/*
 * TMGTop: GPU-accelerated 3D transient thermal topology optimization.
 * Copyright (C) 2026 Yibo Yang
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of TMGTop. TMGTop is free software: you can
 * redistribute it and/or modify it under the terms of the GNU General
 * Public License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * TMGTop is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with TMGTop. If not, see <https://www.gnu.org/licenses/>.
 */

#include <petscksp.h>
#include <petscvec.h>
#include <petscmat.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "mma.h"
#include "gpu_kernels.h"
#include "zslab_partition_3d_v2.hpp"
#include <cuda_runtime.h>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::array;
using std::cout;
using std::endl;
using std::string;
using std::vector;
namespace fs = std::filesystem;

struct HeatPatch {
    double x0 = 0.0, x1 = 0.0;
    double y0 = 0.0, y1 = 0.0;
    double weight = 1.0;
    string name;
};

struct CoolingStrip {
    double x0 = 0.0, x1 = 0.0;
    double y0 = 0.0, y1 = 0.0;
    double zInterface = 0.0; // 0 => bottom boundary, otherwise horizontal internal interface z = const
    double h = 1500.0;
    string name;
};

struct Config3D {
    double Lx = 100e-3;
    double Ly = 60e-3;
    double Lz = 50e-3;
    int nelx = 200;
    int nely = 120;
    int nelz = 100;

    double volumeFraction = 0.15;
    double rmin = 2;
    double eta = 0.5;
    double beta0 = 1.0;
    double betaMax = 16.0;
    int betaUpdateEvery = 40;

    double kLow = 0.27;
    double kHigh = 202.4;
    double cvLow = 9.0e5;
    double cvHigh = 2719.0 * 870.0;
    double penalK = 2.5;
    double penalC = 1.5;

    double Tc = 25.0;      // ambient temperature for air cooling
    double hconv = 60.0;   // air-cooling-level convection coefficient on cooling strips

    double stripWidth = 5e-3;
    double stripXMargin = 5e-3;
    double stripZ2 = 14e-3;
    double stripZ3 = 28e-3;

    double tf = 1000.0;
    int nt = 1000;

    double q0 = 6.0e4;
    double qStage1End = 250.0;
    double qStage2End = 500.0;
    double qStage3End = 750.0;
    double qStage1Scale = 1.0;
    double qStage2Scale = 0.7;
    double qStage3Scale = 0.7;
    double qStage4Scale = 0.7;
    bool usePiecewiseFlux = true;

    double heatPatchCenterX = 50e-3;
    double heatPatchCenterY = 30e-3;
    double heatPatchSizeX = 15e-3;
    double heatPatchSizeY = 15e-3;
    double heatPatchWeight = 1.0;

    int maxIter = 220;
    double changeTol = 1e-2;
    int minIterBeforeStop = 180;
    int changeStableIters = 5;

    string matType = "aijcusparse";
    string vecType = "cuda";
    string backend = "gpu"; // gpu or cpu; both use the same PETSc/discretization path
    string kspType = "cg";
    string pcType = "jacobi";
    double kspRtol = 1e-5;
    int kspMaxIt = 500;

    int checkpointInterval = 32;
    string historyMode = "checkpoint"; // checkpoint, full, or host_full
    string historyHostMemory = "pageable"; // pageable or pinned (GPU host/checkpoint histories)

    // Fixed-design performance mode. Warmups and measured repetitions execute in one
    // process, so CUDA/PETSc initialization is excluded without changing the design.
    bool benchmarkFixedDesign = false;
    int benchmarkWarmupIters = 0;
    int benchmarkMeasuredIters = 0;

    string outDir = "results_3d_opt_1热源3冷却带_200×120×100_0.15_1000s";
    string outPrefix = "case_3d_opt";
    bool writeVTK = true;
    // Production default. The command-line option -write_every_iter can
    // override this value for individual runs.
    int writeEveryIter = 10;
    bool saveProblemPreview = true;
    bool saveFinalDensity = true;
    int previewScale = 10;

    // Restart / warm-start options.
    // The VTK restart file is expected to contain CELL_DATA -> SCALARS Density,
    // i.e., the saved physical density field xPhys from a previous iteration.
    string restartVtk = "";
    int restartIter = 0;
    bool restartInvertProjection = true;

    bool useMMA = true;
    double mma_a0 = 1.0;
    double mma_a = 0.0;
    double mma_c = 1000.0;
    double mma_d = 0.0;

    int seqAijNnzPerRow = 32;
    int mpiAijDiagNnzPerRow = 32;
    int mpiAijOffdiagNnzPerRow = 16;

    int stepPrintEvery = 250;
};

static bool gUseCudaBackend = true;

struct Mesh3D {
    int nelx = 0, nely = 0, nelz = 0;
    int nnx = 0, nny = 0, nnz = 0;
    PetscInt numNodes = 0;
    PetscInt numElems = 0;
    double dx = 0.0, dy = 0.0, dz = 0.0, Ve = 0.0;
    vector<Vector3d> nodes;
    vector<array<PetscInt, 8>> edof;
    vector<Vector3d> centers;
};

struct Filter3D {
    int nelx = 0, nely = 0, nelz = 0;
    double rmin = 0.0;
    int r = 0;
    vector<double> Hs;
};

struct Projection3D {
    vector<double> xTilde;
    vector<double> xPhys;
    vector<double> dProj;
};

static inline void cudaCheckThrow(GpuError err, const char* expr) {
    cudaError_t cerr = static_cast<cudaError_t>(err);
    if (cerr != cudaSuccess) throw std::runtime_error(string("CUDA error in ") + expr + ": " + cudaGetErrorString(cerr));
}
#define CUDA_SAFE(call) cudaCheckThrow((call), #call)

static void petscCheck(PetscErrorCode ierr, const char* expr) {
    if (ierr) throw std::runtime_error(string("PETSc call failed: ") + expr + ", ierr=" + std::to_string(ierr));
}
#define PETSC_CHECK_THROW(call) petscCheck((call), #call)

struct DeviceBuffers3D {
    vector<float> T_ckpt_host;  // owned checkpoint slice only
    vector<float> T_full_host;  // owned full temperature history in CPU RAM for host_full mode
    vector<float> T_seg_host;   // CPU backend work/ghost history
    float* T_ckpt_pinned = nullptr;
    float* T_full_pinned = nullptr;
    size_t T_ckpt_count = 0;
    size_t T_full_count = 0;
    bool hostPinned = false;
    float* T_ckpt_tmp = nullptr; // temp owned device float column
    float* T_seg = nullptr;      // work history on device
    double* dk = nullptr;
    double* dcv = nullptr;
    double* dc_dx = nullptr;
    int* designEdof = nullptr;   // local design to work-node edof
    double* Ke = nullptr;
    double* Ce = nullptr;
    int segLen = 0;
    int numCkpt = 0;
    int nOwned = 0;
    int nWork = 0;
    int nLocalElems = 0;
};

static void destroyDeviceBuffers3D(DeviceBuffers3D& db) {
    db.T_ckpt_host.clear();
    db.T_full_host.clear();
    db.T_seg_host.clear();
    if (db.T_ckpt_pinned) CUDA_SAFE(cudaFreeHost(db.T_ckpt_pinned));
    if (db.T_full_pinned) CUDA_SAFE(cudaFreeHost(db.T_full_pinned));
    if (db.T_ckpt_tmp) CUDA_SAFE(gpuFreePtr(db.T_ckpt_tmp));
    if (db.T_seg) CUDA_SAFE(gpuFreePtr(db.T_seg));
    if (db.dk) CUDA_SAFE(gpuFreePtr(db.dk));
    if (db.dcv) CUDA_SAFE(gpuFreePtr(db.dcv));
    if (db.dc_dx) CUDA_SAFE(gpuFreePtr(db.dc_dx));
    if (db.designEdof) CUDA_SAFE(gpuFreePtr(db.designEdof));
    if (db.Ke) CUDA_SAFE(gpuFreePtr(db.Ke));
    if (db.Ce) CUDA_SAFE(gpuFreePtr(db.Ce));
    db = DeviceBuffers3D{};
}

static float* checkpointHostData(DeviceBuffers3D& db) {
    return db.hostPinned ? db.T_ckpt_pinned : db.T_ckpt_host.data();
}
static const float* checkpointHostData(const DeviceBuffers3D& db) {
    return db.hostPinned ? db.T_ckpt_pinned : db.T_ckpt_host.data();
}
static float* fullHostData(DeviceBuffers3D& db) {
    return db.hostPinned ? db.T_full_pinned : db.T_full_host.data();
}
static const float* fullHostData(const DeviceBuffers3D& db) {
    return db.hostPinned ? db.T_full_pinned : db.T_full_host.data();
}

static const PetscScalar* getVecCudaReadPtr(Vec v) {
    const PetscScalar* ptr = nullptr;
    PETSC_CHECK_THROW(VecCUDAGetArrayRead(v, &ptr));
    return ptr;
}
static void restoreVecCudaReadPtr(Vec v, const PetscScalar* ptr) {
    PETSC_CHECK_THROW(VecCUDARestoreArrayRead(v, &ptr));
}

struct PetscSystem3D {
    Mat K = nullptr;
    Mat A = nullptr;
    Vec Cdiag = nullptr;
    Vec Fshape = nullptr;
    Vec Fcool = nullptr;
    Vec Tn = nullptr;
    Vec rhs = nullptr;
    Vec lambda = nullptr;
    Vec tmp = nullptr;
    KSP ksp = nullptr;
};

static inline PetscInt nodeId(int i, int j, int k, int nnx, int nny) {
    return static_cast<PetscInt>(k) * static_cast<PetscInt>(nny) * static_cast<PetscInt>(nnx)
         + static_cast<PetscInt>(j) * static_cast<PetscInt>(nnx)
         + static_cast<PetscInt>(i);
}
static inline PetscInt elemId(int i, int j, int k, int nelx, int nely) {
    return static_cast<PetscInt>(k) * static_cast<PetscInt>(nely) * static_cast<PetscInt>(nelx)
         + static_cast<PetscInt>(j) * static_cast<PetscInt>(nelx)
         + static_cast<PetscInt>(i);
}

static void ensureDir(const string& path) { fs::create_directories(path); }

static inline double stageBegin(MPI_Comm comm) {
    if (gUseCudaBackend) CUDA_SAFE(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    return MPI_Wtime();
}

static inline double stageElapsedMax(MPI_Comm comm, double t0) {
    if (gUseCudaBackend) CUDA_SAFE(cudaDeviceSynchronize());
    const double local = MPI_Wtime() - t0;
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global;
}

static inline double cudaUsedGB() {
    if (!gUseCudaBackend) return 0.0;
    size_t freeB = 0, totalB = 0;
    cudaError_t e = cudaMemGetInfo(&freeB, &totalB);
    if (e != cudaSuccess) return 0.0;
    return static_cast<double>(totalB - freeB) / (1024.0 * 1024.0 * 1024.0);
}

static double processPeakRssGB() {
    std::ifstream is("/proc/self/status");
    std::string key;
    while (is >> key) {
        if (key == "VmHWM:") {
            double kib = 0.0;
            std::string unit;
            is >> kib >> unit;
            return kib / (1024.0 * 1024.0);
        }
        std::string rest;
        std::getline(is, rest);
    }
    return 0.0;
}

struct HistoryTransferStats {
    std::uint64_t h2dBytes = 0;
    std::uint64_t d2hBytes = 0;
    double h2dSeconds = 0.0;
    double d2hSeconds = 0.0;
    long long h2dCalls = 0;
    long long d2hCalls = 0;
};

static inline void updatePeakGpuGB(double& peak) {
    peak = std::max(peak, cudaUsedGB());
}

static inline double cudaUsedGBMax(MPI_Comm comm) {
    const double local = cudaUsedGB();
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global;
}

static inline void updatePeakGpuGBAll(MPI_Comm comm, double& peak) {
    peak = std::max(peak, cudaUsedGBMax(comm));
}

static inline double bytesToGB(double bytes) {
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

struct MemoryDiagnostics3D {
    double fullHistoryTotalGB = 0.0;       // global temperature history, Nt+1 states, single precision
    double fullHistoryRankMaxGB = 0.0;     // max owned-slice full history per GPU/rank
    double checkpointStorageTotalGB = 0.0; // host checkpoint storage across ranks
    double checkpointStorageRankMaxGB = 0.0;
    double hostFullStorageTotalGB = 0.0;  // host full-history storage across ranks
    double hostFullStorageRankMaxGB = 0.0;
    double segmentBufferRankMaxGB = 0.0;   // device segment buffer, max over ranks
    double checkpointTmpRankMaxGB = 0.0;   // device temp checkpoint vector, max over ranks
    double sensitivityBufferRankMaxGB = 0.0;
    double historyGpuRankMaxGB = 0.0;      // segment + temp + sensitivity/history-related device buffers
    double gpuAfterBufferInitGB = 0.0;     // measured CUDA used memory after history/device buffers are allocated
};

static MemoryDiagnostics3D computeMemoryDiagnostics(const Config3D& cfg, const Mesh3D& mesh,
                                                     const ZSlabPartition3D& part,
                                                     const DeviceBuffers3D& db, MPI_Comm comm) {
    MemoryDiagnostics3D m;
    const double sp = 4.0;
    const double dp = 8.0;
    const double ip = 4.0;

    const double localFull = static_cast<double>(part.nOwnedNodes) * static_cast<double>(cfg.nt + 1) * sp;
    const double localCkpt = static_cast<double>(part.nOwnedNodes) * static_cast<double>(db.numCkpt) * sp;
    const double localHostFull = static_cast<double>(db.T_full_count) * sp;
    const double localSeg  = static_cast<double>(part.nWorkNodes)  * static_cast<double>(db.segLen + 1) * sp;
    const double localTmp  = static_cast<double>(part.nOwnedNodes) * sp;
    const double localSens = static_cast<double>(db.nLocalElems) * (3.0 * dp + 8.0 * ip) + 2.0 * 64.0 * dp;
    const double localHistGpu = gUseCudaBackend ? (localSeg + localTmp + localSens) : 0.0;

    double sumCkpt = 0.0, sumHostFull = 0.0;
    double maxFull = 0.0, maxCkpt = 0.0, maxHostFull = 0.0, maxSeg = 0.0, maxTmp = 0.0, maxSens = 0.0, maxHistGpu = 0.0;
    double localCkptGB = bytesToGB(localCkpt);
    double localHostFullGB = bytesToGB(localHostFull);
    double localFullGB = bytesToGB(localFull);
    double localSegGB  = bytesToGB(localSeg);
    double localTmpGB  = bytesToGB(localTmp);
    double localSensGB = bytesToGB(localSens);
    double localHistGpuGB = bytesToGB(localHistGpu);

    MPI_Allreduce(&localCkptGB, &sumCkpt, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&localHostFullGB, &sumHostFull, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&localFullGB, &maxFull, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localCkptGB, &maxCkpt, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localHostFullGB, &maxHostFull, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localSegGB, &maxSeg, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localTmpGB, &maxTmp, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localSensGB, &maxSens, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localHistGpuGB, &maxHistGpu, 1, MPI_DOUBLE, MPI_MAX, comm);

    m.fullHistoryTotalGB = bytesToGB(static_cast<double>(mesh.numNodes) * static_cast<double>(cfg.nt + 1) * sp);
    m.fullHistoryRankMaxGB = maxFull;
    m.checkpointStorageTotalGB = sumCkpt;
    m.checkpointStorageRankMaxGB = maxCkpt;
    m.hostFullStorageTotalGB = sumHostFull;
    m.hostFullStorageRankMaxGB = maxHostFull;
    m.segmentBufferRankMaxGB = maxSeg;
    m.checkpointTmpRankMaxGB = maxTmp;
    m.sensitivityBufferRankMaxGB = maxSens;
    m.historyGpuRankMaxGB = maxHistGpu;
    m.gpuAfterBufferInitGB = gUseCudaBackend ? cudaUsedGBMax(comm) : 0.0;
    return m;
}

static std::string classifyMemoryBottleneck(const MemoryDiagnostics3D& mem, double fixedSolverPCGB) {
    if (fixedSolverPCGB > 1.5 * std::max(mem.historyGpuRankMaxGB, 1e-12)) return "PC-dominated";
    if (mem.historyGpuRankMaxGB > 1.5 * std::max(fixedSolverPCGB, 1e-12)) return "history-dominated";
    return "mixed";
}


static void readOptions(Config3D& cfg) {
    PetscBool flg = PETSC_FALSE;
    PetscInt itmp = 0;
    char buf[256];
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-Lx", &cfg.Lx, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-Ly", &cfg.Ly, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-Lz", &cfg.Lz, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-nelx", &itmp, &flg)); if (flg) cfg.nelx = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-nely", &itmp, &flg)); if (flg) cfg.nely = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-nelz", &itmp, &flg)); if (flg) cfg.nelz = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-volfrac", &cfg.volumeFraction, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-rmin", &cfg.rmin, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-beta0", &cfg.beta0, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-beta_max", &cfg.betaMax, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-beta_update_every", &itmp, &flg)); if (flg) cfg.betaUpdateEvery = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-eta", &cfg.eta, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-k_low", &cfg.kLow, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-k_high", &cfg.kHigh, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-cv_low", &cfg.cvLow, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-cv_high", &cfg.cvHigh, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-pk", &cfg.penalK, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-pc", &cfg.penalC, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-Tc", &cfg.Tc, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-hconv", &cfg.hconv, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-strip_width", &cfg.stripWidth, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-strip_x_margin", &cfg.stripXMargin, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-strip_z2", &cfg.stripZ2, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-strip_z3", &cfg.stripZ3, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-tf", &cfg.tf, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-nt", &itmp, &flg)); if (flg) cfg.nt = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-q0", &cfg.q0, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-q_stage1_end", &cfg.qStage1End, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-q_stage2_end", &cfg.qStage2End, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-q_stage3_end", &cfg.qStage3End, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-q_stage1_scale", &cfg.qStage1Scale, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-q_stage2_scale", &cfg.qStage2Scale, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-q_stage3_scale", &cfg.qStage3Scale, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-q_stage4_scale", &cfg.qStage4Scale, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-use_piecewise_flux", &itmp, &flg)); if (flg) cfg.usePiecewiseFlux = (itmp != 0);
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-heat_patch_center_x", &cfg.heatPatchCenterX, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-heat_patch_center_y", &cfg.heatPatchCenterY, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-heat_patch_size_x", &cfg.heatPatchSizeX, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-heat_patch_size_y", &cfg.heatPatchSizeY, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-heat_patch_weight", &cfg.heatPatchWeight, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-max_iter", &itmp, &flg)); if (flg) cfg.maxIter = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-change_tol", &cfg.changeTol, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-min_iter_before_stop", &itmp, &flg)); if (flg) cfg.minIterBeforeStop = std::max(0, static_cast<int>(itmp));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-change_stable_iters", &itmp, &flg)); if (flg) cfg.changeStableIters = std::max(1, static_cast<int>(itmp));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-checkpoint_interval", &itmp, &flg)); if (flg) cfg.checkpointInterval = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-history_mode", buf, sizeof(buf), &flg)); if (flg) cfg.historyMode = buf;
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-history_host_memory", buf, sizeof(buf), &flg)); if (flg) cfg.historyHostMemory = buf;
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-backend", buf, sizeof(buf), &flg)); if (flg) cfg.backend = buf;
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-benchmark_fixed_design", &itmp, &flg)); if (flg) cfg.benchmarkFixedDesign = (itmp != 0);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-benchmark_warmup_iters", &itmp, &flg)); if (flg) cfg.benchmarkWarmupIters = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-benchmark_measured_iters", &itmp, &flg)); if (flg) cfg.benchmarkMeasuredIters = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetReal(nullptr, nullptr, "-ksp_rtol", &cfg.kspRtol, &flg));
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-ksp_max_it", &itmp, &flg)); if (flg) cfg.kspMaxIt = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-step_print_every", &itmp, &flg)); if (flg) cfg.stepPrintEvery = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-write_every_iter", &itmp, &flg)); if (flg) cfg.writeEveryIter = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-write_vtk", &itmp, &flg)); if (flg) cfg.writeVTK = (itmp != 0);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-preview_scale", &itmp, &flg)); if (flg) cfg.previewScale = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-save_problem_preview", &itmp, &flg)); if (flg) cfg.saveProblemPreview = (itmp != 0);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-save_final_density", &itmp, &flg)); if (flg) cfg.saveFinalDensity = (itmp != 0);
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-output_dir", buf, sizeof(buf), &flg)); if (flg) cfg.outDir = buf;
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-case_name", buf, sizeof(buf), &flg)); if (flg) cfg.outPrefix = buf;
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-mat_type", buf, sizeof(buf), &flg)); if (flg) cfg.matType = buf;
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-vec_type", buf, sizeof(buf), &flg)); if (flg) cfg.vecType = buf;
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-ksp_type", buf, sizeof(buf), &flg)); if (flg) cfg.kspType = buf;
    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-pc_type", buf, sizeof(buf), &flg)); if (flg) cfg.pcType = buf;

    PETSC_CHECK_THROW(PetscOptionsGetString(nullptr, nullptr, "-restart_vtk", buf, sizeof(buf), &flg)); if (flg) cfg.restartVtk = buf;
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-restart_iter", &itmp, &flg)); if (flg) cfg.restartIter = static_cast<int>(itmp);
    PETSC_CHECK_THROW(PetscOptionsGetInt(nullptr, nullptr, "-restart_invert_projection", &itmp, &flg)); if (flg) cfg.restartInvertProjection = (itmp != 0);
}

static vector<HeatPatch> defaultHeatPatches(const Config3D& cfg) {
    if (cfg.heatPatchSizeX <= 0.0 || cfg.heatPatchSizeY <= 0.0) {
        throw std::runtime_error("Heat patch size must be positive.");
    }
    const double x0 = cfg.heatPatchCenterX - 0.5 * cfg.heatPatchSizeX;
    const double x1 = cfg.heatPatchCenterX + 0.5 * cfg.heatPatchSizeX;
    const double y0 = cfg.heatPatchCenterY - 0.5 * cfg.heatPatchSizeY;
    const double y1 = cfg.heatPatchCenterY + 0.5 * cfg.heatPatchSizeY;
    if (x0 < 0.0 || x1 > cfg.Lx || y0 < 0.0 || y1 > cfg.Ly) {
        throw std::runtime_error("Central heat patch extends outside the design domain.");
    }
    if (cfg.heatPatchWeight <= 0.0) {
        throw std::runtime_error("Heat patch weight must be positive.");
    }
    return {
        {x0, x1, y0, y1, cfg.heatPatchWeight, "Q_center"},
    };
}

static vector<CoolingStrip> defaultCoolingStrips(const Config3D& cfg) {
    const double x0 = cfg.stripXMargin;
    const double x1 = cfg.Lx - cfg.stripXMargin;

    // 三根都在底部 z = 0，沿 x 方向铺开，在 y 方向并排分开
    const double w = cfg.stripWidth;

    // 建议：在 Ly = 60 mm 内，把三根 5 mm 带放在 y = 12, 30, 48 mm 附近
    const double yc1 = 10e-3;
    const double yc2 = 30e-3;
    const double yc3 = 50e-3;

    return {
        {x0, x1, yc1 - 0.5 * w, yc1 + 0.5 * w, 0.0, cfg.hconv, "S1"},
        {x0, x1, yc2 - 0.5 * w, yc2 + 0.5 * w, 0.0, cfg.hconv, "S2"},
        {x0, x1, yc3 - 0.5 * w, yc3 + 0.5 * w, 0.0, cfg.hconv, "S3"},
    };
}

static Mesh3D buildMesh3D(const Config3D& cfg) {
    Mesh3D mesh;
    mesh.nelx = cfg.nelx; mesh.nely = cfg.nely; mesh.nelz = cfg.nelz;
    mesh.nnx = cfg.nelx + 1; mesh.nny = cfg.nely + 1; mesh.nnz = cfg.nelz + 1;
    mesh.dx = cfg.Lx / static_cast<double>(cfg.nelx);
    mesh.dy = cfg.Ly / static_cast<double>(cfg.nely);
    mesh.dz = cfg.Lz / static_cast<double>(cfg.nelz);
    mesh.Ve = mesh.dx * mesh.dy * mesh.dz;
    mesh.numNodes = static_cast<PetscInt>(mesh.nnx) * mesh.nny * mesh.nnz;
    mesh.numElems = static_cast<PetscInt>(cfg.nelx) * cfg.nely * cfg.nelz;

    mesh.nodes.resize(static_cast<size_t>(mesh.numNodes));
    for (int k = 0; k < mesh.nnz; ++k)
        for (int j = 0; j < mesh.nny; ++j)
            for (int i = 0; i < mesh.nnx; ++i) {
                const PetscInt id = nodeId(i, j, k, mesh.nnx, mesh.nny);
                mesh.nodes[static_cast<size_t>(id)] = Vector3d(i * mesh.dx, j * mesh.dy, k * mesh.dz);
            }

    mesh.edof.resize(static_cast<size_t>(mesh.numElems));
    mesh.centers.resize(static_cast<size_t>(mesh.numElems));
    for (int k = 0; k < cfg.nelz; ++k)
        for (int j = 0; j < cfg.nely; ++j)
            for (int i = 0; i < cfg.nelx; ++i) {
                const PetscInt e = elemId(i, j, k, cfg.nelx, cfg.nely);
                const PetscInt n0 = nodeId(i,     j,     k,     mesh.nnx, mesh.nny);
                const PetscInt n1 = nodeId(i + 1, j,     k,     mesh.nnx, mesh.nny);
                const PetscInt n2 = nodeId(i + 1, j + 1, k,     mesh.nnx, mesh.nny);
                const PetscInt n3 = nodeId(i,     j + 1, k,     mesh.nnx, mesh.nny);
                const PetscInt n4 = nodeId(i,     j,     k + 1, mesh.nnx, mesh.nny);
                const PetscInt n5 = nodeId(i + 1, j,     k + 1, mesh.nnx, mesh.nny);
                const PetscInt n6 = nodeId(i + 1, j + 1, k + 1, mesh.nnx, mesh.nny);
                const PetscInt n7 = nodeId(i,     j + 1, k + 1, mesh.nnx, mesh.nny);
                mesh.edof[static_cast<size_t>(e)] = {n0, n1, n2, n3, n4, n5, n6, n7};
                mesh.centers[static_cast<size_t>(e)] = Vector3d((i + 0.5) * mesh.dx, (j + 0.5) * mesh.dy, (k + 0.5) * mesh.dz);
            }
    return mesh;
}

static void hex8ShapeDerivatives(const Vector3d& q, Eigen::Matrix<double, 8, 3>& dN) {
    const double xi = q(0), eta = q(1), zeta = q(2);
    const double s = 0.125;
    const double X[8] = {-1, 1, 1,-1,-1, 1, 1,-1};
    const double E[8] = {-1,-1, 1, 1,-1,-1, 1, 1};
    const double Z[8] = {-1,-1,-1,-1, 1, 1, 1, 1};
    for (int a = 0; a < 8; ++a) {
        dN(a, 0) = s * X[a] * (1.0 + E[a] * eta) * (1.0 + Z[a] * zeta);
        dN(a, 1) = s * E[a] * (1.0 + X[a] * xi)  * (1.0 + Z[a] * zeta);
        dN(a, 2) = s * Z[a] * (1.0 + X[a] * xi)  * (1.0 + E[a] * eta);
    }
}

static Eigen::Matrix<double, 8, 8> AXTH8_N_cpp(const array<PetscInt, 8>& elem, const vector<Vector3d>& nodes) {
    Eigen::Matrix<double, 8, 8> KE = Eigen::Matrix<double, 8, 8>::Zero();
    Eigen::Matrix<double, 8, 3> coord;
    for (int a = 0; a < 8; ++a) coord.row(a) = nodes[static_cast<size_t>(elem[a])].transpose();

    const double gp = 1.0 / std::sqrt(3.0);
    const double qv[2] = {-gp, gp};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                Vector3d q(qv[i], qv[j], qv[k]);
                Eigen::Matrix<double, 8, 3> dNdxi;
                hex8ShapeDerivatives(q, dNdxi);
                Matrix3d J = coord.transpose() * dNdxi;
                Matrix3d invJ = J.inverse();
                Eigen::Matrix<double, 8, 3> dNdx = dNdxi * invJ;
                Eigen::Matrix<double, 3, 8> B;
                B.row(0) = dNdx.col(0).transpose();
                B.row(1) = dNdx.col(1).transpose();
                B.row(2) = dNdx.col(2).transpose();
                KE += B.transpose() * B * J.determinant();
            }
    return KE;
}

static Filter3D buildFilter3D(const Mesh3D& mesh, double rmin) {
    Filter3D f;
    f.nelx = mesh.nelx; f.nely = mesh.nely; f.nelz = mesh.nelz;
    f.rmin = rmin;
    f.r = static_cast<int>(std::ceil(rmin)) - 1;
    f.Hs.assign(static_cast<size_t>(mesh.numElems), 0.0);

    for (int k1 = 0; k1 < mesh.nelz; ++k1)
        for (int j1 = 0; j1 < mesh.nely; ++j1)
            for (int i1 = 0; i1 < mesh.nelx; ++i1) {
                const PetscInt e1 = elemId(i1, j1, k1, mesh.nelx, mesh.nely);
                double s = 0.0;
                for (int k2 = std::max(0, k1 - f.r); k2 <= std::min(mesh.nelz - 1, k1 + f.r); ++k2)
                    for (int j2 = std::max(0, j1 - f.r); j2 <= std::min(mesh.nely - 1, j1 + f.r); ++j2)
                        for (int i2 = std::max(0, i1 - f.r); i2 <= std::min(mesh.nelx - 1, i1 + f.r); ++i2) {
                            const double dist = std::sqrt(static_cast<double>((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2) + (k1 - k2) * (k1 - k2)));
                            s += std::max(0.0, rmin - dist);
                        }
                f.Hs[static_cast<size_t>(e1)] = (s > 0.0) ? s : 1.0;
            }
    return f;
}

static vector<double> filterApply3D(const Filter3D& f, const Mesh3D& mesh, const vector<double>& x) {
    vector<double> y(static_cast<size_t>(mesh.numElems), 0.0);
    for (int k1 = 0; k1 < mesh.nelz; ++k1)
        for (int j1 = 0; j1 < mesh.nely; ++j1)
            for (int i1 = 0; i1 < mesh.nelx; ++i1) {
                const PetscInt e1 = elemId(i1, j1, k1, mesh.nelx, mesh.nely);
                double s = 0.0;
                for (int k2 = std::max(0, k1 - f.r); k2 <= std::min(mesh.nelz - 1, k1 + f.r); ++k2)
                    for (int j2 = std::max(0, j1 - f.r); j2 <= std::min(mesh.nely - 1, j1 + f.r); ++j2)
                        for (int i2 = std::max(0, i1 - f.r); i2 <= std::min(mesh.nelx - 1, i1 + f.r); ++i2) {
                            const PetscInt e2 = elemId(i2, j2, k2, mesh.nelx, mesh.nely);
                            const double dist = std::sqrt(static_cast<double>((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2) + (k1 - k2) * (k1 - k2)));
                            const double w = std::max(0.0, f.rmin - dist);
                            s += w * x[static_cast<size_t>(e2)];
                        }
                y[static_cast<size_t>(e1)] = s;
            }
    return y;
}

static Projection3D project3D(const Filter3D& f, const Mesh3D& mesh, const vector<double>& x, double beta, double eta) {
    Projection3D pr;
    pr.xTilde = filterApply3D(f, mesh, x);
    pr.xPhys.resize(x.size(), 0.0);
    pr.dProj.resize(x.size(), 0.0);
    const double den = std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta));
    for (size_t e = 0; e < x.size(); ++e) {
        pr.xTilde[e] /= f.Hs[e];
        const double arg = beta * (pr.xTilde[e] - eta);
        pr.xPhys[e] = (std::tanh(beta * eta) + std::tanh(arg)) / den;
        pr.dProj[e] = beta * (1.0 - std::tanh(arg) * std::tanh(arg)) / den;
    }
    return pr;
}

static vector<double> filterTransposeWeighted3D(const Filter3D& f, const Mesh3D& mesh, const vector<double>& v) {
    return filterApply3D(f, mesh, v);
}

static double simpK(double xPhys, const Config3D& cfg) {
    return cfg.kLow + std::pow(std::clamp(xPhys, 0.0, 1.0), cfg.penalK) * (cfg.kHigh - cfg.kLow);
}
static double simpCv(double xPhys, const Config3D& cfg) {
    return cfg.cvLow + std::pow(std::clamp(xPhys, 0.0, 1.0), cfg.penalC) * (cfg.cvHigh - cfg.cvLow);
}
static bool centerInPatch(const Vector3d& c, const HeatPatch& p) {
    return (c(0) >= p.x0 && c(0) <= p.x1 && c(1) >= p.y0 && c(1) <= p.y1);
}
static double qScale(double t, const Config3D& cfg) {
    if (!cfg.usePiecewiseFlux) return 1.0;
    if (t <= cfg.qStage1End) return cfg.qStage1Scale;
    if (t <= cfg.qStage2End) return cfg.qStage2Scale;
    if (t <= cfg.qStage3End) return cfg.qStage3Scale;
    return cfg.qStage4Scale;
}
static double patchArea(const HeatPatch& p) {
    return std::max(0.0, p.x1 - p.x0) * std::max(0.0, p.y1 - p.y0);
}
static void validatePiecewiseFluxSchedule(const Config3D& cfg) {
    if (!cfg.usePiecewiseFlux) return;
    if (cfg.qStage1End < 0.0 || cfg.qStage2End < cfg.qStage1End ||
        cfg.qStage3End < cfg.qStage2End || cfg.qStage3End > cfg.tf) {
        throw std::runtime_error("Piecewise heat schedule must satisfy 0 <= q_stage1_end <= q_stage2_end <= q_stage3_end <= tf.");
    }
}
static bool pointInCoolingStripXY(const Vector3d& c, const CoolingStrip& s) {
    return (c(0) >= s.x0 && c(0) <= s.x1 && c(1) >= s.y0 && c(1) <= s.y1);
}

static bool extractStripFaceNodes(const Mesh3D& mesh, PetscInt e, const CoolingStrip& s, PetscInt fn[4]) {
    const int elemsPerLayer = mesh.nelx * mesh.nely;
    const int k = static_cast<int>(e) / elemsPerLayer;
    const auto& en = mesh.edof[static_cast<size_t>(e)];
    const Vector3d& c = mesh.centers[static_cast<size_t>(e)];
    if (!pointInCoolingStripXY(c, s)) return false;

    if (std::abs(s.zInterface) < 1e-14) {
        if (k != 0) return false;
        fn[0] = en[0]; fn[1] = en[1]; fn[2] = en[2]; fn[3] = en[3];
        return true;
    }

    const int kFace = std::max(1, std::min(mesh.nelz - 1, static_cast<int>(std::lround(s.zInterface / mesh.dz))));
    const int kLower = kFace - 1;
    if (k != kLower) return false;
    fn[0] = en[4]; fn[1] = en[5]; fn[2] = en[6]; fn[3] = en[7];
    return true;
}

static void quad4FaceRobin(double h, double area, double Tinf, double Kf[16], double ff[4]) {
    const double M[16] = {
        4.0, 2.0, 1.0, 2.0,
        2.0, 4.0, 2.0, 1.0,
        1.0, 2.0, 4.0, 2.0,
        2.0, 1.0, 2.0, 4.0
    };
    const double scale = h * area / 36.0;
    for (int i = 0; i < 16; ++i) Kf[i] = scale * M[i];
    const double f = h * Tinf * area / 4.0;
    for (int i = 0; i < 4; ++i) ff[i] = f;
}
static vector<double> buildPatchWeightPerTopElement(const Mesh3D& mesh, const vector<HeatPatch>& patches) {
    vector<double> w(static_cast<size_t>(mesh.numElems), 0.0);
    const int kTop = mesh.nelz - 1;
    for (int j = 0; j < mesh.nely; ++j)
        for (int i = 0; i < mesh.nelx; ++i) {
            const PetscInt e = elemId(i, j, kTop, mesh.nelx, mesh.nely);
            const Vector3d& c = mesh.centers[static_cast<size_t>(e)];
            for (const auto& p : patches) if (centerInPatch(c, p)) { w[static_cast<size_t>(e)] = p.weight; break; }
        }
    return w;
}

static Vec createEmptyPetscVecLocal(PetscInt nLocal, PetscInt nGlobal, MPI_Comm comm, const std::string& vecType)
{
    Vec v = nullptr;
    PETSC_CHECK_THROW(VecCreate(comm, &v));
    PETSC_CHECK_THROW(VecSetSizes(v, nLocal, nGlobal));
    PETSC_CHECK_THROW(VecSetType(v, vecType.c_str()));
    PETSC_CHECK_THROW(VecSetFromOptions(v));
    PETSC_CHECK_THROW(VecSetUp(v));
    return v;
}

static Mat createEmptyDistributedAIJMatrix(PetscInt nLocal, PetscInt nGlobal, MPI_Comm comm, const Config3D& cfg)
{
    Mat M = nullptr;
    PETSC_CHECK_THROW(MatCreate(comm, &M));
    PETSC_CHECK_THROW(MatSetSizes(M, nLocal, nLocal, nGlobal, nGlobal));
    PETSC_CHECK_THROW(MatSetType(M, cfg.matType.c_str()));
    PETSC_CHECK_THROW(MatSetFromOptions(M));
    PetscMPIInt size = 1;
    MPI_Comm_size(comm, &size);
    if (size == 1) {
        PETSC_CHECK_THROW(MatSeqAIJSetPreallocation(M, cfg.seqAijNnzPerRow, nullptr));
    } else {
        PETSC_CHECK_THROW(MatMPIAIJSetPreallocation(M, cfg.mpiAijDiagNnzPerRow, nullptr, cfg.mpiAijOffdiagNnzPerRow, nullptr));
    }
    PETSC_CHECK_THROW(MatSetUp(M));
    PETSC_CHECK_THROW(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    return M;
}

static void verifyVecOwnership(const Vec v, PetscInt expectedStart, PetscInt expectedLocalSize, const char* name)
{
    PetscInt s = 0, e = 0;
    PETSC_CHECK_THROW(VecGetOwnershipRange(v, &s, &e));
    if (s != expectedStart || (e - s) != expectedLocalSize) {
        throw std::runtime_error(std::string("Unexpected PETSc ownership for ") + name);
    }
}

static void assembleSystemDistributed(const Config3D& cfg, const Mesh3D& mesh, const ZSlabPartition3D& part,
                                      const vector<double>& xPhys, const vector<double>& patchWeight,
                                      const vector<CoolingStrip>& coolingStrips,
                                      const Eigen::Matrix<double, 8, 8>& KeUnit, const Eigen::Matrix<double, 8, 8>& CeDiagUnit,
                                      PetscSystem3D& ps)
{
    PETSC_CHECK_THROW(MatZeroEntries(ps.K));
    PETSC_CHECK_THROW(VecZeroEntries(ps.Cdiag));
    PETSC_CHECK_THROW(VecZeroEntries(ps.Fshape));
    PETSC_CHECK_THROW(VecZeroEntries(ps.Fcool));

    const double faceArea = mesh.dx * mesh.dy;
    for (PetscInt e : part.ownedElemIds) {
        const auto& en = mesh.edof[static_cast<size_t>(e)];
        const double kappa = simpK(xPhys[static_cast<size_t>(e)], cfg);
        const double cv = simpCv(xPhys[static_cast<size_t>(e)], cfg);
        double Ke[64];
        double CeDiag[8];
        for (int a = 0; a < 8; ++a) {
            CeDiag[a] = cv * CeDiagUnit(a, a);
            for (int b = 0; b < 8; ++b) Ke[8 * a + b] = kappa * KeUnit(a, b);
        }
        PETSC_CHECK_THROW(MatSetValues(ps.K, 8, en.data(), 8, en.data(), Ke, ADD_VALUES));
        PETSC_CHECK_THROW(VecSetValues(ps.Cdiag, 8, en.data(), CeDiag, ADD_VALUES));
        if (patchWeight[static_cast<size_t>(e)] > 0.0) {
            const PetscInt fn[4] = {en[4], en[5], en[6], en[7]};
            const double qw = patchWeight[static_cast<size_t>(e)];
            const double fe[4] = {qw * faceArea / 4.0, qw * faceArea / 4.0, qw * faceArea / 4.0, qw * faceArea / 4.0};
            PETSC_CHECK_THROW(VecSetValues(ps.Fshape, 4, fn, fe, ADD_VALUES));
        }

        PetscInt fnCool[4];
        for (const auto& strip : coolingStrips) {
            if (extractStripFaceNodes(mesh, e, strip, fnCool)) {
                double Kf[16], ff[4];
                quad4FaceRobin(strip.h, faceArea, cfg.Tc, Kf, ff);
                PETSC_CHECK_THROW(MatSetValues(ps.K, 4, fnCool, 4, fnCool, Kf, ADD_VALUES));
                PETSC_CHECK_THROW(VecSetValues(ps.Fcool, 4, fnCool, ff, ADD_VALUES));
            }
        }
    }

    PETSC_CHECK_THROW(MatAssemblyBegin(ps.K, MAT_FINAL_ASSEMBLY));
    PETSC_CHECK_THROW(MatAssemblyEnd(ps.K, MAT_FINAL_ASSEMBLY));
    PETSC_CHECK_THROW(VecAssemblyBegin(ps.Cdiag));
    PETSC_CHECK_THROW(VecAssemblyEnd(ps.Cdiag));
    PETSC_CHECK_THROW(VecAssemblyBegin(ps.Fshape));
    PETSC_CHECK_THROW(VecAssemblyEnd(ps.Fshape));
    PETSC_CHECK_THROW(VecAssemblyBegin(ps.Fcool));
    PETSC_CHECK_THROW(VecAssemblyEnd(ps.Fcool));
}

static void rebuildOperator(const Config3D& cfg, PetscSystem3D& ps) {
    const double dt = cfg.tf / static_cast<double>(cfg.nt);
    if (!ps.A) PETSC_CHECK_THROW(MatDuplicate(ps.K, MAT_DO_NOT_COPY_VALUES, &ps.A));
    PETSC_CHECK_THROW(MatCopy(ps.K, ps.A, SAME_NONZERO_PATTERN));
    PETSC_CHECK_THROW(VecCopy(ps.Cdiag, ps.tmp));
    PETSC_CHECK_THROW(VecScale(ps.tmp, 1.0 / dt));
    PETSC_CHECK_THROW(MatDiagonalSet(ps.A, ps.tmp, ADD_VALUES));
}

static void applyBOperator(const Config3D& cfg, PetscSystem3D& ps, Vec x, Vec y) {
    const double dt = cfg.tf / static_cast<double>(cfg.nt);
    PETSC_CHECK_THROW(VecPointwiseMult(y, ps.Cdiag, x));
    PETSC_CHECK_THROW(VecScale(y, 1.0 / dt));
}

static vector<double> toStd(const VectorXd& v) {
    vector<double> x(static_cast<size_t>(v.size()));
    for (Eigen::Index i = 0; i < v.size(); ++i) x[static_cast<size_t>(i)] = v(i);
    return x;
}
static VectorXd toEigen(const vector<double>& v) {
    VectorXd x(v.size());
    for (size_t i = 0; i < v.size(); ++i) x(static_cast<Eigen::Index>(i)) = v[i];
    return x;
}

static void writeVTK(const string& filename, const Mesh3D& mesh, const vector<double>& nodalT, const vector<double>& cellX) {
    std::ofstream os(filename);
    if (!os) throw std::runtime_error("Failed to open VTK file: " + filename);
    os << "# vtk DataFile Version 3.0\n3D transient heat conduction result\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    os << "POINTS " << mesh.numNodes << " float\n";
    for (PetscInt n = 0; n < mesh.numNodes; ++n) {
        const auto& p = mesh.nodes[static_cast<size_t>(n)];
        os << p(0) << ' ' << p(1) << ' ' << p(2) << '\n';
    }
    os << "CELLS " << mesh.numElems << ' ' << mesh.numElems * 9 << "\n";
    for (PetscInt e = 0; e < mesh.numElems; ++e) {
        const auto& en = mesh.edof[static_cast<size_t>(e)];
        os << 8; for (int a = 0; a < 8; ++a) os << ' ' << en[a]; os << '\n';
    }
    os << "CELL_TYPES " << mesh.numElems << "\n";
    for (PetscInt e = 0; e < mesh.numElems; ++e) os << 12 << '\n';
    os << "POINT_DATA " << mesh.numNodes << "\nSCALARS Temperature float 1\nLOOKUP_TABLE default\n";
    for (double v : nodalT) os << v << '\n';
    os << "CELL_DATA " << mesh.numElems << "\nSCALARS Density float 1\nLOOKUP_TABLE default\n";
    for (double v : cellX) os << v << '\n';
}

static void saveDensityCSV(const string& filename, const Mesh3D& mesh, const vector<double>& xPhys) {
    std::ofstream os(filename);
    os << std::setprecision(16);
    for (int k = 0; k < mesh.nelz; ++k) {
        for (int j = 0; j < mesh.nely; ++j) {
            for (int i = 0; i < mesh.nelx; ++i) {
                const PetscInt e = elemId(i,j,k,mesh.nelx,mesh.nely);
                os << xPhys[static_cast<size_t>(e)];
                if (i < mesh.nelx - 1) os << ',';
            }
            os << '\n';
        }
        os << '\n';
    }
}


static double betaAfterCompletedIterations(const Config3D& cfg, int completedIter) {
    double beta = cfg.beta0;
    if (cfg.betaUpdateEvery <= 0) return std::min(beta, cfg.betaMax);
    for (int i = cfg.betaUpdateEvery; i <= completedIter; i += cfg.betaUpdateEvery) {
        beta = std::min(2.0 * beta, cfg.betaMax);
    }
    return beta;
}

static double inverseHeavisideProjection(double xPhys, double beta, double eta) {
    const double xp = std::clamp(xPhys, 1.0e-9, 1.0 - 1.0e-9);
    if (beta <= 1.0e-14) return xp;

    const double den = std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta));
    double y = xp * den - std::tanh(beta * eta);
    y = std::clamp(y, -1.0 + 1.0e-12, 1.0 - 1.0e-12);
    return std::clamp(eta + 0.5 * std::log((1.0 + y) / (1.0 - y)) / beta, 1.0e-3, 1.0);
}

static vector<double> loadDensityFromVTK(const string& filename, PetscInt expectedNumElems) {
    std::ifstream is(filename);
    if (!is) throw std::runtime_error("Failed to open restart VTK file: " + filename);

    std::string line;
    bool foundDensity = false;
    while (std::getline(is, line)) {
        if (line.find("SCALARS Density") != std::string::npos) {
            foundDensity = true;
            break;
        }
    }
    if (!foundDensity) {
        throw std::runtime_error("Restart VTK file does not contain CELL_DATA SCALARS Density: " + filename);
    }

    // Skip LOOKUP_TABLE line.
    if (!std::getline(is, line)) {
        throw std::runtime_error("Restart VTK file ended before density values: " + filename);
    }

    vector<double> rho(static_cast<size_t>(expectedNumElems), 0.0);
    for (PetscInt e = 0; e < expectedNumElems; ++e) {
        double v = 0.0;
        if (!(is >> v)) {
            throw std::runtime_error("Restart VTK density count is smaller than mesh.numElems in: " + filename);
        }
        rho[static_cast<size_t>(e)] = std::clamp(v, 1.0e-3, 1.0);
    }
    return rho;
}

static void allocateHostFloatStorage(vector<float>& pageable, float*& pinned,
                                     size_t count, bool usePinned)
{
    if (count == 0) return;
    if (usePinned) {
        CUDA_SAFE(cudaHostAlloc(reinterpret_cast<void**>(&pinned), count * sizeof(float), cudaHostAllocPortable));
        std::fill(pinned, pinned + count, 0.0f);
    } else {
        pageable.assign(count, 0.0f);
    }
}

static void initDeviceBuffers3D(DeviceBuffers3D& db,
                                const ZSlabPartition3D& part,
                                const Config3D& cfg,
                                const Eigen::Matrix<double,8,8>& KeGeo,
                                const Eigen::Matrix<double,8,8>& CeGeo)
{
    db.nOwned = part.nOwnedNodes;
    db.nWork = part.nWorkNodes;
    db.nLocalElems = static_cast<int>(part.localDesignElems.size());
    db.hostPinned = gUseCudaBackend && cfg.historyHostMemory == "pinned";

    if (cfg.historyMode == "full") {
        // GPU full-history: store all work-node history on device.
        db.segLen = cfg.nt;
        db.numCkpt = 0;
    } else if (cfg.historyMode == "host_full") {
        // Host full-history: store all owned-node history in CPU RAM; keep only two work columns on GPU.
        db.segLen = 1;
        db.numCkpt = 0;
        db.T_full_count = static_cast<size_t>(db.nOwned) * static_cast<size_t>(cfg.nt + 1);
        allocateHostFloatStorage(db.T_full_host, db.T_full_pinned, db.T_full_count, db.hostPinned);
    } else {
        // Checkpoint-recompute: store sparse owned checkpoints in CPU RAM and one segment on GPU.
        db.segLen = cfg.checkpointInterval;
        db.numCkpt = cfg.nt / cfg.checkpointInterval + 1;
        db.T_ckpt_count = static_cast<size_t>(db.nOwned) * static_cast<size_t>(db.numCkpt);
        allocateHostFloatStorage(db.T_ckpt_host, db.T_ckpt_pinned, db.T_ckpt_count, db.hostPinned);
    }

    if (!gUseCudaBackend) {
        db.T_seg_host.assign(static_cast<size_t>(db.nWork) * static_cast<size_t>(db.segLen + 1), 0.0f);
        return;
    }

    CUDA_SAFE(gpuAllocFloats(&db.T_ckpt_tmp, static_cast<size_t>(db.nOwned)));
    CUDA_SAFE(gpuAllocFloats(&db.T_seg, static_cast<size_t>(db.nWork) * static_cast<size_t>(db.segLen + 1)));
    CUDA_SAFE(gpuAllocDoubles(&db.dk, static_cast<size_t>(db.nLocalElems)));
    CUDA_SAFE(gpuAllocDoubles(&db.dcv, static_cast<size_t>(db.nLocalElems)));
    CUDA_SAFE(gpuAllocDoubles(&db.dc_dx, static_cast<size_t>(db.nLocalElems)));
    CUDA_SAFE(gpuAllocInts(&db.designEdof, static_cast<size_t>(db.nLocalElems) * 8));
    CUDA_SAFE(gpuAllocDoubles(&db.Ke, 64));
    CUDA_SAFE(gpuAllocDoubles(&db.Ce, 64));

    vector<int> edofFlat(static_cast<size_t>(db.nLocalElems) * 8);
    for (size_t e = 0; e < static_cast<size_t>(db.nLocalElems); ++e)
        for (int a = 0; a < 8; ++a) edofFlat[8 * e + a] = part.localDesignEdofWork[e][a];

    double KeFlat[64], CeFlat[64];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
        KeFlat[8 * i + j] = KeGeo(i, j);
        CeFlat[8 * i + j] = CeGeo(i, j);
    }
    CUDA_SAFE(gpuMemcpyHtoD(db.designEdof, edofFlat.data(), sizeof(int) * edofFlat.size()));
    CUDA_SAFE(gpuMemcpyHtoD(db.Ke, KeFlat, sizeof(double) * 64));
    CUDA_SAFE(gpuMemcpyHtoD(db.Ce, CeFlat, sizeof(double) * 64));
    CUDA_SAFE(gpuZeroDoubles(db.dc_dx, static_cast<size_t>(db.nLocalElems)));
}

static void storeOwnedVecColumnToCheckpoint(Vec ownedVec, int nOwned, int col, DeviceBuffers3D& db,
                                            HistoryTransferStats& transfer)
{
    float* hdst = checkpointHostData(db) + static_cast<size_t>(col) * static_cast<size_t>(nOwned);
    if (gUseCudaBackend) {
        const PetscScalar* dptr = getVecCudaReadPtr(ownedVec);
        CUDA_SAFE(gpuStoreHistoryColumnFloat(nOwned, 0, reinterpret_cast<const double*>(dptr), db.T_ckpt_tmp));
        restoreVecCudaReadPtr(ownedVec, dptr);
        const double t0 = MPI_Wtime();
        CUDA_SAFE(gpuMemcpyDtoH(hdst, db.T_ckpt_tmp, static_cast<size_t>(nOwned) * sizeof(float)));
        transfer.d2hSeconds += MPI_Wtime() - t0;
        transfer.d2hBytes += static_cast<std::uint64_t>(nOwned) * sizeof(float);
        ++transfer.d2hCalls;
    } else {
        const PetscScalar* ptr = nullptr;
        PETSC_CHECK_THROW(VecGetArrayRead(ownedVec, &ptr));
        for (int i = 0; i < nOwned; ++i) hdst[i] = static_cast<float>(PetscRealPart(ptr[i]));
        PETSC_CHECK_THROW(VecRestoreArrayRead(ownedVec, &ptr));
    }
}

static void loadCheckpointOwnedSliceToDistributedVec(const ZSlabPartition3D& part, DeviceBuffers3D& db,
                                                     int ckIdx, Vec globalVec, HistoryTransferStats& transfer)
{
    const float* hsrc = checkpointHostData(db) + static_cast<size_t>(ckIdx) * static_cast<size_t>(db.nOwned);
    PetscScalar* owned = nullptr;
    if (gUseCudaBackend) {
        PETSC_CHECK_THROW(VecCUDAGetArrayWrite(globalVec, &owned));
        const double t0 = MPI_Wtime();
        CUDA_SAFE(gpuMemcpyHtoD(db.T_ckpt_tmp, hsrc, static_cast<size_t>(part.nOwnedNodes) * sizeof(float)));
        transfer.h2dSeconds += MPI_Wtime() - t0;
        transfer.h2dBytes += static_cast<std::uint64_t>(part.nOwnedNodes) * sizeof(float);
        ++transfer.h2dCalls;
        CUDA_SAFE(gpuCopyFloatToDouble(reinterpret_cast<double*>(owned), db.T_ckpt_tmp, static_cast<size_t>(part.nOwnedNodes)));
        PETSC_CHECK_THROW(VecCUDARestoreArrayWrite(globalVec, &owned));
    } else {
        PETSC_CHECK_THROW(VecGetArrayWrite(globalVec, &owned));
        for (int i = 0; i < part.nOwnedNodes; ++i) owned[i] = static_cast<PetscScalar>(hsrc[i]);
        PETSC_CHECK_THROW(VecRestoreArrayWrite(globalVec, &owned));
    }
}

static void storeOwnedVecColumnToHostFull(Vec ownedVec, int nOwned, int col, DeviceBuffers3D& db,
                                          HistoryTransferStats& transfer)
{
    float* hdst = fullHostData(db) + static_cast<size_t>(col) * static_cast<size_t>(nOwned);
    if (gUseCudaBackend) {
        const PetscScalar* dptr = getVecCudaReadPtr(ownedVec);
        CUDA_SAFE(gpuStoreHistoryColumnFloat(nOwned, 0, reinterpret_cast<const double*>(dptr), db.T_ckpt_tmp));
        restoreVecCudaReadPtr(ownedVec, dptr);
        const double t0 = MPI_Wtime();
        CUDA_SAFE(gpuMemcpyDtoH(hdst, db.T_ckpt_tmp, static_cast<size_t>(nOwned) * sizeof(float)));
        transfer.d2hSeconds += MPI_Wtime() - t0;
        transfer.d2hBytes += static_cast<std::uint64_t>(nOwned) * sizeof(float);
        ++transfer.d2hCalls;
    } else {
        const PetscScalar* ptr = nullptr;
        PETSC_CHECK_THROW(VecGetArrayRead(ownedVec, &ptr));
        for (int i = 0; i < nOwned; ++i) hdst[i] = static_cast<float>(PetscRealPart(ptr[i]));
        PETSC_CHECK_THROW(VecRestoreArrayRead(ownedVec, &ptr));
    }
}

static void loadHostFullOwnedSliceToDistributedVec(const ZSlabPartition3D& part, DeviceBuffers3D& db,
                                                   int timeStep, Vec globalVec, HistoryTransferStats& transfer)
{
    const float* hsrc = fullHostData(db) + static_cast<size_t>(timeStep) * static_cast<size_t>(db.nOwned);
    PetscScalar* owned = nullptr;
    if (gUseCudaBackend) {
        PETSC_CHECK_THROW(VecCUDAGetArrayWrite(globalVec, &owned));
        const double t0 = MPI_Wtime();
        CUDA_SAFE(gpuMemcpyHtoD(db.T_ckpt_tmp, hsrc, static_cast<size_t>(part.nOwnedNodes) * sizeof(float)));
        transfer.h2dSeconds += MPI_Wtime() - t0;
        transfer.h2dBytes += static_cast<std::uint64_t>(part.nOwnedNodes) * sizeof(float);
        ++transfer.h2dCalls;
        CUDA_SAFE(gpuCopyFloatToDouble(reinterpret_cast<double*>(owned), db.T_ckpt_tmp, static_cast<size_t>(part.nOwnedNodes)));
        PETSC_CHECK_THROW(VecCUDARestoreArrayWrite(globalVec, &owned));
    } else {
        PETSC_CHECK_THROW(VecGetArrayWrite(globalVec, &owned));
        for (int i = 0; i < part.nOwnedNodes; ++i) owned[i] = static_cast<PetscScalar>(hsrc[i]);
        PETSC_CHECK_THROW(VecRestoreArrayWrite(globalVec, &owned));
    }
}

static void storeWorkVecColumnToHistory(Vec workVec, int nWork, int col, DeviceBuffers3D& db)
{
    if (gUseCudaBackend) {
        const PetscScalar* dptr = getVecCudaReadPtr(workVec);
        CUDA_SAFE(gpuStoreHistoryColumnFloat(nWork, col, reinterpret_cast<const double*>(dptr), db.T_seg));
        restoreVecCudaReadPtr(workVec, dptr);
    } else {
        const PetscScalar* ptr = nullptr;
        PETSC_CHECK_THROW(VecGetArrayRead(workVec, &ptr));
        float* dst = db.T_seg_host.data() + static_cast<size_t>(col) * static_cast<size_t>(nWork);
        for (int i = 0; i < nWork; ++i) dst[i] = static_cast<float>(PetscRealPart(ptr[i]));
        PETSC_CHECK_THROW(VecRestoreArrayRead(workVec, &ptr));
    }
}

static void accumulateSensitivityStep(const ZSlabPartition3D& part,
                                      DeviceBuffers3D& db,
                                      Vec workLambda,
                                      int timeColN,
                                      int timeColNp1,
                                      const vector<double>& dkLocal,
                                      const vector<double>& dcvLocal,
                                      double dt,
                                      const Eigen::Matrix<double,8,8>& KeGeo,
                                      const Eigen::Matrix<double,8,8>& CeGeo,
                                      vector<double>& dcHost)
{
    const int ndLocal = static_cast<int>(part.localDesignElems.size());
    if (gUseCudaBackend) {
        const PetscScalar* dLamNp1 = getVecCudaReadPtr(workLambda);
        CUDA_SAFE(gpuSensitivityAccumulateHistoryFloatHex8(ndLocal, db.designEdof, db.T_seg, part.nWorkNodes,
                    timeColN, timeColNp1,
                    reinterpret_cast<const double*>(dLamNp1), db.dk, db.dcv,
                    dt, db.Ke, db.Ce, db.dc_dx));
        restoreVecCudaReadPtr(workLambda, dLamNp1);
        return;
    }

    const PetscScalar* lambda = nullptr;
    PETSC_CHECK_THROW(VecGetArrayRead(workLambda, &lambda));
    const float* tn = db.T_seg_host.data() + static_cast<size_t>(timeColN) * static_cast<size_t>(part.nWorkNodes);
    const float* tnp1 = db.T_seg_host.data() + static_cast<size_t>(timeColNp1) * static_cast<size_t>(part.nWorkNodes);
    for (int e = 0; e < ndLocal; ++e) {
        const auto& edof = part.localDesignEdofWork[static_cast<size_t>(e)];
        double accum = 0.0;
        for (int a = 0; a < 8; ++a) {
            const double la = PetscRealPart(lambda[edof[a]]);
            double row = 0.0;
            for (int b = 0; b < 8; ++b) {
                const double kb = dkLocal[static_cast<size_t>(e)] * KeGeo(a,b) * static_cast<double>(tnp1[edof[b]]);
                const double cb = dcvLocal[static_cast<size_t>(e)] * CeGeo(a,b)
                                * (static_cast<double>(tnp1[edof[b]]) - static_cast<double>(tn[edof[b]])) / dt;
                row += kb + cb;
            }
            accum += la * row;
        }
        dcHost[static_cast<size_t>(e)] += accum;
    }
    PETSC_CHECK_THROW(VecRestoreArrayRead(workLambda, &lambda));
}

static vector<double> gatherDistributedVecToRoot(Vec v, MPI_Comm comm)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    PetscInt nLocal = 0, nGlobal = 0, start = 0, end = 0;
    PETSC_CHECK_THROW(VecGetLocalSize(v, &nLocal));
    PETSC_CHECK_THROW(VecGetSize(v, &nGlobal));
    PETSC_CHECK_THROW(VecGetOwnershipRange(v, &start, &end));

    vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
    }
    int nLocalInt = static_cast<int>(nLocal);
    int startInt = static_cast<int>(start);
    MPI_Gather(&nLocalInt, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, comm);
    MPI_Gather(&startInt, 1, MPI_INT, rank == 0 ? displs.data() : nullptr, 1, MPI_INT, 0, comm);

    const PetscScalar* arr = nullptr;
    PETSC_CHECK_THROW(VecGetArrayRead(v, &arr));
    vector<double> local(static_cast<size_t>(nLocal));
    for (PetscInt i = 0; i < nLocal; ++i) local[static_cast<size_t>(i)] = PetscRealPart(arr[i]);
    PETSC_CHECK_THROW(VecRestoreArrayRead(v, &arr));

    vector<double> global;
    if (rank == 0) global.resize(static_cast<size_t>(nGlobal));
    MPI_Gatherv(local.data(), nLocalInt, MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm);
    return global;
}


struct IterStageTimes {
    double filter = 0.0;
    double assemble = 0.0;
    double operatorBuild = 0.0;
    double pcSetup = 0.0;
    double forward = 0.0;
    double sensPrep = 0.0;
    double adjoint = 0.0;
    double update = 0.0;
    double vtk = 0.0;
    double total = 0.0;
    double avgKspIts = 0.0;
    double avgFwdKspIts = 0.0;
    double avgAdjKspIts = 0.0;
    double maxKspResidual = 0.0;
    long long failedKspSolves = 0;
    long long forwardSolves = 0;
    long long adjointSolves = 0;
    long long recomputeSolves = 0;
    double peakGpuGB = 0.0;
    double memAfterPCSetupGB = 0.0;
    double fixedSolverPCGB = 0.0;
    double hostPeakRssRankMaxGB = 0.0;
    double hostPeakRssTotalGB = 0.0;
    std::uint64_t historyH2DBytesTotal = 0;
    std::uint64_t historyD2HBytesTotal = 0;
    double historyH2DTimeRankMax = 0.0;
    double historyD2HTimeRankMax = 0.0;
};

static void recordKspOutcome(KSP ksp, IterStageTimes& stage) {
    PetscReal residual = 0.0;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PETSC_CHECK_THROW(KSPGetResidualNorm(ksp, &residual));
    PETSC_CHECK_THROW(KSPGetConvergedReason(ksp, &reason));
    stage.maxKspResidual = std::max(stage.maxKspResidual, static_cast<double>(residual));
    if (reason < 0) ++stage.failedKspSolves;
}

static cv::Point mapXY(double x, double y, double Lx, double Ly, int ox, int oy, int W, int H)
{
    const int px = ox + static_cast<int>(std::round((x / Lx) * W));
    const int py = oy + H - static_cast<int>(std::round((y / Ly) * H));
    return {px, py};
}

static cv::Point mapXZ(double x, double z, double Lx, double Lz, int ox, int oy, int W, int H)
{
    const int px = ox + static_cast<int>(std::round((x / Lx) * W));
    const int py = oy + H - static_cast<int>(std::round((z / Lz) * H));
    return {px, py};
}

static void saveProblemPreviewPNG(const Config3D& cfg,
                                  const std::vector<HeatPatch>& patches,
                                  const std::vector<CoolingStrip>& strips,
                                  const std::string& filename)
{
    const int s = std::max(4, cfg.previewScale);
    const int margin = 24;
    const int topW = static_cast<int>(std::round(cfg.Lx * 1e3 * s));
    const int topH = static_cast<int>(std::round(cfg.Ly * 1e3 * s));
    const int sideW = topW;
    const int sideH = static_cast<int>(std::round(cfg.Lz * 1e3 * s));
    const int textPad = 70;
    const int W = margin * 3 + topW + sideW;
    const int H = margin * 3 + std::max(topH, sideH) + textPad;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(248, 248, 248));

    auto fmt = [](double value, int precision) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(precision) << value;
        return os.str();
    };

    const int topOx = margin;
    const int topOy = margin + textPad;
    const int sideOx = margin * 2 + topW;
    const int sideOy = margin + textPad;

    cv::Scalar border(60, 90, 110);
    cv::rectangle(img, cv::Rect(topOx, topOy, topW, topH), border, 2);
    cv::rectangle(img, cv::Rect(sideOx, sideOy, sideW, sideH), border, 2);

    cv::putText(img, "Top view (central heat patch + air-cooling strip footprints)", {topOx, topOy - 22},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(20,20,20), 1, cv::LINE_AA);
    cv::putText(img, "X-Z section at mid-Y (heat patch + cooling strips)", {sideOx, sideOy - 22},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(20,20,20), 1, cv::LINE_AA);

    for (const auto& st : strips) {
        auto p0 = mapXY(st.x0, st.y0, cfg.Lx, cfg.Ly, topOx, topOy, topW, topH);
        auto p1 = mapXY(st.x1, st.y1, cfg.Lx, cfg.Ly, topOx, topOy, topW, topH);
        cv::Rect rr(cv::Point(std::min(p0.x,p1.x), std::min(p0.y,p1.y)), cv::Point(std::max(p0.x,p1.x), std::max(p0.y,p1.y)));
        cv::rectangle(img, rr, cv::Scalar(232, 242, 252), cv::FILLED);
        cv::rectangle(img, rr, cv::Scalar(145, 176, 210), 1);
    }

    const cv::Scalar patchFill(50, 95, 230);
    const cv::Scalar patchStroke(18, 35, 120);
    for (size_t i = 0; i < patches.size(); ++i) {
        const auto& hp = patches[i];
        auto p0 = mapXY(hp.x0, hp.y0, cfg.Lx, cfg.Ly, topOx, topOy, topW, topH);
        auto p1 = mapXY(hp.x1, hp.y1, cfg.Lx, cfg.Ly, topOx, topOy, topW, topH);
        cv::Rect rr(cv::Point(std::min(p0.x,p1.x), std::min(p0.y,p1.y)), cv::Point(std::max(p0.x,p1.x), std::max(p0.y,p1.y)));
        cv::rectangle(img, rr, patchFill, cv::FILLED);
        cv::rectangle(img, rr, patchStroke, 2);
        cv::putText(img, hp.name, {rr.x + 4, rr.y - 4}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(10,10,10), 1, cv::LINE_AA);
    }

    const double midY = 0.5 * cfg.Ly;
    for (const auto& hp : patches) {
        if (midY < hp.y0 || midY > hp.y1) continue;
        auto p0 = mapXZ(hp.x0, cfg.Lz, cfg.Lx, cfg.Lz, sideOx, sideOy, sideW, sideH);
        auto p1 = mapXZ(hp.x1, cfg.Lz, cfg.Lx, cfg.Lz, sideOx, sideOy, sideW, sideH);
        cv::Rect rr(std::min(p0.x, p1.x), sideOy + 2, std::max(2, std::abs(p1.x - p0.x)), 6);
        cv::rectangle(img, rr, patchFill, cv::FILLED);
        cv::rectangle(img, rr, patchStroke, 1);
        cv::putText(img, hp.name, {rr.x + 4, rr.y + 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, patchStroke, 1, cv::LINE_AA);
    }

    for (size_t i = 0; i < strips.size(); ++i) {
        const auto& st = strips[i];
        int zPix = 0;
        if (std::abs(st.zInterface) < 1e-14) zPix = sideOy + sideH - 2;
        else zPix = mapXZ(0.0, st.zInterface, cfg.Lx, cfg.Lz, sideOx, sideOy, sideW, sideH).y;
        auto p0 = mapXZ(st.x0, 0.0, cfg.Lx, cfg.Lz, sideOx, sideOy, sideW, sideH);
        auto p1 = mapXZ(st.x1, 0.0, cfg.Lx, cfg.Lz, sideOx, sideOy, sideW, sideH);
        cv::Rect rr(std::min(p0.x,p1.x), zPix - 3, std::abs(p1.x - p0.x), 6);
        cv::rectangle(img, rr, cv::Scalar(180, 130, 60), cv::FILLED);
        cv::rectangle(img, rr, cv::Scalar(120, 80, 30), 1);
        cv::putText(img, st.name, {rr.x + 4, rr.y - 6}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(40,40,40), 1, cv::LINE_AA);
    }

    cv::putText(img, "Single central heat patch: " + fmt(cfg.heatPatchSizeX * 1e3, 1) + " x " + fmt(cfg.heatPatchSizeY * 1e3, 1) + " mm",
                {margin + 10, margin + 26},
                cv::FONT_HERSHEY_SIMPLEX, 0.95, cv::Scalar(20,20,20), 2, cv::LINE_AA);
    cv::putText(img, "Top/side boundaries: adiabatic", {margin + 10, H - 36},
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(50,50,50), 1, cv::LINE_AA);
    cv::putText(img, "Cooling: 3 bottom air-cooling strips, h=" + fmt(cfg.hconv, 1) + " W/m^2K, variable load enabled", {margin + 10, H - 12},
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(50,50,50), 1, cv::LINE_AA);

    if (!cv::imwrite(filename, img)) {
        throw std::runtime_error("Failed to write preview PNG: " + filename);
    }
}

int main(int argc, char** argv) {

    PetscInitialize(&argc, &argv, nullptr, nullptr);
    try {
        MPI_Comm comm = PETSC_COMM_WORLD;
        int rank = 0, size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        Config3D cfg;
        readOptions(cfg);
        if (cfg.backend != "gpu" && cfg.backend != "cpu") {
            throw std::runtime_error("-backend must be gpu or cpu");
        }
        gUseCudaBackend = (cfg.backend == "gpu");
        if (cfg.historyHostMemory != "pageable" && cfg.historyHostMemory != "pinned") {
            throw std::runtime_error("-history_host_memory must be pageable or pinned");
        }
        if (!gUseCudaBackend && cfg.historyHostMemory == "pinned") {
            throw std::runtime_error("Pinned host history is a GPU transfer experiment; use pageable for -backend cpu");
        }
        cfg.benchmarkWarmupIters = std::max(0, cfg.benchmarkWarmupIters);
        cfg.benchmarkMeasuredIters = std::max(0, cfg.benchmarkMeasuredIters);
        validatePiecewiseFluxSchedule(cfg);
        if (cfg.checkpointInterval <= 0) cfg.checkpointInterval = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(cfg.nt)))));
        if (cfg.historyMode != "checkpoint" && cfg.historyMode != "full" && cfg.historyMode != "host_full") {
            throw std::runtime_error("-history_mode must be checkpoint, full, or host_full");
        }
        if (cfg.historyMode == "full" && gUseCudaBackend && size > 1 && rank == 0) {
            std::cerr << "Warning: GPU full-history mode stores work/ghost history on device; multi-rank runs may use extra ghost history memory." << std::endl;
        }
        if (gUseCudaBackend) {
            if (size > 1) {
                if (cfg.matType == "aijcusparse") cfg.matType = "mpiaijcusparse";
                if (cfg.vecType == "cuda") cfg.vecType = "mpicuda";
            }
        } else {
            cfg.matType = (size > 1) ? "mpiaij" : "seqaij";
            cfg.vecType = (size > 1) ? "mpi" : "seq";
        }

        ensureDir(cfg.outDir);
        const auto patches = defaultHeatPatches(cfg);
        const auto mesh = buildMesh3D(cfg);
        const auto fctx = (rank == 0) ? buildFilter3D(mesh, cfg.rmin) : Filter3D{};
        const auto patchWeight = buildPatchWeightPerTopElement(mesh, patches);
        const auto coolingStrips = defaultCoolingStrips(cfg);
        if (rank == 0 && cfg.saveProblemPreview) {
            saveProblemPreviewPNG(cfg, patches, coolingStrips, (fs::path(cfg.outDir) / (cfg.outPrefix + "_problem_preview.png")).string());
        }

        auto part = buildUniformZSlabPartition3D(mesh, comm, nodeId, elemId);
        buildLocalDesignEdofWork3D(mesh, part);
		
		for (size_t e = 0; e < part.localDesignEdofWork.size(); ++e) {
			for (int a = 0; a < 8; ++a) {
				int idw = part.localDesignEdofWork[e][a];
				if (idw < 0 || idw >= part.nWorkNodes) {
					std::cerr << "[Rank " << rank << "] localDesignEdofWork OOB: "
							  << "e=" << e << ", a=" << a
							  << ", idw=" << idw
							  << ", nWorkNodes=" << part.nWorkNodes << std::endl;
					MPI_Abort(comm, 91);
				}
			}
		}

        if (rank == 0) {
              cout << "3D optimization PETSc CPU/CUDA benchmark (single central heat patch, 3 air-cooling strips)" << endl;
            cout << "mesh: " << mesh.nelx << " x " << mesh.nely << " x " << mesh.nelz
                 << ", nodes=" << mesh.numNodes << ", elems=" << mesh.numElems << ", ranks=" << size << endl;
            cout << "backend=" << cfg.backend
                 << ", mat_type=" << cfg.matType
                 << ", vec_type=" << cfg.vecType
                 << ", ksp_type=" << cfg.kspType
                 << ", pc_type=" << cfg.pcType
                 << ", history_mode=" << cfg.historyMode
                 << ", history_host_memory=" << cfg.historyHostMemory
                 << ", checkpoint_interval=" << cfg.checkpointInterval << endl;
              const auto& patch = patches.front();
              cout << "heat patch: center(mm)=(" << 0.5 * (patch.x0 + patch.x1) * 1e3
                  << ", " << 0.5 * (patch.y0 + patch.y1) * 1e3 << ")"
                  << ", size(mm)=(" << (patch.x1 - patch.x0) * 1e3
                  << " x " << (patch.y1 - patch.y0) * 1e3 << ")"
                  << ", q0=" << cfg.q0 << " W/m^2" << endl;
              if (cfg.usePiecewiseFlux) {
                 cout << "load schedule(s): [0, " << cfg.qStage1End << "] x" << cfg.qStage1Scale
                     << ", (" << cfg.qStage1End << ", " << cfg.qStage2End << "] x" << cfg.qStage2Scale
                     << ", (" << cfg.qStage2End << ", " << cfg.qStage3End << "] x" << cfg.qStage3Scale
                     << ", (" << cfg.qStage3End << ", " << cfg.tf << "] x" << cfg.qStage4Scale << endl;
              } else {
                 cout << "load schedule(s): uniform x1.0 over [0, " << cfg.tf << "]" << endl;
              }
              cout << "heat patch power(W): stage1=" << cfg.q0 * cfg.qStage1Scale * patchArea(patch) * patch.weight
                  << ", stage2=" << cfg.q0 * cfg.qStage2Scale * patchArea(patch) * patch.weight
                  << ", stage3=" << cfg.q0 * cfg.qStage3Scale * patchArea(patch) * patch.weight
                  << ", stage4=" << cfg.q0 * cfg.qStage4Scale * patchArea(patch) * patch.weight << endl;
              cout << "cooling strips: width=" << cfg.stripWidth * 1e3
    				 << " mm, h=" << cfg.hconv
    				 << " W/m^2K, arranged side-by-side on bottom surface" << endl;
        }
        cout << "[Rank " << rank << "] ez=[" << part.ez0 << "," << part.ez1 << ")"
             << ", ownedNodes=" << part.nOwnedNodes << ", workNodes=" << part.nWorkNodes
             << ", localElems=" << part.nOwnedElems << endl;

        const auto KeGeo = AXTH8_N_cpp(mesh.edof.front(), mesh.nodes);
        Eigen::Matrix<double,8,8> CeGeo = Eigen::Matrix<double,8,8>::Zero();
        for (int a = 0; a < 8; ++a) CeGeo(a,a) = mesh.Ve / 8.0;

        PetscSystem3D ps;
        ps.K = createEmptyDistributedAIJMatrix(part.nOwnedNodes, mesh.numNodes, comm, cfg);
        ps.Cdiag = createEmptyPetscVecLocal(part.nOwnedNodes, mesh.numNodes, comm, cfg.vecType);
        ps.Fshape = createEmptyPetscVecLocal(part.nOwnedNodes, mesh.numNodes, comm, cfg.vecType);
        ps.Fcool = createEmptyPetscVecLocal(part.nOwnedNodes, mesh.numNodes, comm, cfg.vecType);
        ps.Tn = createEmptyPetscVecLocal(part.nOwnedNodes, mesh.numNodes, comm, cfg.vecType);
        ps.rhs = createEmptyPetscVecLocal(part.nOwnedNodes, mesh.numNodes, comm, cfg.vecType);
        ps.lambda = createEmptyPetscVecLocal(part.nOwnedNodes, mesh.numNodes, comm, cfg.vecType);
        ps.tmp = createEmptyPetscVecLocal(part.nOwnedNodes, mesh.numNodes, comm, cfg.vecType);
        verifyVecOwnership(ps.Tn, part.ownedNodeStart, part.nOwnedNodes, "Tn");

        PETSC_CHECK_THROW(KSPCreate(comm, &ps.ksp));
        PETSC_CHECK_THROW(KSPSetType(ps.ksp, cfg.kspType.c_str()));
        PETSC_CHECK_THROW(KSPSetInitialGuessNonzero(ps.ksp, PETSC_TRUE));
        PC pc = nullptr;
        PETSC_CHECK_THROW(KSPGetPC(ps.ksp, &pc));
        PETSC_CHECK_THROW(PCSetType(pc, cfg.pcType.c_str()));
        PETSC_CHECK_THROW(KSPSetTolerances(ps.ksp, cfg.kspRtol, PETSC_DEFAULT, PETSC_DEFAULT, cfg.kspMaxIt));
        PETSC_CHECK_THROW(KSPSetFromOptions(ps.ksp));

        DeviceBuffers3D db;
        initDeviceBuffers3D(db, part, cfg, KeGeo, CeGeo);
        const MemoryDiagnostics3D memDiag = computeMemoryDiagnostics(cfg, mesh, part, db, comm);
        if (rank == 0) {
            cout << "MEMORY_CASE"
                 << " scaleMesh=" << mesh.nelx << "x" << mesh.nely << "x" << mesh.nelz
                 << " ranks=" << size
                 << " backend=" << cfg.backend
                 << " historyMode=" << cfg.historyMode
                 << " historyHostMemory=" << cfg.historyHostMemory
                 << " interval=" << cfg.checkpointInterval
                 << " nt=" << cfg.nt
                 << " nodes=" << mesh.numNodes
                 << " elems=" << mesh.numElems
                 << " numCkpt=" << db.numCkpt
                 << " segLen=" << db.segLen
                 << " fullHistoryTotalGB=" << memDiag.fullHistoryTotalGB
                 << " fullHistoryRankMaxGB=" << memDiag.fullHistoryRankMaxGB
                 << " checkpointStorageTotalGB=" << memDiag.checkpointStorageTotalGB
                 << " checkpointStorageRankMaxGB=" << memDiag.checkpointStorageRankMaxGB
                 << " hostFullStorageTotalGB=" << memDiag.hostFullStorageTotalGB
                 << " hostFullStorageRankMaxGB=" << memDiag.hostFullStorageRankMaxGB
                 << " segmentBufferRankMaxGB=" << memDiag.segmentBufferRankMaxGB
                 << " checkpointTmpRankMaxGB=" << memDiag.checkpointTmpRankMaxGB
                 << " sensitivityBufferRankMaxGB=" << memDiag.sensitivityBufferRankMaxGB
                 << " historyGpuRankMaxGB=" << memDiag.historyGpuRankMaxGB
                 << " gpuAfterBufferInitGB=" << memDiag.gpuAfterBufferInitGB
                 << endl;
        }

        Vec workT = nullptr, workLam = nullptr;
        VecScatter scatterT = nullptr, scatterLam = nullptr;
        createScatterToWork3D(ps.Tn, part, gUseCudaBackend, &workT, &scatterT);
        PETSC_CHECK_THROW(VecDuplicate(workT, &workLam));
        createScatterToExistingWork3D(ps.lambda, part, workLam, &scatterLam);

        const int nd = static_cast<int>(mesh.numElems);
        const int ndLocal = static_cast<int>(part.localDesignElems.size());
        vector<int> designCounts, designDispls, gatheredElemIds;
        vector<double> gatheredDcDxLocal;
        int localDesignCount = ndLocal;
        if (rank == 0) designCounts.resize(size);
        MPI_Gather(&localDesignCount, 1, MPI_INT, rank == 0 ? designCounts.data() : nullptr, 1, MPI_INT, 0, comm);
        int totalGatheredDesign = 0;
        if (rank == 0) {
            designDispls.resize(size, 0);
            for (int r = 0; r < size; ++r) { designDispls[r] = totalGatheredDesign; totalGatheredDesign += designCounts[r]; }
            gatheredElemIds.resize(totalGatheredDesign);
            gatheredDcDxLocal.resize(totalGatheredDesign);
        }
        MPI_Gatherv(ndLocal > 0 ? part.localDesignElems.data() : nullptr, ndLocal, MPI_INT,
                    rank == 0 ? gatheredElemIds.data() : nullptr,
                    rank == 0 ? designCounts.data() : nullptr,
                    rank == 0 ? designDispls.data() : nullptr,
                    MPI_INT, 0, comm);

        vector<double> x(static_cast<size_t>(mesh.numElems), cfg.volumeFraction);
        VectorXd xEigen, xminVec, xmaxVec, xold1, xold2, low, upp;
        if (rank == 0) {
            xEigen = VectorXd::Constant(mesh.numElems, cfg.volumeFraction);
            xminVec = VectorXd::Constant(mesh.numElems, 1e-3);
            xmaxVec = VectorXd::Ones(mesh.numElems);
            xold1 = xEigen; xold2 = xEigen; low = xminVec; upp = xmaxVec;

            if (!cfg.restartVtk.empty()) {
                const double betaRestart = betaAfterCompletedIterations(cfg, cfg.restartIter);
                vector<double> rhoRestart = loadDensityFromVTK(cfg.restartVtk, mesh.numElems);
                for (PetscInt e = 0; e < mesh.numElems; ++e) {
                    const double rho = rhoRestart[static_cast<size_t>(e)];
                    const double xin = cfg.restartInvertProjection
                        ? inverseHeavisideProjection(rho, betaRestart, cfg.eta)
                        : std::clamp(rho, 1.0e-3, 1.0);
                    xEigen(static_cast<Eigen::Index>(e)) = xin;
                }
                xold1 = xEigen;
                xold2 = xEigen;
                low = xminVec;
                upp = xmaxVec;
                x = toStd(xEigen);

                cout << "RESTART_FROM_VTK"
                     << " file=" << cfg.restartVtk
                     << " restartIter=" << cfg.restartIter
                     << " betaRestart=" << betaRestart
                     << " invertProjection=" << (cfg.restartInvertProjection ? 1 : 0)
                     << " meanX=" << xEigen.mean()
                     << endl;
            }
        }
        MPI_Bcast(x.data(), nd, MPI_DOUBLE, 0, comm);

        const double dt = cfg.tf / static_cast<double>(cfg.nt);
        vector<double> Q_t(static_cast<size_t>(cfg.nt + 1), 0.0);
        for (int n = 0; n <= cfg.nt; ++n) {
            const double t = n * dt;
            Q_t[static_cast<size_t>(n)] = cfg.q0 * qScale(t, cfg);
        }

        double beta = betaAfterCompletedIterations(cfg, cfg.restartIter);
        double change = 1e9;
        int iter = std::max(0, cfg.restartIter);
        int smallChangeStreak = 0;
        string stopReason = "running";

        while (true) {
            int doIter = 0;
            if (rank == 0) {
                if (cfg.benchmarkFixedDesign) {
                    const int benchmarkTotal = cfg.benchmarkWarmupIters + cfg.benchmarkMeasuredIters;
                    doIter = (iter < benchmarkTotal) ? 1 : 0;
                } else {
                    const bool hitMaxIter = (iter >= cfg.maxIter);
                    const bool minimumIterationReached = (iter >= cfg.minIterBeforeStop);
                    const bool finalProjectionStage = (beta >= cfg.betaMax);
                    const bool changeIsStable = (smallChangeStreak >= cfg.changeStableIters);
                    const bool converged = minimumIterationReached && finalProjectionStage && changeIsStable;
                    doIter = (!hitMaxIter && !converged) ? 1 : 0;
                    if (!doIter) stopReason = hitMaxIter ? "max_iter" : "stable_change_at_final_beta";
                }
            }
            MPI_Bcast(&doIter, 1, MPI_INT, 0, comm);
            if (!doIter) break;
            ++iter;
            const bool isWarmup = cfg.benchmarkFixedDesign && iter <= cfg.benchmarkWarmupIters;
            const bool isMeasured = !cfg.benchmarkFixedDesign || !isWarmup;
            const int benchmarkRep = cfg.benchmarkFixedDesign ? (iter - cfg.benchmarkWarmupIters) : iter;

            IterStageTimes stage;
            HistoryTransferStats historyTransfer;
            const double tIter0 = stageBegin(comm);

            // betaUsed belongs to the state, objective and gradient evaluated
            // in this iteration. beta may be advanced only after the MMA step.
            const double betaUsed = beta;
            Projection3D pr;
            {
                const double t0 = stageBegin(comm);
                if (rank == 0) pr = project3D(fctx, mesh, x, betaUsed, cfg.eta);
                else {
                    pr.xPhys.resize(x.size());
                    if (ndLocal > 0) {} // silence warnings
                }
                MPI_Bcast(pr.xPhys.data(), nd, MPI_DOUBLE, 0, comm);
                stage.filter = stageElapsedMax(comm, t0);
            }

            {
                const double t0 = stageBegin(comm);
                assembleSystemDistributed(cfg, mesh, part, pr.xPhys, patchWeight, coolingStrips, KeGeo, CeGeo, ps);
                stage.assemble = stageElapsedMax(comm, t0);
            }

            {
                const double t0 = stageBegin(comm);
                rebuildOperator(cfg, ps);
                PETSC_CHECK_THROW(KSPSetOperators(ps.ksp, ps.A, ps.A));
                stage.operatorBuild = stageElapsedMax(comm, t0);
            }

            {
                const double t0 = stageBegin(comm);
                PETSC_CHECK_THROW(KSPSetUp(ps.ksp));
                stage.pcSetup = stageElapsedMax(comm, t0);
                stage.memAfterPCSetupGB = cudaUsedGBMax(comm);
                stage.fixedSolverPCGB = std::max(0.0, stage.memAfterPCSetupGB - memDiag.historyGpuRankMaxGB);
                updatePeakGpuGBAll(comm, stage.peakGpuGB);
            }

            PETSC_CHECK_THROW(VecSet(ps.Tn, cfg.Tc));
            if (cfg.historyMode == "full") {
                scatterGlobalToWork3D(scatterT, ps.Tn, workT);
                storeWorkVecColumnToHistory(workT, part.nWorkNodes, 0, db);
            } else if (cfg.historyMode == "host_full") {
                storeOwnedVecColumnToHostFull(ps.Tn, part.nOwnedNodes, 0, db, historyTransfer);
            } else {
                storeOwnedVecColumnToCheckpoint(ps.Tn, part.nOwnedNodes, 0, db, historyTransfer);
            }

            double J = 0.0;
            double gradientL2 = 0.0;
            double gradientLinf = 0.0;
            double gradientSum = 0.0;
            vector<double> TlastGlobal;
            {
                const double t0 = stageBegin(comm);
                long long fwdItsSum = 0;
                long long fwdSolves = 0;
                for (int n = 0; n < cfg.nt; ++n) {
                    applyBOperator(cfg, ps, ps.Tn, ps.rhs);
                    PETSC_CHECK_THROW(VecAXPY(ps.rhs, Q_t[static_cast<size_t>(n + 1)], ps.Fshape));
                    PETSC_CHECK_THROW(VecAXPY(ps.rhs, 1.0, ps.Fcool));
                    PETSC_CHECK_THROW(KSPSolve(ps.ksp, ps.rhs, ps.Tn));
                    recordKspOutcome(ps.ksp, stage);
                    PetscInt itsNow = 0;
                    PETSC_CHECK_THROW(KSPGetIterationNumber(ps.ksp, &itsNow));
                    fwdItsSum += static_cast<long long>(itsNow);
                    ++fwdSolves;

                    PetscScalar dotq = 0.0;
                    PETSC_CHECK_THROW(VecDot(ps.Tn, ps.Fshape, &dotq));
                    J += dt * Q_t[static_cast<size_t>(n + 1)] * PetscRealPart(dotq);

                    if (cfg.historyMode == "full") {
                        scatterGlobalToWork3D(scatterT, ps.Tn, workT);
                        storeWorkVecColumnToHistory(workT, part.nWorkNodes, n + 1, db);
                    } else if (cfg.historyMode == "host_full") {
                        storeOwnedVecColumnToHostFull(ps.Tn, part.nOwnedNodes, n + 1, db, historyTransfer);
                    } else if ((n + 1) % cfg.checkpointInterval == 0) {
                        storeOwnedVecColumnToCheckpoint(ps.Tn, part.nOwnedNodes, (n + 1) / cfg.checkpointInterval, db, historyTransfer);
                    }

                    if (rank == 0 && cfg.stepPrintEvery > 0 &&
                        (((n + 1) % cfg.stepPrintEvery == 0) || (n + 1 == cfg.nt))) {
                        cout << "  [forward] step=" << (n + 1) << "/" << cfg.nt
                             << ", q=" << Q_t[static_cast<size_t>(n + 1)]
                             << ", kspIts=" << itsNow
                             << ", partialJ=" << J << endl;
                    }
                }
                stage.avgFwdKspIts = (fwdSolves > 0) ? static_cast<double>(fwdItsSum) / static_cast<double>(fwdSolves) : 0.0;
                stage.forwardSolves = fwdSolves;
                stage.forward = stageElapsedMax(comm, t0);
                updatePeakGpuGBAll(comm, stage.peakGpuGB);
            }

            vector<double> dkLocal(static_cast<size_t>(ndLocal)), dcvLocal(static_cast<size_t>(ndLocal)), dc_dxPhysLocal(static_cast<size_t>(ndLocal), 0.0);
            {
                const double t0 = stageBegin(comm);
                for (int k = 0; k < ndLocal; ++k) {
                    const int e = part.localDesignElems[static_cast<size_t>(k)];
                    const double xp = std::max(pr.xPhys[static_cast<size_t>(e)], 1e-12);
                    dkLocal[static_cast<size_t>(k)] = cfg.penalK * (cfg.kHigh - cfg.kLow) * std::pow(xp, cfg.penalK - 1.0);
                    dcvLocal[static_cast<size_t>(k)] = cfg.penalC * (cfg.cvHigh - cfg.cvLow) * std::pow(xp, cfg.penalC - 1.0);
                }
                if (gUseCudaBackend) {
                    CUDA_SAFE(gpuMemcpyHtoD(db.dk, dkLocal.data(), sizeof(double) * dkLocal.size()));
                    CUDA_SAFE(gpuMemcpyHtoD(db.dcv, dcvLocal.data(), sizeof(double) * dcvLocal.size()));
                    CUDA_SAFE(gpuZeroDoubles(db.dc_dx, static_cast<size_t>(ndLocal)));
                } else {
                    std::fill(dc_dxPhysLocal.begin(), dc_dxPhysLocal.end(), 0.0);
                }
                PETSC_CHECK_THROW(VecSet(ps.lambda, 0.0));
                stage.sensPrep = stageElapsedMax(comm, t0);
            }

            {
                const double t0 = stageBegin(comm);
                long long adjItsSum = 0;
                long long adjSolves = 0;

                if (cfg.historyMode == "full") {
                    for (int ng = cfg.nt - 1; ng >= 0; --ng) {
                        const int revStep = cfg.nt - ng;
                        scatterGlobalToWork3D(scatterLam, ps.lambda, workLam);
                        accumulateSensitivityStep(part, db, workLam, ng, ng + 1,
                                                  dkLocal, dcvLocal, dt, KeGeo, CeGeo, dc_dxPhysLocal);

                        applyBOperator(cfg, ps, ps.lambda, ps.rhs);
                        PETSC_CHECK_THROW(VecAXPY(ps.rhs, -Q_t[static_cast<size_t>(ng + 1)], ps.Fshape));
                        PETSC_CHECK_THROW(KSPSolve(ps.ksp, ps.rhs, ps.lambda));
                        recordKspOutcome(ps.ksp, stage);
                        PetscInt itsNow = 0;
                        PETSC_CHECK_THROW(KSPGetIterationNumber(ps.ksp, &itsNow));
                        adjItsSum += static_cast<long long>(itsNow);
                        ++adjSolves;

                        if (rank == 0 && cfg.stepPrintEvery > 0 &&
                            ((revStep % cfg.stepPrintEvery == 0) || (revStep == cfg.nt))) {
                            cout << "  [adjoint-full] revStep=" << revStep << "/" << cfg.nt
                                 << ", globalStep=" << (ng + 1)
                                 << ", kspIts=" << itsNow << endl;
                        }
                    }
                } else if (cfg.historyMode == "host_full") {
                    // Host full-history mode: avoid forward recomputation.
                    // For each reverse step, load T^n and T^{n+1} from CPU RAM to GPU,
                    // scatter them to work/ghost layout, and reuse the same two-column T_seg buffer.
                    for (int ng = cfg.nt - 1; ng >= 0; --ng) {
                        const int revStep = cfg.nt - ng;

                        loadHostFullOwnedSliceToDistributedVec(part, db, ng, ps.Tn, historyTransfer);
                        scatterGlobalToWork3D(scatterT, ps.Tn, workT);
                        storeWorkVecColumnToHistory(workT, part.nWorkNodes, 0, db);

                        loadHostFullOwnedSliceToDistributedVec(part, db, ng + 1, ps.Tn, historyTransfer);
                        scatterGlobalToWork3D(scatterT, ps.Tn, workT);
                        storeWorkVecColumnToHistory(workT, part.nWorkNodes, 1, db);

                        scatterGlobalToWork3D(scatterLam, ps.lambda, workLam);
                        accumulateSensitivityStep(part, db, workLam, 0, 1,
                                                  dkLocal, dcvLocal, dt, KeGeo, CeGeo, dc_dxPhysLocal);

                        applyBOperator(cfg, ps, ps.lambda, ps.rhs);
                        PETSC_CHECK_THROW(VecAXPY(ps.rhs, -Q_t[static_cast<size_t>(ng + 1)], ps.Fshape));
                        PETSC_CHECK_THROW(KSPSolve(ps.ksp, ps.rhs, ps.lambda));
                        recordKspOutcome(ps.ksp, stage);
                        PetscInt itsNow = 0;
                        PETSC_CHECK_THROW(KSPGetIterationNumber(ps.ksp, &itsNow));
                        adjItsSum += static_cast<long long>(itsNow);
                        ++adjSolves;

                        if (rank == 0 && cfg.stepPrintEvery > 0 &&
                            ((revStep % cfg.stepPrintEvery == 0) || (revStep == cfg.nt))) {
                            cout << "  [adjoint-host-full] revStep=" << revStep << "/" << cfg.nt
                                 << ", globalStep=" << (ng + 1)
                                 << ", kspIts=" << itsNow << endl;
                        }
                    }
                } else {
                    const int segLen = cfg.checkpointInterval;
                    const int numFullSegs = cfg.nt / segLen;
                    const int tailLen = cfg.nt - numFullSegs * segLen;
                    const int segCount = numFullSegs + (tailLen > 0 ? 1 : 0);

                    for (int seg = segCount - 1; seg >= 0; --seg) {
                        int segStart = 0, segEnd = 0, thisSegLen = 0;
                        if (tailLen > 0 && seg == numFullSegs) { segStart = numFullSegs * segLen; segEnd = cfg.nt; }
                        else { segStart = seg * segLen; segEnd = std::min(cfg.nt, segStart + segLen); }
                        thisSegLen = segEnd - segStart;
                        const int ckIdx = segStart / segLen;

                        loadCheckpointOwnedSliceToDistributedVec(part, db, ckIdx, ps.Tn, historyTransfer);
                        scatterGlobalToWork3D(scatterT, ps.Tn, workT);
                        storeWorkVecColumnToHistory(workT, part.nWorkNodes, 0, db);
                        for (int localStep = 0; localStep < thisSegLen; ++localStep) {
                            const int globalStep = segStart + localStep;
                            applyBOperator(cfg, ps, ps.Tn, ps.rhs);
                            PETSC_CHECK_THROW(VecAXPY(ps.rhs, Q_t[static_cast<size_t>(globalStep + 1)], ps.Fshape));
                            PETSC_CHECK_THROW(VecAXPY(ps.rhs, 1.0, ps.Fcool));
                            PETSC_CHECK_THROW(KSPSolve(ps.ksp, ps.rhs, ps.Tn));
                            recordKspOutcome(ps.ksp, stage);
                            ++stage.recomputeSolves;
                            scatterGlobalToWork3D(scatterT, ps.Tn, workT);
                            storeWorkVecColumnToHistory(workT, part.nWorkNodes, localStep + 1, db);
                        }

                        for (int localStep = thisSegLen - 1; localStep >= 0; --localStep) {
                            const int ng = segStart + localStep;
                            const int revStep = cfg.nt - ng;
                            scatterGlobalToWork3D(scatterLam, ps.lambda, workLam);
                            accumulateSensitivityStep(part, db, workLam, localStep, localStep + 1,
                                                      dkLocal, dcvLocal, dt, KeGeo, CeGeo, dc_dxPhysLocal);

                            applyBOperator(cfg, ps, ps.lambda, ps.rhs);
                            PETSC_CHECK_THROW(VecAXPY(ps.rhs, -Q_t[static_cast<size_t>(ng + 1)], ps.Fshape));
                            PETSC_CHECK_THROW(KSPSolve(ps.ksp, ps.rhs, ps.lambda));
                            recordKspOutcome(ps.ksp, stage);
                            PetscInt itsNow = 0;
                            PETSC_CHECK_THROW(KSPGetIterationNumber(ps.ksp, &itsNow));
                            adjItsSum += static_cast<long long>(itsNow);
                            ++adjSolves;

                            if (rank == 0 && cfg.stepPrintEvery > 0 &&
                                ((revStep % cfg.stepPrintEvery == 0) || (revStep == cfg.nt))) {
                                cout << "  [adjoint] revStep=" << revStep << "/" << cfg.nt
                                     << ", globalStep=" << (ng + 1)
                                     << ", kspIts=" << itsNow << endl;
                            }
                        }
                    }
                }
                if (gUseCudaBackend) {
                    CUDA_SAFE(gpuMemcpyDtoH(dc_dxPhysLocal.data(), db.dc_dx, sizeof(double) * dc_dxPhysLocal.size()));
                }
                stage.avgAdjKspIts = (adjSolves > 0) ? static_cast<double>(adjItsSum) / static_cast<double>(adjSolves) : 0.0;
                stage.adjointSolves = adjSolves;
                stage.avgKspIts = (stage.avgFwdKspIts + stage.avgAdjKspIts) * 0.5;
                stage.adjoint = stageElapsedMax(comm, t0);
                updatePeakGpuGBAll(comm, stage.peakGpuGB);
            }

            {
                const double t0 = stageBegin(comm);
                MPI_Gatherv(ndLocal > 0 ? dc_dxPhysLocal.data() : nullptr, ndLocal, MPI_DOUBLE,
                            rank == 0 ? gatheredDcDxLocal.data() : nullptr,
                            rank == 0 ? designCounts.data() : nullptr,
                            rank == 0 ? designDispls.data() : nullptr,
                            MPI_DOUBLE, 0, comm);

                if (rank == 0) {
                    VectorXd dc_dxPhysGlobal = VectorXd::Zero(nd);
                    for (int k = 0; k < totalGatheredDesign; ++k) dc_dxPhysGlobal(gatheredElemIds[k]) = gatheredDcDxLocal[k];
                    vector<double> temp(pr.xPhys.size()), volTemp(pr.xPhys.size());
                    for (size_t e = 0; e < pr.xPhys.size(); ++e) {
                        temp[e] = dc_dxPhysGlobal(static_cast<Eigen::Index>(e)) * pr.dProj[e] / fctx.Hs[e];
                        volTemp[e] = pr.dProj[e] / fctx.Hs[e];
                    }
                    const auto dc_dx_vec = filterTransposeWeighted3D(fctx, mesh, temp);
                    const auto dfdx0_vec = filterTransposeWeighted3D(fctx, mesh, volTemp);
                    const VectorXd dc_dx = toEigen(dc_dx_vec);
                    const VectorXd dfdx0 = toEigen(dfdx0_vec);
                    gradientL2 = dc_dx.norm();
                    gradientLinf = dc_dx.cwiseAbs().maxCoeff();
                    gradientSum = dc_dx.sum();
                    const double fval0 = std::accumulate(pr.xPhys.begin(), pr.xPhys.end(), 0.0) - cfg.volumeFraction * static_cast<double>(mesh.numElems);

                    if (!cfg.useMMA) throw std::runtime_error("Only MMA path is implemented.");
                    MMAResult mmaRes = mmasub_m1(
                        static_cast<int>(mesh.numElems), iter,
                        xEigen, xminVec, xmaxVec,
                        xold1, xold2,
                        J, dc_dx,
                        fval0, dfdx0,
                        low, upp,
                        cfg.mma_a0, cfg.mma_a, cfg.mma_c, cfg.mma_d);
                    if (cfg.benchmarkFixedDesign) {
                        // Execute MMA so the full iteration cost is represented, but do not
                        // change x, beta or the subsequent measured repetitions.
                        change = 0.0;
                    } else {
                        VectorXd xnew = mmaRes.xmma.cwiseMax(xminVec).cwiseMin(xmaxVec);
                        change = (xnew - xEigen).cwiseAbs().maxCoeff();
                        xold2 = xold1; xold1 = xEigen; low = mmaRes.low; upp = mmaRes.upp; xEigen = xnew; x = toStd(xEigen);
                        const double betaBeforeUpdate = beta;
                        if (iter % cfg.betaUpdateEvery == 0 && beta < cfg.betaMax) beta = std::min(2.0 * beta, cfg.betaMax);
                        const bool betaUpdated = (beta > betaBeforeUpdate);
                        if (betaUpdated || beta < cfg.betaMax || iter < cfg.minIterBeforeStop) {
                            smallChangeStreak = 0;
                        } else if (change <= cfg.changeTol) {
                            ++smallChangeStreak;
                        } else {
                            smallChangeStreak = 0;
                        }
                    }

                    cout << "iter=" << iter
                         << ", J=" << J
                         << ", vol=" << (std::accumulate(pr.xPhys.begin(), pr.xPhys.end(), 0.0) / static_cast<double>(mesh.numElems))
                         << ", change=" << change
                         << ", betaUsed=" << betaUsed
                         << ", betaNext=" << beta
                         << ", changeStreak=" << smallChangeStreak
                         << ", sample=" << (isWarmup ? "warmup" : "measured")
                         << ", benchmarkRep=" << benchmarkRep << endl;
                    cout << "  timing[s]: filter=" << stage.filter
                         << ", asm=" << stage.assemble
                         << ", formA=" << stage.operatorBuild
                         << ", pcSetup=" << stage.pcSetup
                         << ", forward=" << stage.forward
                         << ", sensPrep=" << stage.sensPrep
                         << ", adjoint=" << stage.adjoint
                         << ", avgFwdIts=" << stage.avgFwdKspIts
                         << ", avgAdjIts=" << stage.avgAdjKspIts
                         << ", avgIts=" << stage.avgKspIts
                         << ", peakGpuGB=" << stage.peakGpuGB
                         << ", memAfterPCSetupGB=" << stage.memAfterPCSetupGB
                         << ", fixedSolverPCGB=" << stage.fixedSolverPCGB
                         << ", memoryBottleneck=" << classifyMemoryBottleneck(memDiag, stage.fixedSolverPCGB);
                }
                stage.update = stageElapsedMax(comm, t0);
            }

            {
                const double t0 = stageBegin(comm);
                if (rank == 0) {
                    cout << ", update=" << stage.update;
                }
                if (!cfg.benchmarkFixedDesign && cfg.writeVTK && (iter % cfg.writeEveryIter == 0)) {
                    TlastGlobal = gatherDistributedVecToRoot(ps.Tn, comm);
                    if (rank == 0) {
                        // Publish a VTK file only after it has been written completely.
                        // A power loss during output may leave the .tmp file incomplete,
                        // but the automatic restart launcher deliberately ignores .tmp.
                        const fs::path vtkPath = fs::path(cfg.outDir) /
                            (cfg.outPrefix + "_iter" + std::to_string(iter) + ".vtk");
                        const fs::path vtkTmpPath = vtkPath.string() + ".tmp";
                        writeVTK(vtkTmpPath.string(), mesh, TlastGlobal, pr.xPhys);
                        std::error_code renameError;
                        fs::rename(vtkTmpPath, vtkPath, renameError);
                        if (renameError) {
                            throw std::runtime_error("Failed to publish completed VTK file: " +
                                                     vtkPath.string() + ": " + renameError.message());
                        }
                    }
                }
                stage.vtk = stageElapsedMax(comm, t0);
            }

            stage.total = stageElapsedMax(comm, tIter0);
            {
                const double localRss = processPeakRssGB();
                MPI_Allreduce(&localRss, &stage.hostPeakRssRankMaxGB, 1, MPI_DOUBLE, MPI_MAX, comm);
                MPI_Allreduce(&localRss, &stage.hostPeakRssTotalGB, 1, MPI_DOUBLE, MPI_SUM, comm);
                unsigned long long localH2D = static_cast<unsigned long long>(historyTransfer.h2dBytes);
                unsigned long long localD2H = static_cast<unsigned long long>(historyTransfer.d2hBytes);
                unsigned long long globalH2D = 0, globalD2H = 0;
                MPI_Allreduce(&localH2D, &globalH2D, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
                MPI_Allreduce(&localD2H, &globalD2H, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
                stage.historyH2DBytesTotal = static_cast<std::uint64_t>(globalH2D);
                stage.historyD2HBytesTotal = static_cast<std::uint64_t>(globalD2H);
                MPI_Allreduce(&historyTransfer.h2dSeconds, &stage.historyH2DTimeRankMax, 1, MPI_DOUBLE, MPI_MAX, comm);
                MPI_Allreduce(&historyTransfer.d2hSeconds, &stage.historyD2HTimeRankMax, 1, MPI_DOUBLE, MPI_MAX, comm);
                double globalResidual = 0.0;
                long long globalFailed = 0;
                MPI_Allreduce(&stage.maxKspResidual, &globalResidual, 1, MPI_DOUBLE, MPI_MAX, comm);
                MPI_Allreduce(&stage.failedKspSolves, &globalFailed, 1, MPI_LONG_LONG_INT, MPI_MAX, comm);
                stage.maxKspResidual = globalResidual;
                stage.failedKspSolves = globalFailed;
            }
            if (rank == 0 && isMeasured) {
                cout << ", vtk=" << stage.vtk
                     << ", total=" << stage.total << endl;
                cout << "BENCH iter=" << iter
                     << " benchmarkRep=" << benchmarkRep
                     << " backend=" << cfg.backend
                     << " ranks=" << size
                     << " matType=" << cfg.matType
                     << " vecType=" << cfg.vecType
                     << " kspType=" << cfg.kspType
                     << " pcType=" << cfg.pcType
                     << " kspRtol=" << cfg.kspRtol
                     << " historyHostMemory=" << cfg.historyHostMemory
                     << " J=" << J
                     << " gradientL2=" << gradientL2
                     << " gradientLinf=" << gradientLinf
                     << " gradientSum=" << gradientSum
                     << " filter=" << stage.filter
                     << " assemble=" << stage.assemble
                     << " operatorBuild=" << stage.operatorBuild
                     << " pcSetup=" << stage.pcSetup
                     << " avgKspIts=" << stage.avgKspIts
                     << " avgFwdIts=" << stage.avgFwdKspIts
                     << " avgAdjIts=" << stage.avgAdjKspIts
                     << " forward=" << stage.forward
                     << " sensPrep=" << stage.sensPrep
                     << " adjoint=" << stage.adjoint
                     << " update=" << stage.update
                     << " vtk=" << stage.vtk
                     << " recomputeSolves=" << stage.recomputeSolves
                     << " forwardSolves=" << stage.forwardSolves
                     << " adjointSolves=" << stage.adjointSolves
                     << " total=" << stage.total
                     << " maxKspResidual=" << stage.maxKspResidual
                     << " failedKspSolves=" << stage.failedKspSolves
                     << " peakGpuGB=" << stage.peakGpuGB
                     << " hostPeakRssRankMaxGB=" << stage.hostPeakRssRankMaxGB
                     << " hostPeakRssTotalGB=" << stage.hostPeakRssTotalGB
                     << " historyH2DBytesTotal=" << stage.historyH2DBytesTotal
                     << " historyD2HBytesTotal=" << stage.historyD2HBytesTotal
                     << " historyH2DTimeRankMax=" << stage.historyH2DTimeRankMax
                     << " historyD2HTimeRankMax=" << stage.historyD2HTimeRankMax
                     << " usefulDofStepsPerSec=" << (static_cast<double>(mesh.numNodes) * 2.0 * cfg.nt / std::max(stage.total, 1e-30))
                     << " executedDofStepsPerSec=" << (static_cast<double>(mesh.numNodes) * (2.0 * cfg.nt + stage.recomputeSolves) / std::max(stage.total, 1e-30))
                     << " memAfterPCSetupGB=" << stage.memAfterPCSetupGB
                     << " fixedSolverPCGB=" << stage.fixedSolverPCGB
                     << " fullHistoryTotalGB=" << memDiag.fullHistoryTotalGB
                     << " checkpointStorageTotalGB=" << memDiag.checkpointStorageTotalGB
                     << " checkpointStorageRankMaxGB=" << memDiag.checkpointStorageRankMaxGB
                     << " hostFullStorageTotalGB=" << memDiag.hostFullStorageTotalGB
                     << " hostFullStorageRankMaxGB=" << memDiag.hostFullStorageRankMaxGB
                     << " segmentBufferRankMaxGB=" << memDiag.segmentBufferRankMaxGB
                     << " historyGpuRankMaxGB=" << memDiag.historyGpuRankMaxGB
                     << " memoryBottleneck=" << classifyMemoryBottleneck(memDiag, stage.fixedSolverPCGB)
                     << " historyMode=" << cfg.historyMode << endl;
                cout << "TABLE12 iter=" << iter
                     << " ranks=" << size
                     << " interval=" << cfg.checkpointInterval
                     << " nt=" << cfg.nt
                     << " nodes=" << mesh.numNodes
                     << " elems=" << mesh.numElems
                     << " fullHistoryTotalGB=" << memDiag.fullHistoryTotalGB
                     << " checkpointStorageTotalGB=" << memDiag.checkpointStorageTotalGB
                     << " checkpointStorageRankMaxGB=" << memDiag.checkpointStorageRankMaxGB
                     << " hostFullStorageTotalGB=" << memDiag.hostFullStorageTotalGB
                     << " hostFullStorageRankMaxGB=" << memDiag.hostFullStorageRankMaxGB
                     << " segmentBufferRankMaxGB=" << memDiag.segmentBufferRankMaxGB
                     << " historyGpuRankMaxGB=" << memDiag.historyGpuRankMaxGB
                     << " memAfterPCSetupGB=" << stage.memAfterPCSetupGB
                     << " fixedSolverPCGB=" << stage.fixedSolverPCGB
                     << " peakGpuGB=" << stage.peakGpuGB
                     << " pcSetup=" << stage.pcSetup
                     << " avgKspIts=" << stage.avgKspIts
                     << " forward=" << stage.forward
                     << " adjoint=" << stage.adjoint
                     << " total=" << stage.total
                     << " memoryBottleneck=" << classifyMemoryBottleneck(memDiag, stage.fixedSolverPCGB)
                     << " status=OK" << endl;
            } else if (rank == 0 && isWarmup) {
                cout << ", vtk=" << stage.vtk << ", total=" << stage.total << endl;
                cout << "WARMUP iter=" << iter << " total=" << stage.total << endl;
            }

            MPI_Bcast(&change, 1, MPI_DOUBLE, 0, comm);
            MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, comm);
            MPI_Bcast(x.data(), nd, MPI_DOUBLE, 0, comm);
        }
        if (rank == 0 && !cfg.benchmarkFixedDesign) {
            cout << "OPTIMIZATION_STOP reason=" << stopReason
                 << " iter=" << iter
                 << " change=" << change
                 << " beta=" << beta
                 << " changeStreak=" << smallChangeStreak
                 << "/" << cfg.changeStableIters
                 << " minIterBeforeStop=" << cfg.minIterBeforeStop << endl;
        }
        if (rank == 0 && cfg.saveFinalDensity && !cfg.benchmarkFixedDesign) {
            auto prFinal = project3D(fctx, mesh, x, beta, cfg.eta);
            saveDensityCSV((fs::path(cfg.outDir) / (cfg.outPrefix + "_density.csv")).string(), mesh, prFinal.xPhys);
        }

        if (scatterT) PETSC_CHECK_THROW(VecScatterDestroy(&scatterT));
        if (scatterLam) PETSC_CHECK_THROW(VecScatterDestroy(&scatterLam));
        if (workT) PETSC_CHECK_THROW(VecDestroy(&workT));
        if (workLam) PETSC_CHECK_THROW(VecDestroy(&workLam));
        destroyDeviceBuffers3D(db);
        if (ps.ksp) PETSC_CHECK_THROW(KSPDestroy(&ps.ksp));
        if (ps.K) PETSC_CHECK_THROW(MatDestroy(&ps.K));
        if (ps.A) PETSC_CHECK_THROW(MatDestroy(&ps.A));
        if (ps.Cdiag) PETSC_CHECK_THROW(VecDestroy(&ps.Cdiag));
        if (ps.Fshape) PETSC_CHECK_THROW(VecDestroy(&ps.Fshape));
        if (ps.Fcool) PETSC_CHECK_THROW(VecDestroy(&ps.Fcool));
        if (ps.Tn) PETSC_CHECK_THROW(VecDestroy(&ps.Tn));
        if (ps.rhs) PETSC_CHECK_THROW(VecDestroy(&ps.rhs));
        if (ps.lambda) PETSC_CHECK_THROW(VecDestroy(&ps.lambda));
        if (ps.tmp) PETSC_CHECK_THROW(VecDestroy(&ps.tmp));

        PetscFinalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        PetscFinalize();
        return 1;
    }
}
