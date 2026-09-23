/*
 * Copyright (C) 2026 Yibo Yang
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gpu_kernels.h"
#include <cuda_runtime.h>

static constexpr int BLOCK = 256;

cudaError_t gpuAllocDoubles(double** ptr, size_t n) { return cudaMalloc((void**)ptr, n * sizeof(double)); }
cudaError_t gpuAllocFloats(float** ptr, size_t n) { return cudaMalloc((void**)ptr, n * sizeof(float)); }
cudaError_t gpuAllocInts(int** ptr, size_t n) { return cudaMalloc((void**)ptr, n * sizeof(int)); }
cudaError_t gpuFreePtr(void* ptr) { return ptr ? cudaFree(ptr) : cudaSuccess; }
cudaError_t gpuMemcpyHtoD(void* dst, const void* src, size_t bytes) { return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); }
cudaError_t gpuMemcpyDtoH(void* dst, const void* src, size_t bytes) { return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); }
cudaError_t gpuZeroDoubles(double* ptr, size_t n) { return cudaMemset(ptr, 0, n * sizeof(double)); }
cudaError_t gpuZeroFloats(float* ptr, size_t n) { return cudaMemset(ptr, 0, n * sizeof(float)); }
cudaError_t gpuMemcpyDtoD(void* dst, const void* src, size_t bytes) { return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice); }
cudaError_t gpuStoreHistoryColumn(int n, int col, const double* d_vec, double* d_hist) {
    return cudaMemcpy(d_hist + static_cast<size_t>(col) * static_cast<size_t>(n), d_vec, static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToDevice);
}
cudaError_t gpuCopyColumn(int n, const double* d_hist, int srcCol, double* d_hist2, int dstCol) {
    const double* src = d_hist + static_cast<size_t>(srcCol) * static_cast<size_t>(n);
    double* dst = d_hist2 + static_cast<size_t>(dstCol) * static_cast<size_t>(n);
    return cudaMemcpy(dst, src, static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToDevice);
}

__global__ static void initHist0Kernel(int n, double value, double* hist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hist[i] = value;
}

cudaError_t gpuInitHistory0(int n, double value, double* d_hist)
{
    initHist0Kernel << <(n + BLOCK - 1) / BLOCK, BLOCK >> > (n, value, d_hist);
    return cudaGetLastError();
}

__global__ static void initHist0KernelFloat(int n, float value, float* hist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hist[i] = value;
}

cudaError_t gpuInitHistory0Float(int n, double value, float* d_hist)
{
    initHist0KernelFloat<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(n, static_cast<float>(value), d_hist);
    return cudaGetLastError();
}

__global__ static void storeHistoryColumnFloatKernel(int n, const double* __restrict__ src, float* __restrict__ dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<float>(src[i]);
}

cudaError_t gpuStoreHistoryColumnFloat(int n, int col, const double* d_vec, float* d_hist)
{
    float* dst = d_hist + static_cast<size_t>(col) * static_cast<size_t>(n);
    storeHistoryColumnFloatKernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(n, d_vec, dst);
    return cudaGetLastError();
}

cudaError_t gpuCopyColumnFloat(int n, const float* d_hist, int srcCol, float* d_hist2, int dstCol)
{
    const float* src = d_hist + static_cast<size_t>(srcCol) * static_cast<size_t>(n);
    float* dst = d_hist2 + static_cast<size_t>(dstCol) * static_cast<size_t>(n);
    return cudaMemcpy(dst, src, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice);
}

__global__ static void copyFloatToDoubleKernel(size_t n, const float* __restrict__ src, double* __restrict__ dst)
{
    size_t i = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (i < n) dst[i] = static_cast<double>(src[i]);
}

cudaError_t gpuCopyFloatToDouble(double* dst, const float* src, size_t count)
{
    copyFloatToDoubleKernel<<<(count + BLOCK - 1) / BLOCK, BLOCK>>>(count, src, dst);
    return cudaGetLastError();
}

__global__ static void sensitivityKernel(int nd,
    const int* __restrict__ edof,
    const double* __restrict__ hist,
    int nrow,
    int col_n,
    int col_np1,
    const double* __restrict__ lambda_np1,
    const double* __restrict__ dk,
    const double* __restrict__ dcv,
    double thk,
    double dt,
    const double* __restrict__ Ke,
    const double* __restrict__ Ce,
    double* __restrict__ out)
{
    int kd = blockIdx.x * blockDim.x + threadIdx.x;
    if (kd >= nd) return;

    double Te[4], dTe[4], le[4], Kle[4] = { 0.0,0.0,0.0,0.0 }, Cle[4] = { 0.0,0.0,0.0,0.0 };
    const int* an = edof + 4 * kd;
    const double* Tn_col = hist + static_cast<size_t>(col_n) * static_cast<size_t>(nrow);
    const double* Tnp1_col = hist + static_cast<size_t>(col_np1) * static_cast<size_t>(nrow);

#pragma unroll
    for (int a = 0; a < 4; ++a) {
        int ia = an[a];
        double tnp1 = Tnp1_col[ia];
        double tn = Tn_col[ia];
        Te[a] = tnp1;
        dTe[a] = (tnp1 - tn) / dt;
        le[a] = lambda_np1[ia];
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            Kle[i] += Ke[4 * i + j] * le[j];
            Cle[i] += Ce[4 * i + j] * le[j];
        }
    }

    double sK = 0.0, sC = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sK += Te[i] * Kle[i];
        sC += dTe[i] * Cle[i];
    }

    out[kd] += dt * ((dk[kd] * thk) * sK + dcv[kd] * sC);
}

cudaError_t gpuSensitivityAccumulate(int nd,
    const int* d_edof,
    const double* d_hist,
    int nrow,
    int col_n,
    int col_np1,
    const double* d_lambda_np1,
    const double* d_dk,
    const double* d_dcv,
    double thk,
    double dt,
    const double* d_Ke,
    const double* d_Ce,
    double* d_out)
{
    sensitivityKernel << <(nd + BLOCK - 1) / BLOCK, BLOCK >> > (nd, d_edof, d_hist, nrow, col_n, col_np1,
        d_lambda_np1, d_dk, d_dcv, thk, dt, d_Ke, d_Ce, d_out);
    return cudaGetLastError();
}

__global__ static void sensitivityKernelHistoryFloat(int nd,
    const int* __restrict__ edof,
    const float* __restrict__ hist,
    int nrow,
    int col_n,
    int col_np1,
    const double* __restrict__ lambda_np1,
    const double* __restrict__ dk,
    const double* __restrict__ dcv,
    double thk,
    double dt,
    const double* __restrict__ Ke,
    const double* __restrict__ Ce,
    double* __restrict__ out)
{
    int kd = blockIdx.x * blockDim.x + threadIdx.x;
    if (kd >= nd) return;

    double Te[4], dTe[4], le[4], Kle[4] = { 0.0,0.0,0.0,0.0 }, Cle[4] = { 0.0,0.0,0.0,0.0 };
    const int* an = edof + 4 * kd;
    const float* Tn_col = hist + static_cast<size_t>(col_n) * static_cast<size_t>(nrow);
    const float* Tnp1_col = hist + static_cast<size_t>(col_np1) * static_cast<size_t>(nrow);

#pragma unroll
    for (int a = 0; a < 4; ++a) {
        int ia = an[a];
        double tnp1 = static_cast<double>(Tnp1_col[ia]);
        double tn = static_cast<double>(Tn_col[ia]);
        Te[a] = tnp1;
        dTe[a] = (tnp1 - tn) / dt;
        le[a] = lambda_np1[ia];
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            Kle[i] += Ke[4 * i + j] * le[j];
            Cle[i] += Ce[4 * i + j] * le[j];
        }
    }

    double sK = 0.0, sC = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sK += Te[i] * Kle[i];
        sC += dTe[i] * Cle[i];
    }

    out[kd] += dt * ((dk[kd] * thk) * sK + dcv[kd] * sC);
}

cudaError_t gpuSensitivityAccumulateHistoryFloat(int nd,
    const int* d_edof,
    const float* d_hist,
    int nrow,
    int col_n,
    int col_np1,
    const double* d_lambda_np1,
    const double* d_dk,
    const double* d_dcv,
    double thk,
    double dt,
    const double* d_Ke,
    const double* d_Ce,
    double* d_out)
{
    sensitivityKernelHistoryFloat<<<(nd + BLOCK - 1) / BLOCK, BLOCK>>>(nd, d_edof, d_hist, nrow, col_n, col_np1,
        d_lambda_np1, d_dk, d_dcv, thk, dt, d_Ke, d_Ce, d_out);
    return cudaGetLastError();
}


__global__ static void sensitivityKernelHistoryFloatHex8(int nd,
    const int* __restrict__ edof,
    const float* __restrict__ hist,
    int nrow,
    int col_n,
    int col_np1,
    const double* __restrict__ lambda_np1,
    const double* __restrict__ dk,
    const double* __restrict__ dcv,
    double dt,
    const double* __restrict__ Ke,
    const double* __restrict__ Ce,
    double* __restrict__ out)
{
    int kd = blockIdx.x * blockDim.x + threadIdx.x;
    if (kd >= nd) return;

    double Te[8], dTe[8], le[8], Kle[8] = {0.0}, Cle[8] = {0.0};
    const int* an = edof + 8 * kd;
    const float* Tn_col = hist + static_cast<size_t>(col_n) * static_cast<size_t>(nrow);
    const float* Tnp1_col = hist + static_cast<size_t>(col_np1) * static_cast<size_t>(nrow);

#pragma unroll
    for (int a = 0; a < 8; ++a) {
        int ia = an[a];
        double tnp1 = static_cast<double>(Tnp1_col[ia]);
        double tn = static_cast<double>(Tn_col[ia]);
        Te[a] = tnp1;
        dTe[a] = (tnp1 - tn) / dt;
        le[a] = lambda_np1[ia];
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            Kle[i] += Ke[8 * i + j] * le[j];
            Cle[i] += Ce[8 * i + j] * le[j];
        }
    }

    double sK = 0.0, sC = 0.0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sK += Te[i] * Kle[i];
        sC += dTe[i] * Cle[i];
    }

    out[kd] += dt * (dk[kd] * sK + dcv[kd] * sC);
}

cudaError_t gpuSensitivityAccumulateHistoryFloatHex8(int nd,
    const int* d_edof,
    const float* d_hist,
    int nrow,
    int col_n,
    int col_np1,
    const double* d_lambda_np1,
    const double* d_dk,
    const double* d_dcv,
    double dt,
    const double* d_Ke,
    const double* d_Ce,
    double* d_out)
{
    sensitivityKernelHistoryFloatHex8<<<(nd + BLOCK - 1) / BLOCK, BLOCK>>>(nd, d_edof, d_hist, nrow, col_n, col_np1,
        d_lambda_np1, d_dk, d_dcv, dt, d_Ke, d_Ce, d_out);
    return cudaGetLastError();
}
