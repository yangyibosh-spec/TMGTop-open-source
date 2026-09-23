/*
 * Copyright (C) 2026 Yibo Yang
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once
#include <petscksp.h>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>

struct ZSlabPartition3D {
    int rank = 0;
    int nranks = 1;

    // owned element slabs [ez0, ez1)
    int ez0 = 0;
    int ez1 = 0;

    // non-overlapping owned node planes [kOwned0, kOwned1] inclusive
    int kOwned0 = 0;
    int kOwned1 = -1;

    // work node planes needed by locally owned elements [kWork0, kWork1] inclusive
    int kWork0 = 0;
    int kWork1 = -1;

    PetscInt ownedNodeStart = 0;
    int nOwnedNodes = 0;
    int nWorkNodes = 0;
    int nOwnedElems = 0;

    std::vector<PetscInt> ownedNodeIds;
    std::vector<PetscInt> workNodeIds;
    std::vector<int> globalNodeToWork;
    std::vector<PetscInt> ownedElemIds;
    std::vector<int> localDesignElems;
    std::vector<std::array<int,8>> localDesignEdofWork;
};

template <class Mesh3D, class NodeIdFn, class ElemIdFn>
static ZSlabPartition3D buildUniformZSlabPartition3D(
    const Mesh3D& mesh,
    MPI_Comm comm,
    NodeIdFn nodeId,
    ElemIdFn elemId)
{
    ZSlabPartition3D p;
    MPI_Comm_rank(comm, &p.rank);
    MPI_Comm_size(comm, &p.nranks);

    const int base = mesh.nelz / p.nranks;
    const int rem  = mesh.nelz % p.nranks;
    p.ez0 = p.rank * base + std::min(p.rank, rem);
    p.ez1 = p.ez0 + base + ((p.rank < rem) ? 1 : 0);

    // Match the 2D owned-node convention: interface plane belongs to the lower-rank slab.
    p.kOwned0 = (p.ez0 == 0) ? 0 : (p.ez0 + 1);
    p.kOwned1 = (p.ez1 == mesh.nelz) ? mesh.nelz : p.ez1;
    if (p.kOwned1 < p.kOwned0) {
        throw std::runtime_error("Invalid z-slab owned node range.");
    }

    // Local elements [ez0, ez1) require node planes ez0..ez1.
    p.kWork0 = p.ez0;
    p.kWork1 = p.ez1;

    const int nxy = mesh.nnx * mesh.nny;
    p.ownedNodeStart = static_cast<PetscInt>(p.kOwned0) * static_cast<PetscInt>(nxy);
    p.nOwnedNodes = (p.kOwned1 - p.kOwned0 + 1) * nxy;

    p.globalNodeToWork.assign(static_cast<size_t>(mesh.numNodes), -1);
    p.ownedNodeIds.reserve(static_cast<size_t>(p.nOwnedNodes));
    p.workNodeIds.reserve(static_cast<size_t>((p.kWork1 - p.kWork0 + 1) * nxy));

    for (int k = p.kOwned0; k <= p.kOwned1; ++k) {
        for (int j = 0; j < mesh.nny; ++j) {
            for (int i = 0; i < mesh.nnx; ++i) {
                p.ownedNodeIds.push_back(nodeId(i, j, k, mesh.nnx, mesh.nny));
            }
        }
    }

    for (int k = p.kWork0; k <= p.kWork1; ++k) {
        for (int j = 0; j < mesh.nny; ++j) {
            for (int i = 0; i < mesh.nnx; ++i) {
                PetscInt gid = nodeId(i, j, k, mesh.nnx, mesh.nny);
                p.globalNodeToWork[static_cast<size_t>(gid)] = static_cast<int>(p.workNodeIds.size());
                p.workNodeIds.push_back(gid);
            }
        }
    }
    p.nWorkNodes = static_cast<int>(p.workNodeIds.size());

    for (int k = p.ez0; k < p.ez1; ++k) {
        for (int j = 0; j < mesh.nely; ++j) {
            for (int i = 0; i < mesh.nelx; ++i) {
                PetscInt e = elemId(i, j, k, mesh.nelx, mesh.nely);
                p.ownedElemIds.push_back(e);
                p.localDesignElems.push_back(static_cast<int>(e));
            }
        }
    }
    p.nOwnedElems = static_cast<int>(p.ownedElemIds.size());

    return p;
}

template <class Mesh3D>
static void buildLocalDesignEdofWork3D(
    const Mesh3D& mesh,
    ZSlabPartition3D& part)
{
    part.localDesignEdofWork.clear();
    part.localDesignEdofWork.reserve(part.localDesignElems.size());

    for (int e : part.localDesignElems) {
        std::array<int,8> edofW{};
        bool ok = true;
        for (int a = 0; a < 8; ++a) {
            const PetscInt g = mesh.edof[static_cast<size_t>(e)][a];
            const int lid = part.globalNodeToWork[static_cast<size_t>(g)];
            if (lid < 0) { ok = false; break; }
            edofW[a] = lid;
        }
        if (!ok) throw std::runtime_error("Failed to map global node to local work node in 3D partition.");
        part.localDesignEdofWork.push_back(edofW);
    }
}

static inline void createScatterToWork3D(Vec globalVec, const ZSlabPartition3D& part, bool useGpu, Vec* workVec, VecScatter* scatter)
{
    VecCreateSeq(PETSC_COMM_SELF, part.nWorkNodes, workVec);
    VecSetType(*workVec, useGpu ? "seqcuda" : "seq");

    IS isFrom = nullptr, isTo = nullptr;
    ISCreateGeneral(PETSC_COMM_SELF,
        static_cast<PetscInt>(part.workNodeIds.size()),
        part.workNodeIds.data(), PETSC_COPY_VALUES, &isFrom);
    ISCreateStride(PETSC_COMM_SELF,
        static_cast<PetscInt>(part.workNodeIds.size()), 0, 1, &isTo);
    VecScatterCreate(globalVec, isFrom, *workVec, isTo, scatter);
    ISDestroy(&isFrom);
    ISDestroy(&isTo);
}

static inline void createScatterToExistingWork3D(Vec globalVec, const ZSlabPartition3D& part, Vec workVec, VecScatter* scatter)
{
    IS isFrom = nullptr, isTo = nullptr;
    ISCreateGeneral(PETSC_COMM_SELF,
        static_cast<PetscInt>(part.workNodeIds.size()),
        part.workNodeIds.data(), PETSC_COPY_VALUES, &isFrom);
    ISCreateStride(PETSC_COMM_SELF,
        static_cast<PetscInt>(part.workNodeIds.size()), 0, 1, &isTo);
    VecScatterCreate(globalVec, isFrom, workVec, isTo, scatter);
    ISDestroy(&isFrom);
    ISDestroy(&isTo);
}

static inline void scatterGlobalToWork3D(VecScatter scatter, Vec globalVec, Vec workVec)
{
    VecScatterBegin(scatter, globalVec, workVec, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter, globalVec, workVec, INSERT_VALUES, SCATTER_FORWARD);
}
