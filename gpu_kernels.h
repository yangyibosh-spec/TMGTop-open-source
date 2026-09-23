/*
 * Copyright (C) 2026 Yibo Yang
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once
#include <cstddef>

// When compiled by nvcc, use the real cudaError_t.
// When compiled by the host C++ compiler (g++), use int (layout-compatible).
#ifdef __CUDACC__
#include <cuda_runtime.h>
using GpuError = cudaError_t;
#else
using GpuError = int;
#endif

GpuError gpuAllocDoubles(double** ptr, size_t n);
GpuError gpuAllocFloats(float** ptr, size_t n);
GpuError gpuZeroFloats(float* ptr, size_t n);
GpuError gpuInitHistory0Float(int n, double value, float* d_hist);
GpuError gpuStoreHistoryColumnFloat(int n, int col, const double* d_vec, float* d_hist);
GpuError gpuCopyColumnFloat(int n, const float* d_hist, int srcCol, float* d_hist2, int dstCol);
GpuError gpuCopyFloatToDouble(double* dst, const float* src, size_t count);
GpuError gpuSensitivityAccumulateHistoryFloat(int nd,
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
    double* d_out);
GpuError gpuAllocInts(int** ptr, size_t n);
GpuError gpuFreePtr(void* ptr);
GpuError gpuMemcpyHtoD(void* dst, const void* src, size_t bytes);
GpuError gpuMemcpyDtoH(void* dst, const void* src, size_t bytes);
GpuError gpuMemcpyDtoD(void* dst, const void* src, size_t bytes);
GpuError gpuZeroDoubles(double* ptr, size_t n);
GpuError gpuInitHistory0(int n, double value, double* d_hist);
GpuError gpuStoreHistoryColumn(int n, int col, const double* d_vec, double* d_hist);
GpuError gpuCopyColumn(int n, const double* d_hist, int srcCol, double* d_hist2, int dstCol);
GpuError gpuSensitivityAccumulate(int nd,
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
    double* d_out);
GpuError gpuSensitivityAccumulateHistoryFloatHex8(int nd,
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
    double* d_out);
