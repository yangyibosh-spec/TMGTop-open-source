# Copyright (C) 2026 Yibo Yang
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)
project(TopOpt LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Default for the RTX A4000 used in the reference runs. Users can override it,
# for example with -DCMAKE_CUDA_ARCHITECTURES=89.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA compute architectures")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

# PETSc: 先试 PETSc，再试 petsc
pkg_check_modules(PETSC IMPORTED_TARGET PETSc)
if(NOT PETSC_FOUND)
    pkg_check_modules(PETSC IMPORTED_TARGET petsc)
endif()
if(NOT PETSC_FOUND)
    message(FATAL_ERROR "PETSc not found via pkg-config.")
endif()

# OpenCV
pkg_check_modules(OPENCV IMPORTED_TARGET opencv4)
if(NOT OPENCV_FOUND)
  pkg_check_modules(OPENCV IMPORTED_TARGET opencv)
endif()
if(NOT OPENCV_FOUND)
  message(FATAL_ERROR "OpenCV not found via pkg-config.")
endif()

add_executable(TopOpt3DGPU4Rank
  main.cpp
  mma.cpp
  mma.h
  gpu_kernels.cu
  gpu_kernels.h
  zslab_partition_3d_v2.hpp
)

target_include_directories(TopOpt3DGPU4Rank PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(TopOpt3DGPU4Rank PRIVATE
  PkgConfig::PETSC
  PkgConfig::OPENCV
  OpenMP::OpenMP_CXX
  Eigen3::Eigen
  CUDA::cudart
)

set_target_properties(TopOpt3DGPU4Rank PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
