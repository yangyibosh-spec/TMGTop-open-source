#!/usr/bin/env bash
# Copyright (C) 2026 Yibo Yang
# SPDX-License-Identifier: GPL-3.0-or-later

# Copy this file to int64_env.sh and adapt PETSC_DIR/PETSC_ARCH to the local
# PETSc installation. The real int64_env.sh is machine-specific and should not
# be committed to the repository.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PETSC_DIR="${PETSC_DIR:-/opt/petsc}"
export PETSC_ARCH="${PETSC_ARCH:-arch-linux-cuda-int64}"

PETSC_PREFIX="$PETSC_DIR/$PETSC_ARCH"
export PATH="$PETSC_PREFIX/bin:$PATH"
export PKG_CONFIG_PATH="$PETSC_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$PETSC_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export MPIEXEC="${MPIEXEC:-$PETSC_PREFIX/bin/mpiexec}"
export EXE="${EXE:-$SCRIPT_DIR/build-int64/TopOpt3DGPU4Rank}"
