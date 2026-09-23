# TMGTop: GPU-Accelerated 3D Transient Thermal Topology Optimization

This repository contains the complete Case-2 implementation used to optimize
three-dimensional transient heat-conduction structures at multiple mesh
resolutions. The solver combines MPI domain decomposition, PETSc linear
solvers, CUDA sensitivity kernels and the Method of Moving Asymptotes (MMA).

## Author

Yibo Yang  
School of Chemical Engineering, Dalian University of Technology  
Dalian 116024

## Supported platform

The reference implementation is intended for Linux or WSL2 with NVIDIA GPUs.
The source may be portable to other systems, but the supplied launch scripts
use Bash, MPI and Linux utilities.

Required software:

- CMake 3.20 or newer;
- a C++17/CUDA 17 compatible compiler and the CUDA Toolkit;
- MPI;
- PETSc built with MPI, CUDA, real double-precision scalars and 64-bit indices;
- Eigen3;
- OpenCV;
- pkg-config and OpenMP.

The reference runs used four NVIDIA RTX A4000 GPUs and CUDA architecture 86.
Other GPUs can be selected at CMake configuration time.

## Configure the local environment

Machine paths are deliberately not stored in the repository. Copy the example
and edit the PETSc paths for the local installation:

```bash
cp int64_env_example.sh int64_env.sh
nano int64_env.sh
source int64_env.sh
```

Users who already export `PETSC_DIR`, `PETSC_ARCH`, `PKG_CONFIG_PATH`,
`LD_LIBRARY_PATH`, `MPIEXEC` and `EXE` do not need `int64_env.sh`.

## Build

For the RTX A4000 reference platform:

```bash
source int64_env.sh
cmake -S . -B build-int64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-int64 -j
```

Change `CMAKE_CUDA_ARCHITECTURES` for another GPU architecture. The executable
is written to `build-int64/TopOpt3DGPU4Rank`.

## Run Case 2

The default command runs the five resolutions compared in the paper
(S1, S2, S3, S4 and S7):

```bash
bash run_case2_all_scales_220.sh
```

Run every configured resolution from S1 through S7:

```bash
CASE_SET=all bash run_case2_all_scales_220.sh
```

Run selected cases:

```bash
CASE_SET=S3,S7 bash run_case2_all_scales_220.sh
```

Select GPUs by their visible device IDs:

```bash
GPU_IDS=0,1,2,3 CASE_SET=S7 bash run_case2_all_scales_220.sh
```

Write results to a chosen directory:

```bash
bash run_case2_all_scales_220.sh /path/to/case2_results
```

Resume after an unavoidable interruption:

```bash
AUTO_RESTART=1 CASE_SET=S7 bash run_case2_all_scales_220.sh
```

The VTK restart restores density, iteration and projection continuation state,
but not the full MMA asymptote history. A fresh run is preferred for final
reported results.

## Resolution definitions

| Scale | Elements | Filter radius (cells) | Included in paper comparison |
|---|---:|---:|:---:|
| S1 | 70 x 42 x 35 | 1.5 | Yes |
| S2 | 150 x 90 x 75 | 1.7578125 | Yes |
| S3 | 256 x 154 x 128 | 3.0 | Yes |
| S4 | 320 x 192 x 160 | 3.75 | Yes |
| S5 | 360 x 216 x 180 | 4.21875 | No |
| S6 | 400 x 240 x 200 | 4.6875 | No |
| S7 | 512 x 307 x 256 | 6.0 | Yes |

S2-S7 use the same physical filter radius referenced to S7. The exact scaled
S1 radius would be below one cell, so the reported practical lower bound of
1.5 cells is used.

## Reference Case-2 settings

- domain: 100 x 60 x 50 mm;
- volume fraction: 0.15;
- final time: 1000 s with 1000 time steps;
- one 15 x 15 mm heat-source patch on the top surface;
- three 90 x 5 mm convection strips on the bottom surface;
- ambient temperature: 25 degC;
- convection coefficient: 60 W/(m2 K);
- maximum optimization iterations: 220;
- density projection: beta 1 to 16, updated every 40 iterations;
- VTK output every 10 iterations.

The complete command used for each run is saved as `last_command.txt`. Logs,
intermediate VTK files and the final density CSV are stored under the selected
results directory.

## Main files

- `main.cpp`: transient solver, adjoint sensitivities and optimization loop;
- `gpu_kernels.cu/.h`: CUDA history and sensitivity kernels;
- `mma.cpp/.h`: MMA implementation;
- `zslab_partition_3d_v2.hpp`: MPI z-slab partition and scatter utilities;
- `run_case2_all_scales_220.sh`: reproducible Case-2 launcher.

## Citation

If you use TMGTop in published work, please cite the accompanying TMGTop
publication and the original MMA paper:

K. Svanberg, "The Method of Moving Asymptotes--A New Method for Structural
Optimization," *International Journal for Numerical Methods in Engineering*,
vol. 24, no. 2, pp. 359--373, 1987.
https://doi.org/10.1002/nme.1620240207

## License

TMGTop is free software distributed under the GNU General Public License,
version 3 or (at your option) any later version (`GPL-3.0-or-later`). See
[`LICENSE`](LICENSE) for the complete license text.

The implementation in `mma.cpp` and `mma.h` is a C++ adaptation of the MATLAB
GCMMA-MMA-code developed by Krister Svanberg. The original work and this
derived implementation are distributed under the GNU GPL. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance and
attribution details.
