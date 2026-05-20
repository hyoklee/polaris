# HDF5 VFD-GDS Build and Test on Polaris

**Date:** 2026-05-20  
**System:** Polaris (ALCF) — NVIDIA A100-SXM4-40GB × 4, CUDA 12.9, Driver 570.124.06  
**Branch:** master (`c613c45`)

---

## Summary

| Step | Result |
|------|--------|
| CMake configure | Fixed (3 issues resolved — see below) |
| Build (`libhdf5_vfd_gds.so`, `gds_test`, `h5gds_simple_dset_write`) | **Pass** |
| Single-rank test (`mpiexec -n 1`) | **Pass** — all 6 test cases |
| CTest | **Pass** — 1/1 (1.71 s) |
| Multi-rank test (`mpiexec -n 2`) | **Fail** — file-locking race (see known issue) |

---

## Build Environment

| Component | Path / Version |
|-----------|---------------|
| C/C++ compiler | `gcc-14` / `g++-14` (`/usr/bin`) |
| CUDA compiler | `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/`) |
| NVHPC localrc | `/home/hyoklee/nvhpc/localrc` |
| MPI | Cray MPICH 9.0.1 ofi/nvidia/23.3 |
| HDF5 (parallel) | Built from `~/hdf5/` → installed to `~/hdf5-install/` |
| cuFile header | `/soft/compilers/cudatoolkit/cuda-11.8.0/include/cufile.h` |
| cuFile runtime | RAPIDS 24.10 — `/soft/applications/dask_rapids/rapids-24.10_polaris/lib/libcufile.so` (v1.10.1) |
| cufile.json | `~/cufile.json` (copied from `/soft/compilers/cudatoolkit/cuda-12.8.0/gds/cufile.json`) |

---

## CMake Fixes

### 1. `CUDA::cuFile` target missing (`src/CMakeLists.txt`)

`find_package(CUDAToolkit)` found the toolkit but created no `CUDA::cuFile` target because
libcufile is not installed in the HPC SDK CUDA 12.9 tree.

**Old:**
```cmake
"$<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cuFile>"
```
**Fix:** use `TARGET_EXISTS` so the generator expression is a no-op when the target is absent:
```cmake
"$<$<TARGET_EXISTS:CUDA::cuFile>:CUDA::cuFile>"
```

### 2. cuFile library and header discovery (`CMakeLists.txt`)

Added `find_path(CUFILE_INCLUDE_DIR cufile.h ...)` and `find_library(CUFILE_LIBRARY cufile ...)`
with fallback paths for Polaris (`/soft/compilers/cudatoolkit/cuda-11.8.0/`,
`/soft/applications/dask_rapids/rapids-24.10_polaris/`).

When `CUDA::cuFile` is still absent after `find_package(CUDAToolkit)`, the imported target is
created manually:
```cmake
if(CUFILE_LIBRARY AND CUFILE_INCLUDE_DIR AND NOT TARGET CUDA::cuFile)
  add_library(CUDA::cuFile SHARED IMPORTED)
  set_target_properties(CUDA::cuFile PROPERTIES
    IMPORTED_LOCATION "${CUFILE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUFILE_INCLUDE_DIR}")
endif()
```

### 3. Parallel HDF5 not found

The miniconda HDF5 has `Parallel HDF5: OFF`. Rebuilt HDF5 from `~/hdf5/` with:

```bash
cmake -S ~/hdf5 -B ~/hdf5/build \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DHDF5_ENABLE_PARALLEL=ON \
  -DHDF5_ENABLE_THREADSAFE=ON \
  -DHDF5_ALLOW_UNSUPPORTED=ON \
  -DBUILD_TESTING=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF \
  -DHDF5_BUILD_HL_LIB=OFF \
  -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_CPP_LIB=OFF \
  -DMPI_C_INCLUDE_DIRS=/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include \
  -DMPI_C_LIBRARIES=/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib/libmpi.so \
  -DCMAKE_C_FLAGS="-I/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include" \
  -DCMAKE_INSTALL_PREFIX=~/hdf5-install
make -j8 install
```

---

## vfd-gds CMake Configure Command

```bash
cmake -S /lus/grand/projects/gpu_hack/hyoklee/vfd-gds \
  -B /lus/grand/projects/gpu_hack/hyoklee/vfd-gds/build \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DHDF5_DIR=/home/hyoklee/hdf5-install/cmake \
  -DMPI_C_INCLUDE_DIRS="/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include" \
  -DMPI_C_LIBRARIES="/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib/libmpi.so" \
  -DCMAKE_C_FLAGS="-I/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include"
```

---

## Runtime Requirements on Compute Nodes

- **`CUDA_VISIBLE_DEVICES`** — must be set per rank before cuFile initializes; without it
  `cuFileDriverOpen()` fails on ranks ≥ 1.
- **`MPICH_GPU_SUPPORT_ENABLED=1`** — do **not** set; requires GTL library linkage not present
  in this build.
- **`CUFILE_ENV_PATH_JSON=~/cufile.json`** — points to a config with `"allow_compat_mode": true`
  so cuFile works on Lustre (`/lus/grand`) without native GDS filesystem support.
- **`mpiexec`** — required even for single-rank runs on Polaris compute nodes.

---

## Test Results (Job 7161907, node x3107c0s25b1n0)

### Single-rank (`mpiexec -n 1 gds_test`)

```
==========================
GDS VFD tests
==========================

Extend independent write test on file gds_vfd_test.h5
Collective write test on file gds_vfd_test.h5
writeAll by some with zero row
writeAll by some with zero col
Collective read test on file gds_vfd_test.h5
readAll by some with zero col
readAll by some with zero row
Independent write test on file gds_vfd_test.h5
writeInd by some with zero row
Independent read test on file gds_vfd_test.h5
Extend independent write test #2 on file gds_vfd_test.h5
All GDS VFD tests passed
```

**Exit: 0**

### CTest (`ctest --output-on-failure`)

```
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 1.71 sec
```

**Exit: 0**

### Multi-rank (`mpiexec -n 2 gds_test`)

**Exit: 255 (Failed)**

Rank 0: `H5FD__gds_open(): unable to open file, errno = 2 (No such file or directory)`  
Rank 1: `H5FD__gds_lock(): unable to lock file, errno = 11 (Resource temporarily unavailable)`

---

## Known Issue: Multi-rank Test Failure

The GDS VFD opens files with independent POSIX `open()` calls per rank before registering
with cuFile. When two ranks race to create the same file with `O_TRUNC`, the second rank
can arrive before the first has finished the cuFile registration, causing the Lustre lock
to be unavailable (`EAGAIN`).

This is a VFD design issue: parallel HDF5 collective file creation requires coordination at
the VFD level before individual cuFile registration. The single-rank path (and all individual
I/O tests) works correctly.

---

## PBS Script

`~/polaris/bin/vfd_gds_test.pbs` — submit with:

```bash
qsub ~/polaris/bin/vfd_gds_test.pbs
```

Log written to `~/vfd_gds_test.log`.
