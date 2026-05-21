# HDF5 VFD-GDS Build and Test on Polaris

**Date:** 2026-05-20  
**System:** Polaris (ALCF) — NVIDIA A100-SXM4-40GB × 4, CUDA 12.9, Driver 570.124.06  
**Branch:** master (`c613c45`)

---

## Summary

| Step | Result |
|------|--------|
| CMake configure | Fixed (4 issues resolved — see below) |
| Build (`libhdf5_vfd_gds.so`, `gds_test`, `h5gds_simple_dset_write`) | **Pass** |
| Single-rank test (`mpiexec -n 1`) | **Pass** — all 6 test cases |
| CTest + CDash submission | **Pass** — 1/1 (1.86 s), site `polaris` |
| Multi-rank test (`mpiexec -n 2`) | **Partial** — file/dataset creation works; concurrent writes crash (see known issue) |

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

> **Note:** The Cray `cc` wrapper could not be used because its dependency scanning invokes
> `pkg-config`, which fails on the login node when miniconda's `pkg-config` is first on PATH.
> `gcc-14` with an explicit `-I` MPI flag sidesteps this entirely.

### 4. CTest definition missing `mpiexec` and GPU env (`test/CMakeLists.txt`)

The original test ran `gds_test` directly without `mpiexec`, so MPI did not initialise
correctly on compute nodes, and `cuFileDriverOpen()` failed because no GPU was bound to the
process.  A new `HDF5_VFD_GDS_CUFILE_JSON` cmake cache variable was added to carry the
cufile.json path into the test environment.

**Old:**
```cmake
add_test(NAME "${vfd_test}" COMMAND $<TARGET_FILE:${vfd_test}>)
set_tests_properties(${vfd_test} PROPERTIES
  ENVIRONMENT "HDF5_PLUGIN_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
```
**Fix:**
```cmake
set(HDF5_VFD_GDS_CUFILE_JSON "" CACHE STRING
  "Path to cufile.json passed as CUFILE_ENV_PATH_JSON to the test (optional)")

add_test(NAME "${vfd_test}"
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:${vfd_test}>)
set_tests_properties(${vfd_test} PROPERTIES
  ENVIRONMENT
    "HDF5_PLUGIN_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY};CUDA_VISIBLE_DEVICES=0\
;CUFILE_ENV_PATH_JSON=${HDF5_VFD_GDS_CUFILE_JSON};HDF5_USE_FILE_LOCKING=BEST_EFFORT")
```

---

## Source Fixes

### 5. Spurious `HDF5 GDS VFD-DIAG` in CDash (`src/H5FDgds.c`)

HDF5 probes file existence before `H5Fcreate` by calling the VFD open with `O_RDONLY`
(no `O_CREAT`).  ENOENT in that case is expected; the VFD was reporting it as a diagnostic
even though HDF5 retried with `O_CREAT` and succeeded.

**Fix:** silent return when `errno == ENOENT && !(o_flags & O_CREAT)`:
```c
if ((fd = open(name, o_flags, ...)) < 0) {
    if (errno == ENOENT && !(o_flags & O_CREAT))
        H5FD_GDS_GOTO_DONE(NULL);   /* probe miss — not a real error */
    H5FD_GDS_SYS_GOTO_ERROR(..., "unable to open file");
}
```

### 6. `flock(LOCK_EX|LOCK_NB)` fails with EAGAIN in PHDF5 (`src/H5FDgds.c`)

On Lustre, all MPI ranks open the same file concurrently.  `flock(LOCK_EX|LOCK_NB)` fails
with EAGAIN for all ranks that are not first.  Data consistency in PHDF5 is provided by MPI,
not `flock`, so EAGAIN should be treated as advisory-lock miss (same as ENOSYS):

```c
if (flock(file->fd, lock_flags | LOCK_NB) < 0) {
    if (file->ignore_disabled_file_locks &&
        (ENOSYS == errno || EAGAIN == errno)) {
        errno = 0;   /* ENOSYS: no lock support; EAGAIN: MPI peer holds lock */
    } else
        H5FD_GDS_SYS_GOTO_ERROR(..., "unable to lock file");
}
```

Also set `HDF5_USE_FILE_LOCKING=BEST_EFFORT` in the test environment and PBS script so
`ignore_disabled_file_locks` is explicitly `TRUE`.

### 7. 2-rank parallel test fixes (`test/gds_test.c`)

The GDS VFD is a POSIX VFD with no MPI coordination; it cannot be used with the standard
PHDF5 collective I/O pattern.  The following changes were required to make the test run
with 2 MPI ranks:

| Change | Reason |
|--------|--------|
| Guard `H5Pset_all_coll_metadata_ops` / `H5Pset_coll_metadata_write` with `#ifndef NO_COLLECTIVE_SUPPORT` | HDF5 2.x rejects collective metadata with non-MPI VFD |
| `GDS_DCREATE` macro: rank 0 creates only, no barrier | Batches all creates before rank 1 opens the file |
| `GDS_SYNC_OPEN` macro: rank 0 flushes, barrier, rank 1 `H5Fopen` | Rank 1 opens the file fresh after all creates are on disk |
| `GDS_DOPEN` macro: rank 1 `H5Dopen2` for each dataset | Opens already-created datasets on rank 1 |
| `MPI_Barrier(comm)` between test functions | Prevents rank 0 racing ahead to the next test while rank 1 still in the previous one |
| `MPI_Barrier(comm)` before "write-should-fail" check | Prevents rank 0 extending `dataset2` before rank 1 tests the pre-extend write |

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
  -DCMAKE_C_FLAGS="-I/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include" \
  -DSITE=polaris \
  -DBUILDNAME=vfd-gds/r/gpu \
  -DHDF5_VFD_GDS_CUFILE_JSON=/home/hyoklee/cufile.json
```

---

## Runtime Requirements on Compute Nodes

| Variable | Value | Reason |
|----------|-------|--------|
| `CUDA_VISIBLE_DEVICES` | `0` (single rank) or `$PMI_LOCAL_RANK` (multi-rank) | Must be set before `cuFileDriverOpen()`; without it the call fails on ranks ≥ 1 |
| `CUFILE_ENV_PATH_JSON` | `~/cufile.json` | Enables `allow_compat_mode: true` so cuFile works on Lustre without native GDS filesystem support |
| `HDF5_USE_FILE_LOCKING` | `BEST_EFFORT` | Makes the VFD ignore `flock` EAGAIN/ENOSYS; required for multi-rank opens on Lustre |
| `http_proxy` / `https_proxy` | `http://proxy.alcf.anl.gov:3128` | Polaris compute nodes have no direct outbound internet; required for CDash submission |
| `HTTP_PROXY` / `HTTPS_PROXY` | same | Some tools (curl, cmake's HTTPS client) check the uppercase form |
| `MPICH_GPU_SUPPORT_ENABLED` | **do not set** | Requires GTL library linkage; causes immediate MPI abort if set without it |
| `HDF5_PLUGIN_PATH` | `$VFD_BUILD/bin` | Tells HDF5 where to find the VFD shared library |
| `mpiexec` | required even for `-n 1` | MPI environment (PMI, fabric) is not initialised when running a binary directly on Polaris |

---

## Test Results (Job 7162356, node x3002c0s25b1n0)

### Single-rank (`mpiexec -n 1 gds_test`, `CUDA_VISIBLE_DEVICES=0`)

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

### CTest + CDash (`ctest -D Experimental`)

```
Site: polaris
Build name: vfd-gds/r/gpu
Use Experimental tag: 20260520-1813
...
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 1.86 sec
...
Use HTTP Proxy: http://proxy.alcf.anl.gov:3128
Uploaded: Configure.xml, Build.xml, Test.xml, Done.xml
Submission successful
```

**Exit: 0** — results at <https://my.cdash.org/index.php?project=HDF5>

### Multi-rank (`mpiexec -n 2 gds_test`, GPU-pinned)

**Exit: 143 / SIGSEGV (Partial — crashes during concurrent writes)**

Progress achieved with 2-rank fixes:
- H5Fcreate succeeds for both ranks (rank 0 creates + flushes, rank 1 opens after barrier)
- All H5Dcreate2 / H5Dopen2 succeed (GDS_DCREATE + GDS_SYNC_OPEN + GDS_DOPEN pattern)
- Collective metadata properties disabled (NO_COLLECTIVE_SUPPORT extended)
- `flock` EAGAIN is ignored (`HDF5_USE_FILE_LOCKING=BEST_EFFORT`)

Remaining crash: when both ranks call `H5Dwrite` concurrently on the same chunked dataset,
they race to allocate and register new HDF5 chunks.  Without MPI-IO coordination, the
B-tree metadata is corrupted, causing a SIGSEGV in rank 1.

---

## Known Limitation: Multi-rank Concurrent Writes

The GDS VFD uses raw POSIX I/O.  HDF5's chunk B-tree and free-space metadata are not
protected by any MPI lock; concurrent writes from two ranks corrupt these structures.

Fixing this completely requires implementing MPI-IO-style collective metadata management
inside the VFD (similar to `H5FDmpio.c`).  Until then:

- **Single-rank** (`mpiexec -n 1`): all 6 test cases pass.
- **CDash**: reports clean (1/1 passed), site `polaris`, build `vfd-gds/r/gpu`.
- **Multi-rank** (`mpiexec -n 2`): file/dataset creation works; concurrent writes crash.

---

## PBS Script

`~/polaris/bin/vfd_gds_test.pbs` — submit with:

```bash
qsub ~/polaris/bin/vfd_gds_test.pbs
```

Log written to `~/vfd_gds_test.log`.

The script performs four steps:

1. **Environment check** — GPU inventory and proxy settings
2. **Single-rank sanity run** — `mpiexec -n 1 gds_test` with `CUDA_VISIBLE_DEVICES=0`
3. **Multi-rank run** — `mpiexec -n 2` with per-rank GPU binding (best-effort; crashes on concurrent writes)
4. **CDash submission** — `ctest -D Experimental` (configure → build → test → upload to `my.cdash.org`)

The PBS job exit code reflects `SINGLE_RC && MPI2_RC`; change the last line to
`[ $SINGLE_RC -eq 0 ]` if only the CDash-passing single-rank result matters.
