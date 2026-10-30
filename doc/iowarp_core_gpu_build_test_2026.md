# IOWarp Core GPU Build & Test on Polaris — 2026-05-22

## Overview

This report documents the process of building and testing
[iowarp/core](https://github.com/iowarp/core) (fork: hyoklee/core)
on Polaris (ALCF) with CUDA support enabled, resolving all build
errors and CTest failures until **211/211 tests pass**.

**Platform:** Polaris compute node — AMD EPYC 7543P, 4× A100 GPUs (arch sm_80)  
**Compiler:** gcc-14 / g++-14, nvcc (CUDA 12.9 via NVIDIA HPC SDK 25.5)  
**CMake preset:** `cuda-debug`  
**CDash project:** HERMES (`hk/r/cnd/gpu`)  
**Final result:** 0 compiler errors, 211/211 tests passed

---

## Build Environment

```
Host compiler:  gcc-14 / g++-14
CUDA compiler:  nvcc (CUDA 12.9, NVIDIA HPC SDK 25.5)
CUDA host:      g++-14
Architecture:   native (sm_80, A100)
Build type:     Debug
```

### CMakePresets.json — `cuda-debug` preset

```json
{
  "name": "cuda-debug",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Debug",
    "CMAKE_C_COMPILER": "gcc-14",
    "CMAKE_CXX_COMPILER": "g++-14",
    "CMAKE_CUDA_HOST_COMPILER": "g++-14",
    "CLIO_CORE_ENABLE_RUNTIME": "ON",
    "CLIO_CORE_ENABLE_CUDA": "ON",
    "CMAKE_CUDA_ARCHITECTURES": "native",
    "CMAKE_CUDA_STANDARD": "17",
    "CLIO_CORE_ENABLE_IO_URING": "ON"
  }
}
```

---

## Issues Found and Fixed

### 1. nvcc / cudafe++ Internal Error (stl_construct.h)

**Error:**
```
/usr/include/c++/14/bits/stl_construct.h(88): internal error:
  assertion failed at: "lexical.c", line 22310 in find_allocated_name_reference
nvcc error: 'cudafe++' died due to signal 6
```

**Root cause:** `wrp_core_enable_cuda(20)` sets `CMAKE_CUDA_STANDARD=20`
globally. CUDA device code compiled in C++20 mode triggers a cudafe++
assertion failure in GCC 14's `<bits/stl_construct.h>` when complex
templates (e.g. `std::vector` constructors) appear in device-compiled TUs.
Additionally, `-fcoroutines` (added by the top-level CMakeLists for C++20)
was being passed through to nvcc and forwarded to the host compiler,
causing further failures.

**Affected targets:** `clio_run_cxx_gpu` (library), `test_gpu_kernel_stress`
(test executable).

**Fix 1 — `cmake/ClioCoreCommon.cmake`:**
Added `CUDA_STANDARD 17` and `-Xcompiler=-fno-coroutines` to both
`add_cuda_library()` and `add_cuda_executable()` so every CUDA target in
the project uses C++17 regardless of the global setting:

```cmake
# In add_cuda_library() and add_cuda_executable():
set_target_properties(${TARGET} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    ...
)
target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-coroutines>
)
```

**Fix 2 — `context-runtime/src/CMakeLists.txt`:**
Also set the property explicitly on `clio_run_cxx_gpu` as belt-and-suspenders.

**Fix 3 — `context-runtime/test/unit/CMakeLists.txt`:**
Same per-target override on `test_gpu_kernel_stress`.

---

### 2. `chi::` Namespace Alias in CUDA-Compiled Templates

**Error:** Same cudafe++ crash, second occurrence after upstream merge.

**Root cause:** Upstream renamed the primary namespace from `chi` to
`clio::run` and made `chi` a backward-compat alias
(`namespace chi = clio::run`). When `chi::SomeType<T>` appears as a
template argument in code compiled by nvcc's cudafe++ frontend, the
alias resolution triggers the same `find_allocated_name_reference`
assertion.

**Fix:** Comprehensive replacement of `chi::` with `clio::run::` (or
unqualified names inside `namespace clio::run {}`) across all headers
and source files processed by nvcc:

| File | Change |
|------|--------|
| `include/clio_runtime/ipc_manager.h` | `chi::priv::vector` → `priv::vector`, `chi::ipc::` → `ipc::` |
| `include/clio_runtime/types.h` | `std::hash<chi::UniqueId>` → `std::hash<clio::run::UniqueId>` |
| `include/clio_runtime/worker.h` | `CLIO_CUR_WORKER` macro: unqualified → `clio::run::Worker` |
| `include/clio_runtime/gpu/future.h` | Conversion operator return type |
| `include/clio_runtime/gpu/gpu_ipc_manager.h` | `GpuTaskQueue`, `gpu::IpcManager` references |
| All module headers (MOD_NAME, admin, bdev) | sed bulk replacement |
| `src/gpu/gpu2cpu_init_hip.cc` | Namespace, helper function signatures |
| GPU test files | `test_gpu_kernel_stress_gpu.cc`, `test_gpu_vector_gpu.cc` |

---

### 3. GPU Worker Threshold in Scheduler

**Symptom:** `cr_gpu_kernel_stress_cuda` timed out (test hung indefinitely).

**Root cause:** `default_sched.cc` assigned the GPU worker only when
`total_workers >= 5`, but the default runtime config uses 4 workers
(`[sched, gpu, net_send, net_recv]`).

**Fix — `context-runtime/src/scheduler/default_sched.cc`:**
```cpp
// Before:
if (total_workers >= 5) { gpu_worker_ = ... }
// After:
if (total_workers >= 4) { gpu_worker_ = ... }
```

---

### 4. IPC Socket Race Condition (cr_client_retry_client_death_ipc)

**Symptom:** IPC-mode client tests failed to connect: "socket not found".

**Root cause:** With 4 GPUs, `ServerInitGpuQueues` (cudaMallocHost for
4× 16 MB pinned buffers) takes several seconds. `WaitForServer()` on
the client side checked for the memfd (created early) but the IPC
Unix-domain socket was only created after GPU init completed.

**Fix — `context-runtime/src/ipc_manager.cc`:**
Moved IPC socket creation to immediately after `ServerInitShm`, before
the GPU initialization block:

```cpp
// Create IPC socket early (before slow GPU init)
{
  std::string ipc_path = ctp::SystemInfo::GetMemfdPath(
      "chimaera_" + std::to_string(port) + ".ipc");
  client_ipc_transport_ = ctp::lbm::TransportFactory::Get(
      ipc_path, kSocket, kServer, "ipc", 0);
}
// ... GPU init follows ...
```

---

### 5. Port 9413 Conflicts Between Tests

**Symptom:** 60+ test failures when tests ran concurrently, all trying
to bind the same chimaera runtime port.

**Fix:** Added `RESOURCE_LOCK "chimaera_runtime"` to all tests that start
a chimaera runtime server (in CR, CTE, CAE, CEE CMakeLists.txt files).

---

### 6. Upstream API Removal: `UnregisterEventManager()`

**Error:**
```
'class ctp::lbm::Transport' has no member named 'UnregisterEventManager'
```

**Fix:** Removed the call in `ipc_manager.cc` — the new transport
implementation handles cleanup internally.

---

### 7. Accidental Core Dump Commits

**Problem:** Core dump files (`core.46479` ~2 GB, `core.46922` ~1 GB)
from the restart integration test were accidentally committed in one
of the fix commits, blocking `git push` with GitHub's 100 MB file
size limit.

**Fix:**
1. Used `git-filter-repo` to rewrite history and remove the files.
2. Added `core` and `core.*` patterns to `.gitignore`.
3. Force-pushed the rewritten branch to `origin/main`.

---

## Final Test Results

**PBS Job:** 7164665 on Polaris debug queue  
**Build time:** ~4 minutes  
**Test time:** 411.63 seconds

```
0 Compiler errors
0 Compiler warnings
211/211 tests passed
```

### Key tests verified

| Test | Result | Notes |
|------|--------|-------|
| `cr_gpu_kernel_stress_cuda` | Passed (2.19 s) | GPU→CPU task submission stress test |
| `cte_gpu_vector_cuda` | Passed | GPU vector with CUDA |
| `cte_restart_integration` | Passed (20.38 s) | Multi-process restart workflow |
| `cr_client_retry_client_death_ipc` | Passed | IPC socket race condition fixed |
| `cr_ipc_transport_ipc` | Passed | IPC transport mode |
| All 211 tests | Passed | — |

---

## Commits

All fixes are on `hyoklee/core:main`, ahead of `iowarp/core:main` (v1.5.8):

| Commit | Description |
|--------|-------------|
| `73ab54bb` | chore: add core dump pattern to .gitignore |
| `71ab8960` | fix(build): set CUDA_STANDARD 17 for all CUDA targets |
| `a57229b9` | fix(headers): replace chi:: namespace alias comprehensively |
| `662f7496` | fix(headers): use clio::run:: prefix in CLIO_CUR_WORKER macro |
| `1ca5b947` | fix(runtime): remove UnregisterEventManager call |
| `863712dd` | fix(headers): replace chi:: in template args (nvcc crash) |
| `889c9041` | fix(build): strip -fcoroutines from CUDA compile options |
| `422f58df` | fix(build): set CUDA_STANDARD 17 on GPU library target |
| `0bb4fa4b` | merge: sync upstream/main (clio namespace rename + v1.5.8) |
| `079552fa` | fix: resolve all ctest failures after upstream merge |
| `18acdfb6` | fix(runtime): cr_ipc_transport_ipc UAF in SocketTransport |

---

*Generated 2026-05-22 — hyoklee@hdfgroup.org*
