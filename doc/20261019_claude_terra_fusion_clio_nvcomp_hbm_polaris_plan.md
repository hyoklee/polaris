# Terra Fusion on Polaris: CLIO's own nvCOMP GPU tiering, plan and estimate

Follow-up to `20261016_claude_terra_fusion_congruence_polaris.md` (runs A–C)
and `20261017_claude_terra_fusion_gpu_inflate_polaris.md` (option D). Those
runs measured GPU HBM tiering for **data transfer** (run B) and **nvCOMP
decompression outside CLIO** (option D, plain nvCOMP with no CLIO runtime).
The part of CLAUDE.md still open is decompression done **by CLIO itself**
with data in the GPU HBM tier.

## What clio-core can do (read from `origin/dev` 03c4ba2b, 2026-10-19)

* The `clio_cte_compressor` chimod sits in front of `clio_cte_core` and
  compresses each blob with the codec the put asks for, or with
  `default_compress` for writers that ask for none. It is built only with
  `CLIO_CTE_ENABLE_COMPRESS=ON`. The Polaris builds so far have it OFF.
* **nvCOMP codecs (wire ids 11–16) only run on device-resident data.** A put
  with a GPU codec compresses device → device, and a get decodes into a device
  page. For a host source the put fails on purpose:
  `compressor: GPU codec '…' requested but the device codec path did not run
  … Failing the put rather than silently substituting a CPU codec`
  (`compressor_runtime.cc` ~3186). A host-destined read of a GPU-compressed
  blob also fails (~3496).
* So **the congruence pipeline cannot use CLIO's GPU decompression as it
  stands.** Its data comes in from S3 through CAE and is read through FUSE by
  netCDF4/HDF5, all host memory. Setting `default_compress: nvcomp-zstd` would
  make every page put of the import fail.
* The path that does work is **gpu_vector**: a paged device vector whose pages
  CTE stores in tiers (HBM first). The compressor encodes and decodes those
  pages with nvCOMP on the GPU. `clio_gpu_vector_stream_bench` measures it:
  a write pass, then a cold read pass that faults every page back.
* gpu_vector is built only with `CLIO_GPU_CLANG=ON`. Its device API is C++20
  coroutines, which nvcc cannot compile. clio-core has not yet been built with
  clang-CUDA on Polaris. clang 22.1.4 (`/soft/compilers/llvm`) compiles CUDA
  12.9.1 for sm_80 here ("partially supported" warning only).
* Even with a host path, these granules would gain little: zstd on 1 MiB
  pages of the files gives 0.97 (O33737) and 0.975 (O18825), and only 15% and
  9% of pages pass the compressor's ≥ 1/8 saving bar (120 random pages each).
  The rest would be stored raw.

## Test

Patch: `clio_gpu_vector_stream_bench` gains `--data-file` (fill the vector
with real bytes instead of the synthetic pattern) and `--codec`
(lz4|snappy|zstd|gdeflate|deflate|ans instead of lz4 only). Commit 1a043c03 on
`polaris-tf-congruence`.

Inputs, written once on the compute node from the cached granules
(`src/tf/tf_dump_arrays.py`):

| File | What | Size |
| --- | --- | --- |
| `A_O33737.bin`, `A_O18825.bin` | the arrays the pipeline reads, decoded, i.e. what a GPU consumer holds | 2.05 / 2.78 GB |
| `H_O33737.bin`, `H_O18825.bin` | 2 GiB of each granule's own bytes as stored (deflated HDF5) | 2 × 2.15 GB |

Matrix (one bench process and runtime per cell, GPU 0, 16 blocks, 256 KiB
pages, batched reads of 8 pages):

| Tier option | CTE tiers | Question |
| --- | --- | --- |
| `fit` | HBM holds all of it | cost of nvCOMP encode/decode when nothing spills |
| `dram` | HBM = ¼ of raw size, spill to DRAM | does compression keep more in HBM and read faster? |
| `lustre` | HBM = ¼, spill to a file on `/eagle` | same, with a slow spill medium |
| `host` | pinned host memory holds all (no HBM) | the DRAM-tier baseline for `fit` |

Codecs: raw, nvCOMP zstd, lz4, gdeflate, deflate, ans. The H files run `fit`
and `dram` only. That gives 72 runs, plus two smoke runs and two per-page
fault runs.

Per run, record: stored MB and ratio, write and read GB/s of the logical
bytes, a position-weighted checksum against the source, and GPU 0 memory and
utilisation (1 s `nvidia-smi`).

## Estimates

Each run reads its 2–2.8 GB input (Eagle, then page cache), starts a runtime
(~5–10 s), writes and reads 8–11 k pages, and checksums on the host (~1–2 s).
The stream bench's own notes give ~110 µs per fault round trip, and up to
~7.6 ms per fault when an nvCOMP decode switches context on the device. That
puts a run between ~20 s (raw, batched) and ~90 s (GPU codec, per-page). The
per-run cap is 10 min.

| Option | Scope | Node-hours | Wall time |
| --- | --- | --- | --- |
| 1. Minimal | build + A files, `fit` only, raw/zstd/lz4 (6 runs) | ~1.0 | ~1 h + queue |
| **2. Standard (submitted)** | build + the matrix above (~76 runs) | **~1 build + 1–2.5 run ≈ 2–3.5** | **~2–3.5 h + queue** |
| 3. Full | Standard + page-size sweep (64 KiB / 1 MiB) + per-page faults for every cell | ~5–6 | ~1 day (capacity runs one job at a time) |
| Not possible as asked | the congruence pipeline end to end with `default_compress: nvcomp-*` | — | fails at the first page put (above) |

Build: debug queue, 1 h cap. A first clang-CUDA build of the tree is ~20–40
min on 32 cores; the stream bench is built first so a failure elsewhere does
not block the run. Run: capacity queue, 3 h walltime, held until the build
succeeds (`depend=afterok`). Runs are ordered most important first, and the
job stops starting new ones 15 min before the walltime.

Storage: inputs ~8.9 GB in `/eagle/radix-io/hyoklee/tf/tf_clio_nvcomp/data`
(kept); Lustre spill ≤ 5.6 GB per run, deleted after it.

## Risks

* First clang-CUDA build of clio-core on Polaris: gcc-14 libstdc++ with clang 22,
  CUDA 12.9 newer than clang knows, and nvCOMP 6.0 from the pip wheel (clio's
  CI uses 5.x). Any of these can stop the build.
* The stream bench has only been run upstream on other GPUs (sm_89 in CI).
* `--tier-type hbm` in the bench config is a CTE hbm bdev, as in run B, which is
  one `cudaMalloc` on GPU 0 next to the bench's own page cache.

## Scripts

* `pbs/build_tf_nvcomp_polaris.pbs` → `/eagle/radix-io/hyoklee/tf/build_tf_nvcomp`
* `pbs/tf_clio_nvcomp_polaris.pbs` → `/eagle/radix-io/hyoklee/tf/tf_clio_nvcomp/run_<job>/`
* `src/tf/tf_dump_arrays.py`

## Jobs

| Job | Queue | What | Result |
| --- | --- | --- | --- |
| 7762190 | debug | clang-CUDA build with compressor + nvCOMP | exit 2 after 2 min 20 s: issue 1 |
| 7762191 | capacity | data prep + smoke + matrix (afterok 7762190) | deleted by PBS with the failed build |
| 7762243 | debug | build, with the issue 1 fix | queued |
| 7762244 | capacity | data prep + smoke + matrix (afterok 7762243) | held on the build |

## Issues

1. **clio-core bug: the compressor chimod does not link nlohmann_json.**
   `compressor/src/models/qtable_predictor.cc` includes `nlohmann/json.hpp`,
   but `clio_cte_compressor_runtime` never links `nlohmann_json::nlohmann_json`,
   so it builds only where the header is in a system include path (upstream
   CI). On Polaris it is in `~/deps/nlohmann_json`: `fatal error:
   nlohmann/json.hpp: No such file or directory`. Fixed by linking the target
   when it exists, commit e8c5b033 on `polaris-tf-congruence` (local, not
   pushed). The clang-CUDA toolchain itself configured cleanly: clang 22.1.4,
   CUDA 12.9.1, nvCOMP 6.0.0.12 found.
