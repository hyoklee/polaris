# Terra Fusion congruence test on Polaris: S3 → CAE → CTE tiers → A100 regrid

Run on **Polaris**, 2026-10-16, one node per run (AMD EPYC 7543P, 512 GB DDR4,
4× A100-SXM4-40GB). Repeats the Aurora congruence pipeline
([~/aurora/doc/20261009_claude_terra_fusion_congruence_aurora.md](../../aurora/doc/20261009_claude_terra_fusion_congruence_aurora.md))
on the two P108 granules holding the most and least congruent blocks of the
Aurora P108 run. Plan:
[20261016_claude_terra_fusion_congruence_polaris_plan.md](20261016_claude_terra_fusion_congruence_polaris_plan.md)
(Standard: runs A, B, C).

**Status:** complete. Smoke test and runs A (DRAM), B (GPU HBM first) and
C (Lustre first) done. Run C's first attempt hit a clio-core libaio bug
(issue 9), fixed and rerun.

## Headline

* **Both target blocks reproduce Aurora.** O33737 block 13: C5 = **0.9661**
  (Aurora 0.966). O18825 block 0: C5 = **0.2376** (Aurora 0.238). Weakest
  sensor MOPITT for both. Identical in all three GPU passes and the CPU pass.
* **Ingest: 83.5 GB at 112 MB/s** (12.4 min) through the ALCF proxy, 16 × 32 MiB
  striped ranged GETs. Same ceiling as Aurora (108 MB/s). Both files matched S3
  on size and 5 × 1 MiB samples.
* **A100 regrid vs pytaf (CPU): 15.6–21.7× faster**, 2.8–3.7× for the whole
  granule. Aurora never measured this.
* **With the regrid on the GPU, CPU-side read/decompression is the floor.** For
  O18825 the band-31 `contrast` stage (38 s, netCDF4 reads only) takes longer
  than the whole `congruence` stage (27 s).
* **GPU HBM tiering does not help this pipeline.** Run B held all of O33737 in
  HBM: ingest ~12% slower (99 vs 112 MB/s) and analysis exactly as fast as from
  DRAM (38 s), because netCDF4/HDF5 reads and decompresses on the CPU.
* **The Lustre tier is slow cold and DRAM-fast warm.** Run C: first reads
  2–6× slower in the read-heavy stages, O33737's first clean pass 78 s vs 38 s;
  repeated passes match DRAM.
* **One clio-core bug found and fixed:** the libaio backend failed writes when
  its queue filled (`-EAGAIN`), which aborted every S3 import into the Lustre
  tier (issue 9).

## Granules

| Granule | Size | Blocks | Target block | Aurora P108 C5 |
| --- | --- | --- | --- | --- |
| O33737 (most congruent) | 29,847,863,689 B | 35 | 13 | 0.966 |
| O18825 (least congruent) | 53,643,715,661 B | 78 | 0 | 0.238 |

Transfer list: [`omni/terra_fusion_p108_pair.omni.yaml`](../omni/terra_fusion_p108_pair.omni.yaml).

## Software

* clio-core: worktree `~/clio-core.dev.tf`, branch `polaris-tf-congruence` =
  `origin/dev` (03c4ba2b) + PR #1296 (S3 resume with backoff, still open),
  commit c1b7709a, used for the smoke test and runs A and B. Run C's rerun used
  34ad1fd7, which adds the libaio fix (issue 9) on top. Release, gcc-14, nvcc 12.9, sm_80, `CAE_ENABLE_S3`,
  `CLIO_CORE_ENABLE_CUDA`, `CLIO_CTP_ENABLE_CUDA`, `CLIO_CTE_ENABLE_FUSE_ADAPTER`.
  Built by [`pbs/build_tf_polaris.pbs`](../pbs/build_tf_polaris.pbs) (job
  7755498, 7 min 56 s: AWS SDK 6 min, clio-core ~2 min).
* AWS SDK for C++ 1.11.599 (s3 only) and libfuse 3.10.5 headers in
  `/eagle/radix-io/hyoklee/tf/deps`.
* Python: `conda` module (PyTorch 2.8.0, CUDA 12.9) + venv with netCDF4 1.7.5
  (`/eagle/radix-io/hyoklee/tf/venv-tf`); pytaf built in place with
  `setup.py.omp` (`/eagle/radix-io/hyoklee/tf/pytaf`).
* Pipeline: [`src/tf/`](../src/tf), copied from `~/aurora/src/tf`. Changes:
  `tf_congruence.py` selects `cuda` when available; `tf_granule.sh` logs the GPU.
* Run script: [`pbs/tf_congruence_polaris.pbs`](../pbs/tf_congruence_polaris.pbs),
  `MODE=dram|hbm|lustre`. GPU 0 belongs to the runtime in every mode (the HBM
  tier is one `cudaMalloc` on the runtime's device); analyses use GPUs 1–3.

Each run: ingest + verify each granule, pass 1 (GPU) overlapped with the next
ingest as on Aurora, then clean passes one granule at a time: GPU passes 2–3
and one pytaf (CPU) pass.

## Smoke test (job 7755731, debug, MODE=hbm, O33737 only)

| | |
| --- | --- |
| HBM tier | 36 GB `cudaMalloc` on GPU 0: `nvidia-smi` 37,323 MiB used |
| Ingest | 29.8 GB in 292 s = 102 MB/s, verified |
| Placement | all in HBM: tier free 38,655 → 8,807 MB |
| Result | O33737 blk 13 C5 = 0.9661, 35 blocks, regrid 3.2 s on GPU 1 |
| Job | 5 min 53 s, 0 runtime errors |

## Run A: DRAM (job 7755769, capacity)

Tiers: DRAM 120 GB (score 1.0), Lustre 200 GB (0.2). 24.5 min from start to
cleanup, 0 runtime errors.

### Ingest

| Granule | GB | s | MB/s |
| --- | --- | --- | --- |
| O33737 | 29.8 | 263 | 113 |
| O18825 | 53.6 | 480 | 111 |
| **total** | **83.5** | **743** | **112** |

### Analysis, clean passes (s)

| Granule | Pass | blocks | sources | congruence | contrast | regrid | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O18825 | GPU p2 | 1 | 8 | 27 | 38 | 14.06 | 76 |
| O18825 | GPU p3 | 1 | 8 | 26 | 39 | 14.06 | 75 |
| O18825 | CPU | 1 | 8 | 230 | 38 | 219.1 | 279 |
| O33737 | GPU p2 | 0 | 7 | 11 | 17 | 3.36 | 38 |
| O33737 | GPU p3 | 1 | 8 | 10 | 18 | 3.36 | 38 |
| O33737 | CPU | 1 | 8 | 78 | 18 | 72.9 | 105 |

Pass 1 (overlapped with O18825's ingest) was slower for O33737: `blocks` 13 s,
`congruence` 55 s, the same ingest contention Aurora saw.

### GPU vs CPU regrid

| | O18825 | O33737 |
| --- | --- | --- |
| Regrid: pytaf (CPU, 1 thread) | 219.1 s | 72.9 s |
| Regrid: A100 (PyTorch, float64) | 14.1 s | 3.36 s |
| **Regrid speedup** | **15.6×** | **21.7×** |
| Granule: CPU regrid | 279 s | 105 s |
| Granule: GPU regrid | 75 s | 38 s |
| **End-to-end speedup** | **3.7×** | **2.8×** |

The regrid includes CUDA start-up in each process (first GPU call). Two GPU
passes agree within 1%.

### Target blocks

| Granule, block | GPU p1 | GPU p2 | GPU p3 | CPU | Aurora |
| --- | --- | --- | --- | --- | --- |
| O33737 blk 13 | 0.9661 | 0.9661 | 0.9661 | 0.9661 | 0.966 |
| O18825 blk 0 | 0.2376 | 0.2376 | 0.2376 | 0.2376 | 0.238 |

ncell 1,165 and 7,061; weakest sensor MOPITT in every pass.

### GPU use

| GPU | Peak memory | Peak utilisation | Role |
| --- | --- | --- | --- |
| 0 | 615 MiB | 1% | runtime CUDA context (no HBM tier in this mode) |
| 1 | 12,371 MiB | 100% | regrid |
| 2 | 12,371 MiB | 100% | regrid |
| 3 | – | – | unused |

## Run B: GPU HBM first (job 7755770, capacity)

Tiers: HBM 36 GB on GPU 0 (score 1.0), DRAM 120 GB (0.6), Lustre 200 GB (0.2).
25.6 min from start to cleanup, 0 runtime errors. GPU 0: 37,323 MiB used once
the tier was created.

### Placement

| Tier | After ingest | Holds |
| --- | --- | --- |
| HBM (GPU 0) | 0.2 MB free of 38,655 MB | all of O33737 (29.8 GB) + ~8.8 GB of O18825 |
| DRAM | 84,012 MB free of 128,849 MB | ~44.8 GB, the rest of O18825 |
| Lustre | untouched | – |

So O33737 is read entirely from GPU memory, and O18825 is split ~16% HBM /
~84% DRAM.

### Ingest

| Granule | GB | s | MB/s | Run A (DRAM) |
| --- | --- | --- | --- | --- |
| O33737 | 29.8 | 302 | 98 | 113 |
| O18825 | 53.6 | 534 | 100 | 111 |
| **total** | **83.5** | **836** | **99** | **112** |

Ingest into HBM was ~12% slower than into DRAM. Aurora showed the same
direction (67–80 vs 104–111 MB/s). One run each through a shared proxy, so the
size of the gap is approximate.

### Analysis, clean passes (s), against run A

| Granule | Pass | sources | congruence | contrast | regrid | total | Run A total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O18825 | GPU p2 | 8 | 27 | 39 | 13.95 | 76 | 76 |
| O18825 | GPU p3 | 8 | 27 | 38 | 13.95 | 76 | 75 |
| O18825 | CPU | 8 | 229 | 39 | 217.6 | 278 | 279 |
| O33737 | GPU p2 | 8 | 10 | 18 | 3.25 | 38 | 38 |
| O33737 | GPU p3 | 8 | 10 | 18 | 3.24 | 38 | 38 |
| O33737 | CPU | 7 | 78 | 18 | 72.7 | 105 | 105 |

**Reading from HBM made no difference to the analysis.** O33737, held entirely
in GPU memory, ran its clean passes in 38 s in both runs, and every stage
matches run A to within 1 s. The consumer is netCDF4/HDF5 on the CPU reading
through FUSE, so every byte held in HBM is copied back to the host before it
is used, and the read path is dominated by CPU-side HDF5 decompression, not by
where the compressed bytes sit. GPU 0 reached 33% utilisation serving those
reads (1% in run A).

### Target blocks

Identical to run A in every pass: O33737 blk 13 C5 = 0.9661, O18825 blk 0
C5 = 0.2376, weakest sensor MOPITT.

### GPU use

| GPU | Peak memory | Peak utilisation | Role |
| --- | --- | --- | --- |
| 0 | 37,407 MiB | 33% | HBM tier, serving reads to the host |
| 1 | 17,889 MiB | 100% | regrid |
| 2 | 12,371 MiB | 100% | regrid |
| 3 | 5 MiB | – | unused |

## Run C: Lustre first

**First attempt (job 7755847) failed in ingest**: every import into the Lustre
tier aborted after 4–6 GB (3 tries per granule, 6 failures, 2,320
`ModifyExistingData: WRITE FAILED - wrote 0 bytes ... rc=2` errors). Cause and
fix: issue 9 below.

**Rerun (job 7755914)** with the fix (clio-core 34ad1fd7). Tiers: Lustre tier
file on `/eagle/radix-io` 200 GB (score 1.0), DRAM 8 GB (0.1). 26.7 min from
start to cleanup, 0 runtime errors.

### Placement and ingest

All 83.5 GB went to the Lustre tier (free 214,748 → 131,257 MB); the DRAM
tier was untouched. Tier file: stripe count 1, stripe size 1 MiB (the Eagle
default).

| Granule | GB | s | MB/s | Run A (DRAM) | Run B (HBM) |
| --- | --- | --- | --- | --- | --- |
| O33737 | 29.8 | 306 | 97 | 113 | 98 |
| O18825 | 53.6 | 487 | 110 | 111 | 100 |
| **total** | **83.5** | **793** | **105** | **112** | **99** |

### Analysis (s): cold reads are slow, warm reads match DRAM

| Granule | Pass | blocks | sources | congruence | contrast | ASTER read | total | Run A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O33737 | GPU p1 (overlapped) | 32 | 32 | 92 | 42 | 15.4 | – | 13 / 11 / 55 / 24 |
| O18825 | GPU p1 (overlapped) | 24 | 22 | 51 | 39 | 28.4 | – | 2 / 7 / 28 / 39 |
| O18825 | GPU p2 | 1 | 8 | 27 | 39 | 5.2 | 78 | 76 |
| O33737 | GPU p2 | 15 | 23 | 20 | 17 | 10.9 | **78** | **38** |
| O18825 | GPU p3 | 2 | 8 | 26 | 39 | 5.0 | 75 | 75 |
| O33737 | GPU p3 | 1 | 8 | 11 | 17 | 2.2 | 39 | 38 |
| O18825 | CPU | 1 | 8 | 228 | 39 | 5.1 | 277 | 279 |
| O33737 | CPU | 0 | 7 | 78 | 18 | 2.1 | 105 | 105 |

(Run A column for p1: blocks / sources / congruence / contrast.)

* **First reads from the Lustre tier are 2–6× slower** in the stages that read
  the most: ASTER reads took 15.4 s and 28.4 s in pass 1 against 2.0 s and 4.6 s
  from DRAM, and `blocks`/`sources` 3–12× longer.
* **O33737's first clean pass took 78 s, twice DRAM's 38 s**; its third pass,
  39 s, matched DRAM. O18825's clean passes matched DRAM from pass 2, as it had
  been read in pass 1 only ~2 minutes earlier.
* So repeated reads come from a cache, not from the OSTs. Which one was not
  identified: aligned file-tier I/O goes through an `O_DIRECT` fd (no client
  page cache), and `lfs ladvise -a dontneed` ran without an error but its
  effect was not measured. Node page cache stayed 28–35 GB, below the 83.5 GB
  held in the tier.
* Regrid and the CPU-bound stages are unchanged (GPU regrid 13.9 / 3.25 s,
  CPU 216.7 / 72.4 s).

### Target blocks

Identical to runs A and B in every pass: O33737 blk 13 C5 = 0.9661, O18825
blk 0 C5 = 0.2376, weakest sensor MOPITT.

### GPU use

GPU 0: 459 MiB (runtime CUDA context, no HBM tier). GPUs 1 and 2: 17,889 and
12,371 MiB peak, 100% during the regrid.

## Comparison of the three tiers

| | A: DRAM | B: GPU HBM first | C: Lustre first |
| --- | --- | --- | --- |
| Where the data sat | DRAM | HBM 38.6 GB + DRAM 44.8 GB | Lustre 83.5 GB |
| Ingest | **112 MB/s** | 99 MB/s | 105 MB/s |
| O33737 first clean pass | 38 s | 38 s (all in HBM) | 78 s (cold) |
| O33737 warm pass | 38 s | 38 s | 39 s |
| O18825 clean pass | 75–76 s | 76 s | 75–78 s |
| GPU regrid O18825 / O33737 | 14.1 / 3.36 s | 13.9 / 3.25 s | 13.9 / 3.25 s |
| C5 (both targets) | match | match | match |
| Job wall time | 24.5 min | 25.6 min | 26.7 min |

**Answer to the plan's question (GPU HBM tiering).** For this pipeline the HBM
tier costs ~12% ingest speed and gives nothing back on reads: the consumer is
netCDF4/HDF5 on the CPU, so data held in HBM is copied to the host and then
decompressed there, the same work as from DRAM. DRAM is the right primary tier,
Lustre is a usable overflow for data read more than once, and HBM only pays
when the consumer runs on the GPU. That is the case option D (GPU
decompression) would create; this measurement is the argument for or against
building it.

Option D was measured on 2026-10-17: `20261017_claude_terra_fusion_gpu_inflate_polaris.md`.

## Issues found on Polaris

1. **AWS SDK for C++ not installed.** `CAE_ENABLE_S3` needs it; built 1.11.599
   (s3 only) in the build job, 6 min.
2. **No libfuse3 headers.** Polaris has `libfuse3.so.3.10.5` but no
   `fuse3/fuse.h` or `fuse3.pc`. Installed the 3.10.5 headers from the release
   tarball with a `fuse3.pc` pointing at the system library.
3. **PR #1296 (S3 resume/backoff) is still open.** Without it, 156 of 199
   imports failed on Aurora. Cherry-picked onto `dev` (applied cleanly).
4. **The existing Polaris build could not be reused.** `build_gpu_c` is Debug
   with FUSE off and S3 off; a separate Release build was needed.
5. **`clio_run status` reports ready before the HBM tier exists.** Creating the
   HBM bdev (CUDA init + one 36 GB `cudaMalloc`) takes seconds after `status`
   succeeds. The first smoke test (job 7755652) checked too early and stopped.
   The script now waits until `cte_tiers` lists every tier. The same
   ready-before-compose behaviour was seen on Aurora for the clio-fs pool.
6. **The HBM confirmation log line is hidden at `CTP_LOG_LEVEL=warning`.**
   `HBM bdev: N bytes of DEVICE memory allocated` is info level. The script now
   checks GPU 0's memory with `nvidia-smi` instead, which is direct proof.
7. **`-o max_threads=64` breaks the FUSE mount.** libfuse 3.10.5 rejects it
   (`fuse: unknown option(s)`, added in 3.12); job 7755697 failed on it. Dropped.
8. **`capacity` counts running jobs against the 2-job queue limit.** With one
   job running and one queued, a third `qsub` is refused (`would exceed queue
   capacity's per-project limit`), so runs A, B, C go in one or two at a time.
9. **clio-core bug: the libaio backend fails writes when its queue is full.**
    With the Lustre tier first, 16 concurrent S3 streams put 1 MiB pages into
    the file bdev faster than the libaio context (`io_depth`) drains them.
    `io_submit` then returns `-EAGAIN`, `LinuxAioAsyncIO::SubmitIO` returned
    `kInvalidIoToken`, and the file bdev reported a write of 0 bytes (`rc=2`,
    `fs_bdev_transport.cc`), which aborts the import. The POSIX AIO backend
    already handles this errno by doing the I/O synchronously (#1156); the
    libaio one did not. Fixed the same way in
    `context-transport-primitives/io/include/clio_ctp/io/libaio_io.h`,
    commit 34ad1fd7 on `polaris-tf-congruence` (local, not pushed). The build
    picks libaio because `libaio.so` is in `~/miniconda3`, and io_uring is off.
    Runs A and B never wrote to the file tier, so the fix does not touch them.
    The run script also passed the wrong name to `lfs ladvise`: the bdev
    appends `_node<N>` to the configured path.
10. **Login-node thread limit.** `git grep` failed with `failed to create
   thread: Resource temporarily unavailable` on `polaris-login-01`;
   `-c grep.threads=1` works.

## Caveats

* Each pass is one run; the two clean GPU passes agree within 1%.
* The ingest rate is set by the shared ALCF proxy.
* pytaf ran with `OMP_NUM_THREADS=1`, as on Aurora and ares (ares part 3: its
  OpenMP only hurts), so the CPU regrid baseline is single-threaded.
* The ares reference data is not on Polaris; comparisons are with the Aurora
  P108 values for the two target blocks only.
