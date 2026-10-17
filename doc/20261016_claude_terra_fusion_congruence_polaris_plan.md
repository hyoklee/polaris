# Terra Fusion congruence test on Polaris: plan (Standard)

Plan written 2026-10-16, before any build or job. Results and issues go in a
separate report under `doc/`.

## Task

Rerun the Aurora congruence pipeline
([~/aurora/doc/20261009_claude_terra_fusion_congruence_aurora.md](../../aurora/doc/20261009_claude_terra_fusion_congruence_aurora.md))
on the two P108 granules with the most and least congruent blocks, on one
Polaris GPU node, and measure the NVIDIA GPU and GPU HBM when CLIO uses GPU
memory tiering during data transfer and decompression.

| Granule | Size | Aurora P108 result (job 8934328) |
|---|---|---|
| O33737 (most congruent, block 13) | 29,847,863,689 B (29.8 GB) | C5 = 0.966 |
| O18825 (least congruent, block 0) | 53,643,715,661 B (53.6 GB) | C5 = 0.238 |
| **Total** | **83.5 GB** | |

Source: `s3://terrafusiondatasampler/P108/TERRA_BF_L1B_O33737_20060422005935_F000_V001.h5`
and `.../TERRA_BF_L1B_O18825_20030703010024_F000_V001.h5`.

Node: Polaris compute, AMD EPYC 7543P (32 cores), 512 GB DDR4,
4× NVIDIA A100 40 GB.

## What already exists

* clio-core `dev` (`~/clio-core.dev`, 9998073a) already has the Aurora work:
  striped S3 import, `cfs::` destinations and the clio-fs pool wait
  (PRs #1258, #1262).
* The HBM bdev is CUDA-native (`cudaMalloc`), so the GPU tier needs no port.
* `tf_congruence.py` selects its device from `TF_TORCH_DEVICE`.

## Setup

| Step | Why | Estimate |
|---|---|---|
| AWS C++ SDK, S3 only | CAE's S3 import needs it; not on Polaris | ~45 min, 1 node (debug) |
| libfuse3 headers | `libfuse3.so.3.10.5` is installed, the headers are not | ~10 min, login node |
| Apply PR #1296 (S3 resume/backoff) | still open; on Aurora 156 of 199 imports failed without it | patch onto `dev` |
| clio-core build: Release, CUDA + S3 + FUSE | existing `build_gpu_c` is Debug with FUSE off | ~30 min, 1 node (debug) |
| Python env + pytaf | `conda` module (PyTorch CUDA) + venv with netCDF4; pytaf is imported by every stage | ~20 min |
| Port scripts from `~/aurora` | XPU/`ZE_*` → CUDA (`TF_TORCH_DEVICE=cuda`, `CUDA_VISIBLE_DEVICES`), Eagle paths, `radix-io`, `filesystems=home:eagle` | small |
| Smoke test, O33737 only | first CUDA + S3 + FUSE run on Polaris; FUSE on compute nodes unverified | ~30 min, debug |

Setup total: ~2–3 node-hours, ~half a day of wall time.

## Runs (1 node each, `capacity` queue, 2 h walltime)

Each job ingests both granules once through the ALCF proxy (expected
70–115 MB/s striped, 13–20 min), verifies them against S3, then runs the
analysis three times with the GPU regrid and once with the CPU (pytaf) regrid.
Aurora never measured GPU vs CPU regrid.

| Run | CLIO tiers | Measures | Estimate |
|---|---|---|---|
| A. DRAM | DRAM holds everything | Polaris reference; GPU vs CPU regrid | 45–60 min |
| B. GPU HBM first | HBM ~36 GB on GPU 0 → DRAM → Lustre | ingest into HBM; reading HBM-tier data through FUSE. O33737 fits in HBM, O18825 spills | 50–70 min |
| C. Lustre first | Eagle tier file (≤100 GB) | disk baseline | 50–70 min |

Per run, record: ingest MB/s per granule, per-stage times (`blocks`,
`sources`, `congruence`, `contrast`), regrid time on GPU and CPU, tier
placement (`cte_tiers`), GPU memory and utilisation (`nvidia-smi` sampling),
host memory, and C5 for O33737 block 13 and O18825 block 0 against Aurora.

Analysis GPUs: with an HBM tier on GPU 0, analyses use GPUs 1–3.

## Totals

| | Node-hours | Wall time |
|---|---|---|
| Setup | 2–3 | ~half day |
| Runs A, B, C | ~3 | ~1 day (serial: `capacity` runs 1 job per project) |
| **Standard total** | **~5–6** | **~1–1.5 days**, plus queue waits |

Storage: DRAM tier ≤100 GB of 512 GB; Lustre tier ≤100 GB on
`/eagle/radix-io` (2.2 TB free on 2026-10-16), deleted after each job.

## Not in this plan

**GPU decompression (option D).** clio-core's optional nvCOMP codecs apply to
data CLIO compresses itself. The granules are HDF5, decompressed on the CPU by
the HDF5 library inside netCDF4, and nothing routes HDF5 chunks to nvCOMP.
Doing that (direct chunk reads into HBM, nvCOMP deflate) is a prototype of
days, not a configuration. Decide after run B.

## Risks and open questions

* FUSE mounts on Polaris compute nodes are unverified (the smoke test checks).
* Proxy throughput from Polaris compute nodes is unknown.
* One HBM bdev is one `cudaMalloc` on one GPU (~38 GB usable on an A100 40 GB).
  Whether several HBM bdevs can sit on different GPUs is to be checked.
* `capacity` allows 1 running and 2 queued jobs per project. The 3-hourly
  clio-core test (`~/bin/clio_core_gpu_dev_c_3h.sh`) submits to the same
  queue, so it competes for that slot.
* The ares reference data is not on Polaris; the comparison is with Aurora's
  P108 values.
