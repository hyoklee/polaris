# Terra Fusion on Polaris, option D: GPU (nvCOMP) inflate of the HDF5 chunks

Status: **complete** (job 7757064, debug queue, x3203c0s1b0n0, 8 min)

Follow-up to `20261016_claude_terra_fusion_congruence_polaris.md` (runs A–C),
which found that GPU HBM tiering did not speed up the congruence pipeline.
HDF5 inside netCDF4 inflates every chunk on the CPU, so data held in HBM is
copied back to the host first. Option D of the plan
(`20261016_claude_terra_fusion_congruence_polaris_plan.md`) asks the
remaining question: if the compressed chunks are inflated on the A100, how
much is saved?

## Answer

**Inflating HDF5's own chunks on the A100 is no faster than one CPU core, and
usually much slower.** Each HDF5 chunk is a single deflate stream, and nvCOMP
decodes one stream nearly serially (8–220 MiB/s against zlib's 0.4–1.4 GiB/s
on one core). It only gains by decoding many streams at once, and these
granules have few, large chunks: 6–9 MiB ASTER, 10.5 MiB MODIS, and MISR
Red_Radiance as one 720 MiB chunk.

**If CLIO re-encoded the data at ingest into 64 KiB nvCOMP chunks, GPU decode
takes 0.1–0.16 s per granule, not 6–8 s.** nvCOMP Zstd keeps the same size as
HDF5's deflate (1.00× and 0.98×) and decodes all the pipeline's data in
0.106 s for O33737 and 0.158 s for O18825. That is ~60× faster than HDF5's path.

**At most, that would save about 6–8 s of a 38 s or 76 s pass (~16% and ~11%),
and a 16-thread CPU inflate already saves ~5–6 s of it without a GPU.** Inflate
is under a quarter of a pass. The rest is pytaf CPU regrid in `contrast`, the
CPU parts of `congruence`, and file opens.

## Ceiling, from run A (before any new job)

Per-sensor read times in `sources.log` and `aster_read_s` of run A (DRAM tier,
clean GPU passes). Each covers a netCDF4 read, HDF5 inflate, and conversion to
float64:

| Granule | MODIS | MISR | CERES+MOPITT | ASTER (blocks + congruence) | all reads | pass total |
| --- | --- | --- | --- | --- | --- | --- |
| O33737 | 3.2 s | 2.7 s | 0.02 s | ~1 + 2.0 s | **~9 s** | 38 s |
| O18825 | 3.1 s | 2.7 s | 0.01 s | ~1 + 4.6 s | **~11 s** | 76 s |

So even with free decompression, a pass can't get faster by more than ~25% for
O33737 or ~15% for O18825. The rest of each pass is CPU pytaf regrid in
`contrast` (17 / 38 s), the non-regrid part of `congruence`, netCDF4 file and
group opens, and Python start-up.

## Test

`pbs/tf_gpu_inflate_polaris.pbs` (1 node, GPU 0, no CLIO runtime; capacity for
the fetch, then `-q debug -l walltime=00:45:00` once the granules were cached):

1. Fetches both granules to `/eagle/radix-io/hyoklee/tf/granules/` with 16
   parallel 64 MiB ranged GETs through the ALCF proxy, resuming any cut
   connection. 5 × 1 MiB of each is checked against S3. The files are kept
   for later runs.
2. Lists the exact datasets and selections the pipeline reads (ASTER
   geolocation and TIR ImageData10; MODIS geolocation and EV_1KM_Emissive
   band 31; MISR geolocation and AN Red_Radiance; CERES, MOPITT) and their
   chunks via the HDF5 chunk index.
3. `tf_gpu_inflate.py` times:
   * a pread of the raw chunks;
   * the same reads through h5py (HDF5's CPU inflate);
   * zlib (+ unshuffle) on 1 and 16 threads;
   * the host→GPU copy;
   * nvCOMP 6.0 Deflate (RAW) batched inflate, with the compressed chunks
     already in GPU memory (the HBM-tier case), one call per dataset and one
     for the whole granule.
   Every GPU chunk is checked byte-for-byte against zlib, and one chunk per
   dataset against h5py. Chunks over 8 MiB stay on the CPU.
4. `tf_gpu_reencode.py` times one deflate stream on the GPU as it grows, then
   re-encodes the same data in 64 KiB chunks with each nvCOMP codec (Zstd,
   GDeflate, Deflate, LZ4, ANS, Bitcomp, Cascaded) and decodes it from GPU
   memory. Each codec runs in its own process.

Shared dataset list: `src/tf/tf_h5chunks.py`.

Estimate: fetch ~14 min at the ~100 MB/s the runs saw, benchmark a few
minutes per granule; ~0.5 node-hour. Actual: fetch 12 min, benchmark 8 min,
~1.3 node-hours in all with the failed attempts.

## Results

Logs: `/eagle/radix-io/hyoklee/tf/tf_gpu_inflate/run_7757064/` (`job.log`,
`inflate_<orbit>.json`, `reencode_<orbit>.json`, `gpu.log`). Granules fetched
at 118 and 116 MB/s by job 7756692 and kept in `/eagle/radix-io/hyoklee/tf/granules/`.

### What the pipeline inflates

The only filter in these granules is deflate (no shuffle, no fletcher32). The
ASTER geolocation that `tf_aster_blocks.py` reads is contiguous and
uncompressed, as are CERES and MOPITT.

| Granule | Stage / sensor | Chunks | Largest chunk (MiB) | Compressed GB | Uncompressed GB | h5py s | zlib ×1 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O33737 | congruence / ASTER TIR | 105 | 6.2 | 0.334 | 0.544 | 1.79 | 1.55 |
| | sources / MODIS | 60 | 10.5 | 0.469 | 0.660 | 2.70 | 2.65 |
| | sources / MISR | 3 | 720 | 0.295 | 0.849 | 1.95 | 1.86 |
| | **all** | **168** | | **1.10** | **2.05** | **6.6** | **6.1** |
| O18825 | congruence / ASTER TIR | 234 | 9.0 | 0.798 | 1.303 | 3.77 | 3.74 |
| | sources / MODIS | 57 | 10.5 | 0.444 | 0.627 | 2.57 | 2.62 |
| | sources / MISR | 3 | 720 | 0.288 | 0.849 | 1.90 | 1.89 |
| | **all** | **294** | | **1.53** | **2.78** | **8.5** | **8.3** |

h5py time ≈ zlib on one thread: almost all of HDF5's read time is inflate.
Run A's netCDF4 read times (~9 and ~11 s) also include the float64 conversion.

### HDF5 chunks as stored: CPU vs GPU

All times are for the whole granule. The compressed chunks already sit in GPU
memory (host→GPU copy 0.05 / 0.09 s, not included).

| | O33737 | O18825 |
| --- | --- | --- |
| zlib, 1 thread (HDF5's path) | 6.1 s | 8.3 s |
| zlib, 16 threads | **1.9 s** | **2.1 s** |
| nvCOMP, chunks ≤ 8 MiB, one call per dataset | 104.2 s (zlib ×1: 1.55 s) | 191.8 s (zlib ×1: 2.88 s) |
| nvCOMP, chunks ≤ 8 MiB, all in one call | 1.47 s (zlib ×1: 1.55 s) | 1.49 s (zlib ×1: 2.88 s) |
| chunks > 8 MiB, left on the CPU (MODIS, MISR, some ASTER) | 63 datasets, 1.51 GB, 4.5 s | 88 datasets, 1.73 GB, 5.4 s |

Every GPU-inflated chunk matched zlib byte for byte (105 and 206 chunks), and
one chunk per dataset matched h5py.

One call per dataset was a single 6–9 MiB stream each time, at ~1.4 s per ASTER
geolocation chunk (zlib: 0.02 s). With every chunk in one call, the batch takes
as long as its slowest stream, which is still ~1.4 s.

**One deflate stream of real data**, the first N MiB of the largest dataset
(MISR Red_Radiance), re-deflated with zlib level 4:

| Stream | O33737 GPU | O33737 zlib | O18825 GPU | O18825 zlib |
| --- | --- | --- | --- | --- |
| 64 KiB | 63 MiB/s | 832 MiB/s | 6.7 MiB/s | 406 MiB/s |
| 1 MiB | 185 MiB/s | 1,380 MiB/s | 9.1 MiB/s | 568 MiB/s |
| 16 MiB | 216 MiB/s | 1,127 MiB/s | 8.5 MiB/s | 494 MiB/s |

The GPU rate per stream depends on the data: O18825's MISR radiances decode
25× slower than O33737's, while zlib slows only 2×.

### Re-encoded at ingest: 64 KiB nvCOMP chunks

These are the same bytes, uncompressed from the HDF5 chunks, compressed on the
GPU with `uncomp_chunk_size=65536` and batched 32 datasets per call. Decode
starts from GPU memory, which is the HBM-tier case. Every codec's output
matched the original exactly.

| Codec | O33737 size vs HDF5 | O33737 decode | O18825 size vs HDF5 | O18825 decode |
| --- | --- | --- | --- | --- |
| **Zstd** | **1.00×** | **0.106 s** (19 GB/s) | **0.98×** | **0.158 s** (18 GB/s) |
| GDeflate | 1.13× | 0.072 s (28 GB/s) | 1.14× | 0.105 s (26 GB/s) |
| Deflate | 1.12× | 0.179 s (11 GB/s) | 1.14× | 0.265 s (10 GB/s) |
| LZ4 | 1.35× | 0.063 s (33 GB/s) | 1.35× | 0.092 s (30 GB/s) |
| Cascaded | 1.36× | 0.036 s (57 GB/s) | 1.41× | 0.056 s (50 GB/s) |
| ANS | 1.48× | 0.036 s (58 GB/s) | 1.46× | 0.047 s (59 GB/s) |
| Bitcomp | 1.52× | 0.034 s (61 GB/s) | 1.51× | 0.050 s (55 GB/s) |

GPU encode at ingest took 0.04–0.6 s per granule. That's negligible next to
the ~4–8 min download.

### What it would save per pass

| Inflate path | O33737 | O18825 | Saved vs HDF5 (O33737 / O18825) |
| --- | --- | --- | --- |
| HDF5, 1 thread (today) | 6.1 s | 8.3 s | — |
| zlib, 16 CPU threads | 1.9 s | 2.1 s | 4.2 / 6.2 s |
| HDF5 chunks on the GPU | ≥ 1.47 + 4.5 (CPU) s | ≥ 1.49 + 5.4 (CPU) s | ~0 / ~1.4 s |
| Zstd 64 KiB, GPU (HBM tier) | 0.11 s + copy back | 0.16 s + copy back | ~6 / ~8 s |

Against a pass of 38 s (O33737) or 76 s (O18825), the best case is ~16% or
~11%. Only ~1.5–2 s of that is beyond what multithreaded CPU inflate gets.
Getting it also needs the analysis to keep the decoded arrays on the GPU, or
to copy 2–3 GB back to the host (~0.2–0.5 s).

### What option D would take in CLIO

1. **Transcode at ingest.** CAE imports a granule, then re-encodes each HDF5
   chunk, or the slices the analysis reads, into 64 KiB nvCOMP Zstd or GDeflate
   chunks in the HBM tier. The file is no longer HDF5 there.
2. **A GPU reader.** The pipeline asks for a dataset and selection, and gets a
   CuPy or Torch array decoded in HBM. netCDF4 and h5py can't do this; it would
   be a new API, or an HDF5 VOL/filter that returns device memory.
3. **Move the remaining CPU stages to the GPU** (the `contrast` regrid, 17 / 38 s),
   or the inflate savings are lost among them. The cheapest real speed-up here
   is that regrid, not decompression: `tf_congruence.py`'s GPU regrid is
   already 15–22× faster than pytaf.

### Issues

1. **`decode([...])` raised TypeError.** nvCOMP 6.0's Python API takes a batch
   only as `nvcomp.as_arrays(list)` (a `BatchArray`).
2. **Decoded output became invalid** (`cudaErrorInvalidValue` on copy-back).
   CuPy views from `from_dlpack` don't keep the nvCOMP batch alive, so the
   outputs are now copied into CuPy-owned arrays.
3. **Re-encoding crashed every codec with illegal memory access.** That run sent
   all 294 datasets (2.8 GB) in one `encode` call, as views into one buffer.
   The run that passed sends separate allocations, batches of 32, and copies
   the outputs. A probe job (7757040) passed every codec on small and 200 MB
   inputs, so the exact trigger (batch size, total size, or views) wasn't
   isolated.
4. **Per-dataset GPU decode stalled a run for >5 min** (job 7756764) on what
   zlib does in 6 s. Chunks larger than `GPU_MAX_CHUNK_MB` (8) are now left to
   the CPU.
5. **Downloading through the ALCF proxy cut ~1 in 4 of the 256 MiB ranged GETs**
   at ~200 MB. The fetcher now uses 64 MiB parts and resumes from the last byte
   written.
6. **Compute nodes can't see the login node's `/tmp`.** The first probe job
   (7756643) ran from the session scratch directory and exited 2 with no log.
   Probe scripts now live on Eagle.

### Jobs

| Job | Queue | What | Result |
| --- | --- | --- | --- |
| 7756643 | debug | nvCOMP probe from login `/tmp` | exit 2, no log (issue 6) |
| 7756644 | capacity | fetch + benchmark | cancelled: 256 MiB GETs cut (issue 5) |
| 7756692 | capacity | fetch + benchmark | granules fetched; GPU step TypeError (issue 1) |
| 7756764 | debug | benchmark | cancelled after 5 min in per-dataset GPU decode (issue 4) |
| 7756917 | debug | benchmark, 8 MiB cap, re-encode | CPU/GPU numbers; crashes at copy-back and re-encode (issues 2, 3) |
| 7757040 | debug | nvCOMP encode/decode probe | all pass |
| 7757064 | debug | benchmark, all fixes | **complete, the numbers above** |
