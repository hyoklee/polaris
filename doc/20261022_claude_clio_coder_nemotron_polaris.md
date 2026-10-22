# Nemotron 3 Nano for Clio Coder on Polaris

Goal: run NVIDIA Nemotron 3 Nano on a Polaris GPU node with the model stored on
/eagle, and drive it with Clio Coder.

## Setup

| What | Where |
| --- | --- |
| Ollama v0.40.8, official Linux build (CUDA 12 and 13 runners) | `/eagle/radix-io/hyoklee/ollama/v0.40.8` (`bin/ollama`, `lib/ollama`) |
| Model `nemotron-3-nano:latest` (Ollama id 7487f38ecf24): 31.6B total / ~3B active, hybrid Mamba-Transformer MoE (`nemotron_h_moe`), Q4_K_M, 1M context, tools + thinking | `/eagle/radix-io/hyoklee/models/ollama` (23 GB) |
| Clio Coder 0.6.1 target `polaris-nemotron` (runtime `ollama`, `http://127.0.0.1:11519`, context window 131072) | `~/.config/clio-coder/settings.yaml`. Chat and fleet defaults stay on `alcf-sophia`; runs select it with `--target polaris-nemotron` |
| Job script | `pbs/clio_coder_nemotron_polaris.pbs`, output under `/eagle/radix-io/hyoklee/clio_coder_test/run_<jobid>` |

The tarball's sha256 matched the release's `sha256sum.txt`. The model was
pulled on a login node through the ALCF proxy (`OLLAMA_MODELS` set to the
/eagle directory; a private server port; no GPU), then served on a compute
node with `OLLAMA_HOST=127.0.0.1:11519`, `OLLAMA_CONTEXT_LENGTH=131072`,
`OLLAMA_KEEP_ALIVE=-1` and `OLLAMA_FLASH_ATTENTION=1`.

To use it interactively on a compute node (e.g. `qsub -I`):

```bash
export OLLAMA_MODELS=/eagle/radix-io/hyoklee/models/ollama OLLAMA_HOST=127.0.0.1:11519
export OLLAMA_CONTEXT_LENGTH=131072 OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1
/eagle/radix-io/hyoklee/ollama/v0.40.8/bin/ollama serve > /tmp/ollama.log 2>&1 &
clio-coder --target polaris-nemotron        # or: clio-coder run --target polaris-nemotron "<task>"
```

## Results (job 7776849, debug queue, 5 min 25 s)

**Ollama on the GPUs.** It found all four A100s (CUDA, compute 8.0, runner
`cuda_v13`, driver 13.0). The model loaded 100% on GPU (53/53 layers) with a
131072-token context, on GPU 0 alone: peak 24,735 MiB of 40,960. GPUs 1–3
stayed idle, so the other three could each serve another copy.

| Request (`/api/generate`, thinking off) | Load | Prompt | Output |
| --- | --- | --- | --- |
| First (cold) | **200 s** (issue 6) | 26 tokens | 57 tokens at 8.7 tok/s |
| Second (warm) | 0 s | 37 tokens at 632 tok/s | 193 tokens at **202 tok/s** |

The April CPU run took 2 min 48 s for one short answer. Warm on the GPU, a
193-token answer takes 1 s.

**Clio Coder, answer mode** (`--turn-mode answer`): "explain
`src/nvcomp_probe/nvcomp_temp_probe.cu` in at most 6 bullets". It finished in
**10 s** with one `read` tool call and changed no file. The answer was
accurate on codecs, chunking, data patterns and checks, with one error (issue
7).

**Clio Coder, change mode** (`--turn-mode change --autonomy yolo`): a
throwaway project where `median` is wrong for even-length input and
`sample_variance` divides by n. Two tests fail. Clio Coder finished in **12 s**
with 2 `read`, 2 `edit` and 2 `bash` calls. It fixed both bugs in `stats.py`
correctly: it averages the two middle values, and uses n − 1. It ran the tests
itself and left the tests alone. The job's own test run afterwards: **OK**.

**Verdict:** Nemotron 3 Nano runs comfortably on one Polaris A100 and works
as a Clio Coder backend for small read-and-edit tasks. These two tasks are
small. Long multi-file changes are untested.

## Issues

1. **The previous copy of the model is unreachable.** `~/.ollama/models` is a
   symlink to `/lus/grand/projects/gpu_hack/iowarp/models`. That directory is
   mode `drwxrws---`, group `gpu_hack`, and this account is now only in
   `radix-io`, so it is "Permission denied". `pbs/ollama_nemo.pbs` also still
   uses `-A gpu_hack -q gpu_hack`. **Fix:** a fresh copy on /eagle (above); the
   symlink was left as is, and everything here sets `OLLAMA_MODELS`
   explicitly.

2. **The April 30 run used the CPU, not the GPUs.** Its log shows
   `inference compute id=cpu library=cpu` and `model weights device=CPU
   size="22.6 GiB"`; one short answer took 2 min 48 s. Cause: `ollama` on the
   PATH is conda-forge's `ollama 0.20.7 cpu_hacd46da_0`, a CPU-only build with
   no CUDA libraries (`~/miniconda3/lib/ollama` has only `libggml-cpu-*`).
   **Fix:** the official release tarball, which ships `lib/ollama/cuda_v12`
   and `cuda_v13`. The compute nodes' driver is 580.65.06, new enough for
   both.

3. **The April job overran its walltime** ("walltime 785 exceeded limit
   720"): the script ends with `wait` on `ollama serve`, which never exits.
   The new script kills the server on exit.

4. **The Clio Coder target cannot be verified from a login node.**
   `clio-coder configure` probes the server and model list; with no server
   running it warns "could not verify model" and "Connection refused", and
   saves the target anyway with `--force`. Harmless, but the target only
   works where the server runs (the compute node, on 127.0.0.1).

5. **The server takes about a minute to answer.** `ollama serve` took 61 s
   before `/api/version` answered: GPU discovery across the four A100s plus
   starting from /eagle. The job waited up to 60 s and only just made it; it now
   waits up to 180 s.

6. **The first load takes 200 s: Lustre read speed.** Loading 23 GB from
   `/eagle` took 200 s, about 115 MB/s. After that, `OLLAMA_KEEP_ALIVE=-1`
   keeps the model resident, so only the first request in a job pays it. To
   cut this, stage the model to the node's local SSD at the start of the job
   (copy `$OLLAMA_MODELS` to `/local/scratch` and point `OLLAMA_MODELS` there)
   and measure whether that is faster.

7. **One factual error in the code explanation.** Clio Coder said the
   `total|zero` argument decides whether decompression "allocate[s] the full
   temporary size" or "the minimal size of 0". In fact it decides what total
   uncompressed size is passed to nvCOMP's temp-size query: the real total,
   or 0 as clio-core does. The temp is then sized from the answer. This is a
   plausible but wrong reading of a terse comment; check the model's
   explanations of unfamiliar code.

8. **Change mode returns a JSON summary.** The `turn_end` message in change
   mode is a JSON object (`{"summary": "Fixed two bugs …"}`); answer mode
   returns Markdown. Scripts that read the final text must handle both.
