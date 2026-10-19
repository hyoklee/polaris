#!/usr/bin/env python3
"""Option D: inflate the HDF5 chunks the congruence pipeline reads on the GPU.

The pipeline (tf_aster_blocks.py, tf_sources.py, tf_congruence.py) reads its
datasets through netCDF4, so HDF5 inflates every deflate chunk on the CPU, one
chunk at a time. This measures, for exactly those datasets and selections:

  io       pread of the raw (compressed) chunks named by the chunk index
  h5py     the same selections through h5py (HDF5's own CPU inflate, warm cache)
  zlib1    zlib inflate + unshuffle of the raw chunks, 1 thread (HDF5's path)
  zlibN    the same on N threads (what a parallel CPU reader could do)
  h2d      copy of the compressed chunks to the GPU
  gpu      nvCOMP Deflate (RAW) batched inflate + unshuffle on the GPU, with
           the compressed chunks already in GPU memory: what CLIO's HBM tier
           could hand an analysis that inflates on the device

and checks every GPU-inflated chunk against zlib, and one chunk per dataset
against h5py. GPU timings are reported two ways: one decode call per dataset
(a drop-in for one h5py read) and one call for the whole granule.

Usage: tf_gpu_inflate.py <granule.h5> <out.json>
Environment: MODIS_BAND_IDX (10), CERES_FIELD (WN_Radiance), MOPITT_CHAN (4),
  MOPITT_STATE (0), INFLATE_THREADS (16), GPU_BATCH_GB (8), GPU_MAX_CHUNK_MB (8).
"""
import json, os, sys, time, zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tf_h5chunks import inventory, h5sel, unshuffle_np, zlib_strip

F, OUT = sys.argv[1], sys.argv[2]
NTHR = int(os.environ.get('INFLATE_THREADS', '16'))
BATCH = float(os.environ.get('GPU_BATCH_GB', '8')) * 2**30
# A chunk is one deflate stream, which nvCOMP decodes almost serially: a 1 GB
# granule's chunks of tens to hundreds of MB kept an A100 busy for >5 min
# (job 7756764). Larger chunks than this are left to the CPU and reported.
MAX_CHUNK = float(os.environ.get('GPU_MAX_CHUNK_MB', '8')) * 2**20


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ---- 1. inventory --------------------------------------------------------
f = h5py.File(F, 'r')
t0 = time.time()
DS = inventory(f)
log(f'{len(DS)} datasets, {sum(len(d["ck"]) for d in DS)} chunks, '
    f'inventory {time.time() - t0:.1f}s; filters seen:',
    sorted({tuple(d['filters']) for d in DS}))
unsupported = [d['path'] for d in DS if not d['gpu_ok']]
if unsupported:
    log(f'{len(unsupported)} datasets not GPU-inflatable here (CPU only):', unsupported[:8])

# ---- 2. io: raw chunks ---------------------------------------------------
fd = os.open(F, os.O_RDONLY)
t = time.time()
for d in DS:
    t1 = time.time()
    d['raw'] = [os.pread(fd, n, off) for _, off, n, _ in d['ck']]
    d['io_s'] = time.time() - t1
    d['comp_bytes'] = sum(len(b) for b in d['raw'])
    d['unc_bytes'] = d['raw_chunk'] * len(d['ck'])
T = dict(io_s=time.time() - t)
log(f'io: {sum(d["comp_bytes"] for d in DS)/1e9:.2f} GB compressed in {T["io_s"]:.1f}s')

# ---- 3. h5py: the pipeline's own path (HDF5 CPU inflate) ------------------
t = time.time()
for d in DS:
    t1 = time.time()
    a = f[d['path']][h5sel(d['sel'])]
    d['h5py_s'] = time.time() - t1
    del a
T['h5py_s'] = time.time() - t
log(f'h5py: {T["h5py_s"]:.1f}s')


# ---- 4. zlib on the CPU: 1 thread, then N --------------------------------
def inflate_cpu(d, raw):
    if not d['gpu_ok'] or not d['filters']:
        return raw
    out = zlib.decompress(raw[:-4] if d['fl32'] else raw)
    return unshuffle_np(out, d['itemsize']) if d['shuffle'] else out


ZOUT = {}
t = time.time()
for i, d in enumerate(DS):
    t1 = time.time()
    ZOUT[i] = [inflate_cpu(d, r) for r in d['raw']]
    d['zlib1_s'] = time.time() - t1
T['zlib1_s'] = time.time() - t
log(f'zlib x1: {T["zlib1_s"]:.1f}s')

jobs = [(i, j) for i, d in enumerate(DS) for j in range(len(d['raw']))]
t = time.time()
with ThreadPoolExecutor(NTHR) as ex:
    list(ex.map(lambda ij: inflate_cpu(DS[ij[0]], DS[ij[0]]['raw'][ij[1]]), jobs))
T['zlibN_s'] = time.time() - t
log(f'zlib x{NTHR}: {T["zlibN_s"]:.1f}s')

# check: one chunk per dataset against h5py
bad = 0
for i, d in enumerate(DS):
    if not d['ck'] or not d['gpu_ok']:
        continue
    off = d['ck'][0][0]
    sl = tuple(slice(o, min(o + c, n)) for o, c, n in zip(off, d['chunks'], d['shape']))
    ref = f[d['path']][sl]
    got = np.frombuffer(ZOUT[i][0], d['dtype']).reshape(d['chunks'])
    got = got[tuple(slice(0, s.stop - s.start) for s in sl)]
    if not np.array_equal(ref, got, equal_nan=ref.dtype.kind == 'f'):
        bad += 1
        log('MISMATCH zlib vs h5py', d['path'])
T['h5py_check_bad'] = bad

# ---- 5. GPU: nvCOMP Deflate --------------------------------------------
import cupy as cp
from nvidia import nvcomp

dev = int(os.environ.get('INFLATE_GPU', '0'))
cp.cuda.Device(dev).use()
codec = nvcomp.Codec(algorithm='Deflate', bitstream_kind=nvcomp.BitstreamKind.RAW,
                     device_id=dev)


def unshuffle_cp(b, itemsize):
    n = b.size // itemsize
    if itemsize == 1 or n == 0:
        return b
    body = b[:n * itemsize].reshape(itemsize, n).T.ravel()
    return cp.concatenate([body, b[n * itemsize:]]) if b.size > n * itemsize else body


def to_cp(a):
    return cp.from_dlpack(a) if hasattr(a, '__dlpack__') else cp.asarray(a)


def gpu_batch(items):
    """items: [(i, j, device uint8 array of raw deflate)] -> {(i,j): cupy bytes}."""
    outs = codec.decode(nvcomp.as_arrays([x for _, _, x in items]))
    res = {}
    for (i, j, _), o in zip(items, outs):
        # copy: the views die with the nvCOMP batch (job 7756917 read freed memory)
        o = to_cp(o).view(cp.uint8).ravel().copy()
        res[(i, j)] = unshuffle_cp(o, DS[i]['itemsize']) if DS[i]['shuffle'] else o
    return res


# stage the compressed chunks in GPU memory (the HBM-resident case)
G = [i for i, d in enumerate(DS) if d['gpu_ok'] and d['ck'] and d['raw_chunk'] <= MAX_CHUNK]
for i, d in enumerate(DS):
    d['gpu_skipped'] = d['gpu_ok'] and d['raw_chunk'] > MAX_CHUNK
T['gpu_skipped'] = [(d['path'], round(d['raw_chunk'] / 2**20, 1)) for d in DS if d['gpu_skipped']]
T['gpu_skipped_unc_GB'] = sum(d['unc_bytes'] for d in DS if d['gpu_skipped']) / 1e9
T['gpu_skipped_zlib1_s'] = sum(d['zlib1_s'] for d in DS if d['gpu_skipped'])
log(f"gpu: {len(G)} datasets; {len(T['gpu_skipped'])} with chunks > {MAX_CHUNK/2**20:.0f} MiB left to the CPU "
    f"({T['gpu_skipped_unc_GB']:.2f} GB, zlib x1 {T['gpu_skipped_zlib1_s']:.1f}s):", T['gpu_skipped'][:12])
t = time.time()
DEV = {}
for i in G:
    d = DS[i]
    stripped = [zlib_strip(r, d['fl32']) for r in d['raw']]
    host = np.frombuffer(b''.join(stripped), np.uint8)
    buf = cp.asarray(host)
    o = 0
    DEV[i] = []
    for s in stripped:
        DEV[i].append(buf[o:o + len(s)]); o += len(s)
cp.cuda.Device(dev).synchronize()
T['h2d_s'] = time.time() - t
log(f'h2d: {T["h2d_s"]:.2f}s (incl. zlib header strip + concat on the host)')

# warm-up: CUDA context, nvCOMP kernels
if G:
    i0 = max(G, key=lambda i: len(DEV[i]))
    gpu_batch([(i0, 0, DEV[i0][0])]); cp.cuda.Device(dev).synchronize()

# (a) one decode call per dataset
t = time.time()
GOUT = {}
for i in G:
    t1 = time.time()
    GOUT.update(gpu_batch([(i, j, x) for j, x in enumerate(DEV[i])]))
    cp.cuda.Device(dev).synchronize()
    DS[i]['gpu_s'] = time.time() - t1
    d = DS[i]
    log(f"  gpu {d['gpu_s']:7.3f}s  zlib1 {d['zlib1_s']:7.3f}s  {len(d['ck']):4d} x "
        f"{d['raw_chunk']/2**20:6.2f} MiB  {d['path']}")
T['gpu_per_ds_s'] = time.time() - t
T['zlib1_same_ds_s'] = sum(DS[i]['zlib1_s'] for i in G)
log(f'gpu per dataset: {T["gpu_per_ds_s"]:.2f}s vs zlib x1 {T["zlib1_same_ds_s"]:.2f}s on the same datasets')

# verify every GPU chunk against zlib
mism = 0
for (i, j), o in GOUT.items():
    if cp.asnumpy(o).tobytes() != ZOUT[i][j]:
        mism += 1
T['gpu_vs_zlib_mismatch'] = mism
T['gpu_chunks'] = len(GOUT)
log(f'gpu vs zlib: {len(GOUT)} chunks, {mism} mismatches')
del GOUT

# (b) whole granule in as few calls as fit GPU_BATCH_GB uncompressed
allj = sorted(((i, j, x) for i in G for j, x in enumerate(DEV[i])),
              key=lambda t: -DS[t[0]]['raw_chunk'])
t = time.time()
batch, acc, ncalls = [], 0, 0
for it in allj:
    if batch and acc + DS[it[0]]['raw_chunk'] > BATCH:
        gpu_batch(batch); cp.cuda.Device(dev).synchronize(); batch, acc = [], 0; ncalls += 1
    batch.append(it); acc += DS[it[0]]['raw_chunk']
if batch:
    gpu_batch(batch); cp.cuda.Device(dev).synchronize(); ncalls += 1
T['gpu_all_s'] = time.time() - t
T['gpu_all_calls'] = ncalls
log(f'gpu whole granule: {T["gpu_all_s"]:.2f}s in {ncalls} call(s)')

# ---- 6. report -----------------------------------------------------------
def agg(key):
    g = defaultdict(lambda: defaultdict(float))
    for d in DS:
        k = key(d)
        g[k]['n_ds'] += 1; g[k]['chunks'] += len(d['ck'])
        g[k]['comp_GB'] += d['comp_bytes'] / 1e9; g[k]['unc_GB'] += d['unc_bytes'] / 1e9
        g[k]['sel_GB'] += d['nbytes_sel'] / 1e9
        g[k]['max_chunk_MB'] = max(g[k]['max_chunk_MB'], d['raw_chunk'] / 2**20)
        for s in ('io_s', 'h5py_s', 'zlib1_s', 'gpu_s'):
            g[k][s] += d.get(s, 0.0)
    return {k: dict(v) for k, v in g.items()}


rep = dict(granule=os.path.basename(F), gpu=cp.cuda.runtime.getDeviceProperties(dev)['name'].decode(),
           nvcomp=nvcomp.__version__, threads=NTHR, totals=T,
           by_stage_sensor={f'{a}/{b}': v for (a, b), v in agg(lambda d: (d['stage'], d['sensor'])).items()},
           datasets=[{k: v for k, v in d.items() if k not in ('raw', 'ck')} | {'nchunks': len(d['ck'])}
                     for d in DS])
json.dump(rep, open(OUT, 'w'), indent=1, default=str)

print(f'\n{"stage/sensor":20} {"chunks":>7} {"maxMB":>7} {"compGB":>7} {"uncGB":>7} '
      f'{"io":>6} {"h5py":>6} {"zlib1":>6} {"gpu":>6}')
for k, v in sorted(rep['by_stage_sensor'].items()):
    print(f'{k:20} {int(v["chunks"]):7d} {v["max_chunk_MB"]:7.1f} {v["comp_GB"]:7.2f} '
          f'{v["unc_GB"]:7.2f} {v["io_s"]:6.2f} {v["h5py_s"]:6.2f} {v["zlib1_s"]:6.2f} {v["gpu_s"]:6.2f}')
print('totals:', json.dumps(T))
