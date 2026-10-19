#!/usr/bin/env python3
"""Option D, part 2: what GPU inflate costs if CLIO re-encodes at ingest.

nvCOMP decodes many independent streams in parallel, but each stream on its own
is decoded almost serially. HDF5 stores each chunk as one deflate stream, and
these granules use chunks of several MB up to 755 MB (MISR Red_Radiance), so
the GPU sees a few huge streams. This measures:

  1. stream size: GPU (nvCOMP Deflate RAW) vs zlib on one stream of real data,
     cut at 64 KiB .. MAX_STREAM_MB, the same bytes re-deflated with zlib
  2. re-encode: the pipeline's datasets (uncompressed, from the HDF5 chunks)
     compressed on the GPU in 64 KiB nvCOMP chunks with each ALGOS codec;
     compressed size against HDF5's, and decode time with the compressed data
     already in GPU memory (the HBM-tier case), checked against the original

Usage: tf_gpu_reencode.py <granule.h5> <out.json>
Environment: ALGOS (Deflate,GDeflate,LZ4,Zstd,ANS,Bitcomp,Cascaded),
  MAX_STREAM_MB (16), ZLEVEL (4, the zlib level for part 1).
"""
import json, os, subprocess, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tf_h5chunks import inventory

F, OUT = sys.argv[1], sys.argv[2]
ALGOS = os.environ.get('ALGOS', 'Deflate,GDeflate,LZ4,Zstd,ANS,Bitcomp,Cascaded').split(',')
MAX_STREAM = int(float(os.environ.get('MAX_STREAM_MB', '16')) * 2**20)
ZLEVEL = int(os.environ.get('ZLEVEL', '4'))
CHILD = os.environ.get('REENCODE_CHILD') == '1'   # one codec, part 2 only


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


import cupy as cp
from nvidia import nvcomp
dev = cp.cuda.Device(0); dev.use()


def sync():
    dev.synchronize()


def to_cp(a):
    return (cp.from_dlpack(a) if hasattr(a, '__dlpack__') else cp.asarray(a)).view(cp.uint8).ravel()


# ---- uncompressed pipeline data, from the HDF5 chunks ------------------------
f = h5py.File(F, 'r')
DS = [d for d in inventory(f) if d['gpu_ok']]
fd = os.open(F, os.O_RDONLY)
hdf5_comp = 0
plain = []                       # one bytes object per dataset (all its chunks)


def inflate_ds(d):
    raws = [os.pread(fd, n, off) for _, off, n, _ in d['ck']]
    return sum(map(len, raws)), b''.join(zlib.decompress(r) for r in raws)


t = time.time()
with ThreadPoolExecutor(16) as ex:
    for c, p in ex.map(inflate_ds, DS):
        hdf5_comp += c; plain.append(p)
unc = sum(map(len, plain))
log(f'{len(DS)} deflate datasets: HDF5 {hdf5_comp/1e9:.3f} GB compressed, '
    f'{unc/1e9:.3f} GB uncompressed (ratio {unc/hdf5_comp:.2f}), read+inflate {time.time()-t:.1f}s')
R = dict(granule=os.path.basename(F), datasets=len(DS), hdf5_comp_bytes=hdf5_comp,
         unc_bytes=unc, stream=[], reencode={})

# ---- 1. one stream, growing ---------------------------------------------------
big = max(plain, key=len) if not CHILD else b''
codec = nvcomp.Codec(algorithm='Deflate', bitstream_kind=nvcomp.BitstreamKind.RAW, device_id=0)
size = 64 << 10
while size <= min(MAX_STREAM, len(big)):
    src = big[:size]
    c = zlib.compress(src, ZLEVEL)[2:-4]
    x = cp.asarray(np.frombuffer(c, np.uint8))
    codec.decode(nvcomp.as_arrays([x])); sync()                     # warm-up
    t = time.time(); o = codec.decode(nvcomp.as_arrays([x])); sync(); g = time.time() - t
    ok = cp.asnumpy(to_cp(list(o)[0])).tobytes() == src
    t = time.time(); zlib.decompress(c, -15); z = time.time() - t
    R['stream'].append(dict(MiB=size / 2**20, gpu_s=g, zlib_s=z, ok=ok))
    log(f'one stream {size/2**20:8.3f} MiB: gpu {g*1e3:9.2f} ms ({size/2**20/g:8.1f} MiB/s)  '
        f'zlib {z*1e3:8.2f} ms ({size/2**20/z:7.1f} MiB/s)  ok={ok}')
    if g > 20:
        log('stopping: one stream took > 20 s'); break
    size *= 4

# ---- 2. re-encode in 64 KiB nvCOMP chunks --------------------------------------
# Each codec runs in its own process: an illegal address in one poisons the CUDA
# context for all that follow (job 7756917).
if not CHILD:
    for alg in ALGOS:
        sub = f'{OUT}.{alg}'
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), F, sub],
                            env=dict(os.environ, ALGOS=alg, REENCODE_CHILD='1')).returncode
        try:
            R['reencode'][alg] = json.load(open(sub))['reencode'][alg]
            os.remove(sub)
        except Exception as e:
            R['reencode'][alg] = dict(error=f'child rc={rc}: {e}')
            log(f'{alg:9} failed: child rc={rc}')
    json.dump(R, open(OUT, 'w'), indent=1)
    sys.exit(0)

# one cupy allocation per dataset (aligned), not views into one buffer; empty
# inputs (a selection touching no chunk) are dropped: nvCOMP does not guard them
views = [cp.asarray(np.frombuffer(p, np.uint8)) for p in plain if len(p)]
log(f'{len(views)} non-empty inputs, {min(v.size for v in views)} .. {max(v.size for v in views)} bytes')
NB = int(os.environ.get('REENCODE_BATCH', '32'))


def batched(fn, xs):
    out = []
    for k in range(0, len(xs), NB):
        b = fn(nvcomp.as_arrays(xs[k:k + NB]))
        out.extend(to_cp(e).copy() for e in b)                     # own the bytes
        del b
    sync()
    return out
for alg in ALGOS:
    try:
        cdc = nvcomp.Codec(algorithm=alg, device_id=0, uncomp_chunk_size=65536)
        batched(cdc.encode, views)                                  # warm-up
        t = time.time(); enc = batched(cdc.encode, views); te = time.time() - t
        csize = sum(int(e.size) for e in enc)
        batched(cdc.decode, enc)                                    # warm-up
        t = time.time(); db = batched(cdc.decode, enc); td = time.time() - t
        ok = all(bool(cp.array_equal(a, v)) for a, v in zip(db, views))
        R['reencode'][alg] = dict(comp_bytes=csize, ratio=unc / csize, vs_hdf5=csize / hdf5_comp,
                                  encode_s=te, decode_s=td, ok=ok)
        log(f'{alg:9} ratio {unc/csize:5.2f} ({csize/hdf5_comp:5.2f}x HDF5 size)  '
            f'encode {te:6.3f}s  decode {td:6.3f}s ({unc/td/1e9:6.1f} GB/s)  ok={ok}')
        del enc, db
    except Exception as e:
        R['reencode'][alg] = dict(error=str(e)[:300])
        log(f'{alg:9} failed: {str(e)[:200]}')
json.dump(R, open(OUT, 'w'), indent=1)
