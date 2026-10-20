"""Dump the NVCOMP_NATIVE (manager / "HLIF") stream header that the installed
nvCOMP writes, to check the offsets clio-core's ParseBlobChunks assumes
(compressor_runtime.cc: magic 0x52911326 at +0, orig size at +16, chunk count
at +24, chunk size at +48, data start = 72 + 16 * nchunks at +56, then the
chunk offset and size tables). Issue 4 of
doc/20261019_claude_terra_fusion_clio_nvcomp_hbm_polaris_plan.md.

Each page is 256 KiB, 30% zeros + 70% compressible text-like bytes, compressed
with 64 KiB chunks like clio's NvComp (kChunkSize = 1 << 16).

usage: python nvcomp_hlif_probe.py
"""
import struct

import numpy as np
from nvidia import nvcomp

PAGE = 256 << 10
CHUNK = 64 << 10
MAGIC = 0x52911326

rng = np.random.default_rng(108)
page = np.zeros(PAGE, np.uint8)
page[int(PAGE * 0.3):] = rng.integers(0, 16, PAGE - int(PAGE * 0.3), dtype=np.uint8)
print('nvcomp', nvcomp.__version__)

for algo in ['LZ4', 'Snappy', 'Zstd', 'GDeflate', 'Deflate', 'ANS']:
    try:
        codec = nvcomp.Codec(algorithm=algo, bitstream_kind=nvcomp.BitstreamKind.NVCOMP_NATIVE,
                             chunk_size=CHUNK)
        enc = codec.encode(nvcomp.as_array(page))
        b = bytes(np.asarray(enc.cpu()).view(np.uint8))
    except Exception as e:   # one codec failing must not hide the others
        print(f'{algo}: ERROR {e}')
        continue
    q = struct.unpack_from('<16Q', b, 0)
    print(f'\n{algo}: {len(b)} bytes')
    for i in range(0, 160, 16):
        print(f'  +{i:3d}  {b[i:i + 16].hex(" ", 4)}')
    print('  u64 words:', ' '.join(f'{i * 8}:{v}' for i, v in enumerate(q)))
    for name, val in [('magic', MAGIC), ('orig', PAGE), ('chunk', CHUNK), ('nchunks', PAGE // CHUNK)]:
        offs = [o for o in range(0, min(len(b) - 4, 256), 4)
                if struct.unpack_from('<I', b, o)[0] == val]
        print(f'  {name}={val} found at u32 offsets {offs}')
