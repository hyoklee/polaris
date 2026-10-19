"""Write the input files of the CLIO GPU-tier benchmark (tf_clio_nvcomp_polaris.pbs)
for one granule:

  A_<orbit>.bin  the arrays the congruence pipeline reads, decoded by h5py and
                 concatenated in pipeline order: what a GPU consumer would hold
  H_<orbit>.bin  SAMPLE_MB of the granule's own bytes as stored (deflated HDF5),
                 random 1 MiB pages: what CLIO holds after an S3/CAE import

usage: python tf_dump_arrays.py <granule.h5> <out dir>
"""
import json
import os
import random
import sys
import time

import h5py
import numpy as np

from tf_h5chunks import pipeline_reads, h5sel

F, OUT = sys.argv[1], sys.argv[2]
SAMPLE_MB = int(os.environ.get('SAMPLE_MB', '2048'))
orbit = os.path.basename(F).split('_')[3]
os.makedirs(OUT, exist_ok=True)

t = time.time()
meta = []
with h5py.File(F, 'r') as f, open(os.path.join(OUT, f'A_{orbit}.bin'), 'wb') as o:
    for stage, sensor, path, sel in pipeline_reads(f):
        a = np.ascontiguousarray(f[path][h5sel(sel)])
        meta.append(dict(stage=stage, sensor=sensor, path=path, dtype=a.dtype.str,
                         shape=list(a.shape), offset=o.tell(), nbytes=a.nbytes))
        o.write(a.tobytes())
print(f'A_{orbit}.bin: {len(meta)} arrays, {sum(m["nbytes"] for m in meta) / 1e9:.3f} GB '
      f'in {time.time() - t:.1f} s', flush=True)

t = time.time()
size = os.path.getsize(F)
pages = sorted(random.Random(108).sample(range(size >> 20), SAMPLE_MB))
with open(F, 'rb') as i, open(os.path.join(OUT, f'H_{orbit}.bin'), 'wb') as o:
    for p in pages:
        i.seek(p << 20)
        o.write(i.read(1 << 20))
print(f'H_{orbit}.bin: {SAMPLE_MB} x 1 MiB random pages in {time.time() - t:.1f} s', flush=True)

with open(os.path.join(OUT, f'A_{orbit}.json'), 'w') as j:
    json.dump(meta, j, indent=1)
