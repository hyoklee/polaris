#!/usr/bin/env python3
"""Rechunk the oversized chunks of an HDF5 granule (prototype of rechunk-on-ingest).

    python tf_rechunk.py <src.h5> <dst.h5>
    THRESH_MB  rechunk a dataset whose chunk is larger than this, uncompressed (32)
    TARGET_MB  largest new chunk, uncompressed (4)
    RECHUNK_INCLUDE  comma-separated dataset paths; only these are rechunked (default:
               every oversized one). Rechunking all 631 oversized datasets of
               O10903 took 1,555 s, most of them ASTER arrays the pipeline reads
               whole, so the prototype rechunks just what it slices:
               RECHUNK_INCLUDE=MISR/AN/Data_Fields/Red_Radiance
               (not INCLUDE: `module load` prepends compiler include dirs to it)

The rest of the file is copied with H5Ocopy, which moves the compressed chunks
as they are (no inflate, no deflate). Each oversized dataset is then recreated
from the source's own creation property list (same filters, zlib level, fill
value), with only the chunk shape changed: axes are split from the first one
on until a chunk fits TARGET_MB. MISR Red_Radiance (180,512,2048) in one
755 MB chunk becomes {1,512,2048}, one SOM block per chunk, as in ares part 3.

Dimension scales: every DIMENSION_LIST in these granules points into another
top-level group (O12068: 422 of 422), and H5Ocopy remaps references only
within one copy call, so after the per-group copies all of them dangle and
netCDF4 refuses the file (NetCDF: HDF error). The scale links are therefore
rebuilt by name with H5DSattach_scale once everything is in place.

The data is read one SOURCE chunk at a time (iter_chunks), so every source
chunk is inflated exactly once. Slicing the new layout out of the source
instead re-inflates the whole source chunk per slice (ares part 3's trap).

Prints one JSON line: datasets rechunked, bytes, and copy/rechunk seconds.
"""
import json, math, os, re, sys, time
import h5py, numpy as np

THRESH = float(os.environ.get('THRESH_MB', 32)) * 2**20
TARGET = float(os.environ.get('TARGET_MB', 4)) * 2**20
INCLUDE = {p.strip().strip('/') for p in os.environ.get('RECHUNK_INCLUDE', '').split(',') if p.strip()}


def new_chunks(chunks, itemsize):
    c = list(chunks); size = math.prod(c) * itemsize
    for i in range(len(c)):
        if size <= TARGET:
            break
        nc = max(1, math.ceil(c[i] / math.ceil(size / TARGET)))
        size = size // c[i] * nc; c[i] = nc
    return tuple(c)


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    t0 = time.time()
    src = h5py.File(src_path, 'r')
    big, scales = [], {}
    def collect(n, o):
        if isinstance(o, h5py.Dataset) and 'DIMENSION_LIST' in o.attrs:
            scales[n] = [[sc.name for sc in dim.values()] for dim in o.dims]
    src.visititems(collect)
    src.visititems(lambda n, o: big.append(n) if isinstance(o, h5py.Dataset) and o.chunks
                   and (not INCLUDE or n in INCLUDE) and math.prod(o.chunks) * o.dtype.itemsize > THRESH else None)
    dst = h5py.File(dst_path, 'w')
    for k, v in src.attrs.items():
        dst.attrs.create(k, v, dtype=src.attrs.get_id(k).dtype)
    for k in src:
        src.copy(src[k], dst, name=k)
    t_copy = time.time() - t0

    t1 = time.time(); report = []
    for name in big:
        s = src[name]
        nc = new_chunks(s.chunks, s.dtype.itemsize)
        parent, leaf = name.rsplit('/', 1) if '/' in name else ('/', name)
        attrs = [(k, s.attrs[k], s.attrs.get_id(k).dtype) for k in s.attrs
                 if k not in ('DIMENSION_LIST', 'REFERENCE_LIST')]
        del dst[name]
        dcpl = s.id.get_create_plist().copy()
        dcpl.set_chunk(nc)
        dsid = h5py.h5d.create(dst[parent].id, leaf.encode(), s.id.get_type(),
                               s.id.get_space(), dcpl=dcpl)
        d = h5py.Dataset(dsid)
        for k, v, dt in attrs:
            d.attrs.create(k, v, dtype=dt)
        tr = time.time()
        for sel in s.iter_chunks():
            d[sel] = s[sel]
        report.append(dict(name=name, shape=s.shape, old=s.chunks, new=nc,
                           old_MB=round(s.id.get_storage_size() / 2**20, 1),
                           new_MB=round(d.id.get_storage_size() / 2**20, 1),
                           s=round(time.time() - tr, 2)))
    # Rebuild every scale link (see the docstring): drop the copied, dangling
    # REFERENCE_LIST / DIMENSION_LIST attributes, then attach by name.
    t2 = time.time()
    for n in {sn for dims in scales.values() for names in dims for sn in names}:
        if 'REFERENCE_LIST' in dst[n].attrs:
            del dst[n].attrs['REFERENCE_LIST']
    for n, dims in scales.items():
        d = dst[n]
        if 'DIMENSION_LIST' in d.attrs:
            del d.attrs['DIMENSION_LIST']
        for i, names in enumerate(dims):
            for sn in names:
                d.dims[i].attach_scale(dst[sn])
    t_scales = time.time() - t2
    dst.close(); src.close()
    print(json.dumps(dict(src=src_path, dst=dst_path, copy_s=round(t_copy, 1),
                          rechunk_s=round(t2 - t1, 1), scales_s=round(t_scales, 1),
                          scale_links=sum(len(x) for d in scales.values() for x in d),
                          src_GB=round(os.path.getsize(src_path) / 1e9, 2),
                          dst_GB=round(os.path.getsize(dst_path) / 1e9, 2),
                          rechunked=report)))


if __name__ == '__main__':
    main()
