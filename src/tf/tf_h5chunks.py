"""HDF5 chunk helpers shared by tf_gpu_inflate.py and tf_gpu_reencode.py: the
datasets and selections the congruence pipeline reads, their filters and the
chunks a selection touches."""
import os
import numpy as np

BAND_IDX = int(os.environ.get('MODIS_BAND_IDX', '10'))
CERES_FIELD = os.environ.get('CERES_FIELD', 'WN_Radiance')
MOPITT_CHAN = int(os.environ.get('MOPITT_CHAN', '4'))
MOPITT_STATE = int(os.environ.get('MOPITT_STATE', '0'))
H5Z_DEFLATE, H5Z_SHUFFLE, H5Z_FLETCHER32 = 1, 2, 3


def pipeline_reads(f):
    """(stage, sensor, path, selection) as the pipeline reads them. A selection
    is a tuple of (lo, hi) per axis, or None for the whole dataset."""
    r = []
    for gn in sorted(f['ASTER']):                       # tf_aster_blocks.py
        g = f['ASTER'][gn]
        if 'Geolocation' in g:
            for v in ('Latitude', 'Longitude'):
                r.append(('blocks', 'ASTER', g['Geolocation'][v].name, None))
    for gn in sorted(f['ASTER']):                       # tf_congruence.py
        g = f['ASTER'][gn]
        if 'Geolocation' in g and 'TIR' in g and 'ImageData10' in g['TIR']:
            t = g['TIR']
            for v in ('Latitude', 'Longitude'):
                r.append(('congruence', 'ASTER', t['Geolocation'][v].name, None))
            r.append(('congruence', 'ASTER', t['ImageData10'].name, None))
    for gn in f['MODIS']:                               # tf_sources.py
        g = f['MODIS'][gn]
        if '_1KM' not in g: continue
        k = g['_1KM']
        for v in ('Latitude', 'Longitude'):
            r.append(('sources', 'MODIS', k['Geolocation'][v].name, None))
        ev = k['Data_Fields']['EV_1KM_Emissive']
        sel = ((BAND_IDX, BAND_IDX + 1),) + tuple((0, n) for n in ev.shape[1:])
        r.append(('sources', 'MODIS', ev.name, sel))
    m = f['MISR']
    for v in ('GeoLatitude', 'GeoLongitude'):
        r.append(('sources', 'MISR', m['Geolocation'][v].name, None))
    r.append(('sources', 'MISR', m['AN']['Data_Fields']['Red_Radiance'].name, None))
    for gn in f['CERES']:
        for fm in ('FM1', 'FM2'):
            if fm not in f['CERES'][gn]: continue
            c = f['CERES'][gn][fm]
            for v in ('Latitude', 'Longitude'):
                r.append(('sources', 'CERES', c['Time_and_Position'][v].name, None))
            r.append(('sources', 'CERES', c['Radiances'][CERES_FIELD].name, None))
    for gn in f['MOPITT']:
        g = f['MOPITT'][gn]
        if 'Geolocation' in g and 'Data_Fields' in g:
            for v in ('Latitude', 'Longitude'):
                r.append(('sources', 'MOPITT', g['Geolocation'][v].name, None))
            mr = g['Data_Fields']['MOPITTRadiances']
            sel = tuple((0, n) for n in mr.shape[:3]) + \
                ((MOPITT_CHAN, MOPITT_CHAN + 1), (MOPITT_STATE, MOPITT_STATE + 1))
            r.append(('sources', 'MOPITT', mr.name, sel))
            break
    return r


def filters(ds):
    p = ds.id.get_create_plist()
    return [p.get_filter(i)[0] for i in range(p.get_nfilters())]


def chunks(ds, sel):
    """[(chunk offset, file offset, nbytes, filter_mask)] intersecting sel."""
    out = []
    cs = ds.chunks
    def cb(ci):
        if sel is None or all(o < hi and o + c > lo
                              for o, c, (lo, hi) in zip(ci.chunk_offset, cs, sel)):
            out.append((ci.chunk_offset, ci.byte_offset, ci.size, ci.filter_mask))
    ds.id.chunk_iter(cb)
    return out


def h5sel(sel):
    return tuple(slice(lo, hi) for lo, hi in sel) if sel else ()


def unshuffle_np(b, itemsize):
    n = len(b) // itemsize
    if itemsize == 1 or n == 0:
        return b
    a = np.frombuffer(b, np.uint8)
    body = a[:n * itemsize].reshape(itemsize, n).T.ravel()
    return body.tobytes() + bytes(a[n * itemsize:])


def zlib_strip(raw, fl32):
    """zlib stream (2-byte header, deflate, adler32) -> raw deflate."""
    if fl32:
        raw = raw[:-4]
    assert raw[0] & 0x0F == 8 and not raw[1] & 0x20, 'not a plain zlib stream'
    return raw[2:-4]


def inventory(f):
    """One dict per pipeline read: path, stage, sensor, dtype, chunks, filters,
    gpu_ok (deflate [+ shuffle, fletcher32] only) and ck, the chunk list."""
    DS = []
    for stage, sensor, path, sel in pipeline_reads(f):
        ds = f[path]
        fl = filters(ds) if ds.chunks else []
        d = dict(stage=stage, sensor=sensor, path=path, shape=list(ds.shape),
                 dtype=ds.dtype.str, itemsize=ds.dtype.itemsize,
                 chunks=list(ds.chunks) if ds.chunks else None, filters=fl,
                 sel=sel, nbytes_sel=int(np.prod([hi - lo for lo, hi in sel]) if sel
                                         else np.prod(ds.shape)) * ds.dtype.itemsize)
        d['gpu_ok'] = bool(ds.chunks) and H5Z_DEFLATE in fl and \
            set(fl) <= {H5Z_DEFLATE, H5Z_SHUFFLE, H5Z_FLETCHER32}
        d['shuffle'] = H5Z_SHUFFLE in fl
        d['fl32'] = H5Z_FLETCHER32 in fl
        d['ck'] = chunks(ds, sel) if ds.chunks else []
        if any(c[3] for c in d['ck']):
            d['gpu_ok'] = False                # a filter was skipped for some chunk
        d['raw_chunk'] = int(np.prod(ds.chunks)) * ds.dtype.itemsize if ds.chunks else 0
        DS.append(d)
    return DS
