"""Build `regrid_inputs*.npz`: the strip-wide per-sensor (lat, lon, value)
arrays that tf_congruence.py consumes.

This step existed only as an ad-hoc snippet in the original session, so
tf_congruence.py's documented "needs regrid_inputs.npz" could not actually be
satisfied from the repo. Extracted here from tf_discrepancy.py's loader half so
the congruence stage is reproducible.

The one new knob is MODIS_BAND_IDX. `EV_1KM_Emissive` carries 16 bands
(band_names = 20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36) and parts 2-4
always read index 0 -- band 20 at 3.75 um, which mixes reflected solar with
thermal emission. Part 4 found MODIS the weakest sensor in 19 of 32 blocks and
could not say whether that is a processing fault or that band's physics. This
parameter is what makes the control possible:

    MODIS_BAND_IDX=0   band 20   3.75 um   reflective + thermal (the original)
    MODIS_BAND_IDX=8   band 29   8.55 um   nearest ASTER TIR-10 (8.29 um)
    MODIS_BAND_IDX=10  band 31   11.03 um  clean thermal window

Usage:
    MODIS_BAND_IDX=10 python3 bin/tf_sources.py      # -> regrid_inputs_b31.npz
"""
import netCDF4, numpy as np, json, time, os, sys

F = os.environ.get(
    'TF_GRANULE',
    '/mnt/common/datasets-staging/TERRA_BF_L1B_O10204_20011118010522_F000_V001.h5')
BAND_IDX = int(os.environ.get('MODIS_BAND_IDX', '0'))
# CERES channel. LW_Radiance (parts 2-10) is BROADBAND longwave (~5-100 um);
# WN_Radiance is the 8-12 um WINDOW channel, which overlaps ASTER TIR-10
# (8.29 um) and MODIS band 31 (11.03 um) directly. Swapping it is the CERES-side
# analogue of part 10's MODIS band control, and separates a spectral cause
# (broadband integral) from a geometric one (~20 km footprint vs a 2 km grid).
CERES_FIELD = os.environ.get('CERES_FIELD', 'LW_Radiance')
# MOPITT channel index into MOPITTRadiances[...,chan,state]. Only 4..7 carry
# data. The file does not record a wavelength per index, so these are reported
# by index rather than by an assumed band.
MOPITT_CHAN = int(os.environ.get('MOPITT_CHAN', '4'))
MOPITT_STATE = int(os.environ.get('MOPITT_STATE', '0'))
PAD = 0.05

blocks = json.load(open('aster_blocks.json'))
d = netCDF4.Dataset(F)
LA0 = min(b['lat0'] for b in blocks) - PAD; LA1 = max(b['lat1'] for b in blocks) + PAD
LO0 = min(b['lon0'] for b in blocks) - PAD; LO1 = max(b['lon1'] for b in blocks) + PAD


def ok(la, lo):
    return (np.isfinite(la) & np.isfinite(lo) & (np.abs(la) <= 90) &
            (np.abs(lo) <= 180) & ((la != 0) | (lo != 0)))


def vmask(var, a):
    """Valid-data mask from the variable's own metadata. valid_min/valid_max is
    the only reliable test: MOPITT carries a SECOND sentinel (-8888) beyond its
    _FillValue of -9999, and a fill-value-only test lets it through."""
    m = np.isfinite(a)
    vmin = getattr(var, 'valid_min', None); vmax = getattr(var, 'valid_max', None)
    vr = getattr(var, 'valid_range', None)   # CERES spells it this way
    if vr is not None and len(vr) == 2:
        if vmin is None: vmin = vr[0]
        if vmax is None: vmax = vr[1]
    if vmin is not None: m &= (a >= float(vmin))
    if vmax is not None: m &= (a <= float(vmax))
    fv = getattr(var, '_FillValue', None)
    if fv is not None: m &= (a != float(fv))
    if vmin is None and vmax is None: m &= (np.abs(a) < 1e30)
    return m


SRC = {}; T = {}

# ---- MODIS -----------------------------------------------------------------
t = time.time(); mla = []; mlo = []; mv = []; band_label = None
for gn, g in d['MODIS'].groups.items():
    if '_1KM' not in g.groups: continue
    G = g['_1KM']['Geolocation']
    la = np.asarray(G['Latitude'][:], 'f8'); lo = np.asarray(G['Longitude'][:], 'f8')
    m = ok(la, lo) & (la >= LA0) & (la <= LA1) & (lo >= LO0) & (lo <= LO1)
    if not m.any(): continue
    EV = g['_1KM']['Data_Fields']['EV_1KM_Emissive']
    if band_label is None:
        names = getattr(EV, 'band_names', '')
        parts = [s.strip() for s in str(names).split(',') if s.strip()]
        band_label = parts[BAND_IDX] if BAND_IDX < len(parts) else str(BAND_IDX)
    v = np.asarray(EV[BAND_IDX, :, :], 'f8')
    m &= vmask(EV, v)
    mla.append(la[m]); mlo.append(lo[m]); mv.append(v[m])
SRC['MODIS'] = (np.concatenate(mla), np.concatenate(mlo), np.concatenate(mv))
T['read_MODIS'] = time.time() - t

# ---- MISR ------------------------------------------------------------------
t = time.time()
gl = d['MISR']['Geolocation']
la = np.asarray(gl['GeoLatitude'][:], 'f8'); lo = np.asarray(gl['GeoLongitude'][:], 'f8')
m = ok(la, lo) & (la >= LA0) & (la <= LA1) & (lo >= LO0) & (lo <= LO1)
rr = d['MISR']['AN']['Data_Fields']['Red_Radiance']      # single 755 MB chunk
if os.environ.get('TF_MISR_SUBSET', '0') == '1':
    # Read only the SOM blocks the strip touches. On the original single chunk
    # this inflates all 755 MB anyway; on a rechunked file ({1,512,2048}, see
    # tf_rechunk.py) it inflates just those blocks. The values are the same.
    bl = np.flatnonzero(m.any(axis=(1, 2)))
    b0, b1 = (int(bl[0]), int(bl[-1]) + 1) if bl.size else (0, 0)
    v = np.full(m.shape, np.nan)
    if b1 > b0:
        v[b0:b1] = np.asarray(rr[b0:b1], 'f8')[:, ::4, ::4]
    T['MISR_blocks'] = b1 - b0
else:
    v = np.asarray(rr[:], 'f8')                          # must read the whole chunk
    v = v[:, ::4, ::4]   # (180,512,2048) radiance vs (180,128,512) geoloc
m &= vmask(rr, v)
SRC['MISR'] = (la[m], lo[m], v[m]); T['read_MISR'] = time.time() - t

# ---- CERES -----------------------------------------------------------------
t = time.time(); cla = []; clo = []; cv = []
for gn, g in d['CERES'].groups.items():
    for fm in ('FM1', 'FM2'):
        if fm not in g.groups: continue
        tp = g[fm]['Time_and_Position']
        la = np.asarray(tp['Latitude'][:], 'f8'); lo = np.asarray(tp['Longitude'][:], 'f8')
        m = ok(la, lo) & (la >= LA0) & (la <= LA1) & (lo >= LO0) & (lo <= LO1)
        if not m.any(): continue
        LW = g[fm]['Radiances'][CERES_FIELD]
        v = np.asarray(LW[:], 'f8'); m &= vmask(LW, v)
        cla.append(la[m]); clo.append(lo[m]); cv.append(v[m])
SRC['CERES'] = (np.concatenate(cla), np.concatenate(clo), np.concatenate(cv))
T['read_CERES'] = time.time() - t

# ---- MOPITT ----------------------------------------------------------------
t = time.time()
# The MOPITT granule group is named for the DATE, so it differs per orbit.
# Hardcoding one name silently restricted this script to a single granule.
_mop = None
for _gn, _g in d['MOPITT'].groups.items():
    if 'Geolocation' in _g.groups and 'Data_Fields' in _g.groups:
        _mop = _g
        break
if _mop is None:
    raise SystemExit('no MOPITT granule with Geolocation+Data_Fields in ' + F)
G = _mop['Geolocation']
la = np.asarray(G['Latitude'][:], 'f8'); lo = np.asarray(G['Longitude'][:], 'f8')
m = ok(la, lo) & (la >= LA0) & (la <= LA1) & (lo >= LO0) & (lo <= LO1)
MR = _mop['Data_Fields']['MOPITTRadiances']
v = np.asarray(MR[:, :, :, MOPITT_CHAN, MOPITT_STATE], 'f8')   # only 4..7 hold data
m &= vmask(MR, v)
SRC['MOPITT'] = (la[m], lo[m], v[m]); T['read_MOPITT'] = time.time() - t

out = {}
# A sensor with no points inside the ASTER strip is a real data condition, not a
# bug: O10437 has only 2 ASTER granules, so its strip is small enough that the
# sparse sensors (CERES ~60/block, MOPITT ~17/block) can miss it entirely. This
# used to surface as "zero-size array to reduction operation minimum", which
# names neither the sensor nor the cause.
empty = [k for k, (a, b, c) in SRC.items() if a.size == 0]
if empty:
    raise SystemExit(
        f"no points inside the ASTER strip for: {', '.join(empty)}.\n"
        f"  strip lat {LA0:.2f}..{LA1:.2f}  lon {LO0:.2f}..{LO1:.2f}\n"
        f"  This granule has {len(blocks)} ASTER block(s); a short strip can miss\n"
        f"  the sparse sensors entirely. Nothing to assimilate -- skipping.")

for k, (a, b, c) in SRC.items():
    good = np.isfinite(c)
    a = np.ascontiguousarray(a[good]); b = np.ascontiguousarray(b[good])
    c = np.ascontiguousarray(c[good])
    out[k + '_lat'] = a; out[k + '_lon'] = b; out[k + '_val'] = c
    print(f'# {k:7} {a.size:8d} pts  read {T["read_"+k]:6.2f}s  '
          f'val[{c.min():.4g},{c.max():.4g}]', flush=True)

tag = f'b{band_label}'
if CERES_FIELD != 'LW_Radiance':
    tag += '_' + CERES_FIELD.replace('_Radiance', '').replace('_Filtered', 'F')
if (MOPITT_CHAN, MOPITT_STATE) != (4, 0):
    tag += f'_m{MOPITT_CHAN}{MOPITT_STATE}'
path = os.environ.get('TF_NPZ', f'regrid_inputs_{tag}.npz')
np.savez(path, **out)
json.dump(dict(timings=T, modis_band_idx=BAND_IDX, modis_band=band_label,
               ceres_field=CERES_FIELD, mopitt_chan=MOPITT_CHAN,
               mopitt_state=MOPITT_STATE, npz=path, granule=F),
          open(f'sources_{tag}.json', 'w'), indent=1)
print(f'\n# MODIS band index {BAND_IDX} -> band {band_label}; CERES {CERES_FIELD}')
print(f'# wrote {path}')
