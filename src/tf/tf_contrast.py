"""Per-block scene contrast: MODIS band-31 brightness temperature mean and sd.

Part 13/17 of the ares series showed C5 is partly a scene-contrast measure, so
the collection ranking drops blocks whose band-31 BT spatial sd is under a 2 K
floor. The ares repo carries the resulting `bt31_mean`/`bt31_sd` fields in
data/all_granule_ranking.json but not the script that computed them, so this
reconstructs them; tf_compare_ares.py checks the reconstruction against the
ares values block by block.

BT is taken over the block's 0.02 deg target grid: the MODIS points
tf_sources.py extracted (EV_1KM_Emissive band 31 radiance, within 0.05 deg of
the block) are regridded nearest-neighbour within 3 km with pytaf, exactly as
tf_congruence.py does, and the inverse Planck function at 11.03 um is applied to
every cell that got a value. This reproduces the ares values exactly (O11835:
max |d mean| = max |d sd| = 0.000 K over its 14 blocks); the same statistic over
the raw MODIS pixels differs by up to 0.33 K.

Usage (in a granule's work directory, after tf_congruence.py):
    TF_GRANULE=... TF_NPZ=regrid_inputs_b31_WN.npz TF_TAG=b31_WN python3 tf_contrast.py
    -> contrast_<TAG>.json   {blk: {"bt31_mean": K, "bt31_sd": K, "n": points}}
"""
import json, os, sys
import numpy as np
import netCDF4
sys.path.insert(0, os.path.expanduser(os.environ.get('PYTAF_DIR', '~/src/TerraFusion/pytaf')))
import pytaf

F = os.environ['TF_GRANULE']
NPZ = os.environ.get('TF_NPZ', 'regrid_inputs_b31_WN.npz')
TAG = os.environ.get('TF_TAG', 'b31_WN')
BAND_IDX = int(os.environ.get('MODIS_BAND_IDX', '10'))   # band 31
LAMBDA_UM = 11.03                                          # band 31 centre
C1 = 1.191042e8      # 2hc^2, W um^4 m^-2 sr^-1
C2 = 1.4387752e4     # hc/k,  um K
RES, PAD, MAXR = 0.02, 0.05, 3000.0   # tf_congruence.py's grid, pad, MODIS radius

d = netCDF4.Dataset(F)
# Terra Fusion stores EV_1KM_Emissive as RADIANCE (W m^-2 um^-1 sr^-1; band 31
# spans ~3-10 here), unlike MODIS L1B's scaled integers. Apply
# radiance_scales/offsets only if a granule carries them.
scale, offset, units = 1.0, 0.0, None
for g in d['MODIS'].groups.values():
    if '_1KM' in g.groups:
        ev = g['_1KM']['Data_Fields']['EV_1KM_Emissive']
        attrs = ev.ncattrs()
        units = ev.getncattr('units') if 'units' in attrs else None
        if 'radiance_scales' in attrs:
            scale = float(np.ravel(ev.radiance_scales)[BAND_IDX])
            offset = float(np.ravel(ev.radiance_offsets)[BAND_IDX])
        break
print(f'# EV_1KM_Emissive units={units!r} scale={scale} offset={offset}')



def regrid(sla, slo, sval, tla, tlo, maxr):
    """tf_congruence.py's pytaf nearest-neighbour regrid."""
    nS, nT = sla.size, tla.size
    nnid = np.full(nT, -1, dtype=np.int32); nnd = np.zeros((1, nT))
    pytaf.find_nn_block_index(sla.reshape(1, -1).copy(), slo.reshape(1, -1).copy(), nS,
                              tla.reshape(1, -1).copy(), tlo.reshape(1, -1).copy(),
                              nnid, nnd, nT, maxr)
    tv = np.full((1, nT), np.nan)
    pytaf.interpolate_nn(sval.reshape(1, -1).copy(), tv, nnid, nT)
    out = tv.ravel().copy(); out[nnid < 0] = np.nan
    return out


D = np.load(NPZ)
la, lo, val = D['MODIS_lat'], D['MODIS_lon'], D['MODIS_val']
blocks = json.load(open('aster_blocks.json'))
rows = json.load(open(f'congruence_{TAG}.json'))
out = {}
for r in rows:
    b = blocks[r['blk']]
    glat = np.arange(b['lat0'], b['lat1'] + RES, RES)
    glon = np.arange(b['lon0'], b['lon1'] + RES, RES)
    TLA, TLO = np.meshgrid(glat, glon, indexing='ij')
    m = ((la >= b['lat0'] - PAD) & (la <= b['lat1'] + PAD) &
         (lo >= b['lon0'] - PAD) & (lo <= b['lon1'] + PAD))
    if m.sum() < 5:
        continue
    g = regrid(la[m].copy(), lo[m].copy(), val[m].copy(), TLA.ravel(), TLO.ravel(), MAXR)
    rad = scale * (g[np.isfinite(g)] - offset)          # W m^-2 um^-1 sr^-1
    rad = rad[rad > 0]
    if rad.size < 10:
        continue
    bt = C2 / (LAMBDA_UM * np.log1p(C1 / (LAMBDA_UM ** 5 * rad)))
    out[str(r['blk'])] = dict(bt31_mean=float(bt.mean()), bt31_sd=float(bt.std()),
                              n=int(bt.size))
json.dump(out, open(f'contrast_{TAG}.json', 'w'), indent=1)
print(f'# contrast for {len(out)}/{len(rows)} blocks -> contrast_{TAG}.json')
