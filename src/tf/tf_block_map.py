#!/usr/bin/env python3
"""Map one congruence block: where it is, and what each sensor saw there.

    python tf_block_map.py <work/ORBIT dir> <block index> <s3 key> <out.png> [title]

    PYTAF_DIR   pytaf checkout (as tf_congruence.py)
    BUCKET_URL  https://terrafusiondatasampler.s3.amazonaws.com

The block's five fields are rebuilt exactly as tf_congruence.py builds them:
pytaf nearest-neighbour regrid onto the 0.02 deg target grid with the same
per-sensor radii, rank z-scores, the cells all sensors cover, re-standardised
and sign-aligned to the common mode. MODIS/MISR/CERES/MOPITT come from the
run's regrid_inputs_b31_WN.npz. ASTER TIR band 10 is read from the granule in
S3 with HTTP range requests (the granule itself was deleted after analysis;
only the ASTER granule's TIR group is fetched, not the 25-60 GB file). The
recomputed C5 is printed next to the ranked one as a check.

Left: an orthographic locator with the orbit's ASTER blocks and this one.
Right: one panel per sensor, sign-aligned rank z on a shared diverging scale
(blue low, red high), so congruent sensors show the same pattern; the panel
title carries the sensor's loading on the common mode.
"""
import io, json, os, sys, urllib.request
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap

sys.path.insert(0, os.path.expanduser(os.environ.get('PYTAF_DIR', '~/src/TerraFusion/pytaf')))
import pytaf

WORK, BLK, KEY, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
TITLE = sys.argv[5] if len(sys.argv) > 5 else ''
URL = os.environ.get('BUCKET_URL', 'https://terrafusiondatasampler.s3.amazonaws.com') + '/' + KEY
RES, PAD = 0.02, 0.05
MAXR = {'ASTER': 3000.0, 'MODIS': 3000.0, 'MISR': 3000.0, 'CERES': 25000.0, 'MOPITT': 25000.0}
INST = ['ASTER', 'MODIS', 'MISR', 'CERES', 'MOPITT']

# Reference palette (dataviz skill): text inks, surface, blue<->red diverging
# pair through a neutral gray midpoint.
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#8a8984', '#fcfcfb'
LAND, SEA = '#e4e3de', '#f4f6f8'
DIV = LinearSegmentedColormap.from_list(
    'div', ['#174f9a', '#2a78d6', '#f0efec', '#e34948', '#a12a29'])
DIV.set_bad(SURF)


class RangeFile(io.RawIOBase):
    """Read-only file over HTTP range GETs, 4 MiB pages, cached."""
    PAGE = 4 << 20

    def __init__(s, url):
        s.url, s.pos, s.cache, s.nreq = url, 0, {}, 0
        r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60)
        s.size = int(r.headers['Content-Length'])

    def readable(s): return True
    def seekable(s): return True
    def tell(s): return s.pos

    def seek(s, off, whence=0):
        s.pos = off if whence == 0 else s.pos + off if whence == 1 else s.size + off
        return s.pos

    def page(s, i):
        if i not in s.cache:
            a = i * s.PAGE; b = min(a + s.PAGE, s.size) - 1
            for attempt in range(5):
                try:
                    req = urllib.request.Request(s.url, headers={'Range': f'bytes={a}-{b}'})
                    s.cache[i] = urllib.request.urlopen(req, timeout=120).read(); break
                except OSError:
                    if attempt == 4: raise
            s.nreq += 1
        return s.cache[i]

    def readinto(s, buf):
        n = min(len(buf), s.size - s.pos); out = 0
        while out < n:
            i, o = divmod(s.pos + out, s.PAGE); p = s.page(i)
            k = min(len(p) - o, n - out); buf[out:out + k] = p[o:o + k]; out += k
        s.pos += n; return n


def ok(la, lo):
    return (np.isfinite(la) & np.isfinite(lo) & (np.abs(la) <= 90) &
            (np.abs(lo) <= 180) & ((la != 0) | (lo != 0)))


def nc_read(ds):
    """Read like netCDF4 does by default: unpack scale/offset, and the
    attributes vmask tests, so the mask matches tf_congruence.py."""
    a = np.asarray(ds[()], 'f8'); at = ds.attrs
    fv = at.get('_FillValue'); m = np.isfinite(a)
    if fv is not None: m &= a != float(np.ravel(fv)[0])
    sf, ao = at.get('scale_factor'), at.get('add_offset')
    if sf is not None: a = a * float(np.ravel(sf)[0])
    if ao is not None: a = a + float(np.ravel(ao)[0])
    for att, op in (('valid_min', np.greater_equal), ('valid_max', np.less_equal)):
        v = at.get(att)
        if v is not None: m &= op(a, float(np.ravel(v)[0]))
    return a, m


def regrid(sla, slo, sval, tla, tlo, maxr):
    nS, nT = sla.size, tla.size
    nnid = np.full(nT, -1, dtype=np.int32); nnd = np.zeros((1, nT))
    pytaf.find_nn_block_index(sla.reshape(1, -1).copy(), slo.reshape(1, -1).copy(), nS,
                              tla.reshape(1, -1).copy(), tlo.reshape(1, -1).copy(),
                              nnid, nnd, nT, maxr)
    tv = np.full((1, nT), np.nan)
    pytaf.interpolate_nn(sval.reshape(1, -1).copy(), tv, nnid, nT)
    out = tv.ravel().copy(); out[nnid < 0] = np.nan
    return out


def rankz(a):
    out = np.full(a.shape, np.nan); m = np.isfinite(a)
    if m.sum() < 10: return out
    r = np.empty(m.sum()); r[np.argsort(a[m], kind='stable')] = np.arange(m.sum())
    out[m] = (r - r.mean()) / (r.std() if r.std() > 0 else 1)
    return out


# --- the block's fields -------------------------------------------------------
blocks = json.load(open(f'{WORK}/aster_blocks.json'))
b = blocks[BLK]
la = np.arange(b['lat0'], b['lat1'] + RES, RES); lo = np.arange(b['lon0'], b['lon1'] + RES, RES)
TLA, TLO = np.meshgrid(la, lo, indexing='ij'); tla, tlo = TLA.ravel(), TLO.ravel()

f = {}
rf = RangeFile(URL)
with h5py.File(io.BufferedReader(rf, buffer_size=1 << 20), 'r') as h:
    g = h['ASTER'][b['granule']]['TIR']
    av, vm = nc_read(g['ImageData10'])
    ala = np.asarray(g['Geolocation']['Latitude'][()], 'f8').ravel()
    alo = np.asarray(g['Geolocation']['Longitude'][()], 'f8').ravel()
    am = ok(ala, alo) & vm.ravel(); av = av.ravel()
    if am.sum() > 100:
        f['ASTER'] = regrid(ala[am], alo[am], av[am], tla, tlo, MAXR['ASTER'])
print(f'# ASTER {b["granule"]}: {rf.nreq} range GETs ({rf.nreq * 4} MiB)', flush=True)

D = np.load(f'{WORK}/regrid_inputs_b31_WN.npz')
for k in INST[1:]:
    sla, slo, sv = D[k + '_lat'], D[k + '_lon'], D[k + '_val']
    m = (sla >= b['lat0'] - PAD) & (sla <= b['lat1'] + PAD) & (slo >= b['lon0'] - PAD) & (slo <= b['lon1'] + PAD)
    if m.sum() >= 5: f[k] = regrid(sla[m], slo[m], sv[m], tla, tlo, MAXR[k])

names = [k for k in INST if k in f]
M = np.vstack([rankz(f[k]) for k in names])
good = np.all(np.isfinite(M), axis=0)
M = M[:, good]
for r in range(M.shape[0]):
    M[r] = (M[r] - M[r].mean()) / (M[r].std() if M[r].std() > 0 else 1)
w, v = np.linalg.eigh(np.corrcoef(M)); lead = v[:, int(np.argmax(w))]
sign = np.sign(lead); sign[sign == 0] = 1
Ms = M * sign[:, None]
ws, vs = np.linalg.eigh(np.corrcoef(Ms)); j = int(np.argmax(ws))
C = float(ws[j] / len(names))
load = {names[q]: float(abs(vs[q, j])) for q in range(len(names))}
weak = min(load, key=load.get)

res = [r for r in json.load(open(f'{WORK}/congruence_b31_WN.json')) if r['blk'] == BLK]
ranked = res[0]['congruence'] if res else float('nan')
print(f'# block {BLK}: {int(good.sum())} common cells, C5 recomputed {C:.4f}, ranked {ranked:.4f}', flush=True)

# --- figure -------------------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': MUTED})
fig = plt.figure(figsize=(15.5, 7.6), facecolor=SURF)
gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1], left=0.02, right=0.98,
                      top=0.86, bottom=0.13, wspace=0.12, hspace=0.32)

lat_c, lon_c = (b['lat0'] + b['lat1']) / 2, (b['lon0'] + b['lon1']) / 2
ax = fig.add_subplot(gs[:, 0])
mo = Basemap(projection='ortho', lat_0=lat_c, lon_0=lon_c, resolution='l', ax=ax)
mo.drawmapboundary(fill_color=SEA, color=MUTED, linewidth=0.8)
mo.fillcontinents(color=LAND, lake_color=SEA)
mo.drawcoastlines(color=MUTED, linewidth=0.4)
mo.drawparallels(np.arange(-80, 81, 20), color=MUTED, linewidth=0.3, dashes=[1, 2])
mo.drawmeridians(np.arange(0, 360, 30), color=MUTED, linewidth=0.3, dashes=[1, 2])
for q, bb in enumerate(blocks):
    xs, ys = mo([bb['lon0'], bb['lon1'], bb['lon1'], bb['lon0'], bb['lon0']],
                [bb['lat0'], bb['lat0'], bb['lat1'], bb['lat1'], bb['lat0']])
    ax.fill(xs, ys, color=INK2 if q != BLK else '#e34948', alpha=0.35 if q != BLK else 1,
            linewidth=0)
x, y = mo(lon_c, lat_c)
ax.plot(x, y, 'o', ms=16, mfc='none', mec=INK, mew=1.6)
ax.set_title(f'{len(blocks)} ASTER blocks of the orbit; this block in red',
             color=INK2, fontsize=10)

# Each sensor panel: the block in Mercator, the common cells only.
Z = np.full((len(names), TLA.size), np.nan); Z[:, good] = Ms
vmax = 2.5
POS = dict(zip(INST, [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]))   # fixed per sensor
for k in INST:
    a = fig.add_subplot(gs[POS[k]])
    if k not in names:   # too few source points in the block: an empty panel
        a.set_xticks([]); a.set_yticks([]); a.set_facecolor(SURF)
        a.text(0.5, 0.5, f'no {k} data\nin this block', ha='center', va='center',
               color=MUTED, transform=a.transAxes)
        a.set_title(k, color=MUTED, fontsize=10)
        continue
    q = names.index(k)
    mb = Basemap(projection='merc', llcrnrlat=b['lat0'], urcrnrlat=b['lat1'] + RES,
                 llcrnrlon=b['lon0'], urcrnrlon=b['lon1'] + RES, resolution='i', ax=a)
    X, Y = mb(TLO, TLA)
    pm = mb.pcolormesh(X, Y, Z[q].reshape(TLA.shape), cmap=DIV, vmin=-vmax, vmax=vmax,
                       shading='nearest', rasterized=True)
    mb.drawcoastlines(color=INK, linewidth=0.7)
    # Interior ticks only, so corner labels of the two axes cannot collide.
    mb.drawparallels(np.round(b['lat0'] + (b['lat1'] - b['lat0']) * np.array([0.2, 0.8]), 2), labels=[1, 0, 0, 0],
                     fmt='%.2f', color=MUTED, linewidth=0.3, dashes=[1, 2], fontsize=8)
    mb.drawmeridians(np.round(b['lon0'] + (b['lon1'] - b['lon0']) * np.array([0.2, 0.8]), 2), labels=[0, 0, 0, 1],
                     fmt='%.2f', color=MUTED, linewidth=0.3, dashes=[1, 2], fontsize=8)
    flip = '  (sign flipped)' if sign[q] < 0 else ''
    a.set_title(f'{k}   loading {load[k]:.2f}{"  - weakest" if k == weak else ""}{flip}',
                color=INK, fontsize=10, fontweight='bold' if k == weak else 'normal')

# Sixth slot: the loadings, one bar per sensor in fixed order.
a = fig.add_subplot(gs[1, 3]); a.set_facecolor(SURF)
vals = [load[k] for k in names]
a.barh(range(len(names)), vals, height=0.55, color='#2a78d6')
a.set_yticks(range(len(names)), names); a.invert_yaxis(); a.set_xlim(0, 1)
for i_, v_ in enumerate(vals):
    a.text(v_ + 0.02, i_, f'{v_:.2f}', va='center', color=INK2, fontsize=9)
for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
a.grid(axis='x', color='#e4e3de', linewidth=0.6); a.set_axisbelow(True)
a.set_title('loading on the common mode', color=INK2, fontsize=10)

cax = fig.add_axes([0.37, 0.055, 0.36, 0.022])
cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='both')
cb.set_label('rank z-score on the common cells, sign-aligned to the common mode', color=INK2)
cb.outline.set_edgecolor(MUTED)

orbit = os.path.basename(os.path.normpath(WORK))
ns = 'N' if lat_c >= 0 else 'S'; ew = 'E' if lon_c >= 0 else 'W'
fig.suptitle(f'{TITLE}{orbit} block {BLK}  ·  C5 = {C:.3f}  ·  '
             f'{abs(b["lat0"]):.2f}–{abs(b["lat1"]):.2f}°{ns}, '
             f'{abs(b["lon0"]):.2f}–{abs(b["lon1"]):.2f}°{ew}  ·  {int(good.sum())} common cells',
             x=0.02, ha='left', fontsize=14, color=INK, y=0.965)
rk = [r for r in json.load(open(f'{WORK}/../all_granule_ranking.json'))
      if r['orbit'] == os.path.basename(os.path.normpath(WORK)) and r['blk'] == BLK]
sd = f'band-31 BT {rk[0]["bt31_mean"]:.1f} K, sd {rk[0]["bt31_sd"]:.2f} K  ·  ' if rk else ''
fig.text(0.02, 0.905, f'{len(names)} sensors  ·  {sd}ASTER {b["granule"]}  ·  ASTER TIR band 10, MODIS band 31, MISR AN red, '
         f'CERES WN window, MOPITT channel 4  ·  0.02° grid', color=INK2, fontsize=9.5)
fig.savefig(OUT, dpi=150, facecolor=SURF)
print(f'# wrote {OUT}')
