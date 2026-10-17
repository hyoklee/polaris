"""Rank every ASTER block across every Terra Fusion granule by discrepancy.

Parts 1-15 ranked 32 blocks within one orbit. This pools all granules that
carry ASTER plus the other four sensors and answers the question across the
whole collection: which block, anywhere, is the least congruent.

Discrepancy is C5 = lambda1/n from part 4, computed on the MATCHED
configuration of parts 11-12 (MODIS band 31, CERES WN_Radiance). Lower C5 means
more discrepancy. Kendall's W is carried as the independent cross-check.

A caution the ranking cannot express on its own: C5 is also a scene-contrast
measure (part 13), so a low value flags a featureless scene as readily as a
disturbed one. The per-block cell count and loading spread are printed so a
block at the top can be read rather than just cited.

Usage:
    python3 bin/tf_rank_all.py                 # -> all_granule_ranking.json
    TOP=40 python3 bin/tf_rank_all.py

Aurora: when a granule has contrast_<TAG>.json (tf_contrast.py), its blocks
carry bt31_mean/bt31_sd, and the blocks at or above FLOOR kelvin of band-31 sd
(default 2.0, the ares part-17 floor) are ranked again into
all_granule_ranking_filtered.json.
"""
import json, os, glob, sys
import numpy as np

ROOT = os.environ.get('ROOT', '/mnt/common/hyoklee/tfwork/allgran')
TOP = int(os.environ.get('TOP', '25'))
TAG = os.environ.get('TF_TAG', 'b31_WN')
FLOOR = float(os.environ.get('FLOOR', '2.0'))

rows = []
per_granule = {}
for d in sorted(glob.glob(os.path.join(ROOT, 'O*'))):
    orbit = os.path.basename(d)
    cpath = os.path.join(d, f'congruence_{TAG}.json')
    bpath = os.path.join(d, 'aster_blocks.json')
    if not (os.path.exists(cpath) and os.path.exists(bpath)):
        print(f'# {orbit}: no results, skipped')
        continue
    cg = json.load(open(cpath))
    blocks = json.load(open(bpath))
    kpath = os.path.join(d, f'contrast_{TAG}.json')
    contrast = json.load(open(kpath)) if os.path.exists(kpath) else {}
    for r in cg:
        b = blocks[r['blk']]
        ld = r.get('loadings', {})
        rows.append(dict(
            orbit=orbit, blk=r['blk'], granule=b.get('granule', ''),
            lat0=b['lat0'], lat1=b['lat1'], lon0=b['lon0'], lon1=b['lon1'],
            C5=r['congruence'], W=r.get('kendall_w'),
            Cd=r.get('congruence_dense'), ncell=r.get('ncell'),
            weakest=r.get('weakest'), gap=r.get('load_gap'),
            loadings=ld,
            **{k: contrast[str(r['blk'])][k] for k in ('bt31_sd', 'bt31_mean')
               if str(r['blk']) in contrast}))
    per_granule[orbit] = [r['congruence'] for r in cg]

if not rows:
    print('no results found under', ROOT); sys.exit(1)

rows.sort(key=lambda r: r['C5'])
C = np.array([r['C5'] for r in rows])
W = np.array([r['W'] for r in rows if r['W'] is not None])

print(f'# {len(rows)} blocks from {len(per_granule)} granules'
      f'   C5 min={C.min():.3f} median={np.median(C):.3f} max={C.max():.3f}')
if len(W) == len(C):
    print(f'# C5 vs Kendall W across all blocks: r={np.corrcoef(C, W)[0,1]:.4f}')

print(f'\n# per granule')
print(f'{"orbit":9} {"blocks":>6} {"C5 min":>7} {"median":>7} {"max":>7}')
for o in sorted(per_granule):
    v = np.array(per_granule[o])
    print(f'{o:9} {len(v):6d} {v.min():7.3f} {np.median(v):7.3f} {v.max():7.3f}')

print(f'\n# TOP {TOP} DISCREPANCY BLOCKS, all granules pooled (lowest C5 first)')
print(f'{"rk":>3} {"orbit":9} {"blk":>4} {"lat0":>7} {"lat1":>7} {"lon0":>8} '
      f'{"C5":>6} {"W":>6} {"cells":>6} {"weak":>7} {"gap":>5}  loadings')
for i, r in enumerate(rows[:TOP]):
    ld = ' '.join(f'{k[:4]}={v:.2f}'
                  for k, v in sorted(r['loadings'].items(), key=lambda y: -y[1]))
    w = '   n/a' if r['W'] is None else f'{r["W"]:6.3f}'
    print(f'{i:3d} {r["orbit"]:9} {r["blk"]:4d} {r["lat0"]:7.2f} {r["lat1"]:7.2f} '
          f'{r["lon0"]:8.2f} {r["C5"]:6.3f} {w} '
          f'{r["ncell"]:6d} {str(r["weakest"])[:7]:>7} {r["gap"]:5.2f}  {ld}')

print(f'\n# MOST congruent block overall')
r = rows[-1]
print(f'  {r["orbit"]} blk {r["blk"]}  {r["lat0"]:.2f}-{r["lat1"]:.2f}N  C5={r["C5"]:.3f}')

from collections import Counter
print(f'\n# weakest-sensor tally across all {len(rows)} blocks: '
      f'{dict(Counter(r["weakest"] for r in rows))}')

json.dump(rows, open('all_granule_ranking.json', 'w'), indent=1)
print(f'\n# wrote all_granule_ranking.json ({len(rows)} blocks)')

kept = [r for r in rows if r.get('bt31_sd', -1) >= FLOOR]
if kept:
    print(f'\n# TOP {TOP} after the {FLOOR} K band-31 contrast floor '
          f'({len(kept)} of {len(rows)} blocks kept)')
    for i, r in enumerate(kept[:TOP]):
        print(f'{i:3d} {r["orbit"]:9} {r["blk"]:4d} {r["lat0"]:7.2f} '
              f'{r["lat1"]:7.2f} {r["lon0"]:8.2f} {r["C5"]:6.3f} '
              f'sd={r["bt31_sd"]:5.2f}K BT={r["bt31_mean"]:6.1f}K '
              f'{str(r["weakest"])[:7]:>7} {r["gap"]:5.2f}')
    json.dump(kept, open('all_granule_ranking_filtered.json', 'w'), indent=1)
    print(f'# wrote all_granule_ranking_filtered.json ({len(kept)} blocks)')
