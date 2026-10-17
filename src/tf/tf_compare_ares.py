"""Compare an Aurora congruence run with the ares results, block by block.

The Aurora run reads the same granules (fetched from s3://terrafusiondatasampler
instead of ares' local copies) through the same pipeline, so every block's C5,
Kendall's W, cell count and weakest sensor should agree; a regrid tie resolved
differently on the GPU is the only expected source of a difference.

Usage:
    ROOT=<aurora work root> ARES=~/ares/data python3 tf_compare_ares.py
    -> prints a per-orbit table; writes compare_ares.json under ROOT
"""
import glob, json, os
import numpy as np

ROOT = os.environ['ROOT']
ARES = os.path.expanduser(os.environ.get('ARES', '~/ares/data'))
TAG = os.environ.get('TF_TAG', 'b31_WN')

ares_rank = {}
p = os.path.join(ARES, 'all_granule_ranking.json')
if os.path.exists(p):
    for r in json.load(open(p)):
        ares_rank[(r['orbit'], r['blk'])] = r

report = {}
print(f'{"orbit":8} {"ares":>5} {"aurora":>6} {"matched":>7} {"max|dC5|":>9} '
      f'{"max|dW|":>9} {"ncell=":>6} {"weak=":>5} {"max|dSD|K":>9}')
for apath in sorted(glob.glob(os.path.join(ARES, 'congruence_O*.json'))):
    orbit = os.path.basename(apath)[len('congruence_'):-len('.json')]
    cpath = os.path.join(ROOT, orbit, f'congruence_{TAG}.json')
    a = {r['blk']: r for r in json.load(open(apath))}
    if not os.path.exists(cpath):
        print(f'{orbit:8} {len(a):5d}  (no Aurora result)')
        report[orbit] = dict(ares=len(a), aurora=0)
        continue
    b = {r['blk']: r for r in json.load(open(cpath))}
    kpath = os.path.join(ROOT, orbit, f'contrast_{TAG}.json')
    k = json.load(open(kpath)) if os.path.exists(kpath) else {}
    common = sorted(set(a) & set(b))
    dc = [abs(a[i]['congruence'] - b[i]['congruence']) for i in common]
    dw = [abs(a[i]['kendall_w'] - b[i]['kendall_w']) for i in common]
    ncell = sum(a[i]['ncell'] == b[i]['ncell'] for i in common)
    weak = sum(a[i]['weakest'] == b[i]['weakest'] for i in common)
    dsd = [abs(ares_rank[(orbit, i)]['bt31_sd'] - k[str(i)]['bt31_sd'])
           for i in common
           if (orbit, i) in ares_rank and 'bt31_sd' in ares_rank[(orbit, i)]
           and str(i) in k]
    rep = dict(ares=len(a), aurora=len(b), matched=len(common),
               max_dC5=max(dc) if dc else None, max_dW=max(dw) if dw else None,
               ncell_equal=ncell, weakest_equal=weak,
               max_dbt31_sd=max(dsd) if dsd else None,
               only_ares=sorted(set(a) - set(b)), only_aurora=sorted(set(b) - set(a)))
    report[orbit] = rep
    f = lambda v: '      n/a' if v is None else f'{v:9.2e}'
    print(f'{orbit:8} {len(a):5d} {len(b):6d} {len(common):7d} {f(rep["max_dC5"])} '
          f'{f(rep["max_dW"])} {ncell:6d} {weak:5d} {f(rep["max_dbt31_sd"])}')
json.dump(report, open(os.path.join(ROOT, 'compare_ares.json'), 'w'), indent=1)
print(f'# wrote {os.path.join(ROOT, "compare_ares.json")}')
