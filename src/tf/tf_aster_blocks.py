import netCDF4, numpy as np, json, time
import os as _os
F=_os.environ.get('TF_GRANULE',
  '/mnt/common/datasets-staging/TERRA_BF_L1B_O10204_20011118010522_F000_V001.h5')
d=netCDF4.Dataset(F); t0=time.time()
rows=[]
for gn,g in sorted(d['ASTER'].groups.items()):
    if 'Geolocation' not in g.groups: continue
    G=g['Geolocation']
    la=np.asarray(G['Latitude'][:],'f8'); lo=np.asarray(G['Longitude'][:],'f8')
    m=np.isfinite(la)&np.isfinite(lo)&(np.abs(la)<=90)&(np.abs(lo)<=180)
    if not m.any(): continue
    la,lo=la[m],lo[m]
    # which data groups exist (VNIR/SWIR/TIR) and their native shapes
    have=[k for k in ('VNIR','SWIR','TIR') if k in g.groups]
    rows.append(dict(granule=gn, lat0=la.min(), lat1=la.max(), lon0=lo.min(), lon1=lo.max(),
                     clat=float(la.mean()), clon=float(lo.mean()), bands=have))
rows.sort(key=lambda r:-r['clat'])
print(f'# {len(rows)} ASTER granules, read {time.time()-t0:.1f}s')
print(f'{"#":>3} {"granule":26} {"lat0":>7} {"lat1":>7} {"lon0":>8} {"lon1":>8} {"dlat":>5} bands')
for i,r in enumerate(rows):
    print(f'{i:3d} {r["granule"]:26} {r["lat0"]:7.2f} {r["lat1"]:7.2f} {r["lon0"]:8.2f} {r["lon1"]:8.2f} {r["lat1"]-r["lat0"]:5.2f} {",".join(r["bands"])}')
json.dump(rows, open('aster_blocks.json','w'), indent=1, default=float)
