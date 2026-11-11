"""Quantify how old unscorable/failure episodes persist; labels remain unchanged."""
import json,csv
import numpy as np
from common import ROOT,write
maps={p.stem.removeprefix('mapping_'):np.load(p) for p in sorted((ROOT/'artifacts/groups').glob('mapping_*.npy'))};rows=[]
for run in ('uniform','sgs','sgs-e1'):
    d=np.load(ROOT/'data_cache'/('feedback_'+run+'.npy'),mmap_mode='r')
    for name,m in {'sequence':np.arange(32768),**maps}.items():
        g=m[d['sequence']];order=np.argsort(g,kind='stable');sg=g[order];starts=np.searchsorted(sg,np.arange(int(m.max())+2))
        selected=np.concatenate([order[starts[k]:starts[k+1]][-100:] for k in range(len(starts)-1)])
        w=d[selected];rows.append(dict(run=run,mapping=name,checkpoint=15000,retained_real_episodes=len(w),insufficient_window_fraction=float((w['scored']<100).mean()),native_failure_fraction=float(w['failure'].mean()),
            mature_nonfailure_fraction=float(((w['scored']>=100)&(w['failure']==0)).mean()),feedback_end_before4000_fraction=float((w['tick']<=4000*24).mean()),strict_success_fraction=float(w['success'].mean())))
with (ROOT/'artifacts/feedback/window_eligibility.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(ROOT/'artifacts/feedback/window_eligibility_summary.json',dict(rows=rows,note='Diagnostic only; unsuccessful/unscorable episodes are not removed from any sampler window. All strict labels stay unchanged.'))
print('WINDOW_ELIGIBILITY_PASS '+json.dumps(rows),flush=True)
