"""Separate recency from within-window member weighting; inspect observed group spans."""
import pathlib,json,csv
import numpy as np
from common import ROOT,write,file_sha

OUT=ROOT/'artifacts';G=OUT/'groups';maps={p.stem.removeprefix('mapping_'):np.load(p) for p in sorted(G.glob('mapping_*.npy'))}
bias=[]
for run in ('uniform','sgs','sgs-e1'):
    d=np.load(ROOT/'data_cache'/('feedback_'+run+'.npy'),mmap_mode='r')
    for name,m in maps.items():
        group=m[d['sequence']];order=np.argsort(group,kind='stable');sorted_g=group[order];starts=np.searchsorted(sorted_g,np.arange(4097));gaps=[];full_gaps=[];missing=0
        for k in range(4096):
            w=d[order[starts[k]:starts[k+1]][-100:]];u,inv,n=np.unique(w['sequence'],return_inverse=True,return_counts=True)
            s=np.bincount(inv,weights=w['success'],minlength=len(u));member_mean=(s/n).mean();pooled=w['success'].mean();gaps.append(float(pooled-member_mean))
            if len(u)==8:full_gaps.append(float(pooled-member_mean))
            else:missing+=1
        bias.append(dict(run=run,mapping=name,groups=4096,missing_member_groups=missing,all8_observed_groups=4096-missing,
            mean_abs_same_window_member_weighting_gap_observed_members=float(np.abs(gaps).mean()),
            mean_abs_same_window_member_weighting_gap_all8_groups=float(np.abs(full_gaps).mean()),
            note='Same last100 episodes/time support; compares episode-pooled rate to equal observed-member rates. Contains finite-sample noise; absent members are not imputed.'))
with (OUT/'feedback/same_window_member_mixture.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(bias[0]));w.writeheader();w.writerows(bias)
summary=json.loads((OUT/'feedback/feedback_summary.json').read_text())
for row in summary['window_statistics']:
    if 'mean_abs_exposure_mixture_gap' in row:row['mean_abs_recent_group_vs_lifetime_members_gap']=row.pop('mean_abs_exposure_mixture_gap')
summary['same_window_member_mixture']=bias
summary['recency_bias_note']='The recent-group versus lifetime-member gap mixes temporal policy drift and exposure; it is NOT a pure statistical bias estimate. same_window_member_mixture controls the time support but still includes finite-sample noise.'
write(OUT/'feedback/feedback_summary.json',summary)
with (OUT/'feedback/window_density_and_age.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary['window_statistics'][0]));w.writeheader();w.writerows(summary['window_statistics'])

E1=pathlib.Path('/e1reference/exports/cohort10397_20261103');spans=[]
for mode in ('mean','stochastic'):
    rows=json.loads((E1/('10000-'+mode)/'episode_results.json').read_text());metrics={}
    for seq in sorted({r['sequence_id'] for r in rows}):
        a=np.array([r['values'] for r in rows if r['sequence_id']==seq]);valid=(a[:,0]>=100)&(a[:,7]==0)
        metrics[seq]=dict(p=float(a[:,6].mean()),position=float(a[valid,4].mean()) if valid.any() else np.nan,joint=float((a[valid,1]/a[valid,0]).mean()) if valid.any() else np.nan)
    for g in sorted({int(maps['command'][i]) for i in metrics}):
        ids=[i for i in metrics if maps['command'][i]==g]
        if len(ids)<2:continue
        def span(key):
            v=[metrics[i][key] for i in ids if np.isfinite(metrics[i][key])]
            return float(max(v)-min(v)) if len(v)>1 else None
        spans.append(dict(checkpoint=10000,mode=mode,group_id=g,observed_members=len(ids),sequence_ids=';'.join(map(str,ids)),strict_success_rate_span=span('p'),position_RMSE_span_m=span('position'),joint_good_time_span=span('joint'),scope='Observed subset only;4 resets per sequence; no full8-member frozen groups.'))
with (OUT/'frozen/observed_command_group_spans.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(spans[0]));w.writeheader();w.writerows(spans)
span_summary={}
for mode in ('mean','stochastic'):
    rs=[s for s in spans if s['mode']==mode]
    span_summary[mode]=dict(partial_groups=len(rs),observed_success_span_at_least50pp=sum(s['strict_success_rate_span']>=.5 for s in rs),observed_position_span_at_least10mm=sum(s['position_RMSE_span_m'] is not None and s['position_RMSE_span_m']>=.01 for s in rs),median_observed_position_span_mm=float(np.median([s['position_RMSE_span_m']*1000 for s in rs if s['position_RMSE_span_m'] is not None])))
write(OUT/'frozen/observed_group_span_summary.json',span_summary)
print('BIAS_AND_SPANS_PASS '+json.dumps(dict(mixture=bias,spans=span_summary)),flush=True)
