"""Validate fixed-policy training-pool panels; never touch independent evaluation data."""
import pathlib,json,itertools,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write,file_sha

OUT=ROOT/'artifacts/frozen';OUT.mkdir(parents=True,exist_ok=True)
G=ROOT/'artifacts/groups';maps={p.stem.removeprefix('mapping_'):np.load(p) for p in sorted(G.glob('mapping_*.npy'))}
E1=pathlib.Path('/e1reference/exports/cohort10397_20261103');OLD=pathlib.Path('/reference/analysis/metric-diagnostics-20261102')
sources=[];datasets=[];BOOT=2000;rng=np.random.default_rng(20261104)

selection=json.loads((E1/'selection.json').read_text())
# The diagnostic selection was sampled without replacement within three disjoint
# history strata. Reconstruct strata of selected IDs; the known population sizes
# permit Horvitz-Thompson pair weights. No historical label enters group construction.
ids=np.array(sorted(set(selection['sequence_ids'])),dtype=int)
stratum={int(i):g for i,g in zip(selection['sequence_ids'],selection['groups'])}
pop=selection['population_groups'];strata=[stratum[int(i)] for i in ids];Npop={k:int(v) for k,v in pop.items()}
ns={k:strata.count(k) for k in set(strata)}
assert len(ids)==1024 and sum(ns.values())==1024

def pair_set(mapping,selected,labels=None,population=None):
    inverse={int(v):j for j,v in enumerate(selected)};pairs=[];groups=[];weights=[]
    for g in range(4096):
        seen=[inverse[int(i)] for i in selected[mapping[selected]==g]]
        for a,b in itertools.combinations(seen,2):
            pairs.append((a,b));groups.append(g)
            if labels is not None:
                x,y=labels[a],labels[b];na,nb=ns[x],ns[y];Na,Nb=population[x],population[y]
                inclusion=na*(na-1)/(Na*(Na-1)) if x==y else na/Na*nb/Nb
                weights.append(1/inclusion)
            else:weights.append(1.)
    return np.array(pairs,dtype=int).reshape(-1,2),np.array(groups,dtype=int),np.array(weights)

def from_rows(rows,selected,key='values'):
    by={int(i):[] for i in selected}
    for r in rows:by[int(r['sequence_id'])].append(r)
    assert all(len(v)==4 for v in by.values())
    values=np.array([[r[key] for r in sorted(by[int(i)],key=lambda x:x['replicate'])] for i in selected],dtype=float)
    valid=(values[:,:,0]>=100)&(values[:,:,7]==0);J=values[:,:,6].mean(1)
    def avg(a):return np.divide(np.where(valid,a,0).sum(1),valid.sum(1),out=np.full(len(selected),np.nan),where=valid.sum(1)>0)
    fraction=np.divide(values[:,:,1],values[:,:,0],out=np.zeros_like(values[:,:,0]),where=values[:,:,0]>0)
    return dict(success=J,position=avg(values[:,:,4]),joint_fraction=avg(fraction),failure=values[:,:,7].mean(1)),values

def evaluate(name,cp,mode,selected,metrics,labels=None,population=None):
    # Shared node-Poisson bootstrap across mappings respects overlapping sequence
    # pairs; it does not pretend the different mapping IDs form paired clusters.
    node_weight=rng.poisson(1,(BOOT,len(selected))).astype(np.float32)
    rows=[];boots={};coverage={};details=[]
    for m,mapping in maps.items():
        pairs,groups,weights=pair_set(mapping,selected,labels,population)
        coverage[m]=dict(observed_sequences=len(selected),multi_member_groups=len(set(groups)),within_group_pairs=len(pairs),fully_observed_groups=int(sum(np.isin(np.flatnonzero(mapping==g),selected).all() for g in set(groups))))
        for metric,v in metrics.items():
            a,b=pairs.T;ok=np.isfinite(v[a])&np.isfinite(v[b]);aa,bb=a[ok],b[ok];w=weights[ok];d=np.abs(v[aa]-v[bb])
            if not len(d):continue
            scale=1000 if metric=='position' else 100
            mean=float(np.average(d,weights=w))*scale
            wb=node_weight[:,aa]*node_weight[:,bb]*w[None,:];den=wb.sum(1)
            boot=np.divide(wb@d,den,out=np.full(BOOT,np.nan),where=den>0)*scale
            boots[(m,metric)]=boot
            row=dict(policy=name,checkpoint=cp,action_mode=mode,mapping=m,metric=metric,units='mm' if metric=='position' else 'percentage_points',mean_within_group_abs_difference=mean,
                unweighted_panel_difference=float(d.mean())*scale,ci95_low=float(np.nanquantile(boot,.025)),ci95_high=float(np.nanquantile(boot,.975)),
                observed_pairs=len(d),effective_IPW_pairs=float(w.sum()**2/np.dot(w,w)),multi_member_groups=len(set(groups[ok])),
                correction='Known stratum pair-inclusion IPW' if labels is not None else 'Panel-conditional, selection probabilities not reconstructed')
            rows.append(row)
            for g in sorted(set(groups[ok])):
                gg=groups[ok]==g;details.append(dict(policy=name,checkpoint=cp,action_mode=mode,mapping=m,group_id=int(g),metric=metric,observed_pair_count=int(gg.sum()),mean_difference=float(np.average(d[gg],weights=w[gg]))*scale))
    comparisons=[]
    for m in maps:
        if m=='command':continue
        for metric in metrics:
            if ('command',metric) not in boots or (m,metric) not in boots:continue
            d=boots[('command',metric)]-boots[(m,metric)]
            point=next(x['mean_within_group_abs_difference'] for x in rows if x['mapping']=='command' and x['metric']==metric)-next(x['mean_within_group_abs_difference'] for x in rows if x['mapping']==m and x['metric']==metric)
            comparisons.append(dict(policy=name,checkpoint=cp,action_mode=mode,random_mapping=m,metric=metric,command_minus_random=point,ci95_low=float(np.nanquantile(d,.025)),ci95_high=float(np.nanquantile(d,.975)),command_better_CI95=bool(np.nanquantile(d,.975)<0)))
    return rows,comparisons,coverage,details

allrows=[];comparisons=[];details=[];coverage={};vectors={}
for cp in (4000,8000,10000):
    for mode in ('mean','stochastic'):
        folder=E1/(str(cp)+'-'+mode);meta=json.loads((folder/'result.json').read_text());path=folder/'episode_results.json'
        assert meta['passed'] and meta['training_history_updates']==0 and meta['model_weights_unchanged']
        rows=json.loads(path.read_text());metrics,values=from_rows(rows,ids)
        vectors[(cp,mode)]=metrics
        rs,cs,cover,ds=evaluate('WBC-SGS-E1',cp,mode,ids,metrics,strata,Npop)
        allrows+=rs;comparisons+=cs;details+=ds;coverage['E1_'+str(cp)+'_'+mode]=cover
        sources.extend([dict(path=str(f),sha256=file_sha(f)) for f in (path,folder/'result.json')])
        print('FROZEN_E1 '+str(cp)+' '+mode+' '+json.dumps(cover),flush=True)
for mode in ('mean','stochastic'):
    for old,new in ((4000,8000),(8000,10000)):
        metrics={k:vectors[(new,mode)][k]-vectors[(old,mode)][k] for k in ('success','position','joint_fraction')}
        rs,cs,cover,ds=evaluate('WBC-SGS-E1-progress',str(old)+'-'+str(new),mode,ids,metrics,strata,Npop)
        allrows+=rs;comparisons+=cs;details+=ds
for cp in (8000,10000):
    for mode in ('mean','stochastic'):
        folder=OLD/(str(cp)+'-'+mode);path=folder/'episode_diagnostics.json'
        data=json.loads(path.read_text());rows=data['rows'];selected=np.array(sorted({x['sequence_id'] for x in rows}),dtype=int)
        metrics,_=from_rows(rows,selected,'formal_values');rs,cs,cover,ds=evaluate('WBC-SGS-original',cp,mode,selected,metrics)
        allrows+=rs;comparisons+=cs;details+=ds;coverage['original_'+str(cp)+'_'+mode]=cover
        sources.append(dict(path=str(path),sha256=file_sha(path)))
        print('FROZEN_ORIGINAL '+str(cp)+' '+mode+' '+json.dumps(cover),flush=True)
for filename,rows in [('within_group_control_metrics.csv',allrows),('command_vs_random_frozen.csv',comparisons),('observed_group_details.csv',details)]:
    with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
primary=[x for x in comparisons if x['policy']=='WBC-SGS-E1' and x['checkpoint'] in (8000,10000)]
coverage_ok=all(c['multi_member_groups']>=64 for key,cc in coverage.items() if key.startswith(('E1_8000','E1_10000')) for c in cc.values())
continuous_ok=all(x['command_better_CI95'] for x in primary if x['metric'] in ('position','joint_fraction'))
success_degradation=any(x['ci95_low']>0 for x in primary if x['metric']=='success')
frozen_pass=coverage_ok and continuous_ok and not success_degradation
summary=dict(passed=True,frozen_GO_condition_passed=frozen_pass,coverage_condition=coverage_ok,continuous_consistency_condition=continuous_ok,
    significant_strict_heterogeneity_degradation=success_degradation,primary_comparisons=primary,coverage=coverage,rows=allrows,comparisons=comparisons,sources=sources,
    limitations=['Existing panels are selected by historical feedback strata, not cluster-complete uniform task samples.',
    'Primary E1 panel uses known without-replacement stratum pair-inclusion weights. Pair ESS may be far lower than raw pair count. Original SGS panel is reported conditionally only.',
    'Only4 reset trials per sequence; binary success rates have25-percentage-point resolution. Within-group dispersion includes measurement noise.',
    'Bootstrap is shared sequence-node Poisson; intervals condition on one trained seed and the existing diagnostic panels, and do not establish new PPO benefit.',
    'Position/time continuous metrics condition on mature nonfailure episodes; native failures and all-trial strict success are separate.',
    'No independent1024 evaluation sequence is part of clustering or these training-pool panels.'])
write(OUT/'frozen_summary.json',summary)
fig,axes=plt.subplots(2,2,figsize=(12,7))
for row,mode in enumerate(('mean','stochastic')):
    for col,metric in enumerate(('position','joint_fraction')):
        ax=axes[row,col]
        for m in maps:
            rs=[x for x in allrows if x['policy']=='WBC-SGS-E1' and x['action_mode']==mode and x['metric']==metric]
            rs=sorted([x for x in rs if x['mapping']==m],key=lambda x:x['checkpoint'])
            ax.plot([x['checkpoint'] for x in rs],[x['mean_within_group_abs_difference'] for x in rs],'-o',label=m,ms=4)
        ax.set_title(mode+' | '+metric+' within-group dispersion');ax.set_xlabel('Frozen checkpoint');ax.set_ylabel('mm' if metric=='position' else 'Percentage points');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Fixed-policy consistency, training-panel IPW; lower is more homogeneous');fig.tight_layout();fig.savefig(OUT/'frozen_consistency.png',dpi=180);plt.close(fig)
print('FROZEN_ANALYSIS_PASS condition='+str(frozen_pass),flush=True)
