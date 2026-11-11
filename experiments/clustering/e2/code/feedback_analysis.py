"""Chronological estimator replay; logged events are fixed, no counterfactual training."""
import pathlib,json,csv,time,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write,file_sha

OUT=ROOT/'artifacts/feedback';OUT.mkdir(parents=True,exist_ok=True);CACHE=ROOT/'data_cache';CACHE.mkdir(exist_ok=True)
N=32768;H=100
maps={p.stem.removeprefix('mapping_'):np.load(p) for p in sorted((ROOT/'artifacts/groups').glob('mapping_*.npy'))}
dtype=np.dtype([('sequence','i4'),('tick','i4'),('physical_steps','i2'),('scored','i2'),('good','i2'),('position','f4'),('success','u1'),('failure','u1')])

def extract(kind):
    paths=[pathlib.Path('/e1reference' if kind=='sgs-e1' else '/reference')/'runs'/('formal-wbc-'+kind)/'terminal_feedback.jsonl',
           pathlib.Path('/continue_reference/runs')/('continue-wbc-'+kind)/'terminal_feedback.jsonl']
    chunks=[];previous=0
    for path in paths:
        for line in path.open():
            r=json.loads(line);tick=r['control_step'];assert previous<=tick;previous=tick
            assert r['pool_identity'].startswith('wbc:train:')
            v=np.asarray(r['rows']);ids=np.asarray(r['task_ids']);a=np.empty(len(v),dtype=dtype)
            a['sequence']=ids;a['tick']=tick;a['physical_steps']=r['physical_steps'];a['scored']=v[:,0];a['good']=v[:,1];a['position']=v[:,4];a['success']=v[:,6];a['failure']=v[:,7]
            chunks.append(a)
        print('EXTRACTED '+kind+' '+str(path),flush=True)
    data=np.concatenate(chunks);assert np.all(data['sequence']>=0) and np.all(data['sequence']<N)
    expected=json.loads((pathlib.Path('/continue_reference/runs')/('continue-wbc-'+kind)/'result.json').read_text())['summary']['feedback_accepted']
    assert len(data)==expected,(kind,len(data),expected)
    np.save(CACHE/('feedback_'+kind+'.npy'),data)
    return data,[dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)) for p in paths]

def window_statistics(data,mapping,cp,name,run):
    count=np.searchsorted(data['tick'],cp*24,side='right');d=data[:count];g=mapping[d['sequence']]
    order=np.argsort(g,kind='stable');sorted_g=g[order];starts=np.searchsorted(sorted_g,np.arange(int(mapping.max())+2))
    ages=[];sizes=[];diversity=[];largest=[];p=[];bias=[];contributions=[]
    seq_n=np.bincount(d['sequence'],minlength=N);seq_s=np.bincount(d['sequence'],weights=d['success'],minlength=N)
    seq_rate=np.divide(seq_s,seq_n,out=np.full(N,.5),where=seq_n>0)
    for k in range(len(starts)-1):
        indices=order[starts[k]:starts[k+1]][-H:];w=d[indices];size=len(w);sizes.append(size)
        if size:
            ages.extend(((cp*24-w['tick'])/24).tolist());u,c=np.unique(w['sequence'],return_counts=True)
            diversity.append(len(u));largest.append(float(c.max()/size));p.append(float(w['success'].mean()))
            # Legacy-history bias diagnostic only: exposure-weighted group mean vs
            # equal-member historical means, all from nonstationary logged policies.
            members=np.flatnonzero(mapping==k);bias.append(float(w['success'].mean()-seq_rate[members].mean()))
            contributions.append(float(1/np.sum((c/size)**2)))
        else:diversity.append(0);largest.append(0);p.append(.5)
    values=np.array(ages);sizes=np.array(sizes);p=np.array(p)
    return dict(run=run,mapping=name,checkpoint=cp,groups=len(sizes),feedback_events=len(d),mean_window_samples=float(sizes.mean()),median_window_samples=float(np.median(sizes)),empty_groups=int((sizes==0).sum()),full_windows=int((sizes==100).sum()),
        mean_feedback_age_updates=float(values.mean()) if len(values) else None,median_feedback_age_updates=float(np.median(values)) if len(values) else None,p95_feedback_age_updates=float(np.quantile(values,.95)) if len(values) else None,
        median_members_per_window=float(np.median(diversity)),median_largest_member_share=float(np.median(largest)),mean_effective_members=float(np.mean(contributions)) if contributions else None,
        mean_abs_recent_group_vs_lifetime_members_gap=float(np.abs(bias).mean()) if bias else None,zero_window_fraction=float((p==0).mean()),near50_window_fraction=float(((p>=.4)&(p<=.6)).mean()))

class Ring:
    def __init__(self,mapping):
        self.mapping=mapping;g=int(mapping.max())+1
        self.y=np.zeros((g,H),dtype=np.uint8);self.member=np.zeros((g,H),dtype=np.int32);self.head=np.zeros(g,dtype=np.int64);self.length=np.zeros(g,dtype=np.int64);self.total=np.zeros(g,dtype=np.int64)
        self.member_n=np.zeros(N,dtype=np.int64);self.member_s=np.zeros(N,dtype=np.int64)
    def predictions(self,i):
        g=int(self.mapping[i]);n=int(self.length[g]);s=int(self.total[g]);p=s/n if n else .5
        other_n=n-int(self.member_n[i]);other_s=s-int(self.member_s[i]);loo=other_s/other_n if other_n else .5
        return p,loo
    def update(self,i,y):
        g=int(self.mapping[i]);h=int(self.head[g]);n=int(self.length[g])
        if n==H:
            old=int(self.y[g,h]);j=int(self.member[g,h]);self.total[g]-=old;self.member_n[j]-=1;self.member_s[j]-=old
        else:self.length[g]+=1
        self.y[g,h]=y;self.member[g,h]=i;self.total[g]+=y;self.member_n[i]+=1;self.member_s[i]+=y;self.head[g]=(h+1)%H

def prequential(data):
    names=['sequence']+list(maps);rings=[Ring(np.arange(N))]+[Ring(m) for m in maps.values()]
    loss=np.zeros((15001,len(names)),dtype=np.float64);loo_loss=np.zeros_like(loss);counts=np.zeros(15001,dtype=np.int64)
    for j,(i,tick,y) in enumerate(zip(data['sequence'],data['tick'],data['success'])):
        i=int(i);y=int(y);u=(int(tick)+23)//24;counts[u]+=1
        for k,r in enumerate(rings):
            p,loo=r.predictions(i);loss[u,k]+=(p-y)**2;loo_loss[u,k]+=(loo-y)**2;r.update(i,y)
        if j%300000==0:print('PREQUENTIAL '+str(j)+'/'+str(len(data)),flush=True)
    rows=[];comparisons=[];rng=np.random.default_rng(20261104)
    for first,last in ((10001,12500),(12501,15000)):
        for k,name in enumerate(names):rows.append(dict(first_update=first,last_update=last,estimator=name,episodes=int(counts[first:last+1].sum()),Brier=float(loss[first:last+1,k].sum()/counts[first:last+1].sum()),leave_member_out_Brier=float(loo_loss[first:last+1,k].sum()/counts[first:last+1].sum())))
        block_loss=loss[first:last+1].reshape(25,100,len(names)).sum(1);block_count=counts[first:last+1].reshape(25,100).sum(1)
        idx=rng.integers(0,25,(4000,25));command=names.index('command')
        for name in names:
            if name=='command':continue
            k=names.index(name);change=block_loss[:,command]-block_loss[:,k]
            boots=change[idx].sum(1)/block_count[idx].sum(1)
            comparisons.append(dict(first_update=first,last_update=last,comparison='command-minus-'+name,mean_Brier_change=float(change.sum()/block_count.sum()),ci95_low=float(np.quantile(boots,.025)),ci95_high=float(np.quantile(boots,.975))))
    return rows,comparisons

all_stats=[];all_sources=[];prediction_rows=[];prediction_compare=[];hist_control=[]
for kind in ('uniform','sgs','sgs-e1'):
    data,sources=extract(kind);all_sources+=sources
    for cp in (4000,8000,10000,12500,15000):
        for name,mapping in {'sequence':np.arange(N),**maps}.items():all_stats.append(window_statistics(data,mapping,cp,name,kind))
    if kind=='uniform':
        prediction_rows,prediction_compare=prequential(data)
        late=[]
        for first,last in ((10001,12500),(12501,15000)):
            d=data[(data['tick']>(first-1)*24)&(data['tick']<=last*24)];n=np.bincount(d['sequence'],minlength=N);s=np.bincount(d['sequence'],weights=d['success'],minlength=N)
            valid=(d['scored']>=100)&(d['failure']==0);vn=np.bincount(d['sequence'][valid],minlength=N)
            ps=np.bincount(d['sequence'][valid],weights=d['position'][valid],minlength=N);fs=np.bincount(d['sequence'][valid],weights=d['good'][valid]/d['scored'][valid],minlength=N)
            rates=np.divide(s,n,out=np.full(N,np.nan),where=n>0);position=np.divide(ps,vn,out=np.full(N,np.nan),where=vn>0);fraction=np.divide(fs,vn,out=np.full(N,np.nan),where=vn>0)
            late.append(dict(success=rates,position=position,joint_fraction=fraction))
            for name,m in maps.items():
                for metric,v in late[-1].items():
                    means=[]
                    for k in range(4096):
                        z=v[np.flatnonzero(m==k)];z=z[np.isfinite(z)]
                        if len(z)>=2:means.append(float(np.var(z)))
                    hist_control.append(dict(run='uniform logged policies; NOT frozen',first_update=first,last_update=last,mapping=name,metric=metric,mean_within_group_variance=float(np.mean(means))))
        # Store per-sequence late proxies for review; they never enter the grouping script.
        with (OUT/'uniform_late_sequence_proxies.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['sequence_id','p_10001_12500','p_12501_15000','position_10001_12500','position_12501_15000','joint_fraction_10001_12500','joint_fraction_12501_15000'])
            for i in range(N):w.writerow([i,late[0]['success'][i],late[1]['success'][i],late[0]['position'][i],late[1]['position'][i],late[0]['joint_fraction'][i],late[1]['joint_fraction'][i]])
    print('FEEDBACK_RUN_PASS '+kind,flush=True);del data
for filename,rows in [('window_density_and_age.csv',all_stats),('prequential_Brier.csv',prediction_rows),('prequential_comparisons.csv',prediction_compare),('historical_within_group_proxies.csv',hist_control)]:
    with (OUT/filename).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
random_better=all(x['ci95_high']<0 for x in prediction_compare if 'random_' in x['comparison'])
sequence_not_degraded=all(x['mean_Brier_change']<=0 for x in prediction_compare if x['comparison']=='command-minus-sequence')
summary=dict(passed=True,estimator_GO_condition_passed=random_better and sequence_not_degraded,command_better_than_all_random_in_both_blocks_CI95=random_better,command_not_worse_than_sequence_in_both_blocks=sequence_not_degraded,
    prediction_rows=prediction_rows,prediction_comparisons=prediction_compare,window_statistics=all_stats,historical_control_proxies=hist_control,sources=all_sources,
    limitations=['All feedback was generated by changing policies; none of these history rates is a fixed-policy difficulty label.',
    'Prequential prediction uses the identical chronological Uniform event stream, before each outcome is added. It is an estimator diagnostic, not simulated grouped-sampler training.',
    'SGS/E1 legacy streams are nonuniformly sampled; window contribution imbalance and gap to equal-member historical means are exposure-mixture diagnostics only.',
    'Group100 is100 real ended episodes across8 sequences, not800 trials. More events per group reduce age but can increase smoothing bias across differing members.',
    'Feedback age is measured since terminal measurement in PPO-update units; intra-episode latency and control-state dependence remain.',
    'Brier intervals use100-update blocks and condition on one policy trajectory; no across-training-seed causal claim.'])
write(OUT/'feedback_summary.json',summary)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for name in ('sequence','random_20261104','command'):
    rs=[x for x in all_stats if x['run']=='uniform' and x['mapping']==name]
    axes[0].plot([x['checkpoint'] for x in rs],[x['mean_window_samples'] for x in rs],'-o',label=name,ms=4)
    axes[1].plot([x['checkpoint'] for x in rs],[x['mean_feedback_age_updates'] for x in rs],'-o',label=name,ms=4)
axes[0].set_title('Real episodes per history window');axes[0].set_ylabel('Episodes');axes[1].set_title('Historical feedback age');axes[1].set_ylabel('PPO updates')
for ax in axes[:2]:ax.set_xlabel('Update');ax.legend(fontsize=7);ax.grid(alpha=.2)
names=['sequence']+list(maps)
for j,(first,last) in enumerate(((10001,12500),(12501,15000))):
    rr=[next(x for x in prediction_rows if x['first_update']==first and x['estimator']==n)['Brier'] for n in names]
    axes[2].bar(np.arange(len(names))+(j-.5)*.35,rr,.35,label=str(first)+'-'+str(last))
axes[2].set_xticks(np.arange(len(names)),names,rotation=35);axes[2].set_ylabel('Brier score (lower is better)');axes[2].set_title('Forward prediction on Uniform logs');axes[2].legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'feedback_density_age_prediction.png',dpi=180);plt.close(fig)
print('FEEDBACK_ANALYSIS_PASS condition='+str(summary['estimator_GO_condition_passed']),flush=True)
