import csv, json, math, os, pathlib, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from common import ROOT,write,file_sha

REFERENCE=pathlib.Path('/reference'); HOST='/data/go2_piper_sgs_e1_20261102'
delivery=ROOT/'delivery'; delivery.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text())
def host(p): return HOST+'/'+str(p.relative_to(ROOT))
summary=dict(scope='E1 scheduled sampling feedback, seed42; CP10000 strict deterministic held-out primary endpoint.',
             source=read(ROOT/'source_manifest.json'),preflight=read(ROOT/'audit/preflight.json'),runs=[],generated_unix=time.time(),
             limitation='One training seed. Sequence pairing is not replication across training seeds. No attribution to individual schedule components.')
for kind in ('flat','wbc'):
    label='formal-'+kind+'-sgs-e1'; folder=ROOT/'runs'/label; out=delivery/label; out.mkdir(exist_ok=True)
    result=read(folder/'result.json'); assert result['completed_updates']==10000
    updates=[json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()]
    assert [r['completed_updates'] for r in updates]==list(range(1,10001))
    curves={}; paired={}
    for method in ('uniform','sgs','sgs-e1'):
        base=ROOT if method=='sgs-e1' else REFERENCE
        runlabel='formal-'+kind+'-'+method
        es=[read(base/'evaluations'/runlabel/str(u).zfill(5)/'result.json') for u in range(1000,10001,1000)]
        assert all(e['train_history_updates']==0 and e['episodes']==1024 for e in es)
        curves[method]=es
    def labels(base,runlabel):
        rows=read(base/'evaluations'/runlabel/'10000/episode_results.json')
        return {x['sequence_id']:bool(x['values'][6]) for x in rows}
    new=labels(ROOT,label)
    for method in ('sgs','uniform'):
        old=labels(REFERENCE,'formal-'+kind+'-'+method); assert new.keys()==old.keys() and len(new)==1024
        paired[method]=dict(both_success=sum(new[i] and old[i] for i in new),
                            new_success_old_failure=sum(new[i] and not old[i] for i in new),
                            new_failure_old_success=sum(not new[i] and old[i] for i in new),
                            both_failure=sum(not new[i] and not old[i] for i in new))
    with (out/'strict_evaluation_curve.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['update','method','strict_joint_success','ee_position_rmse_m','native_failure_rate'])
        for method,es in curves.items():
            for u,e in zip(range(1000,10001,1000),es):w.writerow([u,method,e['joint_success_rate'],e['pooled_eligible_rmse'][2],e['native_failure_rate']])
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for method,es in curves.items():
        x=list(range(1000,10001,1000))
        axes[0].plot(x,[100*e['joint_success_rate'] for e in es],'-o',label=method)
        axes[1].plot(x,[100*e['pooled_eligible_rmse'][2] for e in es],'-o',label=method)
        axes[2].plot(x,[100*e['native_failure_rate'] for e in es],'-o',label=method)
    for ax,title in zip(axes,['Strict joint success (%)','EE position RMSE (cm)','Native failure (%)']):
        ax.set_title(title); ax.set_xlabel('PPO updates'); ax.grid(alpha=.2);ax.legend()
    fig.suptitle(kind+' | unchanged 1024-sequence deterministic evaluation | seed42');fig.tight_layout();fig.savefig(out/'strict_comparison.png',dpi=170);plt.close(fig)
    ea=EventAccumulator(str(folder/'policy'),size_guidance={'scalars':0}); ea.Reload()
    scalars={k:ea.Scalars(k) for k in ea.Tags()['scalars']}
    with (out/'all_native_scalars.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['tag','native_step','wall_time','value'])
        for tag,points in scalars.items():
            for p in points:w.writerow([tag,p.step,p.wall_time,p.value])
    tags=[k for k in scalars if not k.endswith('/time')]
    for page,offset in enumerate(range(0,len(tags),12),1):
        subset=tags[offset:offset+12];fig,axes=plt.subplots(math.ceil(len(subset)/3),3,figsize=(16,3*math.ceil(len(subset)/3)),squeeze=False)
        for ax,tag in zip(axes.flat,subset):
            points=scalars[tag];x=np.array([p.step+1 for p in points]);y=np.array([p.value for p in points]);ax.plot(x,y,alpha=.5,lw=.5)
            if len(y)>=100:ax.plot(x[99:],np.convolve(y,np.ones(100)/100,'valid'))
            ax.set_title(tag,fontsize=9);ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(subset):]:ax.set_visible(False)
        fig.tight_layout();fig.savefig(out/('native_curves_'+str(page)+'.png'),dpi=140);plt.close(fig)
    mechanisms=[read(p) for p in sorted((folder/'sampling').glob('*.json'))]
    switches=[read(p) for p in sorted((folder/'switches').glob('*.json'))]
    write(out/'mechanism_timeline.json',dict(snapshots=mechanisms,switches=switches))
    cp=folder/'policy/model_9999.pt';link=out/'final_checkpoint.pt'
    if not link.exists():link.symlink_to(os.path.relpath(cp,out))
    result.update(evaluations=curves,paired_final=paired,checkpoint=host(cp),training_log_directory=host(folder),
                  videos=[dict(path=host(p),bytes=p.stat().st_size,sha256=file_sha(p)) for p in (ROOT/'replays'/label/'videos').glob('*.mp4')])
    assert result['videos'];write(out/'result.json',result);summary['runs'].append(result)
write(delivery/'summary.json',summary)
lines=['E1 scheduled SGS sampling feedback; primary endpoint is CP10000 under original strict deterministic evaluation.',summary['limitation']]
for r in summary['runs']:
    lines.extend([r['task'],str({m:e[-1]['joint_success_rate'] for m,e in r['evaluations'].items()}),
                  'Paired final counts '+str(r['paired_final']),'Checkpoint '+r['checkpoint'],
                  'Updates '+str(r['completed_updates'])+'; transitions '+str(r['environment_transitions'])+'; training loop seconds '+str(r['training_loop_seconds'])])
(delivery/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('E1_REPORT_COMPLETE',host(delivery/'summary.json'),flush=True)
