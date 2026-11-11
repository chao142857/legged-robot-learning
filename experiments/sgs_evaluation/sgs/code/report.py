import csv,json,pathlib,hashlib,math,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from common import ROOT,write,file_sha

def host(p):return '/data/go2_piper_sgs_20261102/'+str(p.relative_to(ROOT))
delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
summary=dict(source=json.loads((ROOT/'source_manifest.json').read_text()),protocol=json.loads((ROOT/'protocol_frozen.json').read_text()),engineering=json.loads((ROOT/'ENGINEERING_SHORT_PASS.json').read_text()),runs=[],generated_unix=time.time())
for kind in ('flat','wbc'):
    for mode in ('uniform','sgs'):
        label='formal-'+kind+'-'+mode;run=ROOT/'runs'/label;out=delivery/label;out.mkdir(exist_ok=True)
        result=json.loads((run/'result.json').read_text());assert result['completed_updates']==10000 and result['environment_transitions']==983040000
        updates=[json.loads(s) for s in (run/'updates.jsonl').read_text().splitlines()]
        assert [r['completed_updates'] for r in updates]==list(range(1,10001))
        evaluations=[json.loads((ROOT/'evaluations'/label/str(n).zfill(5)/'result.json').read_text()) for n in range(1000,10001,1000)]
        assert all(e['train_history_updates']==0 for e in evaluations)
        result['evaluations']=evaluations;result['all_update_records_contiguous']=True
        result['checkpoint']=host(pathlib.Path(result['checkpoint']))
        ea=EventAccumulator(str(run/'policy'),size_guidance={'scalars':0});ea.Reload()
        scalars={k:ea.Scalars(k) for k in ea.Tags()['scalars']}
        with (out/'all_native_scalars.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['tag','native_step','axis','wall_time','value'])
            for tag,points in scalars.items():
                for x in points:w.writerow([tag,x.step,'time_seconds' if tag.endswith('/time') else 'iteration_zero_based',x.wall_time,x.value])
        tags=[t for t in scalars if not t.endswith('/time')]
        for page,start in enumerate(range(0,len(tags),12)):
            subset=tags[start:start+12];fig,axs=plt.subplots(math.ceil(len(subset)/3),3,figsize=(16,3*math.ceil(len(subset)/3)),squeeze=False)
            for ax,tag in zip(axs.flat,subset):
                pts=scalars[tag];x=np.array([v.step+1 for v in pts]);y=np.array([v.value for v in pts]);ax.plot(x,y,lw=.5,alpha=.4)
                if len(y)>=100:ax.plot(x[99:],np.convolve(y,np.ones(100)/100,'valid'),lw=1)
                ax.set_title(tag,fontsize=9);ax.grid(alpha=.2);ax.set_xlabel('PPO updates')
            for ax in list(axs.flat)[len(subset):]:ax.set_visible(False)
            fig.suptitle(label+' | seed42');fig.tight_layout();fig.savefig(out/('native_curves_'+str(page+1)+'.png'),dpi=140);plt.close(fig)
        cp=pathlib.Path(result['checkpoint'].replace('/data/go2_piper_sgs_20261102','/workspace'))
        link=out/'final_checkpoint.pt'
        if not link.exists():link.symlink_to(pathlib.Path(__import__('os').path.relpath(cp,out)))
        videos=[]
        for p in (ROOT/'replays'/label/'videos').glob('*.mp4'):videos.append(dict(path=host(p),bytes=p.stat().st_size,sha256=file_sha(p)))
        assert videos;result['videos']=videos
        result['training_log_directory']=host(run);result['sampling_distributions']=host(run/'sampling')
        write(out/'result.json',result);summary['runs'].append(result)
    arms=[r for r in summary['runs'] if r['task'].lower().endswith(kind)]
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    for r in arms:
        es=r['evaluations'];x=np.arange(1000,10001,1000)
        axs[0,0].plot(x,[e['joint_success_rate'] for e in es],'-o',label=r['mode'])
        axs[0,1].plot(x,[e['native_failure_rate'] for e in es],'-o',label=r['mode'])
        for j,ax in enumerate([axs[0,2],axs[1,0],axs[1,1],axs[1,2]]):ax.plot(x,[e['pooled_eligible_rmse'][j] if e['pooled_eligible_rmse'] is not None else np.nan for e in es],'-o',label=r['mode'])
    for ax,title in zip(axs.flat,['Held-out joint success rate','Native failure rate','XY velocity RMSE (m/s)','Yaw-rate RMSE (rad/s)','EE position RMSE (m)','EE orientation RMSE (rad)']):ax.set_title(title);ax.set_xlabel('PPO updates');ax.grid(alpha=.2);ax.legend()
    fig.suptitle(kind+' | identical independent1024-sequence evaluation | seed42 training');fig.tight_layout();fig.savefig(delivery/(kind+'_uniform_vs_sgs.png'),dpi=160);plt.close(fig)
write(delivery/'summary.json',summary)
lines=['GO2+Piper fixed32768-sequence Uniform/SGS comparison','Native source 4e12109d1eccb1d9da2e4a35f189fabea3f13677','Four fresh seed42 runs; no short checkpoint initialization; original native10k reference preserved.','']
for r in summary['runs']:
    e=r['evaluations'][-1]
    lines.extend([r['task']+' '+r['mode'],f"PPO updates {r['completed_updates']}; transitions {r['environment_transitions']}; loop seconds {r['training_loop_seconds']}; paused eval seconds {r['paused_evaluation_seconds']}",f"Final held-out joint success {e['joint_success_rate']}; native failure {e['native_failure_rate']}; eligible pooled RMSE {e['pooled_eligible_rmse']}",'Checkpoint '+r['checkpoint'],'SHA256 '+r['checkpoint_sha256'],'Videos '+str(r['videos']),''])
(delivery/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print('FINAL_REPORT_COMPLETE',host(delivery/'summary.json'),flush=True)
