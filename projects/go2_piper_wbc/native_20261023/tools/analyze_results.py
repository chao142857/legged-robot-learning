"""CPU-only aggregation and figures. Raw data and training remain untouched."""
from publication_paths import research_path
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser=argparse.ArgumentParser()
parser.add_argument('--partial',action='store_true')
args=parser.parse_args()
ROOT=Path(research_path('go2_piper_upstream_wbc_20261023'))
OUT=ROOT/('analysis_partial' if args.partial else 'analysis')
OUT.mkdir(exist_ok=True)
B0=Path(research_path('LeggedManip_Lab_go2_piper_native_wbc_20261022/docs/go2_piper_wbc_v1/results'))


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def csvwrite(name,rows):
    if not rows: return
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)


def rolling(x,n=100):
    x=np.array(x,dtype=float)
    size=min(n,len(x))
    return np.convolve(x,np.ones(size)/size,mode='valid')


def training():
    rows=[json.loads(s) for s in (ROOT/'formal_seed21/updates.jsonl').read_text().splitlines() if s.strip()]
    assert [r['completed_updates'] for r in rows]==list(range(1,len(rows)+1))
    if not args.partial: assert len(rows)==3000
    run=next(p for p in (ROOT/'native_logs/rsl_rl/go2_piper_wbc').iterdir()
             if not 'smoke' in p.name and p.is_dir())
    ea=EventAccumulator(str(run),size_guidance={'scalars':0});ea.Reload()
    scalars={tag:{e.step+1:e.value for e in ea.Scalars(tag)} for tag in ea.Tags()['scalars']}
    flat=[]
    for row in rows:
        item={k:row[k] for k in ('completed_updates','env_steps','elapsed_s','learning_rate','kl_mean','kl_max',
              'actor_grad_norm_mean','actor_grad_norm_max','critic_grad_norm_mean','critic_grad_norm_max')}
        item.update(row['losses']);item.update(row['metrics'])
        item.update(std_min=min(row['action_std']),std_max=max(row['action_std']))
        for k in scalars:
            item['native/'+k]=scalars[k].get(row['completed_updates'],float('nan'))
        flat.append(item)
    csvwrite('training_metrics.csv',flat)
    residual=max(abs(sum(v for k,v in r['metrics'].items() if k.startswith('reward_rate/'))*.02-r['metrics']['reward_step']) for r in rows)
    report=dict(completed_updates=len(rows),environment_steps=rows[-1]['env_steps'],elapsed_s=rows[-1]['elapsed_s'],
                source='observer + native TensorBoard',native_log_dir=str(run),
                reward_sum_max_abs_residual=residual,
                kl_mean_max=max(r['kl_mean'] for r in rows),
                kl_minibatch_mean_max=max(r['kl_max'] for r in rows),
                lr_min=min(r['learning_rate'] for r in rows),lr_max=max(r['learning_rate'] for r in rows),
                actor_grad_norm_max=max(r['actor_grad_norm_max'] for r in rows),
                critic_grad_norm_max=max(r['critic_grad_norm_max'] for r in rows),
                final=rows[-1],native_final={k:scalars[k].get(len(rows)) for k in scalars},
                native_scalar_tags=list(scalars))
    dump('training_summary.json',report)
    x=np.array([r['completed_updates'] for r in rows])
    fig,axes=plt.subplots(3,3,figsize=(15,11))
    specs=[('native/Train/mean_reward','Native mean episode reward',False),
           ('native/Train/mean_episode_length','Native episode length (steps)',False),
           ('kl_mean','Mean minibatch KL',True),('learning_rate','Adaptive LR',True),
           ('value','Critic loss',True),('surrogate','Actor surrogate loss',False),
           ('entropy','Entropy',False),('actor_grad_norm_max','Actor pre-clip norm max',True),
           ('critic_grad_norm_max','Critic pre-clip norm max',True)]
    for ax,(key,label,log) in zip(axes.flat,specs):
        y=[r.get(key,np.nan) for r in flat]
        ax.plot(x,y,alpha=.45,lw=.6);smooth=rolling(y);ax.plot(x[len(x)-len(smooth):],smooth,lw=1.3)
        if log: ax.set_yscale('log')
        ax.set_title(label);ax.set_xlabel('Completed updates');ax.grid(alpha=.2)
    fig.suptitle('Native GO2-PIPER-WBC — seed 21; raw + trailing 100-update mean')
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(key,label,root) in zip(axes.flat,[('xy_mse','XY RMSE (m/s)',True),('yaw_mse','Yaw RMSE (rad/s)',True),
                    ('ee_position_mse','EE position RMSE (m)',True),('ee_orientation_rad','EE angle (rad)',False),
                    ('base_height','Base height (m)',False),('nonfoot_fraction','Non-foot contact step fraction',False)]):
        y=np.array([r['metrics'][key] for r in rows]); y=np.sqrt(y) if root else y
        ax.plot(x,y,lw=.7);ax.set_title(label);ax.set_xlabel('Completed updates');ax.grid(alpha=.2)
    fig.suptitle('Native training distribution — stochastic rollout; pre-reset states')
    fig.tight_layout();fig.savefig(OUT/'training_physics.png',dpi=160);plt.close(fig)
    if (B0/'learning_curve.csv').exists():
        with (B0/'learning_curve.csv').open() as f: old=list(csv.DictReader(f))
        with (B0/'audit_v2/training_reward_curve.csv').open() as f: oldreward=list(csv.DictReader(f))
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        pairs=[('kl_mean','kl_mean','Mean KL'),('learning_rate','lr','Adaptive LR'),('value','value_loss','Value loss'),
               ('entropy','entropy','Entropy'),('native/Train/mean_reward','Train/mean_reward','Episode reward (different lengths)'),
               ('native/Train/mean_episode_length','Train/mean_episode_length','Episode length (steps)')]
        for i,(ax,(newkey,oldkey,label)) in enumerate(zip(axes.flat,pairs)):
            oldrows=old if i<4 else oldreward
            ax.plot([float(r['update']) for r in oldrows],[float(r[oldkey]) for r in oldrows],label='90-task B0',alpha=.6,lw=.7)
            ax.plot(x,[r.get(newkey,np.nan) for r in flat],label='Native',alpha=.7,lw=.7)
            if i<3: ax.set_yscale('log')
            ax.set_title(label);ax.set_xlabel('Completed updates');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Descriptive comparison only: native 20 s vs custom 8 s; different task distributions')
        fig.tight_layout();fig.savefig(OUT/'b0_learning_comparison.png',dpi=160);plt.close(fig)
    return report


def evaluations():
    files=sorted((ROOT/'evaluation').glob('cp_*/*/seed_*/summary.json'))
    if not args.partial: assert len(files)==24, f'Expected 24 evaluation runs including CP0, found {len(files)}'
    runs=[]
    rows=[]
    for p in files:
        s=json.loads(p.read_text());cp=int(p.parents[2].name[3:])
        s['checkpoint']=cp;s['path']=str(p.parent);runs.append(s)
        for window,groups in s['groups'].items():
            for group,metrics in groups.items():
                rows.append(dict(checkpoint=cp,mode=s['mode'],seed=s['seed'],window=window,group=group,**metrics))
    # Keys are consistent for populated groups; keep zero-sample groups in JSON.
    populated=[r for r in rows if r['samples']>0]
    csvwrite('evaluation_by_seed_group.csv',populated)
    dump('evaluation_runs.json',runs)
    pooled=[]
    for cp in [0,1000,2000,3000]:
        for mode in ['deterministic','stochastic']:
            selected=[r for r in runs if r['checkpoint']==cp and r['mode']==mode]
            if not selected: continue
            for window in ['all','after_1s']:
                for group in selected[0]['groups'][window]:
                    items=[r['groups'][window][group] for r in selected if r['groups'][window][group]['samples']>0]
                    n=sum(i['samples'] for i in items)
                    rec=dict(checkpoint=cp,mode=mode,window=window,group=group,samples=n,
                             total_episodes=sum(r['num_episodes'] for r in selected),
                             survived_20s=sum(r['survived_20s'] for r in selected),
                             contact_episodes=sum(r['episodes_with_nonfoot_contact'] for r in selected),
                             overspeed_episodes=sum(r['episodes_with_overspeed_diagnostic'] for r in selected))
                    if n:
                        for key in ['xy_rmse','yaw_rmse','ee_position_rmse','xy_zero_velocity_rmse','yaw_zero_velocity_rmse']:
                            rec[key]=float(np.sqrt(sum(i[key]**2*i['samples'] for i in items)/n))
                        for key in ['ee_position_mean','ee_orientation_mean','base_height_mean','nonfoot_step_fraction',
                                    'foot_contact_fraction','low_base_step_fraction','commanded_xy_speed_mean','actual_xy_speed_mean',
                                    'command_actual_xy_dot_mean','commanded_abs_yaw_mean','actual_abs_yaw_mean']:
                            rec[key]=sum(i[key]*i['samples'] for i in items)/n
                        rec['xy_command_response_gain']=rec['command_actual_xy_dot_mean']/rec['xy_zero_velocity_rmse']**2 if rec['xy_zero_velocity_rmse']>0 else None
                        rec['xy_mse_reduction_vs_stationary']=1-rec['xy_rmse']**2/rec['xy_zero_velocity_rmse']**2 if rec['xy_zero_velocity_rmse']>0 else None
                    pooled.append(rec)
    dump('evaluation_pooled.json',pooled)
    csvwrite('evaluation_pooled.csv',[r for r in pooled if r['samples']>0])
    if not runs: return
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,(key,label) in zip(axes.flat,[('xy_rmse','XY RMSE (m/s)'),('yaw_rmse','Yaw RMSE (rad/s)'),
                 ('ee_position_rmse','EE position RMSE (m)'),('ee_orientation_mean','EE angle (rad)'),
                 ('base_height_mean','Base height (m)'),('nonfoot_step_fraction','Non-foot contact fraction')]):
        for mode in ['deterministic','stochastic']:
            subset=[r for r in pooled if r['mode']==mode and r['window']=='after_1s' and r['group']=='all']
            ax.plot([r['checkpoint'] for r in subset],[r.get(key,np.nan) for r in subset],'-o',label=mode)
        if key=='nonfoot_step_fraction': ax.set_yscale('log')
        ax.set_title(label);ax.set_xlabel('Completed updates');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Native evaluation — pooled valid samples, age > 1 s, three evaluation seeds')
    fig.tight_layout();fig.savefig(OUT/'native_evaluation.png',dpi=160);plt.close(fig)
    for s in runs:
        if s['checkpoint']!=3000 or s['seed']!=7001: continue
        with np.load(Path(s['path'])/'trajectories.npz') as z:
            valid=z['valid'];cmdspeed=np.linalg.norm(z['base_cmd'][:,:,:2],axis=-1)
            # Choose median EE error among long-enough episodes with a moving-command majority.
            candidates=[i for i in range(valid.shape[1]) if valid[:,i].sum()>=300 and (cmdspeed[:,i][valid[:,i]]>=.1).mean()>.5]
            if not candidates: candidates=list(range(valid.shape[1]))
            ordered=sorted(candidates,key=lambda i:float(z['ee_position_error'][:,i][valid[:,i]].mean()))
            env_id=ordered[len(ordered)//2];mask=valid[:,env_id];t=z['age'][:,env_id][mask]*.02
            fig,axes=plt.subplots(4,2,figsize=(15,12),sharex=True)
            for ax,dim,label in [(axes[0,0],0,'Base vx (m/s)'),(axes[0,1],1,'Base vy (m/s)'),(axes[1,0],2,'Base yaw rate (rad/s)')]:
                ax.plot(t,z['base_cmd'][:,env_id,dim][mask],label='command');ax.plot(t,z['base_actual'][:,env_id,dim][mask],label='actual');ax.set_ylabel(label);ax.legend()
            axes[1,1].plot(t,z['ee_position_error'][:,env_id][mask],label='position (m)')
            axes[1,1].plot(t,z['ee_orientation_error'][:,env_id][mask],label='angle (rad)');axes[1,1].legend()
            for dim,ax in enumerate([axes[2,0],axes[2,1],axes[3,0]]):
                ax.plot(t,z['ee_cmd'][:,env_id,dim][mask],label='target');ax.plot(t,z['ee_actual_mixed'][:,env_id,dim][mask],label='actual');ax.set_ylabel(['EE x / link0 (m)','EE y / link0 (m)','EE z / world (m)'][dim]);ax.legend()
            ax=axes[3,1];ax.plot(t,z['base_height'][:,env_id][mask],label='base z (m)');ax.axhline(.2,color='grey',ls=':');ax.legend(loc='upper left')
            other=ax.twinx();other.plot(t,z['nonfoot_max_force'][:,env_id][mask],color='tab:red',alpha=.5,label='max nonfoot force (N)');other.legend(loc='upper right')
            for ax in axes.flat: ax.set_xlabel('Episode time (s)');ax.grid(alpha=.2)
            fig.suptitle(f"CP3000 {s['mode']} seed7001 env{env_id} — median EE error among moving-command episodes")
            fig.tight_layout();fig.savefig(OUT/f"trajectory_{s['mode']}.png",dpi=160);plt.close(fig)
            dump(f"representative_{s['mode']}.json",dict(env_id=env_id,selection='median EE error among >=6s moving-command-majority episodes, fallback all',source=s['path']))
            # Body origins and native sensor forces from saved states, not a physics rerun or mesh render.
            names=s['articulation_body_names'];sensor_names=s['native_contact_body_names']
            count=min(16,valid.shape[1])
            sample=sorted(range(count),key=lambda i:float(z['ee_position_error'][:,i][valid[:,i]].mean()))[count//2]
            length=int(valid[:,sample].sum())
            times=np.linspace(min(49,length-1),length-1,4).astype(int)
            fig=plt.figure(figsize=(16,4.8))
            for panel,frame in enumerate(times):
                ax=fig.add_subplot(1,4,panel+1,projection='3d')
                points=z['body_pos_world_first16'][frame,sample].copy()
                origin=points[names.index('base')].copy();origin[2]=0;points-=origin
                for leg in ['FL','FR','RL','RR']:
                    chain=['base',f'{leg}_hip',f'{leg}_thigh',f'{leg}_calf',f'{leg}_foot']
                    p=points[[names.index(k) for k in chain]]
                    ax.plot(p[:,0],p[:,1],p[:,2],color='steelblue',lw=2)
                chain=['base']+[f'link{i}' for i in range(7)]+['end_effector']
                p=points[[names.index(k) for k in chain]]
                ax.plot(p[:,0],p[:,1],p[:,2],color='darkorange',lw=2)
                forces=z['all_body_contact_forces'][frame,sample]
                for i,name in enumerate(names):
                    contact=name in sensor_names and forces[sensor_names.index(name)]>1
                    ax.scatter(*points[i],c='red' if contact else 'black',s=22 if contact else 5)
                ax.set_xlim(-.7,.9);ax.set_ylim(-.7,.7);ax.set_zlim(0,1.1)
                ax.set_xlabel('World x offset (m)');ax.set_ylabel('World y offset (m)');ax.set_zlabel('')
                ax.view_init(elev=18,azim=-60)
                ax.set_title(f't={z["age"][frame,sample]*.02:.2f}s, z={z["base_height"][frame,sample]:.3f}m')
            fig.suptitle(f'CP3000 {s["mode"]} seed7001 env{sample}: recorded body origins; z = height (m); red = contact >1 N')
            fig.tight_layout();fig.savefig(OUT/f'posture_{s["mode"]}.png',dpi=160,bbox_inches='tight',pad_inches=.15);plt.close(fig)


training()
evaluations()
dump('artifact_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
     for p in OUT.iterdir() if p.is_file() and p.name!='artifact_manifest.json'})
print('ANALYSIS_COMPLETE',OUT)
