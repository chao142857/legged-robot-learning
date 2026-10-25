"""Plots and evidence index from actual training logs and frozen evaluation data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--additional-evaluation-root', action='append', default=[])
    args=parser.parse_args();root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    updates=[json.loads(line) for line in (root/'train_seed21/updates.jsonl').read_text().splitlines() if line]
    assert [x['completed_updates'] for x in updates]==list(range(1,len(updates)+1))
    x=np.array([r['completed_updates'] for r in updates])
    fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for name,label in [('ee_mse','EE position'),('xy_mse','XY velocity'),('yaw_mse','yaw rate')]:
        axes[0,0].plot(x,[r['metrics'][name]**.5 for r in updates],label=label,lw=.8)
    axes[0,0].set(title='Stochastic training diagnostics: RMSE',xlabel='Completed updates');axes[0,0].legend()
    axes[0,1].semilogy(x,[r['learning_rate'] for r in updates],label='native adaptive LR')
    axes[0,1].set(title='Learning rate (no manual schedule change)',xlabel='Completed updates');axes[0,1].legend()
    for name in ('kl_mean','kl_max'):axes[1,0].plot(x,[r[name] for r in updates],label=name,lw=.8)
    axes[1,0].axhline(.01,c='black',ls=':',label='desired KL');axes[1,0].set(title='PPO minibatch KL');axes[1,0].legend()
    for name in ('actor_grad_norm_max','critic_grad_norm_max'):axes[1,1].semilogy(x,[max(r[name],1e-9) for r in updates],label=name,lw=.8)
    axes[1,1].set(title='Pre-clip gradient norms');axes[1,1].legend()
    for name in ('time_out','base_contact','bad_orientation','nonfoot_contact'):
        axes[2,0].plot(x,[r['metrics'].get('termination/'+name,0)*24 for r in updates],label=name,lw=.8)
    axes[2,0].set(title='Termination events per rollout (terms may overlap)');axes[2,0].legend()
    axes[2,1].plot(x,[np.mean(r['action_std'][:12]) for r in updates],label='legs')
    axes[2,1].plot(x,[np.mean(r['action_std'][12:]) for r in updates],label='arm')
    axes[2,1].set(title='Action distribution std');axes[2,1].legend()
    for a in axes.flat:a.grid(alpha=.2)
    fig.savefig(out/'training_diagnostics.png',dpi=150);plt.close(fig)
    train_rows=[]
    for r in updates:
        row={k:r[k] for k in ('completed_updates','env_steps','elapsed_s','learning_rate','kl_mean','kl_max','actor_grad_norm_max','critic_grad_norm_max')}
        row.update(r['metrics']);train_rows.append(row)
    with (out/'training_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(train_rows[0]));writer.writeheader();writer.writerows(train_rows)

    eval_rows=[];details=[];index=[]
    evaluation_roots=[root/'evaluation']+[Path(p) for p in args.additional_evaluation_root]
    seen_evaluations=set()
    for path in sorted(p for base in evaluation_roots for p in base.glob('*/*/seed_*')):
        if not path.is_dir() or not (path/'manifest.json').exists():continue
        protocol=json.loads((path/'protocol.json').read_text());summary=json.loads((path/'summary.json').read_text())
        tracking=json.loads((path/'tracking_metrics.json').read_text());label=path.parent.parent.name
        video=protocol['video']['enabled']
        identity=(label,protocol['task']['family'],protocol['seed'],video)
        if identity in seen_evaluations:raise ValueError(f'Duplicate evaluation: {identity}')
        seen_evaluations.add(identity)
        row=dict(label=label,family=protocol['task']['family'],seed=protocol['seed'],video=video,
            num_envs=summary['num_episodes'],complete=summary['complete_count'],failure=summary['failure_count'],
            historical_movement_success=summary['movement_success_count'],historical_joint_5cm=summary['joint_success_5cm_count'],
            historical_safe=summary['safe_count'])
        ee=tracking['windows']['all']['ee']
        row.update(ee_rmse=ee.get('position_rmse_m'),center_counterfactual_rmse=ee.get('center_counterfactual_rmse_m'),
            ee_mse_improvement_vs_center=ee.get('mse_improvement_vs_center'))
        for d,value in tracking['windows']['all']['directions'].items():
            for metric in ('gain','rmse','correct_sign_fraction','mse_improvement_vs_zero','samples'):
                row[d+'_'+metric]=value['command_plateau'].get(metric)
        for name,count in summary['terminations'].items():row['termination_'+name]=count
        eval_rows.append(row);details.append(dict(path=str(path),protocol=protocol,summary=summary,tracking=tracking))
        for name in ('trajectories.npz','full_rollout.mp4','protocol.json','summary.json','tracking_metrics.json','reset_state.json'):
            p=path/name
            if p.exists():index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    if eval_rows:
        keys=list(dict.fromkeys(k for r in eval_rows for k in r))
        with (out/'frozen_evaluation.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(eval_rows)
        main_rows=[r for r in eval_rows if r['family']=='T1' and not r['video']]
        labels=sorted({r['label'] for r in main_rows if r['label'].startswith('cp_')},key=lambda s:int(s[3:]))
        counts=[int(s[3:]) for s in labels]
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        def draw(ax,key,title):
            arrays=[[r[key] for r in main_rows if r['label']==label and r.get(key) is not None] for label in labels]
            means=[np.mean(v) if v else np.nan for v in arrays]
            low=[min(v) if v else np.nan for v in arrays];high=[max(v) if v else np.nan for v in arrays]
            ax.plot(counts,means,marker='o',label='scratch: mean over eval seeds');ax.fill_between(counts,low,high,alpha=.15,label='eval-seed min/max')
            native=[r[key] for r in main_rows if r['label']=='native_cp3000' and r.get(key) is not None]
            if native:ax.axhline(np.mean(native),c='tab:red',ls='--',label='native CP3000')
            ax.set(title=title,xlabel='Completed scratch updates');ax.grid(alpha=.2)
        for ax,d in zip(axes.flat,('forward','backward','left','right')):
            draw(ax,d+'_gain',d+' velocity response gain');ax.axhline(0,c='gray',ls=':');ax.axhline(1,c='gray',ls=':')
        draw(axes[1,1],'ee_rmse','EE position RMSE (valid first-episode samples)')
        draw(axes[1,2],'ee_mse_improvement_vs_center','EE MSE improvement vs mathematical center holding')
        axes[1,2].axhline(0,c='black',ls=':',label='center-holding mathematical reference')
        axes[0,0].legend(fontsize=8);axes[1,2].legend(fontsize=7)
        fig.suptitle('Deterministic frozen evaluation; one training seed, three evaluation seeds')
        fig.savefig(out/'frozen_checkpoint_curves.png',dpi=150);plt.close(fig)
        final=root/'evaluation/cp_5000/T1/seed_7101/trajectories.npz'
        native=root/'evaluation/native_cp3000/T1/seed_7101/trajectories.npz'
        if final.exists() and native.exists():
            datasets={'native CP3000':np.load(native),'scratch CP5000':np.load(final)}
            colors={'native CP3000':'tab:orange','scratch CP5000':'tab:blue'}
            fig,axes=plt.subplots(3,4,figsize=(17,9),layout='constrained')
            for label,data in datasets.items():
                for env in range(4):
                    mask=data['valid'][:,env];t=np.arange(1,len(mask)+1)[mask]*.02
                    for j,axis in enumerate('XYZ'):
                        axes[j,env].plot(t,data['ee_actual_mixed'][mask,env,j],c=colors[label],label=label,lw=1)
                        axes[j,env].plot(t,data['ee_cmd'][mask,env,j],c=colors[label],ls=':',alpha=.8,label=label+' target')
                        axes[j,env].set(title=f'env {env}, {axis}',xlabel='Episode time (s)',ylabel='Mixed-frame position (m)');axes[j,env].grid(alpha=.2)
            axes[0,0].legend(fontsize=7);fig.suptitle('First environment in each initial direction group; no post-reset samples; seed 7101')
            fig.savefig(out/'ee_axes_native_vs_scratch.png',dpi=150);plt.close(fig)
            fig,axes=plt.subplots(4,2,figsize=(13,12),layout='constrained')
            for label,data in datasets.items():
                for env in range(4):
                    mask=data['valid'][:,env];t=np.arange(1,len(mask)+1)[mask]*.02
                    for col,j in enumerate((0,2)):
                        axes[env,col].plot(t,data['base_actual'][mask,env,j],c=colors[label],label=label,lw=1)
                        axes[env,col].plot(t,data['base_cmd'][mask,env,j],c=colors[label],ls=':',label=label+' command')
                        axes[env,col].set(title=f'env {env}: '+('vx (m/s)' if j==0 else 'yaw rate (rad/s)'),xlabel='Episode time (s)');axes[env,col].grid(alpha=.2)
            axes[0,0].legend(fontsize=7);fig.suptitle('Four initial direction groups; actual response and command, seed 7101')
            fig.savefig(out/'direction_traces_native_vs_scratch.png',dpi=150);plt.close(fig)
    for p in sorted((root/'train_seed21').glob('cp_*.pt')):
        index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    for name in ('train.log','train_seed21/updates.jsonl','train_seed21/initialization.json','train_seed21/launch.json','train_seed21/resolved_config_difference.json'):
        p=root/name;index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    pause_path=root/'user_pause_1500.json'
    pause=json.loads(pause_path.read_text()) if pause_path.exists() else None
    pause_seconds=pause.get('pause_wall_seconds',0.) if pause else 0.
    result=dict(completed_updates=len(updates),completed_update_transitions=updates[-1]['env_steps'],
        training_elapsed_s=updates[-1]['elapsed_s']-pause_seconds,
        training_elapsed_wall_s=updates[-1]['elapsed_s'],excluded_user_pause_s=pause_seconds,
        user_pause=pause,evaluation_rows=eval_rows,
        interpretation='Training curves are stochastic diagnostics. Frozen metrics pool valid first-episode samples; failures remain in completion/success denominators. Seed bands are evaluation randomization variability, not independent training-seed confidence intervals.',
        counterfactuals='Fixed-center EE and zero velocity are mathematical references only, not robot rollouts.',
        termination_note='Termination terms can overlap within an episode.',details=details)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'delivery_manifest.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(completed_updates=len(updates),frozen_evaluation_runs=len(eval_rows),indexed_files=len(index),out=str(out))))


if __name__=='__main__':main()
