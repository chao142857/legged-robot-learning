#!/usr/bin/env python3
"""Independently recompute bounded pilot metrics, sampler accounting and frozen inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0,str(repo))
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmarks.go2_piper.wbc_sgs.sampler import TaskSampler
    a.output.mkdir(parents=True,exist_ok=True)
    plan=read(repo/'docs/wbc_sgs/PILOT_PLAN.json')
    protocol=read(repo/'benchmarks/go2_piper/wbc_sgs/success_protocol.json')
    assert read(a.root/'pilot_status.json')['status']=='COMPLETE_TWO_ARMS_FIXED_EVALUATION'
    summary={};curve=[];sampling={};manifests={};episodes={}
    for method in ['uniform','sgs']:
        t=read(a.root/f'pilot_{method}/manifest.json');e=read(a.root/f'eval_{method}/manifest.json')
        manifests[method]=t
        assert t['status']==e['status']=='COMPLETE'
        assert t['result']['completed_updates']==plan['updates']==128
        for run in [f'pilot_{method}',f'eval_{method}']:
            proc=read(a.root/(run+'_process')/'run.json')
            assert proc['returncode']==0 and not proc['timed_out'] and not proc['crash_detected']
        assert t['source_sha256']==plan['source_sha256']
        # Current branch may include a later resume-only repair. Verify executed
        # source against its actual committed revision, never silently substitute it.
        for name,h in t['source_sha256'].items():
            b=subprocess.check_output(['git','show',t['source_revision']+':'+name],cwd=repo)
            assert hashlib.sha256(b).hexdigest()==h
        records=lines(a.root/f'pilot_{method}/raw_episodes.jsonl')
        cp=torch.load(a.root/f'pilot_{method}/checkpoint_0128.pt',map_location='cpu',weights_only=False)
        state=cp['infos']['environment']['sampler']
        replay=TaskSampler(48,13,method)
        replay.record([r['task_id'] for r in records],[r['joint_success'] for r in records])
        for k in ['outcomes','completed_counts','success_counts']:assert replay.state_dict()[k]==state[k],k
        assert len(records)==sum(state['completed_counts'])==t['result']['completed_episodes']
        active=Counter(cp['infos']['environment']['active_task_ids'])
        assert sum(active.values())==128
        assert all(state['sample_counts'][i]==state['completed_counts'][i]+active[i] for i in range(48))
        s=TaskSampler(48,999,method);s.load_state_dict(state)
        sampling[method]=lines(a.root/f'pilot_{method}/sampling.jsonl')
        assert len(sampling[method])==128
        # Recompute history after every logged PPO rollout from raw terminal events.
        for row in sampling[method]:
            prefix=[r for r in records if r['control_step']<=row['update']*24]
            check=TaskSampler(48,13,method)
            check.record([r['task_id'] for r in prefix],[r['joint_success'] for r in prefix])
            assert check.rates()==row['rates']
            assert np.allclose(check.probabilities(),row['probabilities'],atol=1.e-14,rtol=0)
        for r in records:
            assert r['pool_sha256']==plan['pool_sha256'] and r['protocol_sha256']==plan['protocol_sha256']
            sufficient=r['scored_samples']>=protocol['minimum_scored_samples']
            assert r['locomotion']==(sufficient and r['good_fractions']['locomotion']>=.8)
            assert r['manipulation']==(sufficient and r['good_fractions']['manipulation']>=.8)
            expected=r['full_horizon'] and r['safety'] and r['locomotion'] and r['manipulation'] and r['good_fractions']['simultaneous']>=.8
            assert r['joint_success']==expected
        evals=[]
        for update in plan['evaluation_updates']:
            r=read(a.root/f'eval_{method}'/f'checkpoint_{update:04d}_episodes.json')
            assert len(r)==48 and {x['task_id'] for x in r}==set(range(48))
            episodes[method,update]=r
            row=dict(method=method,update=update,transitions=update*128*24,n=48,
                **{k:sum(x[k] for x in r) for k in ['joint_success','locomotion','manipulation','safety']})
            saved=next(x for x in e['result']['scores'] if x['updates']==update)
            assert sha(saved['checkpoint'])==saved['checkpoint_sha256']
            assert all(row[k]/48==saved[k] for k in ['joint_success','locomotion','manipulation','safety'])
            curve.append(row);evals.append(row)
        learning=lines(a.root/f'pilot_{method}/learning.jsonl')
        assert len(learning)==128 and learning[-1]['transitions']==393216
        summary[method]=dict(evaluation=evals,training_episodes=len(records),
            training_joint_successes=sum(r['joint_success'] for r in records),
            training_component_counts={k:sum(r[k] for r in records) for k in ['locomotion','manipulation','safety']},
            training_wall_s=t['result']['training_wall_s'],transitions=393216,
            train_transitions_per_s=393216/t['result']['training_wall_s'],final_sampler=s.summary(),
            minimum_effective_tasks=min(x['effective_tasks'] for x in sampling[method]),
            all48_sampled=all(x>0 for x in state['sample_counts']),
            max_history_length=max(map(len,state['outcomes'])),
            all18_outputs_changed=min(t['result']['output_weight_change_by_joint'])>0)
    for k in ['pool_sha256','protocol_sha256','environment_sha256','agent_sha256','reward_sha256',
              'source_sha256','initial_actor_sha256','initial_critic_sha256','source_revision']:
        assert manifests['uniform'][k]==manifests['sgs'][k],k
    initial_checkpoints={m:torch.load(a.root/f'pilot_{m}/checkpoint_0000.pt',map_location='cpu',weights_only=False) for m in ['uniform','sgs']}
    for key in ['actor_state_dict','critic_state_dict']:
        left=initial_checkpoints['uniform'][key];right=initial_checkpoints['sgs'][key]
        assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
    assert initial_checkpoints['uniform']['infos']['environment']['active_task_ids']==initial_checkpoints['sgs']['infos']['environment']['active_task_ids']
    assert episodes['uniform',0]==episodes['sgs',0], 'Initial fixed physical evaluations must match'
    final_u={r['task_id']:r for r in episodes['uniform',128]}
    final_s={r['task_id']:r for r in episodes['sgs',128]}
    paired={k:dict(uniform_only=sum(final_u[i][k] and not final_s[i][k] for i in range(48)),
                   sgs_only=sum(final_s[i][k] and not final_u[i][k] for i in range(48)),
                   both=sum(final_s[i][k] and final_u[i][k] for i in range(48)))
            for k in ['joint_success','locomotion','manipulation','safety']}
    frozen=read(repo/'docs/wbc_sgs/FROZEN_INPUTS.json')
    for name,item in {**frozen['previous_frozen'],**frozen['wbc_source']['files']}.items():assert sha(repo/name)==item['sha256'],name
    for item in frozen['previous_evidence']:assert sha(item['path'])==item['sha256'],item['path']
    resume=read(a.root/'resume_with_history/manifest.json')
    assert resume['status']=='COMPLETE' and resume['result']['resume_verified']
    assert resume['resumed_historical_episodes']==summary['sgs']['training_episodes']
    cp=torch.load(a.root/'pilot_sgs/checkpoint_0128.pt',map_location='cpu',weights_only=False)
    assert resume['restored_learning_rate']==cp['optimizer_state_dict']['param_groups'][0]['lr']==1.e-5
    data=dict(status='COMPLETE_BOUNDED_PILOT_NO_JOINT_BENEFIT',protocol_sha256=plan['protocol_sha256'],
        pool_sha256=plan['pool_sha256'],evaluated_commit=manifests['uniform']['source_revision'],
        plan_sha256=sha(repo/'docs/wbc_sgs/PILOT_PLAN.json'),
        experiment_branch='research/go2-piper-wbc-sgs-20261019',
        actual_pilot_wall_s=read(a.root/'pilot_status.json')['elapsed_s'],methods=summary,paired_final=paired,
        exact_initial_48_physical_records_match=True,configuration_and_initial_network_hashes_match=True,
        frozen_wbc_files_verified=len(frozen['wbc_source']['files']),previous_frozen_verified=len(frozen['previous_frozen']),
        previous_evidence_verified=len(frozen['previous_evidence']),resume_nonempty_history_and_learning_rate_verified=True,
        interpretation='Both training arms have zero binary joint successes; fixed deterministic success falls from40/48 to0/48 by32updates. SGS cannot prioritize a measured intermediate-success frontier; final q is uniform. No demonstrated joint benefit. Stop; no threshold tuning/long run/three-seed efficacy launch.',
        limitations=['One training seed;48 known fixed tasks, no unseen-geometry evidence',
            'Warm-start mean, random critic, stochastic std=.2: bounded fine-tuning, not training from scratch',
            'Dense reward/success-threshold alignment, cold critic/exploration/optimization are unisolated candidates for degradation, not proven causes',
            'Deterministic mature calibration has nontrivial success; zero training signal does not prove task physical infeasibility',
            'Original self-collision disabled; safety is an observed-contact/posture proxy',
            'Resume-only LR/historical-log accounting repair audited after uninterrupted pilot; fixed evaluation uses original committed checkpoints only'])
    (a.output/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    (a.output/'fixed_learning_curve.json').write_text(json.dumps(curve,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'uniform':'#2878b5','sgs':'#d34f34'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True,constrained_layout=True)
    for ax,key,title in zip(axes.flat,['joint_success','locomotion','manipulation','safety'],['Joint success','Locomotion','Manipulation','Observed safety']):
        for method in ['uniform','sgs']:
            rows=[r for r in curve if r['method']==method]
            ax.plot([r['transitions']/1000 for r in rows],[r[key]/48 for r in rows],
                color=colors[method],label=method,marker='o' if method=='uniform' else 'x',
                linestyle='-' if method=='uniform' else '--',linewidth=2)
        ax.set_title(title);ax.set_ylim(-.04,1.04);ax.grid(alpha=.2);ax.set_xlabel('Training transitions (thousands)')
        ax.set_ylabel('Fraction of 48 fixed episodes')
    axes[0,0].legend();fig.suptitle('Fixed evaluation: one training seed, unchanged 48-task distribution')
    fig.savefig(a.output/'fixed_evaluation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    vmax=max(max(r['probabilities']) for rows in sampling.values() for r in rows)
    for ax,method in zip(axes[0],['uniform','sgs']):
        image=ax.imshow([r['probabilities'] for r in sampling[method]],origin='lower',aspect='auto',
                        vmin=0,vmax=vmax,cmap='viridis',extent=[-.5,47.5,.5,128.5])
        ax.set_title(method+' task probabilities');ax.set_xlabel('Task ID (shuffled, no difficulty order)');ax.set_ylabel('PPO update')
    fig.colorbar(image,ax=axes[0].tolist(),label='Categorical probability')
    for method,shift in [('uniform',-.2),('sgs',.2)]:
        counts=np.array(summary[method]['final_sampler']['sampled'])
        axes[1,0].bar(np.arange(48)+shift,counts/counts.sum(),width=.4,label=method,color=colors[method])
        axes[1,1].plot([r['update'] for r in sampling[method]],[r['effective_tasks'] for r in sampling[method]],color=colors[method],label=method)
    axes[1,0].axhline(1/48,color='grey',linestyle=':',label='Uniform expectation')
    axes[1,0].set_title('Realized assignment frequencies (includes active episodes)');axes[1,0].set_xlabel('Task ID');axes[1,0].legend(fontsize=8)
    axes[1,1].set_title('Effective configurations: 1/sum(q^2)');axes[1,1].set_xlabel('PPO update');axes[1,1].set_ylim(0,50);axes[1,1].legend()
    fig.suptitle('SGS pilot: all recorded joint outcomes zero; final sampling returns to uniform')
    fig.savefig(a.output/'sampling_distribution.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for method in ['uniform','sgs']:
        rows=lines(a.root/f'pilot_{method}/learning.jsonl');x=[r['update'] for r in rows]
        axes[0].plot(x,[r['loss']['value'] for r in rows],label=method,color=colors[method])
        axes[1].plot(x,[r['learning_rate'] for r in rows],label=method,color=colors[method])
        axes[2].plot(x,[r['completed_episodes'] for r in rows],label=method,color=colors[method])
    for ax,title in zip(axes,['PPO value loss','Adaptive learning rate','Completed training episodes (all joint failures)']):
        ax.set_title(title);ax.set_xlabel('PPO update');ax.grid(alpha=.2);ax.legend()
    fig.savefig(a.output/'ppo_diagnostics.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in data.items() if k!='methods'},indent=2))


if __name__=='__main__':main()
