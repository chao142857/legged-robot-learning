"""Read-only analysis of V0.2 evidence, with append-only report snapshots.

No simulator, training, Teacher update, or formal-test-driven model selection.
Use --stage meta after eight forks; --stage final after frozen seed47 transfer.
Incomplete records stay visibly incomplete instead of being interpolated.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from common import ROOT, plain, write

EXPERIENCE = Path('/experience')
META = EXPERIENCE/'meta-eight-starts'
ARMS = ('uniform','fresh','meta')
COLORS = dict(uniform='#4d4d4d',fresh='#e69f00',meta='#0072b2',plus='#009e73',minus='#cc79a7')
METRIC_FIELDS = ('update','joint_success_rate','continuous_joint_score','native_failure_rate',
    'no_valid_window_rate','longest_joint_good_seconds_mean','loco_success_rate',
    'manipulation_rmse_pass_rate','pooled_rmse','safety','wall_seconds',
    'executed_environment_transitions','first_episode_transitions','checkpoint_sha256',
    'actor_sha256','initial_sha256','pool_sha256','seed','episodes','kind',
    'reset_repeat_mean_gap','sequence_mean_standard_error')


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def jsonlines(path, warnings):
    rows=[]
    if not Path(path).exists(): return rows
    with Path(path).open() as f:
        for line_number,line in enumerate(f,1):
            try: rows.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.append(f'Incomplete/invalid JSON line {path}:{line_number}; omitted, evidence unchanged.')
    return rows


def brief(result):
    if result is None: return None
    return {k:result[k] for k in METRIC_FIELDS if k in result}


def evaluation(label):
    path=ROOT/'evaluations'/label/'result.json'
    result=read(path)
    if result is None: return None
    return dict(result,path=str(path),result_sha256=sha(path))


def paired(left,right):
    if left is None or right is None: return None
    assert left['initial_sha256']==right['initial_sha256']
    assert left['pool_sha256']==right['pool_sha256'] and left['episodes']==right['episodes']
    a=np.asarray(left['scores']);b=np.asarray(right['scores'])
    difference=b-a
    seq=difference.reshape(2,-1).mean(0) if left['episodes']==512 else difference
    return dict(continuous_difference=float(difference.mean()),
        paired_sequence_standard_error=float(seq.std(ddof=1)/math.sqrt(len(seq))) if len(seq)>1 else None,
        paired_sequence_difference_sd=float(seq.std(ddof=1)) if len(seq)>1 else None,
        sequences_improved=int((seq>0).sum()),sequences_worse=int((seq<0).sum()),
        mean_absolute_reset_difference=float(abs(difference).mean()),
        strict_difference=right['joint_success_rate']-left['joint_success_rate'],
        physical_rmse_difference={k:right['pooled_rmse'][k]-left['pooled_rmse'][k] for k in left['pooled_rmse']},
        interpretation='Right minus left. Sequence SE is descriptive conditional on these policies; it is not independent Student-seed uncertainty.')


def empirical(p):
    p=np.asarray(p,dtype=float)
    if p.sum()<=0: return None
    p=p/p.sum();positive=p[p>0]
    return dict(tv_uniform=float(abs(p-1/len(p)).sum()/2),effective_sequences=float(1/(p@p)),
                entropy=float(-(positive*np.log(positive)).sum()),covered_sequences=int((p>0).sum()),
                top10_mass=float(np.sort(p)[-max(1,len(p)//10):].sum()))


def exposure(label, probability_path=None):
    dataset=EXPERIENCE/label
    cycles=[]
    for cycle in sorted(dataset.glob('cycle_*')):
        outcome=read(cycle/'outcome.json')
        if outcome is None or not (cycle/'real_draws.npz').exists(): continue
        with np.load(cycle/'real_draws.npz',allow_pickle=False) as data:
            counts=data['counts'].astype(float)
            transitions=data['executed_transitions_per_sequence'].astype(float)
            assigned_steps=np.bincount(data['sequence_id'],weights=data['exposure_steps_until_boundary'],minlength=len(counts))
            probability=None
            if probability_path is not None and Path(probability_path).exists():
                probability=np.load(probability_path,allow_pickle=False)
            else:
                # Reading compact decision metadata alone does not reconstruct q;
                # q is saved exactly in this trusted, local experience checkpoint.
                import torch
                saved=torch.load(cycle/'decision.pt',map_location='cpu',weights_only=False)
                probability=np.asarray(saved['decision']['probability'])
                del saved
            item=dict(cycle=cycle.name,completed_updates=outcome['completed_updates'],
                assigned_episodes=int(counts.sum()),proposed=outcome['distribution'],
                empirical_draws=empirical(counts),physical_all_versions=empirical(transitions),
                physical_current_decision=empirical(assigned_steps),
                cycle_version_exposure=outcome.get('cycle_version_exposure'),
                actual_draw_tv_from_proposal=float(abs(counts/counts.sum()-probability).sum()/2) if counts.sum() else None,
                actual_draw_probability_mass_seen=float(probability[counts>0].sum()),
                expected_unique_sequences_under_iid_proposal=float((-np.expm1(counts.sum()*np.log1p(-probability))).sum()),
                all_version_transitions=int(transitions.sum()),current_decision_exposure_steps=int(assigned_steps.sum()),
                zero_physics_exposure_draws=outcome['zero_physics_exposure_draws'],
                real_draws_sha256=sha(cycle/'real_draws.npz'),committed=(cycle/'COMMITTED.json').exists())
            cycles.append(item)
    return dict(cycles=cycles,interpretation='Proposal q is distinct from sparse empirical episode draws. Physical exposure includes carried-over episodes; current-decision exposure is reported separately. Empirical TV has multinomial sampling noise and is not itself a learned-policy distance.')


def meta_summary(warnings):
    points=[]
    for index,point in enumerate(sorted(META.glob('point_*'))):
        decision=read(point/'decision.json');update=read(point/'update.json')
        if decision is None: continue
        index=int(point.name.split('_')[-1]);seed=decision['student_seed'];start=decision['start_update']
        label=f'p{index:02d}-seed{seed}-cp{start:05d}'
        before=evaluation(label+'-before')
        after={a:evaluation(label+'-'+a+'-after') for a in ('uniform','plus','minus')}
        repeats={a:evaluation(label+'-'+a+'-after-repeat') for a in after}
        item=dict(index=index,student_seed=seed,start_update=start,decision=decision,teacher_update=update,
            engineering=read(point/'engineering.json'),before=brief(before),
            after={a:brief(r) for a,r in after.items()},
            gain_from_same_start={a:paired(before,r) for a,r in after.items()},
            gain_over_uniform={a:paired(after['uniform'],after[a]) for a in ('plus','minus')},
            repeat_evaluation_noise={a:paired(after[a],repeats[a]) for a in after if repeats[a] is not None},
            before_repeat_noise=paired(before,evaluation(label+'-before-repeat')),
            same_policy_restore_and_training_noise=paired(after['uniform'],evaluation(label+'-uniform-repeat-after')),
            exposure={a:exposure(label+'-'+a,point/f'{a}.npy') for a in after},
            update_json_sha256=sha(point/'update.json') if update is not None else None)
        points.append(item)
    differences=[p['teacher_update']['paired_reward_difference'] for p in points if p['teacher_update']]
    return dict(points=points,completed_teacher_updates=len(differences),
        mean_antithetic_reward_difference=float(np.mean(differences)) if differences else None,
        positive_antithetic_differences=int(np.sum(np.asarray(differences)>0)),
        negative_antithetic_differences=int(np.sum(np.asarray(differences)<0)),
        no_claim_of_causal_significance=True)


def tree_hash(value):
    """Same typed CPU hash contract as fork_state, without importing Isaac Lab."""
    import torch
    h=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):item=item.detach().cpu().contiguous().numpy()
        if isinstance(item,np.ndarray):h.update(f'array:{item.dtype}:{item.shape}:'.encode());h.update(item.tobytes())
        elif isinstance(item,dict):
            h.update(f'dict:{len(item)}:'.encode())
            for key in sorted(item,key=lambda k:(type(k).__name__,repr(k))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            h.update(f'{type(item).__name__}:{len(item)}:'.encode())
            for child in item:visit(child)
        else:h.update(f'{type(item).__name__}:{item!r};'.encode())
    visit(value);return h.hexdigest()


def frozen_validation(label,arm,result):
    import torch
    source=ROOT/'teachers'/('meta-final.pt' if arm=='meta' else 'fresh.pt')
    if not source.exists():return dict(available=False,passed=None)
    expected=torch.load(source,map_location='cpu',weights_only=False)['teacher']
    full_hash=tree_hash(expected)
    network_hash=hashlib.sha256()
    for k,v in sorted(expected['network'].items()):
        network_hash.update(k.encode());network_hash.update(v.detach().cpu().contiguous().numpy().tobytes())
    teacher_network_hash=network_hash.hexdigest()
    checked=0;mismatches=[]
    for cycle in sorted((EXPERIENCE/label).glob('cycle_*')):
        path=cycle/'decision.pt'
        if not path.exists():continue
        saved=torch.load(path,map_location='cpu',weights_only=False)['decision']
        if saved['teacher_sha256']!=teacher_network_hash or tree_hash(saved['teacher'])!=full_hash:
            mismatches.append(str(path))
        checked+=1
    reported=result.get('teacher_state_sha256') if result else None
    passed=(not mismatches and (reported==full_hash if result else True)) if checked else (False if result else None)
    return dict(available=True,passed=passed,teacher_checkpoint=str(source),teacher_checkpoint_sha256=sha(source),
        expected_complete_teacher_hash=full_hash,expected_network_hash=teacher_network_hash,
        completed_run_reported_teacher_hash=reported,checked_decision_snapshots=checked,mismatches=mismatches,
        coverage='All available decision snapshots plus completed-run final state receipt; no mutation or formal-test selection.')


def transfer_summary(warnings):
    result={}
    for arm in ARMS:
        label='transfer-seed47-'+arm;run=ROOT/'runs'/label
        evaluations=[]
        for path in sorted((ROOT/'evaluations').glob(label+'-*/result.json')):
            r=read(path)
            if r is not None:evaluations.append(dict(brief(r),path=str(path),result_sha256=sha(path)))
        final=read(run/'result.json')
        result[arm]=dict(label=label,student_seed=47,result=final,initialization=read(run/'initialization.json'),
            strict_test_curve=sorted([r for r in evaluations if r['kind']=='test'],key=lambda r:r['update']),
            development_curve=sorted([r for r in evaluations if r['kind']=='development'],key=lambda r:r['update']),
            frozen_teacher=frozen_validation(label,arm,final),
            sampling=exposure(label),latest_training=read(run/'latest.json'))
    initial=[x['initialization'] for x in result.values() if x['initialization'] is not None]
    same_initialization=len(initial)==3 and all(x==initial[0] for x in initial[1:])
    return dict(arms=result,all_three_student_initializations_identical=same_initialization,
                available_initializations=len(initial),test_in_teacher_inputs=False,
                selection='All fixed every1000 checkpoints and final10k; missing points left absent.')


def interval_union(intervals):
    total=0.;end=None
    for start,stop in sorted(intervals):
        if end is None or start>end:total+=stop-start;end=stop
        elif stop>end:total+=stop-end;end=stop
    return total


def cost_summary(warnings):
    jobs=[];intervals=[];now=time.time()
    for directory in sorted((ROOT/'jobs').glob('*')):
        record=read(directory/'result.json') or read(directory/'launch.json')
        if record is None:continue
        job={k:record.get(k) for k in ('label','gpu','started_unix','finished_unix','wall_seconds','exit_code')}
        job['complete']=job['finished_unix'] is not None
        if job['wall_seconds'] is None:job['wall_seconds']=now-job['started_unix']
        jobs.append(job)
        if job['gpu']:intervals.append((job['started_unix'],job['finished_unix'] or now))
    run_costs=[];simulator_faults=[]
    for run in sorted((ROOT/'runs').glob('*')):
        if not run.is_dir():continue
        rows=jsonlines(run/'updates.jsonl',warnings)
        updates=[r['completed_updates'] for r in rows]
        if len(updates)!=len(set(updates)):warnings.append(f'Duplicate PPO update log IDs in {run}')
        result=read(run/'result.json')
        fault=run/'fault.txt'
        if fault.exists():
            lines=fault.read_text(errors='replace').splitlines()
            job_result=read(ROOT/'jobs'/run.name/'result.json') or {}
            simulator_faults.append(dict(run=run.name,path=str(fault),sha256=sha(fault),
                final_traceback_lines=lines[-12:],reported_container_exit_code=job_result.get('exit_code'),
                completed_result_present=bool(result and result.get('passed')),
                note='Isaac application shutdown can return container exit0 after a Python exception; the preserved fault receipt is independent evidence.'))
        run_costs.append(dict(run=run.name,logged_new_ppo_updates=len(set(updates)),
            first_logged_update=min(updates) if updates else None,last_logged_update=max(updates) if updates else None,
            completed=bool(result and result.get('passed')),reported_wall_seconds=result.get('wall_seconds') if result else None,
            reported_evaluation_wait_seconds=result.get('evaluation_seconds') if result else None,
            reported_active_training_and_startup_seconds=max(0,result['wall_seconds']-result.get('evaluation_seconds',0)) if result else None,
            note='Count logs actually emitted by this process; resumed smoke cumulative new_updates is not summed.'))
    eval_costs=[]
    for path in sorted((ROOT/'evaluations').glob('*/result.json')):
        r=read(path)
        eval_costs.append(dict(label=path.parent.name,kind=r['kind'],wall_seconds=r['wall_seconds'],
            executed_environment_transitions=r['executed_environment_transitions'],
            first_episode_transitions=r['first_episode_transitions'],passed=r['passed']))
    pre=sum(r['logged_new_ppo_updates'] for r in run_costs if not r['run'].startswith('transfer-'))
    transfer=sum(r['logged_new_ppo_updates'] for r in run_costs if r['run'].startswith('transfer-'))
    train_transitions=(pre+transfer)*98304
    evaluation_transitions=sum(r['executed_environment_transitions'] for r in eval_costs)
    if simulator_faults:
        warnings.append(f"{len(simulator_faults)} preserved simulator fault receipt(s) found, including any failed attempts later superseded by successful retries. Container exit0 does not override a fault receipt; see cost.simulator_faults.")
    return dict(jobs=jobs,runs=run_costs,evaluations=eval_costs,
        expected_pre_transfer_updates=2604,expected_meta_three_arm_updates=2400,
        expected_restore_noise_extra_updates=200,expected_smoke_updates=4,expected_transfer_updates=30000,
        actual_logged_pre_transfer_updates=pre,actual_logged_transfer_updates=transfer,
        actual_logged_training_transitions=train_transitions,evaluation_executed_transitions=evaluation_transitions,
        total_observed_environment_transitions=train_transitions+evaluation_transitions,
        evaluation_first_episode_transitions=sum(r['first_episode_transitions'] for r in eval_costs),
        evaluation_reported_wall_seconds=sum(r['wall_seconds'] for r in eval_costs),
        gpu_job_wall_seconds_sum_including_overlap=sum(j['wall_seconds'] for j in jobs if j['gpu']),
        gpu_job_interval_union_seconds=interval_union(intervals),
        cpu_job_wall_seconds_sum=sum(j['wall_seconds'] for j in jobs if not j['gpu']),
        failed_jobs=[j for j in jobs if j['exit_code'] not in (None,0)],
        simulator_faults=simulator_faults,
        interpretation='GPU-job interval union avoids double-counting paused parent training and nested evaluation jobs. It measures GPU allocation duration, not device active compute. Incomplete/failed jobs are included in elapsed allocation; training transition counts are logged lower bounds if a crash preceded logging. Source-checkpoint pretraining cost is inherited and excluded; all new meta, noise, smoke and transfer costs are included.')


def figures(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    paths=[]
    points=summary['meta']['points'];x=np.arange(len(points))
    if points:
        fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
        for offset,arm in ((-.18,'plus'),(.18,'minus')):
            y=[100*p['gain_over_uniform'][arm]['continuous_difference'] if p['gain_over_uniform'][arm] else np.nan for p in points]
            ax.bar(x+offset,y,.34,label=arm,color=COLORS[arm])
        ax.axhline(0,color='#777',linewidth=.8);ax.set_xticks(x,[f"s{p['student_seed']} / {p['start_update']//1000}k" for p in points])
        ax.set(ylabel='Development continuous score gain vs Uniform (pp)',xlabel='Fixed Student starting checkpoint',
               title='Counterfactual 100-update branches: same-start comparison')
        ax.legend();ax.grid(axis='y',alpha=.2)
        fig.text(.01,-.045,'One paired perturbation per starting state; this is not independent-seed evidence of algorithm superiority.',fontsize=9)
        path=out/'meta_branch_gain.png';fig.savefig(path,bbox_inches='tight');plt.close(fig);paths.append(str(path))
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for arm in ('uniform','plus','minus'):
            axes[0,0].plot(x,[100*p['gain_from_same_start'][arm]['continuous_difference'] if p['gain_from_same_start'][arm] else np.nan for p in points],marker='o',label=arm,color=COLORS[arm])
        for arm in ('plus','minus'):
            axes[0,1].plot(x,[100*p['decision'][arm]['tv_uniform'] for p in points],marker='o',label=arm,color=COLORS[arm])
        axes[0,1].plot(x,[100*p['teacher_update']['policy_after']['tv_uniform'] if p['teacher_update'] else np.nan for p in points],
            marker='s',linewidth=2.2,color='#222222',label='Updated mean policy (no exploration)')
        for key,label in (('gradient_norm_before_clip','Gradient norm'),('parameter_delta_l2','Parameter change L2')):
            axes[1,0].plot(x,[p['teacher_update'][key] if p['teacher_update'] else np.nan for p in points],marker='o',label=label)
        for arm in ('plus','minus'):
            actual=[];proposed=[]
            for p in points:
                cycles=p['exposure'][arm]['cycles']
                actual.append(cycles[-1]['empirical_draws']['effective_sequences'] if cycles and cycles[-1]['empirical_draws'] else np.nan)
                proposed.append(p['decision'][arm]['effective_sequences'])
            axes[1,1].plot(x,proposed,label=arm+' proposal',color=COLORS[arm])
            axes[1,1].plot(x,actual,'o--',label=arm+' realized draws',color=COLORS[arm])
        for ax,title,ylabel in zip(axes.ravel(),['Real development gain','Proposal distance from Uniform','Teacher update magnitude','Proposal vs finite-sample exposure'],['Percentage points','TV (%)','L2 norm','Effective sequence count']):
            ax.set(title=title,ylabel=ylabel,xlabel='Teacher decision index');ax.grid(alpha=.2);ax.legend(fontsize=8)
        path=out/'meta_teacher_diagnostics.png';fig.savefig(path);plt.close(fig);paths.append(str(path))
    transfer=summary['transfer']['arms']
    if any(v['development_curve'] or v['strict_test_curve'] for v in transfer.values()):
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
        for arm,r in transfer.items():
            for ax,kind,metric in ((axes[0],'strict_test_curve','joint_success_rate'),(axes[1],'development_curve','continuous_joint_score')):
                rows=r[kind];available={q['update']:q[metric] for q in rows}
                grid=range(1000,10001,1000) if kind=='strict_test_curve' else range(0,10001,100)
                ax.plot(list(grid),[100*available.get(u,np.nan) for u in grid],marker='o' if kind=='strict_test_curve' else None,label=arm,color=COLORS[arm])
        for ax,title in zip(axes,['Independent strict test: 5 cm / 80%','Development continuous joint compliance']):
            ax.set(title=title,xlabel='Student PPO updates',ylabel='Percent',xlim=(0,10000),ylim=(0,100));ax.legend();ax.grid(alpha=.2)
        path=out/'transfer_learning_curves.png';fig.savefig(path);plt.close(fig);paths.append(str(path))
        metrics=[('ee_position_m',100,'EE position RMSE (cm)'),('ee_orientation_rad',180/np.pi,'EE orientation RMSE (deg)'),('velocity_m_s',1,'Loco XY RMSE (m/s)'),('yaw_rad_s',1,'Yaw RMSE (rad/s)')]
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for ax,(key,scale,title) in zip(axes.ravel(),metrics):
            for arm,r in transfer.items():
                rows={q['update']:q for q in r['strict_test_curve']};grid=list(range(1000,10001,1000))
                ax.plot(grid,[rows[u]['pooled_rmse'][key]*scale if u in rows else np.nan for u in grid],marker='o',label=arm,color=COLORS[arm])
            ax.set(title=title,xlabel='PPO updates',xlim=(0,10000));ax.grid(alpha=.2);ax.legend(fontsize=8)
        path=out/'transfer_physical_errors.png';fig.savefig(path);plt.close(fig);paths.append(str(path))
        fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
        safety_metrics=[('native_failure_rate','Native failure (%)'),
            ('body_tilt_over_30deg_step_fraction_mean','Body tilt >30 deg: mean step fraction (%)'),
            ('joint_soft_limit_step_fraction_mean','Joint soft limit: mean step fraction (%)')]
        for ax,(key,title) in zip(axes,safety_metrics):
            for arm,r in transfer.items():
                rows={q['update']:q for q in r['strict_test_curve']};grid=list(range(1000,10001,1000))
                ax.plot(grid,[100*rows[u]['safety'][key] if u in rows else np.nan for u in grid],marker='o',label=arm,color=COLORS[arm])
            ax.set(title=title,xlabel='PPO updates',xlim=(0,10000));ax.grid(alpha=.2);ax.legend(fontsize=8)
        path=out/'transfer_safety.png';fig.savefig(path);plt.close(fig);paths.append(str(path))
    return paths


def text_report(summary):
    cost=summary['cost'];meta=summary['meta'];transfer=summary['transfer']
    lines=[f"nSGS V0.2 {summary['stage']} report",f"Created UTC: {summary['created_utc']}",
        f"Completed Teacher updates: {meta['completed_teacher_updates']}/8",
        f"New logged Student updates: meta/noise/smoke {cost['actual_logged_pre_transfer_updates']}/2604; transfer {cost['actual_logged_transfer_updates']}/30000",
        f"New training transitions: {cost['actual_logged_training_transitions']:,}",
        f"Extra evaluation transitions: {cost['evaluation_executed_transitions']:,}",
        f"GPU allocation interval union: {cost['gpu_job_interval_union_seconds']/3600:.3f} h (not active kernel time)",
        '', 'Fixed-start counterfactual results (continuous development score, percentage points):']
    for p in meta['points']:
        gains=p['gain_over_uniform']
        show=lambda a: f"{100*gains[a]['continuous_difference']:+.3f}" if gains[a] else 'missing'
        lines.append(f"  seed {p['student_seed']} CP{p['start_update']}: q+−Uniform {show('plus')}; q−−Uniform {show('minus')}")
        noise=p['same_policy_restore_and_training_noise']
        if noise:lines.append(f"    Uniform restore/retrain repeat difference {100*noise['continuous_difference']:+.3f} pp; mean absolute paired reset difference {100*noise['mean_absolute_reset_difference']:.3f} pp.")
    lines.extend(['','Frozen Teacher transfer, new Student seed47:'])
    for arm,r in transfer['arms'].items():
        tests=r['strict_test_curve'];last=tests[-1] if tests else None
        lines.append(f"  {arm}: "+(f"CP{last['update']} strict {100*last['joint_success_rate']:.3f}%, continuous {100*last['continuous_joint_score']:.3f}%" if last else 'no formal evaluation yet')+
                     f"; frozen-state validation {r['frozen_teacher']['passed']}.")
    lines.extend(['','Interpretation and limits:']+['  '+x for x in summary['limitations']])
    if summary['warnings']:lines.extend(['','Warnings:']+['  '+x for x in summary['warnings']])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['meta','final'],required=True)
    args=parser.parse_args();warnings=[]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=ROOT/'results'/f'{args.stage}_{stamp}';out.mkdir(parents=True,exist_ok=False)
    summary=dict(stage=args.stage,created_utc=stamp,source_protocol=read(ROOT/'PROTOCOL.json'),
        source_manifest=read(ROOT/'source_manifest.json'),status_snapshot=read(ROOT/'STATUS.json'),
        meta=meta_summary(warnings),transfer=transfer_summary(warnings),cost=cost_summary(warnings),
        limitations=[
            'Eight chosen starting states from two existing Student seeds train one Teacher; this is a feasibility study, not a multi-Teacher statistical comparison.',
            'A single antithetic pair at each starting state is noisy. Same-policy restore/retrain and repeated development evaluations quantify different noise sources; PhysX hidden caches are not serialized.',
            'The development pool supplies Teacher feedback and state. Its learning curve is not an independent generalization estimate. Fixed formal-test results never train or select Teacher parameters.',
            'The transfer comparison uses one previously unseen Student seed47, identical initial actor/critic/optimizer and equal 10k budgets; both Teachers remain frozen. Any superiority claim needs additional held-out seeds.',
            'Command/state projection to eight dimensions is fixed and lossy. Missing feedback remains explicitly masked; it does not become a synthetic success or failure.',
            'Finite episode draws across32768 sequences create large empirical distribution noise even under Uniform. Proposal probabilities, realized draws and carried-over physical exposure are reported separately.',
            'Physical RMSEs are pooled over eligible measured steps; failures and invalid windows remain zero in the primary continuous development score. Safety rates are diagnostic native observations, not new reward terms.',
            'All recorded checkpoints remain preserved; report generation performs no training, policy selection, or modification of earlier experiments.'],warnings=warnings)
    summary['complete']=bool(summary['meta']['completed_teacher_updates']==8 and
        (args.stage=='meta' or all(r['result'] and r['result']['passed'] and r['result']['completed_updates']==10000 for r in summary['transfer']['arms'].values())))
    for arm,r in summary['transfer']['arms'].items():
        if r['frozen_teacher']['passed'] is False:warnings.append(f'Frozen Teacher identity mismatch for {arm}; investigate before any comparative conclusion.')
    summary['figures']=figures(summary,out)
    write(out/'summary.json',summary);(out/'report.txt').write_text(text_report(summary))
    # Every invocation retains its unique snapshot; canonical links only point
    # to the latest report and never replace checkpoints or experience records.
    shutil.copy2(out/'summary.json',ROOT/'results'/f'{args.stage}_summary.json')
    shutil.copy2(out/'report.txt',ROOT/'results'/f'{args.stage}_report.txt')
    print('V02_REPORT '+json.dumps(dict(stage=args.stage,complete=summary['complete'],summary=str(out/'summary.json'),figures=summary['figures']),allow_nan=False),flush=True)


if __name__=='__main__':main()
