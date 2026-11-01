"""Two frozen PhysX cells after the current learner/saved-node follow-up.

Only a causal diagnostic: never a policy candidate, teacher or training update.
"""
import argparse, fcntl, hashlib, os, pathlib, time, traceback
import numpy as np
from gpu_budget import Budget, read, write
from stage3_pipeline import wait


def main(b):
    write(b.p/'REPAIR2_PROCESS.json',dict(pid=os.getpid(),independent_of_chat=True))
    while not (b.p/'SAVED500_DONE.json').exists():
        if (b.p/'RESEARCH_CLOSED.json').exists():
            write(b.p/'REPAIR2_SKIPPED.json',dict(reason='Research closed; no reopening or final-seed tuning'));return
        if (b.p/'SAVED500_ERROR.json').exists():raise RuntimeError('Preserved saved-node evaluation error needs repair')
        time.sleep(5)
    if (b.p/'RESEARCH_CLOSED.json').exists():return
    with (b.p/'learning.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (b.p/'FROZEN_SELECTION.json').exists() and not b.meter()['active']
        if b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']<500:
            write(b.p/'REPAIR2_SKIPPED.json',dict(reason='Preserve final reserve'));return
        nodes=read(b.p/'CANDIDATE_NODES.json')
        safe=next(n for n in nodes if n['label']=='legKL002-n200-seed22')
        bad=next(n for n in nodes if n['label']=='legKL002-n400-seed22')
        assert not safe['decision']['severe_physical_failures'] and bad['decision']['severe_physical_failures']
        native=pathlib.Path(read(b.p/'EVAL_INDEX.json')['legKL002-n400-seed22-A-6181031'])
        with np.load(native/'evaluation/trajectories.npz',allow_pickle=False) as z:
            points=np.argwhere(z['valid'].astype(bool)&(z['nonfoot_max_force']>1))
            assert len(points)>0
            t,e=map(int,points[0]);failure=(t+1)*.02
        assert failure>1.45,'A late leg-function intervention cannot diagnose an earlier failure'
        cfg=read(native/'config.json');cfg.update(mode='probe',entry='component_probe_entry.py',phase='P2',resume=False,
            diagnostic_parent_checkpoint=safe['checkpoint']['checkpoint'],diagnostic_parent_sha256=safe['checkpoint']['checkpoint_sha256'])
        plan=dict(current=bad['checkpoint'],safe_leg_function=safe['checkpoint'],protocol='A',seed=6181031,
            first_native_failure_s=failure,env=e,modes=['current','parent_legs'],start_s=1.,blend_s=.25,
            no_PPO_updates=True,no_inference_policy_success=True,all_original_physics_and_scores_unchanged=True)
        f=b.p/'REPAIR2_PLAN_FROZEN.json'
        if f.exists():assert read(f)==plan
        else:write(f,plan)
        runs=[]
        for mode in plan['modes']:
            cfg['component_probe']=dict(mode=mode,start_s=1.,blend_s=.25,first_known_failure_s=failure)
            receipt=b.p/('REPAIR2_'+mode+'_RUN.json')
            if receipt.exists():run=pathlib.Path(read(receipt)['run'])
            else:run=b.launch(cfg,'repair2-seed22-'+mode,timeout=300);write(receipt,dict(run=str(run),diagnostic_only=True))
            wait(b,run);runs.append(run)
        a=np.load(runs[0]/'evaluation/trajectories.npz',allow_pickle=False)
        c=np.load(runs[1]/'evaluation/trajectories.npz',allow_pickle=False)
        fields=['diag_actor_obs','component_current_mean','control_processed_target','control_joint_position',
            'control_joint_velocity','ee_cmd','base_cmd','kin_link0_pos','kin_link0_quat','component_contacts_history_w']
        prefix={k:bool(np.array_equal(a[k][:49],c[k][:49])) for k in fields}
        realized=read(runs[0]/'evaluation/component_realized_start.json')==read(runs[1]/'evaluation/component_realized_start.json')
        summaries=[read(r/'evaluation/summary.json') for r in runs]
        dynamics=[read(r/'evaluation/dynamic_metrics.json') for r in runs]
        native_summary=read(native/'evaluation/summary.json')
        reproduced=all(summaries[0][k]==native_summary[k] for k in ('safe_count','complete_count','joint_limit_violation_count','nonfoot_contact_count'))
        rows=[dict(run=str(r),mode=m,strict=s['safe_count'],complete=s['complete_count'],hard=s['joint_limit_violation_count'],
            nonfoot=s['nonfoot_contact_count'],EE_after1s_m=d['windows']['after1s']['EE_absolute_rmse_m'])
            for r,m,s,d in zip(runs,plan['modes'],summaries,dynamics)]
        supported=bool(realized and all(prefix.values()) and reproduced and rows[0]['nonfoot']>0 and
            rows[1]['hard']==rows[1]['nonfoot']==0 and rows[1]['complete']==64 and rows[1]['strict']>=61 and
            rows[1]['EE_after1s_m']<=rows[0]['EE_after1s_m']+.01)
        write(b.p/'REPAIR2_DONE.json',dict(hypothesis_supported=supported,rows=rows,native_scalar_reproduced=reproduced,
            initial_realization_equal=realized,first49_actual_control_prefix_equal=prefix,plan_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            diagnostic_only_not_single_policy_success=True,scope='Whole-leg function intervention, not a unique instantaneous calf-scalar cause or hidden PhysX restore.',clock=b.meter()))
        a.close();c.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();b=Budget(a.root)
    try:main(b)
    except BaseException as error:
        write(b.p/'REPAIR2_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),clock=b.meter()));raise
