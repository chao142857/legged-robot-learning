"""Conditional joint-function retention using the unchanged native trainer.

Run only if matched full-function restoration removes the actual contact.
Cover both common-parent seeds before optional later safe continuations.
"""
import argparse, fcntl, hashlib, os, pathlib, time, traceback
from gpu_budget import Budget, read, write
from stage3_pipeline import train, tests, decision, wait


def main(b):
    assert not (b.p/'COMPLETED.json').exists() and not (b.p/'RESEARCH_CLOSED.json').exists(), 'Closed research is read-only'
    extension=read(b.p/'ROBUST_NODE_EXTENSION_PREREGISTRATION.json');assert extension['extra_updates']==100 and extension['training_seed']==22
    plan_path=b.p/'STEADY_KL001_PLAN_FROZEN.json';plan=read(plan_path)
    assert plan['leg_weight']==.02 and plan['steady_mean_weight']>0 and plan['training_seeds']==[22,21]
    assert plan['only_changed_learning_factor']=='add_observable_steady_goal_full_mean_preservation'
    write(b.p/'STEADY_KL001_V2_PROCESS.json',dict(pid=os.getpid(),startup_state='Replay completed receipts before native continuation',independent_of_chat=True))
    while not (b.p/'REPAIR3V2_DONE.json').exists():
        if (b.p/'RESEARCH_CLOSED.json').exists():return
        if (b.p/'REPAIR3V2_ERROR.json').exists():raise RuntimeError('Preserved causal diagnostic error needs repair')
        time.sleep(5)
    if not any(v['mode']=='parent_all' and v['removes_contact'] for v in read(b.p/'REPAIR3V2_DONE.json')['checks']):
        write(b.p/'STEADY_KL001_NOT_ADMITTED.json',dict(reason='Matched full-function restoration did not support joint preservation; no blind training'));return
    assert not (b.p/'FROZEN_SELECTION.json').exists() and not (b.p/'RESEARCH_CLOSED.json').exists()
    with (b.p/'learning.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for job in b.meter()['active']:
            run=pathlib.Path(job['run']);assert job['purpose']=='research' and job['mode']=='train' and run.name.endswith(('steadyKL001-seed21-window-to2','steadyKL001-seed21-window-to100','steadyKL001-seed21-window-to200','steadyKL001-seed21-window-to400'))
            wait(b,run)  # Existing independent GPU process finishes; never restarted.
        marker=b.p/'LEARNING_REPAIR_NEEDED.json';archive=b.p/'REPAIR3_STOP_BEFORE_STEADYKL001.json'
        if marker.exists():
            original=marker.read_bytes()
            if archive.exists():assert archive.read_bytes()==original
            else:archive.write_bytes(original)
            assert archive.read_bytes()==original;marker.unlink()
        else:assert archive.exists(), 'A resumed coordinator must retain original admission/stop evidence'
        write(b.p/'ACTIVE_LEARNING_VARIANT.json',dict(pid=os.getpid(),recipe_id=plan['recipe_id'],plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest()))
        base=read(b.p/'P1_BASE_CONFIG.json');base.pop('probe',None);base.update(mode='eval',entry='evaluate_entry.py',phase='P3',num_envs=64)
        parent=read(b.p/'CHECKPOINTS.json')['parent'];assert parent['checkpoint_sha256']==plan['common_parent_sha256']
        objective=dict(schema='original210_fixed_leg_gaussian_KL_v1',weight=.02,anchor_sha256=parent['checkpoint_sha256'],steady_mean_weight=plan['steady_mean_weight'],steady_mean_schema='observable_steady_XYZ_full_mean_v1',steady_xyz_tolerance_m=1e-7)
        nodes=read(b.p/'CANDIDATE_NODES.json');states=[]

        def remaining():return b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']
        def write_once(path,value):
            if path.exists():assert read(path)==value
            else:write(path,value)
        def close_time(reason):write(b.p/'RESEARCH_CLOSED.json',dict(reason=reason,recipe_id=plan['recipe_id'],clock=b.meter()))
        def record(state,cp,run,n):
            label='steadyKL001-n'+str(n)+'-seed'+str(state['seed'])
            existing=next((v for v in nodes if v['label']==label),None)
            previous=state['branch'][-1]['decision'] if state['branch'] else None
            if existing is not None:
                assert existing['checkpoint']['checkpoint_sha256']==cp['checkpoint_sha256']
                assert existing['recipe_id']==plan['recipe_id'] and existing['actual_checkpoint_iteration']==1800+n
                node=existing;d=node['decision']
            else:
                rows=tests(b,base,cp,label,phase='P2' if n<=200 else 'P3');d=decision(rows,previous)
                control=next((v for v in nodes if v['recipe_id']=='arm_synergy_fixed_legKL_002' and v['training_seed']==state['seed'] and v['additional_actual_updates']==n),None)
                if control:d['single_factor_matched_002_control']=dict(label=control['label'],checkpoint=control['checkpoint'])
                node=dict(label=label,evaluation_prefix=label,recipe_id=plan['recipe_id'],experiment_id='steadyKL001-seed'+str(state['seed']),training_seed=state['seed'],additional_actual_updates=n,actual_checkpoint_iteration=1800+n,checkpoint=cp,decision=d,run=str(run))
                nodes.append(node);write(b.p/'CANDIDATE_NODES.json',nodes);write(b.p/(label+'_DECISION.json'),d)
                write(b.p/'LEARNING_STATE.json',dict(state='steadyKL001_full_physical_node',node=node,clock=b.meter()))
            state['branch'].append(node)
            if previous is not None:state['flat']=state['flat']+1 if d['learning_change']<.02 else 0
            if d['severe_physical_failures'] or state['flat']>=2:
                state['stopped']=True
                stop=b.p/('steadyKL001-seed'+str(state['seed'])+'-PHYSICAL_STOP.json')
                if not stop.exists():write(stop,dict(node=node,reason='Actual unsafe policy or two flat physical milestones; never resume this weight'))
            safe=lambda x:x['decision']['original_all_development_gates'] and not x['decision']['severe_physical_failures']
            state['qualified']=len(state['branch'])>=2 and all(safe(v) for v in state['branch'][-2:])
            return d

        # These native training/reset boundaries match the previous .02 recipe.
        # Extra300 evaluation reads a checkpoint already saved inside to400;
        # it cannot change the generated PPO data or400 policy.
        for seed in plan['training_seeds']:
            cfg=dict(read(b.p/'NATIVE_TRAIN_CONFIG.json'));cfg.update(parent);cfg.update(mode='train',entry='train_entry.py',phase='P2',
                resume=False,training_seed=seed,actor_recipe=read(b.p/'SYNERGY_RECIPE.json'),gradient_audit=True,leg_retention=objective,
                leg_anchor_checkpoint=parent['checkpoint'],leg_anchor_sha256=parent['checkpoint_sha256'])
            state=dict(seed=seed,cfg=cfg,current=dict(cfg),actual=0,branch=[],flat=0,stopped=False,qualified=False);states.append(state)
            for window in (2,98,100,200):
                # Fit full remaining evaluations as well as the useful learner.
                allowance=1100 if state['actual']==200 else 500
                receipt=b.p/('steadyKL001-seed'+str(seed)+'-window-to'+str(state['actual']+window)+'_RUN.json')
                if not receipt.exists() and remaining()<window*6.5+allowance:
                    close_time('A complete controlled learning/evaluation window cannot fit; final reserve protected');return
                target=state['actual']+window
                run,cp=train(b,dict(state['current'],updates=window),'steadyKL001-seed'+str(seed)+'-window-to'+str(target))
                result=read(run/'ppo/result.json');state['actual']=result['completed_updates']-1800
                if state['actual']!=target:
                    assert result['time_stopped'],'Unexpected partial native window'
                    close_time('Time-stopped partial checkpoint retained but not labeled as a completed physical milestone');return
                state['current']=dict(cfg,**cp,resume=True,phase='P3' if target>=200 else 'P2')
                if target==2:
                    write_once(b.p/('steadyKL001-seed'+str(seed)+'-SMOKE.json'),dict(run=str(run),restore=read(run/'ppo/restore-evidence.json'),zero_mean=read(run/'ppo/zero-mean-attachment.json'),result=result));continue
                if target==100:
                    write_once(b.p/('steadyKL001-seed'+str(seed)+'-TRAIN100.json'),dict(run=str(run),checkpoint=cp,physical_matrix_not_yet_open=True));continue
                if target==400:
                    middle=run/'ppo/checkpoints/cp_2100.pt';metadata=read(middle.with_suffix('.json'))
                    assert metadata['completed_updates']==2100 and hashlib.sha256(middle.read_bytes()).hexdigest()==metadata['sha256']
                    record(state,dict(cp,checkpoint=str(middle),checkpoint_sha256=metadata['sha256']),run,300)
                    if state['stopped']:break
                record(state,cp,run,target)
                if state['stopped']:break
            write_once(b.p/('steadyKL001-seed'+str(seed)+'-FIRST_COVERAGE.json'),dict(actual_training_updates=state['actual'],stopped=state['stopped'],qualified=state['qualified'],branch=state['branch']))

        # Balanced continuation: both seeds have their first controlled window
        # before spending optional time on safe400->500 etc. No lifetime quota.
        while any(not s['qualified'] and not s['stopped'] for s in states):
            for state in states:
                if state['qualified'] or state['stopped']:continue
                if remaining()<1300:
                    close_time('No complete safe100-update continuation and full matrix fit; final reserve protected');return
                target=state['actual']+100
                run,cp=train(b,dict(state['current'],updates=100),'steadyKL001-seed'+str(state['seed'])+'-window-to'+str(target))
                result=read(run/'ppo/result.json')
                if result['completed_updates']-1800!=target:
                    assert result['time_stopped'];close_time('Time-stopped partial continuation retained, no false milestone');return
                state['actual']=target;state['current']=dict(state['cfg'],**cp,resume=True,phase='P3')
                record(state,cp,run,target)
        qualified_before_extra=all(s['qualified'] and not s['stopped'] for s in states)
        base_pairs=[dict(training_seed=s['seed'],labels=[v['label'] for v in s['branch'][-2:]],checkpoint_sha256=[v['checkpoint']['checkpoint_sha256'] for v in s['branch'][-2:]]) for s in states]
        extra_receipt=b.p/'ROBUST_NODE_EXTRA_DONE.json'
        if qualified_before_extra and not extra_receipt.exists():
            state=next(s for s in states if s['seed']==22)
            near=state['actual']==400 and state['branch'][-2]['decision']['worst_seed_X_amplitude_m']<extension['adjacent_amplitude_trigger_m']
            if near and remaining()>=1300:
                run,cp=train(b,dict(state['current'],updates=100),'steadyKL001-seed22-window-to500')
                result=read(run/'ppo/result.json')
                if result['completed_updates']==2300:
                    d=record(state,cp,run,500)
                    write(extra_receipt,dict(new_training_updates=100,node=state['branch'][-1],base_qualified_pairs=base_pairs,unsafe_extra_never_resumed=bool(d['severe_physical_failures']),original_formal_gates_unchanged=True,clock=b.meter()))
                else:
                    assert result['time_stopped'];write(extra_receipt,dict(partial_checkpoint_retained_but_not_eligible=True,actual_completed_updates=result['completed_updates'],base_qualified_pairs=base_pairs,clock=b.meter()))
            else:write(extra_receipt,dict(skipped=True,reason='No near-boundary trigger or complete window does not fit final reserve',base_qualified_pairs=base_pairs,clock=b.meter()))
        if qualified_before_extra:
            write(b.p/'FINAL_RECIPE_ADMISSION.json',dict(recipe_ids=[plan['recipe_id']],reason='Both common-parent continuation seeds have consecutive safe full original development nodes',formal_seeds_unopened=True))
            write(b.p/'RESEARCH_CLOSED.json',dict(reason='Replicated original full passes and bounded robustness follow-up complete; freeze for unchanged formal acceptance',clock=b.meter()))
        else:
            write(b.p/'LEARNING_REPAIR_NEEDED.json',dict(owner='steadyKL001',reason='Evidence-admitted joint-function control ended without two-seed safe consecutive success; do not mechanically extend unsafe weights',states=states,clock=b.meter()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();b=Budget(a.root)
    try:main(b)
    except BaseException as error:
        write(b.p/'STEADY_KL001_V2_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),clock=b.meter()));raise
