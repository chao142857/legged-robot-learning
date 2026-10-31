"""Controlled coefficient-only native PPO continuation from common CP1800.

Reuses the existing collector, native PPO, full evaluator, receipt/resume and
time supervisor. No optimizer, task, sampler, reward or actor is reimplemented.
"""
import argparse, fcntl, hashlib, os, pathlib, traceback
from gpu_budget import Budget, read, write
from stage3_pipeline import train, tests, decision


def main(b, plan_path):
    plan_path=pathlib.Path(plan_path); assert plan_path.parent==b.p
    plan=read(plan_path); tag=plan['label_prefix']; recipe=plan['recipe_id']
    assert tag=='legKL002' and plan['single_changed_training_factor']=='fixed_leg_Gaussian_KL_weight_0.01_to_0.02'
    assert not (b.p/'FROZEN_SELECTION.json').exists() and not (b.p/'COMPLETED.json').exists()
    support=read(b.p/'REPAIR1_LEG_FUNCTION_CAUSAL_COMPARISON.json'); assert support['hypothesis_supported']
    parent=read(b.p/'CHECKPOINTS.json')['parent']; assert parent['checkpoint_sha256']==plan['common_parent_sha256']
    assert plan['training_seeds']==[21,22] and plan['leg_retention']['weight']==.02
    assert plan['leg_retention']['anchor_sha256']==parent['checkpoint_sha256']
    process=dict(pid=os.getpid(),recipe_id=recipe,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),independent_of_chat=True)
    write(b.p/(tag+'_PROCESS.json'),process); write(b.p/'ACTIVE_LEARNING_VARIANT.json',process)
    # Preserve the completed/stopped first recipe verbatim. Its failure is never
    # cleared from candidate nodes, physical curves or the original stop file.
    marker=b.p/'LEARNING_REPAIR_NEEDED.json'
    if marker.exists():
        archive=b.p/'REPAIR1_LEARNING_STOP_ORIGINAL.json'
        if archive.exists():assert archive.read_bytes()==marker.read_bytes()
        else:archive.write_bytes(marker.read_bytes())
        assert archive.read_bytes()==marker.read_bytes(); marker.unlink()
    write(b.p/'REPAIR1_CONTINUE.json',dict(plan=str(plan_path),process=process,old_stop_preserved=True,clock=b.meter()))
    base=read(b.p/'P1_BASE_CONFIG.json'); base.pop('probe',None); base.update(mode='eval',entry='evaluate_entry.py',phase='P2',num_envs=64)
    nodes=read(b.p/'CANDIDATE_NODES.json')
    for seed in plan['training_seeds']:
        cfg=dict(read(b.p/'NATIVE_TRAIN_CONFIG.json')); cfg.update(parent); cfg.update(mode='train',entry='train_entry.py',phase='P2',resume=False,
            training_seed=seed,actor_recipe=read(b.p/'SYNERGY_RECIPE.json'),gradient_audit=True,leg_retention=plan['leg_retention'],
            leg_anchor_checkpoint=parent['checkpoint'],leg_anchor_sha256=parent['checkpoint_sha256'])
        current=dict(cfg); actual=0; previous=None; plateau=0; branch=[]; experiment=tag+'-seed'+str(seed)
        while True:
            left=b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']
            if actual==0: window=2
            elif actual==2: window=98
            elif actual==100: window=100
            else:
                r=read(pathlib.Path(branch[-1]['run'])/'ppo/result.json'); seconds=r['training_loop_seconds']/r['updates']
                window=200 if left>200*seconds+700 else 100 if left>100*seconds+700 else 0
            if not window or left<700+window*6.5:
                write(b.p/'RESEARCH_CLOSED.json',dict(reason='Complete research window cannot fit remaining time; final3600s protected',plan=recipe,clock=b.meter())); return
            label=experiment+'-window-to'+str(actual+window)
            run,cp=train(b,dict(current,updates=window),label); result=read(run/'ppo/result.json'); actual=int(result['completed_updates'])-1800
            current=dict(cfg,**cp,resume=True,phase='P3' if actual>=200 else 'P2')
            if actual==2:
                write(b.p/(experiment+'-SMOKE.json'),dict(run=str(run),restore=read(run/'ppo/restore-evidence.json'),zero_mean=read(run/'ppo/zero-mean-attachment.json'),result=result)); continue
            if b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']<500:
                write(b.p/'RESEARCH_CLOSED.json',dict(reason='Preserve final reserve; newest checkpoint lacks complete development matrix and is ineligible',clock=b.meter())); return
            label=tag+'-n'+str(actual)+'-seed'+str(seed); rows=tests(b,base,cp,label,phase='P2' if actual<=200 else 'P3'); d=decision(rows,previous)
            matched=next((n for n in nodes if n['recipe_id']=='arm_synergy_fixed_legKL_001' and n['training_seed']==seed and n['additional_actual_updates']==actual),None)
            d['coefficient_only_matched_001_control']=None if matched is None else dict(label=matched['label'],checkpoint=matched['checkpoint'],decision={k:v for k,v in matched['decision'].items() if k not in ('rows','same_parent_seed_schedule_unconstrained_control')})
            node=dict(label=label,evaluation_prefix=label,recipe_id=recipe,experiment_id=experiment,training_seed=seed,
                additional_actual_updates=actual,actual_checkpoint_iteration=result['completed_updates'],checkpoint=cp,decision=d,run=str(run))
            if not any(n['checkpoint']['checkpoint_sha256']==cp['checkpoint_sha256'] for n in nodes):nodes.append(node)
            branch.append(node); write(b.p/'CANDIDATE_NODES.json',nodes); write(b.p/(label+'_DECISION.json'),d)
            write(b.p/'LEARNING_STATE.json',dict(state='variant_full_physical_milestone',node=node,clock=b.meter()))
            if d['severe_physical_failures']:
                write(b.p/(experiment+'-PHYSICAL_STOP.json'),dict(reason='Actual hard/contact/severe physical failure; this checkpoint is never resumed',node=node)); break
            if previous is not None: plateau=plateau+1 if d['learning_change']<.02 else 0
            if len(branch)>=2 and d['original_all_development_gates'] and branch[-2]['decision']['original_all_development_gates']:
                write(b.p/(experiment+'-DEVELOPMENT_PASS.json'),dict(nodes=branch[-2:],formal_PASS=False)); break
            if actual>=200 and d['worst_seed_X_gain']<.12 and d['worst_seed_X_amplitude_m']<.004: break
            if plateau>=2:
                write(b.p/(experiment+'-PLATEAU.json'),dict(reason='Two physical milestones without substantive X improvement',nodes=branch[-2:])); break
            previous=d
    if all((b.p/(tag+'-seed'+str(s)+'-DEVELOPMENT_PASS.json')).exists() for s in plan['training_seeds']):
        write(b.p/'FINAL_RECIPE_ADMISSION.json',dict(recipe_ids=[recipe],reason='Both common-parent continuation seeds have two complete safe original-gate development nodes',formal_seeds_unopened=True))
        write(b.p/'RESEARCH_CLOSED.json',dict(reason='Replicated coefficient-only recipe has consecutive full development passes; proceed to unchanged formal acceptance',recipe_id=recipe,clock=b.meter())); return
    write(b.p/'LEARNING_REPAIR_NEEDED.json',dict(reason='Coefficient-only controlled branches stopped/completed without two-seed consecutive success; diagnose actual evidence before another factor',recipe_id=recipe,clock=b.meter()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True); parser.add_argument('--plan',required=True); args=parser.parse_args(); b=Budget(args.root)
    with (b.p/'learning.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b,args.plan)
        except BaseException as error:
            write(b.p/'LEARNING_VARIANT_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),clock=b.meter())); raise
