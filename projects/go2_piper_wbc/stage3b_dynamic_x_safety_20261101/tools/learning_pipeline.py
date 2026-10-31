"""Evidence-admitted single-factor native PPO windows, no lifetime count quota."""
import argparse,fcntl,hashlib,json,os,pathlib,time,traceback
from gpu_budget import Budget,read,write
from stage3_pipeline import train,tests,decision

def main(b):
    assert not (b.p/'COMPLETED.json').exists(),'Closed study is read-only'
    write(b.p/'LEARNING_PROCESS.json',dict(pid=os.getpid(),independent_of_chat=True))
    while not (b.p/'CAUSAL_DONE.json').exists():
        if (b.p/'CAUSAL_ERROR.json').exists():
            write(b.p/'LEARNING_STATE.json',dict(state='waiting_for_actual_causal_interface_repair',no_GPU_training_started=True))
        time.sleep(3)
    physics=read(b.p/'COMPONENT_PHYSICS.json');rows=physics['rows']
    # Minimal support: replacing only legs removes at least one actual event
    # while preserving an excited dynamic X response. This does not claim that
    # full frozen-leg composition is safe: retain all18trainable functions.
    paired=[]
    for old in [r for r in rows if r['mode']=='current']:
        changed=next((r for r in rows if (r['group'],r['protocol'],r['seed'],r['mode'])==(old['group'],old['protocol'],old['seed'],'parent_legs')),None)
        if changed and changed['hard']+changed['nonfoot']<old['hard']+old['nonfoot']:paired.append((old,changed))
    dynamic=[v for _,v in paired if v['protocol'] in ('T1','X') and v['X']['synchronous_gain']>=.35 and v['X']['actual_fitted_amplitude_m']>=.008 and v['X']['correlation']>=.5]
    assert paired and dynamic and physics['comparisons'],'Physical causal support missing; no blind regularization'
    parent=read(b.p/'CHECKPOINTS.json')['parent']
    objective=dict(schema='original210_fixed_leg_gaussian_KL_v1',weight=.01,anchor_sha256=parent['checkpoint_sha256'])
    plan=dict(recipe_id='arm_synergy_fixed_legKL_001',single_new_factor='fixed original-CP1800 leg Gaussian KL in native PPO loss',objective=objective,
        common_parent=parent,training_seeds=[22,21],all18D_actor_parameters_trainable=True,original27_rewards_Uniform30_70_LR_entropy_PD_DR_unchanged=True,
        first_slices=[2,98,100],then_actual_throughput_bounded=True,model_data_counters_not_caps=True,final_seeds_unopened=True,
        calibration='CPU_LEG_KL_SCALE_AUDIT:0.01 gives approximately25%-38% of native actor preclip-gradient scale at old200/400 fixed-state probes. Different batches; not exact native PPO gradient attribution.',
        physical_support_cells=[dict(group=o['group'],protocol=o['protocol'],seed=o['seed'],before_hard=o['hard'],before_nonfoot=o['nonfoot'],after_hard=v['hard'],after_nonfoot=v['nonfoot']) for o,v in paired],
        interpretation='Soft leg-function preservation while allowing actual joint leg/arm coadaptation. Frozen diagnostic composition is never a trained-policy success.')
    f=b.p/'LEG_KL001_PLAN_FROZEN.json'
    if f.exists():assert read(f)==plan
    else:write(f,plan)
    base=read(b.p/'P1_BASE_CONFIG.json');base.pop('probe',None);base.update(mode='eval',entry='evaluate_entry.py',phase='P2',num_envs=64)
    allnodes=read(b.p/'CANDIDATE_NODES.json') if (b.p/'CANDIDATE_NODES.json').exists() else []
    oldroot=pathlib.Path(b.a['prior_closed_stage3_root'])
    for seed in plan['training_seeds']:
        cfg=dict(read(b.p/'NATIVE_TRAIN_CONFIG.json'),phase='P2',mode='train',entry='train_entry.py',
            training_seed=seed,actor_recipe=read(b.p/'SYNERGY_RECIPE.json'),gradient_audit=True,leg_retention=objective,
            leg_anchor_checkpoint=parent['checkpoint'],leg_anchor_sha256=parent['checkpoint_sha256'])
        current=dict(cfg);actual=0;previous=None;plateau=0;branch=[];experiment='legKL001-seed'+str(seed)
        while True:
            left=b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']
            if actual==0:window=2
            elif actual==2:window=98
            elif actual==100:window=100
            else:
                last_result=read(pathlib.Path(branch[-1]['run'])/'ppo/result.json');seconds=last_result['training_loop_seconds']/last_result['updates']
                window=200 if left>200*seconds+700 else 100 if left>100*seconds+700 else 0
            if not window or left<700+window*6.5:
                write(b.p/'RESEARCH_CLOSED.json',dict(reason='No complete useful training+development window fits remaining research time; final reserve protected',clock=b.meter()));return
            label=experiment+'-window-to'+str(actual+window)
            run,cp=train(b,dict(current,updates=window),label)
            result=read(run/'ppo/result.json');actual=int(result['completed_updates'])-1800
            current=dict(cfg,**cp,resume=True,phase='P3' if actual>=200 else 'P2')
            if actual==2:
                write(b.p/(experiment+'-SMOKE.json'),dict(run=str(run),zero_Gaussian_initialization=read(run/'ppo/zero-mean-attachment.json'),restore=read(run/'ppo/restore-evidence.json'),native_PPO_result=result));continue
            if b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']<500:
                write(b.p/'RESEARCH_CLOSED.json',dict(reason='Latest time-stopped checkpoint preserved unselected; full development matrix cannot fit research reserve',clock=b.meter()));return
            eval_label='legKL001-n'+str(actual)+'-seed'+str(seed)
            evaluated=tests(b,base,cp,eval_label,phase='P2' if actual<=200 else 'P3');d=decision(evaluated,previous)
            controls=read(oldroot/'SEED22_NODES.json') if seed==22 else read(oldroot/'CANDIDATE_NODES.json')
            matched=next((n for n in controls if n['label']=='synergy'+str(actual)+'-seed'+str(seed)),None)
            d['same_parent_seed_schedule_unconstrained_control']=matched
            node=dict(label=eval_label,evaluation_prefix=eval_label,recipe_id=plan['recipe_id'],experiment_id=experiment,training_seed=seed,
                additional_actual_updates=actual,actual_checkpoint_iteration=result['completed_updates'],checkpoint=cp,decision=d,run=str(run))
            if not any(n['checkpoint']['checkpoint_sha256']==cp['checkpoint_sha256'] for n in allnodes):allnodes.append(node)
            branch.append(node);write(b.p/'CANDIDATE_NODES.json',allnodes);write(b.p/(eval_label+'_DECISION.json'),d)
            write(b.p/'LEARNING_STATE.json',dict(state='milestone_physically_evaluated',node=node,clock=b.meter()))
            if d['severe_physical_failures']:
                write(b.p/(experiment+'-PHYSICAL_STOP.json'),dict(reason='Actual hard/contact/severe safety event; never resume this checkpoint',node=node));break
            if previous is not None:plateau=plateau+1 if d['learning_change']<.02 else 0
            if len(branch)>=2 and d['original_all_development_gates'] and branch[-2]['decision']['original_all_development_gates']:
                write(b.p/(experiment+'-DEVELOPMENT_PASS.json'),dict(nodes=branch[-2:],formal_PASS=False));break
            if actual>=200 and d['worst_seed_X_gain']<.12 and d['worst_seed_X_amplitude_m']<.004:
                write(b.p/(experiment+'-NO_DIRECTIONAL_LEARNING.json'),dict(node=node,reason='First200 lacks a meaningful physical X direction; do not append same-recipe windows'));break
            if plateau>=2:
                write(b.p/(experiment+'-PLATEAU.json'),dict(nodes=branch[-2:],reason='Two full physical milestones without substantive X gain'));break
            previous=d
        # Independent continuationseed21 always starts from immutable common
        # CP1800, never from a failed seed22 weight or a different seed's RNG.
    if all((b.p/('legKL001-seed'+str(seed)+'-DEVELOPMENT_PASS.json')).exists() for seed in plan['training_seeds']):
        write(b.p/'RESEARCH_CLOSED.json',dict(reason='Both independent common-parent continuation seeds have two consecutive full development nodes passing unchanged gates; freeze for new formal seeds',clock=b.meter()));return
    write(b.p/'LEARNING_REPAIR_NEEDED.json',dict(reason='Two preregistered continuation-seed branches completed/stopped. Review actual physical outcomes before another mechanism.',nodes=allnodes,clock=b.meter()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();b=Budget(a.root)
    with (b.p/'learning.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b)
        except BaseException as error:write(b.p/'LEARNING_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),clock=b.meter()));raise
