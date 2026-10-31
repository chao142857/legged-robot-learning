"""Independent CPU closer. No formal seed is opened before frozen selection."""
import argparse,fcntl,hashlib,json,os,pathlib,time,traceback
from gpu_budget import Budget,read,write,read_state
from stage3_pipeline import evaluate,wait,report
FINAL_SEEDS=(511031,521031,531031)

def selection(b):
    curve=report(b,'before_final');groups={}
    for r in curve['rows']:
        if r['label'].startswith(('final-','frozen1800-','nativeControl')):continue
        groups.setdefault(r['checkpoint_sha256'],[]).append(r)
    options=[]
    for sha,rows in groups.items():
        if len(rows)!=8 or any({r['protocol'] for r in rows if r['seed']==seed}!={'A','T0','T1','X'} for seed in (6181031,6281031)):continue
        xs=[r for r in rows if r['protocol']=='X'];safe=all(r['nonfoot']==0 and r['joint_limit']==0 and r['strict_safe']/r['n']>=.95 and r['complete']/r['n']>=.95 for r in rows)
        options.append(dict(checkpoint=rows[0]['checkpoint'],checkpoint_sha256=sha,actor_recipe=read(pathlib.Path(rows[0]['run'])/'config.json').get('actor_recipe'),
            min_X_gain=min(r['X_after1s']['synchronous_gain'] for r in xs),safe=safe,registered_retention=all(all(r.get('retention_checks',{}).values()) for r in rows),labels=[r['label'] for r in rows]))
    acceptable=[r for r in options if r['safe']]
    parent=read(b.p/'CHECKPOINTS.json')['parent']
    chosen=max(acceptable,key=lambda r:r['min_X_gain']) if acceptable else dict(parent,reason='No new safe full-matrix candidate; keep immutable parent')
    # Freeze the adjacent node on this exact recipe/seed lineage, including
    # the matched native control. Never combine capabilities across branches.
    import re
    match=re.match(r'^(synergy|matchedNative)([0-9]+)-seed([0-9]+)-',chosen.get('labels',[''])[0])
    earlier=[]
    if match:
        branch,count,seed=match.group(1),int(match.group(2)),match.group(3)
        for option in options:
            other=re.match(r'^(synergy|matchedNative)([0-9]+)-seed([0-9]+)-',option['labels'][0])
            if other and other.group(1)==branch and other.group(3)==seed and int(other.group(2))<count:earlier.append((int(other.group(2)),option))
    previous=max(earlier,key=lambda item:item[0])[1] if earlier else None
    adjacent=None if previous is None else dict(checkpoint={k:v for k,v in previous.items() if k in ('checkpoint','checkpoint_sha256','actor_recipe') and v is not None},labels=previous['labels'],frozen_before_final_seeds=True)
    return dict(candidate=chosen,adjacent=adjacent,all_development_candidates=options,used_final_seeds=False,rule='Before formal seeds: best worst-development-seed X gain among full8 native20s matrices, no actual nonfoot or hard-limit event, strict and completion>=.95. Original retention gates remain mandatory formal gates; selection is not PASS.')

def main(b):
    write(b.p/'FINALIZER_PROCESS.json',dict(pid=os.getpid()))
    while True:
        clock=b.meter()
        if ((b.p/'RESEARCH_CLOSED.json').exists() or clock['GPU_container_SUM_seconds']>=b.cap-b.reserve) and not clock['active']:break
        write(b.p/'FINALIZER_STATE.json',dict(state='waiting_for_research_close_GPU_idle',clock=clock));time.sleep(5)
    base=read(b.p/'P1_BASE_CONFIG.json');source=str(pathlib.Path(read(b.p/'CPU_RESPONSE_CONFIG.json')['kinematics_source']).parents[1]);base.pop('probe',None);base.update(mode='eval',entry='evaluate_entry.py',phase='P4',num_envs=64)
    path=b.p/'FROZEN_SELECTION.json'
    if not path.exists():write(path,selection(b))
    selected=read(path);candidate={k:v for k,v in selected['candidate'].items() if k in ('checkpoint','checkpoint_sha256','actor_recipe') and v is not None};parent=read(b.p/'CHECKPOINTS.json')['parent']
    for name,cp in [('initial',parent),('candidate',candidate)]:
        for seed in FINAL_SEEDS:
            for protocol in ('A','T0','T1','X'):
                write(b.p/'FINALIZER_STATE.json',dict(state='formal_native20s',model=name,seed=seed,protocol=protocol,clock=b.meter()))
                evaluate(b,dict(base,**cp,seed=seed,protocol=protocol),'final-'+name+'-'+protocol+'-'+str(seed),purpose='final')
    curve=report(b,'final_matrix');rows=[r for r in curve['rows'] if r['label'].startswith('final-')]
    # Relative gates use the new paired initialization, without changing any historical score.
    spec=read(pathlib.Path(source)/'recipe_b0/spec.json');g=spec['learning_trend_gate']['CP2200'];ret=spec['retention_gate'];checks=[]
    for r in [r for r in rows if r['label'].startswith('final-candidate-')]:
        old=next(v for v in rows if v['label']=='final-initial-'+r['protocol']+'-'+str(r['seed']))
        c=dict(full20=r['complete']/r['n']>=ret['completion_fraction_min'],strict_safety=r['strict_safe']/r['n']>=ret['strict_safe_fraction_min'] and r['strict_safe']/r['n']>=old['strict_safe']/old['n']-ret['strict_safe_fraction_drop_vs_parent_max'],EE_retained=r['EE_after1s']<=old['EE_after1s']+ret['A_T0_T1_EE_RMSE_increase_max_m'])
        health=r['native_health']['checks'];c['native_health']=r['native_health']['gate_pass'] if r['protocol']!='T0' else all(v for k,v in health.items() if not k.startswith(('forward/','backward/','left/','right/')))
        if r['protocol'] in ('A','T1'):
            c['direction_retained']=all(abs(r['directions'][d]['gain']-old['directions'][d]['gain'])<=ret['A_T1_direction_gain_change_max'] for d in ('forward','backward','left','right'))
        if r['protocol']=='T1':
            c['YZ_retained']=all(r['axes_after1s'][d]['synchronous_gain']>=old['axes_after1s'][d]['synchronous_gain']-ret['T1_YZ_sync_gain_drop_max'] for d in ('y','z'))
        if r['protocol']=='T0':
            c['drift_retained']=r['base_drift']['mean']<=old['base_drift']['mean']+ret['T0_base_drift_mean_increase_max_m'] and r['base_drift']['p95']<=old['base_drift']['p95']+ret['T0_base_drift_p95_increase_max_m']
        if r['protocol']=='X':
            x=r['X_after1s'];y=old['X_after1s'];c.update(r['absolute_X_gate'])
            c.update(gain_improvement=x['synchronous_gain']>=y['synchronous_gain']+g['X_gain_improvement_vs_CP1800_min'],correlation_improvement=x['correlation']>=y['correlation']+g['X_correlation_improvement_min'],demeaned_RMSE=x['demeaned_rmse_m']<=y['demeaned_rmse_m']*g['X_demeaned_RMSE_fraction_of_parent_max'],absolute_RMSE=x['absolute_rmse_m']<=y['absolute_rmse_m']+g['X_absolute_RMSE_increase_max_m'])
        # User's preregistered branch-stop rule applies to final acceptance too:
        # never hide an actual hard/contact event inside a passing fraction.
        # Native scores, their thresholds and all original checks above stay.
        c.update(user_zero_actual_hard_events=r['joint_limit']==0,user_zero_actual_nonfoot_events=r['nonfoot']==0)
        checks.append(dict(protocol=r['protocol'],seed=r['seed'],checks=c,passed=all(c.values())))
    write(b.p/'FINAL_MATRIX.json',dict(rows=rows,checks=checks,single_candidate_numeric_gates=all(c['passed'] for c in checks),original_two_consecutive_saved_nodes_requirement='Primary final candidate tested here; adjacent saved-node verification is recorded separately and required for PASS'))
    # Verify an adjacent node on the same final protocols only if frozen before
    # opening final seeds. A single checkpoint can never satisfy stability alone.
    nodes=read(b.p/'CANDIDATE_NODES.json') if (b.p/'CANDIDATE_NODES.json').exists() else []
    ix=next((i for i,n in enumerate(nodes) if n['checkpoint']['checkpoint_sha256']==candidate['checkpoint_sha256']),None)
    adjacent=selected.get('adjacent')
    adjacent_checks=[]
    if adjacent is not None:
        for seed in FINAL_SEEDS:
            for protocol in ('A','T0','T1','X'):
                evaluate(b,dict(base,**adjacent['checkpoint'],seed=seed,protocol=protocol),'final-adjacent-'+protocol+'-'+str(seed),purpose='final')
        curve=report(b,'final_adjacent')
        # Reuse exactly the same registered check implementation on this node by
        # replacing the candidate rows in a local derived calculation only.
        for r in [r for r in curve['rows'] if r['label'].startswith('final-adjacent-')]:
            old=next(v for v in curve['rows'] if v['label']=='final-initial-'+r['protocol']+'-'+str(r['seed']))
            c=dict(full20=r['complete']/r['n']>=ret['completion_fraction_min'],strict_safety=r['strict_safe']/r['n']>=ret['strict_safe_fraction_min'] and r['strict_safe']/r['n']>=old['strict_safe']/old['n']-ret['strict_safe_fraction_drop_vs_parent_max'],EE_retained=r['EE_after1s']<=old['EE_after1s']+ret['A_T0_T1_EE_RMSE_increase_max_m'])
            health=r['native_health']['checks'];c['native_health']=r['native_health']['gate_pass'] if r['protocol']!='T0' else all(v for k,v in health.items() if not k.startswith(('forward/','backward/','left/','right/')))
            if r['protocol'] in ('A','T1'):c['direction_retained']=all(abs(r['directions'][d]['gain']-old['directions'][d]['gain'])<=ret['A_T1_direction_gain_change_max'] for d in ('forward','backward','left','right'))
            if r['protocol']=='T1':c['YZ_retained']=all(r['axes_after1s'][d]['synchronous_gain']>=old['axes_after1s'][d]['synchronous_gain']-ret['T1_YZ_sync_gain_drop_max'] for d in ('y','z'))
            if r['protocol']=='T0':c['drift_retained']=r['base_drift']['mean']<=old['base_drift']['mean']+ret['T0_base_drift_mean_increase_max_m'] and r['base_drift']['p95']<=old['base_drift']['p95']+ret['T0_base_drift_p95_increase_max_m']
            if r['protocol']=='X':
                x=r['X_after1s'];y=old['X_after1s'];c.update(r['absolute_X_gate']);c.update(gain_improvement=x['synchronous_gain']>=y['synchronous_gain']+g['X_gain_improvement_vs_CP1800_min'],correlation_improvement=x['correlation']>=y['correlation']+g['X_correlation_improvement_min'],demeaned_RMSE=x['demeaned_rmse_m']<=y['demeaned_rmse_m']*g['X_demeaned_RMSE_fraction_of_parent_max'],absolute_RMSE=x['absolute_rmse_m']<=y['absolute_rmse_m']+g['X_absolute_RMSE_increase_max_m'])
            c.update(user_zero_actual_hard_events=r['joint_limit']==0,user_zero_actual_nonfoot_events=r['nonfoot']==0)
            adjacent_checks.append(dict(protocol=r['protocol'],seed=r['seed'],checks=c,passed=all(c.values())))
    write(b.p/'CONSECUTIVE_NODE_FINAL.json',dict(adjacent=adjacent,checks=adjacent_checks,passed=len(adjacent_checks)==12 and all(c['passed'] for c in adjacent_checks),selection_not_changed_after_final_seeds=True))
    if b.p/'runs' in pathlib.Path(candidate['checkpoint']).parents:
        run_parent=pathlib.Path(candidate['checkpoint']).parents[2];cfg=dict(read(run_parent/'config.json'),**candidate,resume=True,updates=0,phase='P4')
        receipt=b.p/'FINAL_RESTORE_RUN.json'
        if receipt.exists():run=pathlib.Path(read(receipt)['run'])
        else:run=b.launch(cfg,'final-native-learner-restore',purpose='final',timeout=180);write(receipt,dict(run=str(run)))
        wait(b,run);assert read(run/'ppo/result.json')['passed']
    actual_slots={r['run']:r['simulator_transition_slots'] for r in report(b,'resource_counts')['rows']}
    resources=[];unmeasured_partial=[];totals=dict(PPO_updates=0,Adam_calls=0,training_transitions_used=0,eval_transition_slots=0,probe_transition_slots=0,restore_inference_transition_slots=0,training_or_restore_reset_transition_slots=0)
    for j in read(b.p/'LEDGER.json')['jobs']:
        run=pathlib.Path(j['run']);row=dict(run=str(run),mode=j['mode'],phase=j['phase'],cid=j['cid'],state=read_state(j['cid']))
        updates=[json.loads(x) for x in (run/'ppo/updates.jsonl').read_text().splitlines()] if (run/'ppo/updates.jsonl').exists() else []
        err=read(run/'ppo/error.json') if (run/'ppo/error.json').exists() else None
        calls=err['optimizer_calls_observed'] if err else sum(r['optimizer_steps_this_update'] for r in updates)
        totals['PPO_updates']+=len(updates);totals['Adam_calls']+=calls;totals['training_transitions_used']+=sum(r['transitions'] for r in updates)
        row.update(PPO_updates=len(updates),Adam_calls=calls,training_transitions_used=sum(r['transitions'] for r in updates),error=err)
        if (run/'ppo/result.json').exists():
            result=read(run/'ppo/result.json');n=read(run/'config.json')['num_envs']
            row['reset_reward_compute_transition_slots']=result.get('reset_control_steps',0)*n
            totals['training_or_restore_reset_transition_slots']+=row['reset_reward_compute_transition_slots']
            if result.get('kind')=='final_checkpoint_restore_zero_updates':
                assert result['updates']==0
                row['restore_inference_transition_slots']=result.get('restore_inference_transition_slots',result['inference_steps']*n)
                totals['restore_inference_transition_slots']+=row['restore_inference_transition_slots']
        s=run/'evaluation/summary.json'
        if s.exists():
            s=read(s);row.update(evaluation_episodes=s['num_episodes'],evaluation_requested_steps=s['requested_steps'])
            if j['mode']=='probe':
                row['diagnostic_requested_transition_slots']=s['num_episodes']*s['requested_steps'];totals['probe_transition_slots']+=row['diagnostic_requested_transition_slots']
            else:row['actual_transition_slots']=actual_slots[str(run)];totals['eval_transition_slots']+=row['actual_transition_slots']
        elif j['mode'] in ('eval','probe'):
            unmeasured_partial.append(dict(run=str(run),mode=j['mode'],GPU_cost_included=True,partial_transition_count='Not recorded reliably by failed native evaluator; not silently counted as zero'))
        resources.append(row)
    clock=b.meter();assert not clock['active'] and clock['GPU_container_SUM_seconds']<=b.cap
    finalpass=len(checks)==12 and all(c['passed'] for c in checks) and len(adjacent_checks)==12 and all(c['passed'] for c in adjacent_checks)
    write(b.p/'RESOURCES_FINAL.json',dict(clock=clock,totals=totals,jobs=resources,unmeasured_partial_evaluations=unmeasured_partial,update_counters_not_caps=True,old_Stage2_pretraining_and_control_cost_not_recharged=True))
    write(b.p/'COMPLETED.json',dict(state='Stage3 original PPO Uniform bounded experiment closed',clock=clock,selected=candidate,original_baseline_PASS=finalpass,primary_checks_passed=all(c['passed'] for c in checks),consecutive_nodes_passed=len(adjacent_checks)==12 and all(c['passed'] for c in adjacent_checks),SuperMimic_used=False,SGS_ready=finalpass))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);v=a.parse_args();b=Budget(v.root)
    with (b.p/'finalize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b)
        except BaseException as e:write(b.p/'FINALIZER_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),clock=b.meter()));raise
