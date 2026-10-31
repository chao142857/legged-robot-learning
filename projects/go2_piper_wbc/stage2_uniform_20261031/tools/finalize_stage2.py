"""Independent CPU finalizer: freeze selection, new-seed native20s tests, restore, close SUM ledger."""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess,time,traceback
from gpu_budget import Budget,read,write,read_state
from pipeline import wait
from continue_pipeline import report

def final_eval(b,cfg,label):
    idx=read(b.p/'EVAL_INDEX.json')
    if label in idx:r=pathlib.Path(idx[label])
    else:r=b.launch(cfg,label,purpose='final',timeout=240);idx[label]=str(r);write(b.p/'EVAL_INDEX.json',idx)
    wait(b,r);s=read(r/'evaluation/summary.json');assert s['num_episodes']==64 and s['complete_episode_evaluation'] and s['actor_unchanged'] and s['critic_unchanged']

def select(curve,initial):
    """Choose on development physics; keep original registration as a separate result.

    An absolute change from a weak parent can flag a direction that improved
    toward gain=1. This is not a reason to hide that candidate from final tests.
    Neither this selection rule nor its diagnostic rewrites the registered gate.
    """
    baseline={(r['protocol'],r['seed']):r for r in curve['rows'] if r['label'].startswith('frozen1800-')}
    groups={}
    for r in curve['rows']:
        if r['label'].startswith('final-') or r['label'].startswith('frozen1800-'):continue
        groups.setdefault(r['checkpoint_sha256'],[]).append(r)
    candidates=[]
    for sha,rows in groups.items():
        xs=[r for r in rows if r['protocol']=='X']
        if len(xs)<2:continue
        safety=min(r['strict_safe']/r['n'] for r in rows)
        retained=all(all(r.get('retention_checks',{}).values()) for r in rows)
        quality_checks=[];direction_diagnostics=[]
        for r in rows:
            check=dict(r.get('retention_checks',{}));old=baseline.get((r['protocol'],r['seed']))
            if not check or old is None:quality_checks.append(False);continue
            for name in ('forward','backward','left','right'):
                if name not in check:continue
                parent_gain=old['directions'][name]['gain'];gain=r['directions'][name]['gain']
                diagnostic=parent_gain is not None and gain is not None and abs(gain-1)<=abs(parent_gain-1)+.15
                direction_diagnostics.append(dict(label=r['label'],direction=name,parent_gain=parent_gain,current_gain=gain,original_registered_check=check[name],distance_to_ideal_diagnostic=diagnostic))
                check[name]=diagnostic
            quality_checks.append(all(check.values()) and r['complete']/r['n']>=.95)
        score=min(r['X_after1s']['synchronous_gain'] for r in xs)
        full_development=all({r['protocol'] for r in rows if r['seed']==seed}=={'A','T0','T1','X'} for seed in (8101,8102))
        candidates.append(dict(checkpoint=xs[0]['checkpoint'],checkpoint_sha256=sha,min_safe_fraction=safety,retained=retained,development_physical_quality=all(quality_checks),complete_development_matrix=full_development,direction_diagnostics=direction_diagnostics,min_X_gain=score,development_labels=[r['label'] for r in rows]))
    acceptable=[x for x in candidates if x['development_physical_quality'] and x['complete_development_matrix'] and x['min_safe_fraction']>=.95]
    chosen=max(acceptable,key=lambda x:x['min_X_gain']) if acceptable else dict(initial,reason='No trained candidate retained frozen original health/safety; preserve initialization')
    return dict(candidate=chosen,all_development_candidates=candidates,selection_used_final_seeds=False,
        selection_rule='Best worst-seed X gain among full A/T0/T1/X development matrices with strict safety and physical-quality checks. Absolute parent-relative direction change remains an independent original registered gate.',
        registered_final_gates_unchanged=True,rule_version='development_selection_v2_before_final_seeds')

def main(b,source):
    write(b.p/'FINALIZER_PROCESS.json',dict(pid=os.getpid()))
    while True:
        m=b.meter();closing=(b.p/'RESEARCH_CLOSED.json').exists() or m['GPU_container_SUM_seconds']>=b.cap-b.reserve
        if closing and not m['active']:break
        write(b.p/'FINALIZER_STATE.json',dict(state='waiting_for_research_close_and_GPU_idle',clock=m));time.sleep(5)
    curve=report(b,source,'pre_final_selection');base=read(pathlib.Path(read(b.p/'FIRST_RUN.json')['run'])/'config.json')
    initial={k:base[k] for k in ('checkpoint','checkpoint_sha256')}
    selection_path=b.p/'FROZEN_SELECTION.json'
    if selection_path.exists():selection=read(selection_path)
    else:selection=select(curve,initial);write(selection_path,selection)
    candidate={k:selection['candidate'][k] for k in initial}
    for label,cp in [('initial',initial),('candidate',candidate)]:
        for seed in (381031,391031,401031):
            for protocol in ('A','T0','T1','X'):
                write(b.p/'FINALIZER_STATE.json',dict(state='independent_native20s',model=label,protocol=protocol,seed=seed,clock=b.meter()))
                final_eval(b,dict(base,**cp,seed=seed,protocol=protocol),'final-'+label+'-'+protocol+'-'+str(seed))
    curve=report(b,source,'final_matrix');rows=[r for r in curve['rows'] if r['label'].startswith('final-')]
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
        checks.append(dict(protocol=r['protocol'],seed=r['seed'],checks=c,passed=all(c.values())))
    write(b.p/'FINAL_MATRIX.json',dict(rows=rows,checks=checks,single_candidate_numeric_gates=all(c['passed'] for c in checks),original_two_consecutive_saved_nodes_requirement='Not replaced by one candidate test; report separately before declaring baseline PASS'))
    if b.p/'runs' in pathlib.Path(candidate['checkpoint']).parents:
        parent=pathlib.Path(candidate['checkpoint']).parents[2];cfg=dict(read(parent/'config.json'),**candidate,resume=True,updates=0)
        index=b.p/'FINAL_RESTORE_RUN.json'
        if index.exists():run=pathlib.Path(read(index)['run'])
        else:run=b.launch(cfg,'final-full-learner-restore',purpose='final',timeout=180);write(index,dict(run=str(run)))
        wait(b,run);assert read(run/'ppo/result.json')['passed']
    usage=[]
    for j in read(b.p/'LEDGER.json')['jobs']:
        run=pathlib.Path(j['run']);record=dict(run=str(run),mode=j['mode'],purpose=j['purpose'],cid=j['cid'],state=read_state(j['cid']))
        if (run/'ppo/result.json').exists():record['PPO']=read(run/'ppo/result.json')
        elif (run/'ppo/updates.jsonl').exists():
            updates=[json.loads(line) for line in (run/'ppo/updates.jsonl').read_text().splitlines()];record.update(completed_updates=len(updates),completed_PPO_transitions=sum(r['transitions'] for r in updates),observed_Adam_calls=sum(r['optimizer_steps_this_update'] for r in updates),partial_failure=read(run/'ppo/error.json') if (run/'ppo/error.json').exists() else None)
        usage.append(record)
    clock=b.meter();assert not clock['active'] and clock['GPU_container_SUM_seconds']<=b.cap
    write(b.p/'RESOURCES_FINAL.json',dict(clock=clock,jobs=usage,updated_counters_are_not_caps=True))
    write(b.p/'COMPLETED.json',dict(state='original_PPO_stage2_bounded_experiment_complete',clock=clock,selected=candidate,task_numeric_gates=all(c['passed'] for c in checks),formal_baseline_PASS=False,note='Conservative pending full gate/provenance review; no engineering-only PASS. No further GPU launch from this ledger.'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--source',required=True);v=a.parse_args();b=Budget(v.root)
    with (b.p/'finalize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b,v.source)
        except BaseException as e:write(b.p/'FINALIZER_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),clock=b.meter()));raise
