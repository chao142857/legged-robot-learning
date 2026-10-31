"""Persistent targeted frozen-policy causal probes, inside new grant only."""
import argparse,fcntl,hashlib,json,os,pathlib,time,traceback
from gpu_budget import Budget,read,write
from stage3_pipeline import wait

def node_checkpoint(old,label):
    run=pathlib.Path(read(old/(label+'_RUN.json'))['run']);cfg=read(run/'config.json');result=read(run/'ppo/result.json')
    ck=run/'ppo/checkpoints'/pathlib.Path(result['checkpoint']['path']).name
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==result['checkpoint']['sha256']
    cp=dict(checkpoint=str(ck),checkpoint_sha256=result['checkpoint']['sha256'])
    if cfg.get('actor_recipe') is not None:cp['actor_recipe']=cfg['actor_recipe']
    return cp

def main(b):
    assert not (b.p/'COMPLETED.json').exists(),'Closed study is read-only'
    write(b.p/'CAUSAL_PROCESS.json',dict(pid=os.getpid(),CPU_coordinator_GPU_jobs_sequential=True))
    old=pathlib.Path(b.a['prior_closed_stage3_root']);assert (old/'COMPLETED.json').exists()
    parent=read(b.p/'CHECKPOINTS.json')['parent'];base=read(b.p/'P1_BASE_CONFIG.json');base.pop('probe',None)
    for k in ('actor_recipe','resume'):base.pop(k,None)
    base.update(mode='probe',entry='component_probe_entry.py',phase='P1',num_envs=64,
        diagnostic_parent_checkpoint=parent['checkpoint'],diagnostic_parent_sha256=parent['checkpoint_sha256'])
    groups=[dict(label='synergy600-seed22',cells=[('A',6181031,1.98),('T1',6281031,8.66),('X',6281031,8.70)]),
            dict(label='matchedNative200-seed22',cells=[('A',6181031,2.66)]),
            dict(label='synergy100-seed21',cells=[('A',6181031,7.16)])]
    plan=dict(groups=groups,modes=['current','parent_legs','parent_arm','parent_all'],intervention_start_s=1.,blend_s=.25,
        complete20s=True,original_task_reward_DR_PD_limits_and_safety_unchanged=True,
        inference_composition_diagnostic_only=True,PPO_updates=0,formal_seeds_not_used=True,
        hypothesis='Discriminate changed leg functions from arm-induced dynamics and their interaction; never infer cause from correlation alone.',
        stage3_unsafe_checkpoints_diagnosed_readonly_never_resumed_for_training=True)
    path=b.p/'CAUSAL_PLAN.json'
    if path.exists():assert read(path)==json.loads(json.dumps(plan)), 'Restart plan changed beyond JSON tuple/list encoding'
    else:write(path,plan)
    indexfile=b.p/'COMPONENT_PROBE_INDEX.json'
    if not indexfile.exists():write(indexfile,{})
    for group in groups:
        cp=node_checkpoint(old,group['label'])
        for protocol,seed,first_time in group['cells']:
            for mode in plan['modes']:
                assert b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']>450, 'Preserve final reserve before diagnostic start'
                label='causal-'+group['label']+'-'+protocol+'-'+str(seed)+'-'+mode
                index=read(indexfile)
                stopfile=b.p/'CAUSAL_MODE_STOP.json'
                stops=read(stopfile) if stopfile.exists() else {}
                if mode in stops and label not in index:
                    omittedfile=b.p/'CAUSAL_OMITTED_CELLS.json'
                    omitted=read(omittedfile) if omittedfile.exists() else []
                    if not any(row['label']==label for row in omitted):
                        omitted.append(dict(label=label,group=group['label'],protocol=protocol,seed=seed,mode=mode,reason=stops[mode],not_counted_as_success=True))
                        write(omittedfile,omitted)
                    continue
                cfg=dict(base,**cp,protocol=protocol,seed=seed,diagnostic_group=group['label'],
                    component_probe=dict(mode=mode,start_s=1.,blend_s=.25,first_known_failure_s=first_time))
                if label in index:run=pathlib.Path(index[label])
                else:
                    run=b.launch(cfg,label,timeout=300);index[label]=str(run);write(indexfile,index)
                write(b.p/'CAUSAL_STATE.json',dict(state='real_frozen_PhysX_probe',label=label,run=str(run),clock=b.meter()))
                wait(b,run)
                result=read(run/'evaluation/component_result.json')
                assert result['current_unchanged'] and result['parent_unchanged'] and not result['trained_policy_success']
            # Derive only completed current/intervention pairs. Native scores
            # and the full cold prefix are checked before moving to next cell.
            write(b.p/'CAUSAL_STATE.json',dict(state='CPU_native_metric_prefix_and_DR_check',protocol=protocol,seed=seed,group=group['label'],clock=b.meter()))
            from reduce_components import summarize
            summarize(b.p)
    result=read(b.p/'COMPONENT_PHYSICS.json')
    omitted=read(b.p/'CAUSAL_OMITTED_CELLS.json') if (b.p/'CAUSAL_OMITTED_CELLS.json').exists() else []
    assert len(result['rows'])==20-len(omitted) and len(result['comparisons'])==15-len(omitted)
    write(b.p/'CAUSAL_DONE.json',dict(planned_rows=20,rows=len(result['rows']),comparisons=len(result['comparisons']),omitted_unsafe_diagnostic_cells=omitted,mode_assessments=result['mode_assessments'],
        single_policy_success_claim=False,requires_evidence_selected_training_mechanism=True,clock=b.meter()))
    write(b.p/'CAUSAL_STATE.json',dict(state='complete_real_causal_probes',clock=b.meter()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();b=Budget(a.root)
    with (b.p/'causal.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b)
        except BaseException as error:
            write(b.p/'CAUSAL_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),clock=b.meter()));raise
