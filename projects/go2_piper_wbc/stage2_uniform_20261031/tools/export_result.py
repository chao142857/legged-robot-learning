"""Offline export of completed native PPO receipts; never chooses or trains a policy."""
import argparse,datetime,hashlib,json,pathlib

def read(path):return json.loads(pathlib.Path(path).read_text())

def export(delivery,compact,roots):
    states=delivery['states']
    completed=states['COMPLETED']['value']
    resources=states['RESOURCES_FINAL']['value']
    matrix=states['FINAL_MATRIX']['value']
    selected=states['FROZEN_SELECTION']['value']['candidate']
    restored=delivery['FINAL_RESTORE_RUN_result']
    assert completed['state']=='original_PPO_stage2_bounded_experiment_complete'
    assert not resources['clock']['active']
    assert resources['clock']['GPU_container_SUM_seconds']<=14400
    assert restored['passed'] and restored['updates']==0 and restored['learner_adam_unchanged']
    assert restored['loaded_checkpoint_sha256']==selected['checkpoint_sha256']
    rows=matrix['rows']
    assert len(rows)==24 and len({r['label'] for r in rows})==24
    compact_final={r['case']:r for r in compact['physics'] if r['case'].startswith('final-')}
    assert set(compact_final)=={r['label'] for r in rows},'Refresh completed compact evidence before export'
    assert all(compact_final[r['label']]['trajectory_sha256']==r['raw_trajectory_sha256'] for r in rows)
    for model in ('initial','candidate'):
        assert {(r['protocol'],r['seed']) for r in rows if r['label'].startswith('final-'+model+'-')}=={(p,s) for p in ('A','T0','T1','X') for s in (381031,391031,401031)}
    jobs=resources['jobs'];training=[];evaluations=[];restore_slots=0;job_usage=[]
    for job in jobs:
        run=pathlib.Path(job['run']).name
        state=job['state'];assert not state['Running']
        def timestamp(s):return datetime.datetime.fromisoformat(s[:26]+'+00:00').timestamp()
        seconds=timestamp(state['FinishedAt'])-timestamp(state['StartedAt'])
        job_usage.append(dict(run=run,mode=job['mode'],purpose=job['purpose'],exit_code=state['ExitCode'],GPU_container_seconds=seconds))
        if 'PPO' in job:
            result=job['PPO']
            if result['updates']:
                training.append(dict(run=run,new_updates=result['updates'],new_rollout_transitions=result['new_transitions'],
                    Adam_minibatches=result['optimizer_steps'],parent_completed_updates=result['parent_completed_updates'],
                    completed_updates=result['completed_updates'],checkpoint_sha256=result['checkpoint']['sha256'],
                    loop_seconds=result['training_loop_seconds'],measured_loop_transitions_per_second=result['full_ppo_loop_transitions_per_second'],
                    actor_changed=result['actor_changed'],critic_changed=result['critic_changed'],Adam_changed=result['adam_changed'],
                    finite=result['finite_checks_passed'],KL_min=result['kl_min'],KL_max=result['kl_max']))
            else:restore_slots+=result.get('inference_steps',0)*result['identity']['num_envs']
    for r in rows:
        evaluations.append(dict(case=r['label'],protocol=r['protocol'],seed=r['seed'],n=r['n'],complete=r['complete'],strict_safe=r['strict_safe'],
            native_joint_5cm=r['joint_5cm'],nonfoot=r['nonfoot'],overspeed=r['overspeed'],joint_limit=r['joint_limit'],
            native_terminations=r['terminations'],
            full20=r['full20'],orientation_RMS_rad=r['orientation_RMS_rad'],after1s_X=r['X_after1s'],
            directions=r['directions'],base_drift=r['base_drift'],native_health=r['native_health'],
            checkpoint_sha256=r['checkpoint_sha256'],raw_trajectory_sha256=r['raw_trajectory_sha256']))
    changes=[];direction_changes=[]
    for r in rows:
        if r['protocol'] not in ('A','T1') or not r['label'].startswith('final-candidate-'):continue
        old=next(x for x in rows if x['label']=='final-initial-'+r['protocol']+'-'+str(r['seed']))
        for name in ('forward','backward','left','right'):
            before=old['directions'][name]['gain'];after=r['directions'][name]['gain']
            direction_changes.append(dict(protocol=r['protocol'],seed=r['seed'],direction=name,initial_gain=before,candidate_gain=after,
                original_absolute_change_check=abs(after-before)<=.15,
                distance_to_unit_gain_change=abs(after-1)-abs(before-1),diagnostic_only=True))
    for r in rows:
        if r['protocol']!='X' or not r['label'].startswith('final-candidate-'):continue
        old=next(x for x in rows if x['label']=='final-initial-X-'+str(r['seed']))
        changes.append(dict(seed=r['seed'],initial_gain=old['X_after1s']['synchronous_gain'],candidate_gain=r['X_after1s']['synchronous_gain'],
            gain_delta=r['X_after1s']['synchronous_gain']-old['X_after1s']['synchronous_gain'],
            initial_correlation=old['X_after1s']['correlation'],candidate_correlation=r['X_after1s']['correlation']))
    total_updates=sum(x['new_updates'] for x in training)
    total_transitions=sum(x['new_rollout_transitions'] for x in training)
    assert total_updates==1600 and total_transitions==157286400
    assert abs(sum(j['GPU_container_seconds'] for j in job_usage)-resources['clock']['GPU_container_SUM_seconds'])<.01
    paired=[]
    for protocol in ('A','T0','T1','X'):
        for model in ('initial','candidate'):
            group=[r for r in rows if r['protocol']==protocol and r['label'].startswith('final-'+model+'-')]
            paired.append(dict(model=model,protocol=protocol,n=sum(r['n'] for r in group),complete=sum(r['complete'] for r in group),strict_safe=sum(r['strict_safe'] for r in group),
                native_joint_5cm=sum(r['joint_5cm'] for r in group),nonfoot=sum(r['nonfoot'] for r in group),overspeed=sum(r['overspeed'] for r in group),joint_limit=sum(r['joint_limit'] for r in group),
                EE_full20s_RMSE_m=[r['full20']['ee_position_rmse'] for r in group],XY_full20s_RMSE_mps=[r['full20']['xy_rmse'] for r in group],yaw_full20s_RMSE_radps=[r['full20']['yaw_rmse'] for r in group]))
    record=dict(status='FINAL_COMPLETED',formal_baseline_PASS=completed['formal_baseline_PASS'],original_numeric_gates=matrix['single_candidate_numeric_gates'],
        task='Original historical B0-X Uniform, native PPO, full20s A/T0/T1/X including startup; no SuperMimic',
        original_task_commit='ae1054046f792df8da6362c36ee168ded6b2ebcd',
        selected=dict(checkpoint=selected['checkpoint'],sha256=selected['checkpoint_sha256'],pre_final_development_retained=selected['retained'],selection_used_final_seeds=False),
        common_initialization_sha256='16333b869458ae62c371fd68b92f61e4740c250635f8b57906809ac067a77091',
        final_seeds=[381031,391031,401031],training_seeds=[21],training_windows=training,final_evaluations=evaluations,paired_summary=paired,X_paired_changes=changes,registered_direction_diagnostics=direction_changes,
        original_final_checks=matrix['checks'],two_consecutive_saved_nodes_requirement=matrix['original_two_consecutive_saved_nodes_requirement'],
        resources=dict(clock=resources['clock'],new_PPO_updates=total_updates,new_PPO_rollout_transitions=total_transitions,
            Adam_minibatches=sum(x['Adam_minibatches'] for x in training),evaluation_cases=len(compact['physics']),
            metered_jobs=job_usage,total_metered_containers=len(job_usage),
            evaluation_simulator_transition_slots=sum(x['simulator_transition_slots'] for x in compact['physics']),
            evaluation_first_episode_valid_samples=sum(x['first_episode_valid_samples'] for x in compact['physics']),
            restore_probe_diagnostic_transition_slots=restore_slots,
            predecessor_transitions='Warm-start CP1800 has its own saved historical counter; not charged as new Stage2 data.',predecessor_GPU_cost='Not reconstructed; not zero.'),
        checkpoint_restore=restored,remote_receipt_sha256={n:v['sha256'] for n,v in states.items() if n in ('COMPLETED','RESOURCES_FINAL','FINAL_MATRIX','FROZEN_SELECTION','FINAL_RESTORE_RUN')},
        reproduction=dict(selected_training_run='${STAGE2_ROOT}/runs/20261031T143531-control-nativePPO-first200',
            exact_host_command_and_frozen_source_manifest='${STAGE2_ROOT}/runs/20261031T143531-control-nativePPO-first200/launch.json',
            original_native_task_and_assets_required=True,actor_critic_history_action_dimensions=[210,102,18],
            seed=21,num_envs=4096,updates=200,Uniform_dynamic_fixed_fraction=[.3,.7],native_entropy_coefficient=.005,
            in_container_training_entry='${ISAAC_PYTHON_WRAPPER} /stage2-active/train_entry.py --num-envs 4096 --updates 200',
            in_container_evaluation_entry='${ISAAC_PYTHON_WRAPPER} /stage2-active/evaluate_entry.py',
            requires_explicit_config_and_asset_mounts='Reuse the frozen launch manifest with a new output directory and an active independent budget; the completed Stage2 ledger cannot be reopened.'),
        limitations=['One training seed; three final physics seeds do not establish cross-training-seed robustness.',
            'Native joint_5cm success is not dynamic-X qualification.',
            'After1s dynamic diagnostics do not remove startup from full20s acceptance or strict safety.',
            'Only historical B0-X source recovered; latest offline original-host working tree was not inspected.',
            'Virtual local reward audit is not full return, policy gradient or an executed controller.'])
    return portable(record,roots)

def portable(value,roots):
    if isinstance(value,dict):return {k:portable(v,roots) for k,v in value.items()}
    if isinstance(value,list):return [portable(v,roots) for v in value]
    if isinstance(value,str):
        for original,replacement in roots:value=value.replace(original,replacement)
        return value
    return value

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--delivery',required=True);a.add_argument('--compact',required=True);a.add_argument('--out',required=True);a.add_argument('--stage2-root',required=True);a.add_argument('--original-runtime-root',required=True);c=a.parse_args()
    roots=[(c.stage2_root,'${STAGE2_ROOT}'),(c.original_runtime_root,'${ORIGINAL_RUNTIME_ROOT}')]
    result=export(read(c.delivery),read(c.compact),roots);result['export_input_sha256']={name:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for name,path in [('delivery',c.delivery),('compact',c.compact)]}
    pathlib.Path(c.out).write_text(json.dumps(result,indent=2)+'\n')
