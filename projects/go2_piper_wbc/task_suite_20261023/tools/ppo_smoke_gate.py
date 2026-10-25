"""Read-only, fail-closed preflight for every selected PPO smoke case.

Inspects both updates rather than latest.json. Never imports torch/Isaac or
launches training. Metadata copied without raw files cannot pass full preflight.
"""
from publication_paths import research_path
import argparse
import json
import math
from pathlib import Path
from task_identity import file_sha256, task_identity

NATIVE_CP3000_SHA256='600a2dab9caf18289bb6ec558f86c4e6e86c352b621f9a6d83ff73f209cc08b6'
NATIVE_ACTOR_SHA256='5de058e562012c68bbe65631b4e93b245e13be5b5ebfe4b64b251d67efbe8956'
NATIVE_CRITIC_SHA256='8848ae6fd19efd33c5c4d7a74a62c90a64596901edf4a08b21b5bd55718f4693'
CASES={'t0':('T0','plane'),'t1':('T1','plane'),'up':('T2','stairs_up'),'down':('T2','stairs_down')}
REQUIRED_FILES=('protocol.json','transfer.json','updates.jsonl','summary.json',
                'candidate_identity.json','cp_final.pt','cp_final.json')


def finite_errors(value,path='root'):
    errors=[]
    if isinstance(value,float) and not math.isfinite(value):
        errors.append('nonfinite:'+path)
    elif isinstance(value,dict):
        for key,item in value.items(): errors.extend(finite_errors(item,path+'/'+str(key)))
    elif isinstance(value,list):
        for i,item in enumerate(value): errors.extend(finite_errors(item,path+'/'+str(i)))
    return errors


def inspect_updates(rows):
    errors=finite_errors(rows,'updates')
    if len(rows)!=2: errors.append('updates:expected_exactly_2_rows')
    if [row.get('completed_updates') for row in rows]!=[1,2]:
        errors.append('updates:expected_sequential_completed_updates_1_2')
    for i,row in enumerate(rows):
        for key in ('kl_mean','kl_max','actor_grad_norm_mean','actor_grad_norm_max',
                    'critic_grad_norm_mean','critic_grad_norm_max','learning_rate'):
            value=row.get(key)
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
                errors.append(f'updates:{i+1}:missing_or_nonfinite_{key}')
        kl=row.get('kl_mean')
        if isinstance(kl,(int,float)) and math.isfinite(kl) and (kl<0 or kl>=.1):
            errors.append(f'updates:{i+1}:kl_mean_out_of_bounds:{kl}')
        if not isinstance(row.get('losses'),dict) or not row['losses']:
            errors.append(f'updates:{i+1}:missing_losses')
        if not isinstance(row.get('metrics'),dict) or not row['metrics']:
            errors.append(f'updates:{i+1}:missing_metrics')
        std=row.get('action_std',[])
        if not isinstance(std,list) or len(std)!=18 or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in std):
            errors.append(f'updates:{i+1}:invalid_action_std')
    return errors


def current_dependencies(suite_root):
    root=Path(suite_root)
    native=root.parent/'source/LeggedManip_Lab/LeggedManip_Lab'
    prefix=native/'tasks/manager_based/leggedmanip_lab'
    sources={p:root/p for p in ('task_suite/config.py','task_suite/mdp.py','task_suite/profiles.json',
        'tools/run_suite.py','tools/telemetry.py','tools/task_identity.py','tools/audit_geometry_runtime.py')}
    for p in ('leggedmanip_lab_env_cfg.py','mdp/rewards.py','mdp/pose_command_wbc.py',
              'config/go2_piper/wbc_env_cfg.py','config/go2_piper/agents/rsl_rl_ppo_cfg.py'):
        sources['native/'+p]=prefix/p
    base=native/'assets/go2_piper'
    assets={'robot_articulation_cfg':base/'go2_piper_articulation_cfg.py','robot_usd':base/'go2_piper.usd'}
    for name in ('physics','sensor','base','robot'):
        assets['robot_usd_configuration_'+name]=base/'configuration'/('go2_piper_'+name+'.usd')
    return dict(sources=sources,assets=assets)


def inspect_smoke(smoke,admission_path,suite_root,native_checkpoint,case):
    smoke=Path(smoke); errors=[]; checked=[]; objects={}
    def fail(message): errors.append(message)
    def read(name,path):
        try:
            value=json.loads(Path(path).read_text())
            if not isinstance(value,dict): raise ValueError('expected JSON object')
            errors.extend(finite_errors(value,name)); return value
        except Exception as exc:
            fail(name+':unreadable:'+str(exc)); return {}
    manifest=read('manifest',smoke/'manifest.json')
    for name in REQUIRED_FILES:
        path=smoke/name; entry=manifest.get(name)
        if not isinstance(entry,dict): fail('manifest:missing_entry:'+name); continue
        if not path.is_file(): fail('manifest:missing_file:'+name); continue
        if path.stat().st_size!=entry.get('bytes') or file_sha256(path)!=entry.get('sha256'):
            fail('manifest:hash_or_size_mismatch:'+name)
        else: checked.append(name)
    for name in ('protocol','transfer','summary','candidate_identity','cp_final'):
        objects[name]=read(name,smoke/(name+'.json'))
    try:
        rows=[json.loads(line) for line in (smoke/'updates.jsonl').read_text().splitlines() if line.strip()]
        if not all(isinstance(row,dict) for row in rows): raise ValueError('each update must be an object')
        errors.extend(inspect_updates(rows))
    except Exception as exc:
        rows=[]; fail('updates:unreadable:'+str(exc))
    summary=objects['summary']; protocol=objects['protocol']; transfer=objects['transfer']; identity=objects['candidate_identity']
    admission=read('admission',admission_path)
    if summary.get('completed_updates')!=2 or summary.get('stopped_reason')!='bounded run completed':
        fail('summary:not_exactly_two_completed_updates')
    cp_final=objects['cp_final']
    if cp_final.get('completed_updates')!=2 or cp_final.get('sha256')!=manifest.get('cp_final.pt',{}).get('sha256') or cp_final.get('bytes')!=manifest.get('cp_final.pt',{}).get('bytes'):
        fail('cp_final:checkpoint_metadata_mismatch')
    if transfer.get('strict_load') is not True or transfer.get('bitwise_equal')!={'actor':True,'critic':True}:
        fail('transfer:not_strict_complete_bitwise_load')
    if transfer.get('optimizer_state_entries')!=0 or transfer.get('iteration')!=0:
        fail('transfer:not_fresh_optimizer_iteration_zero')
    if transfer.get('actor_sha256')!=NATIVE_ACTOR_SHA256 or transfer.get('critic_sha256')!=NATIVE_CRITIC_SHA256:
        fail('transfer:not_native_cp3000_actor_critic')
    if transfer.get('observation_dims')!={'policy':[210],'critic':[102]}:
        fail('transfer:unexpected_observation_dimensions')
    native_checkpoint=Path(native_checkpoint)
    if not native_checkpoint.is_file() or file_sha256(native_checkpoint)!=NATIVE_CP3000_SHA256:
        fail('native_checkpoint:missing_or_sha_mismatch')
    if protocol.get('checkpoint_sha256')!=NATIVE_CP3000_SHA256 or Path(protocol.get('checkpoint',''))!=native_checkpoint:
        fail('protocol:not_original_native_checkpoint')
    if protocol.get('mode')!='train' or protocol.get('requested_updates')!=2 or protocol.get('seed')!=21:
        fail('protocol:not_expected_two_update_smoke')
    if protocol.get('ppo_schedule')!='adaptive' or protocol.get('learning_rate')!=5e-4:
        fail('protocol:changed_native_ppo_configuration')
    if protocol.get('initialization')!='strict complete actor/critic, fresh optimizer, iteration zero':
        fail('protocol:unexpected_initialization')
    if protocol.get('admission')!=admission:
        fail('protocol:embedded_admission_changed')
    task=identity.get('task',{})
    try:
        key,spec=task_identity(task)
        if not task or any(o.get('admission_key')!=key for o in (identity,protocol,admission,summary)):
            fail('identity:task_key_mismatch')
        if protocol.get('task')!=spec or admission.get('task')!=spec:
            fail('identity:task_metadata_mismatch')
        family,terrain=CASES[case]
        if task.get('family')!=family or task.get('terrain')!=terrain or task.get('difficulty')!='easy' or task.get('ee_mode')!='smooth':
            fail('identity:unexpected_task_family_terrain_or_profile')
        current_profile=json.loads((Path(suite_root)/'task_suite/profiles.json').read_text())['easy']
        if task.get('params')!=current_profile:
            fail('identity:task_parameters_differ_from_current_easy')
    except Exception as exc: fail('identity:unverifiable:'+str(exc))
    for flag in ('geometry_pass','kinematics_pass','physics_pass'):
        if admission.get(flag) is not True: fail('admission:not_passed:'+flag)
    if admission.get('exact_profile_only') is not True or admission.get('interval_or_bank_admission') is not False:
        fail('admission:not_exact_profile_only')
    bindings=identity.get('bindings',{})
    if bindings!=admission.get('bindings'):
        fail('identity:admission_source_asset_bindings_differ')
    for group,paths in current_dependencies(suite_root).items():
        recorded=bindings.get(group,{})
        if set(recorded)!=set(paths): fail('identity:unexpected_file_set:'+group)
        for name,path in paths.items():
            if not path.is_file() or file_sha256(path)!=recorded.get(name):
                fail('identity:current_dependency_mismatch:'+group+'/'+name)
    # Verify the original physical evidence chain too; no replacement paths or
    # metadata-only exports can satisfy the actual launch gate.
    evidence=admission.get('evidence',[])
    if not isinstance(evidence,list) or not evidence: fail('admission:missing_evidence_chain')
    else:
        for item in evidence:
            path=Path(item.get('path',''))
            if not path.is_absolute(): path=Path(admission_path).parent/path
            if not path.is_file() or file_sha256(path)!=item.get('sha256'):
                fail('admission:changed_or_missing_evidence:'+str(path))
    return dict(case=case,smoke=str(smoke),passed=not errors,errors=errors,
        updates=len(rows),kl_means=[row.get('kl_mean') if isinstance(row.get('kl_mean'),(int,float)) and math.isfinite(row['kl_mean']) else None for row in rows],verified_manifest_files=checked,
        checkpoint_sha256=NATIVE_CP3000_SHA256,gate_sha256=file_sha256(__file__),
        full_preflight=True,training_started=False)


def preflight_all(output,suite_root,native_checkpoint,cases,tag='v1'):
    results=[]
    for case in cases:
        try:
            results.append(inspect_smoke(Path(output)/('train_'+case+'_smoke_'+tag),
                Path(output)/'audit'/('admission_'+case+'_v1.json'),suite_root,native_checkpoint,case))
        except Exception as exc:
            results.append(dict(case=case,passed=False,errors=['gate_exception:'+repr(exc)],training_started=False))
    return dict(passed=bool(results) and all(r['passed'] for r in results),cases=results,training_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,help='Existing execution output root; read-only')
    p.add_argument('--suite-root',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--checkpoint',default=research_path('go2_piper_upstream_wbc_20261023/formal_seed21/cp_3000.pt'))
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--tag',default='v1')
    a=p.parse_args()
    result=preflight_all(a.out,a.suite_root,a.checkpoint,a.cases,a.tag)
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result['passed'] else 2)
