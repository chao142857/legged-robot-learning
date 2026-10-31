"""Independent new grant initialization, strictly after all Stage3 closure proofs.

Inactive preparation. JSON payload supplies explicitly versioned new tools.
No old ledger is copied as a new ledger, reopened, increased, or modified.
"""
import argparse,hashlib,json,pathlib,shutil,subprocess,sys,time
APPROVED_ROOT=pathlib.Path('/data/go2_piper_wbc_stage3b_dynamic_x_20261101').resolve()
APPROVED_GRANT_ID='stage3b-original-uniform-dynamicX-safety-20261101-28800'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(old,new,payload):
    old,new=pathlib.Path(old),pathlib.Path(new)
    assert new.resolve()==APPROVED_ROOT and payload['grant_id']==APPROVED_GRANT_ID,'No cloned grant or fresh-directory budget reset'
    assert old!=new and not new.exists(), 'Never reset/reinitialize an existing study'
    assert payload['approved_GPU_seconds_SUM']==28800 and payload['final_reserve_seconds']==3600
    assert payload['user_approved'] and payload['actor_update_cap'] is None and payload['transition_cap'] is None
    for name,item in payload['tools'].items():
        assert pathlib.Path(name).name==name and name.endswith('.py')
        assert digest(item['source'].encode())==item['sha256']
        compile(item['source'],name,'exec')
    namespace={'__name__':'inactive_handoff_guard'}
    exec(compile(payload['tools']['handoff_guard.py']['source'],'handoff_guard.py','exec'),namespace)
    handoff=namespace['handoff'](old) # must complete before creating new directory
    assert shutil.disk_usage(new.parent).free>=50*1024**3, 'Need real free disk for native trajectories; no history deletion'
    # Preregistered new seeds are checked again against actual completed jobs.
    seeds=set(payload['formal_seeds']['stage3b_formal_seeds'])
    assert seeds=={611101,621101,631101} and not seeds & set(handoff['old_formal_seeds_consumed'])
    for root in [old,old.parent/'go2_piper_wbc_stage2_20261031']:
        for f in (root/'runs').glob('*/config.json'):
            cfg=json.loads(f.read_text());assert cfg.get('seed') not in seeds and cfg.get('training_seed') not in seeds
    new.mkdir();(new/'runs').mkdir();shutil.copytree(old/'active',new/'active',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name,item in payload['tools'].items():(new/'active'/name).write_text(item['source'])
    for name in ('BASE_CONTAINER.json','CPU_RUNTIME.json','CPU_RESPONSE_CONFIG.json','P1_BASE_CONFIG.json','NATIVE_TRAIN_CONFIG.json','CHECKPOINTS.json','SYNERGY_RECIPE.json'):
        shutil.copyfile(old/name,new/name)
    plan=dict(total_GPU_container_SUM_seconds=28800,final_reserve_seconds=3600,
        elastic_minutes=dict(handoff=20,safety_causality=90,evidence_selected_training=150,refinement_and_replication=160,formal_close=60),
        phase_allocations_are_not_earlier_hard_caps=True,counts_accounting_only=True,
        no_old_balance_transfer=True,no_parallel_GPU=True,no_SuperMimic_SGS_M3_or_hardware=True,
        frozen_original_task='ae1054046f792df8da6362c36ee168ded6b2ebcd',baseline_thresholds_unchanged=True)
    write(new/'RESOURCE_PLAN.json',plan)
    auth=dict(approved=True,approved_GPU_seconds_SUM=28800,final_reserve_seconds=3600,
        grant_id=payload['grant_id'],approval_text=payload['approval_text'],plan_sha256=digest((new/'RESOURCE_PLAN.json').read_bytes()),
        actor_update_cap=None,transition_cap=None,prior_closed_stage3_root=str(old),
        prior_closed_file_sha256=handoff['prior_closed_file_sha256'])
    write(new/'AUTHORIZATION.json',auth);write(new/'LEDGER.json',dict(grant_id=payload['grant_id'],jobs=[]))
    write(new/'HANDOFF.json',handoff);write(new/'FORMAL_SEEDS_PREREGISTRATION.json',payload['formal_seeds'])
    # Historical development context is explicitly labelled and never charged
    # to this new grant. New candidates cannot be selected from these labels.
    old_index=json.loads((old/'EVAL_INDEX.json').read_text())
    index={label:path for label,path in old_index.items() if label.startswith('frozen1800-')}
    write(new/'EVAL_INDEX.json',index)
    write(new/'HISTORICAL_CONTEXT.json',dict(source_closed_root=str(old),evaluation_labels=list(index),
        old_cost_not_recharged=True,not_new_training_or_evaluation=True,
        old_formal_results_never_claimed_unseen=True))
    write(new/'SOURCE_PROVENANCE.json',dict(old_closed_source_sha256={p.name:digest(p.read_bytes()) for p in (old/'active').glob('*.py')},
        new_active_sha256={p.name:digest(p.read_bytes()) for p in (new/'active').glob('*.py')},
        payload_tools_sha256={n:v['sha256'] for n,v in payload['tools'].items()}))
    with (new/'budget_supervisor.log').open('x') as log:
        child=subprocess.Popen(['python3','-u',str(new/'active/gpu_budget.py'),'--root',str(new)],stdout=log,stderr=log,start_new_session=True)
    deadline=time.monotonic()+10
    while not (new/'GPU_CLOCK.json').exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.1)
    assert child.poll() is None
    assert json.loads((new/'GPU_CLOCK.json').read_text())['GPU_container_SUM_seconds']==0
    print(json.dumps(dict(root=str(new),budget_supervisor_pid=child.pid,independent_cap=28800,
        old_GPU_seconds=handoff['old_actual_GPU_seconds_SUM'],route=handoff['route'],GPU_containers_started=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);a=p.parse_args()
    main(a.old,a.new,json.load(sys.stdin))
