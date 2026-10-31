"""Read-only Stage3 closure check and explicit independent Stage3b handoff.

Run only after original Stage3 completion. No directory, ledger, checkpoint,
container or authorization is modified by this reader.
"""
import argparse, hashlib, json, pathlib, subprocess

REQUIRED = ('LEDGER.json','GPU_CLOCK.json','COMPLETED.json','FROZEN_SELECTION.json',
            'FINAL_MATRIX.json','CONSECUTIVE_NODE_FINAL.json','RESOURCES_FINAL.json','FINAL_RESTORE_RUN.json')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def handoff(old):
    old=pathlib.Path(old)
    for name in REQUIRED:assert (old/name).is_file(), 'Stage3 not fully closed: '+name
    done=read(old/'COMPLETED.json');c=read(old/'GPU_CLOCK.json')
    assert c['cap_seconds']==21600 and not c['active'] and c['GPU_container_SUM_seconds']<=21600
    assert not done['clock']['active'] and not (old/'FINALIZER_ERROR.json').exists()
    restored=read(pathlib.Path(read(old/'FINAL_RESTORE_RUN.json')['run'])/'ppo/result.json')
    assert restored['passed'] and restored['updates']==0
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for j in read(old/'LEDGER.json')['jobs']:
        state=json.loads(subprocess.check_output(['docker','inspect',j['cid']],text=True))[0]['State']
        assert not state['Running']
    nodes=read(old/'SEED22_NODES.json')
    safe=[n for n in nodes if not n['decision']['severe_physical_failures']
          and n['decision']['original_retention']]
    assert safe, 'No safe seed22 checkpoint; diagnose from common CP1800 instead'
    best=max(safe,key=lambda n:n['decision']['worst_seed_X_gain'])
    cp=best['checkpoint'];assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
    original_control=read(old/'synergy200-seed22_DECISION.json')
    positive=(original_control['gain_vs_matched_native']>=.02
              and original_control['amplitude_vs_matched_native_m']>=.001)
    repeated_unsafe=any(n['decision']['severe_physical_failures'] for n in nodes)
    route='A_safe_continuation' if positive and not repeated_unsafe else 'B_leg_safety_causal_contrast' if positive else 'C_replication_failure_diagnosis'
    final_seeds=sorted({r['seed'] for r in read(old/'FINAL_MATRIX.json')['rows']})
    return dict(prior_closed_stage3_root=str(old),prior_closed_file_sha256={n:sha(old/n) for n in REQUIRED},
        old_actual_GPU_seconds_SUM=c['GPU_container_SUM_seconds'],old_original_baseline_PASS=done['original_baseline_PASS'],
        selected_safe_development_node=best,route=route,old_formal_seeds_consumed=final_seeds,
        new_independent_authorization_seconds=28800,new_final_reserve_seconds=3600,
        prior_unused_seconds_not_transferred=True,new_stage_not_started_by_this_script=True,
        selection_source='Original development matrices; no reuse of old formal seeds as unseen')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);v=p.parse_args()
    print(json.dumps(handoff(v.old),indent=2))
