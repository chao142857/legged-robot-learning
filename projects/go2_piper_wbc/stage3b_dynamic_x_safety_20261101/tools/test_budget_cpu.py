"""Synthetic budget/closure checks only; no real Docker, GPU or history writes."""
from pathlib import Path
import importlib.util,json,tempfile,hashlib
W=Path(__file__).parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('pending_budget',W/'gpu_budget.py')
with tempfile.TemporaryDirectory(prefix='stage3b-budget-fixture-') as td:
    p=Path(td);old=p/'fictional_old';new=p/'fictional_new';old.mkdir();new.mkdir();restore=old/'dummy_restore/ppo';restore.mkdir(parents=True)
    m.APPROVED_ROOT=new.resolve();m.APPROVED_GRANT_ID='fixture-only'
    names=['LEDGER.json','GPU_CLOCK.json','COMPLETED.json','FROZEN_SELECTION.json','FINAL_MATRIX.json','CONSECUTIVE_NODE_FINAL.json','RESOURCES_FINAL.json','FINAL_RESTORE_RUN.json']
    c=dict(cap_seconds=21600,active=[],GPU_container_SUM_seconds=12345)
    values={n:{} for n in names};values.update({'GPU_CLOCK.json':c,'COMPLETED.json':{'clock':c},'FINAL_RESTORE_RUN.json':{'run':str(restore.parent)}})
    for n,d in values.items():m.write(old/n,d)
    m.write(restore/'result.json',dict(passed=True,updates=0));m.write(new/'RESOURCE_PLAN.json',dict(elastic=True))
    auth=dict(approved=True,approved_GPU_seconds_SUM=28800,final_reserve_seconds=3600,plan_sha256=m.sha(new/'RESOURCE_PLAN.json'),prior_closed_stage3_root=str(old),prior_closed_file_sha256={n:m.sha(old/n) for n in names},actor_update_cap=None,transition_cap=None,grant_id='fixture-only')
    m.write(new/'AUTHORIZATION.json',auth);m.write(new/'LEDGER.json',{'jobs':[]});b=m.Budget(new);assert b.meter()['GPU_container_SUM_seconds']==0
    clone=p/'cloned_new';clone.mkdir();m.write(clone/'AUTHORIZATION.json',auth)
    try:m.Budget(clone);raise RuntimeError('cloned grant accepted')
    except AssertionError:pass
    m.write(new/'AUTHORIZATION.json',dict(auth,approved_GPU_seconds_SUM=50400))
    try:m.Budget(new);raise RuntimeError('combined budget accepted')
    except AssertionError:pass
    m.write(new/'AUTHORIZATION.json',auth);m.write(old/'LEDGER.json',{'modified':True})
    try:m.Budget(new);raise RuntimeError('changed old ledger accepted')
    except AssertionError:pass
    m.write(old/'LEDGER.json',{})
    start='2026-11-01T01:00:00.000000000Z';end='2026-11-01T01:00:10.000000000Z'
    m.write(new/'LEDGER.json',{'jobs':[dict(run=str(new/'runs/mock'),cid='fake',phase='P2',grant_id='fixture-only')]})
    original_check_output=m.subprocess.check_output
    def fake_inspect(cmd,text):
        assert cmd==['docker','inspect','--format','{{json .State}}','fake']
        return json.dumps(dict(StartedAt=start,FinishedAt=end,Running=False,Paused=False))+'\n'
    m.subprocess.check_output=fake_inspect
    assert b.meter()['GPU_container_SUM_seconds']==10 and b.phase_seconds('P2')==10
    m.subprocess.check_output=lambda cmd,text:json.dumps(dict(StartedAt=start,FinishedAt=end,Running=True,Paused=True))+'\n'
    original_time=m.time.time;m.time.time=lambda:m.stamp(start)+17
    assert b.meter()['GPU_container_SUM_seconds']==17 and len(b.meter()['active'])==1
    m.subprocess.check_output=original_check_output;m.time.time=original_time
    # Closure guard must reject an incomplete old study before creating a new
    # research directory or trying a GPU/driver query.
    bootstrap=load('pending_bootstrap',W/'bootstrap.py')
    unfinished=p/'unfinished';unfinished.mkdir();never=p/'must_not_be_created'
    bootstrap.APPROVED_ROOT=never.resolve();bootstrap.APPROVED_GRANT_ID='fixture-only'
    source=(W/'handoff_guard.py').read_text()
    payload=dict(approved_GPU_seconds_SUM=28800,final_reserve_seconds=3600,user_approved=True,actor_update_cap=None,transition_cap=None,
                 tools={'handoff_guard.py':dict(source=source,sha256=hashlib.sha256(source.encode()).hexdigest())},grant_id='fixture-only')
    try:bootstrap.main(unfinished,never,payload);raise RuntimeError('incomplete old study accepted')
    except AssertionError:assert not never.exists()
    receipt=dict(CPU_fixture_only=True,no_actual_GPU_or_closed_history_touched=True,independent28800_cap=True,combined50400_rejected=True,cloned_directory_rejected=True,
        closed_old_ledger_mutation_rejected=True,required_full_closure_and_restore=True,finished_container_SUM=True,paused_container_SUM=True,
        batched_Docker_state_read_validated=True,elastic_phases_no_hidden_count_caps=True,new_directory_cannot_precede_old_completion=True,
        source_sha256=hashlib.sha256((W/'gpu_budget.py').read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (W.parent/'evidence/BUDGET_CPU_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
