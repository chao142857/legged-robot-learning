"""Bounded follow-up of already saved native CP2300, never new training.

The .02 continuation's 400-update node is safe while 600 has an actual
contact. Test the real intervening saved node on both declared continuation
seeds before any formal seed. Preserve all unsafe600 results and weights.
"""
import argparse, fcntl, hashlib, os, pathlib, time, traceback
import torch
from gpu_budget import Budget, read, write
from stage3_pipeline import tests, decision


def main(b):
    torch.set_num_threads(1)
    plan=read(b.p/'SAVED500_FOLLOWUP_PLAN.json')
    assert plan['training_seeds']==[21,22] and plan['additional_saved_updates']==500
    assert plan['new_training_updates']==0
    write(b.p/'SAVED500_PROCESS.json',dict(pid=os.getpid(),independent_of_chat=True))
    # The existing learner owns the GPU schedule until it completes both seeds.
    while True:
        if (b.p/'RESEARCH_CLOSED.json').exists():
            write(b.p/'SAVED500_SKIPPED.json',dict(reason='Research already closed; no reopening or formal-seed tuning'));return
        process=read(b.p/'ACTIVE_LEARNING_VARIANT.json')
        try:os.kill(process['pid'],0);alive=True
        except ProcessLookupError:alive=False
        if not alive:break
        time.sleep(5)
    assert not (b.p/'FROZEN_SELECTION.json').exists()
    assert (b.p/'LEARNING_REPAIR_NEEDED.json').exists()
    assert not (b.p/'LEARNING_VARIANT_ERROR.json').exists()
    with (b.p/'learning.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=(b.p/'LEARNING_REPAIR_NEEDED.json').read_bytes()
        archive=b.p/'LEG_KL002_LEARNING_STOP_BEFORE_SAVED500.json'
        if archive.exists():assert archive.read_bytes()==original
        else:archive.write_bytes(original)
        assert archive.read_bytes()==original
        (b.p/'LEARNING_REPAIR_NEEDED.json').unlink()
        write(b.p/'ACTIVE_SAVED_NODE_FOLLOWUP.json',dict(pid=os.getpid(),scope='Evaluation only; completed unsafe weights are never resumed'))
        base=read(b.p/'P1_BASE_CONFIG.json');base.pop('probe',None)
        base.update(mode='eval',entry='evaluate_entry.py',phase='P3',num_envs=64)
        for seed in plan['training_seeds']:
            nodes=read(b.p/'CANDIDATE_NODES.json')
            label='legKL002-n500-seed'+str(seed)
            if any(n['label']==label for n in nodes):continue
            parent=next((n for n in nodes if n['label']=='legKL002-n400-seed'+str(seed)),None)
            end=next((n for n in nodes if n['label']=='legKL002-n600-seed'+str(seed)),None)
            if parent is None or end is None or parent['decision']['severe_physical_failures']:
                write(b.p/('SAVED500_UNAVAILABLE_SEED'+str(seed)+'.json'),dict(reason='No safe400 and completed600 training window; never train from a failed policy to manufacture an intermediate node'));continue
            if b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']<650:
                write(b.p/'RESEARCH_CLOSED.json',dict(reason='Saved-node follow-up cannot fit final reserve',clock=b.meter()));return
            run=pathlib.Path(end['run']);path=run/'ppo/checkpoints/cp_2300.pt'
            raw=path.read_bytes();cp=dict(end['checkpoint'],checkpoint=str(path),checkpoint_sha256=hashlib.sha256(raw).hexdigest())
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['iter']==2299 and saved['infos']['completed_updates']==2300
            assert saved['infos']['root_parent_sha256']==plan['common_parent_sha256']
            assert len(saved['optimizer_state_dict']['state'])>0
            del saved
            rows=tests(b,base,cp,label,phase='P3');d=decision(rows,parent['decision'])
            node=dict(label=label,evaluation_prefix=label,recipe_id=plan['recipe_id'],experiment_id='legKL002-seed'+str(seed),
                training_seed=seed,additional_actual_updates=500,actual_checkpoint_iteration=2300,checkpoint=cp,
                decision=d,run=str(run),evaluation_only_followup=True,followup_plan_sha256=hashlib.sha256((b.p/'SAVED500_FOLLOWUP_PLAN.json').read_bytes()).hexdigest())
            nodes.append(node);write(b.p/'CANDIDATE_NODES.json',nodes);write(b.p/(label+'_DECISION.json'),d)
            write(b.p/'SAVED500_STATE.json',dict(node=node,clock=b.meter(),new_training_updates=0))
        nodes=read(b.p/'CANDIDATE_NODES.json');pairs=[]
        for seed in plan['training_seeds']:
            pair=[next((n for n in nodes if n['label']=='legKL002-n'+str(k)+'-seed'+str(seed)),None) for k in (400,500)]
            qualified=all(n is not None and n['decision']['original_all_development_gates'] and not n['decision']['severe_physical_failures'] for n in pair)
            pairs.append(dict(training_seed=seed,qualified=qualified,nodes=pair))
        success=all(p['qualified'] for p in pairs)
        write(b.p/'SAVED500_DONE.json',dict(pairs=pairs,both_seeds_consecutive_original_development_pass=success,new_training_updates=0,formal_PASS=False,clock=b.meter()))
        if success:
            write(b.p/'FINAL_RECIPE_ADMISSION.json',dict(recipe_ids=[plan['recipe_id']],reason='Both independent common-parent continuation seeds pass safe consecutive native saved400/500 nodes; unsafe600 remains excluded',formal_seeds_unopened=True))
            write(b.p/'RESEARCH_CLOSED.json',dict(reason='Replicated safe native saved400/500 pair; proceed to unchanged formal acceptance',clock=b.meter()))
        else:
            repair=dict(reason='Existing intervening checkpoint does not establish both-seed safe consecutive development; no blind extension',owner='saved500_followup',clock=b.meter())
            write(b.p/'SAVED500_REPAIR_NEEDED.json',repair)
            write(b.p/'LEARNING_REPAIR_NEEDED.json',repair)
    assert not torch.cuda.is_initialized()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();b=Budget(a.root)
    try:main(b)
    except BaseException as error:
        write(b.p/'SAVED500_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),clock=b.meter()));raise
