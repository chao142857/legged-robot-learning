"""Preregistered second continuation seed, same parent and mechanism, no failed-weight continuation."""
import argparse,fcntl,hashlib,os,pathlib,time,traceback
from gpu_budget import Budget,read,write
from stage3_pipeline import train,tests,decision,report

def main(b):
    write(b.p/'SEED22_PROCESS.json',dict(pid=os.getpid()))
    while not (b.p/'MECHANISM_REPAIR_NEEDED.json').exists():
        if (b.p/'LEARNING_ERROR.json').exists():raise RuntimeError('Primary controller has a real error; repair before independent replication')
        time.sleep(5)
    while b.meter()['active']:time.sleep(3)
    first=read(b.p/'MECHANISM_REPAIR_NEEDED.json');assert first['node']['label']=='synergy100-seed21'
    assert first['node']['decision']['severe_physical_failures'],'Seed21 stop must be preserved'
    folder=b.p/'repairs/seed21_hard_limit_stopped';folder.mkdir(exist_ok=True)
    if not (folder/'MECHANISM_REPAIR_NEEDED.json').exists():(b.p/'MECHANISM_REPAIR_NEEDED.json').rename(folder/'MECHANISM_REPAIR_NEEDED.json')
    write(b.p/'SEED22_PLAN.json',dict(training_seed=22,common_parent=read(b.p/'CHECKPOINTS.json')['parent'],recipe_sha256=hashlib.sha256((b.p/'SYNERGY_RECIPE.json').read_bytes()).hexdigest(),
        same_geometry_rewards_LR_entropy_sampler=True,same_window_cuts=[2,100,200],old_seed21_checkpoint_not_resumed=True,
        purpose='Independent continuation replication already specified in RESOURCE_PLAN; distinguish stochastic leg drift from reproducible coupling. Not a second full from-scratch seed.',
        original_hard_limit_threshold_unchanged=True))
    base=read(b.p/'P1_BASE_CONFIG.json');base.pop('probe',None);base.update(mode='eval',entry='evaluate_entry.py',phase='P2',num_envs=64)
    recipe=read(b.p/'SYNERGY_RECIPE.json');cfg=dict(read(b.p/'NATIVE_TRAIN_CONFIG.json'),phase='P2',mode='train',entry='train_entry.py',actor_recipe=recipe,training_seed=22,gradient_audit=True)
    cc=dict(cfg);cc.pop('actor_recipe');ac=dict(cfg);nc=dict(cc);additional=0;previous=None;nodes=[];plateau=0
    for window in (2,98,100):
        run,cp=train(b,dict(ac,updates=window),'synergy'+str(additional+window)+'-seed22');additional+=window
        nr,ncp=train(b,dict(nc,updates=window),'matchedNative'+str(additional)+'-seed22')
        ac=dict(cfg,**cp,resume=True);nc=dict(cc,**ncp,resume=True)
        if window==2:
            write(b.p/'SEED22_SMOKE.json',dict(candidate_run=str(run),control_run=str(nr),zero_output_and_Gaussian=read(run/'ppo/zero-mean-attachment.json'),common_parent=True));continue
        rows=tests(b,base,cp,'synergy'+str(additional)+'-seed22');controls=tests(b,base,ncp,'matchedNative'+str(additional)+'-seed22')
        d=decision(rows,previous);cd=decision(controls);d.update(matched_native_control=cd,gain_vs_matched_native=d['worst_seed_X_gain']-cd['worst_seed_X_gain'],amplitude_vs_matched_native_m=d['worst_seed_X_amplitude_m']-cd['worst_seed_X_amplitude_m'])
        node=dict(label='synergy'+str(additional)+'-seed22',checkpoint=cp,decision=d,run=str(run));nodes.append(node);write(b.p/'SEED22_NODES.json',nodes);write(b.p/(node['label']+'_DECISION.json'),d)
        if d['severe_physical_failures']:
            write(b.p/'REPLICATION_REPAIR_NEEDED.json',dict(reason='Independent seed also has real hard/contact/large safety degradation; stop branch and diagnose coupled control before any repeat',node=node,clock=b.meter()));return
        previous=d
    if previous['gain_vs_matched_native']<.02 or previous['amplitude_vs_matched_native_m']<.001:
        write(b.p/'REPLICATION_REPAIR_NEEDED.json',dict(reason='No reproducible physical improvement over matched control after200; do not extend unchanged recipe',node=nodes[-1],clock=b.meter()));return
    ac['phase']='P3'
    while b.phase_seconds('P3')<b.a['phase_seconds_caps']['P3']-1100:
        result=read(pathlib.Path(nodes[-1]['run'])/'ppo/result.json');speed=result['training_loop_seconds']/result['updates']
        left=min(b.a['phase_seconds_caps']['P3']-b.phase_seconds('P3'),b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds'])
        window=200 if left>200*speed+650 else 100 if left>100*speed+650 else 0
        if not window:break
        run,cp=train(b,dict(ac,updates=window),'synergy'+str(additional+window)+'-seed22');additional+=window
        rows=tests(b,base,cp,'synergy'+str(additional)+'-seed22',phase='P3');d=decision(rows,previous);node=dict(label='synergy'+str(additional)+'-seed22',checkpoint=cp,decision=d,run=str(run));nodes.append(node)
        write(b.p/'SEED22_NODES.json',nodes);write(b.p/(node['label']+'_DECISION.json'),d)
        if d['severe_physical_failures']:
            write(b.p/'REPLICATION_REPAIR_NEEDED.json',dict(reason='Real severe/contact/hard event during longer continuation; no further training from failed model',node=node,clock=b.meter()));return
        plateau=plateau+1 if d['learning_change']<.02 else 0
        if len(nodes)>1 and d['original_all_development_gates'] and nodes[-2]['decision']['original_all_development_gates']:
            write(b.p/'RESEARCH_CLOSED.json',dict(reason='Independent continuation seed has two consecutive saved development nodes meeting unchanged gates; final tests next',nodes=nodes[-2:],clock=b.meter()));return
        if plateau>=2:
            write(b.p/'REPLICATION_REPAIR_NEEDED.json',dict(reason='Two physical milestones plateaued; no automatic extra updates',nodes=nodes[-2:],clock=b.meter()));return
        previous=d;ac=dict(cfg,**cp,resume=True,phase='P3')
    write(b.p/'RESEARCH_CLOSED.json',dict(reason='Measured bounded continuation allocation complete; preserve final reserve',nodes=nodes,clock=b.meter()))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);v=a.parse_args();b=Budget(v.root)
    with (b.p/'replication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b)
        except BaseException as e:write(b.p/'REPLICATION_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),clock=b.meter()));raise
