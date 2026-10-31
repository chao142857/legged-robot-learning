"""Bounded persistent native-PPO windows and unchanged full20s evaluations."""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess,time,traceback
from gpu_budget import Budget,read,write,read_state
DEV=(6181031,6281031)

def wait(b,run):
    j=next(j for j in read(b.p/'LEDGER.json')['jobs'] if j['run']==str(run))
    while read_state(j['cid'])['Running']:
        write(b.p/'LEARNING_STATE.json',dict(pid=os.getpid(),run=str(run),clock=b.meter()));time.sleep(5)
    state=read_state(j['cid']);errors=[str(run/n) for n in ('ppo/error.json','evaluation/error.json') if (run/n).exists()]
    if state['ExitCode'] or errors:raise RuntimeError(dict(run=str(run),exit=state['ExitCode'],errors=errors))

def evaluate(b,cfg,label,purpose='research'):
    idx=read(b.p/'EVAL_INDEX.json');cfg=dict(cfg,mode='eval',entry='evaluate_entry.py',phase='P4' if purpose=='final' else cfg['phase'])
    if label in idx:run=pathlib.Path(idx[label])
    else:
        run=b.launch(cfg,label,purpose=purpose,timeout=300);idx[label]=str(run);write(b.p/'EVAL_INDEX.json',idx)
    wait(b,run);s=read(run/'evaluation/summary.json')
    assert s['num_episodes']==cfg['num_envs'] and s['complete_episode_evaluation'] and s['actor_unchanged'] and s['critic_unchanged']
    assert (run/'evaluation/manifest.json').exists()
    return run

def report(b,tag):
    base=read(b.p/'P1_BASE_CONFIG.json');out=b.p/('PHYSICS_'+tag+'.json')
    subprocess.run([read(b.p/'CPU_RUNTIME.json')['python'],str(b.p/'active/analyze_physics.py'),'--root',str(b.p),'--source',base['native_source'],'--out',str(out)],check=True,timeout=240)
    d=read(out);write(b.p/'PHYSICAL_CURVE.json',d);return d

def tests(b,base,cp,label,phase='P2',seeds=DEV,protocols=('A','T0','T1','X')):
    for seed in seeds:
        for protocol in protocols:evaluate(b,dict(base,**cp,num_envs=64,phase=phase,seed=seed,protocol=protocol),label+'-'+protocol+'-'+str(seed))
    return [r for r in report(b,label)['rows'] if r['label'].startswith(label+'-')]

def train(b,cfg,label,timeout=3600):
    receipt=b.p/(label+'_RUN.json')
    if receipt.exists():run=pathlib.Path(read(receipt)['run'])
    else:run=b.launch(cfg,label,timeout=timeout);write(receipt,dict(run=str(run)))
    wait(b,run);result=read(run/'ppo/result.json');assert result['passed']
    ck=result['checkpoint'];path=run/'ppo/checkpoints'/pathlib.Path(ck['path']).name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ck['sha256']
    cp=dict(checkpoint=str(path),checkpoint_sha256=ck['sha256'])
    if cfg.get('actor_recipe') is not None:cp['actor_recipe']=cfg['actor_recipe']
    return run,cp

def decision(rows,old=None):
    xs=[r for r in rows if r['protocol']=='X'];assert len(xs)==2
    bad=[dict(protocol=r['protocol'],seed=r['seed'],nonfoot=r['nonfoot'],hard_joint=r['joint_limit'],strict_safe=r['strict_safe'],complete=r['complete']) for r in rows if r['nonfoot'] or r['joint_limit'] or r['strict_safe']/r['n']<.90 or r['complete']/r['n']<.95]
    retained=all(all(r.get('retention_checks',{}).values()) for r in rows)
    gain=min(r['X_after1s']['synchronous_gain'] for r in xs)
    amplitude=min(r['X_after1s']['actual_fitted_amplitude_m'] for r in xs)
    correlation=min(r['X_after1s']['correlation'] for r in xs)
    original=all(all(r['absolute_X_gate'].values()) and all(r.get('relative_X_gate',{}).values()) for r in xs) and retained
    return dict(worst_seed_X_gain=gain,worst_seed_X_amplitude_m=amplitude,worst_seed_X_correlation=correlation,original_all_development_gates=original,original_retention=retained,
        severe_physical_failures=bad,learning_change=None if old is None else gain-old['worst_seed_X_gain'],
        engineering_gate_not_formal_PASS=True,rows=[dict(label=r['label'],strict_safe=r['strict_safe'],complete=r['complete'],X_gain=r['X_after1s']['synchronous_gain'],amplitude_m=r['X_after1s']['actual_fitted_amplitude_m']) for r in rows])

def main(b):
    write(b.p/'LEARNING_PROCESS.json',dict(pid=os.getpid()))
    assert (b.p/'P1_DONE.json').exists() and (b.p/'MECHANISM_ACTIVATED.json').exists()
    activation=read(b.p/'MECHANISM_ACTIVATED.json');recipe=read(b.p/'SYNERGY_RECIPE.json');assert activation['recipe_sha256']==hashlib.sha256((b.p/'SYNERGY_RECIPE.json').read_bytes()).hexdigest()
    base=read(b.p/'P1_BASE_CONFIG.json');base.pop('probe',None);base.update(mode='eval',entry='evaluate_entry.py',phase='P2',num_envs=64)
    parent=read(b.p/'CHECKPOINTS.json')['parent'];control=read(b.p/'CHECKPOINTS.json')['candidate']
    cfg=dict(read(b.p/'NATIVE_TRAIN_CONFIG.json'),phase='P2',mode='train',entry='train_entry.py',actor_recipe=recipe,training_seed=21,gradient_audit=True)
    control_cfg=dict(cfg);control_cfg.pop('actor_recipe')
    previous=None;failures=0;best=None;nodes=[];current_cfg=dict(cfg);current_control=dict(control_cfg);additional=0
    for window in (2,98,100):
        label='synergy'+str(additional+window)+'-seed21'
        run,cp=train(b,dict(current_cfg,updates=window),label);additional+=window
        control_run,control_cp=train(b,dict(current_control,updates=window),'matchedNative'+str(additional)+'-seed21')
        current_control=dict(control_cfg,**control_cp,resume=True)
        if window==2:
            write(b.p/'SYNERGY_NATIVE_SMOKE.json',dict(run=str(run),restore_and_Gaussian_checks=read(run/'ppo/zero-mean-attachment.json'),result=read(run/'ppo/result.json')))
            tests(b,base,parent,'frozen1800')
            write(b.p/'PAIRED_SCHEDULE.json',dict(parent=parent,seed=21,common_window_cuts=[2,100,200],common_Uniform_rewards_physics_optimizer=True,
                reason='Match all physical-reset boundaries; cached Stage2 uninterrupted200 remains a diagnostic, not the causal control for a split-window candidate',control_run=str(control_run),candidate_run=str(run)))
            current_cfg=dict(cfg,**cp,resume=True);continue
        rows=tests(b,base,cp,label)
        control_rows=tests(b,base,control_cp,'matchedNative'+str(additional)+'-seed21')
        d=decision(rows,previous);cd=decision(control_rows)
        d['matched_native_control']=cd;d['gain_vs_matched_native']=d['worst_seed_X_gain']-cd['worst_seed_X_gain'];d['amplitude_vs_matched_native_m']=d['worst_seed_X_amplitude_m']-cd['worst_seed_X_amplitude_m']
        write(b.p/(label+'_DECISION.json'),d);nodes.append(dict(label=label,checkpoint=cp,decision=d,run=str(run)))
        write(b.p/'CANDIDATE_NODES.json',nodes)
        if d['severe_physical_failures']:
            write(b.p/'MECHANISM_REPAIR_NEEDED.json',dict(reason='Actual hard/contact/large safety or direction degradation; stop this branch, analyze rather than blindly extend',node=nodes[-1],clock=b.meter()));return
        if previous is not None and d['learning_change']<.02:failures+=1
        elif previous is not None:failures=0
        if best is None or d['worst_seed_X_gain']>best['decision']['worst_seed_X_gain']:best=nodes[-1]
        previous=d;current_cfg=dict(cfg,**cp,resume=True)
    # First200 is a diagnostic mechanism window, never an automatic claim of PASS.
    initial=[r for r in report(b,'after200')['rows'] if r['label'].startswith('frozen1800-')]
    start=decision(initial)
    if previous['worst_seed_X_gain']<start['worst_seed_X_gain']+.04 or previous['worst_seed_X_amplitude_m']<start['worst_seed_X_amplitude_m']+.0015 or previous['gain_vs_matched_native']<.02 or previous['amplitude_vs_matched_native_m']<.001:
        write(b.p/'MECHANISM_REPAIR_NEEDED.json',dict(reason='First200 did not show substantial reproducible physical X progress; inspect head gradients, actions and native reward before another mechanism',nodes=nodes,initial=start,clock=b.meter()));return
    previous_phase='P3';current_cfg['phase']='P3'
    while b.phase_seconds('P3')<b.a['phase_seconds_caps']['P3']-1100:
        # Actual measured throughput determines useful slice; it does not create
        # an artificial lifetime update quota.
        result=read(pathlib.Path(nodes[-1]['run'])/'ppo/result.json');speed=result['training_loop_seconds']/result['updates']
        left=min(b.a['phase_seconds_caps']['P3']-b.phase_seconds('P3'),b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds'])
        window=200 if left>200*speed+650 else 100 if left>100*speed+650 else 0
        if not window:break
        label='synergy'+str(additional+window)+'-seed21';run,cp=train(b,dict(current_cfg,updates=window),label);additional+=window
        rows=tests(b,base,cp,label,phase='P3');d=decision(rows,previous);nodes.append(dict(label=label,checkpoint=cp,decision=d,run=str(run)));write(b.p/'CANDIDATE_NODES.json',nodes);write(b.p/(label+'_DECISION.json'),d)
        if d['severe_physical_failures']:
            write(b.p/'MECHANISM_REPAIR_NEEDED.json',dict(reason='Real severe/contact/hard-limit event after continuation',node=nodes[-1],clock=b.meter()));return
        failures=failures+1 if d['learning_change']<.02 else 0
        if best is None or d['worst_seed_X_gain']>best['decision']['worst_seed_X_gain']:best=nodes[-1]
        if len(nodes)>1 and d['original_all_development_gates'] and nodes[-2]['decision']['original_all_development_gates']:
            write(b.p/'RESEARCH_CLOSED.json',dict(reason='Two consecutive saved nodes satisfy unchanged development gates; freeze for final seeds',nodes=nodes[-2:],clock=b.meter()));return
        if failures>=2:
            write(b.p/'MECHANISM_REPAIR_NEEDED.json',dict(reason='Two physical milestones without substantial X progress; no mechanical same-recipe extension',nodes=nodes[-2:],clock=b.meter()));return
        previous=d;current_cfg=dict(cfg,**cp,resume=True,phase='P3')
    write(b.p/'RESEARCH_CLOSED.json',dict(reason='Useful bounded window or phase-time limit reached; preserve all formal reserve',nodes=nodes,clock=b.meter()))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);v=a.parse_args();b=Budget(v.root)
    with (b.p/'learning.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b)
        except BaseException as e:write(b.p/'LEARNING_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),clock=b.meter()));raise
