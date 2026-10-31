"""Persistent paired native-PPO continuation, conditional on actual frozen physics."""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess,time,traceback
from gpu_budget import Budget,read,write,read_state
from pipeline import evaluate,wait

def report(b,source,tag):
    out=b.p/('PHYSICS_'+tag+'.json')
    subprocess.run([read(b.p/'CPU_RUNTIME.json')['python'],str(b.p/'analyze_physics.py'),'--root',str(b.p),'--source',source,'--out',str(out)],check=True)
    result=read(out);write(b.p/'PHYSICAL_CURVE.json',result);return result

def train(b,cfg,label):
    index=b.p/(label+'_RUN.json')
    if index.exists():run=pathlib.Path(read(index)['run'])
    else:
        run=b.launch(cfg,label,timeout=5400);write(index,dict(run=str(run)))
    wait(b,run);result=read(run/'ppo/result.json');assert result['passed']
    cp=result['checkpoint'];path=run/'ppo/checkpoints'/pathlib.Path(cp['path']).name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['sha256']
    return run,dict(checkpoint=str(path),checkpoint_sha256=cp['sha256'])

def tests(b,base,checkpoint,label,protocols=('A','X')):
    for seed in (8101,8102):
        for protocol in protocols:
            evaluate(b,dict(base,**checkpoint,seed=seed,protocol=protocol),label+'-'+protocol+'-'+str(seed))

def rows_for(curve,label,protocol):
    return [r for r in curve['rows'] if r['label'].startswith(label+'-') and r['protocol']==protocol]

def safe_to_continue(rows):
    # Engineering catastrophic-degradation guard, never the formal pass gate.
    return len(rows)>=4 and all(r['complete']/r['n']>=.90 and r['strict_safe']/r['n']>=.85 for r in rows)

def main(b,source):
    write(b.p/'CONTINUATION_PROCESS.json',dict(pid=os.getpid()))
    while not (b.p/'PAIRED200_DONE.json').exists():
        if (b.p/'PIPELINE_ERROR.json').exists():raise RuntimeError('First paired window needs specific error repair; see PIPELINE_ERROR')
        write(b.p/'CONTINUATION_STATE.json',dict(state='waiting_for_original_paired200',clock=b.meter()));time.sleep(5)
    plan=read(b.p/'TRAIN_PLAN.json');base=read(pathlib.Path(read(b.p/'FIRST_RUN.json')['run'])/'config.json')
    curve=report(b,source,'paired200');active={}
    for branch in ('control','coverage'):
        tests200=rows_for(curve,branch+'200','A')+rows_for(curve,branch+'200','X')
        if not safe_to_continue(tests200):
            write(b.p/(branch+'_DEGRADATION.json'),dict(state='branch paused after material physical degradation',rows=tests200));continue
        parent=pathlib.Path(read(b.p/(branch.upper()+'_RUN.json'))['run']);r=read(parent/'ppo/result.json');cp=r['checkpoint']
        cfg=dict(read(parent/'config.json'),resume=True,updates=200,checkpoint=str(parent/'ppo/checkpoints'/pathlib.Path(cp['path']).name),checkpoint_sha256=cp['sha256'])
        write(b.p/'CONTINUATION_STATE.json',dict(state='paired400',branch=branch,clock=b.meter()))
        run,ck=train(b,cfg,branch+'400');tests(b,base,ck,branch+'400');active[branch]=(run,ck)
    curve=report(b,source,'paired400')
    comparisons={}
    for branch in active:
        now=rows_for(curve,branch+'400','X');earlier=rows_for(curve,branch+'200','X')
        gain=sum(r['X_after1s']['synchronous_gain'] for r in now)/len(now)
        previous=sum(r['X_after1s']['synchronous_gain'] for r in earlier)/len(earlier)
        delta=sum(r['relative_to_original']['X_gain_delta'] for r in now)/len(now)
        corr=sum(r['X_after1s']['correlation'] for r in now)/len(now)
        comparisons[branch]=dict(gain=gain,gain_vs_initial=delta,gain_change_since200=gain-previous,correlation=corr,
            retention=all(all(r['retention_checks'].values()) for r in now+rows_for(curve,branch+'400','A')))
    write(b.p/'PAIRED400_DONE.json',dict(comparisons=comparisons,clock=b.meter()))
    # Only measured learning earns a further window. A fixed iteration slice is not the experiment budget.
    eligible=[k for k,v in comparisons.items() if v['retention'] and v['gain_vs_initial']>=.03 and v['gain_change_since200']>=-.02]
    if not eligible:
        write(b.p/'STAGE2_ANALYSIS_NEEDED.json',dict(reason='No retained, reproducible dynamic-X learning trend from coverage/control after400; diagnose before a new mechanism',comparisons=comparisons));return
    chosen=max(eligible,key=lambda k:comparisons[k]['gain']);run,ck=active[chosen];previous=comparisons[chosen]
    while True:
        parent=read(run/'ppo/result.json');remaining=b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']
        seconds_per_update=parent['training_loop_seconds']/parent['updates']
        # Adapt the ordinary slice to measured throughput; no hidden global update quota.
        window=400 if remaining>=400*seconds_per_update+600 else 200 if remaining>=200*seconds_per_update+600 else 0
        if not window:break
        additional=parent['completed_updates']-1800+window
        cfg=dict(read(run/'config.json'),resume=True,updates=window,**ck)
        label=chosen+str(additional);write(b.p/'CONTINUATION_STATE.json',dict(state='conditional_learning_window',branch=chosen,additional_updates=additional,clock=b.meter()))
        run,ck=train(b,cfg,label);tests(b,base,ck,label,('A','T0','T1','X'));curve=report(b,source,label)
        now=rows_for(curve,label,'X');allrows=[r for r in curve['rows'] if r['label'].startswith(label+'-')]
        gain=sum(r['X_after1s']['synchronous_gain'] for r in now)/len(now);corr=sum(r['X_after1s']['correlation'] for r in now)/len(now)
        decision=dict(branch=chosen,additional_updates=additional,gain=gain,correlation=corr,
            retained=all(all(r['retention_checks'].values()) for r in allrows),
            all_X_gates=all(all(r['absolute_X_gate'].values()) and all(r['relative_X_gate'].values()) for r in now),clock=b.meter())
        write(b.p/(label+'_DECISION.json'),decision)
        if decision['all_X_gates'] and decision['retained']:
            write(b.p/'RESEARCH_CLOSED.json',dict(reason='Development dynamic-X and retention gates met; freeze candidate before independent final test',candidate=ck,decision=decision));return
        if not safe_to_continue(allrows) or not decision['retained'] or (gain<previous['gain']+.02 and corr<previous['correlation']+.04):
            write(b.p/'STAGE2_ANALYSIS_NEEDED.json',dict(reason='Measured plateau or retention decline; no automatic identical extra updates',decision=decision,previous=previous));return
        previous=dict(gain=gain,correlation=corr)
    write(b.p/'RESEARCH_CLOSED.json',dict(reason='Research allocation reached; preserve final evaluation reserve',clock=b.meter()))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--source',required=True);v=a.parse_args();b=Budget(v.root)
    with (b.p/'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b,v.source)
        except BaseException as e:write(b.p/'CONTINUATION_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
