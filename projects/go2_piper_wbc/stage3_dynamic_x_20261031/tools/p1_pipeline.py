"""Persistent P1 cold-start pairings; intervention scores cannot be policy PASS."""
import argparse,fcntl,json,os,pathlib,subprocess,time,traceback
from gpu_budget import Budget,read,write,read_state

def wait(b,run):
    job=next(j for j in read(b.p/'LEDGER.json')['jobs'] if j['run']==str(run))
    while read_state(job['cid'])['Running']:
        write(b.p/'PIPELINE_STATE.json',dict(state='P1_physics',run=str(run),clock=b.meter()));time.sleep(5)
    state=read_state(job['cid'])
    assert state['ExitCode']==0,dict(run=str(run),state=state)
    if (run/'evaluation/error.json').exists():raise RuntimeError(read(run/'evaluation/error.json'))
    assert (run/'evaluation/manifest.json').exists()

def main(b):
    write(b.p/'PIPELINE_PROCESS.json',dict(pid=os.getpid()))
    base=read(b.p/'P1_BASE_CONFIG.json');index=read(b.p/'PROBE_INDEX.json');results=[]
    cases=[('parent','X',0.),('parent','X',.5),('parent','T0',0.),('parent','T0',.5),('parent','T1',0.),('parent','T1',.5),('candidate','X',0.),('candidate','T0',0.),('candidate','T1',0.)]
    for parent,protocol,strength in cases:
        label='P1-'+parent+'-'+protocol+'-s'+str(strength)
        cfg=dict(base,**read(b.p/'CHECKPOINTS.json')[parent],protocol=protocol,probe=dict(base['probe'],strength=strength))
        if label in index:run=pathlib.Path(index[label])
        else:run=b.launch(cfg,label,timeout=420);index[label]=str(run);write(b.p/'PROBE_INDEX.json',index)
        wait(b,run);s=read(run/'evaluation/summary.json');x=read(run/'evaluation/dynamic_metrics.json')['windows']['after1s']['axes']['x']
        row=dict(label=label,protocol=protocol,strength=strength,n=s['num_episodes'],safe=s['safe_count'],complete=s['complete_count'],nonfoot=s['nonfoot_contact_count'],overspeed=s['overspeed_count'],joint_limit=s['joint_limit_violation_count'],full20=s['groups']['all']['all'],X={k:v for k,v in x.items() if k!='per_episode'},diagnostic_not_policy_success=True,run=str(run));results.append(row)
        write(b.p/'P1_CURVE.json',dict(rows=results,clock=b.meter()))
        if parent=='parent' and protocol in ('X','T1') and strength==.5:
            control=next(r for r in results if r['label']=='P1-parent-'+protocol+'-s0.0')
            incomplete_response=x.get('actual_fitted_amplitude_m',0)<.008 or x.get('synchronous_gain',-1)<.35 or x.get('correlation',-1)<.5
            same_physical_safety=row['safe']>=control['safe'] and row['complete']>=control['complete'] and row['nonfoot']<=control['nonfoot'] and row['joint_limit']<=control['joint_limit']
            if incomplete_response and same_physical_safety:cases.append((parent,protocol,1.0))
    # Fifty unmodified native PPO updates expose a complete20s reference period
    # for the native-GAE/total-gradient audit. This is diagnostic, not a new start.
    config=dict(read(b.p/'NATIVE_TRAIN_CONFIG.json'),phase='P1',mode='train',entry='train_entry.py',updates=50,gradient_audit=True)
    receipt=b.p/'P1_NATIVE_GRADIENT_RUN.json'
    if receipt.exists():run=pathlib.Path(read(receipt)['run'])
    else:run=b.launch(config,'P1-native-gradient50',timeout=900);write(receipt,dict(run=str(run),not_a_P2_parent=True))
    job=next(j for j in read(b.p/'LEDGER.json')['jobs'] if j['run']==str(run))
    while read_state(job['cid'])['Running']:
        write(b.p/'PIPELINE_STATE.json',dict(state='P1_native_GAE_gradient_audit',run=str(run),clock=b.meter()));time.sleep(5)
    assert read_state(job['cid'])['ExitCode']==0 and read(run/'ppo/result.json')['passed']
    write(b.p/'PIPELINE_STATE.json',dict(state='P1_CPU_causal_reduction',clock=b.meter()))
    subprocess.run([read(b.p/'CPU_RUNTIME.json')['python'],str(b.p/'active/reduce_p1.py'),'--root',str(b.p)],check=True,timeout=180)
    write(b.p/'P1_DONE.json',dict(rows=results,native_gradient_run=str(run),clock=b.meter(),next='CPU causal reduction and evidence-selected single-mechanism PPO, not a final policy claim'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);c=a.parse_args();b=Budget(c.root)
    with (b.p/'pipeline.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b)
        except BaseException as e:write(b.p/'PIPELINE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),clock=b.meter()));raise
