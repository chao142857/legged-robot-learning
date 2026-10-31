"""Conditional native-PPO entropy-only contrast; reuses this experiment's matched native control."""
import argparse,fcntl,hashlib,json,os,pathlib,time,traceback
from gpu_budget import Budget,read,write
from continue_pipeline import report,train,tests,rows_for

def main(b,source):
    write(b.p/'ENTROPY_TRIAL_PROCESS.json',dict(pid=os.getpid()))
    while not (b.p/'STAGE2_ANALYSIS_NEEDED.json').exists():
        if (b.p/'RESEARCH_CLOSED.json').exists():
            write(b.p/'ENTROPY_TRIAL_DONE.json',dict(executed=False,reason='Earlier research closed without this conditional diagnosis'));return
        if (b.p/'CONTINUATION_ERROR.json').exists() or (b.p/'PIPELINE_ERROR.json').exists():raise RuntimeError('Resolve actual preceding branch error before an entropy experiment')
        time.sleep(5)
    flag=read(b.p/'STAGE2_ANALYSIS_NEEDED.json')
    if not flag['reason'].startswith('No retained, reproducible dynamic-X learning trend from coverage/control after400'):
        write(b.p/'ENTROPY_TRIAL_DONE.json',dict(executed=False,reason='Different later-stage diagnosis requires review; declared400-step hypothesis not applied',boundary=flag));return
    curve=report(b,source,'before_entropy_trial')
    rows=[r for r in curve['rows'] if r['label'].startswith(('control400-','coverage400-'))]
    initial=[r for r in curve['rows'] if r['label'].startswith('frozen1800-')]
    admission=dict(enough_cases=len(rows)>=8,safe_frozen_initialization=len(initial)==12 and all(r['complete']/r['n']>=.95 and r['strict_safe']/r['n']>=.95 for r in initial),
        preceding_mean_not_collapsed=all(r['complete']/r['n']>=.95 and r['strict_safe']/r['n']>=.85 for r in rows),
        weak_X=all(r['X_after1s']['synchronous_gain']<.20 for r in rows if r['protocol']=='X'),GPU_idle=not b.meter()['active'],time_for_meaningful_trial=b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']>3400)
    if not all(admission.values()):
        write(b.p/'ENTROPY_TRIAL_DONE.json',dict(executed=False,reason='Conditional trial not admitted; no claim of a code/physics error',admission=admission));return
    control_run=pathlib.Path(read(b.p/'control400_RUN.json')['run'])
    updates=[json.loads(line) for line in (control_run/'ppo/updates.jsonl').read_text().splitlines()]
    arm_std=sum(sum(r['action_std'][12:])/6 for r in updates[-20:])/min(20,len(updates))
    if arm_std<.15:
        write(b.p/'ENTROPY_TRIAL_DONE.json',dict(executed=False,reason='Exploration already small; declared hypothesis lacks its observational basis',control_arm_std=arm_std));return
    write(b.p/'ENTROPY_TRIAL_PLAN.json',dict(frozen_before_treatment=True,
        admission_version=2,engineering_admission=admission,formal_safety_threshold_unchanged=.95,
        hypothesis='With finite normal-KL updates, weak commanded X coordination and some mean-policy safety regression, test inherited Gaussian exploration regularization from the same safe CP1800. Not an established unique cause; do not resume the regressed control400 actor.',
        common='Own CP1800, complete native learner/Adam/RNG restore, seed21, Uniform.30/.70,200+200 updates with the same cold-reset boundaries; same original network/LR/KL/rewards/DR/actuators',
        single_factor=dict(control_entropy_coef=.005,treatment_entropy_coef=.001),
        observed_control_arm_raw_action_std_mean=arm_std,noise_interpretation='Gaussian request std, not measured EE noise or confirmed failure cause',
        matched_control='Already executed control200/control400 from this same experiment, not a different historical run; complete missing original T0/T1 dev tests below',
        strict_CRN_claim=False,new_global_count_quota=None,root_time_budget_unchanged=True,
        outcomes='Full20s A/T0/T1/X, dynamic X gain/amplitude/correlation and old YZ/locomotion safety; also compare actual18D std and parameter changes'))
    candidate_source=b.p/'optional_source/train_entry_entropy.py';raw=candidate_source.read_bytes();compile(raw.decode(),str(candidate_source),'exec')
    old=(b.p/'active/train_entry.py').read_bytes();backup=b.p/('train_entry.pre_entropy_'+hashlib.sha256(old).hexdigest()+'.py')
    if not backup.exists():backup.write_bytes(old)
    tmp=b.p/'active/train_entry.entropytmp';tmp.write_bytes(raw);os.replace(tmp,b.p/'active/train_entry.py')
    write(b.p/'ENTROPY_SOURCE_ACTIVATED.json',dict(previous_sha256=hashlib.sha256(old).hexdigest(),new_sha256=hashlib.sha256(raw).hexdigest(),unix=time.time(),no_existing_GPU_run_modified=True))
    base=read(pathlib.Path(read(b.p/'FIRST_RUN.json')['run'])/'config.json');plan=read(b.p/'TRAIN_PLAN.json')
    # Avoid duplicate PPO training while completing a contemporaneous retention control.
    for amount,index in [(200,'CONTROL_RUN.json'),(400,'control400_RUN.json')]:
        parent=pathlib.Path(read(b.p/index)['run']);cp=read(parent/'ppo/result.json')['checkpoint']
        ck=dict(checkpoint=str(parent/'ppo/checkpoints'/pathlib.Path(cp['path']).name),checkpoint_sha256=cp['sha256'])
        tests(b,base,ck,'control'+str(amount),('T0','T1'))
    cfg=dict(plan['common'],dynamic_fraction=.3,entropy_coef=.001)
    run=None;ck=None;previous=None
    for amount in (200,400):
        if run is not None:cfg=dict(read(run/'config.json'),resume=True,updates=200,**ck)
        write(b.p/'ENTROPY_TRIAL_STATE.json',dict(state='native_PPO_training',additional_updates=amount,clock=b.meter()))
        run,ck=train(b,cfg,'entropyLow'+str(amount));tests(b,base,ck,'entropyLow'+str(amount),('A','T0','T1','X'))
        curve=report(b,source,'entropyLow'+str(amount))
        evaluated=[r for r in curve['rows'] if r['label'].startswith('entropyLow'+str(amount)+'-')]
        if any(r['complete']/r['n']<.90 or r['strict_safe']/r['n']<.85 for r in evaluated):
            write(b.p/'ENTROPY_TRIAL_ANALYSIS_NEEDED.json',dict(reason='Material physical degradation: stop this training branch before another window',additional_updates=amount,rows=evaluated));return
    while True:
        now=rows_for(curve,'entropyLow'+str(amount),'X');allrows=[r for r in curve['rows'] if r['label'].startswith('entropyLow'+str(amount)+'-')]
        ctrl=rows_for(curve,'control400','X')
        gain=sum(r['X_after1s']['synchronous_gain'] for r in now)/len(now);corr=sum(r['X_after1s']['correlation'] for r in now)/len(now)
        control_gain=sum(r['X_after1s']['synchronous_gain'] for r in ctrl)/len(ctrl)
        retained=all(all(r['retention_checks'].values()) for r in allrows)
        learned=all(all(r['absolute_X_gate'].values()) and all(r['relative_X_gate'].values()) for r in now)
        decision=dict(additional_updates=amount,gain=gain,correlation=corr,matched_control400_gain=control_gain,retained=retained,full_X_gate=learned,clock=b.meter())
        write(b.p/('entropyLow'+str(amount)+'_DECISION.json'),decision)
        if learned and retained:
            write(b.p/'RESEARCH_CLOSED.json',dict(reason='Entropy-only candidate meets development X/retention gates; independent final test next',candidate=ck));break
        trend=gain>=control_gain+.03 if previous is None else gain>=previous['gain']+.02 or corr>=previous['correlation']+.04
        if not retained or not trend:
            write(b.p/'ENTROPY_TRIAL_ANALYSIS_NEEDED.json',dict(reason='No retained real response improvement after a matched entropy-only trial; do not blindly extend',decision=decision));return
        parent=read(run/'ppo/result.json');per=parent['training_loop_seconds']/parent['updates'];remaining=b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds']
        window=400 if remaining>=400*per+600 else 200 if remaining>=200*per+600 else 0
        if not window:
            write(b.p/'RESEARCH_CLOSED.json',dict(reason='No further full training+development window fits protected final reserve',clock=b.meter()));break
        previous=dict(gain=gain,correlation=corr);amount+=window;cfg=dict(read(run/'config.json'),resume=True,updates=window,**ck)
        run,ck=train(b,cfg,'entropyLow'+str(amount));tests(b,base,ck,'entropyLow'+str(amount),('A','T0','T1','X'));curve=report(b,source,'entropyLow'+str(amount))
    write(b.p/'ENTROPY_TRIAL_DONE.json',dict(executed=True,clock=b.meter()))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--source',required=True);v=a.parse_args();b=Budget(v.root)
    with (b.p/'entropy_trial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b,v.source)
        except BaseException as e:write(b.p/'ENTROPY_TRIAL_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
