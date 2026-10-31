"""One evidence-based longer PPO window; preserve registered scores and final time reserve."""
import argparse,fcntl,json,os,pathlib,time,traceback
from gpu_budget import Budget,read,write
from continue_pipeline import report,train,tests,rows_for

def quality(curve,label):
    rows=[r for r in curve['rows'] if r['label'].startswith(label+'-')];checks=[];directions=[]
    for r in rows:
        base=next(v for v in curve['rows'] if v['label']=='frozen1800-'+r['protocol']+'-'+str(r['seed']))
        current=dict(r['retention_checks'])
        for name in ('forward','backward','left','right'):
            if name not in current:continue
            old=base['directions'][name]['gain'];new=r['directions'][name]['gain'];diagnostic=abs(new-1)<=abs(old-1)+.15
            directions.append(dict(case=r['label'],direction=name,initial_gain=old,current_gain=new,registered_absolute_change_pass=current[name],quality_distance_diagnostic=diagnostic))
            current[name]=diagnostic
        checks.append(all(current.values()) and r['strict_safe']/r['n']>=.95 and r['complete']/r['n']>=.95)
    return dict(physical_training_entry=bool(rows) and all(checks),registered_retention_unchanged=all(all(r['retention_checks'].values()) for r in rows),directional_diagnostic=directions)

def main(b,source):
    write(b.p/'FINISH_ENTROPY_PROCESS.json',dict(pid=os.getpid()))
    assert not b.meter()['active'] and not (b.p/'RESEARCH_CLOSED.json').exists()
    curve=report(b,source,'continuation_review400');q=quality(curve,'entropyLow400');assert q['physical_training_entry'],q
    parent=pathlib.Path(read(b.p/'entropyLow400_RUN.json')['run']);control=pathlib.Path(read(b.p/'control400_RUN.json')['run'])
    def std(r):
        rows=[json.loads(x) for x in (r/'ppo/updates.jsonl').read_text().splitlines()][-20:]
        return sum(sum(x['action_std'][12:])/6 for x in rows)/len(rows)
    reduction=1-std(parent)/std(control);assert reduction>=.10
    old_x=rows_for(curve,'entropyLow400','X');assert all(r['X_after1s']['synchronous_gain']>=.10 for r in old_x)
    write(b.p/'LONGER_ENTROPY_PLAN.json',dict(unix=time.time(),frozen_before_new_window=True,entry_review=q,actual_arm_std_reduction=reduction,
        mechanism_unchanged='Native PPO entropy.001; original Uniform.30/.70; full18D and all old task/reward/PD/DR/LR/network fields unchanged',
        reason='First healthy200 window followed by healthy400 core physics and a changed exploration distribution merits a longer bounded window; not a claim that reducing std is task learning.',
        registered_scores_and_final_gates_unchanged=True,new_budget_seconds=0,count_cap=None))
    cp=read(parent/'ppo/result.json')['checkpoint'];ck=dict(checkpoint=str(parent/'ppo/checkpoints'/pathlib.Path(cp['path']).name),checkpoint_sha256=cp['sha256'])
    base=read(pathlib.Path(read(b.p/'FIRST_RUN.json')['run'])/'config.json');window=400;previous_gain=sum(r['X_after1s']['synchronous_gain'] for r in old_x)/2;amount=400
    while True:
        r0=read(parent/'ppo/result.json');remaining=b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds'];assert remaining>window*r0['training_loop_seconds']/r0['updates']+450
        amount+=window;cfg=dict(read(parent/'config.json'),resume=True,updates=window,**ck)
        write(b.p/'FINISH_ENTROPY_STATE.json',dict(state='training',target_additional_updates=amount,clock=b.meter()))
        parent,ck=train(b,cfg,'entropyLow'+str(amount));tests(b,base,ck,'entropyLow'+str(amount),('A','T0','T1','X'));curve=report(b,source,'entropyLow'+str(amount));q=quality(curve,'entropyLow'+str(amount));xs=rows_for(curve,'entropyLow'+str(amount),'X')
        gain=sum(r['X_after1s']['synchronous_gain'] for r in xs)/len(xs);full=all(all(r['absolute_X_gate'].values()) and all(r['relative_X_gate'].values()) for r in xs)
        decision=dict(additional_updates=amount,gain=gain,gain_delta_vs_previous=gain-previous_gain,physical_training_entry=q['physical_training_entry'],registered_retention=q['registered_retention_unchanged'],full_X_gate=full,direction_diagnostic=q['directional_diagnostic'],clock=b.meter())
        write(b.p/('LONG_ENTROPY_'+str(amount)+'_DECISION.json'),decision)
        result=read(parent/'ppo/result.json');remaining=b.cap-b.reserve-b.meter()['GPU_container_SUM_seconds'];per=result['training_loop_seconds']/result['updates']
        if full or not q['physical_training_entry'] or gain<previous_gain+.03 or remaining<200*per+450:break
        previous_gain=gain;window=200
    write(b.p/'RESEARCH_CLOSED.json',dict(reason='Bounded unchanged-mechanism longer-window test completed; final independent evaluation next',decision=decision,all_original_registered_failures_retained=True,clock=b.meter()))
    write(b.p/'FINISH_ENTROPY_DONE.json',dict(checkpoint=ck,clock=b.meter()))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--source',required=True);c=a.parse_args();b=Budget(c.root)
    with (b.p/'finish_entropy.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b,c.source)
        except BaseException as e:write(b.p/'FINISH_ENTROPY_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
