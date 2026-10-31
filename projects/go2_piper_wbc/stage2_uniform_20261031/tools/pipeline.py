"""Persistent original-PPO stages, serialized under the independent time meter."""
import argparse,fcntl,json,os,pathlib,sys,time,traceback
from gpu_budget import Budget,read,write,read_state

def wait(b,run):
 j=next(j for j in read(b.p/'LEDGER.json')['jobs'] if j['run']==str(run))
 while read_state(j['cid'])['Running']:
  write(b.p/'PIPELINE_STATE.json',dict(pid=os.getpid(),active=str(run),clock=b.meter()));time.sleep(5)
 s=read_state(j['cid']);errors=[str(run/x) for x in ['error.json','ERROR.json','ppo/error.json','evaluation/error.json'] if (run/x).exists()]
 if s['ExitCode']!=0 or errors:raise RuntimeError(json.dumps(dict(run=str(run),exit=s['ExitCode'],errors=errors)))
 return run

def evaluate(b,cfg,label):
 idx=read(b.p/'EVAL_INDEX.json') if (b.p/'EVAL_INDEX.json').exists() else {}
 if label in idx:r=pathlib.Path(idx[label])
 else:r=b.launch(cfg,label,timeout=300);idx[label]=str(r);write(b.p/'EVAL_INDEX.json',idx)
 wait(b,r);s=read(r/'evaluation/summary.json');assert s['num_episodes']==cfg['num_envs'] and s['complete_episode_evaluation']
 assert s['actor_unchanged'] and s['critic_unchanged']
 return s

def main(b):
 write(b.p/'PIPELINE_PROCESS.json',dict(pid=os.getpid()))
 first=pathlib.Path(read(b.p/'FIRST_RUN.json')['run']);cfg=read(first/'config.json')
 if not (b.p/'EVAL_INDEX.json').exists():write(b.p/'EVAL_INDEX.json',{'frozen1800-A-8101':str(first)})
 for seed in [8101,8102,8103]:
  for protocol in ['A','T0','T1','X']:
   c=dict(cfg,seed=seed,protocol=protocol);evaluate(b,c,f'frozen1800-{protocol}-{seed}')
 write(b.p/'BASELINE_DONE.json',dict(evaluations=12,algorithm='Original PPO frozen CP1800',clock=b.meter()))
 while not (b.p/'TRAIN_PLAN.json').exists():
  if b.meter()['GPU_container_SUM_seconds']>=b.cap-b.reserve:raise RuntimeError('Training reserve reached during preparation')
  write(b.p/'PIPELINE_STATE.json',dict(pid=os.getpid(),state='baseline_complete_waiting_CPU_preparation',GPU_active=False));time.sleep(5)
 plan=read(b.p/'TRAIN_PLAN.json');assert plan['frozen_before_training']
 # Separate branches share original CP1800, optimizer and complete RNG restore.
 for branch in ['control','coverage']:
  index=b.p/(branch.upper()+'_RUN.json')
  if index.exists():r=pathlib.Path(read(index)['run'])
  else:
   c=dict(plan['common'],**plan[branch]);r=b.launch(c,branch+'-nativePPO-first200',timeout=5400);write(index,dict(run=str(r)))
  wait(b,r)
  result=read(r/'ppo/result.json');assert result['passed']
  cp=result['checkpoint_schedule'][-1];ck=pathlib.Path(r/'ppo/checkpoints'/pathlib.Path(cp['path']).name)
  for seed in [8101,8102]:
   for protocol in ['A','X']:
    c=dict(cfg,seed=seed,protocol=protocol,checkpoint=str(ck),checkpoint_sha256=cp['sha256']);evaluate(b,c,f'{branch}200-{protocol}-{seed}')
 write(b.p/'PAIRED200_DONE.json',dict(control=read(b.p/'CONTROL_RUN.json'),coverage=read(b.p/'COVERAGE_RUN.json'),clock=b.meter(),next='Analyze actual paired physics before next slice; no user approval needed within authorization'))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);v=a.parse_args();b=Budget(v.root)
 with (b.p/'pipeline.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main(b)
  except BaseException as e:write(b.p/'PIPELINE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
