import argparse,copy,json,hashlib,sys
from pathlib import Path
import numpy as np,torch
from common import ROOT,write,file_sha
from cpu_hash import tree_hash
from cl1_core import Learner,CONFIG,replay,gae
from frozen_ops import save,load
P=json.loads((ROOT/'PROTOCOL.json').read_text())
def mem(item):
 assert file_sha(item['checkpoint'])==item['checkpoint_sha256'] and file_sha(item['memory'])==item['memory_sha256']
 cp=load(item['checkpoint']);s=copy.deepcopy(load(item['memory'])['students'][str(item['seed'])]);i=cp['infos']
 assert i['completed_updates']==5000 and i['student_seed']==item['seed'] and s['pending'] is None
 assert s['transitions'][-1]['completed_updates']==5000 and len(s['history'])==len(s['transitions'])
 assert i['optimizer_calls']==100000
 s['cursor']=len(s['transitions']);return s,dict(actor=tree_hash(cp['actor_state_dict']),critic=tree_hash(cp['critic_state_dict']),optimizer=tree_hash(cp['optimizer_state_dict']),rng=tree_hash(i['rng']),environment=tree_hash(i['environment']),memory=tree_hash(s))
def install(stage,learner,students,kind):
 d=ROOT/'teachers'/stage;assert not d.exists();d.mkdir()
 state=dict(learner=copy.deepcopy(learner),students=copy.deepcopy(students),batches=0,execution_kind=kind)
 for n in ['current.pt','initial.pt']:save(d/n,state)
 save(d/'proxy.pt',dict(teacher=dict(cl1_proxy=1,binding=json.loads((ROOT/'prepared/manifest.json').read_text())['binding'])))
 return tree_hash(state)
def main():
 a=argparse.ArgumentParser();a.add_argument('op');a.add_argument('--seed',type=int);a.add_argument('--stage');x=a.parse_args()
 if x.op=='prepare':
  assert file_sha(P['teacher_source'])==P['teacher_source_sha256'];teacher=load(P['teacher_source'])['learner'];assert teacher['version']==20
  students={};receipts={}
  for i in P['collection']+P['validation']:
   s,receipt=mem(i);receipts[str(i['seed'])]=receipt
   if i in P['collection']:students[str(i['seed'])]=s
  install('collect',teacher,students,'stochastic');save(ROOT/'teachers/shared_before.pt',teacher)
  write(ROOT/'PREFLIGHT.json',dict(passed=True,sources=receipts,teacher_hash=tree_hash(teacher),teacher_source_sha256=P['teacher_source_sha256'],budget=2200))
 elif x.op=='finish':
  state=load(ROOT/'teachers/collect/current.pt');s=state['students'][str(x.seed)];ts=s['transitions'][s['cursor']:]
  assert len(ts)==5 and s['pending'] is None and all(t['version']==20 for t in ts)
  assert [t['completed_updates'] for t in ts]==[5100,5200,5300,5400,5500]
  save(ROOT/'fragments'/f'seed{x.seed}.pt',dict(prefix=s['history'][:s['cursor']],transitions=ts,terminal=False))
  before=load(ROOT/'teachers/shared_before.pt')
  for k in ['net','optimizer']:assert tree_hash(before[k])==tree_hash(state['learner'][k])
 elif x.op=='update':
  shared=load(ROOT/'teachers/collect/current.pt')['learner'];original=load(ROOT/'teachers/shared_before.pt')
  for k in ['net','optimizer']:assert tree_hash(shared[k])==tree_hash(original[k])
  save(ROOT/'teachers/update_before.pt',shared)
  fs=[load(ROOT/'fragments'/f'seed{s}.pt') for s in [43,44]];fh=tree_hash(fs);save(ROOT/'fragments/on_policy_batch.pt',fs)
  before=copy.deepcopy(shared);states={'base':before};results={};raw_targets=[]
  baseline=Learner();baseline.load(shared)
  for f in fs:
   ts=f['transitions'];h=replay(baseline.net,f['prefix']+[t['x'] for t in ts])
   with torch.no_grad():_,nv,_=baseline.net(torch.as_tensor(ts[-1]['next_x']),h)
   adv,ret=gae([t['reward'] for t in ts],[t['value'] for t in ts],float(nv),f['terminal'])
   raw_targets.append(dict(advantages=adv.tolist(),returns=ret.tolist(),old_logprobs=[t['logprob'] for t in ts],rewards=[t['reward'] for t in ts]))
  write(ROOT/'fragments/fixed_targets.json',raw_targets)
  for epochs in [2,10]:
   l=Learner();l.load(copy.deepcopy(before));assert tree_hash(l.state())==tree_hash(before)
   result=l.update(copy.deepcopy(fs),max_epochs=epochs);assert tree_hash(fs)==fh
   name=f'epoch{epochs}';states[name]=l.state();results[name]=result
   save(ROOT/'teachers'/f'{name}.pt',l.state())
  assert results['epoch2']['fixed_targets_sha256']==results['epoch10']['fixed_targets_sha256']
  save(ROOT/'teachers/base.pt',before)
  for item in P['validation']:
   s,receipt=mem(item)
   for arm in P['arms']:install(f'{arm}-seed{item["seed"]}',states[arm],{str(item['seed']):s},'meta')
  write(ROOT/'UPDATE_COMPARISON.json',dict(passed=True,source_teacher_hash=tree_hash(before),new_batch_hash=fh,macro_samples=10,results=results,teacher_hashes={k:tree_hash(v) for k,v in states.items()},fixed_behavior_and_targets=True))
 elif x.op=='verify':
  d=ROOT/'teachers'/x.stage;s=load(d/'current.pt');i=load(d/'initial.pt')
  for k in ['net','optimizer','rng']:assert tree_hash(s['learner'][k])==tree_hash(i['learner'][k])
  assert s['students'][str(x.seed)]['pending'] is None
  write(ROOT/'freeze_checks'/f'{x.stage}.json',dict(passed=True,teacher_hash=tree_hash(s['learner'])))
 elif x.op=='report':
  arms={};baseline={}
  for item in P['validation']:
   cp=load(item['checkpoint']);baseline[str(item['seed'])]=cp['infos']['dev_before']
   for arm in P['arms']:
    label=f'validate-{arm}-seed{item["seed"]}';ev=[]
    for p in sorted((ROOT/'evaluations').glob(label+'-*/result.json')):
     d=json.loads(p.read_text());assert d['kind']=='development';ev.append({k:d[k] for k in ['update','continuous_joint_score','joint_success_rate','pooled_rmse','native_failure_rate','safety'] if k in d})
    assert [e['update'] for e in ev]==[5100,5200];arms[label]=ev
  executed={p.parent.name:sum(1 for _ in p.open()) for p in (ROOT/'runs').glob('*/updates.jsonl')};assert sum(executed.values())==2200
  evals=[json.loads(p.read_text()) for p in (ROOT/'evaluations').glob('*/result.json')]
  write(ROOT/'RESULTS.json',dict(primary_endpoint=200,baseline=baseline,arms=arms,descriptive_local_calibration_only=True))
  write(ROOT/'COSTS.json',dict(student_updates=2200,environment_transitions=2200*4096*24,by_segment=executed,development_transitions=sum(e['executed_environment_transitions'] for e in evals),development_wall_seconds=sum(e['wall_seconds'] for e in evals),jobs=[json.loads(p.read_text()) for p in (ROOT/'jobs').glob('*/result.json')],note='Parent Student wall includes eval/RPC waits; nested wall is not additive elapsed time.'))
if __name__=='__main__':main()
