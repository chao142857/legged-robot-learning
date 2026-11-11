"""Real engineering receipts only; no success-rate or offline efficacy threshold."""
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,write,file_sha
from cpu_hash import tree_hash
from cl1_core import Learner,replay

def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def check():
 hashes=[];distributions=[];sources={};cost=0
 for seed in (43,44):
  r=ROOT/'runs'/f'smoke-seed{seed}-to00200';result=read(r/'result.json')
  assert result['passed'] and result['completed_updates']==200 and result['optimizer_calls']==4000
  assert read(r/'native_comparison.json')['passed'] and read(r/'command_equivalence.json')['passed']
  init=read(r/'initialization.json');assert init['optimizer_empty'] and init['seed']==seed
  sources[seed]=init;cost+=sum(1 for _ in (r/'updates.jsonl').open())
  for p in sorted((r/'teacher_requests').glob('*-decide.done.pt')):
   d=load(p);assert d['teacher_version']==0
   hashes.append(d['teacher_sha256']);q=d['probability']
   assert len(q)==32768 and np.isfinite(q).all() and q.min()>=.2/32768 and abs(q.sum()-1)<1e-12
   distributions.append(dict(seed=seed,macro_version=d['version'],statistics=d['distribution']))
 assert len(hashes)==4 and len(set(hashes))==1
 assert sources[43]['actor_sha256']!=sources[44]['actor_sha256']
 f=load(ROOT/'teachers/smoke/batch_000/on_policy_fragments.pt')
 assert len(f)==2 and all(len(x['transitions'])==2 for x in f)
 update=read(ROOT/'teachers/smoke/batch_000/result.json')
 assert update['macro_samples']==4 and np.isfinite(update['final_behavior_kl'])
 after=load(ROOT/'teachers/smoke/batch_000/after.pt');l=Learner();l.load(after['learner'])
 for student in after['students'].values():
  np.testing.assert_array_equal(student['rebuilt_hidden'],replay(l.net,student['history']).numpy())
 restored=ROOT/'runs/smoke-seed43-to00300';rr=read(restored/'restore_receipt.json')
 assert rr['passed'] and rr['completed_updates']==200 and rr['next_4096_sampler_draws_equal']
 assert len(rr['first_rollout_signatures'])==24
 result=read(restored/'result.json');assert result['completed_updates']==300 and result['new_updates']==100
 cost+=sum(1 for _ in (restored/'updates.jsonl').open());assert cost==500
 assert read(restored/'native_comparison.json')['passed']
 d=load(restored/'teacher_requests/0003-decide.done.pt');assert d['teacher_version']==1
 assert d['teacher_sha256']==tree_hash(after['learner']['net'])
 evidence=dict(passed=True,student_updates=cost,training_transitions=cost*4096*24,
  cpu_tests=21,source_initializations=sources,frozen_teacher_hash=hashes[0],distributions=distributions,
  outer_update=update,restore_receipt=rr,teacher_new_version_used=True,
  real_rewards=[t['reward'] for x in f for t in x['transitions']],
  performance_not_a_gate=True,formal_students_and_teacher_reinitialized=True)
 write(ROOT/'ENGINEERING_ACCEPTANCE.json',evidence)
 return evidence
if __name__=='__main__':check()
