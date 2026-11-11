"""Install frozen inference states. Only immutable source checkpoints may be used."""
import argparse,copy,json
from pathlib import Path
import torch
from common import ROOT,write
from checkpoint_pair import pair,sha
from frozen_ops import save,load
from cl1_core import Learner
from cpu_hash import tree_hash
OLD=Path('/data/go2_piper_nsgs_cl1_gen_seed70159_20261110')
REPAIR=Path('/data/go2_piper_cl1_repair_20261110')
BASE=Path('/data/go2_piper_nsgs_cl1_20261109')
def marker():return dict(cl1_proxy=1,binding=json.loads((ROOT/'prepared/manifest.json').read_text())['binding'])
def install(arm,state,ref=None):
 d=ROOT/'teachers'/arm;assert not d.exists();d.mkdir(parents=True)
 state=copy.deepcopy(state);state['eligible_for_on_policy_teacher_update']=False;state['execution_kind']='uniform' if arm in ('uniform','probe') else 'meta'
 save(d/'current.pt',state);save(d/'initial.pt',state)
 save(d/'proxy.pt',dict(teacher=marker(),resume_teacher_snapshot=ref))
 return dict(network_sha256=tree_hash(state['learner']['net']),optimizer_sha256=tree_hash(state['learner']['optimizer']),version=state['learner']['version'])
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['initial','recovery','restore','verify']);p.add_argument('--arm');p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
 if a.mode=='initial':
  bundle=pair(OLD/'runs/uniform-seed70159-to01500/checkpoints/update_01400.pt')
  ref=dict(path=bundle['teacher_path'],sha256=bundle['teacher_sha256'])
  s=load(bundle['teacher_path']);arms={}
  for arm in ('uniform','probe'):arms[arm]=install(arm,s,ref)
  s=load(OLD/'teachers/frozen-base/initial.pt');assert not s['students']['70159']['history']
  arms['frozen-base']=install('frozen-base',s)
  write(ROOT/'INITIAL_LOCK.json',dict(seed=70159,arms=arms,paired_resume=bundle,legacy_prefix_max_update=1400))
 elif a.mode=='recovery':
  accept=json.loads((REPAIR/'REPAIR_RECOVERY_ACCEPTANCE.json').read_text());assert accept['passed']
  src=REPAIR/'teachers/recovery/RecoveryV2.pt';assert sha(src)==accept['teacher_sha256']
  t=load(src)['learner'];state=dict(learner=t,students={'70159':dict(history=[],pending=None,previous=None,transitions=[],cursor=0)},batches=0)
  lock=install('frozen-recovery',state);write(ROOT/'RECOVERY_V2_LOCK.json',dict(source=str(src),sha256=sha(src),**lock))
 elif a.mode=='restore':
  bundle=pair(a.checkpoint);d=ROOT/'teachers'/a.arm;source=load(bundle['teacher_path']);initial=load(d/'initial.pt')
  assert bundle['transfer_kind']==('uniform' if initial['execution_kind']=='uniform' else 'meta')
  allowed=('probe','uniform') if a.arm=='probe' else (a.arm,)
  assert any(a.checkpoint.parent.parent.name.startswith(k+'-seed70159-') for k in allowed), 'Cross-arm Student checkpoint rejected'
  for k in ('net','optimizer'):assert tree_hash(source['learner'][k])==tree_hash(initial['learner'][k])
  source['execution_kind']=initial['execution_kind']
  assert a.out is not None and not a.out.exists()
  # Never edit an archived source state; current.pt is an explicitly mutable working pointer.
  save(a.out.with_name(a.out.stem+'-previous.pt'),load(d/'current.pt'))
  save(d/'current.pt',source)
  save(a.out,dict(teacher=marker(),resume_teacher_snapshot=dict(path=bundle['teacher_path'],sha256=bundle['teacher_sha256'])))
  write(a.out.with_suffix('.json'),bundle)
 else:
  d=ROOT/'teachers'/a.arm;s=load(d/'current.pt');initial=load(d/'initial.pt')
  for k in ('net','optimizer','rng'):assert tree_hash(s['learner'][k])==tree_hash(initial['learner'][k]),k
  write(ROOT/'freeze_checks'/f'{a.arm}.json',dict(passed=True,network_sha256=tree_hash(s['learner']['net']),
      optimizer_unchanged=True,rng_unchanged=True,version=s['learner']['version'],history=len(s['students']['70159']['history'])))
if __name__=='__main__':main()
