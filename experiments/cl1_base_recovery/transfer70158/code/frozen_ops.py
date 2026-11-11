"""Inference-only Teacher RPC. There is deliberately NO update command."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from cl1_core import Learner,observation,replay
from teacher_v02 import FeatureEncoder,Teacher as Decoder
from neural_core import distribution_stats
from common import ROOT,write,file_sha
from cpu_hash import tree_hash

def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('op',choices=['request']);ap.add_argument('--stage',required=True);ap.add_argument('--request',type=Path,required=True);a=ap.parse_args()
 path=ROOT/'teachers'/a.stage/'current.pt';state=load(path);initial=load(path.with_name('initial.pt'))
 l=Learner();l.load(state['learner']);identity=(tree_hash(l.net.state_dict()),tree_hash(l.optimizer.state_dict()))
 assert identity==(tree_hash(initial['learner']['net']),tree_hash(initial['learner']['optimizer']))
 req=load(a.request);s=state['students'][str(req['seed'])];v=req['version'];kind=state['execution_kind']
 if req['kind']=='decide':
  assert s['pending'] is None
  x=observation(req['global_state'],req['dev'],s['previous'],req['dynamic'])
  if s['transitions']:assert np.allclose(x,s['transitions'][-1]['next_x'],atol=1e-6)
  h=replay(l.net,s['history'])
  with torch.no_grad():mean,value,hidden=l.net(torch.as_tensor(x),h)
  preference=np.zeros(8) if kind=='uniform' else mean.numpy().copy()
  action=dict(action=preference,mean=mean.numpy().copy(),value=float(value),logprob=None,
    behavior_distribution='uniform_override' if kind=='uniform' else 'deterministic_mean',
    eligible_for_on_policy_teacher_update=False,version=l.version,hidden_before=h.numpy())
  enc=FeatureEncoder(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False))
  encoded=enc.encode(req['dynamic'],req['progress'],req['progress_observed'],req['partial'])
  decoder=object.__new__(Decoder);decoder.encoder=enc;decoder.config=enc.config
  q=decoder.probability_from_preference(encoded['basis'],preference)
  if kind=='uniform':q=np.full(enc.n,1/enc.n)
  s['pending']=dict(**action,x=x,development_before=req['dev'],macro_version=v);s['history'].append(x)
  s['current_hidden']=hidden.numpy()
  response=dict(version=v,**encoded,global_state=req['global_state'],preference=preference,
    probability=q,distribution=distribution_stats(q),macro_action=action,teacher_version=l.version,
    teacher_sha256=identity[0],deterministic_frozen_mean=True,execution_kind=kind)
 else:
  t=s['pending'];assert t is not None and t['macro_version']==v and t['version']==l.version
  reward=float(req['dev']['continuous_joint_score']-t['development_before']['continuous_joint_score'])
  previous=dict(dev=t['development_before'],action=t['action'])
  t.update(reward=reward,development_after=req['dev'],completed_updates=req['checkpoint_update'],
     next_x=observation(req['global_state'],req['dev'],previous,req['dynamic']),eligible_for_on_policy_teacher_update=False)
  s['transitions'].append(t);s['pending']=None;s['previous']=previous
  response=dict(passed=True,version=l.version,reward=reward)
 assert identity==(tree_hash(l.net.state_dict()),tree_hash(l.optimizer.state_dict()))
 # No sampling is required for deterministic inference; Teacher RNG is unchanged too.
 assert tree_hash(l.state())==tree_hash(state['learner'])
 evidence=a.request.with_suffix('.teacher_after.pt');assert not evidence.exists()
 save(evidence,state);save(path,state);save(a.request.with_suffix('.done.pt'),response)
 write(a.request.with_suffix('.runtime.json'),dict(module=__file__,sha256=file_sha(__file__),teacher_net_sha256=identity[0],
    adam_sha256=identity[1],frozen=True,execution_kind=kind,gaussian_logprob_not_applicable=True))
if __name__=='__main__':main()
