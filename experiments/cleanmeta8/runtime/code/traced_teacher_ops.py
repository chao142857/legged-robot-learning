"""Run original Teacher CLI with observational full-state Adam traces.
No change to gradient, candidates, KL acceptance, LR or checkpoint semantics.
"""
import copy,inspect,sys,hashlib
from pathlib import Path
import torch
import teacher_ops,teacher_v02
from common import ROOT,write,file_sha
from cpu_hash import tree_hash

original=teacher_v02.Teacher.learn_pair

def traced(self,*args,**kwargs):
 out=Path(sys.argv[sys.argv.index('--out')+1]);before=self.state();candidates=[];native=self.optimizer.step
 identity=dict(module_file=str(Path(teacher_v02.__file__).resolve()),module_sha256=file_sha(Path(teacher_v02.__file__)),
  learn_pair_file=inspect.getsourcefile(original),learn_pair_line=original.__code__.co_firstlineno,
  teacher_ops_file=teacher_ops.__file__,teacher_ops_sha256=file_sha(Path(teacher_ops.__file__)),
  torch_version=torch.__version__,fixed_runtime=True)
 expected=__import__('json').loads((ROOT/'CODE_BINDING.json').read_text())['files']['code/teacher_v02.py']
 assert identity['module_sha256']==expected
 def step(*a,**kw):
  row=dict(network_before=copy.deepcopy(self.net.state_dict()),optimizer_before=copy.deepcopy(self.optimizer.state_dict()),
   gradients=[None if p.grad is None else p.grad.detach().clone() for p in self.net.parameters()])
  candidates.append(row);result=native(*a,**kw)
  row.update(network_after=copy.deepcopy(self.net.state_dict()),optimizer_after=copy.deepcopy(self.optimizer.state_dict()))
  return result
 self.optimizer.step=step
 try:
  response=original(self,*args,**kwargs);after=self.state()
 except BaseException as exc:
  torch.save(dict(before=before,candidates=candidates,after_failure=self.state(),source_identity=identity,error=repr(exc)),out/'adam_fault_trace.pt')
  write(out/'adam_fault.json',dict(error=repr(exc),source_identity=identity,candidates_recorded=len(candidates),full_trace_sha256=file_sha(out/'adam_fault_trace.pt')))
  raise
 finally:self.optimizer.step=native
 assert len(candidates)==len(response['step_backtracking'])
 for row in candidates:
  assert tree_hash(row['network_before'])==tree_hash(before['network'])
  assert tree_hash(row['optimizer_before']['state'])==tree_hash(before['optimizer']['state'])
  for old,new in zip(before['optimizer']['param_groups'],row['optimizer_before']['param_groups']):
   assert {k:v for k,v in old.items() if k!='lr'}=={k:v for k,v in new.items() if k!='lr'}
 accepted=not response['zero_signal'] and not response['step_rejected']
 if accepted:
  assert tree_hash(after['optimizer']['state'])==tree_hash(candidates[0]['optimizer_after']['state'])
  old=before['optimizer']['state'];new=after['optimizer']['state']
  assert all(int(v['step'])==(int(old[k]['step']) if k in old else 0)+1 for k,v in new.items())
 else:
  for k in ('network','optimizer'):assert tree_hash(after[k])==tree_hash(before[k])
 assert after['optimizer']['param_groups']==before['optimizer']['param_groups']
 for k in ('rng','encoder','config','binding'):assert tree_hash(after[k])==tree_hash(before[k])
 path=out/'adam_full_trace.pt';assert not path.exists()
 torch.save(dict(before=before,candidates=candidates,after=after,response=response,source_identity=identity),path)
 write(out/'adam_acceptance.json',dict(passed=True,accepted=accepted,rejected=response['step_rejected'],zero_signal=response['zero_signal'],
  logical_before=before['updates'],logical_after=after['updates'],
  adam_steps_before=sorted(set(int(v['step']) for v in before['optimizer']['state'].values())),
  adam_steps_after=sorted(set(int(v['step']) for v in after['optimizer']['state'].values())),
  before_full_state_sha256=tree_hash(before),after_full_state_sha256=tree_hash(after),
  candidates=response['step_backtracking'],full_trace_sha256=file_sha(path),source_identity=identity,
  every_candidate_same_old_state=True,accepted_exactly_one_state_step=accepted,base_param_groups_restored=True))
 return response

if __name__=='__main__':
 teacher_v02.Teacher.learn_pair=traced
 try:teacher_ops.main()
 finally:teacher_v02.Teacher.learn_pair=original
