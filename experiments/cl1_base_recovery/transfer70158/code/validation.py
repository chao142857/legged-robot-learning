"""Fixed JIT warmup followed by two strict read-only command equivalence checks."""
import torch
from commands import UniformPoseWBCCommand,_native_wbc_fixed
from command_check_math import compare_commands

def command_equivalence(term):
 ids=torch.arange(min(term.num_envs,128),device=term.device)
 u=term._env.sequence_adapter.pool.ee_tensor[torch.arange(len(ids),device=term.device),0]
 before=term.pose_command.clone();saved_inputs=term._sequence_inputs;original_rand=torch.rand
 rounds=[];natives=[];actuals=[]
 for index in range(3):
  calls=[]
  def fixed_rand(*args,**kwargs):
   assert tuple(args)==(len(ids),3)
   value=u[:,:3] if len(calls)==0 else u[:,3:6];calls.append(1);return value
  try:
   torch.rand=fixed_rand;UniformPoseWBCCommand._resample_command(term,ids)
   native=term.pose_command[ids].clone()
  finally:
   torch.rand=original_rand;term.pose_command.copy_(before)
  try:
   term._sequence_inputs=u;_native_wbc_fixed(term,ids);actual=term.pose_command[ids].clone()
  finally:
   term._sequence_inputs=saved_inputs;term.pose_command.copy_(before)
  rounds.append(compare_commands(native,actual,len(calls)));natives.append(native);actuals.append(actual)
 native_repeat=compare_commands(natives[1],natives[2]);actual_repeat=compare_commands(actuals[1],actuals[2])
 result=dict(rounds[-1]);result.update(passed=all(x['passed'] for x in rounds[1:]) and native_repeat['passed'] and actual_repeat['passed'],
  fixed_warmup_pairs=1,strict_verification_pairs=2,all_rounds=rounds,native_repeat=native_repeat,actual_repeat=actual_repeat,
  native_first=natives[-1][:3].cpu().tolist(),actual_first=actuals[-1][:3].cpu().tolist(),
  unchanged_tolerances=True,read_only_pose_restored=bool(torch.equal(term.pose_command,before)),random_draws_replaced_only=True)
 assert result['read_only_pose_restored'] and term._sequence_inputs is saved_inputs
 return result
