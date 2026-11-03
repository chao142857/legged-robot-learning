"""Small read-only engineering trace; no learning and no performance-based selection."""
import argparse,json,pathlib,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);app=launcher.app
out=pathlib.Path('/workspace/artifacts/structure');out.mkdir(parents=True,exist_ok=True);env=None
try:
 import numpy as np,torch,gymnasium as gym
 import LeggedManip_Lab.tasks
 from rsl_rl.runners import OnPolicyRunner
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from common import Pool,model_cfg,write,tensor_hash,file_sha
 from commands import SequenceWBC,SequenceVelocity
 from adapter import SequenceAdapter
 cfg,agent=model_cfg('GO2-PIPER-WBC');cfg.scene.num_envs=16;cfg.seed=20261104;cfg.sim.device='cuda:0'
 cfg.commands.ee_pose.class_type=SequenceWBC;cfg.commands.base_velocity.class_type=SequenceVelocity
 env=gym.make('GO2-PIPER-WBC',cfg=cfg);raw=env.unwrapped;events=[]
 class Trace(SequenceAdapter):
  def inputs(self,kind,ids):
   slots=self.cursor[kind][ids].cpu().tolist();env_ids=ids.cpu().tolist()
   for i,s in zip(env_ids,slots):
    if self.owner.generation[i]==1:events.append(dict(kind=kind,env_id=i,sequence_id=int(self.owner.task[i]),slot=s,control_step=self.owner.control_steps,episode_step=int(self.owner.steps[i]),native_episode_length=int(self.env.episode_length_buf[i])))
   return super().inputs(kind,ids)
 pool=Pool('wbc','train');pool.ee=pool.ee[:16];pool.velocity=pool.velocity[:16];pool.n=16;pool.identity+=':structure16'
 adapter=Trace(raw,'wbc','evaluation',pool,out);wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
 term=raw.command_manager.get_term('ee_pose');robot=raw.scene['robot']
 refs=dict(link0_positions=robot.data.body_pos_w[:,term.link0_idx].cpu().tolist(),link0_quaternions=robot.data.body_quat_w[:,term.link0_idx].cpu().tolist(),selection='first4 reset states in fixed16-env seed20261104, captured before loading policy or observing outcomes')
 write(out/'reset_reference_states.json',refs)
 cp='/continue_reference/runs/continue-wbc-uniform/policy/model_14999.pt';runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=agent.device);runner.load(cp)
 policy=runner.get_inference_policy(device=raw.device);before=tensor_hash(runner.alg.actor.state_dict());obs=wrapped.get_observations();samples=[]
 with torch.inference_mode():
  for step in range(1002):
   for i in range(16):
    if adapter.owner.generation[i]==1:
     samples.append(dict(env_id=i,episode_step=int(adapter.owner.steps[i]),ee_slot=int(adapter.cursor['ee'][i])-1,velocity_slot=int(adapter.cursor['velocity'][i])-1,ee_time_left=float(term.time_left[i]),velocity_time_left=float(raw.command_manager.get_term('base_velocity').time_left[i]),standing=bool(raw.command_manager.get_term('base_velocity').is_standing_env[i]),velocity=raw.command_manager.get_command('base_velocity')[i].cpu().tolist()))
   obs,_,dones,_=wrapped.step(policy(obs));policy.reset(dones)
   if len(adapter.first_episode_rows)==16:break
 assert len(adapter.first_episode_rows)==16 and before==tensor_hash(runner.alg.actor.state_dict())
 assert sum(adapter.owner.engine.completed_counts)==0
 write(out/'resample_events.json',events);write(out/'control_samples.json',samples)
 write(out/'result.json',dict(passed=True,environments=16,control_steps=step+1,training_updates=0,training_history_updates=0,checkpoint_sha256=file_sha(cp),actor_unchanged=True,effective=adapter.effective(),first_episode_lengths=[int(r['steps']) for r in adapter.first_episode_rows]))
 print('STRUCTURE_TRACE_PASS '+json.dumps(events),flush=True)
except BaseException:
 (out/'fault.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2));raise
finally:
 if env is not None:env.close()
 app.close()
