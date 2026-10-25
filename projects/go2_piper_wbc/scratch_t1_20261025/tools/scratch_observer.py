"""Read-only scratch telemetry around the installed native RSL-RL PPO."""
import json
import math
import os
import time
from pathlib import Path

import torch

from telemetry import TrainingTelemetry, plain, sha, write_json, model_sha


def command_observations(raw, obs):
    """Extract newest command samples from the actual concatenated input."""
    result = {}
    manager = raw.observation_manager
    for group in ('policy', 'critic'):
        offset = 0
        for name, shape in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
            width = math.prod(shape)
            if name in ('velocity_commands', 'pos_commands'):
                n = 3 if name == 'velocity_commands' else 7
                result[group+'/'+name] = obs[group][:, offset:offset+width].reshape(raw.num_envs, -1, n)[:, -1]
            offset += width
    return result


class InterfaceObserver:
    """Checks observation -> action -> reward commands and pre-reset masks."""
    def __init__(self, env, runner=None):
        self.env, self.raw = env, env.unwrapped
        self.original_step = env.step
        self.original_reward = self.raw.reward_manager.compute
        self.steps = 0
        self.failures = []
        self.max_error = {}
        self.termination_counts = {}
        self.last = None
        self.input_obs = None
        self.input_step = -1
        self.runner = runner
        self.original_act = runner.alg.act if runner is not None else None
        if runner is not None:
            runner.alg.act = self.act
        env.step = self.step
        self.raw.reward_manager.compute = self.reward

    def equal(self, key, a, b, atol=2e-5):
        err = float((a-b).abs().max()) if a.numel() else 0.
        self.max_error[key] = max(self.max_error.get(key, 0.), err)
        if not math.isfinite(err) or err > atol:
            self.failures.append({'key':key,'max_abs_error':err})
            raise RuntimeError(f'Data pipeline mismatch {key}: {err}')

    def capture_input(self, obs):
        # Capture the exact TensorDict supplied by the runner/evaluator. Never
        # call get_observations(): that API recomputes noisy observations.
        self.input_obs = obs
        self.input_step = self.steps

    def act(self, obs, *args, **kwargs):
        self.capture_input(obs)
        return self.original_act(obs, *args, **kwargs)

    def check_input(self):
        r = self.raw
        assert self.input_step == self.steps, 'Actual actor input was not captured for this step'
        obs = self.input_obs
        for name, value in obs.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError('Nonfinite observation '+name)
        self.before = {name:r.command_manager.get_command(name).clone() for name in ('base_velocity','ee_pose')}
        if self.steps < 128 or self.steps % 240 == 0:
            for key, value in command_observations(r, obs).items():
                self.equal(key, value, self.before['base_velocity' if key.endswith('velocity_commands') else 'ee_pose'])

    def step(self, actions):
        self.check_input()
        result = self.original_step(actions)
        obs, reward, done, extras = result
        self.equal('done_mask', done.float(), self.last['done'].float(), 0.)
        if 'time_outs' in extras:
            self.equal('timeout_mask', extras['time_outs'].float(), self.last['timeout'].float(), 0.)
        self.steps += 1
        return result

    def reward(self, *a, **kw):
        result = self.original_reward(*a, **kw)
        r = self.raw
        self.last = {'done':r.reset_buf.clone(), 'timeout':r.reset_time_outs.clone()}
        for name in r.termination_manager.active_terms:
            count = r.termination_manager.get_term(name).sum()
            self.termination_counts[name] = self.termination_counts.get(name, torch.zeros_like(count)) + count
        if self.steps < 128 or self.steps % 240 == 0:
            for name in self.before:
                self.equal('reward_command/'+name, r.command_manager.get_command(name), self.before[name], 0.)
            from telemetry import physical_suite
            p = physical_suite(r)
            expected = {
                'end_effector_position_tracking_exp':torch.exp(-p['ee_position_error'].square()/.1),
                'end_effector_orientation_tracking':p['ee_orientation_reward_error'],
                'track_lin_vel_xy_exp':torch.exp(-p['xy_error_sq']/.25),
                'track_ang_vel_z_exp':torch.exp(-p['yaw_error_sq']/.25),
            }
            for name, value in expected.items():
                cfg = r.reward_manager.get_term_cfg(name)
                i = r.reward_manager.active_terms.index(name)
                self.equal('reward_value/'+name, r.reward_manager._step_reward[:,i], value*cfg.weight)
            self.equal('orientation_root_link0',p['ee_orientation_reward_error'],p['ee_orientation_error'])
            mask = r.reset_terminated | r.reset_time_outs
            self.equal('termination_or_timeout',r.reset_buf.float(),mask.float(),0.)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError('Nonfinite reward')
        return result

    def report(self):
        return {'steps_checked':self.steps,'max_abs_errors':self.max_error,
                'termination_counts':plain(self.termination_counts),'passed':not self.failures,
                'failures':self.failures,
                'scope':'actual actor input/command/reward values and pre-reset done masks; observer itself performs no policy update'}

    def close(self):
        if self.runner is not None:
            self.runner.alg.act = self.original_act
        self.env.step = self.original_step
        self.raw.reward_manager.compute = self.original_reward


class ScratchTelemetry(TrainingTelemetry):
    """Native losses/optimizer/scheduler unchanged; KL alone never stops a run."""
    checkpoints = {500,1000,2000,3000,5000}

    def save(self, filename):
        target = self.out/filename
        if target.exists():
            raise FileExistsError(target)
        saved = self.runner.alg.save()
        saved.update(iter=self.update_count-1, infos={
            'completed_updates':self.update_count,
            'initialization':'from scratch: new actor, critic, Gaussian distribution, Adam and counter',
            'environment_steps':self.update_count*self.env.num_envs*self.runner.cfg['num_steps_per_env'],
            'task_suite':plain(self.env.cfg.task_suite),
            'rng_cpu':torch.get_rng_state(), 'rng_cuda':torch.cuda.get_rng_state_all(),
            'learning_rate':self.runner.alg.learning_rate})
        tmp = target.with_suffix('.partial')
        torch.save(saved,tmp)
        os.replace(tmp,target)
        write_json(target.with_suffix('.json'),dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,
            completed_updates=self.update_count,actor_sha256=model_sha(self.runner.alg.actor),
            critic_sha256=model_sha(self.runner.alg.critic),optimizer_state_entries=len(self.runner.alg.optimizer.state)))

    def reward(self,*a,**kw):
        result=super().reward(*a,**kw)
        with torch.no_grad():
            r=self.env; p=r.cfg.task_suite['params']; d=r.scene['robot'].data
            cmd=r.command_manager.get_command('base_velocity')
            act=torch.cat((d.root_lin_vel_b[:,:2],d.root_ang_vel_b[:,2:3]),-1)
            pose=r.command_manager.get_term('ee_pose')
            from isaaclab.utils.math import quat_apply_inverse
            rel=quat_apply_inverse(d.body_quat_w[:,pose.link0_idx],d.body_pos_w[:,pose.body_idx]-d.body_pos_w[:,pose.link0_idx])
            actual=torch.cat((rel[:,:2],d.body_pos_w[:,pose.body_idx,2:3]),-1)
            target=pose.command[:,:3]
            center=torch.tensor(p['ee_center'],device=r.device)
            row=self.steps[-1]
            for j,axis in enumerate('xyz'):
                row['ee_axis/'+axis+'/target_mean']=target[:,j].mean()
                row['ee_axis/'+axis+'/actual_mean']=actual[:,j].mean()
                row['ee_axis/'+axis+'/mse']=(actual[:,j]-target[:,j]).square().mean()
                row['ee_axis/'+axis+'/center_counterfactual_mse']=(target[:,j]-center[j]).square().mean()
                row['ee_axis/'+axis+'/offset_cross']=((actual[:,j]-center[j])*(target[:,j]-center[j])).mean()
            age=(r.episode_length_buf-1).clamp_min(0)*r.step_dt
            group=(torch.arange(r.num_envs,device=r.device)+(age/p['base_segment_s']).long())%4
            for g,name in enumerate(('forward','backward','left','right')):
                axis=0 if g<2 else 2
                desired=(p['base_linear_speed'] if g<2 else p['base_yaw_rate'])*(1 if g in (0,2) else -1)
                mask=(group==g)&((cmd[:,axis]-desired).abs()<1e-5)
                w=mask.float()
                row['direction/'+name+'/samples']=w.sum()
                row['direction/'+name+'/command_sum']=(w*cmd[:,axis]).sum()
                row['direction/'+name+'/actual_sum']=(w*act[:,axis]).sum()
                row['direction/'+name+'/error_sq_sum']=(w*(act[:,axis]-cmd[:,axis]).square()).sum()
                row['direction/'+name+'/command_sq_sum']=(w*cmd[:,axis].square()).sum()
                row['direction/'+name+'/cross_sum']=(w*act[:,axis]*cmd[:,axis]).sum()
                row['direction/'+name+'/right_sign_sum']=(w*(act[:,axis]*desired>0).float()).sum()
            row['done_episode_age_steps_sum']=(r.episode_length_buf*r.reset_buf).float().sum()
        return result

    def update(self,*a,**kw):
        result=self.original_update(*a,**kw)
        self.update_count+=1
        metrics={k:float(torch.stack([s[k] for s in self.steps]).mean()) for k in self.steps[0]}
        kls=torch.stack(self.kls); grads=torch.stack(self.grads)
        row=dict(completed_updates=self.update_count,
            env_steps=self.update_count*self.env.num_envs*self.runner.cfg['num_steps_per_env'],
            elapsed_s=time.monotonic()-self.started,losses=plain(result),
            learning_rate=self.runner.alg.learning_rate,optimizer_lrs=[x['lr'] for x in self.runner.alg.optimizer.param_groups],
            action_std=plain(self.runner.alg.actor.output_std[0]),kl_mean=float(kls.mean()),kl_max=float(kls.max()),
            actor_grad_norm_mean=float(grads[0::2].mean()),actor_grad_norm_max=float(grads[0::2].max()),
            critic_grad_norm_mean=float(grads[1::2].mean()),critic_grad_norm_max=float(grads[1::2].max()),metrics=metrics,
            cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_max_reserved_bytes=torch.cuda.max_memory_reserved())
        def finite(x):
            if isinstance(x,dict): return all(finite(v) for v in x.values())
            if isinstance(x,list): return all(finite(v) for v in x)
            return not isinstance(x,float) or math.isfinite(x)
        if not finite(row) or not all(bool(torch.isfinite(p).all()) for m in (self.runner.alg.actor,self.runner.alg.critic) for p in m.parameters()):
            write_json(self.out/'nonfinite_diagnostic.json',{'update':self.update_count,'data_repr':repr(row)})
            self.save('emergency_nonfinite.pt')
            raise RuntimeError('Nonfinite PPO state or telemetry; no recipe change')
        self.bad_kl=self.bad_kl+1 if row['kl_mean']>.1 else 0
        row['consecutive_high_kl_updates']=self.bad_kl
        row['optimization_review_flag']=self.bad_kl>=20
        row['stop_policy']='nonfinite/OOM/simulation/pipeline error; high KL flag requires behavior review, never stops alone'
        self.log.write(json.dumps(row,allow_nan=False)+'\n')
        tmp=self.out/'latest.partial.json'; write_json(tmp,row); os.replace(tmp,self.out/'latest.json')
        if self.update_count in self.checkpoints:
            self.save(f'cp_{self.update_count}.pt')
        self.steps.clear();self.kls.clear();self.grads.clear()
        return result
