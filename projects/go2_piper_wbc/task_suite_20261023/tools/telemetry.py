"""Telemetry adapted from the verified 2026-10-23 native observer.

Read-only telemetry around unmodified upstream training calls.

No new random draws, environment resets, reward/command writes, policy forwards,
optimizer changes, or gradients are introduced. clip_grad_norm_ and KL wrappers
return the original result. Checkpoint serialization adds no optimizer updates.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_error_magnitude, euler_xyz_from_quat


def plain(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "to_dict"):
        return plain(x.to_dict())
    if callable(x):
        return f"{x.__module__}:{getattr(x, '__qualname__', type(x).__name__)}"
    return str(x)


def write_json(path, data):
    Path(path).write_text(json.dumps(plain(data), indent=2, ensure_ascii=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def model_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def commands(env):
    result = {}
    for name in env.command_manager.active_terms:
        term = env.command_manager.get_term(name)
        cmd = term.command.detach()
        result[name] = dict(class_name=f"{type(term).__module__}.{type(term).__name__}",
                            cfg=plain(term.cfg), min=plain(cmd.amin(0)), max=plain(cmd.amax(0)),
                            mean=plain(cmd.mean(0)), sample=plain(cmd[:16]),
                            time_left=plain(term.time_left[:16]), command_counter=plain(term.command_counter[:16]))
    return result


def snapshot(env, runner=None):
    result = dict(common_step_counter=env.common_step_counter,
                  episode_length_s=env.max_episode_length_s, max_episode_length=env.max_episode_length,
                  step_dt=env.step_dt, num_envs=env.num_envs,
                  action_dim=env.action_manager.total_action_dim,
                  observation_dims=plain(env.observation_manager.group_obs_dim),
                  commands=commands(env), curriculum=plain(env.cfg.curriculum),
                  rewards=plain(env.cfg.rewards), terminations=plain(env.cfg.terminations),
                  reset_randomization=plain(env.cfg.events),
                  contact_sensor=plain(env.scene.sensors["contact_forces"].cfg),
                  joint_names=env.scene["robot"].joint_names,
                  contact_body_names=env.scene.sensors["contact_forces"].body_names,
                  forbidden_attributes=[k for k in vars(env) if any(s in k.lower() for s in ("task_id", "task_pool", "sgs", "trajectory"))],
                  imported_modules={k: v.__file__ for k, v in sys.modules.copy().items()
                                    if k.startswith("LeggedManip_Lab") and getattr(v, "__file__", None)})
    if runner is not None:
        alg = runner.alg
        result.update(runner_class=f"{type(runner).__module__}.{type(runner).__name__}",
                      runner_cfg=plain(runner.cfg), optimizer=type(alg.optimizer).__name__,
                      optimizer_state_entries=len(alg.optimizer.state),
                      optimizer_defaults=plain(alg.optimizer.defaults),
                      actor_normalizer=str(alg.actor.obs_normalizer), critic_normalizer=str(alg.critic.obs_normalizer),
                      actor_distribution=str(alg.actor.distribution), critic_distribution=str(alg.critic.distribution),
                      actor_sha256=model_sha(alg.actor), critic_sha256=model_sha(alg.critic),
                      algorithm_source=inspect.getfile(type(alg)), runner_source=inspect.getfile(type(runner)))
    return result


def physical(env):
    """Current pre-reset state and the commands that produced this step's reward."""
    robot = env.scene["robot"]
    d = robot.data
    term = env.command_manager.get_term("ee_pose")
    cmd = term.command
    velcmd = env.command_manager.get_command("base_velocity")
    ee = d.body_pos_w[:, term.body_idx]
    eeq = d.body_quat_w[:, term.body_idx]
    linkq = d.body_quat_w[:, term.link0_idx]
    ee_link = quat_apply_inverse(linkq, ee - d.body_pos_w[:, term.link0_idx])
    actual_mixed = torch.cat((ee_link[:, :2], ee[:, 2:3]), dim=-1)
    pos_err = torch.linalg.vector_norm(actual_mixed - cmd[:, :3], dim=-1)
    # Both conventions are retained: the native reward uses root; command metrics use link0.
    ori_err_reward = quat_error_magnitude(eeq, quat_mul(d.root_quat_w, cmd[:, 3:7]))
    ori_err_command = quat_error_magnitude(eeq, quat_mul(linkq, cmd[:, 3:7]))
    vel = torch.cat((d.root_lin_vel_b[:, :2], d.root_ang_vel_b[:, 2:3]), dim=-1)
    sensor = env.scene.sensors["contact_forces"]
    force_hist = sensor.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
    feet = [i for i, name in enumerate(sensor.body_names) if name.endswith("_foot")]
    nonfeet = [i for i in range(len(sensor.body_names)) if i not in feet]
    roll, pitch, yaw = euler_xyz_from_quat(d.root_quat_w)
    roll = torch.remainder(roll + math.pi, 2 * math.pi) - math.pi
    pitch = torch.remainder(pitch + math.pi, 2 * math.pi) - math.pi
    return dict(base_cmd=velcmd, base_actual=vel, ee_cmd=cmd, ee_actual_mixed=actual_mixed,
                ee_actual_quat_world=eeq, ee_target_quat_world=quat_mul(linkq, cmd[:, 3:7]),
                xy_error_sq=(vel[:, :2] - velcmd[:, :2]).square().sum(-1),
                yaw_error_sq=(vel[:, 2] - velcmd[:, 2]).square(),
                ee_position_error=pos_err, ee_orientation_error=ori_err_command,
                ee_orientation_reward_error=ori_err_reward,
                base_height=d.root_pos_w[:, 2], base_roll=roll, base_pitch=pitch,
                base_pos_world=d.root_pos_w, foot_forces=force_hist[:, feet],
                nonfoot_max_force=force_hist[:, nonfeet].amax(-1),
                nonfoot_support_z=sensor.data.net_forces_w[:, nonfeet, 2].clamp(min=0).sum(-1),
                total_support_z=sensor.data.net_forces_w[:, :, 2].clamp(min=0).sum(-1),
                action_abs_max=env.action_manager.action.abs().amax(-1),
                joint_speed_max=d.joint_vel.abs().amax(-1))



def physical_suite(env):
    from task_suite import mdp
    p = physical(env)
    p['support_height'] = mdp.support_height(env)
    p['base_clearance'] = p['base_height'] - p['support_height']
    p['goal_distance'] = mdp.goal_distance(env)
    p['route_progress'] = mdp.route_progress(env)
    p['actions'] = env.action_manager.action
    p['joint_position'] = env.scene['robot'].data.joint_pos
    p['joint_velocity'] = env.scene['robot'].data.joint_vel
    p['applied_torque'] = env.scene['robot'].data.applied_torque
    p['joint_velocity_limit_ratio'] = (p['joint_velocity'].abs() / env.scene['robot'].data.joint_vel_limits).amax(-1)
    bounds = env.scene['robot'].data.joint_pos_limits
    p['joint_position_limit_excess'] = torch.maximum(
        (bounds[..., 0] - p['joint_position']).clamp_min(0),
        (p['joint_position'] - bounds[..., 1]).clamp_min(0)).amax(-1)
    p['reward_terms'] = env.reward_manager._step_reward
    for name in env.termination_manager.active_terms:
        p['termination_' + name] = env.termination_manager.get_term(name)
    return p


def summarize(data, mask):
    import numpy as np
    count = int(mask.sum())
    if not count:
        return {'samples': 0}
    cmd, actual = data['base_cmd'][mask], data['base_actual'][mask]
    xy_energy = float(np.mean(np.sum(cmd[:, :2] ** 2, axis=-1)))
    yaw_energy = float(np.mean(cmd[:, 2] ** 2))
    ee = data['ee_position_error'][mask]
    return dict(samples=count, xy_rmse=float(np.sqrt(np.mean(data['xy_error_sq'][mask]))),
                yaw_rmse=float(np.sqrt(np.mean(data['yaw_error_sq'][mask]))),
                ee_position_rmse=float(np.sqrt(np.mean(ee**2))), ee_position_mean=float(ee.mean()),
                ee_position_p95=float(np.quantile(ee, .95)),
                ee_orientation_mean=float(data['ee_orientation_error'][mask].mean()),
                base_clearance_min=float(data['base_clearance'][mask].min()),
                base_roll_abs_mean=float(np.abs(data['base_roll'][mask]).mean()),
                base_pitch_abs_mean=float(np.abs(data['base_pitch'][mask]).mean()),
                nonfoot_step_fraction=float((data['nonfoot_max_force'][mask] > 1).mean()),
                xy_zero_velocity_rmse=xy_energy**.5, yaw_zero_velocity_rmse=yaw_energy**.5,
                xy_command_response_gain=float(np.mean(np.sum(cmd[:, :2]*actual[:, :2], axis=-1))/xy_energy) if xy_energy>1e-9 else None,
                yaw_command_response_gain=float(np.mean(cmd[:, 2]*actual[:, 2])/yaw_energy) if yaw_energy>1e-9 else None,
                commanded_xy_speed_mean=float(np.linalg.norm(cmd[:, :2], axis=-1).mean()),
                actual_xy_speed_mean=float(np.linalg.norm(actual[:, :2], axis=-1).mean()),
                actual_abs_yaw_mean=float(np.abs(actual[:, 2]).mean()),
                applied_torque_abs_max=float(np.abs(data['applied_torque'][mask]).max()),
                joint_velocity_limit_ratio_max=float(data['joint_velocity_limit_ratio'][mask].max()),
                overspeed_step_fraction=float((data['joint_velocity_limit_ratio'][mask]>1.0001).mean()),
                joint_position_limit_excess_max_rad=float(data['joint_position_limit_excess'][mask].max()),
                joint_limit_violation_step_fraction=float((data['joint_position_limit_excess'][mask]>.001).mean()),
                action_abs_max=float(data['action_abs_max'][mask].max()))


class TrainingTelemetry:
    """Read-only observers around native PPO. Never replaces its loss or optimizer."""
    def __init__(self, runner, out):
        self.runner, self.env, self.out = runner, runner.env.unwrapped, Path(out)
        self.update_count = 0
        self.steps, self.kls, self.grads = [], [], []
        self.started = time.monotonic()
        self.bad_kl = 0
        self.previous_actions = None
        self.previous_done = None
        self.log = (self.out/'updates.jsonl').open('x', buffering=1)
        self.original_reward = self.env.reward_manager.compute
        self.original_kl = runner.alg.actor.get_kl_divergence
        self.original_clip = torch.nn.utils.clip_grad_norm_
        self.original_update = runner.alg.update
        self.env.reward_manager.compute = self.reward
        runner.alg.actor.get_kl_divergence = self.kl
        torch.nn.utils.clip_grad_norm_ = self.clip
        runner.alg.update = self.update

    def save(self, filename):
        saved = self.runner.alg.save()
        saved.update(iter=self.update_count - 1,
                     infos={'completed_updates': self.update_count, 'initialization': 'full CP3000 actor/critic; fresh optimizer',
                            'task_suite': plain(self.env.cfg.task_suite)})
        target = self.out/filename
        if target.exists():
            raise FileExistsError(target)
        torch.save(saved, target)
        write_json(target.with_suffix('.json'), dict(path=str(target), sha256=sha(target),
                   bytes=target.stat().st_size, completed_updates=self.update_count))

    def reward(self, *args, **kwargs):
        result = self.original_reward(*args, **kwargs)
        with torch.no_grad():
            p = physical_suite(self.env)
            row = dict(xy_mse=p['xy_error_sq'].mean(), yaw_mse=p['yaw_error_sq'].mean(),
                       ee_mse=p['ee_position_error'].square().mean(),
                       ee_orientation_rad=p['ee_orientation_error'].mean(),
                       base_clearance=p['base_clearance'].mean(),
                       nonfoot_fraction=(p['nonfoot_max_force']>1).float().mean(),
                       overspeed_fraction=(p['joint_velocity_limit_ratio']>1.0001).float().mean(),
                       joint_limit_fraction=(p['joint_position_limit_excess']>.001).float().mean(),
                       actual_speed=p['base_actual'][:,:2].norm(dim=-1).mean(),
                       commanded_speed=p['base_cmd'][:,:2].norm(dim=-1).mean(),
                       route_progress=p['route_progress'].mean(), goal_distance=p['goal_distance'].mean(),
                       reward=result.mean(), done_count=self.env.reset_buf.float().sum())
            if self.previous_actions is not None:
                valid = ~self.previous_done
                delta = (p['actions'] - self.previous_actions)[valid]
                if delta.numel():
                    row['action_delta_legs_mse'] = delta[:,:12].square().mean()
                    row['action_delta_arm_mse'] = delta[:,12:].square().mean()
                else:
                    row['action_delta_legs_mse'] = result.new_tensor(0.)
                    row['action_delta_arm_mse'] = result.new_tensor(0.)
            else:
                row['action_delta_legs_mse'] = result.new_tensor(0.)
                row['action_delta_arm_mse'] = result.new_tensor(0.)
            self.previous_actions = p['actions'].clone()
            self.previous_done = self.env.reset_buf.bool().clone()
            for name in self.env.termination_manager.active_terms:
                row['termination/'+name] = self.env.termination_manager.get_term(name).float().sum()
            for i, name in enumerate(self.env.reward_manager.active_terms):
                row['reward_rate/'+name] = self.env.reward_manager._step_reward[:,i].mean()
            self.steps.append({k: v.detach() for k,v in row.items()})
        return result

    def kl(self, *args, **kwargs):
        result = self.original_kl(*args, **kwargs)
        self.kls.append(result.detach().mean())
        return result

    def clip(self, *args, **kwargs):
        result = self.original_clip(*args, **kwargs)
        self.grads.append(result.detach())
        if not bool(torch.isfinite(result)):
            self.save('emergency_nonfinite_gradient.pt')
            raise RuntimeError('Nonfinite gradient; stopped without tuning recipe')
        return result

    def update(self, *args, **kwargs):
        result = self.original_update(*args, **kwargs)
        self.update_count += 1
        metrics = {k: float(torch.stack([s[k] for s in self.steps]).mean()) for k in self.steps[0]}
        kls = torch.stack(self.kls)
        grads = torch.stack(self.grads)
        row = dict(completed_updates=self.update_count,
                   env_steps=self.update_count*self.env.num_envs*self.runner.cfg['num_steps_per_env'],
                   elapsed_s=time.monotonic()-self.started, losses=plain(result),
                   learning_rate=self.runner.alg.learning_rate,
                   action_std=plain(self.runner.alg.actor.output_std[0]),
                   kl_mean=float(kls.mean()), kl_max=float(kls.max()),
                   actor_grad_norm_mean=float(grads[0::2].mean()), actor_grad_norm_max=float(grads[0::2].max()),
                   critic_grad_norm_mean=float(grads[1::2].mean()), critic_grad_norm_max=float(grads[1::2].max()),
                   metrics=metrics)
        self.log.write(json.dumps(row, allow_nan=False)+'\n')
        write_json(self.out/'latest.json', row)
        self.bad_kl = self.bad_kl + 1 if row['kl_mean'] > .1 else 0
        values = list(metrics.values()) + [row['kl_mean'], row['kl_max']]
        values += [float(v) for v in result.values()]
        if not all(math.isfinite(v) for v in values):
            self.save('emergency_nonfinite.pt')
            raise RuntimeError('Nonfinite PPO telemetry')
        if self.bad_kl >= 5:
            self.save('emergency_sustained_kl.pt')
            raise RuntimeError('Mean minibatch KL > 0.1 for 5 consecutive updates')
        self.steps.clear(); self.kls.clear(); self.grads.clear()
        return result

    def close(self):
        self.env.reward_manager.compute = self.original_reward
        self.runner.alg.actor.get_kl_divergence = self.original_kl
        torch.nn.utils.clip_grad_norm_ = self.original_clip
        self.runner.alg.update = self.original_update
        self.log.close()
