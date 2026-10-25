"""Independent native-distribution evaluation. Never selects the -Play task."""
import argparse
from pathlib import Path
import os

os.environ.pop('NATIVE_WBC_DIAGNOSTICS', None)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--seed', type=int, default=7001)
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--mode', choices=['deterministic', 'stochastic'], required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import csv
import importlib.metadata
import json
import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
import LeggedManip_Lab.tasks
from native_observer import physical, snapshot, write_json, sha


def summarize(data, mask):
    count = int(mask.sum())
    if count == 0:
        return dict(samples=0)
    def mean(key):
        return float(data[key][mask].mean())
    cmd = data['base_cmd'][mask]
    act = data['base_actual'][mask]
    xy_energy = float(np.mean(np.sum(cmd[:,:2]**2,axis=-1)))
    yaw_energy = float(np.mean(cmd[:,2]**2))
    return dict(samples=count, xy_rmse=float(np.sqrt(mean('xy_error_sq'))),
                yaw_rmse=float(np.sqrt(mean('yaw_error_sq'))),
                ee_position_rmse=float(np.sqrt(np.mean(data['ee_position_error'][mask]**2))),
                ee_position_mean=mean('ee_position_error'),
                ee_position_p95=float(np.quantile(data['ee_position_error'][mask], .95)),
                ee_orientation_mean=mean('ee_orientation_error'),
                ee_orientation_p95=float(np.quantile(data['ee_orientation_error'][mask], .95)),
                ee_orientation_reward_mean=mean('ee_orientation_reward_error'),
                base_height_mean=mean('base_height'),
                base_height_p05=float(np.quantile(data['base_height'][mask], .05)),
                base_roll_abs_mean=float(np.abs(data['base_roll'][mask]).mean()),
                base_pitch_abs_mean=float(np.abs(data['base_pitch'][mask]).mean()),
                nonfoot_step_fraction=float((data['nonfoot_max_force'][mask] > 1).mean()),
                foot_contact_fraction=float((data['foot_forces'][mask] > 1).mean()),
                nonfoot_support_force_fraction=float(data['nonfoot_support_z'][mask].sum()/max(data['total_support_z'][mask].sum(),1e-9)),
                low_base_step_fraction=float((data['base_height'][mask] < .20).mean()),
                xy_zero_velocity_rmse=xy_energy**.5, yaw_zero_velocity_rmse=yaw_energy**.5,
                xy_command_response_gain=float(np.mean(np.sum(cmd[:,:2]*act[:,:2],axis=-1))/xy_energy) if xy_energy>1e-9 else None,
                yaw_command_response_gain=float(np.mean(cmd[:,2]*act[:,2])/yaw_energy) if yaw_energy>1e-9 else None,
                commanded_xy_speed_mean=float(np.linalg.norm(data['base_cmd'][mask, :2],axis=-1).mean()),
                actual_xy_speed_mean=float(np.linalg.norm(data['base_actual'][mask, :2],axis=-1).mean()),
                command_actual_xy_dot_mean=float((data['base_cmd'][mask, :2]*data['base_actual'][mask, :2]).sum(-1).mean()),
                commanded_abs_yaw_mean=float(np.abs(data['base_cmd'][mask, 2]).mean()),
                actual_abs_yaw_mean=float(np.abs(data['base_actual'][mask, 2]).mean()),
                joint_velocity_limit_ratio_max=float(data['joint_velocity_limit_ratio'][mask].max()),
                overspeed_step_fraction=float((data['joint_velocity_limit_ratio'][mask]>np.float32(1.0001)).mean()),
                applied_torque_abs_max=float(np.abs(data['applied_torque'][mask]).max()),
                joint_speed_max=float(data['joint_speed_max'][mask].max()),
                action_abs_max=float(data['action_abs_max'][mask].max()))


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    env_cfg = load_cfg_from_registry('GO2-PIPER-WBC', 'env_cfg_entry_point')
    agent_cfg = load_cfg_from_registry('GO2-PIPER-WBC', 'rsl_rl_cfg_entry_point')
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version('rsl-rl-lib'))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = args.device or 'cuda:0'
    env_cfg.log_dir = str(out)
    assert env_cfg.episode_length_s == 20 and env_cfg.observations.policy.enable_corruption
    base_env = gym.make('GO2-PIPER-WBC', cfg=env_cfg)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    raw = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=raw.device)
    dump_yaml(str(out/'env.yaml'), env_cfg)
    dump_yaml(str(out/'agent.yaml'), agent_cfg)
    write_json(out/'initial_runtime.json', snapshot(raw))
    write_json(out/'protocol.json', dict(checkpoint=args.checkpoint, checkpoint_sha256=sha(args.checkpoint),
               protocol_version=2,
               evaluator_sha256=sha(__file__), observer_sha256=sha(Path(__file__).with_name('native_observer.py')),
               seed=args.seed, mode=args.mode, task='GO2-PIPER-WBC', num_envs=args.num_envs,
               episodes='first complete episode per environment; no post-reset samples',
               control_dt=raw.step_dt, episode_seconds=20, observation_noise=True,
               curriculum='native callbacks unchanged; no range override',
               windows=['all valid steps', 'age > 1 second'],
               contact='any non-foot body > 1 N in native 3-frame contact history; diagnostic only',
               diagnostic_thresholds=dict(xy_rmse=.15, yaw_rmse=.2, ee_position_rmse=.05,
                    ee_orientation_mean=.2, min_base_height=.20,
                    requires_full_20s=True, forbids_nonfoot_contact=True),
               success_is_author_metric=False, overspeed_termination='absent in native task; no new termination added'))
    live = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    arrays = {}
    bodies = []
    original_compute = raw.reward_manager.compute

    def observe(*a, **kw):
        reward = original_compute(*a, **kw)
        p = physical(raw)
        p['valid'] = live
        p['age'] = raw.episode_length_buf.clone()
        p['reward'] = reward
        p['reward_terms'] = raw.reward_manager._step_reward
        p['actions'] = raw.action_manager.action
        robot = raw.scene['robot']
        p['joint_velocity'] = robot.data.joint_vel
        p['joint_velocity_limit_ratio'] = (robot.data.joint_vel.abs()/robot.data.joint_vel_limits).amax(-1)
        p['applied_torque'] = robot.data.applied_torque
        p['all_body_contact_forces'] = raw.scene.sensors['contact_forces'].data.net_forces_w_history.norm(dim=-1).amax(1)
        for name in raw.termination_manager.active_terms:
            p['termination_' + name] = raw.termination_manager.get_term(name)
        for k,v in p.items():
            arrays.setdefault(k, []).append(v.detach().cpu().numpy().copy())
        bodies.append(raw.scene['robot'].data.body_pos_w[:16].detach().cpu().numpy().copy())
        return reward

    raw.reward_manager.compute = observe
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(raw.max_episode_length):
            actions = policy(obs, stochastic_output=args.mode == 'stochastic')
            obs, reward, dones, extras = env.step(actions)
            live &= ~dones.bool()
            if not bool(live.any()):
                break
    data = {k:np.stack(v) for k,v in arrays.items()}
    data['body_pos_world_first16'] = np.stack(bodies)
    for k,v in data.items():
        if np.issubdtype(v.dtype,np.floating) and not np.isfinite(v).all():
            raise RuntimeError(f'Nonfinite evaluation array: {k}')
    np.savez_compressed(out/'trajectories.npz', **data)
    valid = data['valid']
    cmdxy = np.linalg.norm(data['base_cmd'][:,:,:2],axis=-1)
    cmdyaw = np.abs(data['base_cmd'][:,:,2])
    actualxy = np.linalg.norm(data['base_actual'][:,:,:2],axis=-1)
    groups = dict(all=np.ones_like(valid), standing_command=(cmdxy<1e-6)&(cmdyaw<1e-6),
                  moving_command=cmdxy>=.1, yaw_command=cmdyaw>=.1,
                  commanded_and_actually_moving=(cmdxy>=.1)&(actualxy>=.05))
    summary = dict(seed=args.seed, mode=args.mode, num_episodes=args.num_envs, max_steps=data['valid'].shape[0],
                   groups={}, terminations={}, native_reward_term_names=raw.reward_manager.active_terms,
                   native_contact_body_names=raw.scene.sensors['contact_forces'].body_names,
                   articulation_body_names=raw.scene['robot'].body_names)
    for window, windowmask in [('all', valid), ('after_1s', valid & (data['age']>50))]:
        summary['groups'][window] = {k:summarize(data, windowmask & m) for k,m in groups.items()}
    for name in raw.termination_manager.active_terms:
        summary['terminations'][name] = int((data['termination_'+name] & valid).any(0).sum())
    episodes = []
    for i in range(args.num_envs):
        m = valid[:,i]
        s = {k:v[:,i] for k,v in data.items() if k!='body_pos_world_first16'}
        row = dict(env_id=i, steps=int(m.sum()), duration_s=float(m.sum()*.02),
                   native_failure=bool(((s['termination_base_contact']|s['termination_bad_orientation'])&m).any()),
                   nonfoot_contact=bool((s['nonfoot_max_force'][m]>1).any()),
                   overspeed_diagnostic=bool((s['joint_velocity_limit_ratio'][m]>np.float32(1.0001)).any()),
                   min_height=float(s['base_height'][m].min()), return_sum=float(s['reward'][m].sum()))
        row.update(summarize(s,m))
        row['diagnostic_success'] = bool(row['steps']==1000 and not row['native_failure'] and not row['nonfoot_contact']
                          and row['min_height']>=.2 and row['xy_rmse']<.15 and row['yaw_rmse']<.2
                          and row['ee_position_rmse']<.05 and row['ee_orientation_mean']<.2)
        episodes.append(row)
    with (out/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(episodes[0])); writer.writeheader(); writer.writerows(episodes)
    summary['survived_20s'] = sum(r['steps']==1000 and not r['native_failure'] for r in episodes)
    summary['episodes_with_nonfoot_contact'] = sum(r['nonfoot_contact'] for r in episodes)
    summary['episodes_with_overspeed_diagnostic'] = sum(r['overspeed_diagnostic'] for r in episodes)
    summary['per_body_contact_fraction'] = dict(zip(summary['native_contact_body_names'],
                         (data['all_body_contact_forces'][valid]>1).mean(0).tolist()))
    adjacent_valid=valid[1:]&valid[:-1]
    deltas=np.diff(data['actions'],axis=0)[adjacent_valid]
    summary['action_delta_mse'] = dict(legs=float(np.mean(deltas[:,:12]**2)),arm=float(np.mean(deltas[:,12:]**2)))
    summary['diagnostic_success_count'] = sum(r['diagnostic_success'] for r in episodes)
    summary['mean_duration_s'] = float(np.mean([r['duration_s'] for r in episodes]))
    write_json(out/'summary.json', summary)
    write_json(out/'final_runtime.json', snapshot(raw))
    write_json(out/'manifest.json', {p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()})
    print('EVALUATION_COMPLETE', out, flush=True)
    env.close()


try:
    main()
finally:
    app.close()
