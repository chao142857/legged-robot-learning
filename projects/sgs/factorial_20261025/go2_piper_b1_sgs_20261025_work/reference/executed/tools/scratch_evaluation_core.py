"""Historical first-episode evaluator, reused with an explicit args object and optional full video callback."""
import csv
import numpy as np
import torch
from telemetry import plain, write_json, physical_suite, summarize, model_sha

def evaluate(env, runner, out, protocol, args, frame_callback=None, input_callback=None):
    raw = env.unwrapped
    policy = runner.get_inference_policy(device=raw.device)
    initial_actor_sha = model_sha(runner.alg.actor)
    initial_critic_sha = model_sha(runner.alg.critic)
    live = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    initial_position = raw.scene['robot'].data.root_pos_w.detach().cpu().numpy().copy()
    write_json(out/'reset_state.json', dict(root_state=plain(raw.scene['robot'].data.root_state_w),
               joint_pos=plain(raw.scene['robot'].data.joint_pos),
               body_names=raw.scene['robot'].body_names,
               body_pos=plain(raw.scene['robot'].data.body_pos_w),
               support_height=plain(physical_suite(raw)['support_height'])))
    arrays, body_positions = {}, []
    original_reward = raw.reward_manager.compute
    def observe(*a, **kw):
        reward = original_reward(*a, **kw)
        p = physical_suite(raw)
        p['valid'] = live
        p['age_steps'] = raw.episode_length_buf
        p['reward'] = reward
        for k,v in p.items():
            arrays.setdefault(k, []).append(v.detach().cpu().numpy().copy())
        body_positions.append(raw.scene['robot'].data.body_pos_w[:16].detach().cpu().numpy().copy())
        return reward
    raw.reward_manager.compute = observe
    obs = env.get_observations()
    cap = min(args.steps or raw.max_episode_length, raw.max_episode_length)
    try:
        with torch.inference_mode():
            for frame_index in range(cap):
                if input_callback is not None:
                    input_callback(obs)
                actions = policy(obs, stochastic_output=args.stochastic)
                obs, reward, dones, extras = env.step(actions)
                live &= ~dones.bool()
                if frame_callback is not None:
                    frame_callback(frame_index, live, {k:v[-1] for k,v in arrays.items()})
                if not bool(live.any()) and frame_callback is None:
                    break
    finally:
        raw.reward_manager.compute = original_reward
    data = {k: np.stack(v) for k,v in arrays.items()}
    data['initial_base_pos_world'] = initial_position
    data['body_pos_world_first16'] = np.stack(body_positions)
    for k,v in data.items():
        if np.issubdtype(v.dtype, np.floating) and not np.isfinite(v).all():
            raise RuntimeError('Nonfinite physical trajectory: '+k)
    np.savez_compressed(out/'trajectories.npz', **data)
    valid = data['valid']
    cmd = data['base_cmd']
    masks = dict(all=np.ones_like(valid), standing=(np.linalg.norm(cmd,axis=-1)<1e-6),
                 forward=(cmd[:,:,0]>.05), backward=(cmd[:,:,0]<-.05),
                 turning=(np.abs(cmd[:,:,2])>.05))
    summary = dict(num_episodes=args.num_envs, requested_steps=cap, max_steps=raw.max_episode_length,
                   complete_episode_evaluation=cap==raw.max_episode_length,
                   mode=args.mode, family=args.family, terrain=args.terrain, seed=args.seed,
                   stochastic=args.stochastic, groups={}, terminations={},
                   contact_body_names=raw.scene['contact_forces'].body_names,
                   articulation_body_names=raw.scene['robot'].body_names,
                   reward_names=raw.reward_manager.active_terms,
                   admission_key=protocol['admission_key'])
    for window,w in [('all', valid), ('after_1s', valid & (data['age_steps']*raw.step_dt>1.))]:
        summary['groups'][window] = {key:summarize(data,w & mask) for key,mask in masks.items()}
    failure_names = [n for n in raw.termination_manager.active_terms if not raw.termination_manager.get_term_cfg(n).time_out]
    for name in raw.termination_manager.active_terms:
        summary['terminations'][name] = int((data['termination_'+name]&valid).any(0).sum())
    rows = []
    for i in range(args.num_envs):
        mask=valid[:,i]
        sample={k:v[:,i] for k,v in data.items() if k not in ('initial_base_pos_world','body_pos_world_first16')}
        row=dict(env_id=i, steps=int(mask.sum()), duration_s=float(mask.sum()*raw.step_dt),
                 failure=bool(any((sample['termination_'+n]&mask).any() for n in failure_names)),
                 nonfoot_contact=bool((sample['nonfoot_max_force'][mask]>1).any()),
                 base_drift_final_m=float(np.linalg.norm(sample['base_pos_world'][mask][-1,:2]-initial_position[i,:2])),
                 base_drift_max_m=float(np.linalg.norm(sample['base_pos_world'][mask][:,:2]-initial_position[i,:2],axis=-1).max()),
                 route_progress_final_m=float(sample['route_progress'][mask][-1]),
                 goal_distance_final_m=float(sample['goal_distance'][mask][-1]),
                 goal_distance_min_m=float(sample['goal_distance'][mask].min()),
                 return_sum=float(sample['reward'][mask].sum()))
        row.update(summarize(sample, mask))
        row['complete'] = bool(row['steps']==raw.max_episode_length and not row['failure'])
        row['contact_safe'] = bool(row['complete'] and not row['nonfoot_contact'] and row['base_clearance_min']>=.15)
        row['overspeed'] = bool(row['joint_velocity_limit_ratio_max']>1.0001)
        row['joint_limit_violation'] = bool(row['joint_position_limit_excess_max_rad']>.001)
        row['safe'] = bool(row['contact_safe'] and not row['overspeed'] and not row['joint_limit_violation'])
        row['ee_success_10cm'] = bool(row['complete'] and row['ee_position_rmse']<.1 and row['ee_orientation_mean']<.2)
        row['ee_success_5cm'] = bool(row['complete'] and row['ee_position_rmse']<.05 and row['ee_orientation_mean']<.2)
        row['movement_success'] = bool(row['complete'] and row['xy_rmse']<.15 and row['yaw_rmse']<.2)
        if args.family=='T0':
            row['movement_success'] &= row['base_drift_max_m']<.25
        if args.family=='T2':
            row['movement_success'] &= row['goal_distance_final_m']<.3
        row['joint_success_10cm'] = row['safe'] and row['ee_success_10cm'] and row['movement_success']
        row['joint_success_5cm'] = row['safe'] and row['ee_success_5cm'] and row['movement_success']
        rows.append(row)
    with (out/'episodes.csv').open('x') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for key in ('complete','failure','safe','contact_safe','overspeed','joint_limit_violation','nonfoot_contact','ee_success_10cm','ee_success_5cm',
                'movement_success','joint_success_10cm','joint_success_5cm'):
        summary[key+'_count'] = sum(int(r[key]) for r in rows)
    for key in ('base_drift_final_m','base_drift_max_m','route_progress_final_m','goal_distance_final_m','duration_s'):
        summary[key+'_mean']=float(np.mean([r[key] for r in rows]))
    deltas=np.diff(data['actions'],axis=0)[valid[1:]&valid[:-1]]
    summary['action_delta_mse']=dict(legs=float(np.mean(deltas[:,:12]**2)), arm=float(np.mean(deltas[:,12:]**2))) if deltas.size else None
    adjacent = valid[1:] & valid[:-1]
    ee_velocity = np.linalg.norm(np.diff(data['ee_cmd'][:,:,:3],axis=0)/raw.step_dt,axis=-1)[adjacent]
    base_delta = np.linalg.norm(np.diff(data['base_cmd'],axis=0),axis=-1)[adjacent]
    summary['command_dynamics'] = dict(
        ee_mixed_target_speed_mean_mps=float(ee_velocity.mean()) if ee_velocity.size else None,
        ee_mixed_target_speed_p95_mps=float(np.quantile(ee_velocity,.95)) if ee_velocity.size else None,
        ee_mixed_target_speed_max_mps=float(ee_velocity.max()) if ee_velocity.size else None,
        base_command_delta_max=float(base_delta.max()) if base_delta.size else None,
        note='Mixed-coordinate EE command derivative; world-frame body motion is separately logged')
    summary['actor_unchanged']=model_sha(runner.alg.actor)==initial_actor_sha
    summary['critic_unchanged']=model_sha(runner.alg.critic)==initial_critic_sha
    if not summary['actor_unchanged'] or not summary['critic_unchanged']:
        raise RuntimeError('Frozen evaluation modified model state')
    write_json(out/'summary.json', summary)
    return data, summary
