"""Frozen transfer tests and bounded native PPO for parameterized Go2+Piper tasks.

No new PPO implementation; no SGS sampler; no implicit resume or output overwrite.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

# An older experiment's sitecustomize observer must never attach to this run.
os.environ.pop('NATIVE_WBC_DIAGNOSTICS', None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_identity import task_identity, check_admission
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['eval', 'physics', 'train'], required=True)
parser.add_argument('--family', choices=['T0', 'T1', 'T2'], required=True)
parser.add_argument('--terrain', default='stairs_up')
parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='easy')
parser.add_argument('--ee_mode', choices=['fixed', 'smooth'], default='smooth')
parser.add_argument('--profile', help='Exact candidate profile JSON; all resulting task parameters enter admission key')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--updates', type=int, default=20)
parser.add_argument('--steps', type=int, default=0, help='0 runs complete episodes; positive is censored smoke')
parser.add_argument('--seed', type=int, default=7101)
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--admission', help='Required for training: exact task key and passed geometry/IK/physics checks')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.updates <= 100:
    parser.error('--updates must be in [1,100]; long training is disabled')
if not 1 <= args.num_envs <= 1024:
    parser.error('--num_envs must be in [1,1024] for this bounded acceptance')
if args.steps < 0:
    parser.error('--steps must be nonnegative')
if args.mode == 'train' and not args.admission:
    parser.error('Training requires --admission after geometry, kinematic and physics acceptance')
if args.mode == 'train' and args.steps:
    parser.error('--steps is only for evaluation/physics smoke')
if Path(args.out).exists():
    parser.error('Refusing to overwrite an existing output directory')
if not Path(args.checkpoint).is_file():
    parser.error('Checkpoint file not found')
app = AppLauncher(args).app

import csv
import numpy as np
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks
from task_suite.config import build_cfg
from telemetry import (plain, sha, write_json, physical_suite, summarize, snapshot,
                       TrainingTelemetry, model_sha)


def evaluate(env, runner, out, protocol):
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
            for _ in range(cap):
                actions = policy(obs, stochastic_output=args.stochastic)
                obs, reward, dones, extras = env.step(actions)
                live &= ~dones.bool()
                if not bool(live.any()):
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


def main():
    out=Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg=build_cfg(args.family, terrain=args.terrain, difficulty=args.difficulty,
                  num_envs=args.num_envs, seed=args.seed, ee_mode=args.ee_mode,
                  evaluation=args.mode!='train', profile_json=args.profile)
    cfg.sim.device=args.device or 'cuda:0'
    cfg.log_dir=str(out)
    key, task_spec=task_identity(plain(cfg.task_suite))
    suite_root = Path(__file__).resolve().parents[1]
    native_root = Path(LeggedManip_Lab.__file__).resolve().parent
    task_prefix = Path('tasks/manager_based/leggedmanip_lab')
    source_files = {name: suite_root/name for name in (
        'task_suite/config.py','task_suite/mdp.py','task_suite/profiles.json',
        'tools/run_suite.py','tools/telemetry.py','tools/task_identity.py','tools/audit_geometry_runtime.py')}
    for name in ('leggedmanip_lab_env_cfg.py','mdp/rewards.py','mdp/pose_command_wbc.py',
                 'config/go2_piper/wbc_env_cfg.py','config/go2_piper/agents/rsl_rl_ppo_cfg.py'):
        source_files['native/'+name] = native_root/task_prefix/name
    asset_files = dict(robot_articulation_cfg=native_root/'assets/go2_piper/go2_piper_articulation_cfg.py',
                       robot_usd=Path(cfg.scene.robot.spawn.usd_path))
    if args.profile:
        source_files['profile_json'] = Path(args.profile)
    usd = asset_files['robot_usd']
    for name in ('physics','sensor','base','robot'):
        asset_files['robot_usd_configuration_'+name] = usd.parent/'configuration'/('go2_piper_'+name+'.usd')
    bindings = dict(sources={name:sha(path) for name,path in source_files.items()},
                    assets={name:sha(path) for name,path in asset_files.items()})
    write_json(out/'candidate_identity.json',dict(admission_key=key,task=task_spec,bindings=bindings,
        source_paths={name:str(path) for name,path in source_files.items()},
        asset_paths={name:str(path) for name,path in asset_files.items()}))
    admission=check_admission(args.admission,key,source_files=source_files,asset_files=asset_files) if args.mode=='train' else None
    agent_cfg=load_cfg_from_registry('GO2-PIPER-WBC','rsl_rl_cfg_entry_point')
    agent_cfg=handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version('rsl-rl-lib'))
    agent_cfg.seed=args.seed
    agent_cfg.device=cfg.sim.device
    agent_cfg.max_iterations=args.updates
    agent_cfg.save_interval=args.updates
    assert agent_cfg.algorithm.schedule=='adaptive'
    assert agent_cfg.algorithm.learning_rate==5e-4
    assert agent_cfg.actor.hidden_dims==[512,256,128] and agent_cfg.critic.hidden_dims==[512,256,128]
    dump_yaml(str(out/'env.yaml'), cfg)
    dump_yaml(str(out/'agent.yaml'), agent_cfg)
    protocol=dict(task=task_spec, admission_key=key, admission=admission, checkpoint=args.checkpoint,
                  checkpoint_sha256=sha(args.checkpoint), mode=args.mode, seed=args.seed,
                  initialization='strict complete actor/critic, fresh optimizer, iteration zero',
                  observation_noise=cfg.observations.policy.enable_corruption,
                  training_update_cap=100, requested_updates=args.updates if args.mode=='train' else 0,
                  ppo_schedule='adaptive', learning_rate=agent_cfg.algorithm.learning_rate,
                  success_thresholds=dict(xy_rmse=.15,yaw_rmse=.2,ee_position_rmse=[.1,.05],
                    ee_orientation_mean=.2,base_clearance_min=.15,T0_max_drift=.25,T2_final_goal_distance=.3,
                    joint_velocity_limit_ratio=1.0001,joint_position_limit_excess_rad=.001),
                  success_definition='complete first episode without failure; joint success additionally requires no non-foot contact >1 N',
                  censoring='All first-episode valid pre-reset states included; no post-reset frames; shorter smoke cannot succeed',
                  evaluator_sha256=sha(__file__),telemetry_sha256=sha(Path(__file__).with_name('telemetry.py')))
    write_json(out/'protocol.json',protocol)
    base=gym.make('GO2-PIPER-WBC',cfg=cfg)
    env=RslRlVecEnvWrapper(base,clip_actions=agent_cfg.clip_actions)
    runner=OnPolicyRunner(env,agent_cfg.to_dict(),log_dir=str(out/'rsl_rl') if args.mode=='train' else None,
                          device=agent_cfg.device)
    runner.load(args.checkpoint, load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),
                strict=True,map_location=agent_cfg.device)
    assert runner.current_learning_iteration==0 and len(runner.alg.optimizer.state)==0
    assert runner.alg.schedule=='adaptive' and runner.alg.learning_rate==5e-4
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    equality={}
    for name in ('actor','critic'):
        current=getattr(runner.alg,name).state_dict()
        saved=cp[name+'_state_dict']
        equality[name]=set(current)==set(saved) and all(torch.equal(current[k].cpu(),saved[k]) for k in saved)
    if not all(equality.values()):
        raise RuntimeError('Checkpoint transfer not bitwise complete')
    del cp
    write_json(out/'transfer.json',dict(strict_load=True,bitwise_equal=equality,
               optimizer_state_entries=len(runner.alg.optimizer.state),iteration=runner.current_learning_iteration,
               actor_sha256=model_sha(runner.alg.actor),critic_sha256=model_sha(runner.alg.critic),
               observation_dims=plain(env.unwrapped.observation_manager.group_obs_dim)))
    write_json(out/'initial_runtime.json',snapshot(env.unwrapped,runner))
    from audit_geometry_runtime import audit as audit_geometry
    audit_geometry(env.unwrapped, out)
    try:
        if args.mode=='train':
            telemetry=TrainingTelemetry(runner,out)
            telemetry.save('cp_0000.pt')
            try:
                # Command phase, reset grace and route settle time share the
                # episode clock. Preserve their true reset age for this suite.
                runner.learn(num_learning_iterations=args.updates,init_at_random_ep_len=False)
                assert telemetry.update_count==args.updates
                telemetry.save('cp_final.pt')
                write_json(out/'summary.json',dict(completed_updates=telemetry.update_count,
                    stopped_reason='bounded run completed',admission_key=key,
                    final_actor_sha256=model_sha(runner.alg.actor),final_critic_sha256=model_sha(runner.alg.critic)))
            finally:
                telemetry.close()
        else:
            evaluate(env,runner,out,protocol)
        write_json(out/'final_runtime.json',snapshot(env.unwrapped,runner))
        write_json(out/'manifest.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
                   for p in out.rglob('*') if p.is_file()})
        print('SUITE_COMPLETE',out,flush=True)
    finally:
        env.close()

try:
    main()
except Exception as exc:
    import traceback
    if Path(args.out).is_dir():
        (Path(args.out)/'error.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n')
    raise
finally:
    app.close()
