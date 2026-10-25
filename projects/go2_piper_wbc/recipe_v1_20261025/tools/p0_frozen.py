"""Paired headless frozen-policy calibration; no optimizer updates or rendering."""
import argparse
from pathlib import Path
from release_support import release_source_hashes
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--clamp', action='store_true')
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--seed', type=int, default=8101)
parser.add_argument('--workspace', action='store_true')
parser.add_argument('--recipe-protocol', choices=['A','T1','T0'])
parser.add_argument('--admission')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
out = Path(args.out)
if out.exists(): parser.error('Fresh output required')
app = AppLauncher(args).app

import importlib.metadata
import json
import traceback
import numpy as np
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks
from task_suite.config import build_cfg, profile_signature
from telemetry import plain, write_json, sha, snapshot
from scratch_observer import InterfaceObserver
import scratch_evaluation_core as core
from t1_tracking_metrics import tracking_metrics
from recipe_v1.control import SoftLimitJointPositionAction, control_diagnostics, ActuatorMonitor


def main():
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    spec = json.loads((source/'recipe_v1/spec.json').read_text())
    if args.workspace:
        import itertools
        spec['workspace_cloud']=[list(p) for p in itertools.product((.43,.45,.47),(-.03,0.,.03),(.48,.50,.52))]
        args.num_envs=4*len(spec['workspace_cloud'])
    cfg = build_cfg('T1', num_envs=args.num_envs, seed=args.seed, ee_mode='fixed' if args.workspace else 'smooth', evaluation=True,
                    profile_overrides=dict(base_linear_speed=.45,base_yaw_rate=.45,base_segment_s=10.,ee_amplitude=[0.,0.,0.]) if args.workspace else None)
    cfg.recipe_v1 = spec
    if args.workspace:
        from recipe_v1.commands import CloudPoseCommand
        cfg.commands.ee_pose.class_type=CloudPoseCommand
        cfg.task_suite['workspace_cloud']=spec['workspace_cloud']
        for ranges in (cfg.commands.ee_pose.ranges,cfg.commands.ee_pose.limit_ranges):
            ranges.pos_x=(.43,.47);ranges.pos_y=(-.03,.03);ranges.pos_z=(.48,.52)
        cfg.task_suite_signature=profile_signature(cfg.task_suite)
    if args.recipe_protocol:
        if not args.admission:raise ValueError('Recipe evaluation requires P0 admission')
        from recipe_v1.config import build_recipe
        cfg=build_recipe(admission=args.admission,num_envs=args.num_envs,seed=args.seed,evaluation=args.recipe_protocol)
        spec=cfg.recipe_v1
        args.clamp=True
    cfg.sim.device = args.device or 'cuda:0'
    if args.clamp: cfg.actions.joint_pos.class_type = SoftLimitJointPositionAction
    agent = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry('GO2-PIPER-WBC','rsl_rl_cfg_entry_point'), importlib.metadata.version('rsl-rl-lib'))
    agent.seed=args.seed; agent.device=cfg.sim.device; agent.resume=False
    env = RslRlVecEnvWrapper(gym.make('GO2-PIPER-WBC',cfg=cfg), clip_actions=agent.clip_actions)
    raw = env.unwrapped
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint, load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False), strict=True, map_location=agent.device)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    equality={name:all(torch.equal(v.cpu(),checkpoint[name+'_state_dict'][k]) for k,v in getattr(runner.alg,name).state_dict().items()) for name in ('actor','critic')}
    assert all(equality.values()) and len(runner.alg.optimizer.state)==0
    del checkpoint
    write_json(out/'transfer.json',dict(bitwise_equal=equality,strict_load=True,optimizer_loaded=False))
    if args.recipe_protocol:
        from recipe_v1.telemetry import RecipeInterface
        observer=RecipeInterface(env)
    else:observer = InterfaceObserver(env)
    monitor = ActuatorMonitor(env)
    original_physical = core.physical_suite
    low_duration = torch.zeros(raw.num_envs,device=raw.device)
    pose=raw.command_manager.get_term('ee_pose'); robot=raw.scene['robot']
    nominal_link0_offset=float(robot.data.body_pos_w[0,pose.link0_idx,2]-robot.data.root_pos_w[0,2])
    nominal_link0_quat=plain(robot.data.body_quat_w[0,pose.link0_idx])
    def observed_physical(e):
        p = original_physical(e)
        diagnostics = control_diagnostics(e, spec['control']['target_margin_rad'])
        p.update({'control_'+k:v for k,v in diagnostics.items()})
        action_ids=e.action_manager.get_term('joint_pos')._joint_ids
        p['control_actuator_saturation_substep_fraction'] = monitor.fraction[:,action_ids]
        p['control_actuator_substep_count'] = monitor.counts[:,action_ids]
        p['kin_link0_pos']=robot.data.body_pos_w[:,pose.link0_idx]
        p['kin_link0_quat']=robot.data.body_quat_w[:,pose.link0_idx]
        p['kin_ee_pos']=robot.data.body_pos_w[:,pose.body_idx]
        p['kin_ee_quat']=robot.data.body_quat_w[:,pose.body_idx]
        eligible = e.episode_length_buf*e.step_dt > spec['health']['reset_grace_s']
        low = eligible & (p['base_clearance'] < spec['health']['termination_height_m'])
        low_duration[:] = torch.where(low,low_duration+e.step_dt,0.)
        p['low_height'] = low
        p['low_height_duration_s'] = low_duration
        p['shadow_low_termination'] = low_duration >= spec['health']['termination_dwell_s']
        return p
    core.physical_suite = observed_physical
    protocol = dict(admission_key=cfg.task_suite_signature,task=cfg.task_suite,checkpoint=args.checkpoint,
        checkpoint_sha256=sha(args.checkpoint),seed=args.seed,num_envs=args.num_envs,
        mode='eval',horizon_s=20,stochastic=False,control_clamped=args.clamp,
        control_margin_rad=spec['control']['target_margin_rad'],rewards='Recipe V1' if args.recipe_protocol else 'native unchanged',
        low_termination='active plus independent shadow' if args.recipe_protocol else 'shadow only: does not alter rollout',video={'enabled':False},
        recipe_protocol=args.recipe_protocol,
        source_hashes={str(p.relative_to(source)):sha(p) for p in (source/'recipe_v1').glob('*') if p.is_file()})
    protocol['source_hashes'].update(release_source_hashes())
    write_json(out/'protocol.json',protocol)
    write_json(out/'initial_runtime.json',snapshot(raw,runner))
    args.mode='eval'; args.family=cfg.task_suite['family']; args.terrain='plane'; args.stochastic=False; args.steps=0
    try:
        data, summary = core.evaluate(env,runner,out,protocol,args,input_callback=observer.capture_input)
        tracking = tracking_metrics(data,cfg.task_suite['params'],family=args.family)
        write_json(out/'tracking_metrics.json',tracking)
        write_json(out/'interface.json',observer.report())
        valid = data['valid']; after = valid & (data['age_steps']*.02 > 1.)
        per_joint=[]
        names=raw.action_manager.get_term('joint_pos')._joint_names
        for j,name in enumerate(names):
            def values(k):return data['control_'+k][:,:,j][after]
            if not bool(after.any()):
                per_joint.append(dict(name=name,uncovered_after_1s=True))
                continue
            per_joint.append(dict(name=name,raw_abs_p99=float(np.quantile(abs(values('raw_action')),.99)),
                raw_abs_max=float(abs(values('raw_action')).max()),
                target_min_rad=float(values('processed_target').min()),target_max_rad=float(values('processed_target').max()),
                actual_min_rad=float(values('joint_position').min()),actual_max_rad=float(values('joint_position').max()),
                soft_margin_p01_rad=float(np.quantile(values('soft_margin'),.01)),
                hard_limit_violation_fraction=float((values('hard_limit_excess')>.001).mean()),
                soft_limit_violation_fraction=float((values('soft_limit_excess')>.001).mean()),
                target_clip_fraction=float(values('target_was_clipped').mean()),
                shadow_would_clip_fraction=float(values('shadow_target_would_clip').mean()),
                actuator_saturation_fraction=float(values('actuator_saturation_substep_fraction').mean()),
                pd_target_abs_error_mean_rad=float(abs(values('pd_target_error')).mean()),
                pd_target_abs_error_p95_rad=float(np.quantile(abs(values('pd_target_error')),.95))))
        control=dict(per_joint=per_joint,after_1s_valid_samples=int(after.sum()),
            target_limit_excess_max_rad=float(data['control_target_limit_excess'][valid].max()),
            shadow_low_termination_episodes=int((data['shadow_low_termination'] & valid).any(0).sum()),
            low_height_time_fraction=float(data['low_height'][after].mean()) if bool(after.any()) else None,
            saturation_coverage='Every actuator.compute at all decimation substeps; torque traces also keep last substep',
            actuator_substep_counts_min=float(data['control_actuator_substep_count'][valid].min()),
            actuator_substep_counts_max=float(data['control_actuator_substep_count'][valid].max()),
            low_height_duration_s_per_episode=(data['low_height'] & valid).sum(0).astype(float).tolist())
        control['low_height_duration_s_per_episode']=[x*.02 for x in control['low_height_duration_s_per_episode']]
        write_json(out/'control_summary.json',control)
        if args.workspace:
            from recipe_v1.kinematics import export_arm_chain,validate_dynamic_fk,screen_cloud
            chain=export_arm_chain(cfg.scene.robot.spawn.usd_path)
            validation=validate_dynamic_fk(chain,data)
            write_json(out/'kinematic_model.json',dict(chain=chain,validation=validation,asset_sha256=sha(cfg.scene.robot.spawn.usd_path)))
            assert validation['passed'],validation
            action_ids=raw.action_manager.get_term('joint_pos')._joint_ids
            screening=screen_cloud(chain,spec['workspace_cloud'],plain(robot.data.soft_joint_pos_limits[0,action_ids][12:]),nominal_link0_offset,nominal_link0_quat)
            write_json(out/'kinematic_screening.json',screening)
        write_json(out/'manifest.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file()})
        print('P0_FROZEN_COMPLETE',str(out),flush=True)
    finally:
        core.physical_suite=original_physical
        monitor.close(); observer.close(); env.close()


try:
    with torch.inference_mode(): main()
except BaseException as exc:
    print(traceback.format_exc(),flush=True)
    if out.exists():write_json(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    raise
finally:app.close()
