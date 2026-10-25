"""Z1 continuous-stair evaluation reusing the A2 first-trial evaluator, with contact/limit audits."""
import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument('--task',default='Locomotion-Z1-Rough-Blind-BaseOnly',choices=[f'Locomotion-Z1-{variant}Rough-Blind-{profile}' for variant in ('','Team-') for profile in ('BaseOnly','LegSafe')])
parser.add_argument('--fixed-body-command',action='store_true')
parser.add_argument('--checkpoint', required=True, type=Path)
parser.add_argument('--step_height', type=float, required=True)
parser.add_argument('--descending', action='store_true')
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--seconds', type=float, default=16)
parser.add_argument('--speed', type=float, default=0.45)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--video', action='store_true')
parser.add_argument('--side-follow-camera',action='store_true',help='Real-rollout camera only; robot-centered side view')
parser.add_argument('--video-env', type=int, default=0,
    help='Environment shown in the real video; all first trials still evaluated')
parser.add_argument('--randomized', action='store_true',
    help='Retain upstream dynamics events and observation noise on the fixed staircase')
add_launcher_args(parser)
args = parser.parse_args()
if args.video and getattr(args,'visualizer_explicit',False) and getattr(args,'visualizer',None) is None:
    parser.error('--video needs its configured Newton GL visualizer; omit --viz none')
evaluator_source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if not args.checkpoint.is_file():
    parser.error(f'Checkpoint not found: {args.checkpoint}')
if args.step_height <= 0 or args.num_envs < 1 or args.seconds <= 0 or args.speed <= 0:
    parser.error('Step height, number of environments, duration and forward speed must be positive')
if not 0 <= args.video_env < args.num_envs:
    parser.error('Video environment index must be within the simulated environment count')

import gymnasium as gym
import torch
import basic_locomotion_isaaclab.tasks
from register_tasks import register
register()
from z1_cfg import Z1Env as LocomotionEnv
from basic_locomotion_isaaclab.tasks import custom_events
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata as metadata
from stair_terrain import StaircaseCfg


def t(value):
    return value if torch.is_tensor(value) else value.torch


def fixed_commands(e, env_ids=None):
    if args.fixed_body_command:
        e._commands[:]=0.;e._commands[:,0]=args.speed
        return
    quaternion = t(e._robot.data.root_quat_w)
    x,y,z,w = quaternion.unbind(dim=1)  # Isaac Lab EA uses xyzw.
    yaw = torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    # Track a fixed world-X direction through the existing body-velocity command interface.
    # This external command controller uses simulated yaw, not modified actor weights.
    e._commands[:,0] = args.speed*torch.cos(yaw)
    e._commands[:,1] = -args.speed*torch.sin(yaw)
    e._commands[:,2] = (-1.5*yaw).clamp(-0.7,0.7)


custom_events._get_new_random_commands = fixed_commands


class StairEvalEnv(LocomotionEnv):
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        if not torch.is_tensor(ids):
            import warp as wp
            ids = wp.to_torch(ids).long()
        self.episode_length_buf[ids] = 0
        state = t(self._robot.data.default_root_state)[ids].clone()
        state[:,:3] += self._terrain.env_origins[ids]
        state[:,3:7] = torch.tensor([0,0,0,1],device=self.device)
        state[:,7:] = 0
        q = t(self._robot.data.default_joint_pos)[ids].clone()
        # Keep a controlled starting pose in both domains; dynamics events run in super().
        q += 0.03*torch.randn_like(q)
        self._robot.write_root_pose_to_sim(state[:,:7],env_ids=ids)
        self._robot.write_root_velocity_to_sim(state[:,7:],env_ids=ids)
        self._robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)

    def _get_dones(self):
        terminated,truncated = super()._get_dones()
        history = t(self._contact_sensor.data.net_forces_w_history).norm(dim=-1).amax(dim=1)
        # Snapshot before DirectRLEnv automatically resets failed environments.
        self.eval_snapshot = {
            'pos':t(self._robot.data.root_pos_w).clone(),
            'vel':t(self._robot.data.root_lin_vel_b).clone(),
            'commands':self._commands.clone(),
            'quaternion':t(self._robot.data.root_quat_w).clone(),
            'gravity':t(self._robot.data.projected_gravity_b).clone(),
            'forces':t(self._contact_sensor.data.net_forces_w)[:,self._feet_contact_sensor_ids].clone(),
            'foot_vel':t(self._robot.data.body_lin_vel_w)[:,self._feet_ids_robot].clone(),
            'foot_pos':t(self._robot.data.body_pos_w)[:,self._feet_ids_robot].clone(),
            # PaceDCMotor delays returned joint_efforts, while applied_effort
            # telemetry remains the clipped value before that delay.
            'torque':t(self._robot.data.applied_torque).clone(),
            'z1_joint_vel':t(self._robot.data.joint_vel).clone(),
            'z1_nonfoot_forces':t(self._contact_sensor.data.net_forces_w)[:,self._z1_nonfoot_ids].clone(),
            'z1_calf_forces':t(self._contact_sensor.data.net_forces_w)[:,self._calf_contact_sensor_ids].clone(),
            'base_force':t(self._contact_sensor.data.net_forces_w)[:,self._base_contact_sensor_id].clone(),
            'base_hit':(history[:,self._base_contact_sensor_id]>1).any(dim=1),
            'hip_hit':(history[:,self._hip_contact_sensor_ids]>1).any(dim=1),
            'terminated':terminated.clone(),'truncated':truncated.clone(),
        }
        if args.video:
            self.eval_snapshot.update({
                'joint_pos':t(self._robot.data.joint_pos).clone(),
                'joint_vel':t(self._robot.data.joint_vel).clone(),
                'joint_target_policy_order':self._processed_actions.clone(),
                'computed_torque':t(self._robot.data.computed_torque).clone(),
                'base_history_peak':history[:,self._base_contact_sensor_id].amax(dim=-1),
                'hip_history_peak':history[:,self._hip_contact_sensor_ids].amax(dim=-1),
            })
        return terminated,truncated


cfg = parse_env_cfg(args.task,num_envs=args.num_envs,
                    overrides=['physics=newton_mjwarp'])
cfg.seed = args.seed
cfg.episode_length_s = args.seconds+2
cfg.terrain.terrain_generator = TerrainGeneratorCfg(size=(8,3),num_rows=1,num_cols=1,
    border_width=2,seed=args.seed,use_cache=False,sub_terrains={
    'continuous_stairs':StaircaseCfg(step_height=args.step_height,descending=args.descending)})
cfg.terrain.max_init_terrain_level = 0
import isaaclab.sim as sim_utils
cfg.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24,0.31,0.39))
if not args.randomized:
    for field in vars(cfg.events):
        if not field.startswith('_'):
            setattr(cfg.events,field,None)
    cfg.observation_noise_model = None
cfg.video_recorders = []
if args.video:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True,window_width=960,
        window_height=540,max_visible_envs=1,visible_env_indices=[args.video_env],
        eye=(-1.1,-2.8,1.6),lookat=(-1.1,0,0.55))]
    cfg.video_recorders = [VideoRecorderCfg(source='visualizer:newton_gl',
        output_dir=str(args.output.parent),output_filename_prefix=args.output.stem,
        video_length=int(args.seconds/0.02),frame_stride=2,fps=25)]
gym.register('Z1-Continuous-Stairs-Eval',entry_point=StairEvalEnv,disable_env_checker=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
with launch_simulation(cfg,args):
    env = gym.make('Z1-Continuous-Stairs-Eval',cfg=cfg)
    from isaaclab.utils.io import dump_yaml
    dump_yaml(str(args.output.with_suffix('.env.yaml')),cfg)
    e = env.unwrapped
    wrapper = RslRlVecEnvWrapper(env)
    agent = load_cfg_from_registry(args.task,'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent,metadata.version('rsl-rl-lib'))
    runner = OnPolicyRunner(wrapper,agent.to_dict(),log_dir=None,device=e.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=e.device)
    obs = wrapper.get_observations()
    mass = t(e._robot.data.body_mass).sum(dim=1)
    initial_dynamics = {'total_mass_kg':mass.tolist(), 'actuators':{}}
    for name,actuator in e._robot.actuators.items():
        initial_dynamics['actuators'][name] = {
            'stiffness_min_max':[float(t(actuator.stiffness).min()),float(t(actuator.stiffness).max())],
            'damping_min_max':[float(t(actuator.damping).min()),float(t(actuator.damping).max())]}
    origin = e._terrain.env_origins.clone()
    active = torch.ones(args.num_envs,dtype=torch.bool,device=e.device)
    success = active.clone() & False
    fell = success.clone()
    lateral = success.clone()
    max_progress = torch.zeros(args.num_envs,device=e.device)
    terminal_position = torch.zeros((args.num_envs,3),device=e.device)
    survival_steps = torch.zeros(args.num_envs,device=e.device)
    velocity_error_sum = torch.zeros(args.num_envs,device=e.device)
    slip_sum = torch.zeros(args.num_envs,device=e.device)
    slip_count = torch.zeros(args.num_envs,device=e.device)
    torque_limits = torch.full_like(t(e._robot.data.applied_torque),float('inf'))
    for actuator in e._robot.actuators.values():
        torque_limits[:,actuator.joint_indices] = actuator.actuator_effort_limit
    policy_limits = torque_limits[:,e._ids_joints_order]
    near_limit_steps = torch.zeros_like(policy_limits)
    base_contact_steps = survival_steps.clone()
    horizontal_contact_steps = survival_steps.clone()
    base_terminated = success.clone()
    hip_terminated = success.clone()
    tipped_seen = success.clone()
    nonfoot_steps = survival_steps.clone(); calf_steps=survival_steps.clone()
    torque_violation=success.clone(); speed_violation=success.clone()
    peak_torque=survival_steps.clone(); peak_speed=survival_steps.clone()
    traces = []
    video_trace = []
    goal_progress = 1.8+8*0.30+0.4-0.9
    end_ground_delta = -8*args.step_height if args.descending else 8*args.step_height
    for step in range(int(args.seconds/e.step_dt)):
        if args.video and args.side_follow_camera:
            center=t(e._robot.data.root_pos_w)[args.video_env].detach().cpu().numpy()
            for viz in e.sim._visualizers:
                viewer=getattr(viz,'_viewer',None)
                if viewer is not None:
                    shifted=center+viewer.world_offsets.numpy()[args.video_env]
                    viz.set_camera_view((shifted+[0,-2.0,.30]).tolist(),(shifted+[0,0,-.12]).tolist())
        with torch.inference_mode():
            obs,reward,dones,info = wrapper.step(policy(obs))
        snap = e.eval_snapshot
        pos = snap['pos']-origin
        upright = snap['gravity'][:,2] < -0.75
        supported = (snap['forces'].norm(dim=-1)>5).any(dim=1)
        correct_height = (pos[:,2]-(end_ground_delta+cfg.robot.init_state.pos[2])).abs()<0.25
        reached = (pos[:,0]>=goal_progress)&upright&supported&correct_height
        left_lane = pos[:,1].abs()>0.8
        won = active&reached&~snap['terminated']&~snap['truncated']&~left_lane
        success |= won
        fell |= active&snap['terminated']
        base_terminated |= active&snap['terminated']&snap['base_hit']
        hip_terminated |= active&snap['terminated']&snap['hip_hit']
        tipped_seen |= active&(snap['gravity'][:,2]>-0.5)
        lateral |= active&left_lane&~snap['terminated']
        max_progress = torch.maximum(max_progress,torch.where(active,pos[:,0],max_progress))
        terminal_position = torch.where(active[:,None],pos,terminal_position)
        survival_steps += active
        nonfoot_steps += (snap['z1_nonfoot_forces'].norm(dim=-1).amax(dim=-1)>5.)&active
        calf_steps += (snap['z1_calf_forces'].norm(dim=-1).amax(dim=-1)>5.)&active
        tau=snap['torque'].abs().amax(dim=-1); vel=snap['z1_joint_vel'].abs().amax(dim=-1)
        peak_torque=torch.maximum(peak_torque,torch.where(active,tau,peak_torque))
        peak_speed=torch.maximum(peak_speed,torch.where(active,vel,peak_speed))
        torque_violation |= active&(tau>120.001);speed_violation |= active&(vel>20.01)
        near_limit_steps += (snap['torque'][:,e._ids_joints_order].abs()>=0.95*policy_limits)&active[:,None]
        base_contact_steps += (snap['base_force'].norm(dim=-1).amax(dim=-1)>1)&active
        horizontal_force = snap['forces'][:,:,:2].norm(dim=-1)
        horizontal_dominant = ((horizontal_force>4*snap['forces'][:,:,2].abs())&(horizontal_force>5)).any(dim=1)
        horizontal_contact_steps += horizontal_dominant&active
        velocity_error_sum += (snap['vel'][:,0]-snap['commands'][:,0]).abs()*active
        contacts = (snap['forces'].norm(dim=-1)>5)&active[:,None]
        slip_sum += (snap['foot_vel'][:,:,:2].norm(dim=-1)*contacts).sum(dim=1)
        slip_count += contacts.sum(dim=1)
        if args.video and bool(active[args.video_env]):
            i = args.video_env
            video_trace.append({
                'time_seconds':(step+1)*e.step_dt,'environment_index':i,
                'relative_xyz':pos[i].tolist(),
                'projected_gravity':snap['gravity'][i].tolist(),
                'quaternion_xyzw':snap['quaternion'][i].tolist(),
                'joint_positions_policy_order_rad':snap['joint_pos'][i,e._ids_joints_order].tolist(),
                'joint_velocities_policy_order_radps':snap['joint_vel'][i,e._ids_joints_order].tolist(),
                'joint_targets_policy_order_rad':snap['joint_target_policy_order'][i].tolist(),
                'last_pd_computed_torque_nm':snap['computed_torque'][i,e._ids_joints_order].tolist(),
                'last_clipped_pre_delay_torque_nm':snap['torque'][i,e._ids_joints_order].tolist(),
                'foot_relative_xyz':(snap['foot_pos'][i]-origin[i]).tolist(),
                'foot_force_xyz_n':snap['forces'][i].tolist(),
                'calf_force_norms_n':snap['z1_calf_forces'][i].norm(dim=-1).tolist(),
                'nonfoot_force_norms_n':snap['z1_nonfoot_forces'][i].norm(dim=-1).tolist(),
                'base_force_norm_n':float(snap['base_force'][i].norm()),
                'base_contact_history_peak_n':float(snap['base_history_peak'][i]),
                'hip_contact_history_peak_n':float(snap['hip_history_peak'][i]),
                'terminated':bool(snap['terminated'][i]),'truncated':bool(snap['truncated'][i]),
                'success':bool(won[i]),'left_lane':bool(left_lane[i]),
            })
        if step%5 == 0:
            traces.append({'t_s':step*e.step_dt,'env0_xyz':pos[0].tolist(),
                           'env0_first_trial_active':bool(active[0]),
                           'env0_projected_gravity':snap['gravity'][0].tolist(),
                           'env0_quaternion_xyzw':snap['quaternion'][0].tolist(),
                           'env0_body_commands':snap['commands'][0].tolist(),
                           'env0_joint_torque_nm':snap['torque'][0,e._ids_joints_order].tolist(),
                           'env0_foot_relative_xyz':(snap['foot_pos'][0]-origin[0]).tolist(),
                           'env0_foot_force_xyz_n':snap['forces'][0].tolist(),
                           'env0_base_force_norm_n':float(snap['base_force'][0].norm()),
                           'active_count':int(active.sum()),'success_count':int(success.sum())})
        active &= ~(won|snap['terminated']|snap['truncated']|left_lane)
        if not active.any():
            break
    strict_safe=success&(nonfoot_steps==0)&~torque_violation&~speed_violation
    result = {'task':args.task,'parameter_status':cfg.z1_parameter_status,
        'contact_profile':cfg.z1_contact_profile,'asset':cfg.robot.spawn.usd_path,
        'strict_safe_success_count':int(strict_safe.sum()),'strict_safe_success_rate':float(strict_safe.float().mean()),
        'strict_safe_definition':'Eight-riser first-trial traversal with zero sampled nonfoot contacts above5N, no torque>120.001Nm or speed>20.01rad/s',
        'nonfoot_contact_trial_count':int((nonfoot_steps>0).sum()),'calf_contact_trial_count':int((calf_steps>0).sum()),
        'per_trial_nonfoot_contact_fraction':(nonfoot_steps/survival_steps.clamp(min=1)).tolist(),
        'per_trial_calf_contact_fraction':(calf_steps/survival_steps.clamp(min=1)).tolist(),
        'torque_violation_trial_count':int(torque_violation.sum()),'speed_violation_trial_count':int(speed_violation.sum()),
        'per_trial_peak_torque_Nm':peak_torque.tolist(),'per_trial_peak_speed_rad_s':peak_speed.tolist(),
        'strict_safe_sampling_limit':'Control-step samples at50Hz; brief intersample contacts/overshoots may be missed. Not collision geometry or hardware certification.',
        'checkpoint':str(args.checkpoint),
        'evaluator_source_sha256':evaluator_source_hash,
        'terrain_source_sha256':hashlib.sha256((Path(__file__).parent/'stair_terrain.py').read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'height_m':args.step_height,
        'direction':'down' if args.descending else 'up','risers':8,'tread_m':0.30,
        'num_trials':args.num_envs,'seed':args.seed,'speed_command_mps':args.speed,
        'duration_limit_s':args.seconds,'randomized':args.randomized,
        'command_controller':{'mode':'fixed_body_forward' if args.fixed_body_command else 'world_heading_alignment','world_yaw_target_rad':None if args.fixed_body_command else 0,
                              'yaw_gain_per_s':0. if args.fixed_body_command else 1.5,'yaw_rate_limit_radps':0. if args.fixed_body_command else 0.7,
                              'yaw_source':None if args.fixed_body_command else 'simulation root orientation, xyzw quaternion',
                              'real_requirement':'Body velocity command only; no external yaw correction' if args.fixed_body_command else 'IMU yaw with an external heading reference; not a deployment implementation'},
        'fixed_body_command':args.fixed_body_command,
        'domain':'Randomized simulated Z1 dynamics and observation/action noise' if args.randomized else 'Nominal Z1 dynamics, observation noise disabled, native action noise retained',
        'domain_note':'All dynamics are simulated/provisional. Fixed-body-command mode uses no external yaw feedback.' if args.fixed_body_command else 'External world-yaw feedback is supplied; report separately from fixed body commands.',
        'observation_noise_enabled':cfg.observation_noise_model is not None,
        'action_noise_enabled':cfg.action_noise_model is not None,
        'enabled_event_terms':[name for name,value in vars(cfg.events).items()
                               if not name.startswith('_') and value is not None],
        'initial_dynamics':initial_dynamics,
        'video_environment_index':args.video_env if args.video else None,
        'video_trace':video_trace,
        'camera_mode':'robot-centered side follow' if args.side_follow_camera else 'fixed overview',
        'video_trace_scope':'Selected environment at every control step before auto-reset, through its first outcome only; video may continue while other environments finish',
        'recording_scope':'First trial through success, fall/lateral exit, or time limit; recorder flushes buffered real frames at environment close',
        'success_count':int(success.sum()),'success_rate':float(success.float().mean()),
        'fall_count':int(fell.sum()),'lateral_exit_count':int(lateral.sum()),
        'termination_with_base_contact_count':int(base_terminated.sum()),
        'termination_with_hip_contact_count':int(hip_terminated.sum()),
        'tipped_trial_count':int(tipped_seen.sum()),
        'failure_definition':'Legacy fall_count counts native base/hip contact terminations above 1N, not necessarily large orientation falls. tipped_trial_count separately flags projected gravity z>-0.5 before first-trial completion; base and hip reasons may overlap.',
        'incomplete_count':int((~success&~fell&~lateral).sum()),
        'mean_max_progress_m':float(max_progress.mean()),'goal_progress_m':goal_progress,
        'mean_first_trial_duration_s':float(survival_steps.mean()*e.step_dt),
        'mean_abs_forward_velocity_error_mps':float((velocity_error_sum/survival_steps.clamp(min=1)).mean()),
        'mean_contact_foot_horizontal_speed_mps':float((slip_sum/slip_count.clamp(min=1)).mean()),
        'policy_joint_order':[e._robot.joint_names[i] for i in e._ids_joints_order],
        'nominal_actuator_effort_limit_nm':policy_limits[0].tolist(),
        'torque_telemetry_stage':'Z1 provisional PaceDCMotor, zero delay; last physics-step clipped effort sampled at50Hz',
        'mean_sampled_joint_near_limit_fraction':(near_limit_steps/survival_steps.clamp(min=1)[:,None]).mean(dim=0).tolist(),
        'mean_first_trial_base_contact_fraction':float((base_contact_steps/survival_steps.clamp(min=1)).mean()),
        'mean_first_trial_horizontal_dominant_foot_contact_fraction':float((horizontal_contact_steps/survival_steps.clamp(min=1)).mean()),
        'diagnostic_limits':'Near-limit fractions use 95% of nominal actuator effort limits at control-step samples, not the speed-dependent envelope. Horizontal-dominant contact is a force proxy, not geometric proof of a riser contact.',
        'per_trial_success':success.tolist(),'per_trial_max_progress_m':max_progress.tolist(),
        'per_trial_fall':fell.tolist(),'per_trial_lateral_exit':lateral.tolist(),
        'per_trial_duration_s':(survival_steps*e.step_dt).tolist(),
        'terminal_relative_xyz':terminal_position.tolist(),'trace':traces,
        'success_definition':'Cross all eight risers plus 0.4m landing, upright, foot support and matching final elevation before first fall/reset/lateral exit',
        'limitations':('A finite simulated randomized sample is not Sim-to-Real evidence. Newton uses a single shape friction coefficient; independent dynamic joint friction is unsupported. Interval pushes may not occur before a fast trial ends.'
                       if args.randomized else 'Nominal test, not randomized robustness or Sim-to-Real evidence.')
                       +' Feet can touch a riser as well as a tread. Provisional Z1 actuator parameters are not hardware identification.'}
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(args.output)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('per_trial') and k not in ('trace','terminal_relative_xyz')},indent=2))
    env.close()
