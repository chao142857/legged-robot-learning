"""Small task adapters; native actions, rewards and PPO remain in upstream code."""
from __future__ import annotations
import math
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply_inverse, euler_xyz_from_quat
from isaaclab.utils.warp import raycast_mesh
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_wbc import UniformPoseWBCCommand


def terrain_surface_height(env, positions_xy):
    """Raycast true static collision mesh. Missing hits remain inf, never fake flat."""
    if env.cfg.task_suite['family'] != 'T2':
        return torch.zeros(positions_xy.shape[:-1], device=env.device)
    sensor = env.scene.sensors['height_scanner']
    starts = torch.zeros((*positions_xy.shape[:-1], 3), device=env.device)
    starts[..., :2] = positions_xy
    starts[..., 2] = 10.0
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.0
    hits, _, _, _ = raycast_mesh(starts, directions, sensor.meshes['/World/ground'], max_dist=20.)
    return hits[..., 2]


def support_height(env):
    """Median terrain height over base footprint; excludes gaps >.12m below top."""
    if env.cfg.task_suite['family'] != 'T2':
        return torch.zeros(env.num_envs, device=env.device)
    hits = env.scene.sensors['height_scanner'].data.ray_hits_w[..., 2]
    finite = torch.isfinite(hits)
    top = torch.where(finite, hits, -torch.inf).amax(-1, keepdim=True)
    surface = torch.where(finite & (hits >= top-.12), hits, torch.nan)
    height = surface.nanmedian(dim=-1).values
    if not torch.isfinite(height).all():
        raise RuntimeError('Missing terrain ray hits: support height is undefined')
    return height


def goal_distance(env):
    route = env.cfg.task_suite['route']
    if route is None:
        return torch.zeros(env.num_envs, device=env.device)
    goal = env.scene.env_origins[:, :2] + torch.tensor(route['goal_offset'], device=env.device)
    return (env.scene['robot'].data.root_pos_w[:, :2] - goal).norm(dim=-1)


def route_progress(env):
    route = env.cfg.task_suite['route']
    if route is None:
        return torch.zeros(env.num_envs, device=env.device)
    start = torch.tensor(route['start_offset'], device=env.device)
    delta = torch.tensor(route['goal_offset'], device=env.device) - start
    displacement = env.scene['robot'].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2] - start
    return (displacement * delta/delta.norm()).sum(-1)


def reset_suite_root(env, env_ids):
    robot = env.scene['robot']
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    route = env.cfg.task_suite['route']
    if route is not None:
        root[:, :2] = env.scene.env_origins[ids, :2] + torch.tensor(route['start_offset'], device=env.device)
        ground = terrain_surface_height(env, root[:, :2])
        if not torch.isfinite(ground).all():
            raise RuntimeError('Reset has no supporting terrain')
        root[:, 2] = ground + robot.data.default_root_state[ids, 2]
    # Deterministic yaw aligns all command groups with the same spatial frame.
    root[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=env.device)
    root[:, 7:13] = 0.
    robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(root[:, 7:13], env_ids=ids)


def terrain_relative_base_height(env, target_height, std, asset_cfg=None):
    height = env.scene['robot'].data.root_pos_w[:, 2] - support_height(env)
    return torch.exp(-(height-target_height).abs()/std)


def nonfoot_contact(env, threshold=5., grace_s=.5):
    sensor = env.scene.sensors['contact_forces']
    ids = [i for i, name in enumerate(sensor.body_names) if not name.endswith('_foot')]
    force = sensor.data.net_forces_w_history[:, :, ids].norm(dim=-1).amax(dim=(1, 2))
    return (force > threshold) & (env.episode_length_buf*env.step_dt > grace_s)


def route_boundary(env):
    rel = env.scene['robot'].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (rel.abs() > 3.9).any(-1) | (env.scene['robot'].data.root_pos_w[:, 2] < -0.15)


class SuitePoseCommand(UniformPoseWBCCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._suite = env.cfg.task_suite
        self._support_z = torch.zeros(env.num_envs, device=env.device)

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        ground = terrain_surface_height(self.env, self.robot.data.root_pos_w[:, :2])
        self._support_z[env_ids] = ground[env_ids]
        # Isaac Lab resets command terms before clearing episode_length_buf.
        # Explicit reset phase prevents the previous episode's age leaking into
        # the command returned by reset(); non-reset environments stay untouched.
        self._update_command(env_ids, reset_phase=True)

    def _update_command(self, env_ids=None, reset_phase=False):
        ids = slice(None) if env_ids is None else env_ids
        p = self._suite['params']
        t = self.env.episode_length_buf[ids].to(torch.float32) * self.env.step_dt
        if reset_phase:
            t = torch.zeros_like(t)
        phase = t * (2*math.pi/p['ee_period_s'])
        center = torch.tensor(p['ee_center'], device=self.device)
        amp = torch.tensor(p['ee_amplitude'], device=self.device)
        if self._suite['ee_mode'] == 'smooth':
            offsets = torch.stack((torch.sin(phase), torch.cos(phase), -torch.sin(phase)), dim=-1)*amp
        else:
            offsets = torch.zeros(len(t), 3, device=self.device)
        self.pose_command[ids, :3] = center + offsets
        ground = support_height(self.env)[ids]
        # Slew terrain offset to prevent discontinuous EE targets at step edges.
        self._support_z[ids] += (ground-self._support_z[ids]).clamp(-.1*self.env.step_dt, .1*self.env.step_dt)
        self.pose_command[ids, 2] += self._support_z[ids]
        link_z = self.robot.data.body_pos_w[ids, self.link0_idx, 2]
        radius = self.pose_command[ids, :2].norm(dim=-1)
        orientation_offsets = p['ee_orientation_offset_rpy']
        roll = torch.full_like(phase, orientation_offsets[0])
        pitch = -torch.atan2(self.pose_command[ids, 2]-link_z, radius) + orientation_offsets[1]
        yaw = torch.atan2(self.pose_command[ids, 1], self.pose_command[ids, 0]) + orientation_offsets[2]
        self.pose_command[ids, 3:] = quat_from_euler_xyz(roll, pitch, yaw)


class SuiteVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        self._suite = env.cfg.task_suite

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.
        self.is_standing_env[env_ids] = self._suite['family'] == 'T0'

    def _update_command(self):
        p = self._suite['params']
        desired = torch.zeros_like(self.vel_command_b)
        age = self.env.episode_length_buf.to(torch.float32)*self.env.step_dt
        family = self._suite['family']
        if family == 'T1':
            # Balanced groups start in forward, reverse, left-turn and right-turn;
            # every episode subsequently changes direction without a fixed task ID.
            group = (torch.arange(self.num_envs, device=self.device) + (age/p['base_segment_s']).long()) % 4
            desired[:, 0] = torch.where(group == 0, p['base_linear_speed'], torch.where(group == 1, -p['base_linear_speed'], 0.))
            desired[:, 2] = torch.where(group == 2, p['base_yaw_rate'], torch.where(group == 3, -p['base_yaw_rate'], 0.))
        elif family == 'T2':
            route = self._suite['route']
            robot = self.env.scene['robot']
            goal = self.env.scene.env_origins[:, :2]+torch.tensor(route['goal_offset'], device=self.device)
            delta = goal-robot.data.root_pos_w[:, :2]
            distance = delta.norm(dim=-1)
            direction_w = torch.zeros(self.num_envs, 3, device=self.device)
            direction_w[:, :2] = delta / distance.clamp(min=1e-6).unsqueeze(-1)
            direction_b = quat_apply_inverse(robot.data.root_quat_w, direction_w)
            desired[:, :2] = direction_b[:, :2] * torch.minimum(distance, torch.full_like(distance, p['terrain_speed'])).unsqueeze(-1)
            desired[:, 2] = torch.atan2(direction_b[:, 1], direction_b[:, 0]).clamp(-.3, .3)
            desired[distance < route['goal_tolerance_m']] = 0.
            desired[age < route['settle_s']] = 0.
        slew = torch.tensor([p['base_slew_linear'], p['base_slew_linear'], p['base_slew_yaw']], device=self.device)*self.env.step_dt
        self.vel_command_b += (desired-self.vel_command_b).clamp(-slew, slew)
