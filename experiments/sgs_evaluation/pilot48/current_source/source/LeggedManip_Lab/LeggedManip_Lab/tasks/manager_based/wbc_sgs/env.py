"""Record terminal state BEFORE Isaac autoreset; one assignment feeds state + both commands."""
import hashlib
import json
from pathlib import Path
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_apply_inverse, quat_mul, quat_error_magnitude
from benchmarks.go2_piper.wbc_sgs.metrics import JointMetrics
from benchmarks.go2_piper.wbc_sgs.sampler import TaskSampler

ACTION_NAMES = ['FR_hip_joint','FR_thigh_joint','FR_calf_joint','FL_hip_joint','FL_thigh_joint','FL_calf_joint',
                'RR_hip_joint','RR_thigh_joint','RR_calf_joint','RL_hip_joint','RL_thigh_joint','RL_calf_joint'] + [f'joint{i}' for i in range(1,7)]


class FixedTaskWBCEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self.pool = json.loads(Path(cfg.task_pool_path).read_text())
        self.protocol = json.loads(Path(cfg.success_protocol_path).read_text())
        assert [x['id'] for x in self.pool['tasks']] == list(range(len(self.pool['tasks'])))
        self.pool_sha256 = hashlib.sha256(Path(cfg.task_pool_path).read_bytes()).hexdigest()
        self.protocol_sha256 = hashlib.sha256(Path(cfg.success_protocol_path).read_bytes()).hexdigest()
        n = cfg.scene.num_envs
        self.wbc_task_ids = torch.full((n,), -1, device=cfg.sim.device, dtype=torch.long)
        self.task_velocities = torch.tensor([x['velocity'] for x in self.pool['tasks']], device=cfg.sim.device)
        self.task_poses = torch.tensor([x['ee_pose'] for x in self.pool['tasks']], device=cfg.sim.device)
        self.sampler = TaskSampler(len(self.pool['tasks']), cfg.sampler_seed, cfg.sampler_mode)
        self.joint_metrics = JointMetrics(n, cfg.sim.device, self.protocol)
        self.episode_records = []
        self.episode_serial = 0
        self.context = 'train'
        self.resume_ids = None
        # Upstream creates reset_buf only on the first step, after its initial reset.
        self.reset_buf = torch.zeros(n, dtype=torch.bool, device=cfg.sim.device)
        self._log = open(cfg.episode_log_path, 'a', buffering=1) if cfg.episode_log_path else None
        super().__init__(cfg, **kwargs)

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        old = self.wbc_task_ids[ids]
        valid = (old >= 0) & (self.episode_length_buf[ids] > 0)
        finished = ids[valid]
        if len(finished):
            # Manual reset (evaluation/checkpoint restart) discards partials, never fabricates failures.
            is_done = self.reset_buf[finished] & (self.episode_length_buf[finished] > 0)
            scored_ids = finished[is_done]
            if len(scored_ids):
                outcomes = self.joint_metrics.results(scored_ids.tolist(),
                    self.reset_time_outs[scored_ids].tolist(), self.reset_terminated[scored_ids].tolist())
                task_ids = self.wbc_task_ids[scored_ids].tolist()
                self.sampler.record(task_ids, [r['joint_success'] for r in outcomes])
                for env_id, task_id, record in zip(scored_ids.tolist(), task_ids, outcomes):
                    self.episode_serial += 1
                    record.update(episode=self.episode_serial, env_id=env_id, task_id=task_id,
                        sampler=self.sampler.mode, context=self.context, control_step=int(self.common_step_counter),
                        velocity=self.pool['tasks'][task_id]['velocity'], ee_pose=self.pool['tasks'][task_id]['ee_pose'],
                        pool_sha256=self.pool_sha256, protocol_sha256=self.protocol_sha256)
                    self.episode_records.append(record)
                    if self._log: self._log.write(json.dumps(record)+'\n')
        # Every old outcome is credited before sampling ANY new task in this reset batch.
        # Reused IDs after resume do not count as fresh categorical draws.
        if self.resume_ids is not None:
            self.wbc_task_ids[ids] = self.resume_ids[ids]
        else:
            self.wbc_task_ids[ids] = torch.tensor(self.sampler.sample(len(ids)), device=self.device)
        self.joint_metrics.reset(ids)
        super()._reset_idx(ids)
        # Both commands were reset by parent; validate actual sources (not shadow buffers).
        self.assert_task_commands()

    def assert_task_commands(self):
        active = self.wbc_task_ids >= 0
        for name, table in [('base_velocity',self.task_velocities),('ee_pose',self.task_poses)]:
            actual = self.command_manager.get_command(name)[active]
            if not torch.equal(actual, table[self.wbc_task_ids[active]]):
                raise AssertionError('Task/command mismatch: '+name)

    def sampler_checkpoint(self):
        return dict(sampler=self.sampler.state_dict(), active_task_ids=self.wbc_task_ids.cpu().tolist(),
                    partial_episode_steps=self.episode_length_buf.cpu().tolist(), episode_serial=self.episode_serial,
                    pool_sha256=self.pool_sha256, protocol_sha256=self.protocol_sha256,
                    restore_semantics='PPO/sampler/RNG restored; unfinished physics episodes restart same task from s0, no outcome recorded')

    def restore_sampler_checkpoint(self, state):
        assert state['pool_sha256'] == self.pool_sha256 and state['protocol_sha256'] == self.protocol_sha256
        assert len(state['active_task_ids']) == self.num_envs
        self.sampler.load_state_dict(state['sampler'])
        self.episode_serial = state['episode_serial']
        self.resume_ids = torch.tensor(state['active_task_ids'], device=self.device)
        # Reset buffers first: a manual restart cannot be scored as a terminal failure.
        self.episode_length_buf[:] = 0
        self.reset_buf[:] = False
        self.reset()
        self.resume_ids = None

    def close(self):
        if self._log:
            self._log.close(); self._log = None
        super().close()


def apply_task_reset(env, env_ids):
    robot = env.scene['robot']
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    tasks = [env.pool['tasks'][i]['initial'] for i in env.wbc_task_ids[ids].tolist()]
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] = torch.tensor([t['root_pos'] for t in tasks], device=env.device) + env.scene.env_origins[ids]
    rpy = torch.tensor([t['root_rpy'] for t in tasks], device=env.device)
    root[:, 3:7] = quat_from_euler_xyz(rpy[:,0],rpy[:,1],rpy[:,2])
    root[:, 7:13] = torch.tensor([t['root_velocity'] for t in tasks], device=env.device)
    robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
    q = robot.data.default_joint_pos[ids].clone()
    joint_ids,_ = robot.find_joints(ACTION_NAMES, preserve_order=True)
    assert len(joint_ids) == 18
    q[:,joint_ids[:12]] *= torch.tensor([t['leg_scale'] for t in tasks],device=env.device)[:,None]
    q[:,joint_ids[12:]] = torch.tensor([t['arm'] for t in tasks],device=env.device)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    robot.set_joint_position_target(q, env_ids=ids)


def observe_joint_metrics(env):
    env.assert_task_commands()
    robot=env.scene['robot']; sensor=env.scene['contact_forces']
    if not hasattr(env, '_metric_body_ids'):
        env._metric_body_ids=(robot.find_bodies('link0')[0][0],robot.find_bodies('end_effector')[0][0])
        env._nonfoot_ids=[i for i,n in enumerate(sensor.body_names) if not n.endswith('_foot')]
    link,ee=env._metric_body_ids
    v=env.command_manager.get_command('base_velocity'); p=env.command_manager.get_command('ee_pose')
    rel=quat_apply_inverse(robot.data.body_quat_w[:,link],robot.data.body_pos_w[:,ee]-robot.data.body_pos_w[:,link])
    # Use exactly the two reward frames, including base-relative orientation.
    pos=torch.cat([rel[:,:2]-p[:,:2],(robot.data.body_pos_w[:,ee,2]-env.scene.env_origins[:,2]-p[:,2])[:,None]],-1).norm(dim=-1)
    ori=quat_error_magnitude(robot.data.body_quat_w[:,ee],quat_mul(robot.data.root_quat_w,p[:,3:]))
    errors=torch.stack([(robot.data.root_lin_vel_b[:,:2]-v[:,:2]).norm(dim=-1),
                        (robot.data.root_ang_vel_b[:,2]-v[:,2]).abs(),pos,ori],-1)
    force=sensor.data.net_forces_w_history[:,:,env._nonfoot_ids].norm(dim=-1).amax(dim=(1,2))
    tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))
    height=robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]
    safe=(force<=env.protocol['nonfoot_contact_N']) & (tilt<=env.protocol['max_tilt_rad']) & (height>=env.protocol['min_base_height_m'])
    safe &= torch.isfinite(robot.data.joint_pos).all(-1) & torch.isfinite(robot.data.joint_vel).all(-1)
    env.joint_metrics.update(errors,safe,force,env.step_dt)
    env.last_joint_errors=errors.detach().clone()
    return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
