"""Independent Z1 tasks, native IIT observations/PPO, explicitly provisional motors."""
from pathlib import Path
import os
import gymnasium as gym
import torch
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab_newton.physics import NewtonShapeCfg
from basic_locomotion_isaaclab.actuators import PaceDCMotorCfg
from basic_locomotion_isaaclab.tasks.locomotion.locomotion_env import LocomotionEnv
from basic_locomotion_isaaclab.tasks.locomotion.agents.rsl_rl_ppo_cfg import RoughPPORunnerCfg
from experiment_cfg import A2Stairs20CurriculumCfg

ASSET = Path(os.environ['Z1_USD_PATH'])

@configclass
class Z1ShapeCfg(NewtonShapeCfg):
    # All moving links have authored positive mass/inertia. Missing mass on welded
    # foot helpers must stay zero; Newton otherwise infers additional shape mass.
    density: float = 0.0

@configclass
class Z1BaseOnlyCfg(A2Stairs20CurriculumCfg):
    z1_contact_profile: str = 'base_only'
    z1_diagnostic: bool = False
    z1_parameter_status: str = 'provisional: no measured motor PD/armature/friction/delay/torque-speed curve'

    def __post_init__(self):
        super().__post_init__()
        self.robot = self.robot.copy()
        self.robot.spawn.usd_path = str(ASSET)
        self.robot.init_state.pos = (0., 0., .413)
        self.robot.init_state.joint_pos = {'.*_hip_joint':0., '.*_thigh_joint':.9, '.*_calf_joint':-1.8}
        self.robot.actuators = {
            kind: PaceDCMotorCfg(joint_names_expr=[f'.*_{kind}_joint'],
                saturation_effort=120., actuator_effort_limit=120., actuator_velocity_limit=20.,
                joint_effort_limit=120., joint_velocity_limit=20.,
                stiffness=kp, damping=kd, armature=.02, friction=0.,
                dynamic_friction=0., viscous_friction=0., encoder_bias=0., max_delay=0)
            for kind,kp,kd in [('hip',80.,4.),('thigh',180.,8.),('calf',180.,8.)]}
        shape = self.sim.physics.newton_mjwarp.default_shape_cfg
        self.sim.physics.newton_mjwarp.default_shape_cfg = Z1ShapeCfg(
            margin=shape.margin, gap=shape.gap, ke=shape.ke, kd=shape.kd, mu=shape.mu)
        self.desired_base_height = .403
        self.desired_hip_offset_y = .106
        self.desired_hip_offset_x = 0.
        # Isaac Lab's native mass event rescales inertia by new/default mass.
        # Selecting zero-mass welded helpers would compute 0/0 at startup.
        self.events.scale_all_link_masses.params['asset_cfg'] = SceneEntityCfg(
            'robot',body_names=['base','.*_(hip|thigh|calf)'])

@configclass
class Z1LegSafeCfg(Z1BaseOnlyCfg):
    z1_contact_profile: str = 'leg_safe'

class Z1Env(LocomotionEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._calf_contact_sensor_ids, _ = self._contact_sensor.find_bodies(
            ['FL_calf','FR_calf','RL_calf','RR_calf'], preserve_order=True)
        self._z1_nonfoot_ids = (self._base_contact_sensor_id + self._hip_contact_sensor_ids
            + self._thigh_contact_sensor_ids + self._calf_contact_sensor_ids)
        self._undesired_contact_body_ids = (self._base_contact_sensor_id
            if cfg.z1_contact_profile == 'base_only' else self._z1_nonfoot_ids)

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        limits = self._robot.data.joint_pos_limits
        limits = limits if torch.is_tensor(limits) else limits.torch
        ordered = limits[:, self._ids_joints_order]
        self._processed_actions = self._processed_actions.clamp(ordered[:,:,0]+.005, ordered[:,:,1]-.005)
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._ids_joints_order)

    def _get_dones(self):
        if self.cfg.z1_diagnostic:
            zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return zero, zero.clone()  # Diagnostic must not hide falling through automatic reset.
        return super()._get_dones()

    def _get_rewards(self):
        reward = super()._get_rewards()
        forces = self._contact_sensor.data.net_forces_w
        forces = forces if torch.is_tensor(forces) else forces.torch
        effort = self._robot.data.applied_torque
        effort = effort if torch.is_tensor(effort) else effort.torch
        velocity = self._robot.data.joint_vel
        velocity = velocity if torch.is_tensor(velocity) else velocity.torch
        log = self.extras.setdefault('log', {})
        log['Z1/nonfoot_contact_fraction'] = (forces[:,self._z1_nonfoot_ids].norm(dim=-1)>1.).float().mean()
        log['Z1/calf_contact_fraction'] = (forces[:,self._calf_contact_sensor_ids].norm(dim=-1)>1.).float().mean()
        log['Z1/torque_over_120_fraction'] = (effort.abs()>120.001).float().mean()
        log['Z1/speed_over_20_fraction'] = (velocity.abs()>20.01).float().mean()
        return reward

for name,cfg in [('BaseOnly',Z1BaseOnlyCfg), ('LegSafe',Z1LegSafeCfg)]:
    gym.register(f'Locomotion-Z1-Rough-Blind-{name}', entry_point=Z1Env,
        disable_env_checker=True, kwargs={'env_cfg_entry_point':cfg,'rsl_rl_cfg_entry_point':RoughPPORunnerCfg})
