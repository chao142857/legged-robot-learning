"""Position target constraints and explicitly stateful flat-ground health checks."""
import torch
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import ManagerTermBase


class SoftLimitJointPositionAction(JointPositionAction):
    """Keep the native affine/raw interface, then bound the position in radians."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        margin = env.cfg.recipe_v1['control']['target_margin_rad']
        bounds = self._asset.data.soft_joint_pos_limits[:, self._joint_ids].clone()
        self.target_lower = bounds[..., 0] + margin
        self.target_upper = bounds[..., 1] - margin
        assert bool((self.target_lower < self.target_upper).all())
        self.native_targets = self.processed_actions.clone()
        self.target_was_limited = torch.zeros_like(self.raw_actions, dtype=torch.bool)

    def process_actions(self, actions):
        super().process_actions(actions)
        self.native_targets.copy_(self.processed_actions)
        bounded = self.processed_actions.clamp(self.target_lower, self.target_upper)
        self.target_was_limited.copy_((bounded-self.processed_actions).abs() > 1e-7)
        self._processed_actions.copy_(bounded)


def base_clearance(env):
    # Recipe V1 is explicitly restricted to the authored horizontal plane.
    return env.scene['robot'].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def low_height_penalty(env, threshold=.20, scale=.10):
    return ((threshold-base_clearance(env)).clamp_min(0)/scale).square()


def raw_action_tail_penalty(env, threshold=4.):
    return (env.action_manager.action.abs()-threshold).clamp_min(0).square().mean(-1)


def ee_precision(env, std=.05):
    from telemetry import physical
    return torch.exp(-physical(env)['ee_position_error'].square()/std**2)


class SustainedLowHeight(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.duration = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.duration[env_ids] = 0.

    def __call__(self, env, threshold=.16, dwell_s=.5, grace_s=1.):
        active = (env.episode_length_buf*env.step_dt > grace_s) & (base_clearance(env) < threshold)
        self.duration[:] = torch.where(active, self.duration+env.step_dt, 0.)
        return self.duration >= dwell_s


def control_diagnostics(env, margin=.02):
    robot = env.scene['robot']; d = robot.data
    term = env.action_manager.get_term('joint_pos'); ids = term._joint_ids
    soft = d.soft_joint_pos_limits[:, ids]
    q = d.joint_pos[:, ids]
    hard = d.joint_pos_limits[:, ids]
    unbounded = term.raw_actions*term._scale + term._offset
    native = unbounded.clamp(-10., 10.)
    lower, upper = soft[..., 0]+margin, soft[..., 1]-margin
    applied = term.processed_actions
    computed, effort = d.computed_torque[:, ids], d.applied_torque[:, ids]
    return dict(raw_action=term.raw_actions, affine_target=unbounded,
        native_clipped_target=native, processed_target=applied,
        joint_position=q, joint_velocity=d.joint_vel[:, ids],
        soft_limit_excess=torch.maximum((soft[..., 0]-q).clamp_min(0),(q-soft[..., 1]).clamp_min(0)),
        hard_limit_excess=torch.maximum((hard[..., 0]-q).clamp_min(0),(q-hard[..., 1]).clamp_min(0)),
        soft_margin=torch.minimum(q-soft[..., 0],soft[..., 1]-q),
        target_limit_excess=torch.maximum((lower-applied).clamp_min(0),(applied-upper).clamp_min(0)),
        shadow_target_would_clip=(native < lower-1e-7) | (native > upper+1e-7),
        target_was_clipped=(native-applied).abs() > 1e-7,
        computed_torque=computed, applied_torque=effort,
        actuator_saturated_last_substep=(computed-effort).abs() > 1e-5,
        pd_target_error=applied-q)


class ActuatorMonitor:
    """Count saturation on every physics substep without changing compute calls."""
    def __init__(self, env):
        self.env, self.raw = env, env.unwrapped
        self.original_step = env.step
        self.original_computes = {}
        d = self.raw.scene['robot'].data
        self.hits = torch.zeros_like(d.joint_pos)
        self.counts = torch.zeros_like(d.joint_pos)
        self.fraction = torch.zeros_like(d.joint_pos)
        for name, actuator in self.raw.scene['robot'].actuators.items():
            original = actuator.compute
            self.original_computes[name] = original
            def monitored(*a, _act=actuator, _compute=original, **kw):
                result = _compute(*a, **kw)
                ids = _act.joint_indices
                self.hits[:, ids] += ((_act.computed_effort-_act.applied_effort).abs() > 1e-5).float()
                self.counts[:, ids] += 1.
                self.fraction[:, ids] = self.hits[:, ids]/self.counts[:, ids].clamp_min(1.)
                return result
            actuator.compute = monitored
        env.step = self.step

    def step(self, actions):
        self.hits.zero_(); self.counts.zero_(); self.fraction.zero_()
        return self.original_step(actions)

    def close(self):
        self.env.step = self.original_step
        for name, original in self.original_computes.items():
            self.raw.scene['robot'].actuators[name].compute = original
