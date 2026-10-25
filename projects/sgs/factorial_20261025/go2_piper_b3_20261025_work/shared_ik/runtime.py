"""Isaac adapter. Import only after AppLauncher; CPU tests use the core package.

This is the single B2/B3 action composition wrapper. The native physical
manager, reward, observation history and delayed actuators still receive 18D.
"""
import torch
import hashlib
import json
from pathlib import Path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from .controller import ArmOnlyIK
from .mapping import NativeActionMap, PolicyObservationDecoder
from .settings import LEG_NAMES, ARM_NAMES, VERSION
from .model import KINEMATICS


class LegPPOArmIKWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions=None, slew_mode='previous_reference'):
        if clip_actions is not None:
            raise ValueError('Current shared recipe has clip_actions=None; do not clip IK action components')
        super().__init__(env, clip_actions)
        raw = self.unwrapped; robot = raw.scene['robot']
        # The exported FK model is valid only for the exact unchanged USD layers.
        asset_root = Path(robot.cfg.spawn.usd_path).resolve().parent
        for relative, expected in json.loads(KINEMATICS.read_text())['layers'].items():
            suffix = relative.split('/assets/go2_piper/', 1)[1]
            if hashlib.sha256((asset_root/suffix).read_bytes()).hexdigest() != expected:
                raise ValueError('Fixed kinematic model does not match robot USD layer: '+suffix)
        if raw.action_manager.active_terms != ['joint_pos']:
            raise ValueError('Expected the unchanged single native joint_pos action term')
        term = raw.action_manager.get_term('joint_pos')
        self.mapper = NativeActionMap(robot.joint_names, term._joint_names,
                                     term._scale, term._offset,
                                     getattr(term, '_clip', None), raw.device)
        ids = list(range(len(robot.joint_names)))[term._joint_ids] if isinstance(term._joint_ids, slice) else list(term._joint_ids)
        if self.mapper.joint_ids != ids:
            raise ValueError('Native action joint-name/index mismatch')
        obsman = raw.observation_manager
        qcfg = obsman.get_term_cfg('policy', 'joint_pos')
        if getattr(qcfg.func, '__name__', '') != 'joint_pos_rel' or qcfg.scale not in (None, 1., (1.,)):
            raise ValueError('Unexpected actor joint-position transform')
        asset = qcfg.params.get('asset_cfg')
        obs_ids = slice(None) if asset is None else asset.joint_ids
        obs_names = robot.joint_names[obs_ids] if isinstance(obs_ids, slice) else [robot.joint_names[i] for i in obs_ids]
        self.decoder = PolicyObservationDecoder(obsman.active_terms['policy'], obsman.group_obs_term_dim['policy'],
                                               obs_names, robot.data.default_joint_pos[:, obs_ids])
        if self.decoder.width != 210:
            raise ValueError('Latest baseline actor input must remain 210D')
        self.arms = [robot.joint_names.index(n) for n in ARM_NAMES]
        self.controller = ArmOnlyIK(self.num_envs, raw.device, slew_mode=slew_mode)
        self.num_actions = 12
        self._actor_obs = None
        self.last_control = None

    def capture_input(self, obs):
        """Optional explicit hook used by evaluator/runner; no observation getter."""
        self._actor_obs = obs['policy'].detach()

    def get_observations(self):
        obs = super().get_observations()
        self.capture_input(obs)
        return obs

    def reset(self):
        obs, extras = super().reset()
        if hasattr(self, 'controller'):
            self.controller.reset()
        self.capture_input(obs)
        return obs, extras

    def step(self, actions):
        if actions.shape != (self.num_envs, 12):
            raise ValueError('Leg policy must supply [N,12]')
        if self._actor_obs is None:
            raise RuntimeError('Capture the actual actor input before composing actions')
        raw = self.unwrapped; robot = raw.scene['robot']
        state = self.decoder.decode(self._actor_obs)
        limits = robot.data.soft_joint_pos_limits[:, self.arms]
        lo, hi = self.mapper.arm_bounds(self.num_envs)
        lower = limits[..., 0] if lo is None else torch.maximum(limits[..., 0], lo)
        upper = limits[..., 1] if hi is None else torch.minimum(limits[..., 1], hi)
        qref, diag = self.controller.command(**state, dt=raw.step_dt, lower=lower, upper=upper,
                                            velocity_limits=robot.data.joint_vel_limits[:, self.arms])
        if not bool(diag['input_valid'].all()):
            raise RuntimeError('Invalid IK input/frame; finite hold is available only for diagnosis')
        executed, processed = self.mapper.compose(actions, qref)
        self.last_control = dict(raw12=actions.detach().clone(), a_exec18=executed.clone(),
                                 processed_target18=processed.clone(), **diag)
        obs, reward, done, extras = super().step(executed)
        self.controller.reset(done.nonzero(as_tuple=False).flatten())
        self.capture_input(obs)
        return obs, reward, done, extras

    @property
    def action_space(self):
        import gymnasium as gym
        import numpy as np
        return gym.spaces.Box(-np.inf,np.inf,shape=(self.num_envs,12),dtype=np.float32)

    def controller_state_dict(self):
        return self.controller.state_dict()

    def load_controller_state_dict(self, state):
        self.controller.load_state_dict(state)

    def binding(self):
        return dict(controller_version=VERSION, articulation_joint_names=self.mapper.articulation_names,
                    action_joint_names=self.mapper.action_names, action_joint_ids=self.mapper.joint_ids,
                    leg_action_slots=self.mapper.leg_slots, arm_action_slots=self.mapper.arm_slots,
                    joint_observation_names=self.decoder.joint_names,
                    feedback='exact noisy actor input; newest sample of each native term history',
                    action_history='native action_manager.action18',
                    action_rate='sum((action_manager.action18-prev_action18)^2)')
