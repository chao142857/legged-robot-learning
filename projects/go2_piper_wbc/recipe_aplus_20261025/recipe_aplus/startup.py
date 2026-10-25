"""Frozen-policy study only. This class is never selected by A+ training."""
import torch
from recipe_v1.control import SoftLimitJointPositionAction


class StartupRampPositionAction(SoftLimitJointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.tau=float(env.cfg.startup_study['tau_s'])
        self.elapsed=torch.zeros(env.num_envs,device=env.device)
        self.first=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        self.initial_target=torch.zeros_like(self.raw_actions)
        self.initial_actual_q=torch.zeros_like(self.raw_actions)
        self.requested_target=torch.zeros_like(self.raw_actions)
        self.smoothing_delta=torch.zeros_like(self.raw_actions)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.elapsed[env_ids]=0.;self.first[env_ids]=True
        self.smoothing_delta[env_ids]=0.

    def process_actions(self,actions):
        super().process_actions(actions)
        self.requested_target.copy_(self.processed_actions)
        actual=self._asset.data.joint_pos[:,self._joint_ids]
        self.initial_actual_q[self.first]=actual[self.first]
        current=actual.clamp(self.target_lower,self.target_upper)
        self.initial_target[self.first]=current[self.first];self.first[:]=False
        if self.tau>0:
            phase=(self.elapsed/self.tau).clamp(0,1)
            alpha=phase.square()*(3.-2.*phase)
            output=self.initial_target+alpha[:,None]*(self.requested_target-self.initial_target)
            self._processed_actions.copy_(output.clamp(self.target_lower,self.target_upper))
        self.smoothing_delta.copy_(self.processed_actions-self.requested_target)
        self.elapsed+=self._env.step_dt
