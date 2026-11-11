"""Three complete trajectory counters; original strict metric is the shadow oracle."""
import numpy as np
import torch
from metrics import Metrics, physical_errors
from common import PROTOCOL
from e1_sampling import SCHEDULE

class E1Metrics(Metrics):
    def __init__(self, env, kind):
        super().__init__(env, kind)
        self.tri_good = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.long)
        self.tri_tol = self.tol.repeat(3, 1)
        self.tri_tol[:, 2] = torch.tensor([s['position_m'] for s in SCHEDULE],device=env.device,dtype=torch.float64)
        self.fractions = torch.tensor([s['joint_fraction'] for s in SCHEDULE],device=env.device,dtype=torch.float64)
        self.last_finished = None
        self.strict_compared_episodes = 0

    def reset(self, ids):
        super().reset(ids); self.tri_good[ids] = 0

    def capture(self):
        super().capture()
        e = physical_errors(self.env, self.kind)
        eligible = (self.age >= self.settle_steps).all(1)
        # Includes all four conditions on the SAME eligible step, per tolerance.
        self.tri_good += eligible[:, None] & (e[:, None, :] <= self.tri_tol[None, :, :]).all(2)

    def finish(self, ids):
        strict = super().finish(ids)
        n = self.scored[ids]
        rmse = torch.sqrt(self.sq[ids]/n.clamp(min=1)[:,None])
        labels = ((~self.env.reset_terminated[ids])[:,None]
                  & (n >= PROTOCOL['min_scored_steps'])[:,None]
                  & (self.tri_good[ids].double() >= n[:,None]*self.fractions)
                  & (rmse[:,None,:] <= self.tri_tol[None,:,:]).all(2)).cpu().numpy()
        good = self.tri_good[ids].cpu().numpy()
        assert np.array_equal(good[:,2], strict[:,1])
        assert np.array_equal(labels[:,2], strict[:,6].astype(bool)), 'E1 strict shadow differs from original SGS'
        assert (labels[:,0] >= labels[:,1]).all() and (labels[:,1] >= labels[:,2]).all()
        assert (good[:,0] >= good[:,1]).all() and (good[:,1] >= good[:,2]).all()
        self.last_finished = dict(labels=labels, good=good)
        self.strict_compared_episodes += len(ids)
        return strict

    def state_dict(self):
        s = super().state_dict(); s.update(tri_good=self.tri_good.cpu().clone(), strict_compared_episodes=self.strict_compared_episodes)
        return s

    def load_state_dict(self, s):
        super().load_state_dict({k:v for k,v in s.items() if k != 'strict_compared_episodes'})
        self.strict_compared_episodes = s['strict_compared_episodes']
