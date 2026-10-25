"""Pre-reset episode accumulators shared across all four factorial groups.

Computes raw metrics, never SGS labels: label conversion belongs to B1.
"""
from __future__ import annotations
import copy
import math
import numpy as np

FIELDS = ('xy_sq','yaw_sq','ee_position_sq','ee_orientation_sum','clearance_min',
          'velocity_ratio_max','joint_excess_max','nonfoot_contact','steps','failure')

class EpisodeMetrics:
    def __init__(self, num_envs, horizon_steps=1000):
        if num_envs <= 0 or horizon_steps <= 0: raise ValueError('positive dimensions required')
        self.num_envs, self.horizon_steps = int(num_envs), int(horizon_steps)
        self.reset(range(num_envs), first=True)
    def reset(self, env_ids, first=False):
        ids = np.asarray(list(env_ids), dtype=int)
        if first:
            self.data = {k:np.zeros(self.num_envs, dtype=float) for k in FIELDS}
        for k, values in self.data.items(): values[ids] = np.inf if k == 'clearance_min' else 0
    def capture(self, env_ids, *, base_command, base_actual, ee_position_error, ee_orientation_error,
                base_clearance, joint_velocity_limit_ratio, joint_position_limit_excess, nonfoot_max_force,
                failure):
        ids = np.asarray(list(env_ids), dtype=int)
        if len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= self.num_envs).any():
            raise ValueError('unique in-range environment IDs required')
        command, actual = np.asarray(base_command, dtype=float), np.asarray(base_actual, dtype=float)
        if command.shape != (len(ids), 3) or actual.shape != command.shape:
            raise ValueError('Base arrays must be Nx3: vx, vy, yaw_rate')
        vec = [np.asarray(x, dtype=float) for x in (ee_position_error,ee_orientation_error,base_clearance,
            joint_velocity_limit_ratio,joint_position_limit_excess,nonfoot_max_force,failure)]
        if any(v.shape != (len(ids),) for v in vec) or not all(np.isfinite(v).all() for v in [command,actual]+vec):
            raise ValueError('Non-finite or invalid metric sample; abort without converting to success/failure')
        pos, ori, clearance, velocity, excess, force, fail = vec
        if (pos < 0).any() or (ori < 0).any() or (velocity < 0).any() or (excess < 0).any() or (force < 0).any():
            raise ValueError('Invalid unsigned metric')
        d = self.data
        d['xy_sq'][ids] += np.square(actual[:,:2]-command[:,:2]).sum(-1)
        d['yaw_sq'][ids] += np.square(actual[:,2]-command[:,2])
        d['ee_position_sq'][ids] += pos**2
        d['ee_orientation_sum'][ids] += ori
        d['clearance_min'][ids] = np.minimum(d['clearance_min'][ids],clearance)
        d['velocity_ratio_max'][ids] = np.maximum(d['velocity_ratio_max'][ids],velocity)
        d['joint_excess_max'][ids] = np.maximum(d['joint_excess_max'][ids],excess)
        d['nonfoot_contact'][ids] = np.maximum(d['nonfoot_contact'][ids],force > 1.)
        d['failure'][ids] = np.maximum(d['failure'][ids],fail.astype(bool))
        d['steps'][ids] += 1
    def finish(self, env_id):
        i = int(env_id); d=self.data; n=int(d['steps'][i])
        if n <= 0: raise ValueError('Empty episode has no terminal metrics')
        failure=bool(d['failure'][i]); complete=n==self.horizon_steps and not failure
        row={'steps':n,'failure':failure,'complete':complete,
          'xy_rmse':math.sqrt(d['xy_sq'][i]/n),'yaw_rmse':math.sqrt(d['yaw_sq'][i]/n),
          'ee_position_rmse':math.sqrt(d['ee_position_sq'][i]/n),'ee_orientation_mean':float(d['ee_orientation_sum'][i]/n),
          'base_clearance_min':float(d['clearance_min'][i]),'joint_velocity_limit_ratio_max':float(d['velocity_ratio_max'][i]),
          'joint_position_limit_excess_max_rad':float(d['joint_excess_max'][i]),'nonfoot_contact':bool(d['nonfoot_contact'][i])}
        return row
    def state_dict(self):
        return {'schema':1,'num_envs':self.num_envs,'horizon_steps':self.horizon_steps,'data':{k:[None if k == 'clearance_min' and not np.isfinite(x) else float(x) for x in v] for k,v in self.data.items()}}
    def load_state_dict(self, state):
        if state['schema'] != 1 or state['num_envs'] != self.num_envs or state['horizon_steps'] != self.horizon_steps or set(state['data']) != set(FIELDS):
            raise ValueError('Metric state schema/dimensions mismatch')
        data = {k:np.asarray([np.inf if x is None and k == 'clearance_min' else x for x in v],dtype=float) for k,v in copy.deepcopy(state['data']).items()}
        if any(v.shape != (self.num_envs,) for v in data.values()): raise ValueError('Metric state shape mismatch')
        if any(not np.isfinite(v).all() for k,v in data.items() if k != 'clearance_min'):
            raise ValueError('Non-finite metric state')
        if np.isnan(data['clearance_min']).any() or (data['steps'] < 0).any() or (data['steps'] % 1).any():
            raise ValueError('Corrupt metric state')
        self.data=data
