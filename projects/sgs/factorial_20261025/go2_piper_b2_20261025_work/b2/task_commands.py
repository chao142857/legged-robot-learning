"""One common T1 phase-assignment hook for B0/B1/B2/B3; no sampler or IK."""
from __future__ import annotations
import hashlib
import importlib
from pathlib import Path
import numpy as np

EXECUTED_MDP_SHA256 = 'e5b1206233efd404d5c206054b7a1ff6ba6afee6776178f9620944d67fce2c85'

def phases_from_assignments(assignments, pool):
    return [int(pool.tasks[pool.index[a.task_id]].content['base_sequence_phase_offset']) for a in assignments]

def phase_groups(offsets, age_s, segment_s):
    offsets = np.asarray(offsets)
    ages = np.asarray(age_s, dtype=float)
    if offsets.ndim != 1 or ages.shape != offsets.shape or not np.isfinite(ages).all():
        raise ValueError('One finite age for every assigned phase required')
    if not np.isin(offsets, [0, 1, 2, 3]).all() or (ages < 0).any() or segment_s <= 0:
        raise ValueError('Invalid phase, age or segment')
    return (offsets.astype(np.int64) + np.floor(ages / segment_s).astype(np.int64)) % 4

def targets_numpy(offsets, age_s, params):
    """CPU oracle for command targets; native torch class uses the same phase rule."""
    group = phase_groups(offsets, age_s, params['base_segment_s'])
    target = np.zeros((len(group), 3))
    target[:, 0] = np.where(group == 0, params['base_linear_speed'], np.where(group == 1, -params['base_linear_speed'], 0))
    target[:, 2] = np.where(group == 2, params['base_yaw_rate'], np.where(group == 3, -params['base_yaw_rate'], 0))
    return target

def make_native_phase_command():
    """Called only inside an already prepared Isaac runtime, not by CPU preflight."""
    import torch
    module = importlib.import_module('task_suite.mdp')
    path = Path(module.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXECUTED_MDP_SHA256:
        raise RuntimeError('Executed T1 mdp source mismatch')
    class Phase4VelocityCommand(module.SuiteVelocityCommand):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            if self._suite['family'] != 'T1':
                raise ValueError('Shared phase4 proposal admits T1 only')
            phases = env.cfg.factorial_initial_phases
            if len(phases) != env.num_envs or any(int(p) not in range(4) for p in phases):
                raise ValueError('Initial sampler assignments must precede env creation')
            self.phase_offsets = torch.tensor(phases, dtype=torch.long, device=self.device)
        def bind_phases(self, env_ids, phases):
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            values = torch.as_tensor(phases, dtype=torch.long, device=self.device)
            if ids.numel() != values.numel() or not bool(((values >= 0) & (values < 4)).all()):
                raise ValueError('Invalid partial-reset phase assignment')
            self.phase_offsets[ids] = values
        def _update_command(self):
            p = self._suite['params']
            desired = torch.zeros_like(self.vel_command_b)
            age = self.env.episode_length_buf.to(torch.float32) * self.env.step_dt
            group = (self.phase_offsets + (age / p['base_segment_s']).long()) % 4
            desired[:, 0] = torch.where(group == 0, p['base_linear_speed'], torch.where(group == 1, -p['base_linear_speed'], 0.))
            desired[:, 2] = torch.where(group == 2, p['base_yaw_rate'], torch.where(group == 3, -p['base_yaw_rate'], 0.))
            slew = torch.tensor([p['base_slew_linear'], p['base_slew_linear'], p['base_slew_yaw']], device=self.device) * self.env.step_dt
            self.vel_command_b += (desired - self.vel_command_b).clamp(-slew, slew)
    return Phase4VelocityCommand
