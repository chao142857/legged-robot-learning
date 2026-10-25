"""Stateful arm reference generator; no simulator state is accepted here.

Feedback must be decoded from the exact noisy actor input. The model adds a
fixed USD kinematic prior and a lowest-foot-on-flat-ground height assumption.
The only optimization variables are joint1..joint6. No base/leg commands exist.
"""
import copy
import hashlib
import math
import torch
from .model import KinematicModel, KINEMATICS
from .settings import IK, VERSION


class ArmOnlyIK(KinematicModel):
    def __init__(self, num_envs, device='cpu', dtype=torch.float32,
                 settings=None, slew_mode='previous_reference'):
        cfg = copy.deepcopy(IK if settings is None else settings)
        if slew_mode not in ('previous_reference', 'measured_q_legacy'):
            raise ValueError('Unknown reference slew mode')
        for name in ('damping', 'orientation_length_m', 'max_iteration_step_rad',
                     'tracking_speed_rad_s', 'position_tolerance_m',
                     'orientation_tolerance_rad', 'max_abs_position_m'):
            if not math.isfinite(cfg[name]) or cfg[name] <= 0:
                raise ValueError('Invalid controller setting: '+name)
        if not isinstance(cfg['iterations'], int) or cfg['iterations'] <= 0:
            raise ValueError('iterations must be a positive integer')
        super().__init__(device, dtype, cfg)
        self.num_envs, self.slew_mode = num_envs, slew_mode
        self.model_sha = hashlib.sha256(KINEMATICS.read_bytes()).hexdigest()
        self.q_ref = torch.zeros((num_envs, 6), device=device, dtype=dtype).clamp(self.lower, self.upper)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids=None, q_arm=None):
        """Reset only selected rows; optional feedback is selected-row [K,6].

        Without feedback, initialize lazily from the next actor input so reset
        neither reads simulator truth nor recomputes a noisy observation.
        """
        ids = torch.arange(self.num_envs, device=self.q_ref.device) if env_ids is None else torch.as_tensor(env_ids, device=self.q_ref.device, dtype=torch.long)
        if ids.ndim != 1 or ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError('Invalid reset environment IDs')
        if q_arm is None:
            self.q_ref[ids] = torch.zeros_like(self.q_ref[ids]).clamp(self.lower, self.upper)
            self.initialized[ids] = False
        else:
            if q_arm.shape != (len(ids), 6) or not bool(torch.isfinite(q_arm).all()):
                raise ValueError('Reset feedback must be finite selected-row [K,6]')
            self.q_ref[ids] = q_arm.clamp(self.lower, self.upper)
            self.initialized[ids] = True

    def state_dict(self):
        return dict(version=VERSION, model_sha256=self.model_sha,
                    settings=copy.deepcopy(self.settings), slew_mode=self.slew_mode,
                    q_ref=self.q_ref.clone(), initialized=self.initialized.clone())

    def load_state_dict(self, state):
        for key, expected in [('version', VERSION), ('model_sha256', self.model_sha),
                              ('settings', self.settings), ('slew_mode', self.slew_mode)]:
            if state[key] != expected:
                raise ValueError('Controller state binding mismatch: '+key)
        q, valid = state['q_ref'], state['initialized']
        if q.shape != self.q_ref.shape or valid.shape != self.initialized.shape or valid.dtype != torch.bool or not bool(torch.isfinite(q).all()):
            raise ValueError('Invalid controller state tensors')
        self.q_ref.copy_(q.to(self.q_ref)); self.initialized.copy_(valid.to(self.initialized.device))

    @torch.no_grad()
    def command(self, q_leg, q_arm, gravity_b, ee_mixed, dt, lower=None, upper=None, velocity_limits=None):
        if not isinstance(dt, (float, int)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Control dt must be finite and positive')
        n = self.num_envs
        for x, width in [(q_leg, 12), (q_arm, 6), (gravity_b, 3), (ee_mixed, 7)]:
            if x.shape != (n, width):
                raise ValueError('Controller input shape mismatch')
        lo = self.lower.expand(n, -1).clone()
        hi = self.upper.expand(n, -1).clone()
        if lower is not None: lo = torch.maximum(lo, torch.broadcast_to(lower, lo.shape))
        if upper is not None: hi = torch.minimum(hi, torch.broadcast_to(upper, hi.shape))
        if not bool((torch.isfinite(lo) & torch.isfinite(hi) & (lo <= hi)).all()):
            raise ValueError('Nonfinite or empty joint limit intersection')
        speed = torch.full_like(q_arm, self.settings['tracking_speed_rad_s'])
        if velocity_limits is not None:
            velocity_limits = torch.broadcast_to(velocity_limits, speed.shape)
            if not bool((torch.isfinite(velocity_limits) & (velocity_limits > 0)).all()):
                raise ValueError('Invalid velocity limits')
            speed = torch.minimum(speed, velocity_limits)
        finite_q = torch.isfinite(q_arm).all(-1)
        finite = finite_q & torch.isfinite(q_leg).all(-1) & torch.isfinite(gravity_b).all(-1) & torch.isfinite(ee_mixed).all(-1)
        valid = finite & (gravity_b.norm(dim=-1) > 1e-6) & (ee_mixed[:, 3:].norm(dim=-1) > 1e-6)
        valid &= (ee_mixed[:, :3].abs() <= self.settings['max_abs_position_m']).all(-1)
        # Invalid rows never enter FK/linear solve with NaN/Inf. Other rows are
        # evaluated identically; the caller decides whether invalid input stops
        # the run. Holding a finite reference is not reported as IK success.
        q_safe = torch.where(torch.isfinite(q_arm), q_arm, self.q_ref).clamp(lo, hi)
        leg_safe = torch.where(torch.isfinite(q_leg), q_leg, torch.zeros_like(q_leg))
        gravity_safe = torch.where(valid[:, None], gravity_b, self.tensor([0., 0., -1.]))
        command_safe = torch.where(valid[:, None], ee_mixed, self.tensor([.45, 0., .5, 1., 0., 0., 0.]))
        z, row, base_z = self.estimate_link0_height(leg_safe, gravity_safe)
        position, rotation, frame_valid = self.mixed_to_local(command_safe, z, row)
        valid &= frame_valid
        q_ik, diag = self.solve(q_safe, position, rotation, lo, hi)
        valid &= torch.isfinite(q_ik).all(-1) & torch.isfinite(diag['position_residual_m']) & torch.isfinite(diag['orientation_residual_rad'])
        seed = (~self.initialized) & finite_q
        seed_recovery = seed & ((q_arm < lo) | (q_arm > hi)).any(-1)
        self.q_ref[seed] = q_safe[seed]
        self.initialized[seed] = True
        old_ref = self.q_ref.clone()
        ref_recovery = ((old_ref < lo) | (old_ref > hi)).any(-1)
        origin = q_safe if self.slew_mode == 'measured_q_legacy' else old_ref.clamp(lo, hi)
        proposed = origin + (q_ik - origin).clamp(-speed*dt, speed*dt)
        # Invalid mixed command/gravity holds the previous trajectory reference.
        proposed = torch.where(valid[:, None], proposed, old_ref)
        self.q_ref.copy_(proposed.clamp(lo, hi))
        diag['converged'] &= valid
        diag.update(q_IK=q_ik.clone(), q_ref=self.q_ref.clone(), previous_q_ref=old_ref,
                    input_valid=valid, frame_valid=frame_valid & finite,
                    reference_limit_recovery=ref_recovery,
                    initialization_limit_recovery=seed_recovery,
                    measured_limit_violation=finite_q & ((q_arm < lo) | (q_arm > hi)).any(-1),
                    initialized=self.initialized.clone(), estimated_base_z=base_z,
                    estimated_link0_z=z, q_ref_step=(self.q_ref-old_ref),
                    slew_budget=speed*dt)
        return self.q_ref.clone(), diag
