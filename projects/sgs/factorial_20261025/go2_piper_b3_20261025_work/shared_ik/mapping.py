"""Name-bound mapping between policy12, arm q_ref6 and native action18."""
import math
import torch
from .settings import LEG_NAMES, ARM_NAMES


class NativeActionMap:
    def __init__(self, articulation_joint_names, action_joint_names, scale, offset,
                 processed_clip=None, device='cpu', dtype=torch.float32):
        self.articulation_names = list(articulation_joint_names)
        self.action_names = list(action_joint_names)
        expected = LEG_NAMES + ARM_NAMES
        if len(self.action_names) != 18 or len(set(self.action_names)) != 18 or set(self.action_names) != set(expected):
            raise ValueError('Native term must control exactly the 12 leg and 6 arm joints')
        if len(set(self.articulation_names)) != len(self.articulation_names) or not set(expected) <= set(self.articulation_names):
            raise ValueError('Invalid articulation joint name binding')
        self.joint_ids = [self.articulation_names.index(n) for n in self.action_names]
        self.leg_slots = [self.action_names.index(n) for n in LEG_NAMES]
        self.arm_slots = [self.action_names.index(n) for n in ARM_NAMES]
        self.scale = torch.as_tensor(scale, device=device, dtype=dtype).clone()
        self.offset = torch.as_tensor(offset, device=device, dtype=dtype).clone()
        if not bool(torch.isfinite(self.scale).all()) or bool((self.scale == 0).any()) or not bool(torch.isfinite(self.offset).all()):
            raise ValueError('Native affine map must be finite with nonzero scale')
        self.clip = None if processed_clip is None else torch.as_tensor(processed_clip, device=device, dtype=dtype).clone()
        if self.clip is not None and (bool(torch.isnan(self.clip).any()) or bool((self.clip[..., 0] > self.clip[..., 1]).any())):
            raise ValueError('Invalid native processed-action clip')

    def _affine(self, n):
        return (torch.broadcast_to(self.scale, (n, 18)), torch.broadcast_to(self.offset, (n, 18)))

    def arm_bounds(self, n):
        if self.clip is None:
            return None, None
        clip = torch.broadcast_to(self.clip, (n, 18, 2))[:, self.arm_slots]
        return clip[..., 0], clip[..., 1]

    def compose(self, raw_leg12, q_ref6):
        n = raw_leg12.shape[0]
        if raw_leg12.shape != (n, 12) or q_ref6.shape != (n, 6):
            raise ValueError('Expected policy [N,12] and arm reference [N,6]')
        if not bool(torch.isfinite(raw_leg12).all() & torch.isfinite(q_ref6).all()):
            raise ValueError('Nonfinite action composition input')
        scale, offset = self._affine(n)
        action = torch.empty((n, 18), device=raw_leg12.device, dtype=raw_leg12.dtype)
        action[:, self.leg_slots] = raw_leg12
        action[:, self.arm_slots] = (q_ref6-offset[:, self.arm_slots])/scale[:, self.arm_slots]
        processed = action*scale+offset
        if self.clip is not None:
            clip = torch.broadcast_to(self.clip, (n, 18, 2))
            processed = processed.clamp(clip[..., 0], clip[..., 1])
        if not torch.allclose(processed[:, self.arm_slots], q_ref6, atol=2e-6, rtol=1e-6):
            raise ValueError('Arm q_ref is not representable by the native action term')
        return action, processed

    def articulation_targets(self, processed, initial=None):
        """Diagnostic scatter only. Execution must use native manager/actuators."""
        n = len(processed)
        result = torch.full((n, len(self.articulation_names)), torch.nan, device=processed.device, dtype=processed.dtype) if initial is None else initial.clone()
        result[:, self.joint_ids] = processed
        return result


class PolicyObservationDecoder:
    """Decode newest samples from per-term flattened native actor histories."""
    def __init__(self, term_names, term_shapes, joint_observation_names, default_joint_pos):
        self.slices = {}; offset = 0
        for name, shape in zip(term_names, term_shapes):
            width = math.prod(shape)
            self.slices[name] = (offset, offset+width)
            offset += width
        self.width = offset
        self.joint_names = list(joint_observation_names)
        self.default = default_joint_pos.clone()
        if self.default.ndim != 2 or self.default.shape[1] != len(self.joint_names):
            raise ValueError('Observation default joint-position binding mismatch')
        if set(self.joint_names) != set(LEG_NAMES+ARM_NAMES) or len(self.joint_names) != 18:
            raise ValueError('Expected all 18 observed joints exactly once')
        self.leg_ids = [self.joint_names.index(n) for n in LEG_NAMES]
        self.arm_ids = [self.joint_names.index(n) for n in ARM_NAMES]
        for name, width in [('joint_pos', 18), ('projected_gravity', 3), ('pos_commands', 7)]:
            a, b = self.slices[name]
            if (b-a) % width:
                raise ValueError('Unexpected observation history layout: '+name)

    def decode(self, policy_observation):
        if policy_observation.ndim != 2 or policy_observation.shape[1] != self.width:
            raise ValueError('Actor observation shape changed')
        n = len(policy_observation)
        def newest(name, width):
            a, b = self.slices[name]
            return policy_observation[:, a:b].reshape(n, -1, width)[:, -1]
        q = newest('joint_pos', 18) + self.default
        return dict(q_leg=q[:, self.leg_ids], q_arm=q[:, self.arm_ids],
                    gravity_b=newest('projected_gravity', 3), ee_mixed=newest('pos_commands', 7))
