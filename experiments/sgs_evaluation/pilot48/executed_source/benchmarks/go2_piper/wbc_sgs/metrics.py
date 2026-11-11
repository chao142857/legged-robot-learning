"""Episode metrics at 50Hz, separate from unchanged WBC rewards (torch only)."""
import torch


class JointMetrics:
    def __init__(self, count, device, protocol):
        self.protocol = protocol
        self.steps = torch.zeros(count, dtype=torch.long, device=device)
        self.samples = torch.zeros_like(self.steps)
        self.good = torch.zeros(count, 3, dtype=torch.long, device=device)
        self.sq = torch.zeros(count, 4, dtype=torch.float64, device=device)
        self.bad_safety = torch.zeros(count, dtype=torch.bool, device=device)
        self.peak_contact = torch.zeros(count, device=device)

    def reset(self, ids):
        for b in (self.steps, self.samples, self.good, self.sq, self.bad_safety, self.peak_contact):
            b[ids] = 0

    def update(self, errors, safety_ok, contact_force, dt):
        """errors: XY velocity norm, yaw error, mixed EE position norm, EE angle."""
        self.steps += 1
        finite = torch.isfinite(errors).all(-1)
        self.bad_safety |= (~safety_ok) | (~finite)
        self.peak_contact = torch.maximum(self.peak_contact, contact_force)
        # Startup is excluded from tracking ONLY. Safety scores the entire episode.
        valid = self.steps * dt > self.protocol['settling_s'] + 1.e-6
        e = torch.nan_to_num(errors.double(), nan=1.e6, posinf=1.e6, neginf=1.e6)
        self.samples += valid.long()
        self.sq += e.square() * valid[:, None]
        loc = (e[:, 0] <= self.protocol['xy_velocity_mps']) & (e[:, 1] <= self.protocol['yaw_rate_rps'])
        manip = (e[:, 2] <= self.protocol['ee_position_m']) & (e[:, 3] <= self.protocol['ee_orientation_rad'])
        # Track simultaneous success; separate marginal successes alone are insufficient.
        self.good += torch.stack([loc, manip, loc & manip], -1).long() * valid[:, None]

    def results(self, ids, full_horizon, terminated):
        records = []
        for i, complete, failure in zip(ids, full_horizon, terminated):
            n = int(self.samples[i])
            fractions = (self.good[i].double() / max(n, 1)).tolist()
            sufficient = n >= self.protocol['minimum_scored_samples']
            loc = sufficient and fractions[0] >= self.protocol['tracking_fraction']
            manip = sufficient and fractions[1] >= self.protocol['tracking_fraction']
            safety = not bool(self.bad_safety[i]) and not bool(failure)
            joint = bool(complete) and safety and loc and manip and fractions[2] >= self.protocol['tracking_fraction']
            records.append(dict(locomotion=bool(loc), manipulation=bool(manip), safety=safety,
                joint_success=bool(joint), full_horizon=bool(complete), terminated=bool(failure),
                scored_samples=n, control_steps=int(self.steps[i]),
                good_fractions=dict(locomotion=fractions[0], manipulation=fractions[1], simultaneous=fractions[2]),
                rmse=(self.sq[i] / max(n, 1)).sqrt().tolist(),
                rmse_units=['m/s','rad/s','m','rad'], peak_nonfoot_contact_N=float(self.peak_contact[i])))
        return records
