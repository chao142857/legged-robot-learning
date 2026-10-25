"""Bounded numerical IK on the SHA-verified native USD joint chain.

Failure means unresolved feasibility, never proof of impossibility. This checks
sampled poses, not continuous collision-free motion; physics admission is separate.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Arm:
    def __init__(self, data):
        joints = {j['name']: j for j in data['joints']}
        self.chain = [joints[f'joint{i}'] for i in range(1, 7)] + [joints['ee_gripper']]
        self.lo = np.array([j['lower'] for j in self.chain[:6]]) + 1e-4
        self.hi = np.array([j['upper'] for j in self.chain[:6]]) - 1e-4
        self.t0 = np.array([j['T0'] for j in self.chain])
        self.t1 = np.linalg.inv(np.array([j['T1'] for j in self.chain]))
        mount = joints['connector_to_arm_joint']
        self.mount = np.array(mount['T0']) @ np.linalg.inv(np.array(mount['T1']))

    def fk(self, q):
        t = np.eye(4)
        for i in range(7):
            r = np.eye(4)
            if i < 6:
                axis = np.eye(3)['XYZ'.index(self.chain[i]['axis'])]
                r[:3, :3] = Rotation.from_rotvec(axis*q[i]).as_matrix()
            t = t @ self.t0[i] @ r @ self.t1[i]
        return t

    def solve(self, xyz, rot, previous):
        def residual(q):
            t = self.fk(q)
            return np.r_[(t[:3, 3]-xyz)*5, Rotation.from_matrix(rot.T@t[:3, :3]).as_rotvec()]
        best = None
        starts = [previous, (self.lo+self.hi)/2, [0., 1.5, -1.5, 0., .3, 0.]]
        for start in starts:
            result = least_squares(residual, np.clip(start, self.lo, self.hi),
                                   bounds=(self.lo, self.hi), max_nfev=100,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            t = self.fk(result.x)
            pe = float(np.linalg.norm(t[:3, 3]-xyz))
            re = float(np.linalg.norm(Rotation.from_matrix(rot.T@t[:3, :3]).as_rotvec()))
            row = dict(position_residual_m=pe, orientation_residual_rad=re,
                       q=result.x.tolist(), exact=pe < .001 and re < .01)
            if best is None or pe*5+re < best['position_residual_m']*5+best['orientation_residual_rad']:
                best = row
            if row['exact']:
                break
        return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kinematics', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--center', type=float, nargs=3, default=[.45, 0., .5])
    parser.add_argument('--amplitude', type=float, nargs=3, default=[.02, .03, .02])
    parser.add_argument('--period', type=float, default=8.)
    parser.add_argument('--pitch_offset', type=float, default=0.)
    parser.add_argument('--samples', type=int, default=17)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads(Path(args.kinematics).read_text())
    assets = {p: dict(expected=s, actual=sha(Path(args.repo)/p)) for p, s in data['layers'].items()}
    assert all(v['expected'] == v['actual'] for v in assets.values()), 'USD geometry mismatch'
    arm = Arm(data)
    rows = []
    for height, (roll, pitch) in itertools.product([.24, .28, .35], [(0., 0.), (.12, 0.), (-.12, 0.), (0., .12), (0., -.12)]):
        rbase = Rotation.from_euler('xyz', [roll, pitch, 0.]).as_matrix()
        mountz = height + (rbase @ arm.mount[:3, 3])[2]
        previous = (arm.lo+arm.hi)/2
        for mode in ['fixed', 'smooth']:
            for phase in np.linspace(0, 2*np.pi, args.samples if mode=='smooth' else 1):
                # Cartesian bounds are tested densely independent of actual phase
                # schedule; exact runtime trajectories are checked separately.
                cmd = np.array(args.center) + np.array(args.amplitude)*np.array([np.sin(phase), np.sin(phase+.5*np.pi), np.sin(phase+np.pi)])*(mode=='smooth')
                local = np.array([cmd[0], cmd[1], (cmd[2]-mountz-rbase[2,0]*cmd[0]-rbase[2,1]*cmd[1])/rbase[2,2]])
                # Exact suite pointing convention; the native mixed XY/Z
                # target semantics remain unchanged.
                epitch = -np.arctan2(cmd[2]-mountz, np.linalg.norm(cmd[:2])) + args.pitch_offset
                eyaw = np.arctan2(cmd[1],cmd[0])
                rot = Rotation.from_euler('xyz', np.clip([0.,epitch,eyaw],-3.14/4,3.14/3)).as_matrix()
                solved = arm.solve(local, rot, previous)
                previous = np.array(solved['q'])
                rows.append(dict(mode=mode, base_height=height, base_roll=roll, base_pitch=pitch,
                                 phase=float(phase), command_xyz=cmd.tolist(), local_target=local.tolist(), **solved))
    result = dict(scope='Sampled pose feasibility with joint limits; no self-collision proof; native self-collision disabled.',
                  kinematics_sha256=sha(args.kinematics), assets=assets,
                  parameters=dict(center=args.center, amplitude=args.amplitude, period=args.period, pitch_offset=args.pitch_offset, samples=args.samples),
                  total=len(rows), exact_count=sum(r['exact'] for r in rows),
                  all_sampled_poses_pass=all(r['exact'] for r in rows),
                  max_position_residual_m=max(r['position_residual_m'] for r in rows),
                  max_orientation_residual_rad=max(r['orientation_residual_rad'] for r in rows), rows=rows)
    (out/'reachability.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','assets']}))


if __name__ == '__main__':
    main()
