#!/usr/bin/env python3
"""Generate crossed, shuffled fixed tasks; no ranking or policy-derived difficulty labels."""
import argparse
import json
import math
from pathlib import Path
import random


def quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def generate(seed, count=48):
    rng = random.Random(seed)
    speeds = [[0.,0.,0.],[.18,0.,0.],[-.18,.08,0.],[.32,-.12,.15],
              [.5,.15,-.3],[-.35,-.22,.35],[.65,0.,.45],[.15,.3,-.5]]
    poses = [[.44,0.,.5],[.5,.12,.55],[.58,-.18,.4],
             [.62,.22,.6],[.45,-.25,.32],[.64,-.05,.65]]
    tasks = []
    # All 8x6 crossed commands, then permuted; IDs have no difficulty ordering.
    pairs = [(v,p) for v in range(8) for p in range(6)]
    rng.shuffle(pairs)
    for i,(vi,pi) in enumerate(pairs[:count]):
        x,y,z = poses[pi]
        # Source WBC convention: pointing quaternion relative to base; z is world.
        roll = rng.uniform(-.05,.05)
        pitch = -math.atan2(z-.45, math.hypot(x,y))
        yaw = math.atan2(y,x) + rng.uniform(-.12,.12)
        arm = [[0.,0.,0.,0.,0.,0.],[0.,.3,-.5,0.,.2,0.],
               [0.,.7,-.9,0.,.2,0.]][i%3]
        tasks.append(dict(id=i, velocity=speeds[vi], ee_pose=[x,y,z]+quaternion(roll,pitch,yaw),
            initial=dict(root_pos=[rng.uniform(-.08,.08),rng.uniform(-.08,.08),.35],
                root_rpy=[rng.uniform(-.025,.025),rng.uniform(-.025,.025),rng.uniform(-.4,.4)],
                root_velocity=[0.]*6, leg_scale=rng.uniform(.97,1.03), arm=arm),
            factors=dict(velocity_family=vi, pose_family=pi, initial_arm_family=i%3)))
    return dict(version=1, generation_seed=seed, tasks=tasks,
        semantics=dict(velocity='body-frame vx/vy/yaw_rate', ee_xy='link0 frame',
                       ee_z='world z minus environment origin z', ee_quaternion='base-relative wxyz, matching reward',
                       initial='exact physical reset; no unrecorded reset/domain randomization'),
        ordering='seeded shuffle of crossed factors; no manually ranked difficulty')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261019)
    a=p.parse_args();a.output.write_text(json.dumps(generate(a.seed),indent=2)+'\n')
