"""Bounded representative-point IK audit for unadmitted middle/high candidates.

This is a CPU-only audit of the existing SHA-verified native USD joint chain.
It never changes a task source, launches physics/PPO, or grants training admission.
An unsuccessful solve means unresolved numerical feasibility, not impossibility.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from audit_reachability import Arm, sha


def candidates(profiles):
    """Conservative exact points, preserving unsupported route ranges honestly."""
    result = []
    for name, level in [('medium', 'intermediate'), ('hard', 'advanced')]:
        p = copy.deepcopy(profiles['easy'])
        ee = profiles['candidate_ranges']['T0'][level]
        base = profiles['candidate_ranges']['T1'][level]
        terrain = profiles['candidate_ranges']['T2'][level]
        # Center is a common interior reference, clamped to the declared range
        # only at construction; the selected values are subsequently explicit.
        p.update(
            ee_center=[float(np.clip(.45, *ee['ee_center_x_m'])), 0., float(np.clip(.5, *ee['ee_center_z_m']))],
            ee_amplitude=[ee['ee_amplitude_m'][0]]*3,
            ee_period_s=ee['ee_period_s'][1],
            # The .6 pitch is nominal. Added roll is the smallest declared
            # orientation deviation, so this point is inside that level too.
            ee_orientation_offset_rpy=[ee['orientation_offset_rad'][0], .6, 0.],
            base_linear_speed=base['linear_speed_mps'][0],
            base_yaw_rate=base['yaw_rate_radps'][0],
            base_segment_s=base['segment_s'][1],
            terrain_speed=terrain['linear_speed_mps'][0],
            stairs_height=terrain['step_height_m'][0],
            stairs_tread=terrain['tread_m'][1],
            stones_width=terrain['stone_width_m'][1],
            stones_gap=terrain['gap_m'][0],
            stones_height=terrain['height_variation_m'][0],
            route_distance=terrain['route_distance_m'][0],
        )
        result.append((name, level, p))
    return result


def audit_pose_samples(arm, p):
    rows = []
    base_attitudes = [(0.,0.), (.12,0.), (-.12,0.), (0.,.12), (0.,-.12)]
    for height, (roll, pitch) in itertools.product([.24,.28,.35], base_attitudes):
        rbase = Rotation.from_euler('xyz', [roll,pitch,0.]).as_matrix()
        mount_z = height + (rbase @ arm.mount[:3,3])[2]
        previous = (arm.lo+arm.hi)/2
        for mode in ['fixed','smooth']:
            phases = np.linspace(0.,2*np.pi,17) if mode=='smooth' else [0.]
            for phase in phases:
                offsets = np.array([np.sin(phase), np.cos(phase), -np.sin(phase)])*np.array(p['ee_amplitude']) if mode=='smooth' else np.zeros(3)
                command = np.array(p['ee_center'])+offsets
                local = np.array([command[0],command[1],
                    (command[2]-mount_z-rbase[2,0]*command[0]-rbase[2,1]*command[1])/rbase[2,2]])
                extra = p['ee_orientation_offset_rpy']
                desired_rpy = [extra[0], -np.arctan2(command[2]-mount_z, np.linalg.norm(command[:2]))+extra[1],
                               np.arctan2(command[1],command[0])+extra[2]]
                # Runtime SuitePoseCommand does not clamp Euler angles. Audit
                # the exact target; never clip a failed candidate into success.
                desired_rotation = Rotation.from_euler('xyz',desired_rpy).as_matrix()
                solved = arm.solve(local,desired_rotation,previous)
                previous=np.asarray(solved['q'])
                rows.append(dict(mode=mode,base_height=height,base_roll=roll,base_pitch=pitch,
                    phase=float(phase),time_in_cycle_s=float(phase*p['ee_period_s']/(2*np.pi)),
                    command_xyz=command.tolist(),local_target=local.tolist(),target_rpy=desired_rpy,**solved))
    assert len(rows)==270
    return rows


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False); f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kinematics',required=True)
    parser.add_argument('--repo',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--profiles',default=str(Path(__file__).resolve().parents[1]/'task_suite/profiles.json'))
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():
        parser.error('Refusing to overwrite an existing output directory')
    data=json.loads(Path(args.kinematics).read_text())
    profiles=json.loads(Path(args.profiles).read_text())
    assets={p:dict(expected=s,actual=sha(Path(args.repo)/p)) for p,s in data['layers'].items()}
    if not all(v['expected']==v['actual'] for v in assets.values()):
        raise RuntimeError('USD geometry SHA mismatch; refusing candidate IK audit')
    out.mkdir(parents=True,exist_ok=False)
    arm=Arm(data)
    manifest=dict(schema_version=1,scope='Two conservative representative points only; no interval coverage claim',
        assets=assets,kinematics_sha256=sha(args.kinematics),profiles_sha256=sha(args.profiles),
        source_sha256=sha(__file__),arm_solver_sha256=sha(Path(__file__).with_name('audit_reachability.py')),
        numerical_tolerances=dict(position_residual_m=.001,orientation_residual_rad=.01),
        physics_executed=False,training_admission=False,profiles=[])
    write_json(out/'external_parameter_ranges.json',dict(
        candidate_ranges=profiles['candidate_ranges'],orientation_range_semantics=profiles.get('orientation_range_semantics'),
        tested_points='Only the explicit medium/hard profile_overrides.json files',
        untested='All other interior points, range endpoints/corners not identical to these representatives, continuous collision-free trajectories, dynamic velocities, full-body support and locomotion',
        hard_route_lower_bound_m=profiles['candidate_ranges']['T2']['advanced']['route_distance_m'][0],
        current_tile_route_limit_m=3.5,unsupported_route_handling='Reject T2 route>3.5m; do not clamp or enlarge terrain automatically'))
    for name,level,p in candidates(profiles):
        folder=out/name; folder.mkdir()
        write_json(folder/'profile_overrides.json',p)
        rows=audit_pose_samples(arm,p)
        passing=all(row['exact'] for row in rows)
        route_ok=bool(0<p['route_distance']<=3.5)
        route_ideal_s=p['route_distance']/p['terrain_speed']
        reasons=[]
        if not passing:
            reasons.append('One or more sampled poses failed numerical IK tolerances; reject physical smoke until resolved, without claiming mathematical impossibility')
        if not route_ok:
            reasons.append('T2 route exceeds current 3.5m limit for 8m terrain; reject T2 physics even if arm IK passes')
        state=dict(name=name,declared_level=level,parameters=p,total=len(rows),
            exact_count=sum(row['exact'] for row in rows),all_sampled_poses_pass=passing,
            maximum_position_residual_m=max(row['position_residual_m'] for row in rows),
            maximum_orientation_residual_rad=max(row['orientation_residual_rad'] for row in rows),
            failed_samples=[dict(index=i,mode=row['mode'],base_height=row['base_height'],base_roll=row['base_roll'],base_pitch=row['base_pitch'],
                phase=row['phase'],position_residual_m=row['position_residual_m'],orientation_residual_rad=row['orientation_residual_rad'])
                for i,row in enumerate(rows) if not row['exact']],
            geometry_precheck=dict(route_supported=route_ok,current_tile_route_limit_m=3.5,
                requested_route_distance_m=p['route_distance'],nominal_stone_area_m2=p['stones_width']**2,
                measured_mesh_geometry=False),
            timing_precheck=dict(ee_period_s=p['ee_period_s'],flat_episode_s=p['episode_s_flat'],
                ee_cycles_per_flat_episode=p['episode_s_flat']/p['ee_period_s'],
                conservative_ee_speed_upper_bound_mps=float(2*np.pi/p['ee_period_s']*np.linalg.norm(p['ee_amplitude'])),
                ideal_route_time_s=route_ideal_s,terrain_episode_s=p['episode_s_terrain'],
                caveat='Static timing only; period does not establish velocity/acceleration feasibility'),
            status='sampled_ik_pass_requires_geometry_and_physics' if passing else 'rejected_sampled_ik',
            eligible_for_separately_authorized_physics_smoke=dict(T0=passing,T1=passing,T2=passing and route_ok),
            rejection_reasons=reasons,physics_executed=False,training_admission=False,
            range_admission=False,range_endpoints_and_corners_covered=False,
            coverage_note='270 samples = 15 base poses times (1 fixed + 17 smooth phases including repeated 0/2pi endpoint); no self-collision or continuous-path proof',
            orientation_note='Nominal pointing pitch offset .6 rad plus declared level minimum roll deviation; no target clipping',
            profile_overrides=dict(path=str(folder/'profile_overrides.json'),sha256=sha(folder/'profile_overrides.json')))
        write_json(folder/'reachability.json',dict(**state,rows=rows))
        write_json(folder/'summary.json',state)
        manifest['profiles'].append(dict(**state,reachability_path=str(folder/'reachability.json'),
            reachability_sha256=sha(folder/'reachability.json')))
        print(json.dumps({k:state[k] for k in ['name','total','exact_count','status','eligible_for_separately_authorized_physics_smoke','rejection_reasons']}),flush=True)
    write_json(out/'manifest.json',manifest)
    print('CANDIDATE_LEVEL_AUDIT_COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
