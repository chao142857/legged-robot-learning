"""Build one exact easy-task short-training admission from completed evidence.

This tool never starts training. physics_pass means numerical, geometric and
control feasibility for bounded local PPO; it does NOT mean strict safety or
joint task success. Every existing strict evaluation metric remains in output.
Random stepping-stone bank admission is deliberately unsupported until explicit
whole-bank coverage evidence is implemented and independently reviewed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from task_identity import file_sha256, task_identity, check_admission


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value, location):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite value at '+location)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, location+'/'+str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, location+'/'+str(index))


def read_json(path, finite=False):
    path = Path(path)
    require(path.is_file(), 'Missing evidence: '+str(path))
    result = json.loads(path.read_text())
    require(isinstance(result, dict), 'Expected JSON object: '+str(path))
    if finite:
        finite_tree(result, str(path))
    return result


def exact_numbers(actual, expected, name):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and
            np.allclose(a, b, atol=1e-12, rtol=0), 'IK/task parameter mismatch: '+name)


def bool_csv(value):
    require(value in ('True', 'False'), 'Invalid boolean in episodes.csv: '+str(value))
    return value == 'True'


def verify_bindings(identity):
    result = {}
    for group, path_group in (('sources', 'source_paths'), ('assets', 'asset_paths')):
        paths, hashes = identity.get(path_group), identity.get('bindings', {}).get(group)
        require(isinstance(paths, dict) and bool(paths), 'Missing '+path_group)
        require(isinstance(hashes, dict) and set(hashes) == set(paths), 'Binding file-set mismatch: '+group)
        result[group] = {key: Path(path) for key, path in paths.items()}
        for key, path in result[group].items():
            require(path.is_file(), 'Missing current dependency: '+str(path))
            require(file_sha256(path) == hashes[key], 'Stale source/asset binding: '+group+'/'+key)
    return result


def verify_ik(ik, task, bindings):
    p, actual = task['params'], ik.get('parameters', {})
    for ik_key, task_key in (('center', 'ee_center'), ('amplitude', 'ee_amplitude'), ('period', 'ee_period_s')):
        exact_numbers(actual.get(ik_key), p[task_key], ik_key)
    exact_numbers(p['ee_orientation_offset_rpy'][::2], [0., 0.], 'supported roll/yaw offsets')
    exact_numbers(actual.get('pitch_offset'), p['ee_orientation_offset_rpy'][1], 'pitch_offset')
    require(ik.get('all_sampled_poses_pass') is True, 'IK did not pass all sampled poses')
    rows = ik.get('rows', [])
    require(isinstance(rows, list) and len(rows) >= 270 and len(rows) == ik.get('total') == ik.get('exact_count'),
            'IK evidence must contain the complete >=270-row, all-passed audit')
    require(actual.get('samples', 0) >= 17, 'IK phase sampling is too sparse')
    require({row['mode'] for row in rows} == {'fixed', 'smooth'}, 'IK must cover fixed and smooth modes')
    current_usd = {path.name: file_sha256(path) for path in bindings['assets'].values() if path.suffix == '.usd'}
    ik_assets = ik.get('assets', {})
    require(bool(current_usd) and bool(ik_assets), 'IK/current USD asset bindings are missing')
    ik_by_name = {}
    for name, record in ik_assets.items():
        basename = Path(name).name
        require(basename not in ik_by_name, 'Ambiguous IK asset basename: '+basename)
        require(record.get('actual') == record.get('expected'), 'IK used a mismatched USD: '+basename)
        ik_by_name[basename] = record.get('actual')
    require(set(current_usd) == set(ik_by_name), 'IK and evaluated robot USD file sets differ')
    require(current_usd == ik_by_name, 'IK was not performed on the evaluated robot USD assets')
    body_cases = set()
    for row in rows:
        require(row.get('exact') is True and row['position_residual_m'] < .001 and
                row['orientation_residual_rad'] < .01, 'IK row failed exact-pose tolerances')
        phase = row['phase']
        offset = np.asarray(p['ee_amplitude'])*[math.sin(phase), math.cos(phase), -math.sin(phase)]
        expected = np.asarray(p['ee_center']) + (offset if row['mode'] == 'smooth' else 0.)
        exact_numbers(row['command_xyz'], expected, 'sample command_xyz')
        body_cases.add((row['base_height'], row['base_roll'], row['base_pitch']))
        # audit_reachability clamps Euler while SuitePoseCommand does not. Prove
        # the clamp is inactive for these exact accepted samples before reuse.
        roll, pitch = row['base_roll'], row['base_pitch']
        rbase_z = np.array([-math.sin(pitch), math.cos(pitch)*math.sin(roll),
                           math.cos(pitch)*math.cos(roll)])
        delta_world_z = float(rbase_z @ np.asarray(row['local_target']))
        pointing_pitch = -math.atan2(delta_world_z, np.linalg.norm(expected[:2]))+actual['pitch_offset']
        pointing_yaw = math.atan2(expected[1], expected[0])
        require(all(-3.14/4 <= angle <= 3.14/3 for angle in (0., pointing_pitch, pointing_yaw)),
                'IK Euler clamp changed the suite target; this evidence is not equivalent')
    require(len(body_cases) >= 15, 'IK base pose coverage is incomplete')
    return dict(total=ik['total'], exact_count=ik['exact_count'], parameters=actual,
                max_position_residual_m=ik['max_position_residual_m'],
                max_orientation_residual_rad=ik['max_orientation_residual_rad'],
                euler_clamp_inactive_for_all_samples=True)


def verify_geometry(geometry, task, bindings):
    require(task_identity(geometry['task'])[0] == task_identity(task)[0], 'Geometry belongs to a different task')
    require(geometry.get('geometry_only_checks_pass') is True, 'Geometry summary did not pass')
    checks = geometry.get('checks', {})
    require(bool(checks) and all(value in (True, False, None) for value in checks.values()), 'Invalid geometry checks')
    require(all(value is not False for value in checks.values()), 'Geometry/reset contains a failed check')
    required = ['all_sampled_mesh_rays_finite', 'reset_foot_body_centers_above_ground',
                'foot_collision_bounds_available', 'reset_collision_clearance_finite',
                'reset_collision_aabb_clearance_ge_minus_2mm', 'start_goal_rays_finite']
    if task['family'] == 'T2':
        required += ['height_scanner_finite', 'route_has_real_ascending_steps' if task['terrain'] == 'stairs_up'
                     else 'route_has_real_descending_steps']
    require(all(checks.get(name) is True for name in required), 'Missing/unchecked required geometry or reset check')
    audit_source = bindings['sources'].get('tools/audit_geometry_runtime.py')
    require(audit_source is not None and geometry.get('source_sha256') == file_sha256(audit_source),
            'Geometry audit source does not match candidate binding')
    require(geometry.get('unique_origins_tested', 0) >= 1 and len(geometry.get('layouts', [])) >= 1,
            'No actual terrain layout was audited')
    if task['family'] == 'T2':
        require(geometry['unique_origins_tested'] == geometry['all_unique_origins_count'],
                'Not all origins present in this evaluation were geometrically checked')
    return checks


def verify_physics(evaluation, summary, runtime, transfer):
    count, horizon = summary.get('num_episodes', 0), summary.get('max_steps', 0)
    require(isinstance(count, int) and count >= 1 and isinstance(horizon, int) and horizon >= 1,
            'Invalid evaluation episode count/horizon')
    require(summary.get('complete_episode_evaluation') is True and summary.get('requested_steps') == horizon,
            'Censored smoke cannot grant physics admission')
    require(runtime.get('num_envs') == count and runtime.get('max_episode_length') == horizon,
            'Runtime and evaluation episode budget differ')
    require(transfer.get('strict_load') is True and transfer.get('bitwise_equal') == {'actor': True, 'critic': True},
            'CP3000 full actor/critic transfer was not verified')
    require(transfer.get('optimizer_state_entries') == 0 and transfer.get('iteration') == 0,
            'Unexpected transfer optimizer or iteration state')
    require(transfer.get('observation_dims') == {'policy': [210], 'critic': [102]},
            'Only native 210D/102D inputs are admitted in this suite')
    require(summary.get('actor_unchanged') is True and summary.get('critic_unchanged') is True,
            'Evaluation was not frozen')
    with (evaluation/'episodes.csv').open() as stream:
        episodes = list(csv.DictReader(stream))
    require(len(episodes) == count and sorted(int(row['env_id']) for row in episodes) == list(range(count)),
            'Episode CSV rows do not match the environment population')
    episodes.sort(key=lambda row: int(row['env_id']))
    with np.load(evaluation/'trajectories.npz', allow_pickle=False) as trajectory:
        required = {'valid', 'age_steps', 'actions', 'joint_position', 'joint_velocity', 'base_pos_world',
                    'base_cmd', 'base_actual', 'ee_cmd', 'ee_position_error', 'base_clearance',
                    'joint_velocity_limit_ratio', 'joint_position_limit_excess'}
        require(required <= set(trajectory.files), 'Incomplete physical trajectory fields')
        for name in trajectory.files:
            array = trajectory[name]
            require(array.dtype.kind in 'biuf', 'Unsupported trajectory dtype: '+name)
            require(bool(np.isfinite(array).all()), 'Nonfinite physical trajectory: '+name)
        valid = trajectory['valid']
        require(valid.dtype.kind == 'b' and valid.ndim == 2 and valid.shape[1] == count and
                valid.shape[0] <= horizon, 'Invalid first-episode validity mask')
        require(bool(valid[0].all()) and not bool((valid[1:] & ~valid[:-1]).any()),
                'First-episode mask includes post-reset samples')
        steps = valid.sum(axis=0)
        failure = np.zeros(count, dtype=bool)
        for name in summary['terminations']:
            field = 'termination_'+name
            require(field in trajectory.files, 'Missing termination trajectory: '+name)
            flags = trajectory[field]
            require(flags.shape == valid.shape, 'Termination mask shape mismatch: '+name)
            require(int((flags & valid).any(axis=0).sum()) == summary['terminations'][name],
                    'Termination count mismatch: '+name)
            if not runtime['terminations'][name].get('time_out', False):
                failure |= (flags & valid).any(axis=0)
        complete = (steps == horizon) & ~failure
        for index, row in enumerate(episodes):
            require(int(row['steps']) == int(steps[index]) and bool_csv(row['complete']) == bool(complete[index]) and
                    bool_csv(row['failure']) == bool(failure[index]), 'Episode/trajectory completion mismatch')
            require(math.isclose(float(row['duration_s']), steps[index]*runtime['step_dt'], abs_tol=1e-8),
                    'Episode duration differs from the physical step count')
        for field, value in summary.items():
            if field.endswith('_count') and field[:-6] in episodes[0]:
                require(sum(bool_csv(row[field[:-6]]) for row in episodes) == value, 'Safety/result count mismatch: '+field)
        survivors = int(complete.sum())
        require(survivors >= math.ceil(count/2), 'Fewer than half the episodes survived the complete physical horizon')
    return dict(num_episodes=count, complete_count=survivors, minimum_complete_required=math.ceil(count/2),
                full_horizon_steps=horizon, raw_trajectory_finite=True, pre_reset_mask_verified=True,
                strict_policy_safety_success_required=False)


def build(evaluation, ik_path):
    evaluation, ik_path = Path(evaluation).resolve(), Path(ik_path).resolve()
    require(not (evaluation/'error.json').exists(), 'Evaluation has an error.json; cannot admit')
    identity = read_json(evaluation/'candidate_identity.json', finite=True)
    bindings = verify_bindings(identity)
    task = identity['task']
    key, spec = task_identity(task)
    require(key == identity.get('admission_key') and spec == task, 'Invalid candidate identity')
    require(task.get('difficulty') == 'easy' and task.get('family') in ('T0', 'T1', 'T2'),
            'Only an exact easy task can be admitted')
    require(task.get('terrain') != 'stepping_stones', 'Stepping-stone bank coverage is not verified; admission is disabled')
    require(task.get('terrain') in ('plane', 'stairs_up', 'stairs_down'), 'Unsupported exact terrain')
    profiles = read_json(bindings['sources']['task_suite/profiles.json'], finite=True)
    require(task.get('params') == profiles['easy'], 'This is not the exact current easy profile; ranges/other overrides are quarantined')
    geometry = read_json(evaluation/'geometry.json', finite=True)
    summary = read_json(evaluation/'summary.json', finite=True)
    runtime = read_json(evaluation/'initial_runtime.json')
    transfer = read_json(evaluation/'transfer.json', finite=True)
    protocol = read_json(evaluation/'protocol.json', finite=True)
    read_json(evaluation/'reset_state.json', finite=True)
    ik = read_json(ik_path, finite=True)
    for record in (summary, protocol):
        require(record.get('admission_key') == key, 'Evidence task key differs from candidate')
        require(record.get('mode') in ('eval', 'physics'), 'Training output cannot serve as frozen physical admission')
    require(task_identity(protocol['task'])[0] == key and summary.get('family') == task['family'],
            'Protocol/summary task mismatch')
    geometry_checks = verify_geometry(geometry, task, bindings)
    ik_checks = verify_ik(ik, task, bindings)
    evidence_names = ['candidate_identity.json', 'geometry.json', 'summary.json', 'initial_runtime.json',
                      'transfer.json', 'protocol.json', 'reset_state.json', 'episodes.csv', 'trajectories.npz',
                      'geometry_samples.npz', 'final_runtime.json', 'env.yaml', 'agent.yaml']
    manifest = read_json(evaluation/'manifest.json')
    evidence = []
    for name in evidence_names:
        path = evaluation/name
        require(path.is_file() and name in manifest, 'Incomplete evaluation evidence/manifest: '+name)
        digest = file_sha256(path)
        require(manifest[name].get('sha256') == digest and manifest[name].get('bytes') == path.stat().st_size,
                'Evaluation manifest mismatch: '+name)
        evidence.append(dict(path=str(path), sha256=digest, bytes=path.stat().st_size))
    geometry_samples = geometry['samples']
    require(Path(geometry_samples['path']).resolve() == (evaluation/'geometry_samples.npz') and
            geometry_samples['sha256'] == file_sha256(evaluation/'geometry_samples.npz'), 'Geometry sample binding mismatch')
    physics_checks = verify_physics(evaluation, summary, runtime, transfer)
    for path in (evaluation/'manifest.json', ik_path, Path(__file__).resolve()):
        evidence.append(dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size))
    # Re-read dependency hashes after all evidence checks to reject concurrent changes.
    verify_bindings(identity)
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), admission_key=key,
                task=task, exact_profile_only=True, interval_or_bank_admission=False,
                geometry_pass=True, kinematics_pass=True, physics_pass=True,
                physics_pass_definition='Finite complete physical evidence, valid geometry/reset, matched IK, and at least half full-horizon survival; permits bounded local PPO only. Not strict safety or joint task success.',
                strict_safety_pass=summary.get('safe_count') == summary.get('num_episodes'),
                learning_demonstrated=False, long_training_authorized=False, sgs_training_authorized=False,
                geometry_checks=geometry_checks, kinematics_checks=ik_checks, physics_checks=physics_checks,
                strict_evaluation_results=summary, bindings=identity['bindings'], evidence=evidence,
                builder_sha256=file_sha256(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--ik', type=Path, required=True, help='Complete reachability.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refusing to overwrite an existing admission')
    record = build(args.evaluation, args.ik)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    identity = read_json(args.evaluation/'candidate_identity.json')
    check_admission(args.out, record['admission_key'], source_files=identity['source_paths'], asset_files=identity['asset_paths'])
    print(json.dumps(dict(out=str(args.out), sha256=file_sha256(args.out), admission_key=record['admission_key'],
                          complete_count=record['physics_checks']['complete_count'],
                          strict_safety_pass=record['strict_safety_pass'], training_started=False)))


if __name__ == '__main__':
    main()
