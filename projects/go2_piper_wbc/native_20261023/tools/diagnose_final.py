"""CPU-only diagnostics from saved CP3000 trajectories; no simulation or policy calls."""
from publication_paths import research_path
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(research_path('go2_piper_upstream_wbc_20261023'))
OUT = ROOT / 'analysis'
OUT.mkdir(exist_ok=True)


def elapsed_since_change(command):
    change = np.any(np.abs(np.diff(command, axis=0)) > 1e-6, axis=-1)
    last = np.zeros(command.shape[1], dtype=int)
    elapsed = np.ones(command.shape[:2], dtype=int)
    for t in range(1, len(command)):
        last[change[t - 1]] = t
        elapsed[t] = t - last + 1
    return elapsed * .02


def euler(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                     np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))), axis=-1)


def stats(d):
    n = len(d['height'])
    if not n:
        return dict(samples=0)
    result = dict(samples=n)
    for key in ['ee_pos', 'ee_angle', 'height', 'abs_roll', 'abs_pitch',
                'cmd_xy_speed', 'actual_xy_speed']:
        v = d[key]
        result[key] = dict(mean=float(v.mean()), p05=float(np.quantile(v, .05)),
                           p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95)))
    result.update(xy_rmse=float(np.sqrt(d['xy_sq'].mean())),
                  yaw_rmse=float(np.sqrt(d['yaw_sq'].mean())),
                  ee_position_rmse=float(np.sqrt((d['ee_pos']**2).mean())),
                  nonfoot_step_fraction=float((d['nonfoot_force'] > 1).mean()),
                  base_below_020_fraction=float((d['height'] < .20).mean()),
                  nonfoot_support_z_fraction=float(d['nonfoot_support'].sum()/max(d['all_support'].sum(), 1e-9)),
                  feet_contact_fraction=float(d['feet_contact_fraction'].mean()),
                  yaw_response_gain=float(d['yaw_dot'].sum()/d['yaw_cmd_sq'].sum()) if d['yaw_cmd_sq'].sum() else None,
                  joint_speed_limit_ratio_max=float(d['joint_ratio'].max()),
                  joint_speed_limit_ratio_p95=float(np.quantile(d['joint_ratio'], .95)),
                  joint_speed_limit_ratio_p99=float(np.quantile(d['joint_ratio'], .99)),
                  overspeed_step_fraction=float((d['joint_ratio'] > np.float32(1.0001)).mean()),
                  overspeed_step_fractions={str(v): float((d['joint_ratio'] > np.float32(v)).mean()) for v in [1.0001, 1.01, 1.05, 1.1]})
    return result


results = dict(checkpoint=3000, unit_notes='m, m/s, rad, rad/s, N; XY RMSE is vector norm, not per-axis mean',
               windows='All first-episode valid samples; after_initial_1s excludes reset transient only; settled_both_1s excludes first 1 s after either native command change.',
               native_task_unchanged=True, modes={})
for mode in ['deterministic', 'stochastic']:
    collected = {}
    seed_summaries = []
    body_counts = {}
    valid_total = 0
    termination_counts = {}
    action_delta_sums = np.zeros(2)
    action_delta_counts = np.zeros(2)
    overspeed_episode_counts = {str(v): 0 for v in [1.0001, 1.01, 1.05, 1.1]}
    for seed in [7001, 7002, 7003]:
        path = ROOT / f'evaluation/cp_3000/{mode}/seed_{seed}'
        summary = json.loads((path / 'summary.json').read_text())
        seed_summaries.append(summary)
        with np.load(path / 'trajectories.npz') as z:
            names = ['valid', 'age', 'base_cmd', 'base_actual', 'ee_cmd', 'xy_error_sq', 'yaw_error_sq',
                     'ee_position_error', 'ee_orientation_error', 'base_height', 'base_roll', 'base_pitch',
                     'nonfoot_max_force', 'nonfoot_support_z', 'total_support_z', 'foot_forces',
                     'all_body_contact_forces', 'joint_velocity_limit_ratio', 'actions']
            d = {k: z[k] for k in names}
        valid = d['valid']
        valid_total += int(valid.sum())
        for v in [1.0001, 1.01, 1.05, 1.1]:
            overspeed_episode_counts[str(v)] += int(((d['joint_velocity_limit_ratio'] > np.float32(v)) & valid).any(0).sum())
        for i, name in enumerate(summary['native_contact_body_names']):
            body_counts[name] = body_counts.get(name, 0) + int((d['all_body_contact_forces'][:, :, i][valid] > 1).sum())
        for name, n in summary['terminations'].items():
            termination_counts[name] = termination_counts.get(name, 0) + n
        adj = valid[1:] & valid[:-1]
        delta = np.diff(d['actions'], axis=0)[adj]
        for j, sl in enumerate([slice(0, 12), slice(12, 18)]):
            action_delta_sums[j] += np.square(delta[:, sl], dtype=np.float64).sum()
            action_delta_counts[j] += delta[:, sl].size
        cmd_speed = np.linalg.norm(d['base_cmd'][:, :, :2], axis=-1)
        actual_speed = np.linalg.norm(d['base_actual'][:, :, :2], axis=-1)
        settled = (elapsed_since_change(d['base_cmd']) > 1) & (elapsed_since_change(d['ee_cmd']) > 1)
        groups = dict(all=valid, after_initial_1s=valid & (d['age'] > 50), settled_both_1s=valid & settled,
                      standing_settled=valid & settled & (cmd_speed < 1e-6) & (np.abs(d['base_cmd'][:, :, 2]) < 1e-6),
                      moving_command_settled=valid & settled & (cmd_speed >= .1),
                      actual_moving_settled=valid & settled & (cmd_speed >= .1) & (actual_speed >= .05))
        for lo, hi in [(.1, .3), (.3, .5), (.5, .8)]:
            groups[f'ee_world_z_{lo:.1f}_{hi:.1f}_settled'] = valid & settled & (d['ee_cmd'][:, :, 2] >= lo) & (d['ee_cmd'][:, :, 2] < hi)
        source = dict(height=d['base_height'], ee_pos=d['ee_position_error'], ee_angle=d['ee_orientation_error'],
                      abs_roll=np.abs(d['base_roll']), abs_pitch=np.abs(d['base_pitch']),
                      cmd_xy_speed=cmd_speed, actual_xy_speed=actual_speed, xy_sq=d['xy_error_sq'], yaw_sq=d['yaw_error_sq'],
                      nonfoot_force=d['nonfoot_max_force'], nonfoot_support=d['nonfoot_support_z'], all_support=d['total_support_z'],
                      feet_contact_fraction=(d['foot_forces'] > 1).mean(-1),
                      yaw_dot=d['base_cmd'][:, :, 2]*d['base_actual'][:, :, 2], yaw_cmd_sq=d['base_cmd'][:, :, 2]**2,
                      joint_ratio=d['joint_velocity_limit_ratio'])
        for group, mask in groups.items():
            target = collected.setdefault(group, {})
            for name, value in source.items():
                target.setdefault(name, []).append(value[mask])
    result = dict(episodes=768, groups={g: stats({k: np.concatenate(v) for k, v in fields.items()}) for g, fields in collected.items()},
                  terminations=termination_counts,
                  diagnostic_success_count=sum(s['diagnostic_success_count'] for s in seed_summaries),
                  survived_20s=sum(s['survived_20s'] for s in seed_summaries),
                  nonfoot_contact_episodes=sum(s['episodes_with_nonfoot_contact'] for s in seed_summaries),
                  overspeed_episodes=sum(s['episodes_with_overspeed_diagnostic'] for s in seed_summaries),
                  overspeed_episode_counts_by_ratio=overspeed_episode_counts,
                  per_body_contact_step_fraction={k: v/valid_total for k, v in body_counts.items()},
                  raw_action_delta_mse=dict(legs=action_delta_sums[0]/action_delta_counts[0], arm=action_delta_sums[1]/action_delta_counts[1]),
                  per_seed_all_metrics=[dict(seed=s['seed'], **s['groups']['all']['all']) for s in seed_summaries])
    results['modes'][mode] = result
    representative = json.loads((OUT / f'representative_{mode}.json').read_text())
    with np.load(Path(representative['source']) / 'trajectories.npz') as z:
        i = representative['env_id']; mask = z['valid'][:, i]; t = z['age'][:, i][mask]*.02
        target = euler(z['ee_target_quat_world'][:, i][mask]); actual = euler(z['ee_actual_quat_world'][:, i][mask])
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for dim, ax in enumerate(axes):
            ax.plot(t, np.rad2deg(target[:, dim]), label='target')
            ax.plot(t, np.rad2deg(actual[:, dim]), label='actual')
            ax.set_ylabel(['EE roll (deg)', 'EE pitch (deg)', 'EE yaw (deg)'][dim]); ax.grid(alpha=.2); ax.legend()
        axes[-1].set_xlabel('Episode time (s)')
        fig.suptitle(f'CP3000 {mode}, seed7001 env{i}: target / actual EE world orientation\nEuler display wraps at +/-180 degrees; quantitative error uses quaternion geodesic angle')
        fig.tight_layout(); fig.savefig(OUT / f'ee_orientation_{mode}.png', dpi=150); plt.close(fig)

(OUT / 'final_physics_details.json').write_text(json.dumps(results, indent=2) + '\n')
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
bins = ['ee_world_z_0.1_0.3_settled', 'ee_world_z_0.3_0.5_settled', 'ee_world_z_0.5_0.8_settled']
for mode, result in results['modes'].items():
    g = result['groups']
    axes[0].plot([.2, .4, .65], [g[b]['height']['mean'] for b in bins], '-o', label=mode)
    axes[1].plot([.2, .4, .65], [g[b]['ee_position_rmse'] for b in bins], '-o', label=mode)
    axes[2].plot([.2, .4, .65], [100*g[b]['nonfoot_step_fraction'] for b in bins], '-o', label=mode)
for ax, title in zip(axes, ['Mean base height (m)', 'EE position RMSE (m)', 'Non-foot contact (%)']):
    ax.set_title(title); ax.set_xlabel('EE world-z target bin (m)'); ax.set_xticks([.2, .4, .65], ['0.1-0.3', '0.3-0.5', '0.5-0.8']); ax.grid(alpha=.2); ax.legend()
fig.suptitle('Native CP3000: saved samples >1s after both native command changes; no task modifications')
fig.tight_layout(); fig.savefig(OUT / 'conditional_height_contact.png', dpi=150); plt.close(fig)
print('FINAL_PHYSICS_DIAGNOSTICS_COMPLETE', OUT / 'final_physics_details.json')
