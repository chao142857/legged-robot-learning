"""Offline, read-only analysis of bounded Go2+Piper task-suite runs.

Run on the execution host; raw trajectories/checkpoints never leave that host.
This tool does not import Isaac, run simulation, modify runs, or infer admission.
A short smoke has null task-success rates, even when its censored flags are false.

Example:
  python tools/analyze_suite.py --root /path/to/experiment --out /new/report \
      --pair eval_t0_cp3000 eval_t0_post --pair eval_t1_cp3000 eval_t1_post
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


RATES = ('complete', 'failure', 'safe', 'contact_safe', 'overspeed',
         'joint_limit_violation', 'nonfoot_contact', 'ee_success_10cm',
         'ee_success_5cm', 'movement_success', 'joint_success_10cm', 'joint_success_5cm', 'route_endpoint')
METRICS = ('xy_rmse', 'yaw_rmse', 'ee_position_rmse', 'ee_position_p95',
           'ee_orientation_mean', 'base_drift_max_m', 'base_drift_final_m',
           'route_progress_final_m', 'goal_distance_final_m', 'duration_s',
           'base_roll_abs_mean', 'base_pitch_abs_mean', 'base_clearance_min',
           'applied_torque_abs_max', 'joint_velocity_limit_ratio_max',
           'joint_position_limit_excess_max_rad', 'nonfoot_step_fraction',
           'xy_command_response_gain', 'yaw_command_response_gain',
           'actual_xy_speed_mean', 'commanded_xy_speed_mean')
PAIR_METRICS = ('xy_rmse', 'yaw_rmse', 'ee_position_rmse', 'ee_orientation_mean',
                'base_drift_max_m', 'route_progress_final_m', 'goal_distance_final_m',
                'duration_s', 'joint_velocity_limit_ratio_max',
                'applied_torque_abs_max', 'joint_success_10cm', 'joint_success_5cm',
                'movement_success', 'ee_success_10cm', 'safe', 'failure', 'overspeed', 'route_endpoint')


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def scalar(value):
    if value in ('True', 'true'):
        return True
    if value in ('False', 'false'):
        return False
    if value in ('', 'None', None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (dict, list)) else clean(v)
                             for k, v in row.items()})


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mean(values):
    values = [float(v) for v in values if finite(v)]
    return statistics.fmean(values) if values else None


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., center-half), min(1., center+half)]


def quantile(values, p):
    x = sorted(values)
    loc = (len(x)-1)*p
    lo, hi = int(math.floor(loc)), int(math.ceil(loc))
    return x[lo] + (x[hi]-x[lo])*(loc-lo)


def bootstrap_mean(values, seed=20261023):
    if len(values) < 2:
        return [None, None]
    rng = random.Random(seed)
    draws = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(2000)]
    return [quantile(draws, .025), quantile(draws, .975)]


def load_run(path, root):
    protocol = read_json(path/'protocol.json', {})
    summary = read_json(path/'summary.json', {})
    errors = read_json(path/'error.json')
    rows = []
    if (path/'episodes.csv').is_file():
        with (path/'episodes.csv').open(newline='') as f:
            rows = [{k: scalar(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if protocol.get('task', {}).get('family') == 'T2':
        tolerance = protocol.get('success_thresholds', {}).get('T2_final_goal_distance')
        for row in rows:
            distance = row.get('goal_distance_final_m')
            if finite(tolerance) and finite(distance) and isinstance(row.get('complete'), bool):
                row['route_endpoint'] = bool(row['complete'] and distance < tolerance)
    updates, parse_errors = [], []
    if (path/'updates.jsonl').is_file():
        for i, line in enumerate((path/'updates.jsonl').read_text().splitlines(), 1):
            try:
                updates.append(json.loads(line))
            except json.JSONDecodeError as exc:
                parse_errors.append(f'updates.jsonl:{i}: {exc.msg}; possibly interrupted write')
    return dict(name=str(path.relative_to(root)), path=path, protocol=protocol,
                summary=summary, episodes=rows, updates=updates, error=errors,
                parse_errors=parse_errors)


def complete_population(run):
    expected = run['summary'].get('num_episodes')
    ids = [r.get('env_id') for r in run['episodes']]
    return (isinstance(expected, int) and expected > 0 and len(ids) == expected and
            all(finite(i) and int(i) == i for i in ids) and set(ids) == set(range(expected)))


def run_row(run):
    p, s, rows = run['protocol'], run['summary'], run['episodes']
    task = p.get('task', {})
    mode = p.get('mode', s.get('mode', 'train' if run['updates'] else 'unknown'))
    full = s.get('complete_episode_evaluation') is True
    population_complete = complete_population(run)
    status = ('failed' if run['error'] else
              'bounded_train_completed' if mode == 'train' and s.get('stopped_reason') == 'bounded run completed' else
              'train_partial_or_running' if mode == 'train' else
              'full_horizon_evaluation' if full and population_complete else
              'incomplete_evaluation_records' if full else
              'censored_smoke' if rows else 'incomplete_or_unmeasured')
    row = dict(run=run['name'], mode=mode, status=status,
               family=task.get('family', s.get('family')), terrain=task.get('terrain', s.get('terrain')),
               difficulty=task.get('difficulty'), ee_mode=task.get('ee_mode'), seed=p.get('seed', s.get('seed')),
               stochastic=s.get('stochastic'), episodes=len(rows), full_horizon_requested=full,
               evaluation_population_complete=population_complete,
               checkpoint=p.get('checkpoint'), checkpoint_sha256=p.get('checkpoint_sha256'),
               admission_key=p.get('admission_key', s.get('admission_key')),
               requested_steps=s.get('requested_steps'), max_steps=s.get('max_steps'),
               observed_updates=len(run['updates']), requested_updates=p.get('requested_updates'),
               actor_frozen=s.get('actor_unchanged'), critic_frozen=s.get('critic_unchanged'))
    for key in METRICS:
        vals = [r.get(key) for r in rows]
        row[key+'_episode_mean'] = mean(vals)
        row[key+'_episode_n'] = sum(finite(x) for x in vals)
    # Full-horizon requested evaluation includes all early-terminated episodes.
    # Smoke event counts are descriptive, never full-task success/failure rates.
    for key in RATES:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), bool)]
        k, n = sum(vals), len(vals)
        row[key+'_observed_count'] = k if n else None
        rate_valid = full and population_complete and n == len(rows) and n > 0
        row[key+'_rate'] = k/n if rate_valid else None
        row[key+'_wilson95_low'], row[key+'_wilson95_high'] = wilson(k,n) if rate_valid else [None,None]
    for key, value in s.get('action_delta_mse', {}).items() if s.get('action_delta_mse') else []:
        row['action_delta_'+key+'_mse'] = value
    for key, value in s.get('command_dynamics', {}).items():
        if key != 'note':
            row[key] = value
    row['error'] = run['error'].get('error') if isinstance(run['error'], dict) else run['error']
    return row


def groups_rows(run):
    return [dict(run=run['name'], window=window, command_group=group,
                 aggregation='pooled valid pre-reset steps; failed episodes retained until termination', **values)
            for window, groups in run['summary'].get('groups', {}).items()
            for group, values in groups.items()]


def training_row(run):
    updates = run['updates']
    if not updates:
        return None
    row = dict(run=run['name'], observed_updates=len(updates),
               completed_updates=updates[-1].get('completed_updates'),
               env_steps=updates[-1].get('env_steps'), elapsed_s=updates[-1].get('elapsed_s'),
               schedule=run['protocol'].get('ppo_schedule'),
               requested_updates=run['protocol'].get('requested_updates'),
               completion=run['summary'].get('stopped_reason', 'partial, failed, or running'),
               gradient_convention='unclipped norm returned by clip_grad_norm_; >1 alone is not failure')
    for key in ('kl_mean', 'kl_max', 'actor_grad_norm_mean', 'actor_grad_norm_max',
                'critic_grad_norm_mean', 'critic_grad_norm_max', 'learning_rate'):
        values = [u.get(key) for u in updates]
        good = [v for v in values if finite(v)]
        row[key+'_first'] = values[0]
        row[key+'_last'] = values[-1]
        row[key+'_mean'] = mean(values)
        row[key+'_max'] = max(good) if good else None
        row[key+'_nonfinite_or_missing_count'] = len(values)-len(good)
    window = min(5, len(updates))
    row['trend_window_updates'] = window
    keys = sorted({k for u in updates for k in u.get('metrics', {})})
    for key in keys:
        row['train/'+key+'_first_window'] = mean([u.get('metrics',{}).get(key) for u in updates[:window]])
        row['train/'+key+'_last_window'] = mean([u.get('metrics',{}).get(key) for u in updates[-window:]])
    row['training_trend_scope'] = 'on-policy samples including partial episodes; not independent evaluation or learning proof'
    stds = [v for u in updates for v in u.get('action_std', []) if finite(v)]
    row['action_std_min'] = min(stds) if stds else None
    row['action_std_max'] = max(stds) if stds else None
    max_streak = streak = 0
    for u in updates:
        streak = streak+1 if finite(u.get('kl_mean')) and u['kl_mean']>.1 else 0
        max_streak = max(max_streak, streak)
    row['kl_mean_above_0_1_max_consecutive_updates'] = max_streak
    row['kl_mean_above_0_1_update_count'] = sum(finite(u.get('kl_mean')) and u['kl_mean'] > .1 for u in updates)
    return row


def compare_pair(pre, post):
    a, b = pre['protocol'], post['protocol']
    sa, sb = pre['summary'], post['summary']
    checks = {}
    for key in ('admission_key', 'task', 'seed', 'observation_noise', 'evaluator_sha256', 'telemetry_sha256', 'success_thresholds'):
        checks[key] = key in a and key in b and a[key] == b[key]
    for key in ('num_episodes', 'requested_steps', 'max_steps', 'stochastic'):
        checks[key] = key in sa and key in sb and sa[key] == sb[key]
    checks['full_horizon_requested'] = sa.get('complete_episode_evaluation') is True and sb.get('complete_episode_evaluation') is True
    checks['frozen_evaluations'] = all(s.get('actor_unchanged') is True and s.get('critic_unchanged') is True for s in (sa,sb))
    checks['no_runtime_error'] = not pre['error'] and not post['error']
    checks['complete_episode_population'] = complete_population(pre) and complete_population(post)
    ida = read_json(pre['path']/'candidate_identity.json', {})
    idb = read_json(post['path']/'candidate_identity.json', {})
    checks['source_and_asset_bindings_equal'] = all(
        isinstance(ida.get('bindings', {}).get(group), dict) and bool(ida['bindings'][group]) and
        ida['bindings'][group] == idb.get('bindings', {}).get(group) for group in ('sources', 'assets'))
    ta = read_json(pre['path']/'transfer.json', {})
    tb = read_json(post['path']/'transfer.json', {})
    checks['verified_strict_transfer_same_observation_dims'] = (
        all(t.get('strict_load') is True and t.get('bitwise_equal') == {'actor': True, 'critic': True}
            for t in (ta, tb)) and bool(ta.get('observation_dims')) and
        ta.get('observation_dims') == tb.get('observation_dims'))
    reset_a, reset_b = read_json(pre['path']/'reset_state.json',{}), read_json(post['path']/'reset_state.json',{})
    checks['initial_root_and_joints_equal'] = all(k in reset_a and k in reset_b and reset_a[k] == reset_b[k] for k in ('root_state','joint_pos'))
    ar = {r['env_id']: r for r in pre['episodes']}
    br = {r['env_id']: r for r in post['episodes']}
    checks['matching_episode_ids'] = bool(ar) and set(ar) == set(br)
    valid = all(checks.values())
    header = dict(pre_run=pre['name'], post_run=post['name'], comparable=valid,
                  rejected_checks=[k for k,v in checks.items() if not v], checks=checks,
                  checkpoint_changed=bool(a.get('checkpoint_sha256') and b.get('checkpoint_sha256') and
                                          a['checkpoint_sha256'] != b['checkpoint_sha256']),
                  uncertainty='2000 bootstrap resamples of paired env IDs; single training-seed evidence, not across-seed significance')
    result = []
    if valid:
        for key in PAIR_METRICS:
            matched = [i for i in sorted(ar) if finite(ar[i].get(key)) and finite(br[i].get(key))]
            deltas = [float(br[i][key])-float(ar[i][key]) for i in matched]
            if not deltas:
                continue
            ci = bootstrap_mean(deltas)
            degenerate = len(deltas) > 1 and max(deltas)-min(deltas) < 1e-12
            result.append(dict(pre_run=pre['name'], post_run=post['name'], metric=key,
                               n=len(deltas), pre_mean=mean([ar[i][key] for i in matched]),
                               post_mean=mean([br[i][key] for i in matched]),
                               post_minus_pre=mean(deltas), bootstrap95_low=ci[0], bootstrap95_high=ci[1],
                               degenerate_bootstrap=degenerate,
                               interval_warning=('Constant observed paired differences give a degenerate interval; '
                                   'this does not establish certainty or absence of an effect' if degenerate else None),
                               lower_is_better=key not in ('duration_s','route_progress_final_m','joint_success_10cm',
                                   'joint_success_5cm','movement_success','ee_success_10cm','safe','route_endpoint')))
    return header, result


def plot_trajectory(run, out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    file = run['path']/'trajectories.npz'
    if not file.is_file():
        return None
    p, s = run['protocol'], run['summary']
    runtime = read_json(run['path']/'initial_runtime.json',{})
    dt = runtime.get('step_dt')
    if not finite(dt) or dt <= 0:
        raise ValueError('initial_runtime.json missing positive step_dt; cannot label physical time')
    family = p.get('task',{}).get('family', s.get('family'))
    full = s.get('complete_episode_evaluation') is True
    with np.load(file, allow_pickle=False) as data:
        valid = data['valid'].astype(bool)
        base = data['base_pos_world']
        cmd = data['ee_cmd']
        ee = data['ee_actual_mixed']
        i = 0  # Fixed deterministic selection; never cherry-pick the best episode.
        mask = valid[:,i]
        times = np.arange(1,len(mask)+1)*dt
        t = times[mask]
        if not len(t):
            raise ValueError('env 0 has no valid pre-reset samples')
        fig, axes = plt.subplots(2,3,figsize=(12,6.6), constrained_layout=True)
        ax = axes[0,0]
        starts = data['initial_base_pos_world']
        for j in range(min(4,valid.shape[1])):
            good = valid[:,j]
            xy = base[good,j,:2]-starts[j,:2]
            if len(xy):
                ax.plot(xy[:,0],xy[:,1],lw=1,label=f'env {j}')
                ax.scatter(xy[-1,0],xy[-1,1],s=14)
        ax.scatter([0],[0],s=25,marker='x',c='k',label='reset')
        ax.set(xlabel='relative x (m)',ylabel='relative y (m)',title='Measured base paths (env 0-3)')
        ax.axis('equal'); ax.legend(fontsize=7)
        ax = axes[0,1]
        if family == 'T2':
            progress = data['route_progress'][mask,i]
            ax.plot(progress, data['base_pos_world'][mask,i,2],label='base root z',lw=1.4)
            ax.plot(progress, data['support_height'][mask,i],label='raycast support z',lw=1.2,color='k')
            body_names = s.get('articulation_body_names',[])
            if 'body_pos_world_first16' in data:
                bodies = data['body_pos_world_first16']
                for idx,name in enumerate(body_names):
                    if name.endswith('_foot') and idx < bodies.shape[2]:
                        ax.plot(progress,bodies[mask,i,idx,2],lw=.7,alpha=.75,label=name)
            route = p.get('task',{}).get('route') or {}
            if route.get('goal_offset') is not None and route.get('start_offset') is not None:
                distance = float(np.linalg.norm(np.array(route['goal_offset'])-route['start_offset']))
                ax.axvline(distance,ls=':',color='r',lw=1,label='goal distance')
                ax.set_xlim(min(-.1,float(progress.min())-.05),max(distance+.1,float(progress.max())+.05))
            ax.set(xlabel='base route progress (m)',ylabel='world z (m)',
                   title='Visited terrain support + root/foot heights')
            ax.legend(fontsize=6,ncol=2)
        else:
            base_cmd, actual = data['base_cmd'], data['base_actual']
            for k,c in enumerate(('tab:blue','tab:orange')):
                ax.plot(t,actual[mask,i,k],color=c,lw=1,label=('actual vx','actual vy')[k])
                ax.plot(t,base_cmd[mask,i,k],color=c,lw=1,ls='--',label=('target vx','target vy')[k])
            ax.set(xlabel='time (s)',ylabel='body velocity (m/s)',title='Base tracking (env 0)')
            ax.legend(fontsize=7,ncol=2)
        ax=axes[0,2]
        ax.plot(t,data['ee_position_error'][mask,i],label='EE position error (m)')
        ax.plot(t,data['ee_orientation_error'][mask,i],label='EE orientation error (rad)')
        ax.set(xlabel='time (s)',title='Measured EE errors (env 0)'); ax.legend(fontsize=7)
        for k,label in enumerate(('link0 X','link0 Y','world Z')):
            ax=axes[1,k]
            ax.plot(t,ee[mask,i,k],label='actual',lw=1.1)
            ax.plot(t,cmd[mask,i,k],label='desired',lw=1.1,ls='--')
            ax.set(xlabel='time (s)',ylabel=label+' (m)',title=f'Native mixed-frame EE {label} (env 0)')
            ax.legend(fontsize=7)
        for ax in axes.flat:
            ax.grid(alpha=.2)
        scope = ('FULL-HORIZON REQUESTED; failed episodes stop at termination' if full else
                 'CENSORED SMOKE; no complete-task capability conclusion')
        fig.suptitle(run['name']+'\n'+scope,fontsize=10)
        filename = out/(run['name'].replace('/','__')+'__trajectory.png')
        fig.savefig(filename,dpi=110); plt.close(fig)
        return dict(run=run['name'], path=str(filename), bytes=filename.stat().st_size,
                    sha256=sha(filename), source=str(file), source_bytes=file.stat().st_size,
                    selection='env 0 time series; env 0-3 base paths; all valid first-episode frames',
                    frame='EE x/y in link0, z in world; no false world-frame XYZ reconstruction',
                    terrain_scope='raycast support over visited base footprints, not complete unvisited terrain geometry')


def plot_training(run,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    updates=run['updates']
    if not updates:
        return None
    x=[u['completed_updates'] for u in updates]
    fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
    for key in ('kl_mean','kl_max'):
        axes[0,0].plot(x,[u.get(key) for u in updates],label=key)
    axes[0,0].axhline(.01,color='k',ls=':',lw=1,label='native desired KL=.01')
    axes[0,0].set(title='Native PPO KL',ylabel='KL')
    for key in ('actor_grad_norm_mean','critic_grad_norm_mean'):
        axes[0,1].plot(x,[u.get(key) for u in updates],label=key.replace('_grad_norm_mean',''))
    axes[0,1].set(title='Pre-clipping gradient norms',ylabel='L2 norm')
    axes[0,2].plot(x,[u.get('learning_rate') for u in updates],label='adaptive LR')
    axes[0,2].set(title='Native adaptive learning rate')
    for key in ('xy_mse','yaw_mse','ee_mse'):
        axes[1,0].plot(x,[math.sqrt(value) if finite(value:=u.get('metrics',{}).get(key)) and value >= 0 else math.nan for u in updates],label=key.replace('mse','rmse'))
    axes[1,0].set(title='On-policy tracking; not held-out evaluation')
    for key in ('action_delta_legs_mse','action_delta_arm_mse'):
        axes[1,1].plot(x,[u.get('metrics',{}).get(key) for u in updates],label=key.replace('action_delta_',''))
    axes[1,1].set(title='Adjacent actions excluding reset edges')
    for key in ('actual_speed','commanded_speed','nonfoot_fraction'):
        axes[1,2].plot(x,[u.get('metrics',{}).get(key) for u in updates],label=key)
    axes[1,2].set(title='Movement/contact diagnostics (mixed units)')
    for ax in axes.flat:
        ax.set_xlabel('completed PPO updates'); ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.suptitle(run['name']+' — bounded training telemetry',fontsize=10)
    filename=out/(run['name'].replace('/','__')+'__training.png')
    fig.savefig(filename,dpi=110); plt.close(fig)
    return dict(run=run['name'],path=str(filename),bytes=filename.stat().st_size,sha256=sha(filename))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True,help='New directory; existing paths are refused')
    parser.add_argument('--pair',nargs=2,action='append',default=[],metavar=('PRE_RUN','POST_RUN'))
    parser.add_argument('--no-plots',action='store_true',help='Tables only; pure Python standard library')
    parser.add_argument('--hash-trajectories',action='store_true',help='Optional full local read; never copies raw data')
    args=parser.parse_args()
    root=args.root.resolve()
    out=args.out.resolve()
    if not root.is_dir():
        parser.error('--root must exist')
    if out.exists():
        parser.error('Refusing to overwrite existing report output')
    paths=sorted({p.parent for name in ('protocol.json','summary.json','episodes.csv','updates.jsonl','error.json')
                  for p in root.rglob(name)
                  if p.parent != out and (p.parent/'protocol.json').is_file()})
    if not paths:
        parser.error('No task-suite run directories with protocol.json found')
    runs=[load_run(p,root) for p in paths]
    lookup={r['name']:r for r in runs}
    for pre,post in args.pair:
        if pre not in lookup or post not in lookup:
            parser.error(f'Unknown pair: {pre}, {post}; use run paths relative to --root')
    out.mkdir(parents=True,exist_ok=False)
    rows=[run_row(r) for r in runs]
    training=[row for r in runs if (row:=training_row(r))]
    groups=[row for r in runs for row in groups_rows(r)]
    episodes=[dict(run=r['name'],**row) for r in runs for row in r['episodes']]
    pairs,paired_metrics=[],[]
    for pre,post in args.pair:
        header,metrics=compare_pair(lookup[pre],lookup[post])
        pairs.append(header); paired_metrics.extend(metrics)
    warnings=[]
    for row in training:
        if row['kl_mean_above_0_1_update_count']:
            warnings.append(dict(run=row['run'],warning='At least one update had mean KL > .1; a later normal update does not establish stable optimization'))
        if row['kl_mean_above_0_1_max_consecutive_updates'] >= 5:
            warnings.append(dict(run=row['run'],warning='Mean KL exceeded .1 for at least five consecutive updates'))
        missing=[key for key,value in row.items() if key.endswith('_nonfinite_or_missing_count') and value]
        if missing:
            warnings.append(dict(run=row['run'],warning='Nonfinite or absent PPO diagnostics: '+', '.join(missing)))
    for pair in pairs:
        if not pair['comparable']:
            warnings.append(dict(run=pair['post_run'],warning='Rejected paired comparison: '+', '.join(pair['rejected_checks'])))
        elif not pair['checkpoint_changed']:
            warnings.append(dict(run=pair['post_run'],warning='Paired checkpoint hashes unchanged or unavailable; do not claim parameter learning'))
    figures=[]
    provenance=[]
    for run,row in zip(runs,rows):
        warnings.extend(dict(run=run['name'],warning=w) for w in run['parse_errors'])
        if row['status'] in ('censored_smoke','failed','incomplete_or_unmeasured','train_partial_or_running','incomplete_evaluation_records'):
            warnings.append(dict(run=run['name'],warning=row['status']+'; do not claim full-task or training acceptance'))
        if row['mode'] in ('eval','physics') and (row['actor_frozen'] is not True or row['critic_frozen'] is not True):
            warnings.append(dict(run=run['name'],warning='Frozen actor/critic integrity was not confirmed'))
        inputs={}
        for name in ('protocol.json','summary.json','episodes.csv','updates.jsonl','reset_state.json','initial_runtime.json','transfer.json','candidate_identity.json','error.json','manifest.json'):
            file=run['path']/name
            if file.is_file():
                inputs[name]=dict(path=str(file),bytes=file.stat().st_size,sha256=sha(file))
        file=run['path']/'trajectories.npz'
        if file.is_file():
            inputs['trajectories.npz']=dict(path=str(file),bytes=file.stat().st_size)
            if args.hash_trajectories:
                inputs['trajectories.npz']['sha256']=sha(file)
            else:
                original_manifest=read_json(run['path']/'manifest.json',{})
                if 'trajectories.npz' in original_manifest:
                    inputs['trajectories.npz']['producer_sha256_unrechecked']=original_manifest['trajectories.npz'].get('sha256')
        provenance.append(dict(run=run['name'],inputs=inputs))
        if not args.no_plots:
            for fn in (plot_trajectory,plot_training):
                try:
                    artifact=fn(run,out)
                    if artifact:
                        figures.append(artifact)
                except Exception as exc:
                    warnings.append(dict(run=run['name'],warning=f'{fn.__name__}: {type(exc).__name__}: {exc}'))
    coverage=[]
    for family in ('T0','T1','T2'):
        family_rows=[r for r in rows if r['family']==family]
        coverage.append(dict(family=family,run_count=len(family_rows),
                             full_evaluation_runs=[r['run'] for r in family_rows if r['status']=='full_horizon_evaluation'],
                             censored_smoke_runs=[r['run'] for r in family_rows if r['status']=='censored_smoke'],
                             completed_bounded_train_runs=[r['run'] for r in family_rows if r['status']=='bounded_train_completed'],
                             status='unmeasured' if not family_rows else 'observed; task admission must be decided from full evidence'))
    report=dict(schema_version=2,root=str(root),analysis_source_sha256=sha(Path(__file__)),
                runs=rows,training=training,pairs=pairs,paired_metrics=paired_metrics,
                coverage=coverage,warnings=warnings,figures=figures,provenance=provenance,
                interpretation=[
                    'Smoke success/failure rates are null; observed events are only within the censored time window.',
                    'Full-horizon requested rates retain early-terminated failures in the denominator; incomplete episode populations cannot have rates.',
                    'route_endpoint is derived only for T2: complete first episode and final goal distance below the protocol threshold; no EE or safety success is implied.',
                    'Run metrics are episode-equal means; command_groups.csv gives separate pooled-step diagnostics.',
                    'Means before failure are not full-horizon imputed errors; survival and failures must accompany them.',
                    'Bootstrap intervals concern paired environment rollouts for one training seed, not repeated training seeds; constant observed paired differences can give degenerate intervals.',
                    'Pre-clipping gradient norms above max_grad_norm do not imply clipping failure.',
                    'Higher reward/on-policy improvement alone cannot establish held-out joint task learning.',
                    'Unmeasured or untrained T2 is not a passed task. No automatic admission, long training, or SGS launch.'])
    for filename,contents in (('runs.csv',rows),('training.csv',training),('episodes.csv',episodes),
                              ('command_groups.csv',groups),('paired_metrics.csv',paired_metrics),('coverage.csv',coverage)):
        write_csv(out/filename,contents)
    write_json(out/'report.json',report)
    write_json(out/'paired_checks.json',pairs)
    print(json.dumps(dict(out=str(out),runs=len(rows),training_runs=len(training),
                          valid_pairs=sum(p['comparable'] for p in pairs),figures=len(figures),
                          warnings=len(warnings)),ensure_ascii=False))


if __name__=='__main__':
    main()
