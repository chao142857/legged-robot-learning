"""CPU-only final evidence audit; never selects a policy or launches GPU."""
import argparse, datetime, hashlib, json, pathlib, subprocess, time
from gpu_budget import read, stamp


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def slim(r):
    return dict(label=r['label'], protocol=r['protocol'], seed=r['seed'],
        checkpoint_sha256=r['checkpoint_sha256'], n=r['n'], complete=r['complete'],
        strict_safe=r['strict_safe'], hard=r['joint_limit'], nonfoot=r['nonfoot'],
        overspeed=r['overspeed'], native_joint5cm_count=r['joint_5cm'],
        full20s=r['full20'], after1s=r['after1s'],
        EE_orientation_RMS_rad=r['orientation_RMS_rad'], X=r['X_after1s'],
        directions=r['directions'], raw_trajectory_sha256=r['raw_trajectory_sha256'])


def collect(p, out):
    done = read(p/'COMPLETED.json'); a = read(p/'AUTHORIZATION.json')
    resources = read(p/'RESOURCES_FINAL.json'); frozen = read(p/'FROZEN_SELECTION.json')
    matrix = read(p/'FINAL_MATRIX.json'); consecutive = read(p/'CONSECUTIVE_NODE_FINAL.json')
    curve = read(p/'PHYSICAL_CURVE.json'); jobs = read(p/'LEDGER.json')['jobs']
    assert a['approved_GPU_seconds_SUM'] == 28800 and a['final_reserve_seconds'] == 3600
    assert done['clock'] == resources['clock'] and not done['clock']['active']
    assert done['clock']['GPU_container_SUM_seconds'] <= 28800
    assert done['selected']['checkpoint_sha256'] == frozen['candidate']['checkpoint_sha256']
    assert not frozen['used_final_seeds']
    actual_pass = len(matrix['checks']) == 12 and all(r['passed'] for r in matrix['checks'])
    adjacent_pass = len(consecutive['checks']) == 12 and all(r['passed'] for r in consecutive['checks'])
    assert actual_pass == done['primary_checks_passed']
    assert adjacent_pass == done['consecutive_nodes_passed']
    assert (actual_pass and adjacent_pass) == done['original_baseline_PASS']
    prior = pathlib.Path(a['prior_closed_stage3_root'])
    assert all(sha(prior/name) == value for name, value in a['prior_closed_file_sha256'].items())
    cp = frozen['candidate']; assert sha(cp['checkpoint']) == cp['checkpoint_sha256']
    selected_training_run=pathlib.Path(cp['checkpoint']).parents[2]
    selected_training_config=read(selected_training_run/'config.json')
    if frozen['adjacent']:
        previous = frozen['adjacent']['checkpoint']
        assert sha(previous['checkpoint']) == previous['checkpoint_sha256']
    restore_run = pathlib.Path(read(p/'FINAL_RESTORE_RUN.json')['run'])
    restore = read(restore_run/'ppo/result.json')
    assert restore['passed'] and restore['updates'] == restore['optimizer_steps'] == 0
    assert read(restore_run/'config.json')['checkpoint_sha256'] == cp['checkpoint_sha256']
    # Check actual immutable launch snapshots and Docker durations, including
    # paused occupation; do not edit either the active or the closed ledger.
    states = [json.loads(s) for s in subprocess.check_output(
        ['docker', 'inspect', '--format', '{{json .State}}', *[j['cid'] for j in jobs]],
        text=True).splitlines()]
    assert len(states) == len(jobs) and all(not s['Running'] for s in states)
    seconds = sum(stamp(s['FinishedAt']) - stamp(s['StartedAt']) for s in states
                  if not s['StartedAt'].startswith('0001'))
    assert abs(seconds - done['clock']['GPU_container_SUM_seconds']) < .01
    snapshots = []; costs = {}; stochastic = dict(rollout_updates=0,
        rollouts_with_any_hard_excess_above_native1mrad=0,
        rollouts_with_nonfoot_positive_slots=0, nonfoot_positive_transition_slots=0,
        maximum_recorded_all_joint_hard_excess_rad=0., maximum_rollout_mean_nonfoot_fraction=0.)
    for j, state in zip(jobs, states):
        run = pathlib.Path(j['run']); cfg = read(run/'config.json'); launch = read(run/'launch.json')
        assert run.parent == p/'runs' and cfg['grant_id'] == a['grant_id']
        assert all(sha(run/'source'/name) == value for name, value in launch['source_sha256'].items())
        snapshots.append(dict(run=run.name, config_sha256=sha(run/'config.json'),
            launch_sha256=sha(run/'launch.json'), frozen_source_files=len(launch['source_sha256'])))
        key = j['mode']+'/'+j['phase']; seconds_job = 0. if state['StartedAt'].startswith('0001') else stamp(state['FinishedAt'])-stamp(state['StartedAt'])
        costs[key] = costs.get(key, 0.) + seconds_job
        updates = run/'ppo/updates.jsonl'
        if updates.exists():
            for line in updates.open():
                row = json.loads(line); physical = row['physical_reward_time']
                hard = physical['hard_excess_max']; nonfoot = physical['nonfoot_fraction']
                slots = nonfoot*row['transitions']; assert abs(slots-round(slots)) < .01
                stochastic['rollout_updates'] += 1
                stochastic['rollouts_with_any_hard_excess_above_native1mrad'] += int(hard > .001)
                stochastic['rollouts_with_nonfoot_positive_slots'] += int(nonfoot > 0)
                stochastic['nonfoot_positive_transition_slots'] += round(slots)
                stochastic['maximum_recorded_all_joint_hard_excess_rad'] = max(stochastic['maximum_recorded_all_joint_hard_excess_rad'], hard)
                stochastic['maximum_rollout_mean_nonfoot_fraction'] = max(stochastic['maximum_rollout_mean_nonfoot_fraction'], nonfoot)
    assert stochastic['rollout_updates'] == resources['totals']['PPO_updates']
    formal = [r for r in curve['rows'] if r['label'].startswith('final-')]
    assert {r['seed'] for r in formal} == {611101, 621101, 631101}
    assert all(r['n'] == 64 for r in formal)
    result = dict(stage='Stage3b original Uniform PPO dynamicX and safety', closed=True,
        original_baseline_PASS=actual_pass and adjacent_pass, SGS_ready=actual_pass and adjacent_pass,
        recipe='Single18D synergy mean actor with the explicit selected learning objective below; original task/27rewards/Uniform/physics/evaluator unchanged',
        selected_training_recipe_id=cp.get('recipe_id'),selected_training_run=str(selected_training_run),
        selected_learning_objective=selected_training_config.get('leg_retention'),
        common_parent_sha256=read(p/'CHECKPOINTS.json')['parent']['checkpoint_sha256'],
        selected_checkpoint=cp, adjacent=frozen['adjacent'],
        candidate_checks=matrix['checks'], adjacent_checks=consecutive['checks'],
        formal_rows=[slim(r) for r in formal],
        per_training_seed_development=[dict(label=n['label'], training_seed=n['training_seed'],
            recipe=n['recipe_id'], additional_updates=n['additional_actual_updates'],
            checkpoint=n['checkpoint'], decision={k:v for k,v in n['decision'].items()
                if k not in ('rows', 'same_parent_seed_schedule_unconstrained_control')})
            for n in read(p/'CANDIDATE_NODES.json')],
        GPU_container_SUM_seconds=seconds, approved_cap_seconds=28800,
        unused_seconds_not_transferred=28800-seconds, counters=resources['totals'],
        GPU_SUM_seconds_by_mode_and_phase=costs, stochastic_training_rollout_summary=stochastic,
        partial_evaluation_count_uncertainty=resources['unmeasured_partial_evaluations'],
        full_learner_restore=restore, closed_Stage3_SHA_unchanged=True,
        frozen_run_audit=snapshots, audit_source_sha256=sha(__file__),
        limitations=['Shared-parent continuation seeds are not complete from-scratch seeds.',
            'Tracking uses the unchanged original after1s mask; strict safety and failure denominators include startup and all20s.',
            'Native joint5cm is not dynamicX or complete baseline acceptance.',
            'Frozen diagnostic leg/arm compositions are not learned single-policy results.',
            'Stochastic training rollouts can include failed exploration and native resets. Their positive-contact slots are not distinct episodes/events, and the aggregate hard maximum does not identify a joint. Zero hard/contact formal-policy claims must never be extended to all training samples.',
            'Parent pretraining is outside this grant and is not relabeled as new PPO learning.'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'FINAL_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), baseline_PASS=result['original_baseline_PASS'],
        GPU_SUM_seconds=seconds, selected_sha256=cp['checkpoint_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait', action='store_true'); args = parser.parse_args()
    if args.wait:
        while not (args.root/'COMPLETED.json').exists():
            if (args.root/'FINALIZER_ERROR.json').exists():
                raise RuntimeError('Actual finalizer error preserved; repair final evaluation before delivery')
            time.sleep(5)
    name = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    collect(args.root, args.root/'delivery'/name)
