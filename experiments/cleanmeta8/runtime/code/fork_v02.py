"""CPU conversion of immutable historical full student states into V0.2 forks.

The old checkpoint remains authoritative.  The new sampler has an explicitly
new, paired RNG stream; student/global RNG, PPO, physical tensors, native metric
accumulators and all active episode ownership are retained exactly.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def partial_features(adapter, n=32768):
    """Real unfinished-episode measurements, never completed/success feedback."""
    owner, metrics = adapter['owner'], adapter['metrics']
    ids = array(owner['task']).astype(np.int64)
    active = ids >= 0
    ids = ids[active]
    counts = np.bincount(ids, minlength=n)
    scored = np.bincount(ids, weights=array(metrics['scored'])[active], minlength=n)
    good = np.bincount(ids, weights=array(metrics['good'])[active], minlength=n)
    elapsed = np.bincount(ids, weights=array(owner['steps'])[active], minlength=n)
    sums = np.stack([np.bincount(ids, weights=array(metrics['sq'])[active, j], minlength=n)
                     for j in range(4)], 1)
    f = np.zeros((n, 8), dtype=np.float32)
    f[:, 0], f[:, 1] = counts > 0, scored > 0
    f[:, 2] = np.divide(scored, elapsed, out=np.zeros(n), where=elapsed > 0)
    f[:, 3] = np.divide(good, scored, out=np.zeros(n), where=scored > 0)
    f[:, 4:] = np.minimum(np.sqrt(np.divide(sums, scored[:, None],
        out=np.zeros((n, 4)), where=scored[:, None] > 0)) / [.2, .3, .05, np.pi/12], 10)
    assert np.isfinite(f).all()
    return f, dict(active_episodes=int(active.sum()), observed_sequences=int((counts > 0).sum()),
        valid_error_sequences=int((scored > 0).sum()), counts=counts, scored_steps=scored,
        physical_steps=elapsed, control_step=int(owner['control_steps']),
        contribution_to_completed_history=0, scope='Unfinished episodes; no terminal success is imputed.')


def learning_progress(engine, now):
    from teacher_v02 import learning_progress_from_records
    return learning_progress_from_records(engine.recent_records, engine.n, now)


def restored_owner(pool, state):
    """Build a clean owner, so bootstrap-only attributes cannot survive restore.

    In particular sample() lazily creates last_serials. A prepared historical
    checkpoint has never sampled through the new engine, so that field is
    legitimately absent. Loading onto the bootstrap engine would retain its
    unrelated last_serials; a fresh object prevents any such state leakage.
    """
    from neural_owner import NeuralOwner
    owner = NeuralOwner(pool, len(state['task']), seed=0)
    owner.load_state_dict(state)
    owner.check()
    assert set(owner.engine.state_dict()) == set(state['engine'])
    return owner


def convert_adapter(source, feedback_path, sampler_seed):
    from neural_v01 import NeuralSampler
    old = source['owner']
    n = len(old['transitions'])
    envs = len(old['task'])
    eng = NeuralSampler(n, int(sampler_seed))
    state = old['engine']
    assert 'episode_journal' in state, 'Source must retain complete E3 terminal provenance'
    journal = np.asarray(state['episode_journal'])
    assert journal.ndim == 2 and journal.shape[1] == 5
    assert len(journal) == int(old['accepted'])
    assert not len(journal) or np.all(np.diff(journal[:, 2]) >= 0)
    now = int(old['control_steps'])
    assert not len(journal) or int(journal[:, 2].max()) <= now
    eng.sample_counts = np.asarray(state['sample_counts'], dtype=np.int64).copy()
    # Reconstruct last-100 *sequence* labels from true chronological episodes.
    for seq, success, step, env, generation in journal:
        eng.record([int(seq)], [bool(success)])
    assert np.array_equal(eng.completed_counts, state['completed_counts'])
    assert np.array_equal(eng.success_counts, state['success_counts'])
    eng.serial = int(eng.sample_counts.sum())
    first = max(0, now-1000*24)
    prefix = hashlib.sha256()
    count = recent = cutoff = 0
    expected_start = 0
    feedback_paths = [Path(p) for p in feedback_path] if isinstance(feedback_path, (list, tuple)) else [Path(feedback_path)]
    cutoffs = []
    for feedback_path in feedback_paths:
      cutoff = 0
      with feedback_path.open('rb') as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            item = json.loads(raw)
            step = int(item['control_step'])
            if step > now:
                break
            prefix.update(raw)
            cutoff = f.tell()
            rows = np.asarray(item['rows'], dtype=np.float64)
            ids = np.asarray(item['task_ids'], dtype=np.int64)
            env_ids = np.asarray(item['env_ids'], dtype=np.int64)
            generations = np.asarray(item.get('generation', item.get('generations')), dtype=np.int64)
            assert rows.shape == (len(ids), 9)
            expected = journal[expected_start:expected_start+len(rows)]
            actual = np.column_stack((ids, rows[:, 6].astype(np.int64),
                np.full(len(rows), step), env_ids, generations))
            assert np.array_equal(actual, expected), 'Log does not match checkpoint terminal journal'
            expected_start += len(rows)
            count += len(rows)
            # Reconstruct measured EMAs from the actual chronological log too.
            # observe() never updates completed/history counts (recorded above).
            eng.observe(ids, rows, step)
            recent += int(step >= first)*len(rows)
      cutoffs.append(dict(path=str(feedback_path), bytes=cutoff))
    assert count == old['accepted'] == expected_start, 'Incomplete or future source feedback'
    assert np.array_equal(eng.completed_counts, state['completed_counts'])
    new_owner = copy.deepcopy(old)
    new_owner['engine'] = eng.state_dict()
    new_owner['teacher_version'] = np.full(envs, -1, dtype=np.int64)
    new_owner['decision_id'] = -np.arange(1, envs+1, dtype=np.int64)
    new_owner['selected_probability'] = np.zeros(envs, dtype=np.float64)
    new_owner['cycle_version_exposure'] = {}
    result = copy.deepcopy(source)
    result['owner'] = new_owner
    effective = source['summary']['effective_commands']
    import torch
    result['command_stats'] = {name: {key: torch.as_tensor(value).clone()
        for key, value in entries.items()} for name, entries in effective['command_draws'].items()}
    result['counts'] = dict(episodes=source['summary']['episodes'], successes=source['summary']['successes'],
        failures=source['summary']['native_failures'], standing_draws=effective['standing_draws'],
        velocity_draws=effective['velocity_draws'])
    receipt = dict(source_feedback=cutoffs, source_feedback_prefix_bytes=sum(x['bytes'] for x in cutoffs),
        source_feedback_prefix_sha256=prefix.hexdigest(), committed_terminal_records=count,
        recent_records_imported=recent, first_recent_control_step=first,
        last_allowed_control_step=now, duplicate_completed_updates=0,
        old_inflight_teacher_version=-1, old_inflight_decision_ids='negative unique per environment',
        old_inflight_assignment_probability='not retained by E3; zero placeholder, excluded from new decisions',
        new_sampler_seed=int(sampler_seed), new_sampler_rng='NumPy PCG64; paired across all arms',
        source_sampler_rng_retained=True,
        source_sampler_rng_sha256=hashlib.sha256(repr(state['rng_state']).encode()).hexdigest())
    return result, receipt


def prepare_source(source_checkpoint, feedback_path, destination, sampler_seed, student_seed):
    """One immutable prepared checkpoint/context per historical source, reused by all arms."""
    import torch
    from neural_v01 import NeuralSampler
    source_checkpoint, destination = Path(source_checkpoint), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    saved = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
    info = saved['infos']
    assert info['environment'] is not None and info['completed_updates'] > 0
    original = copy.deepcopy(info['adapter'])
    converted, receipt = convert_adapter(original, feedback_path, sampler_seed)
    partial, partial_receipt = partial_features(original)
    eng = NeuralSampler(len(converted['owner']['transitions']), 0)
    eng.load_state_dict(converted['owner']['engine'])
    now = int(converted['owner']['control_steps'])
    dynamic = eng.features(now, converted['owner']['transitions'])
    progress, progress_observed = learning_progress(eng, now)
    source_hash = sha(source_checkpoint)
    receipt.update(source_checkpoint=str(source_checkpoint), source_checkpoint_sha256=source_hash,
        student_seed=int(student_seed), completed_updates=int(info['completed_updates']))
    info['source_adapter_v02'] = original
    info['source_sampler_v02'] = copy.deepcopy(original['owner']['engine'])
    info['adapter'] = converted
    info['v02_preparation'] = receipt
    info['student_seed'] = int(student_seed)
    info['v02_context'] = dict(dynamic=dynamic, partial=partial, partial_receipt=partial_receipt,
        progress=progress, progress_observed=progress_observed, control_step=now,
        source_checkpoint_sha256=source_hash)
    torch.save(saved, destination)
    return receipt | dict(prepared_checkpoint=str(destination), prepared_checkpoint_sha256=sha(destination))
