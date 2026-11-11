"""CPU helpers for bounded, causally ordered V0 -> V0.1 diagnostic forks."""
import hashlib
import json
from pathlib import Path
import numpy as np


def bootstrap_recent(engine, checkpoint, info):
    """Import only committed real feedback, without changing ownership/history counts."""
    import torch
    source = Path(info['experience']) / f"cycle_{info['decision']['version']:03d}" / 'decision.pt'
    receipt = torch.load(source, weights_only=False, map_location='cpu')
    assert receipt['completed_updates'] == info['completed_updates']
    cutoff = int(receipt['feedback_start_bytes'])
    feedback = Path(receipt['feedback_file'])
    assert feedback.is_file() and feedback.stat().st_size >= cutoff
    limit_step = int(info['adapter']['owner']['control_steps'])
    first_step = max(0, limit_step - 1000 * 24)
    samples_before = engine.sample_counts.copy()
    completed_before = engine.completed_counts.copy()
    history_before = engine.history.copy()
    h = hashlib.sha256()
    count = recent = 0
    seen = set()
    with feedback.open('rb') as f:
        while f.tell() < cutoff:
            line = f.readline()
            assert line and f.tell() <= cutoff, 'Cutoff must be at an exact committed line boundary'
            h.update(line)
            item = json.loads(line)
            step = int(item['control_step'])
            assert step <= limit_step, 'Future feedback cannot enter the fork'
            assert len(item['task_ids']) == len(item['rows']) == len(item['decision_ids'])
            count += len(item['rows'])
            if step < first_step:
                continue
            for task, row, serial in zip(item['task_ids'], item['rows'], item['decision_ids']):
                assert serial not in seen, 'Duplicate terminal decision'
                seen.add(serial)
                engine.bootstrap_record(int(task), row, step, int(serial))
                recent += 1
        assert f.tell() == cutoff
    assert count == info['adapter']['owner']['accepted']
    assert np.array_equal(samples_before, engine.sample_counts)
    assert np.array_equal(completed_before, engine.completed_counts)
    assert np.array_equal(history_before, engine.history)
    return dict(source_checkpoint=str(checkpoint), source_decision=str(source),
                source_feedback=str(feedback), source_feedback_prefix_bytes=cutoff,
                source_feedback_prefix_sha256=h.hexdigest(), committed_terminal_records=count,
                recent_records_imported=recent, first_allowed_control_step=first_step,
                last_allowed_control_step=limit_step, duplicate_history_updates=0)


def stratified_permutation(dynamic, fields, seed):
    """Shuffle complete feedback rows only among similar evidence/age strata."""
    rng = np.random.default_rng(seed)
    n = len(dynamic)
    strata = np.zeros(n, dtype=np.int64)
    multiplier = 1
    selected = []
    for name in fields:
        if ('mask' in name or 'observed' in name or name == 'recent') and len(selected) < 3:
            column = dynamic[:, fields.index(name)]
            bins = (column > .5).astype(np.int64)
            strata += multiplier*bins
            multiplier *= 2
            selected.append(name)
    for token in ('age', 'mass'):
        candidates = [name for name in fields if token in name]
        if candidates:
            name = candidates[0]
            values = dynamic[:, fields.index(name)]
            # Cutoffs depend only on this frozen pre-decision snapshot, not outcomes.
            boundaries = np.unique(np.quantile(values, [.25, .5, .75]))
            bins = np.searchsorted(boundaries, values, side='right')
            strata += multiplier*bins
            multiplier *= len(boundaries)+1
            selected.append(name)
    permutation = np.arange(n)
    for group in np.unique(strata):
        ids = np.flatnonzero(strata == group)
        permutation[ids] = rng.permutation(ids)
    assert np.array_equal(np.sort(permutation), np.arange(n))
    return permutation, dict(fields=selected, groups=int(len(np.unique(strata))),
                             moved_fraction=float(np.mean(permutation != np.arange(n))),
                             seed=int(seed), policy='Whole dynamic rows; same command and global state retained.')
