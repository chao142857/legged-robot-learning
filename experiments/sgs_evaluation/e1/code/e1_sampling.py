"""E1: a shared window of real episodes, with three simultaneous binary labels."""
import copy
import numpy as np
from sampling import CachedSampler, Ownership

SCHEDULE = (
    dict(first=1, last=4000, position_m=.10, joint_fraction=.50),
    dict(first=4001, last=8000, position_m=.075, joint_fraction=.60),
    dict(first=8001, last=10000, position_m=.05, joint_fraction=.80),
)

def stage_for(update):
    assert isinstance(update, int) and 1 <= update <= 10000
    return 0 if update <= 4000 else (1 if update <= 8000 else 2)

def rollout_update(completed, short=False):
    u = completed + 1
    # Short only: exercise both exact formal boundaries using native 24-step rollouts.
    return (u + (3950 if u <= 100 else 7850)) if short else u

class E1Sampler(CachedSampler):
    def __init__(self, n, seed=42):
        super().__init__(n, seed, 'sgs')
        self.active_stage = 0
        self.labels = np.zeros((n, self.history, 3), dtype=np.uint8)
        self.episode_ids = np.zeros((n, self.history), dtype=np.uint64)
        self.head = np.zeros(n, dtype=np.int64)
        self.label_success_counts = np.zeros((n, 3), dtype=np.int64)

    def window(self, i):
        size = min(self.completed_counts[i], self.history)
        indices = (np.arange(size) + self.head[i] - size) % self.history
        return self.labels[i, indices], self.episode_ids[i, indices]

    def column_rates(self, stage):
        length = np.minimum(self.completed_counts, self.history)
        sums = self.labels[:, :, stage].sum(1, dtype=np.int64)
        return np.divide(sums, length, out=np.full(self.n, .5), where=length > 0)

    def set_stage(self, stage):
        assert stage in (0, 1, 2)
        if stage == self.active_stage:
            return
        self.active_stage = stage
        self.outcomes = [self.window(i)[0][:, stage].tolist() for i in range(self.n)]
        self.cached_rates = self.column_rates(stage)
        self.success_counts = self.label_success_counts[:, stage].tolist()

    def record(self, *args, **kwargs):
        raise RuntimeError('E1 requires all three labels and a real episode identity')

    def record_multi(self, ids, labels, episode_ids):
        y = np.asarray(labels)
        assert y.shape == (len(ids), 3) and len(ids) == len(episode_ids)
        assert np.isin(y, [0, 1]).all() and (y[:, 0] >= y[:, 1]).all() and (y[:, 1] >= y[:, 2]).all()
        for i, row, eid in zip(ids, y, episode_ids):
            assert 0 <= i < self.n and 0 < int(eid) < 2**64
            assert int(eid) not in self.window(i)[1], 'Repeated episode in shared window'
            h = self.head[i]
            self.labels[i, h] = row
            self.episode_ids[i, h] = eid
            self.head[i] = (h + 1) % self.history
            self.label_success_counts[i] += row
        # Call the unchanged core/cache record directly, once per actual ended episode.
        super().record(list(ids), y[:, self.active_stage].tolist())

    def state_dict(self):
        s = super().state_dict()
        s['e1'] = dict(version=1, active_stage=self.active_stage, labels=self.labels.copy(),
                       episode_ids=self.episode_ids.copy(), head=self.head.copy(),
                       label_success_counts=self.label_success_counts.copy(), schedule=SCHEDULE)
        return copy.deepcopy(s)

    def load_state_dict(self, s):
        x = s['e1']; assert x['version'] == 1 and x['schedule'] == SCHEDULE
        super().load_state_dict(s)
        self.active_stage = x['active_stage']
        for k in ('labels', 'episode_ids', 'head', 'label_success_counts'):
            setattr(self, k, np.array(x[k], copy=True))
        assert self.labels.shape == (self.n, self.history, 3)
        assert np.all(self.labels[:, :, 0] >= self.labels[:, :, 1]) and np.all(self.labels[:, :, 1] >= self.labels[:, :, 2])
        assert self.outcomes == [self.window(i)[0][:, self.active_stage].tolist() for i in range(self.n)]
        np.testing.assert_array_equal(self.cached_rates, self.column_rates(self.active_stage))
        assert self.success_counts == self.label_success_counts[:, self.active_stage].tolist()

    def mechanism(self, stage=None, full=False):
        stage = self.active_stage if stage is None else stage
        rates = self.column_rates(stage)
        # Exactly the original cached formula, evaluated without changing sampler or RNG.
        e = self.epsilon
        w = (rates + e)**(self.concentration*self.target) * (1-rates+e)**(self.concentration*(1-self.target))
        logits = np.log(np.maximum(w, e)+e)/self.temperature
        p = np.exp(logits-logits.max()); p /= p.sum()
        sizes = np.minimum(self.completed_counts, self.history); valid = sizes > 0
        r = rates[valid]
        def frac(mask): return float(mask.mean()) if len(mask) else None
        result = dict(stage=stage+1, position_m=SCHEDULE[stage]['position_m'], joint_fraction=SCHEDULE[stage]['joint_fraction'],
                      sequences=self.n, empty_histories=int((~valid).sum()), histories=int(valid.sum()),
                      all_zero_fraction_among_nonempty=frac(r == 0), all_one_fraction_among_nonempty=frac(r == 1),
                      near_40_60_fraction_among_nonempty=frac((r >= .4) & (r <= .6)),
                      history_lengths_values_counts={str(int(k)):int(v) for k,v in zip(*np.unique(sizes, return_counts=True))},
                      rate_histogram_edges=np.linspace(0,1,21).tolist(), rate_histogram_counts=np.histogram(r,bins=np.linspace(0,1,21))[0].tolist(),
                      top10_probability_mass=float(np.sort(p)[-int(np.ceil(self.n*.1)):].sum()),
                      effective_sequences=float(1/(p*p).sum()))
        if full: result.update(rates=rates.tolist(), probabilities=p.tolist(), history_lengths=sizes.tolist())
        return result

class E1Ownership(Ownership):
    def __init__(self, pool, num_envs, mode='sgs', seed=42):
        assert mode == 'sgs'
        super().__init__(pool, num_envs, mode, seed)
        self.engine = E1Sampler(pool.n, seed)

    def commit(self, token, labels):
        assert token['pool_identity'] == self.pool.identity, 'Foreign pool feedback'
        ids = np.asarray(token['env_ids'])
        assert len(np.unique(ids)) == len(ids)
        assert np.array_equal(self.task[ids], token['task_ids']) and np.array_equal(self.generation[ids], token['generations'])
        assert np.array_equal(self.steps[ids], token['steps']) and np.all(self.steps[ids] > 0)
        assert np.all(self.closed_generation[ids] != self.generation[ids]), 'Duplicate terminal feedback'
        assert np.all(self.generation[ids] < 2**32) and np.all(ids < 2**32)
        eids = (self.generation[ids].astype(np.uint64) << np.uint64(32)) | ids.astype(np.uint64)
        self.engine.record_multi(self.task[ids].tolist(), labels, eids)
        self.accepted += len(ids)
        self.closed_generation[ids] = self.generation[ids]; self.task[ids] = -1
