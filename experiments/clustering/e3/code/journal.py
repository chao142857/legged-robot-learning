"""Append-only episode provenance and exact last-100 rebuilding on remap."""
import numpy as np

from grouped_sampling import GroupedSampler, mapping_hash


class JournalSampler(GroupedSampler):
    def __init__(self, mapping, pool_identity, seed=42, **parameters):
        super().__init__(mapping, pool_identity, seed, **parameters)
        self._journal = np.empty((0, 5), dtype=np.int32)
        self.journal_size = 0

    @property
    def journal(self):
        # Columns: sequence, strict success, end control step, env, generation.
        view = self._journal[:self.journal_size].view()
        view.flags.writeable = False
        return view

    def record(self, ids, successes, *, control_step=None, event_ids=None):
        ids, successes = list(ids), list(successes)
        if control_step is None or event_ids is None:
            raise ValueError("E3 feedback requires native terminal time and episode identity")
        if not ids:
            return
        events = np.asarray(event_ids)
        if events.shape != (len(ids), 2) or len(ids) != len(successes):
            raise ValueError("Mismatched terminal batch")
        rows = np.column_stack((ids, successes, np.full(len(ids), control_step), events))
        if not np.issubdtype(rows.dtype, np.integer) or not np.array_equal(rows.astype(np.int32), rows):
            raise ValueError("Invalid episode provenance")
        new_size = self.journal_size + len(ids)
        if len(self._journal) < new_size:
            expanded = np.empty((max(4096, new_size, 2 * len(self._journal)), 5), dtype=np.int32)
            expanded[:self.journal_size] = self.journal
            self._journal = expanded
        super().record(ids, successes, control_step=control_step, event_ids=event_ids)
        self._journal[self.journal_size:new_size] = rows
        self.journal_size = new_size

    def _rebuilt(self, mapping):
        params = {key: getattr(self, key) for key in ("history", "target", "concentration", "temperature", "epsilon")}
        other = GroupedSampler(mapping, self.pool_identity, **params)
        if other.n != self.n or other.groups != self.groups or not np.array_equal(other.sizes, self.sizes):
            raise ValueError("Remap changed pool size or group capacities")
        rows = self.journal
        group_ids = other.mapping[rows[:, 0]]
        order = np.argsort(group_ids, kind="stable")
        counts = np.bincount(group_ids, minlength=other.groups)
        other.group_completed = counts
        other.group_successes = np.bincount(group_ids, weights=rows[:, 1], minlength=other.groups).astype(np.int64)
        end = 0
        for group, count in enumerate(counts):
            start, end = end, end + int(count)
            window = rows[order[max(start, end - self.history):end]]
            other.window_members[group] = window[:, 0].tolist()
            other.outcomes[group] = window[:, 1].tolist()
            other.window_successes[group] = window[:, 1].sum()
            other.window_steps[group] = window[:, 2].tolist()
            other.window_events[group] = list(map(tuple, window[:, 3:].tolist()))
        return other

    def switch(self, mapping):
        if mapping_hash(mapping) == self.mapping_sha256:
            return False
        other = self._rebuilt(mapping)
        for key in (
            "mapping", "mapping_sha256", "members", "sizes", "group_completed", "group_successes",
            "outcomes", "window_successes", "window_steps", "window_members", "window_events",
        ):
            setattr(self, key, getattr(other, key))
        return True

    def state_dict(self):
        state = super().state_dict()
        state["journal_version"] = 1
        state["episode_journal"] = self.journal.copy()
        return state

    def load_state_dict(self, state):
        rows = np.asarray(state["episode_journal"])
        if state["journal_version"] != 1 or rows.dtype != np.int32 or rows.ndim != 2 or rows.shape[1] != 5:
            raise ValueError("Invalid episode journal format")
        seq, label, step, env, generation = rows.T
        if ((seq < 0) | (seq >= self.n) | (label < 0) | (label > 1) |
                (step < 0) | (env < 0) | (generation < 1)).any() or (np.diff(step) < 0).any():
            raise ValueError("Invalid episode journal values")
        if len(rows) and int(step[-1]) != state["last_control_step"]:
            raise ValueError("Episode journal timestamp mismatch")
        if not np.array_equal(np.bincount(seq, minlength=self.n), state["completed_counts"]):
            raise ValueError("Episode journal count mismatch")
        if not np.array_equal(np.bincount(seq, weights=label, minlength=self.n), state["success_counts"]):
            raise ValueError("Episode journal success count mismatch")
        _, reverse_index = np.unique(env[::-1], return_index=True)
        tail = rows[len(rows) - 1 - reverse_index]
        if dict(zip(tail[:, 3].tolist(), tail[:, 4].tolist())) != state["last_generations"]:
            raise ValueError("Episode journal generation mismatch")
        old = self._journal, self.journal_size
        self._journal, self.journal_size = rows.copy(), len(rows)
        try:
            rebuilt = self._rebuilt(self.mapping)
            for key in ("outcomes", "window_steps", "window_members", "window_events"):
                if getattr(rebuilt, key) != state[key]:
                    raise ValueError(f"Episode journal does not reproduce {key}")
            # Avoid copying the entire journal a second time in the parent validator.
            compact = {k: v for k, v in state.items() if k != "episode_journal"}
            super().load_state_dict(compact)
        except BaseException:
            self._journal, self.journal_size = old
            raise
