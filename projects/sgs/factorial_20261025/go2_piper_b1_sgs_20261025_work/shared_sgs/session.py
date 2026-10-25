"""Episode ownership and state around the unchanged, vendored SGS algorithm.

One session per training arm and seed. All terminal rows from a simulator step
must be committed together, before assigning any reset environment a new task.
"""
from dataclasses import dataclass, asdict
import copy
import math

from .catalog import digest
from .vendor.sampler import TaskSampler

PARAMETERS = dict(history=100, target=.5, concentration=1., temperature=2., epsilon=1e-4)
STATE_VERSION = 1


def nonnegative_int(x):
    return type(x) is int and x >= 0


@dataclass(frozen=True)
class EpisodeAssignment:
    run_id: str
    env_id: int
    episode_id: int
    task_id: str
    content_sha256: str
    support_sha256: str
    support_version: str


@dataclass(frozen=True)
class TerminalFeedback:
    assignment: EpisodeAssignment
    terminated: bool
    timeout: bool
    success: bool
    steps: int
    feedback_rule_version: str
    metrics: dict
    termination_reasons: tuple = ()

    @property
    def truncated(self):
        return self.timeout


class SamplerSession:
    def __init__(self, pool, *, run_id, seed, feedback_rule_version, mode="sgs"):
        if mode not in ("sgs", "uniform") or not run_id or not feedback_rule_version:
            raise ValueError("Training mode, run identity and frozen feedback rule required")
        if type(seed) is not int:
            raise ValueError("SGS uses a separate explicit integer RNG seed")
        self.pool, self.run_id, self.seed = pool, run_id, seed
        self.mode, self.feedback_rule_version = mode, feedback_rule_version
        self.engine = TaskSampler(len(pool.tasks), seed, mode=mode, **PARAMETERS)
        self.active, self.generations, self.closed = {}, {}, {}
        self.tick = 0
        self.active_counts = [0] * len(pool.tasks)
        self.transitions = [0] * len(pool.tasks)
        self.censored_counts = [0] * len(pool.tasks)
        self.censored_transitions = [0] * len(pool.tasks)
        self.completed_transitions = [0] * len(pool.tasks)

    def probabilities(self):
        p = self.engine.probabilities()
        if any(not math.isfinite(x) or x <= 0 for x in p) or not math.isclose(sum(p), 1., abs_tol=1e-12):
            raise ValueError("Invalid SGS probabilities")
        return p

    def _ids(self, env_ids):
        ids = list(env_ids)
        if len(set(ids)) != len(ids) or any(not nonnegative_int(x) for x in ids):
            raise ValueError("Environment slots must be unique nonnegative integers")
        return sorted(ids)

    def assign(self, env_ids, *, initial_task_ids=None):
        ids = self._ids(env_ids)
        if any(i in self.active for i in ids):
            raise ValueError("Old episode must be closed/censored before reassignment")
        if not ids:
            return []
        if initial_task_ids is not None:
            # Optional balanced CP0 reproduces the historical initial commands.
            # It consumes no sampler or policy RNG, and is identical in the pair.
            if self.generations or self.engine.sample_counts != [0] * len(self.pool.tasks):
                raise ValueError("Explicit assignments are only permitted at CP0")
            if set(initial_task_ids) != set(ids) or any(t not in self.pool.index for t in initial_task_ids.values()):
                raise ValueError("Initial assignments must exactly cover the supplied slots")
            chosen = [self.pool.index[initial_task_ids[i]] for i in ids]
            for k in chosen:
                self.engine.sample_counts[k] += 1
        else:
            self.probabilities()
            chosen = self.engine.sample(len(ids))
        result = []
        for i, k in zip(ids, chosen):
            generation = self.generations.get(i, -1) + 1
            token = EpisodeAssignment(self.run_id, i, generation, self.pool.ids[k],
                                      self.pool.tasks[k].content_sha256, self.pool.support_sha256,
                                      self.pool.support_version)
            self.generations[i] = generation
            self.active[i] = dict(assignment=asdict(token), start_tick=self.tick, extra_steps=0)
            self.active_counts[k] += 1
            result.append(token)
        return result

    def assignment(self, env_id):
        return EpisodeAssignment(**self.active[env_id]["assignment"])

    def episode_steps(self, env_id):
        row = self.active[env_id]
        return self.tick - row["start_tick"] + row["extra_steps"]

    def advance_all(self, steps=1):
        """O(number of tasks), includes each old task's terminal transition."""
        if not nonnegative_int(steps):
            raise ValueError("Invalid transition increment")
        self.tick += steps
        for k, count in enumerate(self.active_counts):
            self.transitions[k] += steps * count

    def advance(self, env_ids, steps=1):
        """Partial stepping, useful for CPU adapters without synchronous stepping."""
        ids = self._ids(env_ids)
        if not nonnegative_int(steps) or any(i not in self.active for i in ids):
            raise ValueError("Invalid partial step")
        for i in ids:
            self.active[i]["extra_steps"] += steps
            self.transitions[self.pool.index[self.active[i]["assignment"]["task_id"]]] += steps

    def _validate_feedback(self, row):
        if not isinstance(row, TerminalFeedback) or not isinstance(row.assignment, EpisodeAssignment):
            raise ValueError("Typed terminal feedback required")
        a = row.assignment
        if not nonnegative_int(a.env_id) or not nonnegative_int(a.episode_id):
            raise ValueError("Invalid episode token")
        if (a.run_id != self.run_id or a.support_sha256 != self.pool.support_sha256
                or a.support_version != self.pool.support_version
                or a.task_id not in self.pool.index
                or a.content_sha256 != self.pool.tasks[self.pool.index[a.task_id]].content_sha256):
            raise ValueError("Foreign run/task/support feedback")
        if row.feedback_rule_version != self.feedback_rule_version:
            raise ValueError("Feedback rule mismatch")
        if not isinstance(row.metrics, dict) or not all(isinstance(x, str) for x in row.termination_reasons):
            raise ValueError("Physical metrics and termination reasons required")
        if any(type(v) is not bool for v in (row.terminated, row.timeout, row.success)):
            raise ValueError("Boolean terminal flags and success required")
        if not (row.terminated or row.timeout) or (row.terminated and row.success):
            raise ValueError("Partial episode or true failure labelled successful")
        if not nonnegative_int(row.steps) or row.steps == 0:
            raise ValueError("Terminal feedback needs actual steps")
        fingerprint = digest(asdict(row))
        old = self.closed.get(a.env_id)
        if old and old["assignment"] == asdict(a):
            if old["fingerprint"] != fingerprint:
                raise ValueError("Conflicting duplicate feedback")
            return False, fingerprint
        if a.env_id not in self.active or self.active[a.env_id]["assignment"] != asdict(a):
            raise ValueError("Stale token or incorrect old task ownership")
        if row.steps != self.episode_steps(a.env_id):
            raise ValueError("Terminal duration does not match recorded transitions")
        return True, fingerprint

    def observe_terminal(self, feedbacks):
        rows = sorted(list(feedbacks), key=lambda r: (r.assignment.env_id, r.assignment.episode_id))
        keys = [(r.assignment.env_id, r.assignment.episode_id) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate terminal inside batch; no feedback committed")
        # Validate the whole step before any mutation: invalid batches are atomic.
        checked = [self._validate_feedback(r) for r in rows]
        accepted = 0
        for row, (new, fingerprint) in zip(rows, checked):
            if not new:
                continue
            a = row.assignment
            k = self.pool.index[a.task_id]
            self.engine.record([k], [row.success])
            self.completed_transitions[k] += row.steps
            self.active_counts[k] -= 1
            del self.active[a.env_id]
            self.closed[a.env_id] = dict(assignment=asdict(a), fingerprint=fingerprint, kind="terminal")
            accepted += 1
        return dict(accepted=accepted, duplicates=len(rows)-accepted)

    def censor_active(self, reason, env_ids=None):
        if reason not in ("training_cutoff", "manual_reset", "interruption", "resume_without_physics"):
            raise ValueError("Censor reason must explicitly denote an unfinished episode")
        ids = self._ids(self.active if env_ids is None else env_ids)
        if any(i not in self.active for i in ids):
            raise ValueError("Only active episodes can be censored")
        rows = []
        for i in ids:
            a = self.assignment(i)
            k, steps = self.pool.index[a.task_id], self.episode_steps(i)
            self.censored_counts[k] += 1
            self.censored_transitions[k] += steps
            self.active_counts[k] -= 1
            self.closed[i] = dict(assignment=asdict(a), fingerprint=None, kind=reason)
            rows.append(dict(assignment=asdict(a), reason=reason, steps=steps))
            del self.active[i]
        return rows

    def state_dict(self):
        return copy.deepcopy(dict(version=STATE_VERSION, run_id=self.run_id, seed=self.seed,
            mode=self.mode, support_sha256=self.pool.support_sha256, parameters=PARAMETERS,
            feedback_rule_version=self.feedback_rule_version, engine=self.engine.state_dict(),
            active=self.active, generations=self.generations, closed=self.closed, tick=self.tick,
            active_counts=self.active_counts, transitions=self.transitions,
            censored_counts=self.censored_counts, censored_transitions=self.censored_transitions,
            completed_transitions=self.completed_transitions))

    def load_state_dict(self, state):
        """Validate in a temporary session and commit atomically; JSON supported."""
        expected = self.state_dict()
        for key in ("version", "run_id", "seed", "mode", "support_sha256", "parameters", "feedback_rule_version"):
            if state.get(key) != expected[key]:
                raise ValueError("Incompatible sampler state: " + key)
        candidate = SamplerSession(self.pool, run_id=self.run_id, seed=self.seed,
                                   feedback_rule_version=self.feedback_rule_version, mode=self.mode)
        e = copy.deepcopy(state["engine"])
        n = len(self.pool.tasks)
        for key in ("sample_counts", "completed_counts", "success_counts"):
            if len(e[key]) != n or any(not nonnegative_int(x) for x in e[key]):
                raise ValueError("Invalid engine counters")
        if len(e["outcomes"]) != n or e.get("eval_cursor") != 0 or e.get("empty_history_prior") != .5:
            raise ValueError("Invalid training history metadata")
        for k, h in enumerate(e["outcomes"]):
            if (any(type(x) is not int or x not in (0, 1) for x in h)
                    or len(h) != min(PARAMETERS["history"], e["completed_counts"][k])
                    or not sum(h) <= e["success_counts"][k] <= e["completed_counts"][k] - len(h) + sum(h)
                    or e["completed_counts"][k] > e["sample_counts"][k]):
                raise ValueError("Invalid outcome history")
        candidate.engine.load_state_dict(e)
        for key in ("active_counts", "transitions", "censored_counts", "censored_transitions", "completed_transitions"):
            value = state[key]
            if len(value) != n or any(not nonnegative_int(x) for x in value):
                raise ValueError("Invalid accounting counters")
            setattr(candidate, key, value[:])
        if not nonnegative_int(state["tick"]):
            raise ValueError("Invalid tick")
        candidate.tick = state["tick"]
        for key in ("active", "generations", "closed"):
            value = state[key]
            converted = {int(k): copy.deepcopy(v) for k, v in value.items()}
            if len(converted) != len(value) or any(not nonnegative_int(k) for k in converted):
                raise ValueError("Invalid environment slots")
            setattr(candidate, key, converted)
        if any(not nonnegative_int(v) for v in candidate.generations.values()):
            raise ValueError("Invalid generation")
        counts, active_steps = [0]*n, [0]*n
        for i, row in candidate.active.items():
            a = EpisodeAssignment(**row["assignment"])
            if (a.env_id != i or a.run_id != self.run_id or a.support_sha256 != self.pool.support_sha256
                    or a.support_version != self.pool.support_version
                    or a.task_id not in self.pool.index or a.episode_id != candidate.generations.get(i)
                    or not nonnegative_int(row["start_tick"]) or row["start_tick"] > candidate.tick
                    or not nonnegative_int(row["extra_steps"])):
                raise ValueError("Invalid active episode")
            k = self.pool.index[a.task_id]
            if a.content_sha256 != self.pool.tasks[k].content_sha256:
                raise ValueError("Invalid active content hash")
            counts[k] += 1
            active_steps[k] += candidate.episode_steps(i)
        for i, row in candidate.closed.items():
            a = EpisodeAssignment(**row["assignment"])
            if (a.env_id != i or a.run_id != self.run_id or a.support_sha256 != self.pool.support_sha256
                    or a.support_version != self.pool.support_version
                    or a.task_id not in self.pool.index or not nonnegative_int(a.episode_id)
                    or a.episode_id > candidate.generations.get(i, -1)
                    or a.content_sha256 != self.pool.tasks[self.pool.index[a.task_id]].content_sha256
                    or row["kind"] not in ("terminal", "training_cutoff", "manual_reset", "interruption", "resume_without_physics")
                    or (row["kind"] == "terminal" and (not isinstance(row["fingerprint"], str) or len(row["fingerprint"]) != 64))):
                raise ValueError("Invalid closed episode")
        if set(candidate.generations) != set(candidate.active) | set(candidate.closed):
            raise ValueError("Generation ownership mismatch")
        if sum(g+1 for g in candidate.generations.values()) != sum(e['sample_counts']):
            raise ValueError("Generation/sample count mismatch")
        for i, generation in candidate.generations.items():
            closed = candidate.closed.get(i)
            if i in candidate.active:
                if generation > 0 and (not closed or closed['assignment']['episode_id'] != generation-1):
                    raise ValueError("Previous generation not closed")
            elif not closed or closed['assignment']['episode_id'] != generation:
                raise ValueError("Last generation not closed")
        if counts != candidate.active_counts:
            raise ValueError("Active task counts mismatch")
        for k in range(n):
            if (candidate.transitions[k] != active_steps[k] + candidate.completed_transitions[k] + candidate.censored_transitions[k]
                    or e["sample_counts"][k] != counts[k] + e["completed_counts"][k] + candidate.censored_counts[k]):
                raise ValueError("Exposure/episode accounting mismatch")
        candidate.probabilities()
        self.__dict__.update(candidate.__dict__)

    def summary(self):
        return dict(run_id=self.run_id, mode=self.mode, support_sha256=self.pool.support_sha256,
                    parameters=copy.deepcopy(PARAMETERS), feedback_rule_version=self.feedback_rule_version,
                    tasks=[dict(task_id=t.task_id, content_sha256=t.content_sha256,
                                probability=self.probabilities()[k], sampled=self.engine.sample_counts[k],
                                completed=self.engine.completed_counts[k], successes=self.engine.success_counts[k],
                                censored=self.censored_counts[k], active=self.active_counts[k],
                                transitions=self.transitions[k]) for k, t in enumerate(self.pool.tasks)])


class FixedEvaluation:
    """Fresh, stateless fixed protocol; has no access to a training session."""
    def __init__(self, pool):
        self.pool = pool

    def assignments(self, env_ids):
        ids = list(env_ids)
        if len(ids) != len(set(ids)) or any(not nonnegative_int(i) for i in ids):
            raise ValueError("Invalid evaluation environment slots")
        return {i: self.pool.ids[i % len(self.pool.tasks)] for i in ids}

    def observe_terminal(self, *args, **kwargs):
        raise RuntimeError("Fixed evaluation cannot feed SGS")
