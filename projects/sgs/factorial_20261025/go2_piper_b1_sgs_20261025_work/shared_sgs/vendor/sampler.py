"""SGS Eq.1/2 and Appendix B.2, arXiv:2610.12465v1. CPU, serial reset order."""
import math
import random


class TaskSampler:
    def __init__(self, n, seed, mode="uniform", history=100, target=.5,
                 concentration=1., temperature=2., epsilon=1.e-4):
        if n < 1 or mode not in ("uniform", "sgs", "evaluation"):
            raise ValueError("Invalid sampler size/mode")
        if history < 1 or not 0 <= target <= 1 or concentration < 0 or epsilon <= 0:
            raise ValueError("Invalid SGS hyperparameters")
        self.n, self.mode = n, mode
        self.history, self.target, self.concentration = history, target, concentration
        self.temperature, self.epsilon = max(temperature, 1.), epsilon
        self.rng = random.Random(seed)
        self.outcomes = [[] for _ in range(n)]
        self.sample_counts = [0] * n
        self.completed_counts = [0] * n
        self.success_counts = [0] * n
        self.eval_cursor = 0

    def rates(self):
        # Empty history is a documented neutral p=.5, no fabricated trials.
        # Thus ALL unseen tasks initially have equal weight, also for t != .5.
        return [sum(h) / len(h) if h else .5 for h in self.outcomes]

    def kernel_weights(self):
        t, k, e = self.target, self.concentration, self.epsilon
        return [(p+e)**(k*t) * (1-p+e)**(k*(1-t)) for p in self.rates()]

    def probabilities(self):
        if self.mode != "sgs":
            return [1. / self.n] * self.n
        # Appendix B.2 floors BEFORE forming log(weight + epsilon).
        logits = [math.log(max(w, self.epsilon) + self.epsilon) / self.temperature
                  for w in self.kernel_weights()]
        weights = [math.exp(l-max(logits)) for l in logits]
        return [w / sum(weights) for w in weights]

    def sample(self, count):
        if self.mode == "evaluation":
            ids = [(self.eval_cursor+i) % self.n for i in range(count)]
            self.eval_cursor += count
        else:
            ids = self.rng.choices(range(self.n), weights=self.probabilities(), k=count)
        for i in ids:
            self.sample_counts[i] += 1
        return ids

    def record(self, task_ids, successes):
        if len(task_ids) != len(successes):
            raise ValueError("Task/outcome lengths differ")
        for i, success in zip(task_ids, successes):
            if not 0 <= i < self.n or success not in (True, False, 0, 1):
                raise ValueError("Invalid task or binary outcome")
            self.completed_counts[i] += 1
            self.success_counts[i] += int(success)
            self.outcomes[i].append(int(success))
            self.outcomes[i] = self.outcomes[i][-self.history:]

    def state_dict(self):
        return dict(version=1, n=self.n, mode=self.mode, history=self.history,
                    target=self.target, concentration=self.concentration,
                    temperature=self.temperature, epsilon=self.epsilon,
                    empty_history_prior=.5, outcomes=self.outcomes,
                    sample_counts=self.sample_counts, completed_counts=self.completed_counts,
                    success_counts=self.success_counts, eval_cursor=self.eval_cursor,
                    rng_state=self.rng.getstate())

    def load_state_dict(self, state):
        for k in ("version", "n", "mode", "history", "target", "concentration", "temperature", "epsilon"):
            if self.state_dict()[k] != state[k]:
                raise ValueError("Incompatible sampler state: " + k)
        for key in ("outcomes", "sample_counts", "completed_counts", "success_counts"):
            if len(state[key]) != self.n:
                raise ValueError("Invalid state size")
            setattr(self, key, [list(h) for h in state[key]] if key == "outcomes" else list(state[key]))
        self.eval_cursor = state["eval_cursor"]
        def tuples(x):
            return tuple(tuples(v) for v in x) if isinstance(x, (list, tuple)) else x
        self.rng.setstate(tuples(state["rng_state"]))

    def summary(self):
        p = self.probabilities()
        return dict(mode=self.mode, rates=self.rates(), kernel_weights=self.kernel_weights(),
                    probabilities=p, sampled=self.sample_counts[:], completed=self.completed_counts[:],
                    successes=self.success_counts[:], history_lengths=[len(h) for h in self.outcomes],
                    entropy=-sum(x*math.log(x) for x in p),
                    effective_tasks=1/sum(x*x for x in p), empty_history_prior=.5)
