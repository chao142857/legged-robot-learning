import copy
import numpy as np
from nsgs.sampling.ownership import Ownership
from nsgs.sampling.neural import NeuralSampler


class NeuralOwner(Ownership):
    def __init__(self, pool, num_envs, seed):
        super().__init__(pool, num_envs, "uniform", seed)
        self.mode = "neural"
        self.engine = NeuralSampler(pool.n, seed)
        self.teacher_version = np.full(num_envs, -1, dtype=np.int64)
        self.decision_id = np.full(num_envs, -1, dtype=np.int64)
        self.selected_probability = np.zeros(num_envs)
        self.cycle_version_exposure = {}

    def assign(self, ids):
        selected = super().assign(ids)
        self.teacher_version[ids] = self.engine.version
        self.decision_id[ids] = self.engine.last_serials
        self.selected_probability[ids] = self.engine.probability[selected]
        self.engine.draws.extend(zip(self.decision_id[ids].tolist(), np.asarray(ids).tolist(),
            selected.tolist(), self.generation[ids].tolist(),
            [self.control_steps]*len(ids), self.selected_probability[ids].tolist()))
        return selected

    def advance(self):
        for version, count in zip(*np.unique(self.teacher_version[self.task>=0], return_counts=True)):
            self.cycle_version_exposure[int(version)] = self.cycle_version_exposure.get(int(version),0)+int(count)
        super().advance()

    def token(self, ids):
        return super().token(ids) | dict(teacher_versions=self.teacher_version[ids].copy(),
            decision_ids=self.decision_id[ids].copy(), selected_probabilities=self.selected_probability[ids].copy())

    def commit_rows(self, token, rows):
        ids = token["env_ids"]
        assert np.array_equal(self.teacher_version[ids], token["teacher_versions"])
        assert np.array_equal(self.decision_id[ids], token["decision_ids"])
        super().commit(token, rows[:,6].astype(bool).tolist())
        self.engine.observe(token["task_ids"], rows, self.control_steps)
        for serial, version, steps in zip(token["decision_ids"], token["teacher_versions"], token["steps"]):
            if version == self.engine.version:
                self.engine.draw_end_steps[int(serial)] = int(steps)

    def cycle_decisions(self):
        # One row per actual draw; completed and in-flight duration are distinct from likelihood weight.
        draws = np.array(self.engine.draws, dtype=np.float64).reshape(-1,6)
        active = {int(i):int(s) for i,s in zip(self.decision_id[self.task>=0], self.steps[self.task>=0])}
        exposure = np.array([self.engine.draw_end_steps.get(int(row[0]), active.get(int(row[0]), 0)) for row in draws],dtype=np.int64)
        counts = np.bincount(draws[:,2].astype(np.int64), minlength=self.pool.n)
        assert np.allclose(draws[:,5], self.engine.probability[draws[:,2].astype(np.int64)], atol=1e-15, rtol=0)
        assert len(np.unique(draws[:,0])) == len(draws)
        return draws, exposure, counts

    def state_dict(self):
        state = super().state_dict()
        for key in ("teacher_version", "decision_id", "selected_probability", "cycle_version_exposure"):
            state[key] = copy.deepcopy(getattr(self,key))
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for key in ("teacher_version", "decision_id", "selected_probability", "cycle_version_exposure"):
            setattr(self,key,copy.deepcopy(state[key]))

    def check(self):
        assert int(self.engine.completed_counts.sum()) == self.accepted
        assert self.control_steps*len(self.task) == int(self.transitions.sum())
        assert int(self.engine.sample_counts.sum()) == self.accepted+int(self.censored.sum())+int((self.task>=0).sum())
        assert np.all(self.teacher_version[self.task>=0] <= self.engine.version)
