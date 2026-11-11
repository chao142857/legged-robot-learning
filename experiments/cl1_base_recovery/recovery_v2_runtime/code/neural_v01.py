"""V0.1 CPU teacher: time-local feedback, separate encoders, past-only baseline.

Student/PPO and the V0 likelihood estimator are unchanged. This module stores only
real terminal measurements; an expired observation is unknown, never a success.
"""
import copy
import math
import numpy as np
import torch
from protocol import CONFIG, GLOBAL_FIELDS
from v0_core import Teacher as V0Teacher, NeuralSampler as V0Sampler
from v0_core import state_hash, distribution_stats, development_score, paired_gain


V01_CONFIG = dict(version="nsgs-v01-validation-20261106", window_updates=1000.,
                  half_life_updates=200., controls_per_update=24,
                  command_embedding=32, state_embedding=32,
                  teacher_seed=1047, ridge_l2=1., baseline_warmup=10,
                  baseline_max_records=50, no_old_policy_actor_replay=True)
DYNAMIC_FIELDS = [
    "feedback_observed", "valid_error_observed", "recent_feedback_observed", "recent_error_observed",
    "recent_strict_success", "recent_velocity_rmse_scaled", "recent_yaw_rmse_scaled",
    "recent_position_rmse_scaled", "recent_orientation_rmse_scaled", "recent_good_fraction",
    "recent_failure", "feedback_age_scaled", "valid_error_age_scaled",
    "recent_evidence_log_mass", "recent_effective_log_count", "recent_error_log_mass",
    "recent_error_effective_log_count", "log_sample_count", "log_complete_count",
    "log_transition_count", "history_fill", "last_good_fraction", "last_strict_success",
    "last_failure", "last_velocity_rmse_scaled", "last_yaw_rmse_scaled",
    "last_position_rmse_scaled", "last_orientation_rmse_scaled",
    "last_measurement_available", "last_error_measurement_available",
]
TOLERANCES = np.array([.2, .3, .05, np.pi/12], dtype=np.float64)


class StateBaseline:
    """Eight fixed-scale inputs, intercept, ridge fitted to previous cycles only."""
    def __init__(self):
        self.x = []
        self.y = []
        self.total_seen = 0
        self.ema = 0.
        self.previous_reward = 0.
        self.coefficients = np.zeros(9, dtype=np.float64)

    def features(self, global_state):
        g = np.asarray(global_state, dtype=np.float64)
        assert g.shape == (len(GLOBAL_FIELDS),) and np.isfinite(g).all()
        # Native, predeclared units; no fitting to current/future returns.
        return np.array([g[0], g[8], self.previous_reward, g[2], g[1],
                         np.clip(g[3], 0, 10)/10., g[4], g[5]], dtype=np.float64)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (8,) and np.isfinite(x).all()
        if self.total_seen < V01_CONFIG["baseline_warmup"]:
            return float(self.ema), "ema_warmup"
        return float(np.clip(np.r_[1., x] @ self.coefficients, -1., 1.)), "state_ridge"

    def observe(self, x, reward):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (8,) and np.isfinite(x).all() and math.isfinite(float(reward))
        self.x.append(x.copy())
        self.y.append(float(reward))
        self.x = self.x[-V01_CONFIG["baseline_max_records"]:]
        self.y = self.y[-V01_CONFIG["baseline_max_records"]:]
        self.total_seen += 1
        self.ema = CONFIG["baseline_beta"]*self.ema + (1-CONFIG["baseline_beta"])*float(reward)
        self.previous_reward = float(reward)
        X = np.column_stack((np.ones(len(self.x)), np.asarray(self.x)))
        penalty = np.eye(9)*V01_CONFIG["ridge_l2"]
        penalty[0, 0] = 0.  # free intercept; positive data count makes solve well-defined
        self.coefficients = np.linalg.solve(X.T @ X + penalty, X.T @ np.asarray(self.y))
        assert np.isfinite(self.coefficients).all()

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        assert set(state) == set(vars(self))
        for k, v in state.items():
            setattr(self, k, copy.deepcopy(v))
        assert self.coefficients.shape == (9,)


class TwoBranchNetwork(torch.nn.Module):
    def __init__(self, command_dim):
        super().__init__()
        self.command_dim = int(command_dim)
        self.command_encoder = torch.nn.Sequential(torch.nn.Linear(command_dim, 32),
                                                  torch.nn.LayerNorm(32), torch.nn.Tanh())
        self.state_encoder = torch.nn.Sequential(torch.nn.Linear(len(DYNAMIC_FIELDS)+len(GLOBAL_FIELDS), 32),
                                                torch.nn.LayerNorm(32), torch.nn.Tanh())
        self.head = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
        torch.nn.init.normal_(self.head[-1].weight, 0., .001)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        # Global features interact nonlinearly with sequence state, not a shared
        # additive logit offset (which softmax would cancel).
        command = self.command_encoder(x[:, :self.command_dim])
        state = self.state_encoder(x[:, self.command_dim:])
        return self.head(torch.cat((command, state), 1))


class Teacher(V0Teacher):
    def __init__(self, static, binding, student_seed, mode="fresh", inherited=None):
        # Initialize inherited accounting only; reconstructing both networks is
        # inside fork_rng so no Teacher operation changes Student RNG.
        super().__init__(static, binding, student_seed, mode="fresh", inherited=None)
        self.dim = self.static.shape[1]+len(DYNAMIC_FIELDS)+len(GLOBAL_FIELDS)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(V01_CONFIG["teacher_seed"])
            self.net = TwoBranchNetwork(self.static.shape[1])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=CONFIG["teacher_lr"])
        self.state_baseline = StateBaseline()
        if mode == "continual":
            assert inherited is not None and student_seed not in inherited["seen_student_seeds"]
            self.load(inherited)
            self.seen_student_seeds.append(int(student_seed))
            self.used_versions = set()
            self.baseline = self.reward_second = 0.
            self.state_baseline = StateBaseline()
        else:
            assert mode == "fresh" and inherited is None

    def probabilities(self, dynamic, global_state):
        d = torch.as_tensor(dynamic, dtype=torch.float32).detach()
        g = torch.as_tensor(global_state, dtype=torch.float32).detach()
        assert d.shape == (len(self.static), len(DYNAMIC_FIELDS)) and g.shape == (len(GLOBAL_FIELDS),)
        x = torch.cat((self.static, d, g.expand(len(d), -1)), 1)
        assert not x.requires_grad and torch.isfinite(x).all()
        logits = self.net(x).flatten().double()
        logits = CONFIG["logit_bound"]*torch.tanh(logits/CONFIG["logit_bound"])
        return CONFIG["uniform_mix"]/len(d)+(1-CONFIG["uniform_mix"])*torch.softmax(logits, 0)

    def decide(self, dynamic, global_state, version):
        decision = super().decide(dynamic, global_state, version)
        x = self.state_baseline.features(global_state)
        prediction, method = self.state_baseline.predict(x)
        decision.update(baseline_features=x.copy(), baseline_prediction=prediction,
                        baseline_method=method, baseline_prior_records=self.state_baseline.total_seen)
        return decision

    def learn(self, decision, counts, reward):
        assert decision["baseline_prior_records"] == self.state_baseline.total_seen, "Stale baseline snapshot"
        expected, method = self.state_baseline.predict(decision["baseline_features"])
        assert expected == decision["baseline_prediction"] and method == decision["baseline_method"]
        self.baseline = float(decision["baseline_prediction"])
        # V0 computes sum(actual_counts * log(actual_prob))/100, once per version.
        result = super().learn(decision, counts, reward)
        self.state_baseline.observe(decision["baseline_features"], reward)
        self.baseline = self.state_baseline.ema
        result.update(baseline_method=method, baseline_prior_records=decision["baseline_prior_records"],
                      baseline_fitted_records=self.state_baseline.total_seen,
                      baseline_prediction=expected, baseline_ema_after=self.baseline,
                      baseline_after=self.baseline,
                      branch_gradient_norms={name:math.sqrt(sum(float((p.grad.detach()**2).sum())
                          for p in module.parameters() if p.grad is not None))
                          for name, module in (("command",self.net.command_encoder),
                                               ("state",self.net.state_encoder),("head",self.net.head))})
        return result

    def state(self):
        result = super().state()
        result.update(v01_config=copy.deepcopy(V01_CONFIG), state_baseline=self.state_baseline.state_dict())
        return result

    def load(self, state):
        assert state.get("v01_config") == V01_CONFIG, "Teacher architecture/baseline config mismatch"
        super().load(state)
        self.state_baseline.load_state_dict(state["state_baseline"])


class NeuralSampler(V0Sampler):
    def __init__(self, n, seed):
        super().__init__(n, seed)
        self.recent_records = {}
        self.last_real_row = np.zeros((n, 9), dtype=np.float64)
        self.last_valid_real_row = np.zeros((n, 4), dtype=np.float64)
        self.last_real_row_step = np.full(n, -1, dtype=np.int64)
        self.last_valid_real_row_step = np.full(n, -1, dtype=np.int64)
        self.recent_bootstrap_ids = set()
        self.recent_schema = V01_CONFIG["version"]

    def _add_recent(self, sequence_id, row, step):
        i, step = int(sequence_id), int(step)
        row = np.asarray(row, dtype=np.float64)
        assert 0 <= i < self.n and step >= 0 and row.shape == (9,) and np.isfinite(row).all()
        assert row[0] >= 0 and 0 <= row[1] <= row[0] and row[6] in (0.,1.) and row[7] in (0.,1.)
        if i in self.recent_records:
            assert step >= self.recent_records[i][-1][0], "Feedback cannot travel backwards in time"
        cutoff = step-V01_CONFIG["window_updates"]*V01_CONFIG["controls_per_update"]
        recent = [(s,r) for s,r in self.recent_records.get(i,[]) if s >= cutoff]
        recent.append((step,row.copy()))
        self.recent_records[i] = recent
        self.last_real_row[i] = row
        self.last_real_row_step[i] = step
        if row[0] > 0:
            self.last_valid_real_row[i] = row[2:6]
            self.last_valid_real_row_step[i] = step

    def observe(self, ids, rows, step):
        super().observe(ids, rows, step)
        for i,row in zip(ids, rows):
            self._add_recent(i,row,step)

    def bootstrap_record(self, sequence_id, row, step, decision_id=None):
        """Populate new features from retained real logs, without recounting outcomes."""
        if decision_id is not None:
            key = int(decision_id)
            assert key not in self.recent_bootstrap_ids, "Duplicate logged episode"
            self.recent_bootstrap_ids.add(key)
        self._add_recent(sequence_id,row,step)

    def bootstrap_records(self, ids, rows, steps, decision_ids=None):
        assert len(ids) == len(rows) == len(steps)
        if decision_ids is None:
            decision_ids = [None]*len(ids)
        assert len(decision_ids) == len(ids)
        for i,row,step,serial in zip(ids,rows,steps,decision_ids):
            self.bootstrap_record(i,row,step,serial)

    def features(self, step, transitions):
        step = int(step)
        observed = self.last_feedback_step >= 0
        valid = self.last_valid_step >= 0
        assert np.all(self.last_feedback_step[observed] <= step)
        assert np.all(self.last_valid_step[valid] <= step)
        f = np.zeros((self.n,len(DYNAMIC_FIELDS)),dtype=np.float64)
        f[:,0],f[:,1] = observed,valid
        age = np.where(observed,(step-self.last_feedback_step)/24.,0.)
        valid_age = np.where(valid,(step-self.last_valid_step)/24.,0.)
        f[:,11] = np.log1p(age)/np.log1p(10000.)
        f[:,12] = np.log1p(valid_age)/np.log1p(10000.)
        f[:,17] = np.log1p(self.sample_counts)/np.log1p(10000.)
        f[:,18] = np.log1p(self.completed_counts)/np.log1p(10000.)
        f[:,19] = np.log1p(transitions)/np.log1p(10000000.)
        f[:,20] = self.history_size/100.
        last = self.last_real_row
        f[:,21] = np.divide(last[:,1],last[:,0],out=np.zeros(self.n),where=(last[:,0]>=100)&(last[:,7]==0))
        f[:,22],f[:,23] = last[:,6],last[:,7]
        f[:,24:28] = np.minimum(self.last_valid_real_row/TOLERANCES,10)
        f[:,28] = (self.last_real_row_step>=0)&(self.last_real_row_step==self.last_feedback_step)
        f[:,29] = (self.last_valid_real_row_step>=0)&(self.last_valid_real_row_step==self.last_valid_step)
        f[:,21:24] *= f[:,28,None]
        f[:,24:28] *= f[:,29,None]
        cutoff = step-V01_CONFIG["window_updates"]*24
        for i, records in list(self.recent_records.items()):
            assert records[-1][0] <= step, "Future feedback leakage"
            records = [(s,r) for s,r in records if s>=cutoff]
            if not records:
                del self.recent_records[i]
                continue
            self.recent_records[i] = records
            times = np.array([s for s,r in records],dtype=np.float64)
            rows = np.stack([r for s,r in records])
            w = np.exp2(-(step-times)/(24.*V01_CONFIG["half_life_updates"]))
            mass = w.sum(); ess = mass**2/(w@w)
            f[i,2] = 1.
            f[i,4] = (w@rows[:,6])/mass
            good = np.divide(rows[:,1],rows[:,0],out=np.zeros(len(rows)),where=(rows[:,0]>=100)&(rows[:,7]==0))
            f[i,9] = (w@good)/mass
            f[i,10] = (w@rows[:,7])/mass
            f[i,13:15] = np.log1p([mass,ess])/np.log1p(100.)
            vm = rows[:,0]>0
            if vm.any():
                vw = w[vm]; v_mass = vw.sum(); v_ess=v_mass**2/(vw@vw)
                f[i,3] = 1.
                f[i,5:9] = np.minimum((vw@rows[vm,2:6])/v_mass/TOLERANCES,10.)
                f[i,15:17] = np.log1p([v_mass,v_ess])/np.log1p(100.)
        assert np.isfinite(f).all()
        return f.astype(np.float32)

    def state_dict(self):
        state = super().state_dict()
        # fork_state.tree_hash hashes generic objects by repr. A set's iteration
        # order is not a serialization contract; checkpoint it as a sorted list.
        state["recent_bootstrap_ids"] = sorted(self.recent_bootstrap_ids)
        return state

    def load_state_dict(self, state):
        if "recent_schema" in state:
            assert state["recent_schema"] == V01_CONFIG["version"]
        else:
            self.recent_records = {}
            self.last_real_row.fill(0)
            self.last_valid_real_row.fill(0)
            self.last_real_row_step.fill(-1)
            self.last_valid_real_row_step.fill(-1)
            self.recent_bootstrap_ids = set()
        super().load_state_dict(state)
        self.recent_bootstrap_ids = set(self.recent_bootstrap_ids)


def shuffled_feedback(dynamic, seed):
    """Deterministically break sequence-feedback links within freshness strata.

    Keep exposure counts and ages on their real sequence. Shuffle joint outcome
    fields together, never individual metrics; preserve observed/valid masks.
    """
    d = np.asarray(dynamic,dtype=np.float32)
    assert d.ndim == 2 and d.shape[1] == len(DYNAMIC_FIELDS)
    result = d.copy()
    outcome_cols = [4,5,6,7,8,9,10,21,22,23,24,25,26,27]
    # Two coarse age levels, plus observed and evidence strata, avoid constructing
    # grossly incompatible feedback-age combinations. Empty/singleton bins stay.
    keys = np.column_stack((d[:,:4].astype(np.int64),d[:,28:30].astype(np.int64),
                            np.floor(np.clip(d[:,11],0,1)*4).astype(np.int64),
                            np.floor(np.clip(d[:,13],0,2)*4).astype(np.int64)))
    _,bins = np.unique(keys,axis=0,return_inverse=True)
    rng = np.random.default_rng(seed)
    for k in np.unique(bins):
        ix=np.flatnonzero(bins==k)
        result[np.ix_(ix,outcome_cols)] = d[np.ix_(rng.permutation(ix),outcome_cols)]
    return result
