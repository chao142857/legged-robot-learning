"""CPU-only teacher and feedback state. No Isaac Lab/PPO imports or autograd paths."""
import copy
import hashlib
import math
import numpy as np
import torch
from protocol import CONFIG, DYNAMIC_FIELDS, GLOBAL_FIELDS


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def distribution_stats(p):
    p = np.asarray(p, dtype=np.float64)
    assert np.all(np.isfinite(p)) and np.all(p >= 0) and abs(p.sum()-1) < 1e-10
    return dict(entropy=float(-(p*np.log(p)).sum()), effective_sequences=float(1/(p@p)),
                min=float(p.min()), max=float(p.max()), tv_uniform=float(abs(p-1/len(p)).sum()/2),
                top10_mass=float(np.sort(p)[-max(1, len(p)//10):].sum()))


class Teacher:
    def __init__(self, static, binding, student_seed, mode="fresh", inherited=None):
        self.static = torch.as_tensor(static, dtype=torch.float32).detach().clone()
        self.binding = copy.deepcopy(binding)
        self.dim = self.static.shape[1] + len(DYNAMIC_FIELDS) + len(GLOBAL_FIELDS)
        # Preserve student CPU RNG; no teacher random operations use student RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(CONFIG["teacher_seed"])
            self.net = torch.nn.Sequential(torch.nn.Linear(self.dim, 64), torch.nn.Tanh(),
                                           torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
            torch.nn.init.normal_(self.net[-1].weight, 0., .001)
            torch.nn.init.zeros_(self.net[-1].bias)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=CONFIG["teacher_lr"])
        self.baseline = 0.
        self.reward_second = 0.
        self.updates = 0
        self.seen_student_seeds = [int(student_seed)]
        self.used_versions = set()
        if mode == "continual":
            assert inherited is not None and student_seed not in inherited["seen_student_seeds"]
            self.load(inherited)
            self.seen_student_seeds.append(int(student_seed))
            # Same optimizer, new on-policy run. No old decisions reused.
            self.used_versions = set()
            self.baseline = 0.
            self.reward_second = 0.
        else:
            assert mode == "fresh" and inherited is None

    def probabilities(self, dynamic, global_state):
        d = torch.as_tensor(dynamic, dtype=torch.float32).detach()
        g = torch.as_tensor(global_state, dtype=torch.float32).detach()
        assert d.shape == (len(self.static), len(DYNAMIC_FIELDS)) and g.shape == (len(GLOBAL_FIELDS),)
        x = torch.cat((self.static, d, g.expand(len(d), -1)), dim=1)
        assert not x.requires_grad and torch.isfinite(x).all()
        logits = self.net(x).flatten().double()
        logits = CONFIG["logit_bound"] * torch.tanh(logits/CONFIG["logit_bound"])
        return CONFIG["uniform_mix"]/len(d) + (1-CONFIG["uniform_mix"])*torch.softmax(logits, dim=0)

    def decide(self, dynamic, global_state, version):
        with torch.no_grad():
            p = self.probabilities(dynamic, global_state).numpy().copy()
        assert p.min() >= CONFIG["uniform_mix"]/len(p)
        return dict(version=int(version), dynamic=np.array(dynamic, copy=True),
                    global_state=np.array(global_state, copy=True), probability=p,
                    teacher_sha256=state_hash(self.net.state_dict()), teacher=self.state(),
                    distribution=distribution_stats(p))

    def learn(self, decision, counts, reward):
        version = decision["version"]
        assert version not in self.used_versions, "On-policy experience cannot be replayed"
        assert decision["teacher_sha256"] == state_hash(self.net.state_dict()), "Stale teacher policy"
        counts = torch.as_tensor(counts, dtype=torch.float64).detach()
        assert counts.shape == decision["probability"].shape and torch.all(counts >= 0) and counts.sum() > 0
        assert math.isfinite(float(reward)) and abs(float(reward)) <= 1.000001
        p = self.probabilities(decision["dynamic"], decision["global_state"])
        assert np.allclose(p.detach().numpy(), decision["probability"], atol=1e-13, rtol=1e-10)
        baseline = self.baseline
        advantage = float(reward)-baseline  # scalar measured outside autograd
        loss = -advantage*(counts*p.log()).sum()/CONFIG["score_sum_divisor"]
        before = copy.deepcopy(self.net.state_dict())
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [v.grad for v in self.net.parameters()]
        assert all(g is not None for g in grads)
        if not all(bool(torch.isfinite(g).all()) for g in grads):
            raise FloatingPointError("Nonfinite teacher gradient; retained experience, no student mutation")
        norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), CONFIG["gradient_clip"]))
        layer_norms = [float(torch.linalg.vector_norm(g)) for g in grads]
        self.optimizer.step()
        if not all(bool(torch.isfinite(v).all()) for v in self.net.parameters()):
            self.net.load_state_dict(before)
            self.optimizer.load_state_dict(optimizer_before)
            raise FloatingPointError("Nonfinite teacher parameters; step reverted")
        with torch.no_grad():
            new = self.probabilities(decision["dynamic"], decision["global_state"]).numpy()
        old = decision["probability"]
        kl = float((old*np.log(old/new)).sum())
        rejected = kl > CONFIG["max_teacher_step_kl"]
        if rejected:
            self.net.load_state_dict(before)
            self.optimizer.load_state_dict(optimizer_before)
            new = old.copy()
        delta = math.sqrt(sum(float(((v-before[k])**2).sum()) for k, v in self.net.state_dict().items()))
        beta = CONFIG["baseline_beta"]
        self.baseline = beta*baseline + (1-beta)*float(reward)
        self.reward_second = beta*self.reward_second + (1-beta)*float(reward)**2
        self.updates += 1
        self.used_versions.add(version)
        return dict(version=version, reward=float(reward), baseline_before=baseline, advantage=advantage,
                    baseline_after=self.baseline, reward_second_moment=self.reward_second,
                    loss=float(loss.detach()), gradient_norm_before_clip=norm, layer_gradient_norms=layer_norms,
                    parameter_delta_l2=delta, step_kl=kl, step_rejected=rejected,
                    sampled_decisions=int(counts.sum()), zero_signal=abs(advantage)<1e-12,
                    zero_gradient=norm<1e-12, warning=("trust_step_rejected" if rejected else
                        "zero_real_reward_signal" if abs(advantage)<1e-12 else "zero_gradient" if norm<1e-12 else
                        "large_gradient_clipped" if norm>CONFIG["gradient_clip"] else None),
                    policy_after=distribution_stats(new), teacher_after_sha256=state_hash(self.net.state_dict()))

    def state(self):
        return copy.deepcopy(dict(binding=self.binding, network=self.net.state_dict(),
                                  optimizer=self.optimizer.state_dict(), baseline=self.baseline,
                                  reward_second=self.reward_second, updates=self.updates,
                                  used_versions=sorted(self.used_versions), seen_student_seeds=self.seen_student_seeds))

    def load(self, state):
        assert state["binding"] == self.binding, "Incompatible teacher features/pool/protocol"
        self.net.load_state_dict(state["network"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for key in ("baseline", "reward_second", "updates", "seen_student_seeds"):
            setattr(self, key, copy.deepcopy(state[key]))
        self.used_versions = set(state["used_versions"])


class NeuralSampler:
    """Real terminal feedback plus a distribution frozen for a full teacher cycle."""
    def __init__(self, n, seed):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.probability = np.full(n, 1/n, dtype=np.float64)
        self.version = -1
        self.sample_counts = np.zeros(n, dtype=np.int64)
        self.completed_counts = np.zeros(n, dtype=np.int64)
        self.success_counts = np.zeros(n, dtype=np.int64)
        self.history = np.zeros((n, CONFIG["success_window"]), dtype=np.uint8)
        self.history_size = np.zeros(n, dtype=np.int64)
        self.history_cursor = np.zeros(n, dtype=np.int64)
        self.last_feedback_step = np.full(n, -1, dtype=np.int64)
        self.last_valid_step = np.full(n, -1, dtype=np.int64)
        self.error_ema = np.zeros((n, 4), dtype=np.float64)
        self.good_ema = np.zeros(n, dtype=np.float64)
        self.failure_ema = np.zeros(n, dtype=np.float64)
        self.serial = 0
        self.draws = []
        self.draw_end_steps = {}

    def probabilities(self):
        return self.probability

    def install(self, probability, version):
        p = np.asarray(probability, dtype=np.float64)
        distribution_stats(p)
        assert p.shape == (self.n,) and p.min() >= CONFIG["uniform_mix"]/self.n-1e-15
        assert version == self.version+1
        self.probability = p.copy()
        self.version = int(version)
        self.draws = []
        self.draw_end_steps = {}

    def sample(self, n):
        assert self.version >= 0
        ids = self.rng.choice(self.n, size=n, replace=True, p=self.probability)
        np.add.at(self.sample_counts, ids, 1)
        self.last_serials = np.arange(self.serial, self.serial+n, dtype=np.int64)
        self.serial += n
        return ids

    def record(self, ids, successes):
        # Deterministic env-ID order when the same sequence ends several times in one step.
        for i, success in zip(ids, successes):
            c = self.history_cursor[i]
            self.history[i, c] = int(success)
            self.history_cursor[i] = (c+1) % CONFIG["success_window"]
            self.history_size[i] = min(CONFIG["success_window"], self.history_size[i]+1)
            self.completed_counts[i] += 1
            self.success_counts[i] += int(success)

    def observe(self, ids, rows, step):
        alpha = CONFIG["error_ema_alpha"]
        for i, row in zip(ids, rows):
            fresh = self.last_feedback_step[i] < 0
            fraction = row[1]/row[0] if row[0] >= 100 and row[7] == 0 else 0.
            self.good_ema[i] = fraction if fresh else (1-alpha)*self.good_ema[i]+alpha*fraction
            self.failure_ema[i] = row[7] if fresh else (1-alpha)*self.failure_ema[i]+alpha*row[7]
            self.last_feedback_step[i] = step
            if row[0] > 0:
                self.error_ema[i] = row[2:6] if self.last_valid_step[i] < 0 else (1-alpha)*self.error_ema[i]+alpha*row[2:6]
                self.last_valid_step[i] = step

    def features(self, step, transitions):
        observed = self.last_feedback_step >= 0
        valid = self.last_valid_step >= 0
        rates = np.divide(self.history.sum(1), self.history_size, out=np.zeros(self.n), where=self.history_size>0)
        # Missing ages remain masked; log scaling fixed to total native controls, no fitted outcomes.
        age = np.where(observed, np.maximum(0, step-self.last_feedback_step), 0)
        valid_age = np.where(valid, np.maximum(0, step-self.last_valid_step), 0)
        f = np.column_stack((observed, valid, rates,
                             np.minimum(self.error_ema/np.array([.2,.3,.05,np.pi/12]), 10),
                             self.good_ema, self.failure_ema,
                             np.log1p(age)/np.log1p(240000), np.log1p(valid_age)/np.log1p(240000),
                             np.log1p(self.sample_counts)/np.log1p(10000),
                             np.log1p(self.completed_counts)/np.log1p(10000),
                             np.log1p(transitions)/np.log1p(10000000), self.history_size/100))
        assert f.shape == (self.n, len(DYNAMIC_FIELDS)) and np.isfinite(f).all()
        return f.astype(np.float32)

    def state_dict(self):
        return copy.deepcopy({k:v for k,v in vars(self).items() if k != "rng"}) | {"rng_state":copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        assert state["n"] == self.n
        for key,value in state.items():
            if key != "rng_state":
                setattr(self, key, copy.deepcopy(value))
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


def development_score(rows):
    values = np.array([r["values"] for r in rows], dtype=np.float64)
    valid = (values[:,0] >= 100) & (values[:,7] == 0)
    score = np.divide(values[:,1], values[:,0], out=np.zeros(len(rows)), where=valid)
    assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
    return score


def paired_gain(before, after, repeats=2):
    a,b = np.asarray(before),np.asarray(after)
    assert a.shape == b.shape and a.size % repeats == 0
    delta = (b-a).reshape(repeats, -1).mean(0)
    return dict(reward=float(delta.mean()), sequence_paired_standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                positive_sequences=int((delta>0).sum()), negative_sequences=int((delta<0).sum()),
                repeat_gap_after=float(np.abs(b.reshape(repeats,-1)[0]-b.reshape(repeats,-1)[1]).mean()) if repeats==2 else None)
