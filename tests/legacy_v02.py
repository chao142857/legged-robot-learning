"""nSGS V0.2: low-dimensional, state-conditioned counterfactual teacher.

Only the likelihood of the *sampled preference* is differentiated. Sequence
draws, simulator, PPO, measured branch outcomes and all input features are data.
This module is CPU-only and never imports Isaac Lab or a Student policy.
"""
import copy
import hashlib
import math

import numpy as np
import torch

from nsgs.sampling.neural import DYNAMIC_FIELDS
from nsgs.evaluation.protocol import GLOBAL_FIELDS
from nsgs.teacher.v0_core import distribution_stats, state_hash


V02_CONFIG = dict(version="nsgs-v02-counterfactual-20261106", preference_dim=8,
                  teacher_seed=1048, projection_seed=30261106, exploration_sigma=.5,
                  hidden=32, learning_rate=.01, uniform_mix=.2, logit_bound=4.,
                  mean_bound=2., gradient_clip=1., max_preference_step_kl=.05,
                  max_step_backtracks=8, command_clip=5., basis_clip=8.)
PARTIAL_FIELDS = ["partial_observed", "partial_error_observed", "partial_scored_fraction",
                  "partial_good_fraction", "partial_velocity_rmse_scaled",
                  "partial_yaw_rmse_scaled", "partial_position_rmse_scaled",
                  "partial_orientation_rmse_scaled"]
ENCODING_DESCRIPTION = (
    "Fixed seeded eight-dimensional projection, block-balanced across standardized "
    "training commands, real completed-feedback features, learning-progress plus "
    "availability, and separately marked partial-episode measurements. Command "
    "means/scales and projected-command scales are fitted only to training commands. "
    "Dynamic physical values use declared fixed scales, with explicit missing masks. "
    "No hard groups, sweet-spot function or outcome-fitted projection. Eight dimensions "
    "are a lossy representation and do not preserve all command/state distinctions."
)


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _numpy(value, dtype=np.float64):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def antithetic_loss(mean, plus, minus, sigma, reward_plus, reward_minus, baseline):
    """Score-function loss; sampled preferences and measured rewards are detached.

    Its negative gradient wrt mean is (R+ - R-) epsilon / (2 sigma).
    Both preferences have the same marginal Gaussian policy even though their
    joint construction is antithetic. The shared, action-independent baseline
    cancels from this estimator exactly, but is retained for provenance.
    """
    plus = torch.as_tensor(plus, dtype=mean.dtype, device=mean.device).detach()
    minus = torch.as_tensor(minus, dtype=mean.dtype, device=mean.device).detach()
    rp, rm, b = [float(_numpy(x).item()) for x in (reward_plus, reward_minus, baseline)]
    assert all(math.isfinite(v) for v in (rp, rm, b)) and sigma > 0
    normal = torch.distributions.Normal(mean, float(sigma))
    lp = normal.log_prob(plus).sum(); lm = normal.log_prob(minus).sum()
    return -.5 * ((rp-b)*lp + (rm-b)*lm), lp, lm


class FeatureEncoder:
    """All fitted statistics are command-only; masks never synthesize outcomes."""
    def __init__(self, static_commands, config=None):
        self.config = copy.deepcopy(V02_CONFIG if config is None else config)
        x = _numpy(static_commands)
        assert x.ndim == 2 and len(x) > 1 and x.shape[1] > 0 and np.isfinite(x).all()
        self.n, self.command_dim = x.shape
        self.static_sha256 = array_hash(x)
        self.command_mean = x.mean(0)
        self.command_scale = x.std(0)
        self.command_scale[self.command_scale < 1e-8] = 1.
        x = np.clip((x-self.command_mean)/self.command_scale,
                    -self.config["command_clip"], self.config["command_clip"])
        dims = [self.command_dim, len(DYNAMIC_FIELDS), 2, len(PARTIAL_FIELDS)]
        rng = np.random.default_rng(self.config["projection_seed"])
        # Each block receives equal prior projection variance independent of its
        # number of columns. This is fixed before any branch outcome exists.
        self.projections = [rng.normal(size=(d,self.config["preference_dim"]))/math.sqrt(4*d)
                            for d in dims]
        command = x @ self.projections[0]
        self.projected_command_scale = command.std(0)
        self.projected_command_scale[self.projected_command_scale < .05] = .05
        self.command_basis = command / self.projected_command_scale

    def encode(self, dynamic, progress=None, progress_observed=None, partial=None):
        d = _numpy(dynamic).copy()
        assert d.shape == (self.n,len(DYNAMIC_FIELDS)) and np.isfinite(d).all()
        mask_cols = [0,1,2,3,28,29]
        assert np.isin(d[:,mask_cols], [0.,1.]).all(), "Feedback masks must be explicit booleans"
        # A completed-feedback input must never smuggle unobserved numeric values
        # into the projected basis. Preserve the original masks in the snapshot.
        d[:,[4,9,10,13,14]] *= d[:,2,None]
        d[:,[5,6,7,8,15,16]] *= d[:,3,None]
        d[:,21:24] *= d[:,28,None]
        d[:,24:28] *= d[:,29,None]
        d[:,11] *= d[:,0]; d[:,12] *= d[:,1]
        # Errors already use strict-tolerance units. Fixed ten-tolerance scale;
        # count/age features are already log-normalized in neural_v01.
        ds = d.copy()
        ds[:,[5,6,7,8,24,25,26,27]] = np.clip(ds[:,[5,6,7,8,24,25,26,27]],0,10)/10.
        ds = np.clip(ds,-2,2)
        if progress is None:
            assert progress_observed is None
            progress = np.zeros(self.n); progress_observed = np.zeros(self.n)
        progress = _numpy(progress); po = _numpy(progress_observed)
        assert progress.shape == po.shape == (self.n,) and np.isfinite(progress).all()
        assert np.isin(po,[0.,1.]).all()
        progress = np.clip(progress,-1,1)*po
        pr = np.column_stack((progress,po))
        if partial is None:
            partial = np.zeros((self.n,len(PARTIAL_FIELDS)))
        partial = _numpy(partial).copy()
        assert partial.shape == (self.n,len(PARTIAL_FIELDS)) and np.isfinite(partial).all()
        assert np.isin(partial[:,:2],[0.,1.]).all()
        assert (partial[:,1] <= partial[:,0]).all()
        partial[:,2:4] *= partial[:,0,None]
        partial[:,4:8] *= partial[:,1,None]
        ps = partial.copy()
        ps[:,2:4] = np.clip(ps[:,2:4],0,1)
        ps[:,4:8] = np.clip(ps[:,4:8],0,10)/10.
        basis = self.command_basis + (ds@self.projections[1] + pr@self.projections[2]
                                      + ps@self.projections[3])/self.projected_command_scale
        # Common offsets have no sampling meaning; remove them before a bounded
        # logit transform so uniform missingness/coverage cannot induce bias.
        basis -= basis.mean(0)
        basis = np.clip(basis,-self.config["basis_clip"],self.config["basis_clip"])
        assert np.isfinite(basis).all()
        return dict(basis=basis, dynamic=d, progress=progress,
                    progress_observed=po.copy(), partial=partial,
                    basis_sha256=array_hash(basis), description=ENCODING_DESCRIPTION)

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        assert state["config"] == self.config and state["static_sha256"] == self.static_sha256
        assert state["n"] == self.n and state["command_dim"] == self.command_dim
        for key in vars(self):
            setattr(self,key,copy.deepcopy(state[key]))


def learning_progress_from_records(recent_records, n, step, window_controls=24000):
    """Later-half minus earlier-half real continuous performance, with a mask.

    This descriptive recent change is noisy and observational, not a causal
    estimate. Both fixed halves of the last 1000 updates require evidence. Failed and
    invalid-window episodes score zero, matching the development objective.
    """
    progress=np.zeros(n); observed=np.zeros(n)
    for i, records in recent_records.items():
        valid=[(int(t),_numpy(r)) for t,r in records if step-window_controls <= t <= step]
        if len(valid)<2: continue
        times=np.array([t for t,r in valid]); rows=np.stack([r for t,r in valid])
        split=step-window_controls/2
        early=times<=split; late=times>split
        if not early.any() or not late.any(): continue
        score=np.divide(rows[:,1],rows[:,0],out=np.zeros(len(rows)),
                        where=(rows[:,0]>=100)&(rows[:,7]==0))
        progress[int(i)]=score[late].mean()-score[early].mean(); observed[int(i)]=1.
    return progress,observed


class PreferenceNetwork(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.layers=torch.nn.Sequential(torch.nn.Linear(len(GLOBAL_FIELDS),config["hidden"]),
                                        torch.nn.Tanh(),torch.nn.Linear(config["hidden"],8))
        torch.nn.init.normal_(self.layers[-1].weight,0.,.001)
        torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self, global_state):
        # Global columns are the existing fixed-scale native state schema. Keep
        # extreme losses bounded without fitting a scale to dev/test returns.
        g=torch.clamp(global_state,-10.,10.)
        raw=self.layers(g)
        return self.config["mean_bound"]*torch.tanh(raw/self.config["mean_bound"])


class Teacher:
    def __init__(self, static, binding=None, seed=None, config=None):
        self.config=copy.deepcopy(V02_CONFIG)
        if config is not None: self.config.update(config)
        if seed is not None: self.config["teacher_seed"]=int(seed)
        assert self.config["preference_dim"] == 8 and self.config["uniform_mix"] == .2
        self.binding=copy.deepcopy({} if binding is None else binding)
        self.encoder=FeatureEncoder(static,self.config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.config["teacher_seed"])
            self.net=PreferenceNetwork(self.config).double()
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=self.config["learning_rate"])
        self.rng=np.random.default_rng(self.config["teacher_seed"]+1)
        self.updates=0; self.used_versions=set()

    def mean(self, global_state):
        g=torch.as_tensor(_numpy(global_state).copy(),dtype=torch.float64).detach()
        assert g.shape == (len(GLOBAL_FIELDS),) and torch.isfinite(g).all()
        return self.net(g)

    def probability_from_preference(self, basis, preference):
        basis=_numpy(basis); z=_numpy(preference)
        assert basis.shape==(self.encoder.n,8) and z.shape==(8,)
        assert np.isfinite(basis).all() and np.isfinite(z).all()
        logits=basis@z/math.sqrt(8.)
        logits=self.config["logit_bound"]*np.tanh(logits/self.config["logit_bound"])
        e=np.exp(logits-logits.max()); p=.2/len(e)+.8*e/e.sum()
        assert p.min()>=.2/len(p) and abs(p.sum()-1)<1e-12
        return p

    def decide(self, dynamic, global_state, version, progress=None, progress_observed=None, partial=None):
        encoded=self.encoder.encode(dynamic,progress,progress_observed,partial)
        with torch.no_grad(): mean=self.mean(global_state).numpy().copy()
        p=self.probability_from_preference(encoded["basis"],mean)
        return dict(version=version, **encoded, global_state=_numpy(global_state).copy(),
                    preference=mean, probability=p, distribution=distribution_stats(p),
                    teacher_sha256=state_hash(self.net.state_dict()), teacher=self.state(),
                    deterministic_frozen_mean=True)

    def sample_pair(self, dynamic, global_state, version, progress=None, progress_observed=None, partial=None):
        assert str(version) not in self.used_versions, "Cannot reuse an already trained decision"
        d=self.decide(dynamic,global_state,version,progress,progress_observed,partial)
        epsilon=self.rng.normal(size=8); sigma=self.config["exploration_sigma"]
        mean=d.pop("preference"); d.pop("probability"); d.pop("distribution")
        # Save post-draw RNG state for exact continuation, along with actual epsilon.
        d.update(mean=mean,epsilon=epsilon,sigma=sigma,teacher=self.state(),
                 deterministic_frozen_mean=False,teacher_updates=self.updates)
        for sign,name in ((1,"plus"),(-1,"minus")):
            z=mean+sign*sigma*epsilon
            p=self.probability_from_preference(d["basis"],z)
            log_prob=float(-.5*np.sum(((z-mean)/sigma)**2+2*math.log(sigma)+math.log(2*math.pi)))
            d[name]=dict(preference=z,probability=p,log_probability=log_prob,
                         distribution=distribution_stats(p),probability_sha256=array_hash(p))
        d["pair_tv"]=float(abs(d["plus"]["probability"]-d["minus"]["probability"]).sum()/2)
        return d

    def learn_pair(self, decision, before_dev, uniform_after, plus_after, minus_after):
        version=str(decision["version"])
        assert version not in self.used_versions,"Off-policy replay is forbidden"
        assert decision["teacher_updates"]==self.updates
        assert decision["teacher_sha256"]==state_hash(self.net.state_dict()),"Stale teacher"
        scores=[float(_numpy(x).item()) for x in (before_dev,uniform_after,plus_after,minus_after)]
        assert all(math.isfinite(x) and 0<=x<=1 for x in scores),"Real dev fractions required"
        before,u,p,m=scores; rp=p-before; rm=m-before; baseline=u-before
        assert array_hash(decision["basis"])==decision["basis_sha256"]
        mu=self.mean(decision["global_state"])
        assert np.array_equal(mu.detach().numpy(),decision["mean"])
        sigma=float(decision["sigma"])
        assert sigma==self.config["exploration_sigma"]
        for name,sign in (("plus",1),("minus",-1)):
            arm=decision[name]
            assert np.allclose(arm["preference"],decision["mean"]+sign*sigma*decision["epsilon"],atol=1e-14,rtol=0)
            assert np.array_equal(self.probability_from_preference(decision["basis"],arm["preference"]),arm["probability"])
        loss,lp,lm=antithetic_loss(mu,decision["plus"]["preference"],decision["minus"]["preference"],
                                  sigma,rp,rm,baseline)
        assert abs(float(lp.detach())-decision["plus"]["log_probability"])<1e-10
        assert abs(float(lm.detach())-decision["minus"]["log_probability"])<1e-10
        old=copy.deepcopy(self.net.state_dict()); old_optimizer=copy.deepcopy(self.optimizer.state_dict())
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        grads=[v.grad for v in self.net.parameters()]
        if not all(g is not None and bool(torch.isfinite(g).all()) for g in grads):
            raise FloatingPointError("Nonfinite teacher gradient; update not applied")
        gradient_vector=torch.cat([g.detach().flatten() for g in grads]).numpy().copy()
        norm=float(torch.nn.utils.clip_grad_norm_(self.net.parameters(),self.config["gradient_clip"]))
        zero_signal=abs(rp-rm)<1e-14
        # Backtrack the optimizer step against a predeclared policy-distance
        # bound, never against branch returns. Every attempt starts from the
        # same optimizer state, so Adam moments advance exactly once on success.
        attempts=[];rejected=False;accepted=False;kl=0.;new_mu=decision["mean"].copy()
        base_lrs=[group['lr'] for group in self.optimizer.param_groups]
        if not zero_signal:
            for backtrack in range(self.config['max_step_backtracks']+1):
                self.net.load_state_dict(old);self.optimizer.load_state_dict(old_optimizer)
                multiplier=2.**(-backtrack)
                for group,base_lr in zip(self.optimizer.param_groups,base_lrs):group['lr']=base_lr*multiplier
                self.optimizer.step()
                finite=all(bool(torch.isfinite(v).all()) for v in self.net.parameters())
                with torch.no_grad():candidate_mu=self.mean(decision['global_state']).numpy()
                candidate_kl=float(np.sum((candidate_mu-decision['mean'])**2)/(2*sigma*sigma))
                finite=finite and math.isfinite(candidate_kl)
                attempts.append(dict(backtracks=backtrack,learning_rate_multiplier=multiplier,
                    attempted_learning_rates=[lr*multiplier for lr in base_lrs],
                    preference_kl=candidate_kl if math.isfinite(candidate_kl) else None,finite=finite))
                if finite and candidate_kl<=self.config['max_preference_step_kl']:
                    accepted=True;kl=candidate_kl;new_mu=candidate_mu.copy()
                    # Keep the declared LR for the next independent paired
                    # decision; accepted Adam moments remain at this one step.
                    for group,base_lr in zip(self.optimizer.param_groups,base_lrs):group['lr']=base_lr
                    break
            rejected=not accepted
            if rejected:
                self.net.load_state_dict(old);self.optimizer.load_state_dict(old_optimizer)
        self.updates+=1; self.used_versions.add(version)
        delta=math.sqrt(sum(float(((value-old[key])**2).sum()) for key,value in self.net.state_dict().items()))
        new_p=self.probability_from_preference(decision["basis"],new_mu)
        return dict(version=decision["version"], before_dev=before,uniform_after=u,plus_after=p,minus_after=m,
                    reward_plus=rp,reward_minus=rm,uniform_gain_baseline=baseline,
                    paired_reward_difference=rp-rm,plus_vs_uniform=p-u,minus_vs_uniform=m-u,
                    loss=float(loss.detach()),gradient_vector=gradient_vector,
                    gradient_norm_before_clip=norm,parameter_delta_l2=delta,
                    mean_gradient_ascent=(rp-rm)*decision["epsilon"]/(2*sigma),
                    preference_step_kl=kl,step_rejected=rejected,zero_signal=zero_signal,
                    step_backtracking=attempts,accepted_learning_rate_multiplier=(attempts[-1]['learning_rate_multiplier'] if accepted else 0.),
                    gaussian_entropy=4*math.log(2*math.pi*math.e*sigma*sigma),
                    policy_after=distribution_stats(new_p),teacher_after_sha256=state_hash(self.net.state_dict()),
                    warning=("preference_step_rejected" if rejected else "zero_paired_reward" if zero_signal else
                             "zero_gradient" if norm<1e-12 else "large_gradient_clipped" if norm>self.config["gradient_clip"] else None))

    def state(self):
        return copy.deepcopy(dict(config=self.config,binding=self.binding,encoder=self.encoder.state_dict(),
                                  network=self.net.state_dict(),optimizer=self.optimizer.state_dict(),
                                  rng=self.rng.bit_generator.state,updates=self.updates,
                                  used_versions=sorted(self.used_versions)))

    def load(self,state):
        assert state["config"]==self.config and state["binding"]==self.binding,"Incompatible teacher"
        self.encoder.load_state_dict(state["encoder"])
        self.net.load_state_dict(state["network"],strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.bit_generator.state=copy.deepcopy(state["rng"])
        self.updates=int(state["updates"]); self.used_versions=set(state["used_versions"])
