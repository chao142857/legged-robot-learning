"""Inactive candidate: fixed-anchor leg Gaussian KL inside installed native PPO.

Used only after a physical causal diagnosis supports leg-function drift. This
does not change observations, Gaussian sampling/log-prob, rewards, GAE, native
KL scheduler, or the single18D actor. No checkpoint/anchor is moved at runtime.
"""
import ast, copy, hashlib, inspect, pathlib, textwrap, types
import torch
from telemetry import model_sha, write_json

SCHEMA = 'original210_fixed_leg_gaussian_KL_v1'

def load_anchor(actor,path,expected_sha):
    from rsl_rl.models import MLPModel
    raw=pathlib.Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected_sha
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert 'synergy_head.weight' not in cp['actor_state_dict'], 'Declared common CP1800 leg anchor only'
    anchor=MLPModel.__new__(MLPModel);torch.nn.Module.__init__(anchor)
    for name in ('obs_groups','obs_dim','obs_normalization'):setattr(anchor,name,copy.deepcopy(getattr(actor,name)))
    for name in ('obs_normalizer','mlp'):setattr(anchor,name,copy.deepcopy(getattr(actor,name)))
    cached=actor.distribution._distribution
    anchor.distribution=copy.deepcopy(actor.distribution,{id(cached):None})
    anchor.distribution._distribution=None
    anchor.load_state_dict(cp['actor_state_dict'],strict=True)
    assert all(torch.equal(v.detach().cpu(),cp['actor_state_dict'][key]) for key,v in anchor.state_dict().items())
    return anchor.eval().requires_grad_(False)

def leg_kl(anchor_mean, anchor_std, current_mean, current_std):
    """KL(anchor||current), summed12leg dimensions, averaged actual states."""
    assert anchor_mean.shape[-1] == current_mean.shape[-1] == 18
    am, ast = anchor_mean[..., :12], anchor_std[..., :12]
    cm, cst = current_mean[..., :12], current_std[..., :12]
    return (torch.log(cst / ast) + (ast.square() + (am-cm).square()) / (2*cst.square()) - .5).sum(-1).mean()

class LegRetention:
    def __init__(self, alg, anchor, cfg, out):
        assert cfg['schema'] == SCHEMA and float(cfg['weight']) > 0
        assert not alg.actor.is_recurrent and not alg.symmetry and not alg.rnd
        self.alg, self.anchor, self.cfg = alg, anchor, copy.deepcopy(cfg)
        anchor.eval().requires_grad_(False)
        assert not set(anchor.parameters()) & set(alg.actor.parameters())
        self.fixed_anchor_hash = model_sha(anchor)
        self.values = []
        self.out = pathlib.Path(out)
        source = textwrap.dedent(inspect.getsource(type(alg).update))
        self.original_update_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        points = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'backward'
                  and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'loss']
        assert len(points) == 1, 'Installed native PPO source must have exactly one total-loss backward'
        line = points[0].lineno - 1
        lines = source.splitlines()
        indent = lines[line][:len(lines[line])-len(lines[line].lstrip())]
        # Add to the same native loss before zero_grad/backward. All original
        # total-PPO/GAE gradients and exact Gaussian action likelihood remain.
        lines.insert(line, indent + 'loss = loss + self._stage3b_leg_retention.penalty(batch.observations, distribution_params)')
        patched = '\n'.join(lines)+'\n'
        namespace = dict(type(alg).update.__globals__)
        exec(compile(patched, '<installed_native_PPO_plus_fixed_leg_KL>', 'exec'), namespace)
        alg._stage3b_leg_retention = self
        alg.update = types.MethodType(namespace['update'], alg)
        write_json(self.out/'leg-retention-contract.json', dict(
            **self.cfg, anchor_actor_sha256=self.fixed_anchor_hash,
            native_update_sha256=self.original_update_source_sha256,
            patched_update_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            injection='one additive declared loss before native total-loss backward',
            original_GAE_surrogate_rewards_logprob_and_KL_scheduler_unchanged=True,
            all18D_actor_parameters_trainable=True, inference_external_controller=False))

    def penalty(self, obs, params):
        mean, std = params
        with torch.no_grad():
            anchor_mean = self.anchor(obs, stochastic_output=False)
            distribution = self.anchor.distribution
            # Deterministic inference does not update cached Normal. Read the
            # actual installed state-independent parameter, without a sample
            # or RNG draw and without reusing a stale previous batch shape.
            if distribution.std_type == 'scalar':
                anchor_std = distribution.std_param.expand_as(anchor_mean)
            else:
                assert distribution.std_type == 'log'
                anchor_std = distribution.log_std_param.exp().expand_as(anchor_mean)
        assert mean.shape == anchor_mean.shape
        loss = leg_kl(anchor_mean, anchor_std, mean, std)
        assert torch.isfinite(loss), 'Leg retention numerical error'
        self.values.append(dict(leg_KL=float(loss.detach()), weighted_loss=float((loss*self.cfg['weight']).detach()),
            leg_mean_shift_RMS=float((mean[...,:12].detach()-anchor_mean[...,:12]).square().mean().sqrt()),
            arm_mean_shift_RMS=float((mean[...,12:].detach()-anchor_mean[...,12:]).square().mean().sqrt())))
        return self.cfg['weight']*loss

    def update_summary(self):
        assert model_sha(self.anchor) == self.fixed_anchor_hash, 'Fixed skill anchor changed'
        rows, self.values = self.values, []
        assert len(rows) == self.alg.num_learning_epochs*self.alg.num_mini_batches
        return {key:sum(r[key] for r in rows)/len(rows) for key in rows[0]}
