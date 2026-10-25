"""CPU-only frozen-policy gradient audit. No simulator, optimizer or training entry.

Use only trusted local checkpoints (native RSL checkpoints contain Python RNG
objects). Rollout bundles use torch's restricted weights_only loader.
See docs/go2_piper_gradient_audit/DIAGNOSTIC_PROTOCOL.md for the data contract.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import torch
from torch import nn

OBJECTIVES = ('locomotion', 'manipulation', 'safety_regularization')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class FrozenMLP(nn.Module):
    """Strict adapter for audited RSL 5.0.1 MLPModel (ELU, Identity obs norm)."""
    def __init__(self, state):
        super().__init__()
        dims = [state[f'mlp.{i}.weight'].shape for i in (0, 2, 4, 6)]
        if [d[0] for d in dims[:3]] != [512, 256, 128]:
            raise ValueError('Unsupported architecture; use the actual model adapter')
        layers = []
        for i, (out_dim, in_dim) in enumerate(dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < 3:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.distribution = nn.Module()
        self.distribution.log_std_param = nn.Parameter(torch.zeros(dims[-1][0]))
        self.load_state_dict(state, strict=True)
        self.cpu().eval()

    def forward(self, x):
        return self.mlp(x)

    def normal(self, x):
        return torch.distributions.Normal(self(x), self.distribution.log_std_param.exp())


def component_gae(rewards, values, next_values, terminated, episode_end,
                  gamma=.99, lam=.95):
    """[T,N,K] inputs; next_values use pre-autoreset terminal observations.

    Bootstrap time-limit truncations, stop traces at ALL episode boundaries.
    This principled timeout convention differs from native current-V timeout
    addition. Never feed a scalar total critic expanded over K objectives.
    """
    if rewards.ndim != 3 or rewards.shape != values.shape or values.shape != next_values.shape:
        raise ValueError('Require genuine [T,N,K] component values and rewards')
    if terminated.shape != rewards.shape[:2] or episode_end.shape != terminated.shape:
        raise ValueError('Invalid boundary shape')
    if torch.any(terminated & ~episode_end):
        raise ValueError('Every termination must end an episode')
    out = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]).unsqueeze(-1) * next_values[t] - values[t]
        tail = delta + gamma * lam * (~episode_end[t]).unsqueeze(-1) * tail
        out[t] = tail
    return out.detach()


def complete_mc(rewards, episode_end, valid, gamma=.99):
    """Complete episodic discounted return, zero baseline, [T,N,K].

    valid must be a nonempty prefix per environment. Last valid step must be an
    episode end. Time limits are treated as finite-horizon terminal: caller must
    label this estimand, not claim native continuing-task GAE equivalence.
    """
    if rewards.ndim != 3 or valid.shape != rewards.shape[:2] or episode_end.shape != valid.shape:
        raise ValueError('Bad MC shapes')
    if not valid[0].all() or torch.any(valid[1:] & ~valid[:-1]):
        raise ValueError('Require nonempty valid prefixes, no holes')
    last = valid.long().sum(0) - 1
    if not episode_end[last, torch.arange(valid.shape[1])].all():
        raise ValueError('Incomplete tails: zero bootstrap would manufacture advantages')
    out = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        tail = (rewards[t] + gamma * (~episode_end[t]).unsqueeze(-1) * tail) * valid[t].unsqueeze(-1)
        out[t] = tail
    return out.detach()


def normalize_components(a, mode):
    """Normalize over WHOLE rollout before minibatch/task slicing, never locally."""
    if mode == 'raw':
        return a.detach()
    centered = a - a.mean(0)
    if mode == 'shared_total_std':
        return (centered / (a.sum(-1).std() + 1e-8)).detach()
    if mode == 'per_objective_std':
        return (centered / (a.std(0) + 1e-8)).detach()
    raise ValueError(mode)


def policy_loss(logp, old_logp, advantage, mode='unclipped', total_adv=None, clip=.2):
    ratio = (logp - old_logp.detach()).exp()
    a = advantage.detach()
    if mode == 'unclipped':
        return -(ratio * a).mean()
    if mode == 'separate_clip':
        return torch.maximum(-ratio*a, -ratio.clamp(1-clip, 1+clip)*a).mean()
    if mode == 'total_mask':
        if total_adv is None:
            raise ValueError('Need total advantage for exact common clip mask')
        active = ~(((total_adv >= 0) & (ratio > 1+clip)) |
                   ((total_adv < 0) & (ratio < 1-clip)))
        # Same gradient as decomposing the aggregate surrogate at this theta;
        # not a standalone clipped objective or meaningful absolute loss value.
        return -(ratio * a * active.detach()).mean()
    raise ValueError(mode)


def gradient(loss, model):
    named = list(model.named_parameters())
    grad = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    return {n: (torch.zeros_like(p) if g is None else g.detach()).flatten().double()
            for (n, p), g in zip(named, grad)}, [n for (n, _), g in zip(named, grad) if g is None]


def scopes(g, action_dim=18):
    result = {'all': torch.cat(list(g.values()))}
    result['trunk'] = torch.cat([v for n, v in g.items() if n.startswith(('mlp.0.', 'mlp.2.', 'mlp.4.'))])
    for layer in ('mlp.0', 'mlp.2', 'mlp.4', 'mlp.6', 'distribution'):
        result[layer] = torch.cat([v for n, v in g.items() if n.startswith(layer + '.')])
    if action_dim in (12, 18):
        for name, sl in [('leg_output_rows', slice(0, 12)), ('arm_output_rows', slice(12, 18))]:
            if name.startswith('arm') and action_dim == 12:
                continue
            result[name] = torch.cat([g['mlp.6.weight'].reshape(action_dim, -1)[sl].flatten(),
                                      g['mlp.6.bias'][sl]])
    return result


def pair_stats(a, b, tol=1e-12):
    na, nb = float(a.norm()), float(b.norm())
    dot = float(a @ b)
    usable = na > tol and nb > tol
    cosine = dot / (na*nb) if usable else None
    return dict(dot_product=dot, norm_i=na, norm_j=nb,
                norm_ratio_i_over_j=na/nb if nb > tol else None,
                cosine=cosine, conflict=(cosine < -1e-8) if usable else None,
                degenerate=not usable)


def snapshot_transition(actor_obs, critic_obs, actions, old_logp, old_mu, old_std,
                        task_id, base_level, ee_level):
    """Call immediately BEFORE env.step. Does not sample or alter global RNG."""
    return {k: v.detach().cpu().clone() for k, v in locals().copy().items()}


def validate_bundle(b, checkpoint_sha):
    m = b['metadata']
    for key in ('schema_version', 'behavior_checkpoint_sha256', 'rollout_id', 'seed',
                'estimator', 'estimator_bias', 'on_policy_stochastic', 'control_structure',
                'reward_group_definition', 'reward_unit', 'gamma', 'normalization_scope'):
        if key not in m:
            raise ValueError('Missing provenance: ' + key)
    if m['schema_version'] != 1 or m['behavior_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Wrong schema or behavior checkpoint')
    if m['on_policy_stochastic'] is not True or m['control_structure'] != 'B0_18D':
        raise ValueError('Require stochastic B0 behavior; deterministic replay is not on-policy data')
    if m['estimator'] not in ('complete_mc_finite_horizon', 'crossfit_component_gae'):
        raise ValueError('Unsupported estimator; scalar-critic reward splitting is forbidden')
    if m['normalization_scope'] != 'whole_rollout' or m['reward_unit'] != 'weighted_dt_once':
        raise ValueError('Wrong normalization scope or reward units')
    if m['estimator'] == 'crossfit_component_gae':
        for key in ('value_fit_rollout_ids', 'heldout_value_metrics', 'timeout_convention'):
            if key not in m:
                raise ValueError('Missing value-estimator provenance: ' + key)
        if str(m['rollout_id']) in [str(x) for x in m['value_fit_rollout_ids']]:
            raise ValueError('Critic fitted on diagnostic rollout: not cross-fitted')
    n = b['actor_obs'].shape[0]
    shapes = {'actor_obs': (n,210), 'critic_obs': (n,102), 'actions': (n,18),
              'old_logp': (n,), 'old_mu': (n,18), 'old_std': (n,18),
              'component_advantages': (n,3), 'task_id': (n,), 'base_level': (n,), 'ee_level': (n,)}
    for key, shape in shapes.items():
        if key not in b or tuple(b[key].shape) != shape or not torch.isfinite(b[key]).all():
            raise ValueError('Invalid tensor: ' + key)
    if n < 2 or torch.any(b['old_std'] <= 0):
        raise ValueError('Insufficient samples or invalid std')
    if ('total_returns' in b) != ('old_values' in b):
        raise ValueError('Critic audit needs both native total returns and old values')
    for k in ('total_returns', 'old_values'):
        if k in b and (b[k].shape != (n,) or not torch.isfinite(b[k]).all()):
            raise ValueError('Invalid critic target: ' + k)
    return n


def analyze(checkpoint, bundle, minibatch_size=2048, partition_seed=71, candidate=None):
    if minibatch_size < 2:
        raise ValueError('minibatch_size must be >=2')
    b = bundle
    n = validate_bundle(b, sha256(checkpoint))
    if candidate is not None and sha256(candidate) not in b['metadata'].get('update_candidate_sha256s', []):
        raise ValueError('Candidate lacks same-rollout update provenance')
    c = torch.load(checkpoint, map_location='cpu', weights_only=False)
    behavior = FrozenMLP(c['actor_state_dict'])
    actor = behavior if candidate is None else FrozenMLP(torch.load(candidate, map_location='cpu', weights_only=False)['actor_state_dict'])
    critic = FrozenMLP(c['critic_state_dict'])
    with torch.no_grad():
        dist = behavior.normal(b['actor_obs'])
        checks = dict(mu_error=float((dist.mean-b['old_mu']).abs().max()),
                      std_error=float((dist.stddev-b['old_std']).abs().max()),
                      logp_error=float((dist.log_prob(b['actions']).sum(-1)-b['old_logp']).abs().max()))
        if checks['mu_error'] > 1e-4 or checks['std_error'] > 1e-5 or checks['logp_error'] > 2e-3:
            raise ValueError('Behavior mismatch: '+str(checks))
    gen = torch.Generator().manual_seed(partition_seed)
    order = torch.randperm(n, generator=gen)
    strata = [('all', torch.arange(n))]
    for key in ('task_id', 'base_level', 'ee_level'):
        for value in b[key].unique().tolist():
            ids = torch.where(b[key] == value)[0]
            if len(ids) >= 2:
                strata.append((f'{key}={value}', ids))
    partitions = [(f'minibatch={i//minibatch_size}', order[i:i+minibatch_size])
                  for i in range(0,n,minibatch_size) if len(order[i:i+minibatch_size])>=2]
    rows, clipping = [], []
    for norm in ('raw', 'shared_total_std', 'per_objective_std'):
        advantages = normalize_components(b['component_advantages'], norm)
        for stratum, ids in strata + partitions:
            obs, actions = b['actor_obs'][ids], b['actions'][ids]
            old_lp = b['old_logp'][ids]
            aa = advantages[ids]
            with torch.no_grad():
                d = actor.normal(obs)
                ratio = (d.log_prob(actions).sum(-1)-old_lp).exp()
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(b['old_mu'][ids],b['old_std'][ids]),d).sum(-1)
                clipping.append(dict(normalization=norm,stratum=stratum,n=len(ids),kl_mean=float(kl.mean()),
                                     kl_max_sample=float(kl.max()),clip_fraction=float(((ratio-1).abs()>.2).float().mean()),
                                     active_clip_fraction=float((((aa.sum(-1)>=0)&(ratio>1.2))|
                                         ((aa.sum(-1)<0)&(ratio<.8))).float().mean())))
            for mode in ('unclipped','total_mask','separate_clip'):
                gs = []
                for j in range(3):
                    lp = actor.normal(obs).log_prob(actions).sum(-1)
                    g, unused = gradient(policy_loss(lp,old_lp,aa[:,j],mode,aa.sum(-1)), actor)
                    gs.append(scopes(g))
                for i,j in itertools.combinations(range(3),2):
                    for scope in gs[i]:
                        rows.append(dict(kind='actor',normalization=norm,surrogate=mode,stratum=stratum,
                                         n=len(ids),pair=f'{OBJECTIVES[i]}/{OBJECTIVES[j]}',scope=scope,
                                         dot_i_total=float(gs[i][scope] @ sum(x[scope] for x in gs)),
                                         dot_j_total=float(gs[j][scope] @ sum(x[scope] for x in gs)),
                                         total_norm=float(sum(x[scope] for x in gs).norm()),
                                         **pair_stats(gs[i][scope],gs[j][scope])))
    # Single critic: task-conditioned TOTAL value losses only. Never compare
    # the same scalar value prediction to three unrelated component returns.
    critic_rows, critic_unused = [], []
    if 'total_returns' in b:
        task_grad = {}
        for label, ids in strata:
            if label != 'all' and not label.startswith('task_id='):
                continue
            pred = critic(b['critic_obs'][ids]).squeeze(-1)
            old, target = b['old_values'][ids].detach(), b['total_returns'][ids].detach()
            loss = torch.maximum((pred-target).square(),(old+(pred-old).clamp(-.2,.2)-target).square()).mean()
            g, critic_unused = gradient(loss,critic)
            task_grad[label] = scopes(g,1)
            err = pred.detach()-target
            var = target.var(unbiased=False)
            critic_rows.append(dict(stratum=label,n=len(ids),value_loss=float(loss.detach()),
                explained_variance=float(1-err.var(unbiased=False)/var) if var>1e-12 else None,
                grad_norm=float(task_grad[label]['all'].norm())))
        for i,j in itertools.combinations([x for x in task_grad if x!='all'],2):
            for scope in task_grad[i]:
                rows.append(dict(kind='critic_total_by_task',normalization='native_targets',surrogate='value_clipped',
                                 stratum='task_pairs',pair=f'{i}/{j}',scope=scope,
                                 **pair_stats(task_grad[i][scope],task_grad[j][scope])))
    summaries = []
    keys = sorted({(r['normalization'],r['surrogate'],r['pair'],r['scope']) for r in rows if r['kind']=='actor'})
    for norm,mode,pair,scope in keys:
        rr = [r for r in rows if r['kind']=='actor' and r['stratum'].startswith('minibatch=') and
              (r['normalization'],r['surrogate'],r['pair'],r['scope']) == (norm,mode,pair,scope)]
        valid = [r for r in rr if not r['degenerate']]
        summaries.append(dict(normalization=norm,surrogate=mode,pair=pair,scope=scope,
                              n_minibatches=len(rr),n_valid=len(valid),
                              conflict_rate=sum(r['conflict'] for r in valid)/len(valid) if valid else None))
    return dict(metadata=b['metadata'],checkpoint_sha256=sha256(checkpoint),candidate_sha256=sha256(candidate) if candidate else None,
                behavior_checks=checks,n=n,partition_seed=partition_seed,optimizer_steps=0,device='cpu',
                caveat='Descriptive within-rollout statistics; minibatches are not independent replicates. No H1 verdict.',
                actor_pairs=[r for r in rows if r['kind']=='actor'],
                critic_pairs=[r for r in rows if r['kind']=='critic_total_by_task'],
                clipping=clipping,conflict_rates=summaries,critic=critic_rows,
                critic_unused_parameters=critic_unused,
                critic_status='total-value task-conditioned only' if critic_rows else 'unavailable: native targets absent')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--batch',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--candidate',help='Optional frozen candidate from SAME rollout update, not a later unrelated checkpoint')
    p.add_argument('--minibatch-size',type=int,default=2048)
    p.add_argument('--partition-seed',type=int,default=71)
    a=p.parse_args()
    torch.set_num_threads(1)
    b=torch.load(a.batch,map_location='cpu',weights_only=True)
    result=analyze(a.checkpoint,b,a.minibatch_size,a.partition_seed,a.candidate)
    result['batch_sha256']=sha256(a.batch)
    result['diagnostic_source_sha256']=sha256(__file__)
    with open(a.out,'x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(output=a.out,n=result['n'],optimizer_steps=0,device='cpu')))

if __name__=='__main__': main()
