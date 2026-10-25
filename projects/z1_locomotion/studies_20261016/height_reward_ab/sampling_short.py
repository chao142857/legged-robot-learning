"""Opt-in standing Bernoulli per (seed, env ID, episode ordinal), without consuming RNG."""
import torch
from sampler_audit import terrain_groups

def episode_uniform(ids, episodes, seed):
    # 31-bit integer mixing stays within signed int64 on CPU and CUDA.
    x = ((ids + 1) * 1103515245 + (episodes + 1) * 12345 + int(seed) * 2654435761) & 0x7fffffff
    x = ((x ^ (x >> 16)) * 73244475) & 0x7fffffff
    x = ((x ^ (x >> 16)) * 73244475) & 0x7fffffff
    x = x ^ (x >> 16)
    return x.to(torch.float64) / 2147483648.

def apply_standing(e, env_ids=None):
    fraction = float(e.cfg.standing_env_fraction)
    assert 0 <= fraction < 1
    fresh = not hasattr(e, '_standing_episode_ordinals')
    if fresh:
        e._standing_episode_ordinals = torch.full((e.num_envs,), -1, dtype=torch.long, device=e.device)
        e._standing_short = torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    if env_ids is not None or fresh:
        ids = (torch.arange(e.num_envs, device=e.device) if env_ids is None
               else torch.as_tensor(env_ids, device=e.device, dtype=torch.long))
        e._standing_episode_ordinals[ids] += 1
        e._standing_short[ids] = episode_uniform(ids, e._standing_episode_ordinals[ids], e.cfg.standing_seed) < fraction
    e._commands[e._standing_short, :3] = 0.

def exposure_snapshot(e):
    names, labels = terrain_groups(e)
    motion = e._commands.norm(dim=1) > .01
    return {name: {'envs': int((labels == i).sum()),
                   'standing': int(((labels == i) & e._standing_short).sum()),
                   'nonzero_commands': int(((labels == i) & motion).sum()),
                   'levels': e._terrain.terrain_levels[labels == i].tolist()}
            for i, name in enumerate(names) if bool((labels == i).any())}
