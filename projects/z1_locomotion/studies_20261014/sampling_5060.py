"""Opt-in terrain-stratified standing allocation; native stop/restart schedule stays intact."""
import torch

def terrain_groups(e):
    gen = e.cfg.terrain.terrain_generator
    assert gen.curriculum, 'This mapping requires the pinned curriculum generator'
    names = list(gen.sub_terrains)
    proportions = torch.tensor([gen.sub_terrains[n].proportion for n in names], device=e.device)
    cumulative = (proportions / proportions.sum()).cumsum(0)
    cols = e._terrain.terrain_types.long()
    thresholds = cols.float() / gen.num_cols + .001
    labels = (thresholds[:, None] >= cumulative[None, :]).sum(1)
    return names, labels

def apply_standing(e, env_ids=None):
    fraction = float(e.cfg.standing_env_fraction)
    assert 0 <= fraction < 1
    if not hasattr(e, '_standing_5060'):
        names, labels = terrain_groups(e)
        e._standing_5060 = torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
        e._standing_5060_labels = labels
        e._standing_5060_names = names
        for label in labels.unique():
            ids = torch.where(labels == label)[0]
            k = min(round(fraction * len(ids)), len(ids) - 1)
            e._standing_5060[ids[torch.randperm(len(ids), device=e.device)[:k]]] = True
    elif env_ids is not None:
        ids = torch.as_tensor(env_ids, device=e.device, dtype=torch.long)
        e._standing_5060[ids] = torch.rand(len(ids), device=e.device) < fraction
        for label in e._standing_5060_labels.unique():
            group = torch.where(e._standing_5060_labels == label)[0]
            if bool(e._standing_5060[group].all()):
                e._standing_5060[group[torch.randint(len(group), (), device=e.device)]] = False
    e._commands[e._standing_5060, :3] = 0

def exposure_snapshot(e):
    names, labels = terrain_groups(e)
    standing = getattr(e, '_standing_5060', torch.zeros(e.num_envs, dtype=torch.bool, device=e.device))
    moving = e._commands.norm(dim=1) > .01
    return {name: {'envs': int((labels == i).sum()),
                   'standing_mask': int(((labels == i) & standing).sum()),
                   'nonzero_commands': int(((labels == i) & moving).sum()),
                   'levels': e._terrain.terrain_levels[labels == i].tolist()}
            for i, name in enumerate(names) if bool((labels == i).any())}
