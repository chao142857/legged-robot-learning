"""Read-only audit of the original IIT sampler and live terrain assignment."""
import hashlib, inspect
from pathlib import Path
import torch

def terrain_groups(e):
    gen=e.cfg.terrain.terrain_generator
    assert gen.curriculum
    names=list(gen.sub_terrains)
    weights=torch.tensor([gen.sub_terrains[n].proportion for n in names],device=e.device)
    cumulative=(weights/weights.sum()).cumsum(0)
    columns=e._terrain.terrain_types.long()
    labels=(columns.float()[:,None]/gen.num_cols+.001 >= cumulative[None,:]).sum(1)
    return names,labels

def exposure_snapshot(e):
    names,labels=terrain_groups(e)
    forced=(torch.arange(e.num_envs,device=e.device)<500) & (e.num_envs>500)
    nonzero=e._commands.norm(dim=1)>.01
    return {name:{'envs':int((labels==i).sum()),
                  'legacy_forced_zero':int(((labels==i)&forced).sum()),
                  'nonzero_commands':int(((labels==i)&nonzero).sum()),
                  'zero_commands':int(((labels==i)&~nonzero).sum()),
                  'levels':e._terrain.terrain_levels[labels==i].tolist()}
            for i,name in enumerate(names) if bool((labels==i).any())}

def sampler_identity():
    from basic_locomotion_isaaclab.tasks import custom_events
    fn=custom_events._get_new_random_commands
    source=inspect.getsource(fn)
    path=Path(inspect.getsourcefile(fn))
    assert fn.__module__==custom_events.__name__ and 'num_fixed_envs = 500' in source
    assert 'standing_env_fraction' not in source, 'User requested original sampler for capacity smoke'
    return {'module':fn.__module__,'file':str(path),'function_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source':source,'legacy_rule':'Only num_envs>500: first500 fixed zero; <=500 does not trigger',
            'sampler_modified_this_round':False}
