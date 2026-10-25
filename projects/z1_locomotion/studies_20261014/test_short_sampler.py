from publication_paths import research_path
import json, torch
from pathlib import Path
from types import SimpleNamespace
from sampling_short import apply_standing, episode_uniform
def env():
    return SimpleNamespace(cfg=SimpleNamespace(standing_env_fraction=.061,standing_seed=42),
                           num_envs=4096,device='cpu',_commands=torch.ones(4096,3),
                           episode_length_buf=torch.zeros(4096,dtype=torch.long),max_episode_length=1000)
e=env();ids=torch.arange(4096)
random_before=torch.random.get_rng_state().clone();apply_standing(e,ids)
assert torch.equal(random_before,torch.random.get_rng_state())
mask=e._standing_short.clone();counts=e._standing_episode_ordinals.clone()
e._commands[:]=1;apply_standing(e)
assert torch.equal(mask,e._standing_short) and torch.equal(counts,e._standing_episode_ordinals)
other=env();apply_standing(other,ids.flip(0));assert torch.equal(mask,other._standing_short)
apply_standing(e,ids[:100]);apply_standing(other,ids[:100].flip(0))
assert torch.equal(e._standing_short,other._standing_short)
assert torch.equal(e._standing_short[100:],mask[100:])
first=mask  # Core-only test; native scheduled stop/restart needs the external runtime.
periods=ids[:,None];episodes=torch.arange(100)[None,:]
uniform=episode_uniform(periods,episodes,42)
long_ratio=float((uniform<.061).double().mean())
assert abs(long_ratio-.061)<.002
audit={'passed':True,'seed':42,'standing_fraction':.061,'initial_count':int(first.sum()),
       'initial_fraction':float(first.float().mean()),'hundred_episode_fraction':long_ratio,
       'reset_order_invariant':True,'mask_held_until_own_reset':True,'global_rng_unchanged':True,
       'native_stop_and_restart_tested':False,'scope':'asset-free standing sampler invariants'}
print(json.dumps(audit))
