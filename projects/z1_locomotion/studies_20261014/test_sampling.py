"""CPU-only test of native scheduled stop/restart and small-size terrain coverage."""
from publication_paths import research_path
import json, types, torch
from pathlib import Path
from sampling_5060 import exposure_snapshot
base=Path(research_path('z1_cp5700_isaaclab_5060'))
path=base/'workspace/basic-locomotion-isaaclab/source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks/custom_events.py'
code=path.read_text().split('def _sample_random_commands',1)[1]
namespace={'torch':torch};exec('def _sample_random_commands'+code,namespace)
weights={'flat':.1,'boxes':.05,'star':0.,'random_rough':.05,'hf_pyramid_slope':.1,
         'hf_pyramid_slope_inv':.1,'pyramid_stairs':.3,'pyramid_stairs_inv':.3}
records=[]
for n in (128,256,512,1024,2048,4096):
 torch.manual_seed(42)
 gen=types.SimpleNamespace(curriculum=True,num_cols=20,sub_terrains={k:types.SimpleNamespace(proportion=v) for k,v in weights.items()})
 e=types.SimpleNamespace(num_envs=n,device='cpu',_commands=torch.zeros(n,3),
     cfg=types.SimpleNamespace(standing_env_fraction=.1,terrain=types.SimpleNamespace(terrain_generator=gen)),
     _terrain=types.SimpleNamespace(terrain_types=torch.floor(torch.arange(n)/(n/20)).long(),terrain_levels=torch.zeros(n,dtype=torch.long)),
     max_episode_length=1000,episode_length_buf=torch.full((n,),100))
 ids=torch.arange(n);fn=namespace['_get_new_random_commands'];fn(e,ids)
 initial=exposure_snapshot(e)
 assert len(initial)==7 and all(v['nonzero_commands']>0 for v in initial.values())
 assert bool(e._commands[:min(n,500)].norm(dim=1).gt(.01).any())
 e.episode_length_buf[:]=750;fn(e);assert bool((e._commands==0).all())
 e.episode_length_buf[:]=850;fn(e);restart=exposure_snapshot(e)
 assert all(v['nonzero_commands']>0 for v in restart.values())
 assert float(e._commands[:,0].abs().max())<=.5
 records.append({'num_envs':n,'initial':initial,'restart':restart,'scheduled_stop_all_zero':True})
# Archived task retains its historical fixed-index behavior.
del e.cfg.standing_env_fraction
fn(e,ids);assert bool((e._commands[:500]==0).all())
out={'passed':True,'seed':42,'device':'cpu','sizes':records,'legacy_branch_preserved':True,
     'note':'CPU command schedule test only; live terrain and exposure must also be checked during native PPO'}
(base/'audits/sampling-cpu-test.json').write_text(json.dumps(out,indent=2))
print('SAMPLING_CPU_TEST_PASSED')
