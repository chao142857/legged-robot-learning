"""Original B0 evaluator with only explicit external checkpoint/output/seed."""
import json,pathlib,runpy,sys
c=json.loads(pathlib.Path('/stage2-config.json').read_text());source=pathlib.Path(c['native_source'])
from isaacsim import SimulationApp
SimulationApp.DEFAULT_LAUNCHER_CONFIG=dict(SimulationApp.DEFAULT_LAUNCHER_CONFIG,disable_viewport_updates=True,enable_crashreporter=False)
sys.argv=[str(source/'tools/evaluate_B0.py'),'--checkpoint','/eval/checkpoint.pt','--out','/runs/evaluation','--recipe-protocol',c['protocol'],'--admission',c['admission'],'--num_envs',str(c['num_envs']),'--seed',str(c['seed']),'--headless','--device','cuda:0','--kit_args','--/persistent/isaac/asset_root/cloud=/node-assets --/persistent/isaac/asset_root/default=/node-assets --/persistent/isaac/asset_root/nvidia=/node-assets --/app/extensions/registryEnabled=false']
runpy.run_path(str(source/'tools/evaluate_B0.py'),run_name='__main__')
