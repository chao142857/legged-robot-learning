"""Original B0 evaluator with only explicit external checkpoint/output/seed."""
import json,pathlib,runpy,sys
c=json.loads(pathlib.Path('/stage2-config.json').read_text());source=pathlib.Path(c['native_source'])
from isaacsim import SimulationApp
SimulationApp.DEFAULT_LAUNCHER_CONFIG=dict(SimulationApp.DEFAULT_LAUNCHER_CONFIG,disable_viewport_updates=True,enable_crashreporter=False)
sys.argv=[str(source/'tools/evaluate_B0.py'),'--checkpoint','/eval/checkpoint.pt','--out','/runs/evaluation','--recipe-protocol',c['protocol'],'--admission',c['admission'],'--num_envs',str(c['num_envs']),'--seed',str(c['seed']),'--headless','--device','cuda:0','--kit_args','--/persistent/isaac/asset_root/cloud=/node-assets --/persistent/isaac/asset_root/default=/node-assets --/persistent/isaac/asset_root/nvidia=/node-assets --/app/extensions/registryEnabled=false']
if c.get('actor_recipe') is None:
    runpy.run_path(str(source/'tools/evaluate_B0.py'),run_name='__main__')
else:
    # Only construct the declared network before the original strict native load.
    # All original commands, physics, rewards, safety and metric code execute intact.
    import hashlib
    native=source/'tools/evaluate_B0.py';text=native.read_text()
    needle="    runner.load(args.checkpoint, load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False), strict=True, map_location=agent.device)"
    assert text.count(needle)==1
    replacement="    from synergy_actor import attach\n    recipe=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['infos']['stage3_actor_recipe']\n    assert recipe==json.loads(Path('/stage2-config.json').read_text())['actor_recipe']\n    attach(runner.alg,recipe,optimizer=False)\n"+needle
    overlay=text.replace(needle,replacement)
    pathlib.Path('/runs/evaluator_network_loader.json').write_text(json.dumps(dict(original_evaluator_sha256=hashlib.sha256(text.encode()).hexdigest(),executed_overlay_sha256=hashlib.sha256(overlay.encode()).hexdigest(),change='Explicit actor class construction only before original strict load',physics_metrics_and_safety_unchanged=True),indent=2)+'\n')
    exec(compile(overlay,str(native),'exec'),dict(__name__='__main__',__file__=str(native)))
