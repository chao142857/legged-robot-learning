"""Original evaluator overlay: frozen component diagnosis, never formal PASS."""
import hashlib,json,pathlib,sys
c=json.loads(pathlib.Path('/stage2-config.json').read_text());native=pathlib.Path(c['native_source'])/'tools/evaluate_B0.py'
from isaacsim import SimulationApp
SimulationApp.DEFAULT_LAUNCHER_CONFIG=dict(SimulationApp.DEFAULT_LAUNCHER_CONFIG,disable_viewport_updates=True,enable_crashreporter=False)
sys.argv=[str(native),'--checkpoint','/eval/checkpoint.pt','--out','/runs/evaluation','--recipe-protocol',c['protocol'],'--admission',c['admission'],'--num_envs',str(c['num_envs']),'--seed',str(c['seed']),'--headless','--device','cuda:0','--kit_args','--/persistent/isaac/asset_root/cloud=/node-assets --/persistent/isaac/asset_root/default=/node-assets --/persistent/isaac/asset_root/nvidia=/node-assets --/app/extensions/registryEnabled=false']
s=native.read_text()
if c.get('actor_recipe'):
    needle="    runner.load(args.checkpoint, load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False), strict=True, map_location=agent.device)";assert s.count(needle)==1
    s=s.replace(needle,"    from synergy_actor import attach\n    recipe=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['infos']['stage3_actor_recipe']\n    assert recipe==json.loads(Path('/stage2-config.json').read_text())['actor_recipe']\n    attach(runner.alg,recipe,optimizer=False)\n"+needle)
needle='data, summary = core.evaluate(env,runner,out,protocol,args,input_callback=observer.capture_input)';assert s.count(needle)==1
s=s.replace(needle,'from component_probe import install\n        probe=install(env,runner,observer,core,out)\n        data, summary = core.evaluate(env,runner,out,protocol,args,input_callback=probe.capture_input)\n        probe.finish(data,summary)')
needle="protocol['method']='B0: Vanilla WBC + Uniform'";assert s.count(needle)==1
s=s.replace(needle,"protocol['method']='Frozen component diagnostic only, NOT trained-policy success'\n    protocol['component_probe']=json.loads(Path('/stage2-config.json').read_text())['component_probe']")
pathlib.Path('/runs/component_overlay.json').write_text(json.dumps(dict(original_native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),executed_overlay_sha256=hashlib.sha256(s.encode()).hexdigest(),diagnostic_only=True,optimizer_updates=0,formal_acceptance=False),indent=2)+'\n')
exec(compile(s,str(native),'exec'),dict(__name__='__main__',__file__=str(native)))
