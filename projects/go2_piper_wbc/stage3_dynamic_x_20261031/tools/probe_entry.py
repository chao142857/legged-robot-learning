"""P1-only overlay of the original evaluator; no learned weights change."""
import hashlib,json,pathlib,sys
c=json.loads(pathlib.Path('/stage2-config.json').read_text());source=pathlib.Path(c['native_source']);native=source/'tools/evaluate_B0.py'
from isaacsim import SimulationApp
SimulationApp.DEFAULT_LAUNCHER_CONFIG=dict(SimulationApp.DEFAULT_LAUNCHER_CONFIG,disable_viewport_updates=True,enable_crashreporter=False)
sys.argv=[str(native),'--checkpoint','/eval/checkpoint.pt','--out','/runs/evaluation','--recipe-protocol',c['protocol'],'--admission',c['admission'],'--num_envs',str(c['num_envs']),'--seed',str(c['seed']),'--headless','--device','cuda:0','--kit_args','--/persistent/isaac/asset_root/cloud=/node-assets --/persistent/isaac/asset_root/default=/node-assets --/persistent/isaac/asset_root/nvidia=/node-assets --/app/extensions/registryEnabled=false']
text=native.read_text();needle='data, summary = core.evaluate(env,runner,out,protocol,args,input_callback=observer.capture_input)'
assert text.count(needle)==1
replacement='from dynamic_x_probe import install\n        probe=install(env,runner,observer,core,out)\n        protocol["P1_diagnostic_only"]=True\n        protocol["P1_intervention"]=probe.config\n        data, summary = core.evaluate(env,runner,out,protocol,args,input_callback=probe.capture_input)\n        probe.finish(data,summary)'
overlay=text.replace(needle,replacement)
mark="write_json(out/'protocol.json',protocol)"
assert overlay.count(mark)==1
overlay=overlay.replace(mark,"protocol['method']='P1 frozen actor plus explicit diagnostic-only intervention; not learned PPO success'\n    protocol['P1_probe_config']=json.loads(Path('/stage2-config.json').read_text())['probe']\n    "+mark)
pathlib.Path('/runs/probe_overlay.json').write_text(json.dumps(dict(original_evaluator_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),executed_overlay_sha256=hashlib.sha256(overlay.encode()).hexdigest(),formal_original_evaluator=False,diagnostic_intervention_not_policy_success=True),indent=2)+'\n')
exec(compile(overlay,str(native),'exec'),dict(__name__='__main__',__file__=str(native)))
