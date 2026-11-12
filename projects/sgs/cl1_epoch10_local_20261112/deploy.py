"""Create an isolated experiment; never rewrite source runs."""
import pathlib,shutil,json,hashlib
R=pathlib.Path('/data/go2_piper_cl1_epoch10_local_20261112');S=pathlib.Path('/data/go2_piper_nsgs_cl1_20261109');G=pathlib.Path('/data/go2_piper_nsgs_cl1_gen_seed70159_20261111');PAY=pathlib.Path('/data/cl1_epoch10_payload_20261112')
assert not R.exists();R.mkdir();shutil.copytree(G/'code',R/'code');shutil.copytree(G/'tests',R/'tests')
for n in ['host.py','tmux.sh','evaluation_identity_development.json']:shutil.copy2(G/n,R/n)
for n in ['prepared','pools','parent','deps']:(R/n).symlink_to(G/n)
for n in ['experience','runs','teachers','jobs','evaluations','fragments']:(R/n).mkdir()
# Use the validated new-intervention runner, not same-policy resume-pending reuse.
shutil.copy2(S/'code/train_cl1.py',R/'code/train_cl1.py');shutil.copy2(S/'code/cl1_proxy.py',R/'code/cl1_proxy.py');shutil.copy2(S/'code/cl1_ops.py',R/'code/collection_rpc.py');shutil.copy2(S/'code/cl1_core.py',R/'code/cl1_core_original.py')
p=R/'code/cl1_core.py';t=(S/'code/cl1_core.py').read_text()
t=t.replace('def update(self,fragments):','def update(self,fragments,max_epochs=None):').replace('result=self._update(fragments)','result=self._update(fragments,max_epochs=max_epochs)').replace('def _update(self,fragments):','def _update(self,fragments,max_epochs=None):')
t=t.replace("for epoch in range(CONFIG['epochs']):", "epochs = CONFIG['epochs'] if max_epochs is None else max_epochs\n        assert type(epochs) is int and epochs in (2,10)\n        from cpu_hash import tree_hash\n        targets = tree_hash([(f['transitions'], adv, ret) for f,adv,ret in prepared])\n        for epoch in range(epochs):\n            assert tree_hash([(f['transitions'], adv, ret) for f,adv,ret in prepared]) == targets")
t=t.replace('macro_samples=len(aa),epochs=steps,','macro_samples=len(aa),epochs=steps,epochs_requested=epochs,fixed_targets_sha256=targets,')
p.write_text(t)
for n in ['experiment_ops.py','test_epochs.py']:shutil.copy2(PAY/n,R/'code'/n)
shutil.copy2(PAY/'controller.py',R/'controller.py')
plan={'experiment':'CL1 epoch budget local calibration','collection':[{'seed':s,'root':str(S),'run':f'base-seed{s}-to05000','start':5000,'updates':500} for s in [43,44]],'validation':[{'seed':s,'root':str(pathlib.Path('/data')/root),'run':f'uniform-seed{s}-to05000-a01','start':5000,'updates':200} for s,root in [(70158,'go2_piper_nsgs_cl1_gen_seed70158_fixed_20261110'),(70159,'go2_piper_nsgs_cl1_gen_seed70159_20261111')]],'teacher_source':str(S/'teachers/base/Base.pt'),'budget':2200,'envs':4096,'rollout':24,'collection_sigma':.2,'validation_execution':'frozen deterministic mean','primary_endpoint':200,'secondary_endpoint':100,'evaluation':'fixed development only; no formal test access','arms':['base','epoch2','epoch10'],'only_algorithm_change':'Teacher PPO max epochs2->10; all behavior logprob, GAE targets fixed within batch','selection':'fixed midpoint5000; no performance-based selection','terminal_for_collection_fragments':False,'no_automatic_long_training':True}
for item in plan['collection']+plan['validation']:
 run=pathlib.Path(item['root'])/'runs'/item['run'];item['checkpoint']=str(run/'checkpoints/update_05000.pt');item['memory']=str(run/'teacher_requests/0050-complete.teacher_after.pt')
 for k in ['checkpoint','memory']:item[k+'_sha256']=hashlib.sha256(pathlib.Path(item[k]).read_bytes()).hexdigest()
plan['teacher_source_sha256']=hashlib.sha256(pathlib.Path(plan['teacher_source']).read_bytes()).hexdigest();(R/'PROTOCOL.json').write_text(json.dumps(plan,indent=2))
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'controller.py',R/'PROTOCOL.json',R/'host.py',*(R/'code').glob('*.py')]};(R/'CODE_BINDING.json').write_text(json.dumps({'files':files},indent=2));(R/'source_manifest.json').write_text(json.dumps({'integration_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}));print(R)
