"""Production-size CPU checkpoint tests and immutable-baseline audit; no PPO."""
import pathlib,json,copy,time,importlib.metadata
import numpy as np,torch
from common import ROOT,Pool,PROTOCOL,digest,file_sha,write
from grouped_sampling import GroupedSampler,GroupedOwnership

out=ROOT/'artifacts/tests_only';out.mkdir(parents=True,exist_ok=True);pool=Pool('wbc','train');results=[]
for path in sorted((ROOT/'artifacts/groups').glob('mapping_*.npy')):
    mapping=np.load(path);assert np.array_equal(np.bincount(mapping),np.full(4096,8))
    owner=GroupedOwnership(pool,4096,mapping,42);ids=np.arange(4096);owner.assign(ids);owner.advance();token=owner.token(ids)
    synthetic=((np.arange(4096)*7)%11<5).astype(int).tolist();owner.commit(token,synthetic);owner.assign(ids);owner.check()
    target=out/(path.stem+'_engineering_only.pt')
    torch.save(dict(engineering_only=True,never_use_for_training=True,owner=owner.state_dict(),torch_cpu_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),target)
    state=torch.load(target,map_location='cpu',weights_only=False);restored=GroupedOwnership(pool,4096,mapping,9);restored.load_state_dict(state['owner']);restored.check()
    a=copy.deepcopy(owner.engine);b=copy.deepcopy(restored.engine)
    assert a.sample(4096)==b.sample(4096);assert digest(a.state_dict())==digest(b.state_dict())
    assert digest(owner.state_dict())==digest(restored.state_dict())
    p=np.asarray(owner.engine.probabilities());q=owner.engine.group_probabilities()
    assert abs(p.sum()-1)<1e-12 and abs(q.sum()-1)<1e-12
    np.testing.assert_allclose(np.bincount(mapping,weights=p,minlength=4096),q,rtol=1e-13,atol=1e-15)
    start=time.perf_counter();samples=owner.engine.sample(4096);seconds=time.perf_counter()-start
    assert all(0<=i<32768 for i in samples)
    results.append(dict(mapping=path.name,mapping_sha256=owner.engine.mapping_sha256,passed=True,torch_checkpoint_sha256=file_sha(target),sampler_next4096_equal=True,ownership_exact=True,formula_normalized=True,sample4096_seconds=seconds,synthetic_state_separate_from_training=True))

baseline=[]
for root in (pathlib.Path('/reference'),pathlib.Path('/e1reference'),pathlib.Path('/continue_reference')):
    manifest=json.loads((root/'source_manifest.json').read_text())
    for n,h in manifest['files'].items():assert file_sha(root/'code'/n)==h,(str(root),n)
for label in ('flat-uniform','flat-sgs','flat-sgs-e1','wbc-uniform','wbc-sgs','wbc-sgs-e1'):
    run=pathlib.Path('/continue_reference/runs')/('continue-'+label);result=json.loads((run/'result.json').read_text())
    cp=run/'policy/model_14999.pt';h=file_sha(cp);assert h==result['checkpoint_sha256']
    baseline.append(dict(run=label,checkpoint_sha256=h,unchanged=True))
for n in ('common.py','commands.py','metrics.py','adapter.py','sampling.py','vendor_sampler.py'):
    assert file_sha(ROOT/'code'/n)==file_sha(pathlib.Path('/reference/code')/n),n
frozen=json.loads(pathlib.Path('/reference/protocol_frozen.json').read_text());assert frozen['definition_sha256']==digest(PROTOCOL)
write(ROOT/'artifacts/production_checks.json',dict(passed=True,tests=results,baseline_checkpoints=baseline,base_sources_unchanged=True,strict_protocol_sha256=digest(PROTOCOL),
    versions={k:importlib.metadata.version(k) for k in ('numpy','torch','isaaclab','isaacsim','rsl-rl-lib')},PPO_updates=0,engineering_checkpoint_scope='CPU serialization and4096-env ownership simulation, not PhysX or a PPO training smoke test.'))
print('PRODUCTION_CHECKS_PASS '+json.dumps(results),flush=True)
