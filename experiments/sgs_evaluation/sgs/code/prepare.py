import hashlib,json,pathlib,copy,math,numpy as np
from common import ROOT,PARAMETERS,PROTOCOL,write,digest,file_sha
from vendor_sampler import TaskSampler
from sampling import CachedSampler,Ownership

def tests():
    rng=np.random.default_rng(123)
    a=TaskSampler(32,42,mode='sgs',**PARAMETERS);b=CachedSampler(32,42,'sgs')
    assert a.outcomes==[[] for _ in range(32)] and b.outcomes==a.outcomes
    for _ in range(5):
        ids=rng.integers(0,32,1000).tolist();ys=rng.integers(0,2,1000).tolist();a.record(ids,ys);b.record(ids,ys)
        np.testing.assert_allclose(a.probabilities(),b.probabilities(),rtol=1e-13,atol=1e-15)
        assert a.sample(1000)==b.sample(1000)
    state=b.state_dict();c=CachedSampler(32,42,'sgs');c.load_state_dict(state);assert c.sample(1000)==b.sample(1000)
    synthetic=CachedSampler(3,7,'sgs');synthetic.record([0]*100+[1]*100+[2]*100,[False]*100+[True]*100+[False,True]*50)
    p=synthetic.probabilities();assert p[2]>p[0] and p[2]>p[1] and abs(p[0]-p[1])<1e-12
    class P:n=32;identity='synthetic-only'
    owner=Ownership(P(),4,'sgs');owner.assign(np.arange(4));owner.advance();token=owner.token(np.arange(4));owner.commit(token,[1,0,1,0])
    try:owner.commit(token,[1,0,1,0]);raise RuntimeError('Duplicate was accepted')
    except AssertionError:pass
    owner.assign(np.arange(4));bad=owner.token(np.arange(4));bad['pool_identity']='wrong-task'
    try:owner.commit(bad,[0]*4);raise RuntimeError('Foreign feedback was accepted')
    except AssertionError:pass
    owner.check()
    return dict(passed=True,core_equivalence=True,history_and_rng_restore=True,duplicate_and_foreign_feedback_rejected=True,synthetic_weights=p,synthetic_not_in_any_training_state=True)

if __name__=='__main__':
    (ROOT/'pools').mkdir(exist_ok=True);rows=[]
    for k,seed in [('flat',721101),('wbc',721102)]:
        for split,n,s in [('train',32768,seed),('evaluation',1024,seed+1000000)]:
            p=ROOT/'pools'/(k+'_'+split+'.npz');assert not p.exists()
            rng=np.random.default_rng(s);ee=rng.random((n,4,7),dtype=np.float32);vel=rng.random((n,4,5),dtype=np.float32)
            np.savez_compressed(p,ee=ee,velocity=vel)
            rows.append(dict(task=k,split=split,sequences=n,seed=s,sha256=file_sha(p),shape_ee=list(ee.shape),shape_velocity=list(vel.shape),standing_input_fraction_by_event=(vel[:,:,3]<=.1).mean(axis=0).tolist(),semantics='Fixed independent U[0,1) random-input sequences; transform with actual live native ranges. EE interval 8+2*u seconds; velocity interval exactly10s; native Bernoulli standing input<=0.1; WBC quaternion computed at runtime from robot state and raw euler inputs. Four events per term cover native20s maximum episode.'))
    write(ROOT/'pools/manifest.json',rows);write(ROOT/'audit/success_definition_candidate.json',PROTOCOL)
    write(ROOT/'audit/cpu_engineering_tests.json',tests())
    print('PREPARE_PASS '+json.dumps(dict(pools=rows,success_definition_sha=digest(PROTOCOL))),flush=True)
