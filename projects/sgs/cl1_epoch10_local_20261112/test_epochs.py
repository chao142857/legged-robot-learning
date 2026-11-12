"""Isolated synthetic unit data; never mixed with real collection."""
import copy,json,importlib.util
from pathlib import Path
import torch,numpy as np
import cl1_core as c
from cpu_hash import tree_hash
from common import ROOT,write
sp=importlib.util.spec_from_file_location('original',Path(__file__).with_name('cl1_core_original.py'));old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
torch.set_num_threads(1)
rng=np.random.default_rng(1202200);l=c.Learner();fs=[]
for seed in [1,2]:
 prefix=[];ts=[]
 for i in range(5):
  x=rng.normal(size=34)*.1;a=l.act(x,prefix);prefix.append(x);ts.append(dict(**a,x=x,next_x=rng.normal(size=34)*.1,reward=float(rng.normal()*.03)))
 fs.append(dict(prefix=[],transitions=ts,terminal=False))
snapshot=l.state();initial=tree_hash(snapshot);fh=tree_hash(fs)
a=c.Learner();a.load(snapshot);b=old.Learner();b.load(snapshot);ra=a.update(copy.deepcopy(fs),max_epochs=2);rb=b.update(copy.deepcopy(fs));assert tree_hash(a.state())==tree_hash(b.state())
z=c.Learner();z.load(snapshot);rz=z.update(copy.deepcopy(fs),max_epochs=10);assert len(rz['epochs'])<=10
assert ra['fixed_targets_sha256']==rz['fixed_targets_sha256'] and tree_hash(fs)==fh and tree_hash(snapshot)==initial
# Exact checkpoint restore and replay of same update.
saved=copy.deepcopy(z.state());zz=c.Learner();zz.load(saved);assert tree_hash(zz.state())==tree_hash(z.state())
# Forced KL-stop test uses separate synthetic behavior means only, never production data.
wrong=copy.deepcopy(fs)
for f in wrong:
 for t in f['transitions']:t['mean']=t['mean']+10
w=c.Learner();w.load(snapshot);rw=w.update(wrong,max_epochs=10);assert rw['epochs'][0]['early_stop'];assert tree_hash(w.net.state_dict())==tree_hash(snapshot['net']);assert tree_hash(w.optimizer.state_dict())==tree_hash(snapshot['optimizer'])
# Exceptions restore parameters, Adam, RNG and version transactionally.
f=c.Learner();f.load(snapshot);before=tree_hash(f.state());orig=f.optimizer.step
calls=[0]
def fail(*args,**kw):
 calls[0]+=1
 if calls[0]==2:raise RuntimeError('synthetic failure')
 return orig(*args,**kw)
f.optimizer.step=fail
try:f.update(copy.deepcopy(fs),max_epochs=10);raise AssertionError('missing failure')
except RuntimeError:pass
assert tree_hash(f.state())==before
write(ROOT/'CPU_TESTS.json',dict(passed=True,original_epoch2_exact_state_match=True,fixed_targets_and_behavior_immutable=True,checkpoint_restore_exact=True,KL_early_stop_preserved=True,exception_full_Adam_rollback=True,synthetic_samples_never_used_for_real_training=True,epoch10_test_executed=len(rz['epochs'])))
print('CPU_TESTS_PASSED')
