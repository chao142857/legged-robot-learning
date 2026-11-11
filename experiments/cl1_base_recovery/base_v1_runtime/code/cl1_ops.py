"""CPU-only transactional Teacher decisions and outer updates, append-only evidence."""
import argparse,copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from cl1_core import Learner,observation,replay
from teacher_v02 import FeatureEncoder,Teacher as Decoder
from neural_core import distribution_stats
from common import ROOT,write,file_sha
from cpu_hash import tree_hash

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def read(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('op',choices=['init','request','update','fork-memory','finish-fragment']);ap.add_argument('--stage',required=True);ap.add_argument('--request',type=Path);ap.add_argument('--batch',type=int);ap.add_argument('--length',type=int,default=5);ap.add_argument('--source-memory',type=Path);ap.add_argument('--seed',type=int);a=ap.parse_args()
    import cl1_core,teacher_v02
    identity={m.__name__:dict(path=m.__file__,sha256=file_sha(m.__file__)) for m in (cl1_core,teacher_v02)}
    assert identity['teacher_v02']['sha256']=='734721a9acb59651388b73b112fb0365dab8a593f00e107f4f79b715a3596f96'
    identity['operation']=a.op;identity['stage']=a.stage
    identity_path=(a.request.with_suffix('.runtime.json') if a.request else ROOT/'teachers'/a.stage/f'{a.op}-{a.batch}-{a.seed}.runtime.json')
    write(identity_path,identity)
    root=ROOT/'teachers'/a.stage;path=root/'current.pt';learner=Learner()
    if a.op=='init':
        if a.stage=='base':
            from smoke_acceptance import check
            check()
        assert not path.exists(); root.mkdir(parents=True,exist_ok=True)
        if a.stage=='recovery': learner.load(read(ROOT/'teachers/base/Base.pt')['learner'])
        state=dict(learner=learner.state(),students={str(s):dict(history=[],pending=None,previous=None,transitions=[],cursor=0) for s in (43,44)},batches=0)
        save(path,state);save(root/'initial.pt',state)
        save(root/'proxy.pt',dict(teacher=dict(cl1_proxy=1,binding=json.loads((ROOT/'prepared/manifest.json').read_text())['binding'])));return
    state=read(path);learner.load(state['learner'])
    if a.op=='request':
        request=read(a.request);student=state['students'][str(request['seed'])];v=request['version']
        if request['kind']=='decide':
            assert student['pending'] is None
            x=observation(request['global_state'],request['dev'],student['previous'],request['dynamic'])
            if student['transitions']:
                assert np.allclose(x,student['transitions'][-1]['next_x'],atol=1e-6), 'Boundary observation drift'
            action=learner.act(x,student['history'])
            static=np.load(ROOT/'prepared/command_features.npy',allow_pickle=False)
            encoder=FeatureEncoder(static)
            encoded=encoder.encode(request['dynamic'],request['progress'],request['progress_observed'],request['partial'])
            decoder=object.__new__(Decoder);decoder.encoder=encoder;decoder.config=encoder.config
            q=decoder.probability_from_preference(encoded['basis'],action['action'])
            t=dict(**action,x=x,development_before=request['dev'],macro_version=v)
            student['pending']=t;student['history'].append(x)
            response=dict(version=v,**encoded,global_state=request['global_state'],preference=action['action'],
                probability=q,distribution=distribution_stats(q),macro_action=action,teacher_version=learner.version,
                teacher_sha256=tree_hash(learner.net.state_dict()),deterministic_frozen_mean=False)
        else:
            t=student['pending'];assert t is not None and t['macro_version']==v
            assert t['version']==learner.version
            reward=float(request['dev']['continuous_joint_score']-t['development_before']['continuous_joint_score'])
            prev=dict(dev=t['development_before'],action=t['action'])
            t.update(reward=reward,development_after=request['dev'],completed_updates=request['checkpoint_update'],
                next_x=observation(request['global_state'],request['dev'],prev,request['dynamic']))
            student['transitions'].append(t);student['pending']=None;student['previous']=prev
            response=dict(passed=True,version=learner.version,reward=reward)
        state['learner']=learner.state()
        # Every decision/outcome has immutable before/after snapshots, including RNG, full optimizer, memory and raw transitions.
        evidence=a.request.with_suffix('.teacher_after.pt');assert not evidence.exists()
        save(evidence,state);save(path,state);save(a.request.with_suffix('.done.pt'),response)
    elif a.op=='update':
        assert a.batch==state['batches']
        fragments=[]
        for s in ([] if a.stage=='recovery' else state['students'].values()):
            assert s['pending'] is None
            ts=s['transitions'][s['cursor']:];assert len(ts)==a.length
            assert len(s['history'])==len(s['transitions'])
            fragments.append(dict(prefix=s['history'][:s['cursor']],transitions=ts,
                terminal=ts[-1]['completed_updates']==10000))
        if a.stage=='recovery':
            fragments=state.pop('fragments');assert len(fragments)==4
        before=copy.deepcopy(state);result=learner.update(fragments)
        if a.stage=='smoke':
            assert result['macro_samples']==4 and result['parameter_delta']>0
            assert any(e.get('gradient_norm',0)>0 for e in result['epochs'])
        for s in state['students'].values():
            s['cursor']=len(s['transitions']);s['rebuilt_hidden']=replay(learner.net,s['history']).numpy()
        state['learner']=learner.state();state['batches']+=1
        p=root/f'batch_{a.batch:03d}';p.mkdir(exist_ok=False)
        save(p/'before.pt',before);save(p/'after.pt',state);save(p/'on_policy_fragments.pt',fragments)
        write(p/'result.json',result);save(path,state)
        from cost_breakdown import report
        report()
    elif a.op=='fork-memory':
        assert a.stage=='recovery'
        original=read(a.source_memory)['students'][str(a.seed)]
        assert original['pending'] is None
        state['students'][str(a.seed)]=copy.deepcopy(original)
        state['students'][str(a.seed)]['cursor']=len(original['transitions'])
        save(path,state)
    elif a.op=='finish-fragment':
        assert a.stage=='recovery'
        s=state['students'][str(a.seed)];ts=s['transitions'][s['cursor']:]
        assert len(ts)==5 and s['pending'] is None
        state.setdefault('fragments',[]).append(dict(prefix=s['history'][:s['cursor']],transitions=ts,terminal=True))
        s['cursor']=len(s['transitions']);save(path,state)
if __name__=='__main__':main()
