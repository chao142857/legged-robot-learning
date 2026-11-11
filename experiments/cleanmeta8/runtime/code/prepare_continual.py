"""Prepare new real counterfactual starts; inherited Teacher data is provenance only."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from common import ROOT,file_sha,write,digest,PROTOCOL
from cpu_hash import tree_hash
from neural_v01 import NeuralSampler
from fork_v01 import bootstrap_recent
from fork_v02 import partial_features,learning_progress
from teacher_v02 import Teacher,state_hash

def prepare_source(index):
    src=json.loads((ROOT/'sources.json').read_text())['sources'][index]
    p=Path(src['path']);assert file_sha(p)==src['sha256']
    saved=torch.load(p,map_location='cpu',weights_only=False);i=saved['infos'];u=i['completed_updates'];assert u==src['updates'] and i['student_seed']==46 and i['environment'] is not None
    assert i['pool_sha256']=='86f39ce6b71d17d84c47681fe5e44d0fd7efac605ef366c3d8e91916b3eb56db' and i['protocol_sha256']==digest(PROTOCOL)
    hashes={k:tree_hash(saved[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')};physical=tree_hash(i['environment']);rng_hash=tree_hash(i['rng'])
    original=copy.deepcopy(i['adapter']);owner=i['adapter']['owner'];eng=NeuralSampler(32768,0);eng.load_state_dict(owner['engine'])
    feedback=bootstrap_recent(eng,p,i) if src['family']=='v0' else dict(recent_records_from_complete_checkpoint=True)
    old_counts=eng.completed_counts.copy();old_history=eng.history.copy()
    eng.version=-1;eng.draws=[];eng.draw_end_steps={};eng.rng=np.random.default_rng(248000+index)
    owner['engine']=eng.state_dict();owner['teacher_version']=np.full(4096,-1,dtype=np.int64)
    owner['decision_id']=-np.arange(1,4097,dtype=np.int64);owner['selected_probability']=np.zeros(4096);owner['cycle_version_exposure']={}
    assert np.array_equal(eng.completed_counts,old_counts) and np.array_equal(eng.history,old_history)
    partial,partial_receipt=partial_features(original)
    now=owner['control_steps'];dynamic=eng.features(now,owner['transitions']);progress,po=learning_progress(eng,now)
    receipt=dict(source_checkpoint=str(p),source_checkpoint_sha256=src['sha256'],student_seed=46,completed_updates=u,sampler_seed=248000+index,feedback=feedback,old_inflight_teacher_version=-1,source_rng_preserved=True,source_actor_critic_optimizer_preserved=True,source_physical_state_preserved=True,source_feedback_not_replayed_as_teacher_gradient=True)
    i['v02_preparation']=receipt;i['base_update']=u
    i['v02_context']=dict(dynamic=dynamic,partial=partial,partial_receipt=partial_receipt,progress=progress,progress_observed=po,control_step=now,source_checkpoint_sha256=src['sha256'])
    assert all(tree_hash(saved[k])==v for k,v in hashes.items()) and tree_hash(i['environment'])==physical and tree_hash(i['rng'])==rng_hash
    out=ROOT/'starts'/f'seed46-cp{u:05d}';out.mkdir(parents=True,exist_ok=False);dest=out/'prepared.pt';torch.save(saved,dest)
    write(out/'source.json',src);write(out/'preparation.json',dict(receipt,prepared_checkpoint_sha256=file_sha(dest),actor_critic_optimizer_hashes=hashes,environment_sha256=physical,rng_sha256=rng_hash))
    assert file_sha(p)==src['sha256'];print('SOURCE_PREPARED',u,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-index',type=int);args=parser.parse_args()
    if args.source_index is not None:return prepare_source(args.source_index)
    old=ROOT/'teachers/old-meta.pt';assert file_sha(old)=='8929d0a13c4aaa7b98c70acf584d04390711bbc9ea6df8b61f3bcb556cdbd3b9'
    manifest=json.loads((ROOT/'prepared/manifest.json').read_text());static=np.load(ROOT/'prepared/command_features.npy',allow_pickle=False)
    t=Teacher(static,manifest['binding']);fresh=t.state();fresh_sha=state_hash(t.net.state_dict())
    torch.save(dict(teacher=fresh,purpose='Seed1048 initial Teacher; no real training experience.'),ROOT/'teachers/fresh.pt')
    a=torch.load(old,map_location='cpu',weights_only=False)['teacher'];t.load(a)
    assert t.updates==8 and t.used_versions==set(str(i) for i in range(8))
    assert tree_hash(t.state())==tree_hash(a) and t.optimizer.state_dict()['state']
    write(ROOT/'teachers/inheritance.json',dict(passed=True,old_teacher_file_sha256=file_sha(old),old_teacher_network_sha256=state_hash(t.net.state_dict()),old_teacher_state_sha256=tree_hash(a),old_updates=8,old_versions=sorted(t.used_versions),optimizer_state_retained=True,private_rng_retained=True,fresh_network_sha256=fresh_sha,new_decision_versions=list(range(8,16)),off_policy_reuse=False))
    print('INHERITANCE_PASS: Teacher 8 -> 16 planned real updates; new Student seed48 held out',flush=True)
if __name__=='__main__':main()
