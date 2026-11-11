"""Rebuild exact Teacher states from tiny deltas plus immutable cached encoder."""
import copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from common import ROOT,file_sha,write
from cpu_hash import tree_hash
from teacher_v02 import Teacher

p=ROOT/'teacher_delta_split.pt'
assert file_sha(p)=='2dfcf2a57b4e94865723b6be317a1bf15e2cbfba64725cb0184e2552fda2ee31'
a=torch.load(p,map_location='cpu',weights_only=False)
old_path=ROOT/'teachers/old-meta.pt';old=torch.load(old_path,map_location='cpu',weights_only=False)['teacher']
ref=a['reference'];assert file_sha(old_path)==ref['old_file_sha256'] and tree_hash(old)==ref['old_state_sha256']
for name in ('encoder','config','binding'):assert tree_hash(old[name])==ref[name+'_sha256']
manifest=json.loads((ROOT/'prepared/manifest.json').read_text());assert old['binding']==manifest['binding']
for name,key in [('prepared/command_features.npy','command_features_sha256'),('prepared/wbc_development.npz','development_pool_sha256'),('pools/wbc_train.npz','train_pool_sha256')]:assert file_sha(ROOT/name)==manifest['binding'][key]
receipts={}
for name,data in a['teachers'].items():
 state=copy.deepcopy(old);state.update(data['delta']);assert tree_hash(state)==data['state_sha256']
 t=Teacher(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False),manifest['binding']);t.load(state)
 assert tree_hash(t.state())==data['state_sha256']
 path=ROOT/'teachers'/f'{name}.pt';assert not path.exists()
 torch.save(dict(teacher=state,source_state_sha256=data['state_sha256'],source_file_sha256=data['source_file_sha256'],reconstructed_from_cached_encoder=True),path)
 receipts[name]=dict(state_sha256=tree_hash(state),file_sha256=file_sha(path),source_file_sha256=data['source_file_sha256'],teacher_updates=state['updates'])
(ROOT/'runs').mkdir(exist_ok=True)
write(ROOT/'runs/student_seed48_initialization.json',a['student_initialization'])
write(ROOT/'SOURCE_UNIFORM_REFERENCE.json',dict(initialization=a['student_initialization'],source_host=a['source_host'],source_eval_identities=a['source_eval_identities']))
write(ROOT/'PREFLIGHT.json',dict(passed=True,seed=48,old_teacher_state_sha256=ref['old_state_sha256'],old_teacher_updates=old['updates'],teachers=receipts,student_initial_weights_must_match_uniform=True,all_local_pool_and_features_hashes_match=True,delta_bytes=p.stat().st_size,split_hardware=dict(uniform='RTX5060 Laptop / ubuntu-robot',fresh_old_new='RTX3060 / yanglab'),source_protocol=a['source_protocol'],meta_experience_summary=a['extended_teacher_summary']))
print('TEACHER_STATES_EXACTLY_RECONSTRUCTED; FRESH0 OLD8 NEW16; STUDENT48_INITIAL_REFERENCE_BOUND',flush=True)
