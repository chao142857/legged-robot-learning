"""Read-only local binding checks; no old Student/model/optimizer is loaded."""
import json
import numpy as np
import torch
from common import ROOT,file_sha,write
from cpu_hash import tree_hash
from teacher_v02 import Teacher
manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
binding=manifest['binding'];sources=json.loads((ROOT/'CORE_SOURCE_BINDING.json').read_text())
for rel,expected in sources['files'].items():assert file_sha(ROOT/rel)==expected,rel
revision=json.loads((ROOT/'CHECKER_REVISION.json').read_text()) if (ROOT/'CHECKER_REVISION.json').exists() else None
if revision:
 for rel,expected in revision['new_files'].items():assert file_sha(ROOT/rel)==expected,rel
for rel,key in [('prepared/command_features.npy','command_features_sha256'),('prepared/wbc_development.npz','development_pool_sha256'),('pools/wbc_train.npz','train_pool_sha256')]:assert file_sha(ROOT/rel)==binding[key]
assert file_sha(ROOT/'pools/wbc_evaluation.npz')=='d48b5850285e26b9cec447a0bb4bd6c84493afec5efa0c7725ddcbacbd714afa'
snapshot=torch.load(ROOT/'teachers/fresh.pt',map_location='cpu',weights_only=False)['teacher']
t=Teacher(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False),binding);t.load(snapshot)
assert t.updates==0 and not t.used_versions and not t.optimizer.state_dict()['state']
assert tree_hash(t.state())==tree_hash(snapshot)
write(ROOT/'PREFLIGHT.json',dict(passed=True,student_seed=49,training_sources_and_resumes=[],fresh_teacher_updates=0,fresh_teacher_state_sha256=tree_hash(snapshot),fresh_teacher_file_sha256=file_sha(ROOT/'teachers/fresh.pt'),all_training_and_sampling_core_hashes_match_prior_executed_code=True,read_only_checker_revision=revision,all_pools_and_features_match=True,strict='Native5cm/80%, mean actions, fixed1024 formal test',envs=4096,rollout=24,target_each=10000,arms=['uniform','fresh'],history_starts_empty=True,teacher_frozen=True,first_uniform_model_initialization_becomes_reference_for_all_seed49_arms=True))
print('SEED49_PREFLIGHT_PASS: Uniform/Fresh from zero; freshTeacher0; same fixed pools/config/code',flush=True)
