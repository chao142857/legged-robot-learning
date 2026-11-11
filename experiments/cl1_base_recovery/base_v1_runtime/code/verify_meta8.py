import json,torch,numpy as np
from common import ROOT,file_sha,write
from cpu_hash import tree_hash
from teacher_v02 import Teacher
s=torch.load(ROOT/'teachers/old-meta.pt',map_location='cpu',weights_only=False)['teacher']
assert s['updates']==8 and set(s['used_versions'])==set(str(i) for i in range(8))
binding=json.loads((ROOT/'prepared/manifest.json').read_text())['binding']
t=Teacher(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False),binding);t.load(s)
assert tree_hash(t.state())==tree_hash(s)
write(ROOT/'META8_PREFLIGHT.json',dict(passed=True,teacher_updates=8,teacher_state_sha256=tree_hash(s),file_sha256=file_sha(ROOT/'teachers/old-meta.pt'),student_seed=49,source_student_checkpoint=None,frozen=True))
print('META8_FROZEN_STUDENT49_READY',flush=True)
