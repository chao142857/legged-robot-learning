import json,io,copy,hashlib,inspect
from pathlib import Path
import numpy as np,torch
from common import ROOT,write,file_sha
from teacher_v02 import Teacher
from cpu_hash import tree_hash
s=torch.load(ROOT/'teachers/Fresh.pt',weights_only=False,map_location='cpu')['teacher']
t=Teacher(np.load(ROOT/'prepared/command_features.npy'),s['binding']);t.load(s)
assert tree_hash(t.state())=='b312ce7657bb6a7497e4dfe0062fe996fff0c6c653bd4836cd929d74b47fe2c2'
payload=copy.deepcopy(s);encoder=payload.pop('encoder');buf=io.BytesIO();torch.save(payload,buf);assert len(buf.getvalue())<262144
buf.seek(0);restored=torch.load(buf,weights_only=False);restored['encoder']=copy.deepcopy(encoder);assert tree_hash(restored)==tree_hash(s)
identity=Path(inspect.getfile(Teacher));assert identity.resolve()==(ROOT/'code/teacher_v02.py').resolve()
assert 'copy.deepcopy(old_optimizer)' in identity.read_text()
cat=json.loads((ROOT/'baseline_catalog.json').read_text());assert cat['seed']==49
for arm in ('uniform','fresh'):assert [v['update'] for v in cat['curves'][arm]]==list(range(1000,10001,1000))
assert json.loads((ROOT/'runs/student_seed49_initialization.json').read_text())==cat['initialization']
write(ROOT/'CPU_PREFLIGHT.json',dict(passed=True,actual_teacher_source=str(identity),teacher_source_sha256=file_sha(identity),fresh_identity_exact=True,compact_roundtrip_exact=True,baseline_counts={k:len(v) for k,v in cat['curves'].items()},no_teacher_retraining=True,student_seed=49,student_updates=10000))
print('SEED49_CLEANMETA8_PREFLIGHT_PASS')
