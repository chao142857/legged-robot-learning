"""Final evidence integrity, read-only historical assets."""
from pathlib import Path
import json,hashlib,sys,subprocess,platform
root=Path(sys.argv[1]);files={}
for p in (root/'evidence').glob('*_final.json'):
 for r in json.loads(p.read_text())['rows']:files[r['checkpoint']]=r['checkpoint_sha256']
for name in ('UBUNTU_CHECKPOINTS.json','YANGLAB_CHECKPOINTS.json'):
 p=root/'evidence'/name
 if p.exists():
  for r in json.loads(p.read_text()):files[r['path']]=r['sha256']
p=root/'evidence/V0_V01.json'
if p.exists():
 d=json.loads(p.read_text())
 for r in d['cycles']:files[r['path']]=r['checkpoint_sha256']
 for r in d['sources']:files[r['path']]=r['sha256']
for p,h in files.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
imports=None
if (root/'fixed_runtime').exists():
 sys.path.insert(0,str(root/'fixed_runtime'))
 import teacher_v02,teacher_ops,prepare_continual,prepare_matched
 imports=dict(teacher=teacher_v02.__file__,teacher_ops=teacher_ops.Teacher is teacher_v02.Teacher,
  prepare_continual=prepare_continual.Teacher is teacher_v02.Teacher,prepare_matched=prepare_matched.Teacher is teacher_v02.Teacher)
 assert all(imports[k] for k in ('teacher_ops','prepare_continual','prepare_matched'))
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for sub in ('code','tests') for p in (root/sub).glob('*.py')}
r=dict(passed=True,historical_files_hash_rechecked=len(files),no_historical_mutation=True,canonical_imports=imports,code_hashes=manifest,
 python=platform.python_version(),source_commit='58a1e1a972f6e2404ff2d696e054d1eaed811a2b',verification_script_added_after_code_commit=True)
(root/'evidence/FINAL_INTEGRITY.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
