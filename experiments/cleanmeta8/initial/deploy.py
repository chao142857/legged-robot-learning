"""Run only on Ubuntu: create a new immutable-input experiment root."""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path('/home/alex/dog/go2_piper_cleanmeta8_20261109')
BASE=Path('/home/alex/dog/go2_piper_nsgs_v02_matched_seed48_20261108')
TR0=Path('/home/alex/dog/go2_piper_nsgs_tr0_20261109')
FIX=Path('/home/alex/dog/teacher_adam_audit_20261109/fixed_runtime')
PAYLOAD=Path('/home/alex/dog/cleanmeta8_deployment_20261109')
DATA=Path('/media/alex/Sharedata/go2_piper_cleanmeta8_20261109')
assert not ROOT.exists() and not DATA.exists();ROOT.mkdir();DATA.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2))
shutil.copytree(FIX,ROOT/'code',ignore=shutil.ignore_patterns('__pycache__'))
# The verified runtime is copied byte-for-byte, with observational orchestration additions.
for p in (PAYLOAD/'code').glob('*.py'):assert not (ROOT/'code'/p.name).exists();shutil.copy2(p,ROOT/'code'/p.name)
for name in ('prepared','pools','assets','parent','deps'):(ROOT/name).symlink_to(BASE/name,target_is_directory=True)
for name in ('runs','experience','evaluations'):
 (DATA/name).mkdir();(ROOT/name).symlink_to(DATA/name,target_is_directory=True)
(ROOT/'teachers').mkdir();shutil.copy2(BASE/'teachers/old-meta.pt',ROOT/'teachers/OldMeta8.pt')
shutil.copy2('/home/alex/dog/go2_piper_nsgs_v02_seed48_20261107/teachers/fresh.pt',ROOT/'teachers/Fresh.pt')
for name in ('sources.json','PROTOCOL.json','source_manifest.json','provenance.json'):shutil.copy2(BASE/name,ROOT/name)
shutil.copy2(TR0/'code/python.sh',ROOT/'code/python.sh');shutil.copy2(BASE/'tmux.sh',ROOT/'tmux.sh')
shutil.copy2(PAYLOAD/'controller.py',ROOT/'controller.py')
# Root-relative host wrapper; accept Path values without changing launch behavior.
s=(TR0/'host.py').read_text().replace('command=["bash",str(ROOT/"code/python.sh"),*arguments]','command=["bash",str(ROOT/"code/python.sh"),*map(str,arguments)]')
(ROOT/'host.py').write_text(s)
shutil.copy2(PAYLOAD/'original_fresh_identity.json',ROOT/'original_fresh_identity.json')
shutil.copy2(PAYLOAD/'PROTOCOL_CLEAN.json',ROOT/'PROTOCOL_CLEAN.json')
write(ROOT/'LEGACY_LABELS.json',dict(old_implementation=True,source_audit_commit='c43b36ea4cb3dbdaa3d0f72730c349188a6d5130',
 old_meta8=dict(path=str(BASE/'teachers/old-meta.pt'),sha256=sha(BASE/'teachers/old-meta.pt')),
 old_meta16A='/home/alex/dog/go2_piper_nsgs_v02_seed48_20261107/teachers/new-meta.pt',
 old_meta16B=str(BASE/'teachers/new-meta.pt'),old_student_results='Preserved in original directories; legacy Teacher implementation. No relabel-in-place or overwritten results.'))
files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'controller.py',ROOT/'host.py',*(ROOT/'code').glob('*.py'),ROOT/'code/python.sh']}
assert sha(ROOT/'code/teacher_v02.py')==sha(FIX/'teacher_v02.py')
write(ROOT/'CODE_BINDING.json',dict(files=files,fixed_runtime_source=str(FIX),fixed_runtime_manifest_sha256=sha(FIX/'ADAM_FIX_MANIFEST.json'),
 deployment_code=dict(implementation_fix='58a1e1a972f6e2404ff2d696e054d1eaed811a2b',audit='c43b36ea4cb3dbdaa3d0f72730c349188a6d5130')))
write(ROOT/'STATUS.json',dict(phase='prepared_not_started',meta_budget=2400,student_seed=48,student_arms=['OldMeta8','CleanMeta8'],student_updates_each=10000))
print(ROOT)
