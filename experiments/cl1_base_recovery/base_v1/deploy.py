"""YangLab-only immutable-input CL1 setup. Existing experiments remain untouched."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path('/data/go2_piper_nsgs_cl1_20261109')
BASE=Path('/data/go2_piper_cleanmeta8_seed49_20261109')
PAYLOAD=Path(__file__).resolve().parent
assert not ROOT.exists();ROOT.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert json.loads((BASE/'STATUS.json').read_text())['phase']=='complete'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();assert not gpu,gpu
shutil.copytree(BASE/'code',ROOT/'code',ignore=shutil.ignore_patterns('__pycache__'))
for p in (PAYLOAD/'code').glob('*.py'):assert not (ROOT/'code'/p.name).exists();shutil.copy2(p,ROOT/'code'/p.name)
shutil.copytree(PAYLOAD/'tests',ROOT/'tests')
for name in ('prepared','pools','parent','deps'):(ROOT/name).symlink_to(BASE/name,target_is_directory=True)
for name in ('experience','runs','evaluations','teachers','preserved','runtime'): (ROOT/name).mkdir()
for name in ('STATUS.json','COMPARISON.json','IMPORT_ACCEPTANCE.json','REPORT.md','LAUNCH_RECEIPT.json'):
 if (BASE/name).exists():shutil.copy2(BASE/name,ROOT/'preserved'/name)
write(ROOT/'preserved/ASSETS.json',dict(source=str(BASE),teacher_sha256=sha(BASE/'teachers/CleanMeta8.pt'),
 source_checkpoints='All preserved at original path; no moves/overwrites',files={str(p.relative_to(BASE)):sha(p) for p in (BASE/'runs').glob('*/checkpoints/update_10000.pt')}))
for n in ('host.py','tmux.sh'):shutil.copy2(BASE/n,ROOT/n)
for n in ('controller.py','EXECUTION_PROTOCOL.md','PROTOCOL.json'):shutil.copy2(PAYLOAD/n,ROOT/n)
subprocess.run(['python3',str(ROOT/'code/build_runner.py')],check=True)
assert sha(ROOT/'code/teacher_v02.py')=='734721a9acb59651388b73b112fb0365dab8a593f00e107f4f79b715a3596f96'
files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'host.py',ROOT/'controller.py',*(ROOT/'code').glob('*.py'),*(ROOT/'tests').glob('*.py')]}
write(ROOT/'source_manifest.json',dict(integration_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files=files))
write(ROOT/'STATUS.json',dict(phase='prepared_not_started',source_seeds=[43,44],student_updates_each=10000,recovery_max_updates=4000))
print(ROOT)
