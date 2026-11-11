from pathlib import Path
import json,hashlib,shutil,subprocess
REPAIR=Path('/data/go2_piper_cl1_repair_20261110')
OLD=Path('/data/go2_piper_nsgs_cl1_gen_seed70158_20261110')
ROOT=Path('/data/go2_piper_nsgs_cl1_gen_seed70158_fixed_20261110')
PAYLOAD=REPAIR/'payload'
assert not ROOT.exists();ROOT.mkdir()
shutil.copytree(OLD/'code',ROOT/'code');shutil.copytree(OLD/'tests',ROOT/'tests')
for name in ('host.py','tmux.sh','evaluation_identity_development.json','evaluation_identity_test.json'):shutil.copy2(OLD/name,ROOT/name)
for name in ('prepared','pools','parent','deps'):(ROOT/name).symlink_to(OLD/name,target_is_directory=True)
for name in ('runs','experience','teachers','jobs','evaluations','resume_inputs','segments'):(ROOT/name).mkdir()
shutil.copy2(OLD/'runs/student_seed70158_initialization.json',ROOT/'runs/student_seed70158_initialization.json')
for p in (PAYLOAD/'code').glob('*.py'):
 if p.name.endswith('_before.py'):continue
 shutil.copy2(p,ROOT/'code'/p.name)
for p in (PAYLOAD/'tests').glob('*.py'):shutil.copy2(p,ROOT/'tests'/p.name)
for name in ('gen_fixed_controller.py','install_recovery.py'):shutil.copy2(PAYLOAD/name,ROOT/name)
shutil.copy2(OLD/'code/train_cl1.py',ROOT/'code/train_cl1_legacy.py')
shutil.copy2(OLD/'code/cl1_proxy.py',ROOT/'code/cl1_proxy_legacy.py')
subprocess.run(['python3',str(ROOT/'code/build_fixed_runner.py')],check=True)
(ROOT/'code/cl1_ops.py').write_bytes((PAYLOAD/'code/disabled_update.py').read_bytes())
files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'gen_fixed_controller.py',ROOT/'host.py',*(ROOT/'code').glob('*.py'),*(ROOT/'tests').glob('*.py')]}
(ROOT/'source_manifest.json').write_text(json.dumps(dict(integration_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files=files)))
(ROOT/'PROTOCOL.json').write_text(json.dumps(dict(version='frozen-transfer-repair-v2',seed=70158,arms=['uniform','frozen-base','frozen-recovery'],updates_each=10000,envs=4096,rollout=24,
 teacher_weights_frozen=True,teacher_action='deterministic mean; Uniform overrides to exactly uniform',teacher_logprob=None,
 resume_uniform_from=1400,matched_restore_boundaries=[500,1000,1400,1500,'then every500 to10000'],source_root=str(OLD),recovery_teacher='V2 from Base after normal-selection correction; no hyperparameter changes',
 test_not_teacher_input=True,paired_state_restore_required=True,failed_attempts_preserved=True),indent=2))
bound={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'gen_fixed_controller.py',ROOT/'install_recovery.py',ROOT/'host.py',ROOT/'PROTOCOL.json',*(ROOT/'code').glob('*.py'),*(ROOT/'tests').glob('*.py')]}
(ROOT/'CODE_BINDING.json').write_text(json.dumps(dict(files=bound,teacher_updates_disabled=True),indent=2))
print(ROOT)
