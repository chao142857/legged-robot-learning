"""Read-only references to the unique owners' packages, pinned by file SHA256."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {'sgs':ROOT.parent/'go2_piper_b1_sgs_20261025_work', 'ik':ROOT.parent/'go2_piper_b3_20261025_work'}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def package_files(root, package):
    folders=[root/package]
    if package=='shared_sgs': folders += [root/'integration',root/'reference/executed/task_suite',root/'reference/executed/tools',root/'reference/resolved']
    return sorted(p for folder in folders for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.sh') and '__pycache__' not in p.parts)
def inventory():
    result={owner:{str(p.relative_to(root)):sha(p) for p in package_files(root, 'shared_sgs' if owner=='sgs' else 'shared_ik')} for owner,root in DEFAULTS.items()}
    for name in ('tools/evaluate_b3.py','frozen/native_binding_fixture.json','frozen/native_runner_cfg.json','evidence/native_pair_report.json'):
        result['ik'][name]=sha(DEFAULTS['ik']/name)
    return result
def verify(lock_path=None):
    lock_path = Path(lock_path or ROOT/'protocol/dependencies.lock.json')
    lock = json.loads(lock_path.read_text())
    if lock['schema_version'] != 1 or lock['packages'] != inventory():
        raise RuntimeError('Shared owner files changed or dependency lock incomplete; no private fallback allowed')
    for owner,item in lock['owner_manifests'].items():
        if sha(DEFAULTS[owner]/item['path'])!=item['sha256']:raise RuntimeError('Owner manifest changed: '+owner)
    for filename,expected in lock['assembly_files'].items():
        if sha(ROOT/filename)!=expected:raise RuntimeError('B2 assembly changed: '+filename)
    for filename, expected in lock['protocol_files'].items():
        if sha(ROOT/'protocol'/filename) != expected: raise RuntimeError('Common protocol changed: '+filename)
    return lock

def enable_owner_imports(verify_lock=True):
    if verify_lock: verify()
    for root in DEFAULTS.values():
        if str(root) not in sys.path: sys.path.insert(0,str(root))
    return importlib.import_module('shared_sgs')
