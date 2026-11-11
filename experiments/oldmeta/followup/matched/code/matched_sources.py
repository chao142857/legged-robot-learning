"""CPU-only validation of immutable, relocated historical source snapshots."""
import json,time
from pathlib import Path
from common import file_sha
from cpu_hash import tree_hash

EXPECTED = [(seed, u) for u in (1000, 3000, 6000, 9000) for seed in (43, 44)]
OLD_META_SHA = '8929d0a13c4aaa7b98c70acf584d04390711bbc9ea6df8b61f3bcb556cdbd3b9'

def import_ready(path):
    path=Path(path)
    marker=path.parent/'IMPORT_READY.json'
    if not marker.exists():return False
    receipt=json.loads(marker.read_text())
    assert receipt['passed'] and receipt['checkpoint_sha256']==file_sha(path)
    return True

def wait_for_import(path,root,timeout=86400):
    deadline=time.monotonic()+timeout
    while not import_ready(path):
        fault=Path(root)/'IMPORT_FAULT.txt'
        if fault.exists():raise RuntimeError(fault.read_text())
        if time.monotonic()>deadline:raise TimeoutError('Source import not ready: '+str(path))
        time.sleep(2)

def source_progress(prepared, info):
    prepared = Path(prepared)
    receipt = json.loads((prepared.parent/'export_receipt.json').read_text())
    path = prepared.parent/'progress_row.json'
    assert file_sha(path) == receipt['progress_row_sha256']
    row = json.loads(path.read_text())
    assert row['completed_updates'] == info['completed_updates']
    assert row['native_feedback_episodes'] == info['adapter']['owner']['accepted']
    assert row['learning_rate'] == info['adaptive_learning_rate']
    return row, path

def verify_required(saved, receipt):
    fields = ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict')
    actual = {k: tree_hash(saved[k]) for k in fields}
    actual.update({k: tree_hash(saved['infos'][k]) for k in ('adapter', 'environment', 'rng', 'v02_context')})
    assert actual == receipt['unchanged_required_state_hashes']
    return actual
