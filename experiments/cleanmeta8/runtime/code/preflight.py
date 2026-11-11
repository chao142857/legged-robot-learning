"""Verify local Teacher checkpoints and the seed42 comparison assets on CPU."""

import json
import numpy as np
import torch

from common import ROOT, file_sha, write
from cpu_hash import tree_hash
from teacher_v02 import Teacher


def read(path):
    return json.loads(path.read_text())


baseline = read(ROOT / 'uniform_baseline.json')
assert baseline['seed'] == 42 and baseline['completed_updates'] == 10000
assert [p['update'] for p in baseline['points']] == list(range(1000, 10001, 1000))
initialization = baseline['transfer_initialization']
assert read(ROOT / 'runs/student_seed42_initialization.json') == initialization
binding = read(ROOT / 'prepared/manifest.json')['binding']
for relative, key in (('pools/wbc_train.npz', 'train_pool_sha256'),
                      ('prepared/wbc_development.npz', 'development_pool_sha256'),
                      ('prepared/command_features.npy', 'command_features_sha256')):
    assert file_sha(ROOT / relative) == binding[key], relative
assert binding['train_pool_sha256'] == initialization['pool_sha256']
test = read(ROOT / 'evaluation_identity_test.json')
assert file_sha(ROOT / 'pools/wbc_evaluation.npz') == test['pool_sha256']
assert test['episodes'] == 1024
for point in baseline['points']:
    assert point['pool'] == test['pool_sha256'] and point['protocol'] == test['protocol_sha256']
    assert point['seed'] == test['seed'] and point['episodes'] == test['episodes'] and point['passed']
acceptance = read(ROOT / 'CLEANMETA8_ACCEPTANCE_SOURCE.json')
assert acceptance['passed'] and acceptance['logical_updates'] == 8
features = np.load(ROOT / 'prepared/command_features.npy', allow_pickle=False)
teacher_states = {}
expected = {'Fresh.pt': ('98c577bb969bbfd3362e5959f20a12cee61cbc2c4b8d2bdf529fecaa1c32cef2', 0),
            'CleanMeta8.pt': (acceptance['checkpoint_sha256'], 8)}
for name, (sha, updates) in expected.items():
    path = ROOT / 'teachers' / name
    assert file_sha(path) == sha, name
    saved = torch.load(path, map_location='cpu', weights_only=False)['teacher']
    teacher = Teacher(features, binding)
    teacher.load(saved)
    assert teacher.updates == updates and teacher.used_versions == set(map(str, range(updates)))
    teacher_states[name] = tree_hash(teacher.state())
    assert teacher_states[name] == tree_hash(saved)
assert teacher_states['CleanMeta8.pt'] == acceptance['teacher_state_sha256']
write(ROOT / 'PREFLIGHT.json', dict(passed=True, seed=42, teacher_states=teacher_states,
                                     teacher_files={name: sha for name, (sha, _) in expected.items()},
                                     uniform_baseline_reused=True, expected_student_init=initialization))
print('SEED42_PREFLIGHT_PASS', flush=True)
