"""Prepare the pre-registered generalization round: lock sources, build three frozen/carrier Teacher stages.

Runs inside the Isaac image via host.run (CPU only). ROOT=/workspace is the new round root.
"""
import json
import sys
from pathlib import Path
import torch
from cl1_core import Learner
from cpu_hash import tree_hash
from common import ROOT, write, file_sha

SEED = 70158
SOURCE_ROOT = Path('/data/go2_piper_nsgs_cl1_20261109')
EXCLUDED_SEEDS = (43, 44, 48, 49)

# Pre-registered hashes of the source round (recorded 2026-11-10 14:25 CST).
BASE_PT_SHA = '2a7d5b0a13eaeadbd858ed629dc373611e81683af4f0424674379c9bb4d35dce'
RECOVERY_PT_SHA = '65b708db09527ebe8f7496974a2169cac7bfafc5a66ae6377c748a1cdd66929b'
PROTOCOL_SHA = 'cf58dac3f75c893e25db3010ab179b84ef52b345f82ba4a91447f114c524c248'
EVAL_DEV_SHA = '7f2db4c7c5d6927dfb28b4352dfa1901282884a67537a385c631da043cdd368c'
EVAL_TEST_SHA = 'b38545ec3ddf45c7f6da05c95e048398f9a80501e50b663895d3d4a5150affa8'
INHERITED_CODE = {
    'code/train_cl1.py': 'b724293591d1c58465332929f887c3667dc66bbfcb4604928d434eedea030a6d',
    'code/teacher_v02.py': '734721a9acb59651388b73b112fb0365dab8a593f00e107f4f79b715a3596f96',
    'code/cl1_core.py': 'c77910121e7980e686020ab4b95882408cf4bea87219d045a68c080d1dbf0376',
    'code/evaluate.py': 'acf9e57d145c56e3576960b09edc53a8c28fe81e53bc56338730ce74349b07b6',
    'code/cl1_proxy.py': '530cca313b7b7e09d4766f4b254ce266a516889891397caeedde83099fd36284',
}
ADAPTATION = ROOT / 'GEN_RUNNER_ADAPTATION.json'
# arms: (stage name, teacher source, runner transfer-kind)
ARMS = (('uniform', 'base', 'uniform'),
        ('frozen-base', 'base', 'meta'),
        ('frozen-recovery', 'recovery', 'meta'))


def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    torch.save(v, tmp)
    tmp.replace(p)


def read(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def write_binding():
    binding = {}
    for p in sorted((ROOT / 'code').glob('*.py')):
        binding[str(p.relative_to(ROOT))] = file_sha(p)
    for rel in ('host.py', 'tmux.sh', 'gen_controller.py', 'PROTOCOL.json',
                'evaluation_identity_development.json', 'evaluation_identity_test.json',
                'tests/test_cl1.py', 'tests/test_adam_runtime.py'):
        binding[rel] = file_sha(ROOT / rel)
    write(ROOT / 'CODE_BINDING.json', dict(seed=SEED, source_root=str(SOURCE_ROOT), files=binding))
    return binding


def main():
    assert SEED not in EXCLUDED_SEEDS, SEED
    status = json.loads((SOURCE_ROOT / 'STATUS.json').read_text())
    assert status['phase'] == 'complete', status
    assert not (SOURCE_ROOT / 'FAULT.txt').exists()
    assert status.get('base_student_updates') == 20000 and status.get('recovery_updates') == 2000
    base = SOURCE_ROOT / 'teachers/base/Base.pt'
    recovery = SOURCE_ROOT / 'teachers/recovery/Recovery.pt'
    assert file_sha(base) == BASE_PT_SHA, file_sha(base)
    assert file_sha(recovery) == RECOVERY_PT_SHA, file_sha(recovery)

    # Protocol, evaluation pools and inherited code stay byte-identical to the source round.
    assert file_sha(ROOT / 'PROTOCOL.json') == PROTOCOL_SHA
    assert file_sha(ROOT / 'evaluation_identity_development.json') == EVAL_DEV_SHA
    assert file_sha(ROOT / 'evaluation_identity_test.json') == EVAL_TEST_SHA
    for rel, want in INHERITED_CODE.items():
        got = file_sha(ROOT / rel)
        assert got == want, (rel, got, want)
    adaptation = json.loads(ADAPTATION.read_text())
    assert file_sha(ROOT / 'code/cl1_ops.py') == adaptation['output_sha256']
    assert adaptation['source_sha256'] == '38fa617969860eaccd916e0b065c3add5e759fe166b0865a529dc40cbd6f36a8'

    write_binding()
    if '--bind-only' in sys.argv[1:]:
        print('GEN_PREPARE_BIND_OK')
        return

    manifest = json.loads((ROOT / 'prepared/manifest.json').read_text())
    proxy = dict(teacher=dict(cl1_proxy=1, binding=manifest['binding']))
    lock = dict(seed=SEED, source_root=str(SOURCE_ROOT), registered='2026-11-10',
                rule='seed = 10000 + int(sha256(S)[:8],16) % 90000; S documented in NEXT_ROUND_PLAN.md',
                source_status=status,
                sources=dict(base=dict(path=str(base), sha256=BASE_PT_SHA),
                             recovery=dict(path=str(recovery), sha256=RECOVERY_PT_SHA)),
                arms={})
    for arm, source, kind in ARMS:
        armroot = ROOT / 'teachers' / arm
        assert not armroot.exists(), f'stage {arm} already exists; refusing to overwrite'
        teacher_path = base if source == 'base' else recovery
        snapshot = read(teacher_path)
        learner = Learner()
        learner.load(snapshot['learner'])
        state = dict(learner=learner.state(),
                     students={str(SEED): dict(history=[], pending=None, previous=None,
                                               transitions=[], cursor=0)},
                     batches=0)
        save(armroot / 'current.pt', state)
        save(armroot / 'initial.pt', state)
        save(armroot / 'proxy.pt', proxy)
        lock['arms'][arm] = dict(source=source, transfer_kind=kind,
                                 source_sha256=file_sha(teacher_path),
                                 teacher_net_sha256=tree_hash(learner.net.state_dict()),
                                 teacher_version=int(learner.version))
    write(ROOT / 'BASE_RECOVERY_LOCK.json', lock)
    print('GEN_PREPARE_OK ' + json.dumps({a: v['teacher_net_sha256'] for a, v in lock['arms'].items()}))


if __name__ == '__main__':
    main()
