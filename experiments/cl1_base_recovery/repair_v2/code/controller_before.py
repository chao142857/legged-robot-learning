"""Durable frozen-Teacher generalization orchestration for pre-registered Student seed 70158.

Three arms run serially on one GPU, 20 x 500-update blocks each, Teacher weights frozen:
  uniform        -> Base learner present but probability forced uniform by the runner
  frozen-base    -> Base.pt weights, memory evolves with the new Student history
  frozen-recovery-> Recovery.pt weights, same rule
Every block is its own container job, so a crash resumes from the last completed checkpoint.
"""
import hashlib
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from host import ROOT, EXPERIENCE, KIT, run, write, outside

SEED = 70158
BLOCKS = 20
NAME_PREFIX = 'e2-a8c76417-v02-'
ARMS = (('uniform', 'uniform'), ('frozen-base', 'meta'), ('frozen-recovery', 'meta'))


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def status(phase, **kw):
    write(ROOT / 'STATUS.json', dict(phase=phase, unix=time.time(), **kw))


def job_ok(label):
    r = ROOT / 'jobs' / label / 'result.json'
    if not r.exists():
        return None
    d = read(r)
    assert d.get('exit_code') == 0, (label, d)
    return d


def ensure(label, arguments, gpu, on_tick=None):
    """Run a container job once; tolerate an interrupted attempt by quarantining its directory."""
    if job_ok(label):
        return
    out = ROOT / 'jobs' / label
    if out.exists():
        out.rename(out.with_name(out.name + '.stale-' + str(int(time.time()))))
    # Container names are deterministic per label and shared with earlier rounds on this host.
    subprocess.run(['docker', 'rm', '-f', NAME_PREFIX + label], capture_output=True)
    run(label, list(map(str, arguments)), gpu=gpu, on_tick=on_tick)


def ops(label, op, stage, deterministic=False, **kw):
    arguments = [ROOT / 'code/cl1_ops.py', op, '--stage', stage]
    for k, v in kw.items():
        arguments += ['--' + k.replace('_', '-'), str(v)]
    if deterministic:
        arguments.append('--deterministic-mean')
    ensure('cpu-' + label, arguments, gpu=False)


def evaluate(label, cp, kind):
    out = ROOT / 'evaluations' / label
    ensure('eval-' + label,
           [ROOT / 'code/evaluate.py', '--checkpoint', cp, '--kind', kind,
            '--out', out, '--headless', '--kit_args', KIT], gpu=True)
    a = read(out / 'result.json')
    assert a['passed'] and a['checkpoint_sha256'] == sha(cp)
    assert a['training_history_updates'] == 0 and a['model_weights_unchanged']
    ident = {k: a[k] for k in ('initial_sha256', 'pool_sha256', 'seed', 'episodes', 'protocol_sha256')}
    ref = ROOT / f'evaluation_identity_{kind}.json'
    assert read(ref) == ident, ('evaluation identity drift', kind)
    return a


def train(arm, kind, target, resume=None):
    label = f'{arm}-seed{SEED}-to{target:05d}'
    out = ROOT / 'runs' / label
    dataset = EXPERIENCE / label
    arguments = [ROOT / 'code/train_cl1.py', '--out', out, '--dataset', dataset,
                 '--mode', 'transfer', '--target', target, '--student-seed', SEED,
                 '--teacher-in', ROOT / 'teachers' / arm / 'proxy.pt',
                 '--transfer-kind', kind, '--branch', 'cl1',
                 '--headless', '--kit_args', KIT]
    if resume:
        arguments += ['--resume', resume]

    def tick(process):
        for req in sorted((out / 'evaluation_requests').glob('*.json')):
            if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():
                continue
            a = read(req)
            cp = outside(a['checkpoint'])
            assert cp.parent == out / 'actors' and sha(cp) == a['checkpoint_sha256']
            result = evaluate(label + '-' + req.stem, cp, a['kind'])
            assert result['update'] == a['update']
            write(req.with_suffix('.done.json'), result)
        for req in sorted((out / 'teacher_requests').glob('*.pt')):
            if req.name.endswith(('.done.pt', '.teacher_after.pt')) or req.with_suffix('.done.pt').exists():
                continue
            ops(label + '-' + req.stem, 'request', arm, deterministic=True, request=req)
            assert req.with_suffix('.done.pt').exists()

    status('training', arm=arm, student_seed=SEED, target=target, label=label, teacher_frozen=True)
    if not job_ok(label):
        if out.exists() or dataset.exists():
            stamp = str(int(time.time()))
            subprocess.run(['docker', 'rm', '-f', NAME_PREFIX + label], capture_output=True)
            for p in (out, dataset):
                if p.exists():
                    p.rename(p.with_name(p.name + '.stale-' + stamp))
        ensure(label, arguments, gpu=True, on_tick=tick)
    a = read(out / 'result.json')
    assert not (out / 'fault.txt').exists(), (out / 'fault.txt').read_text()
    assert a['passed'] and a['completed_updates'] == target
    return outside(a['checkpoint'])


def costs():
    jobs = [read(p) for p in (ROOT / 'jobs').glob('*/result.json')]
    evaluations = [read(p) for p in (ROOT / 'evaluations').glob('*/result.json')]
    runs = [read(p) for p in (ROOT / 'runs').glob('*/result.json')]
    updates = {}
    for p in (ROOT / 'runs').glob('*/updates.jsonl'):
        updates[p.parent.name] = sum(1 for _ in p.open())
    write(ROOT / 'COSTS.json', dict(actual_updates_by_segment=updates,
                                    actual_updates=sum(updates.values()),
                                    training_transitions=sum(updates.values()) * 4096 * 24,
                                    evaluation_transitions=sum(x.get('executed_environment_transitions', 0)
                                                               for x in evaluations),
                                    job_wall_seconds=sum(x['wall_seconds'] for x in jobs),
                                    gpu_job_wall_seconds=sum(x['wall_seconds'] for x in jobs if x['gpu']),
                                    evaluations=len(evaluations), completed_segments=len(runs)))


def preflight():
    ps = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True, check=True)
    running = [n for n in ps.stdout.split() if n]
    assert not running, ('refusing to start while containers run: ' + ','.join(running))
    free = shutil.disk_usage('/data').free
    assert free > 80 * (1 << 30), free
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                         capture_output=True, text=True)
    assert not gpu.stdout.strip() or 'not supported' in gpu.stdout, ('GPU busy: ' + gpu.stdout.strip())
    img = subprocess.run(['docker', 'image', 'inspect',
                          'sha256:8ee9b2bb8e6aae08dace71ee1e2141a4d62bcc63e4cf0eac28b347111a486a87'],
                         capture_output=True, text=True)
    assert img.returncode == 0, 'immutable Isaac image missing: ' + img.stderr.strip()


def main():
    preflight()
    status('bootstrap', student_seed=SEED)
    ensure('cpu-tests', [ROOT / 'tests/test_cl1.py'], gpu=False)
    ensure('cpu-adam-regression', [ROOT / 'tests/test_adam_runtime.py'], gpu=False)
    if not (ROOT / 'BASE_RECOVERY_LOCK.json').exists():
        ensure('gen-prepare', [ROOT / 'code/gen_prepare.py'], gpu=False)
    lock = read(ROOT / 'BASE_RECOVERY_LOCK.json')
    assert lock['seed'] == SEED
    if not (ROOT / 'LAUNCH_RECEIPT.json').exists():
        write(ROOT / 'LAUNCH_RECEIPT.json', dict(
            seed=SEED, plan='NEXT_ROUND_PLAN.md', plan_sha256=sha(ROOT / 'NEXT_ROUND_PLAN.md'),
            code_binding_sha256=sha(ROOT / 'CODE_BINDING.json'), teacher_lock=lock,
            arms=[a for a, _ in ARMS], blocks_per_arm=BLOCKS, target=10000,
            gpu=subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(),
            started_unix=time.time(), automatic=False))
    for arm, kind in ARMS:
        status('arm', arm=arm, transfer_kind=kind, student_seed=SEED, teacher_frozen=True)
        resume = None
        for block in range(BLOCKS):
            resume = train(arm, kind, (block + 1) * 500, resume=resume)
            status('arm_block', arm=arm, block=block + 1, student_updates=(block + 1) * 500)
            costs()
        ensure('gen-verify-' + arm, [ROOT / 'code/gen_verify.py', '--arm', arm], gpu=False)
        status('arm_complete', arm=arm, student_updates=10000, teacher_frozen=True)
    ensure('gen-compare', [ROOT / 'code/gen_compare.py'], gpu=False)
    costs()
    status('complete', arms=[a for a, _ in ARMS], seed=SEED, student_updates=len(ARMS) * 10000)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (ROOT / 'FAULT.txt').write_text(traceback.format_exc())
        try:
            costs()
        except BaseException:
            pass
        status('fault', detail=traceback.format_exc())
        raise
