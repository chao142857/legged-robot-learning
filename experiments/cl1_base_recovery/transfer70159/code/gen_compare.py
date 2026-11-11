"""Compare the three generalization arms (uniform / frozen-base / frozen-recovery) for Student seed 70158.

Runs inside the Isaac image via host.run (CPU only). ROOT=/workspace is the new round root.
"""
import json
from pathlib import Path
from common import ROOT, write

SEED = 70158
ARMS = ('uniform', 'frozen-base', 'frozen-recovery')
METRICS = ('joint_success_rate', 'continuous_joint_score')


def read(p):
    return json.loads(Path(p).read_text())


def results(pattern):
    out = []
    for p in sorted((ROOT / 'evaluations').glob(pattern)):
        r = read(p)
        out.append((r['update'], r))
    return sorted(out)


def main():
    rows = {}
    for arm in ARMS:
        dev = results(f'{arm}-seed{SEED}-*-development/result.json')
        test = results(f'{arm}-seed{SEED}-*-test/result.json')
        end_test = test[-1][1] if test else None
        last5 = dev[-5:] if len(dev) >= 5 else dev
        runs = sorted((ROOT / 'runs').glob(f'{arm}-seed{SEED}-to*/updates.jsonl'))
        updates = sum(sum(1 for _ in p.open()) for p in runs)
        wall = 0.0
        for p in sorted((ROOT / 'jobs').glob(f'{arm}-seed{SEED}-*/result.json')):
            wall += read(p).get('wall_seconds', 0.0)
        for p in sorted((ROOT / 'jobs').glob(f'cpu-{arm}-seed{SEED}-*/result.json')):
            wall += read(p).get('wall_seconds', 0.0)
        rows[arm] = dict(
            arm=arm, student_updates=updates,
            final_test=({k: end_test[k] for k in METRICS + ('pooled_rmse', 'native_failure_rate')}
                        if end_test else None),
            development_last5=dict(
                update=[u for u, _ in last5],
                continuous_joint_score=[r['continuous_joint_score'] for _, r in last5],
                joint_success_rate=[r['joint_success_rate'] for _, r in last5],
                ee_position_m=[r['pooled_rmse']['ee_position_m'] for _, r in last5],
                mean_joint_score=sum(r['continuous_joint_score'] for _, r in last5) / len(last5)),
            development_curve=[dict(update=u, continuous_joint_score=r['continuous_joint_score'],
                                    joint_success_rate=r['joint_success_rate'],
                                    ee_position_m=r['pooled_rmse']['ee_position_m'])
                               for u, r in dev],
            test_points=[dict(update=u, joint_success_rate=r['joint_success_rate'],
                              continuous_joint_score=r['continuous_joint_score']) for u, r in test],
            job_wall_seconds=wall)
    comparison = dict(
        seed=SEED, registered='2026-11-10', rule='single pre-registered Student seed; descriptive only (no significance claim)',
        primary='10k test joint_success_rate', arms=rows,
        note='All arms share one Student initialization; Teacher weights frozen; uniform arm probability forced uniform.')
    write(ROOT / 'COMPARISON.json', comparison)

    lines = [f'# nSGS-CL1 generalization — Student seed {SEED}', '',
             '| arm | updates | 10k test strict success | 10k test joint score | 10k EE RMSE | dev last-5 mean J |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r = rows[arm]
        ft = r['final_test'] or {}
        lines.append('| {arm} | {u} | {s:.2f}% | {j:.4f} | {ee:.5f} | {d:.4f} |'.format(
            arm=arm, u=r['student_updates'],
            s=100 * ft.get('joint_success_rate', float('nan')),
            j=ft.get('continuous_joint_score', float('nan')),
            ee=(ft.get('pooled_rmse') or {}).get('ee_position_m', float('nan')),
            d=r['development_last5']['mean_joint_score']))
    lines += ['', 'Single pre-registered seed; descriptive comparison only, no significance claim.', '',
              'Full development curves and cost breakdown: `COMPARISON.json`, `COSTS.json`.']
    (ROOT / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print('GEN_COMPARE_OK')


if __name__ == '__main__':
    main()
