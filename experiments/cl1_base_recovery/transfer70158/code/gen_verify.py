"""Assert that a frozen-Teacher arm never changed Teacher weights and used deterministic mean preferences.

Runs inside the Isaac image via host.run (CPU only). ROOT=/workspace is the new round root.
"""
import argparse
import json
from pathlib import Path
import torch
from cl1_core import Learner
from cpu_hash import tree_hash
from common import ROOT, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    a = ap.parse_args()
    lock = json.loads((ROOT / 'BASE_RECOVERY_LOCK.json').read_text())
    want = lock['arms'][a.arm]['teacher_net_sha256']
    assert not list((ROOT / 'teachers' / a.arm).glob('batch_*')), 'frozen stage received an outer update'

    rows = []
    for p in sorted((ROOT / 'runs').glob(f'{a.arm}-seed*-to*/teacher_requests/*.done.pt')):
        d = torch.load(p, map_location='cpu', weights_only=False)
        if 'teacher_sha256' in d:
            rows.append(dict(path=str(p), teacher_sha256=d['teacher_sha256'],
                             deterministic_frozen_mean=d.get('deterministic_frozen_mean')))
    assert rows, 'no Teacher decisions recorded'
    shas = {r['teacher_sha256'] for r in rows}
    assert shas == {want}, (sorted(shas), want)
    bad = [r['path'] for r in rows if r['deterministic_frozen_mean'] is not True]
    assert not bad, bad

    state = torch.load(ROOT / 'teachers' / a.arm / 'current.pt', map_location='cpu', weights_only=False)
    learner = Learner()
    learner.load(state['learner'])
    final = tree_hash(learner.net.state_dict())
    assert final == want, (final, want)
    write(ROOT / 'evaluations' / f'freeze_check_{a.arm}.json',
          dict(arm=a.arm, decisions=len(rows), teacher_net_sha256=want, final_state_sha256=final,
               deterministic_frozen_mean=True, outer_updates=0, passed=True))
    print('GEN_VERIFY_OK ' + a.arm + ' decisions=' + str(len(rows)))


if __name__ == '__main__':
    main()
