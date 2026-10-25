"""Explicit Z1 entry points. --check is CPU-only and never imports Isaac."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PIN = '8bda7c7df8b223acf649e0b212635a92c8528900'
ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'evaluate-stairs', 'evaluate-rough'))
    p.add_argument('--iit-checkout', type=Path)
    p.add_argument('--asset', type=Path, help='Your independently authorized Z1 USD; never downloaded')
    p.add_argument('--check', action='store_true')
    args, extra = p.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    versions = {}
    for name in ('torch', 'rsl-rl-lib', 'isaaclab', 'numpy', 'gymnasium'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = dict(mode=args.mode, imported_simulator=False, versions=versions,
                  iit_pin=PIN, asset_provided=bool(args.asset and args.asset.is_file()),
                  note='Requires the historical Isaac Lab v3/Newton environment; not the Go2 environment.')
    if args.check:
        print(json.dumps(report, indent=2))
        return
    if not args.iit_checkout or not args.iit_checkout.is_dir():
        p.error('--iit-checkout must identify an independently obtained IIT source checkout')
    head = subprocess.check_output(['git', '-C', str(args.iit_checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PIN:
        p.error('IIT source revision does not match the documented pin')
    if not args.asset or not args.asset.is_file() or args.asset.suffix not in ('.usd', '.usda', '.usdc'):
        p.error('--asset must identify an independently authorized USD, with all its local dependencies')
    env = os.environ.copy()
    env['Z1_USD_PATH'] = str(args.asset.resolve())
    paths = [ROOT/'legacy_20261010', ROOT.parents[1], args.iit_checkout/'source/basic_locomotion_isaaclab']
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get('PYTHONPATH', '')
    if args.mode == 'train':
        code = 'from isaaclab_rl.entrypoints.backends.train_rsl_rl import run; import sys; run(["--external_callback", "register_tasks.register"] + sys.argv[1:])'
        cmd = [sys.executable, '-c', code, *extra]
    else:
        script = 'evaluate_z1_stairs.py' if args.mode == 'evaluate-stairs' else 'evaluate_z1_rough_crossing.py'
        cmd = [sys.executable, str(ROOT/'legacy_20261010'/script), *extra]
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == '__main__':
    main()
