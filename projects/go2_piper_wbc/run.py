"""Versioned Go2/Piper launchers. --check never imports or starts a simulator."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PIN = '4e12109d1eccb1d9da2e4a35f189fabea3f13677'
ENTRIES = {
    'p0-inventory': ('recipe_v1_20261025/tools/p0_inventory.py', []),
    'native-eval': ('native_20261023/tools/evaluate_native.py', []),
    'scratch': ('scratch_t1_20261025/tools/run_t1_scratch.py', []),
    'scratch-eval': ('scratch_t1_20261025/tools/evaluate_t1_scratch.py', []),
    'recipe-audit': ('recipe_v1_20261025/tools/run_recipe_a.py', ['--mode', 'audit']),
    'recipe-train': ('recipe_v1_20261025/tools/run_recipe_a.py', ['--mode', 'train']),
    'recipe-eval': ('recipe_v1_20261025/tools/p0_frozen.py', []),
    'aplus-audit': ('recipe_aplus_20261025/tools/run_aplus.py', ['--mode', 'audit']),
    'aplus-train': ('recipe_aplus_20261025/tools/run_aplus.py', ['--mode', 'train']),
    'startup-eval': ('recipe_aplus_20261025/tools/evaluate_startup.py', []),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('native-train', *ENTRIES))
    p.add_argument('--upstream', type=Path)
    p.add_argument('--runtime-root', type=Path, help='Your external experiment receipts/checkpoints')
    p.add_argument('--native-params', type=Path, help='Source-bound native env.yaml/agent.yaml directory')
    p.add_argument('--diagnostics-out', type=Path, help='Opt-in native observer receipts; fresh external directory')
    p.add_argument('--check', action='store_true')
    args, extra = p.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    versions = {}
    for name in ('torch', 'rsl-rl-lib', 'isaacsim', 'isaaclab', 'numpy', 'gymnasium'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if args.check:
        print(json.dumps(dict(mode=args.mode, imported_simulator=False, versions=versions,
                              upstream_pin=PIN, upstream_available=bool(args.upstream and args.upstream.is_dir()),
                              runtime_inputs_available=bool(args.runtime_root and args.runtime_root.is_dir()),
                              note='Missing simulator/assets/receipts are limitations, not passed runtime tests.'), indent=2))
        return
    if not args.upstream or not args.upstream.is_dir():
        p.error('--upstream must identify an independently obtained LeggedManip_Lab checkout')
    upstream = args.upstream.resolve()
    head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PIN:
        p.error('Upstream source revision does not match the documented pin')
    if subprocess.check_output(['git', '-C', str(upstream), 'diff', 'HEAD', '--', 'source', 'scripts']):
        p.error('Use a clean independent upstream checkout')
    env = os.environ.copy()
    env['LEGGED_NATIVE_CHECKOUT'] = str(upstream)
    if args.runtime_root:
        env['LEGGED_RUNTIME_ROOT'] = str(args.runtime_root.resolve())
    if args.native_params:
        env['LEGGED_NATIVE_PARAMS'] = str(args.native_params.resolve())
    env.pop('NATIVE_WBC_DIAGNOSTICS', None)
    if args.diagnostics_out:
        if args.mode != 'native-train' or args.diagnostics_out.exists():
            p.error('--diagnostics-out is only for native-train and requires a fresh directory')
        env['NATIVE_WBC_DIAGNOSTICS'] = str(args.diagnostics_out.resolve())
    paths = [ROOT, ROOT.parents[1], ROOT/'task_suite_20261023', ROOT/'task_suite_20261023/tools',
             ROOT/'scratch_t1_20261025/tools', ROOT/'recipe_v1_20261025', ROOT/'recipe_v1_20261025/tools',
             ROOT/'recipe_aplus_20261025', ROOT/'recipe_aplus_20261025/tools',
             ROOT/'native_20261023/tools', upstream/'source/LeggedManip_Lab']
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get('PYTHONPATH', '')
    if args.mode == 'native-train':
        script = upstream/'scripts/rsl_rl/train.py'
        prefix = ['--task', 'GO2-PIPER-WBC']
    else:
        rel, prefix = ENTRIES[args.mode]
        script = ROOT/rel
    if not script.is_file():
        p.error('Expected entry point does not exist: ' + str(script))
    # Native outputs go to this independently obtained checkout. No clone, install,
    # credential read, EULA acceptance, GPU reset or background supervisor occurs here.
    if args.diagnostics_out:
        code = 'import runpy,sys; import observer_bootstrap; script=sys.argv.pop(1); runpy.run_path(script,run_name="__main__")'
        cmd = [sys.executable, '-c', code, str(script), *prefix, *extra]
    else:
        cmd = [sys.executable, str(script), *prefix, *extra]
    raise SystemExit(subprocess.call(cmd, cwd=upstream, env=env))


if __name__ == '__main__':
    main()
