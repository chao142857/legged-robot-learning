"""Run the asset-free checks using this interpreter. No simulator or GPU launch."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FACTORIAL = ROOT/'projects/sgs/factorial_20261025'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-ml', action='store_true', help='Fail if optional Torch/RSL dependencies are absent')
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    package_paths = [ROOT, FACTORIAL/'go2_piper_b1_sgs_20261025_work',
                     FACTORIAL/'go2_piper_b2_20261025_work', FACTORIAL/'go2_piper_b3_20261025_work']
    env['PYTHONPATH'] = os.pathsep.join(map(str, package_paths)) + os.pathsep + env.get('PYTHONPATH', '')
    has_torch = importlib.util.find_spec('torch') is not None
    has_rsl = has_torch and importlib.util.find_spec('rsl_rl') is not None and importlib.util.find_spec('tensordict') is not None
    jobs = [
        ('publication', ['tools/audit_publication.py'], True),
        ('sgs_lifecycle', ['-m','unittest','discover','-s',str(FACTORIAL/'go2_piper_b1_sgs_20261025_work/tests'),'-v'], True),
        ('z1_standing_sampler', ['projects/z1_locomotion/studies_20261014/test_short_sampler.py'], has_torch),
        ('task_metrics', ['-m','unittest','discover','-s','projects/go2_piper_wbc/task_suite_20261023/tools','-p','test_*.py','-v'], True),
        ('tracking_metrics', ['-m','unittest','discover','-s','projects/go2_piper_wbc/scratch_t1_20261025/tools','-p','test_*.py','-v'], has_torch),
        ('b2_common', ['-m','unittest','discover','-s',str(FACTORIAL/'go2_piper_b2_20261025_work/tests'),'-p','test_common_cpu.py','-v'], has_torch),
        ('b3_mapping', ['-m','unittest','discover','-s',str(FACTORIAL/'go2_piper_b3_20261025_work/tests'),'-p','test_shared_ik.py','-k','MappingTests','-v'], has_torch),
        ('gradient', ['-m','unittest','discover','-s','projects/ppo_gradient_lab/tests','-p','test_go2_piper_gradients.py','-v'], has_torch),
        ('component_probe', ['-m','unittest','discover','-s','projects/ppo_gradient_lab/tests','-p','test_go2_gradient_probe.py','-v'], has_torch),
        ('native_ppo_microscope', ['-m','unittest','discover','-s','projects/ppo_gradient_lab/tests','-p','test_ppo_update_microscope.py','-v'], has_rsl),
        ('z1_entry', ['projects/z1_locomotion/run.py','train','--check'], True),
        ('go2_entry', ['projects/go2_piper_wbc/run.py','native-train','--check'], True),
    ]
    results=[]
    for name,cmd,available in jobs:
        if not available:
            results.append(dict(name=name,status='failed' if args.require_ml else 'skipped',reason='Optional CPU ML dependencies missing'))
            continue
        start=time.monotonic()
        result=subprocess.run([sys.executable,*cmd],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        results.append(dict(name=name,status='passed' if result.returncode==0 else 'failed',
                            elapsed_s=round(time.monotonic()-start,3),returncode=result.returncode,output=result.stdout))
        print(name,results[-1]['status'],flush=True)
    report=dict(status='failed' if any(x['status']=='failed' for x in results) else 'partial' if any(x['status']=='skipped' for x in results) else 'passed',
                simulator_started=False,gpu_training_started=False,python=sys.version,results=results)
    if args.report:
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('Overall:',report['status'])
    raise SystemExit(1 if report['status']=='failed' else 0)


if __name__ == '__main__':
    main()
