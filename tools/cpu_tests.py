"""Run independent CPU suites in isolated processes (no simulator or training)."""
import os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1]
suites=[('tests',['src','tests']),
 ('experiments/clustering/e2/tests',['experiments/clustering/e2/code']),
 ('experiments/clustering/e3/tests',['experiments/clustering/e3/code','experiments/clustering/e2/code']),
 ('experiments/cl1_base_recovery/repair_v2/tests',['experiments/cl1_base_recovery/transfer70158/code','experiments/cl1_base_recovery/repair_v2/code'])]
failed=[]
for suite,paths in suites:
 env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(r/p) for p in paths),PYTHONDONTWRITEBYTECODE='1')
 print('SUITE',suite,flush=True)
 result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r/suite),'-v'],env=env,cwd=r)
 if result.returncode:failed.append(suite)
if failed:raise SystemExit('Failed: '+', '.join(failed))
