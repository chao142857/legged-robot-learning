"""Execute only a prepared six-job, 8-env, 500-step physics smoke plan.

No PPO/SGS mode exists. Refuses existing outputs and changed profiles/checkpoint.
Preparation is separate, so creating the plan cannot start simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    a=p.parse_args()
    plan_path=Path(a.plan).resolve()
    plan=json.loads(plan_path.read_text())
    root=Path(plan['suite_root'])
    if plan['runner_sha256']!=sha(__file__):
        raise ValueError('Prepared runner SHA does not match this runner')
    if plan.get('training_jobs')!=0 or plan.get('SGS_jobs')!=0 or plan.get('admission') is not False:
        raise ValueError('Only non-admitted physics smoke plans are accepted')
    expected=[('T0','plane','medium'),('T0','plane','hard'),('T1','plane','medium'),('T1','plane','hard'),
              ('T2','stairs_up','medium'),('T2','stairs_down','medium')]
    actual=[(j['family'],j['terrain'],j['difficulty']) for j in plan['jobs']]
    if actual!=expected:
        raise ValueError('Plan must contain exactly the six specified candidates; hard T2 is forbidden')
    if sha(plan['checkpoint'])!=plan['checkpoint_sha256']:
        raise ValueError('Checkpoint hash changed')
    for j in plan['jobs']:
        if j['mode']!='physics' or j['num_envs']!=8 or j['steps']!=500 or j['ee_mode']!='smooth':
            raise ValueError('Smoke budget/mode changed')
        if sha(j['profile'])!=j['profile_sha256']:
            raise ValueError('Prepared exact profile changed')
        if Path(j['output']).exists() or Path(j['log']).exists():
            raise ValueError('Refusing to overwrite existing output or log: '+j['name'])
        params=json.loads(Path(j['profile']).read_text())
        if j['family']!='T2' and any(not(k.startswith('ee_') or k.startswith('base_')) for k in params):
            raise ValueError('Flat profile must omit all unrelated terrain fields')
        if j['family']=='T2' and not 0<params['route_distance']<=3.5:
            raise ValueError('Unsupported T2 route must never run')
    if plan['per_job_timeout_seconds']!=900:
        raise ValueError('Per-job timeout changed')
    status=plan_path.parent/'status.json'
    with status.open('x') as f:
        json.dump(dict(plan_sha256=sha(plan_path),jobs=[]),f)
    rows=[]
    for j in plan['jobs']:
        command=['bash',str(root/'tools/run_python.sh'),str(root/'tools/run_suite.py'),
            '--mode','physics','--family',j['family'],'--terrain',j['terrain'],'--difficulty',j['difficulty'],
            '--ee_mode','smooth','--profile',j['profile'],'--checkpoint',plan['checkpoint'],'--out',j['output'],
            '--num_envs','8','--steps','500','--seed',str(j['seed']),'--headless']
        item=dict(name=j['name'],state='running',command=command,start_unix=time.time())
        rows.append(item)
        def save():
            status.write_text(json.dumps(dict(plan_sha256=sha(plan_path),jobs=rows),indent=2)+'\n')
        save()
        try:
            with Path(j['log']).open('x') as f:
                completed=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=root.parent,timeout=900)
            item.update(returncode=completed.returncode,state='complete' if completed.returncode==0 else 'failed')
        except BaseException as exc:
            item.update(state='failed',error=repr(exc)); raise
        finally:
            item['end_unix']=time.time(); save()
        print(json.dumps({k:v for k,v in item.items() if k!='command'}),flush=True)
        if item['returncode']:
            raise SystemExit(item['returncode'])
    print('CANDIDATE_SMOKE_COMPLETE',plan_path,flush=True)


if __name__=='__main__':
    main()
