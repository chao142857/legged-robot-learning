"""Explicit bounded PPO stage using native OnPolicyRunner, with fail-closed gate.

Smoke: 2 updates x 128 envs. Short: 80 updates x 512 envs, only after
all four two-update smoke cases pass the complete evidence/identity/KL gate.
No resume, long-run, SGS, reward tuning or LR override exists here.
"""
from publication_paths import research_path
import argparse
import json
from pathlib import Path
import subprocess
import time
from ppo_smoke_gate import preflight_all

CASES={'t0':('T0','plane',32),'t1':('T1','plane',32),
       'up':('T2','stairs_up',16),'down':('T2','stairs_down',16)}
ORIGINAL=research_path('go2_piper_upstream_wbc_20261023/formal_seed21/cp_3000.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['smoke','short'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--tag',default='v1')
    a=p.parse_args()
    if len(set(a.cases))!=len(a.cases): p.error('Duplicate cases are not allowed')
    root=Path(__file__).resolve().parents[1]
    status=a.out/('ppo_'+a.phase+'_'+a.tag+'.json')
    if status.exists(): raise SystemExit('Refusing existing stage')
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'logs').mkdir(exist_ok=True)
    stage=dict(schema_version=2,phase=a.phase,state='preflight',start_unix=time.time(),
               training_started=False,jobs=[],preflight=None)
    def save(): status.write_text(json.dumps(stage,indent=2,allow_nan=False)+'\n')
    save()
    def execute(name,arguments):
        dest=a.out/name
        if dest.exists(): raise FileExistsError(dest)
        cmd=['bash',str(root/'tools/run_python.sh'),str(root/'tools/run_suite.py'),
             '--out',str(dest),'--headless']+arguments
        row=dict(name=name,command=cmd,start_unix=time.time(),state='running')
        stage['jobs'].append(row); stage['state']='running'
        if arguments[arguments.index('--mode')+1]=='train': stage['training_started']=True
        save()
        try:
            with (a.out/'logs'/(name+'.log')).open('x') as f:
                run=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                                   cwd=root.parent,timeout=1800)
            row['returncode']=run.returncode
            if run.returncode: raise RuntimeError('Child failed with return code '+str(run.returncode))
            row['state']='complete'
        except BaseException as exc:
            row.update(state='failed',error=repr(exc)); raise
        finally:
            row['end_unix']=time.time(); save()
        print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
        return dest
    try:
        # All path/admission checks precede the first subprocess too.
        for key in a.cases:
            admission=a.out/'audit'/('admission_'+key+'_v1.json')
            if not admission.is_file(): raise ValueError('Missing physical/IK admission: '+str(admission))
            names=['train_'+key+'_'+a.phase+'_'+a.tag]
            if a.phase=='short': names.append('post_'+key+'_short_s7101_'+a.tag)
            for name in names:
                if (a.out/name).exists() or (a.out/'logs'/(name+'.log')).exists():
                    raise FileExistsError('Refusing existing run/log: '+name)
        if a.phase=='short':
            # Check the complete four-case smoke bank before any selected short
            # run; selecting fewer cases cannot bypass another known bad smoke.
            stage['preflight']=preflight_all(a.out,root,ORIGINAL,list(CASES),a.tag)
            save()
            if not stage['preflight']['passed']:
                raise RuntimeError('Short PPO blocked: smoke evidence/identity/update/KL preflight failed; see all case errors')
        for key in a.cases:
            family,terrain,n_eval=CASES[key]
            count=2 if a.phase=='smoke' else 80
            train=execute('train_'+key+'_'+a.phase+'_'+a.tag,
                ['--mode','train','--family',family,'--terrain',terrain,'--ee_mode','smooth',
                 '--checkpoint',ORIGINAL,'--num_envs','128' if a.phase=='smoke' else '512',
                 '--updates',str(count),'--seed','21','--admission',str(a.out/'audit'/('admission_'+key+'_v1.json'))])
            if a.phase=='short':
                execute('post_'+key+'_short_s7101_'+a.tag,
                    ['--mode','eval','--family',family,'--terrain',terrain,'--ee_mode','smooth',
                     '--checkpoint',str(train/'cp_final.pt'),'--num_envs',str(n_eval),'--seed','7101'])
        stage['state']='complete'
    except BaseException as exc:
        stage.update(state='failed',error=repr(exc)); raise
    finally:
        stage['end_unix']=time.time(); save()


if __name__=='__main__':
    main()
