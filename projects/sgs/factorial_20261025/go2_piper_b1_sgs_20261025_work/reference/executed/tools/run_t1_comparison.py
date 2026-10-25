"""Post-training frozen comparison. Never starts or resumes PPO."""
from publication_paths import research_path
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def write(path,obj):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root);source=Path(__file__).resolve().parents[1]
    status=json.loads((root/'status.json').read_text())
    if status['state']!='complete' or status['completed_updates']!=5000:
        raise RuntimeError('Comparison requires the one scratch run to have completed 5000 updates')
    stages=[]
    native=Path(research_path('go2_piper_upstream_wbc_20261023/formal_seed21/cp_3000.pt'))
    models={f'cp_{n}':root/f'train_seed21/cp_{n}.pt' for n in (0,500,1000,2000,3000,5000)}
    models['native_cp3000']=native
    for label,cp in models.items():
        for seed in (7101,7102,7103):
            stages.append(dict(label=label,checkpoint=str(cp),family='T1',seed=seed,num_envs=64,video=False))
    for label in ('native_cp3000','cp_5000'):
        for seed in (7101,7102,7103):
            stages.append(dict(label=label,checkpoint=str(models[label]),family='T0',seed=seed,num_envs=64,video=False))
        stages.append(dict(label=label,checkpoint=str(models[label]),family='T1',seed=7101,num_envs=4,video=True))
    progress_path=root/'comparison_status.json'
    if progress_path.exists():
        raise FileExistsError('Comparison has already been launched; inspect its state before any separate recovery')
    state={'state':'running','protocol':'t1_scratch_comparison_v1','stages':[],'started_unix':time.time()}
    write(progress_path,state)
    for stage in stages:
        out=root/'evaluation'/stage['label']/('video' if stage['video'] else stage['family'])/f"seed_{stage['seed']}"
        out.parent.mkdir(parents=True,exist_ok=True)
        log_path=out.with_suffix('.log')
        if out.exists() or log_path.exists():raise FileExistsError(out)
        cmd=['bash',str(source/'tools/run_python.sh'),str(source/'tools/evaluate_t1_scratch.py'),
             '--checkpoint',stage['checkpoint'],'--out',str(out),'--family',stage['family'],
             '--seed',str(stage['seed']),'--num_envs',str(stage['num_envs']),'--headless','--device','cuda:0']
        if stage['video']:cmd.append('--video')
        item={**stage,'out':str(out),'command':shlex.join(cmd),'state':'running'}
        state['stages'].append(item);write(progress_path,state)
        with log_path.open('x') as log:
            result=subprocess.run(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT)
        item['returncode']=result.returncode
        item['state']='complete' if result.returncode==0 and (out/'manifest.json').exists() else 'failed'
        write(progress_path,state)
        if item['state']=='failed':
            state['state']='failed';write(progress_path,state)
            raise RuntimeError('Evaluation failed; original evidence retained: '+str(out))
    state['state']='complete';state['elapsed_s']=time.time()-state['started_unix'];write(progress_path,state)


if __name__=='__main__':main()
