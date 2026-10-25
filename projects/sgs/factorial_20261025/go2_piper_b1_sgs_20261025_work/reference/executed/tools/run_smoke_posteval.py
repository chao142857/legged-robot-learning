"""Evaluate four completed 2-update smoke checkpoints; no optimization."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
suite=Path(__file__).resolve().parents[1]
stage=a.out/'post_smoke_v1.json'
if stage.exists():
    raise FileExistsError(stage)
cases={'t0':('T0','plane',32),'t1':('T1','plane',32),
       'up':('T2','stairs_up',16),'down':('T2','stairs_down',16)}
rows=[]
for key,(family,terrain,n) in cases.items():
    training=a.out/('train_'+key+'_smoke_v1')
    assert json.loads((training/'summary.json').read_text())['completed_updates']==2
    dest=a.out/('post_'+key+'_smoke_s7101_v1')
    assert not dest.exists()
    cmd=['bash',str(suite/'tools/run_python.sh'),str(suite/'tools/run_suite.py'),
         '--mode','eval','--family',family,'--terrain',terrain,'--ee_mode','smooth',
         '--checkpoint',str(training/'cp_final.pt'),'--out',str(dest),
         '--num_envs',str(n),'--seed','7101','--headless']
    row=dict(name=dest.name,state='running',command=cmd,start_unix=time.time())
    rows.append(row)
    def save():
        stage.write_text(json.dumps(rows,indent=2)+'\n')
    save()
    try:
        with (a.out/'logs'/(dest.name+'.log')).open('x') as f:
            result=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                                  cwd=suite.parent,timeout=900)
        row.update(returncode=result.returncode,state='complete' if result.returncode==0 else 'failed')
    except BaseException as exc:
        row.update(state='failed',error=repr(exc)); raise
    finally:
        row['end_unix']=time.time(); save()
    if result.returncode:
        raise SystemExit(result.returncode)
    print(dest.name,'complete',flush=True)
