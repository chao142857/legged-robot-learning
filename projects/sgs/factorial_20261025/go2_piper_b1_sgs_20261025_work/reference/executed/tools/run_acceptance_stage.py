"""Finite, sequential physics stages. This script cannot launch PPO or SGS."""
from publication_paths import research_path
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--stage', choices=['flat', 'stairs_smoke', 'stairs_full', 'stones_smoke', 'stones_full'], required=True)
p.add_argument('--out', required=True)
p.add_argument('--tag', default='v2')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = Path(a.out)
cp = research_path('go2_piper_upstream_wbc_20261023/formal_seed21/cp_3000.pt')
jobs = {
    'flat': [('T0','plane','smooth',32,0), ('T1','plane','smooth',8,100),
             ('T1','plane','smooth',32,0), ('T0','plane','fixed',32,0), ('T1','plane','fixed',32,0)],
    'stairs_smoke': [('T2','stairs_up','smooth',8,100),('T2','stairs_down','smooth',8,100)],
    'stairs_full': [('T2','stairs_up','smooth',16,0),('T2','stairs_down','smooth',16,0)],
    'stones_smoke': [('T2','stepping_stones','smooth',8,100)],
    'stones_full': [('T2','stepping_stones','smooth',16,0)],
}[a.stage]
status = out/('stage_'+a.stage+'_'+a.tag+'.json')
if status.exists():
    raise SystemExit('Refusing to reuse a stage status path')
history = []
for family, terrain, ee, n, steps in jobs:
    name = f'{"smoke" if steps else "frozen"}_{family.lower()}_{terrain}_{ee}_s7101_{a.tag}'
    dest = out/name
    if dest.exists():
        raise SystemExit('Refusing to overwrite '+str(dest))
    cmd = ['bash',str(root/'tools/run_python.sh'),str(root/'tools/run_suite.py'),
           '--mode','eval' if family!='T2' else 'physics','--family',family,'--terrain',terrain,
           '--ee_mode',ee,'--checkpoint',cp,'--out',str(dest),'--num_envs',str(n),
           '--steps',str(steps),'--seed','7101','--headless']
    item = dict(name=name,command=cmd,start_unix=time.time(),state='running')
    history.append(item)
    status.write_text(json.dumps(history,indent=2)+'\n')
    with (out/'logs'/(name+'.log')).open('x') as f:
        result = subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=root.parent,timeout=900)
    item.update(returncode=result.returncode,end_unix=time.time(),state='complete' if result.returncode==0 else 'failed')
    status.write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps({k:v for k,v in item.items() if k!='command'}),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
