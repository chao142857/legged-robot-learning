"""Reuse the earlier named-joint/FK diagnosis on new native PhysX traces (CPU only)."""
import argparse,ast,csv,hashlib,json,pathlib
import numpy as np
from scipy.spatial.transform import Rotation
from gpu_budget import read,write

def functions(path,names,namespace):
    path=pathlib.Path(path);tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(root,config,out):
    p=pathlib.Path(root);cfg=read(config)
    ns=dict(np=np,Rotation=Rotation,Path=pathlib.Path,json=json,csv=csv,hashlib=hashlib,DT=.02,W=2*np.pi/8,_MODEL_CACHE={})
    fk_sha=functions(cfg['kinematics_source'],['fk'],ns)
    reference_sha=functions(cfg['original_response_source'],['read','sha','mean','fit','flat_harmonic','jac','phase_stats','analyze'],ns)
    chain=read(cfg['chain_file'])['chain'];index=read(p/'EVAL_INDEX.json');summaries=[];joint_rows=[]
    for label,run in index.items():
        if label.split('-')[-2]!='X' or label.startswith('final-'):continue
        directory=pathlib.Path(run)/'evaluation'
        if not (directory/'manifest.json').exists():continue
        cfg_run=read(pathlib.Path(run)/'config.json')
        summary,episodes,joints=ns['analyze'](directory,cfg_run['checkpoint_sha256'],label,list(range(64)),chain)
        summaries.append(dict(label=label,**summary));joint_rows.extend(joints)
    write(out,dict(kind='Derived local geometric/harmonic diagnosis, not a new physical acceptance score',
        diagnostic_window_s=[4,20],formal_gate_window_unchanged='Native after1s X gate plus full20s startup/safety',
        source_SHA256=dict(original_response=reference_sha,original_fk=fk_sha,chain=hashlib.sha256(pathlib.Path(cfg['chain_file']).read_bytes()).hexdigest()),
        summaries=summaries,joints=joint_rows,
        limits='Local Jacobian projection is not a commanded EE trajectory. Target means original limited/ramped request; delayed queue traces only when actually present. No FK/IK action generation or learning.'))
    print(json.dumps(dict(cases=len(summaries),out=str(out))))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--config',required=True);a.add_argument('--out',required=True);c=a.parse_args();main(c.root,c.config,c.out)
