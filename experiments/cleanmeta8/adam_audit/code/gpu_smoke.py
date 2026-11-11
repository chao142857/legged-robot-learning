"""Disposable engineering chain: 3 x 2 Student PPO updates, one CPU Teacher step.
Never register or export this Teacher as a new Meta checkpoint. No formal tests.
"""
import os,sys,json,subprocess,time,hashlib,copy
from pathlib import Path
import numpy as np,torch
from teacher_fixed import Teacher
from cpu_hash import tree_hash
ROOT=Path('/home/alex/dog/teacher_adam_audit_20261109')
BASE=Path('/home/alex/dog/go2_piper_nsgs_tr0_20261109')
S=ROOT/'smoke';S.mkdir(exist_ok=False)
def write(p,v):p.write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not active,active
torch.set_num_threads(1)
source=BASE/'starts/gradient-seed43-cp01000/prepared.pt'
meta=BASE/'teachers/meta8.pt';hashes={str(p):sha(p) for p in (source,meta)}
old=torch.load(meta,map_location='cpu',weights_only=False)['teacher']
t=Teacher(np.load(BASE/'prepared/command_features.npy'),old['binding'],config=old['config']);t.load(old)
template=torch.load(BASE/'experience/meta-eight-starts/point_00/decision.pt',map_location='cpu',weights_only=False)
d=t.sample_pair(template['dynamic'],template['global_state'],'isolated-adam-smoke',template['progress'],template['progress_observed'],template['partial'])
torch.save(d,S/'decision.pt');before=t.state();h=tree_hash(before)
for arm in ('uniform','plus','minus'):np.save(S/(arm+'.npy'),np.ones(32768)/32768 if arm=='uniform' else d[arm]['probability'])
kit=' '.join([f'--/persistent/isaac/asset_root/{k}={BASE}/assets' for k in ('cloud','default','nvidia')]+['--/app/extensions/registryEnabled=false'])
jobs=[];results={};begin=time.time()
def run(label,args):
 p=S/(label+'.log');start=time.time();command=['bash',str(BASE/'code/python.sh'),*map(str,args)]
 with p.open('xb') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 jobs.append(dict(label=label,command=command,exit_code=r.returncode,wall_seconds=time.time()-start))
 write(S/'jobs.json',jobs);assert r.returncode==0,p
for arm in ('uniform','plus','minus'):
 out=S/arm
 run('train-'+arm,[BASE/'code/train_v02.py','--out',out,'--dataset',S/('experience-'+arm),'--mode','branch','--source',source,'--target','1002','--student-seed','43','--branch',arm,'--probability',S/(arm+'.npy'),'--decision-source',S/'decision.pt','--smoke','--headless','--kit_args',kit])
 assert not (out/'fault.txt').exists(),out/'fault.txt'
 result=read(out/'result.json');assert result['passed'] and result['new_updates']==2 and result['teacher_frozen']
 run('eval-'+arm,[BASE/'code/evaluate.py','--checkpoint',result['actor_checkpoint'],'--kind','development','--out',S/('eval-'+arm),'--headless','--kit_args',kit])
 ev=read(S/('eval-'+arm)/'result.json');assert ev['passed'] and ev['training_history_updates']==0 and ev['model_weights_unchanged']
 results[arm]=ev
baseline=read(BASE/'starts/gradient-seed43-cp01000/baseline_dev.json')
for ev in results.values():assert ev['initial_sha256']==baseline['initial_sha256'] and ev['update']==1002
scores=[baseline['continuous_joint_score']]+[results[a]['continuous_joint_score'] for a in ('uniform','plus','minus')]
info=t.learn_pair(d,*scores)
# Save only diagnostic artifact; cannot accidentally become a transfer Teacher.
torch.save(dict(non_reusable_engineering_only=True,teacher_state_for_audit=t.state(),response=info),S/'diagnostic_state.pt')
r=Teacher(np.load(BASE/'prepared/command_features.npy'),old['binding'],config=old['config']);r.load(torch.load(S/'diagnostic_state.pt',weights_only=False)['teacher_state_for_audit'])
assert tree_hash(r.state())==tree_hash(t.state()) and tree_hash(before)==h
assert all(sha(Path(p))==v for p,v in hashes.items())
old_steps={int(v['step']) for v in old['optimizer']['state'].values()};new_steps={int(v['step']) for v in t.state()['optimizer']['state'].values()}
assert new_steps==({n+1 for n in old_steps} if not info['zero_signal'] and not info['step_rejected'] else old_steps)
write(ROOT/'evidence/GPU_SMOKE.json',dict(passed=True,disposable_engineering_only=True,new_meta_teacher_registered=False,
 new_student_updates=6,new_training_transitions=6*4096*24,new_development_transitions=3*512000,
 source_hashes=hashes,source_assets_unchanged=True,scores=scores,old_adam_steps=list(old_steps),new_adam_steps=list(new_steps),
 teacher_gradient_norm=info['gradient_norm_before_clip'],teacher_parameter_change=info['parameter_delta_l2'],attempts=info['step_backtracking'],
 teacher_zero_signal=info['zero_signal'],teacher_rejected=info['step_rejected'],full_state_restore_exact=True,wall_seconds=time.time()-begin,jobs=jobs))
print('ADAM_GPU_SMOKE_PASS',flush=True)
