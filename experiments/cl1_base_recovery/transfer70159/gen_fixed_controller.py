"""Frozen three-arm continuation. Paired checkpoints only; faults stop, never force-remove a live job."""
import argparse,fcntl,hashlib,json,subprocess,time,traceback
from pathlib import Path
from host import ROOT,EXPERIENCE,KIT,run,write,outside
OLD=Path('/data/go2_piper_nsgs_cl1_gen_seed70159_20261110')
SEED=70159

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),**kw))
def cpu(label,args):return run(label,list(map(str,args)),gpu=False)
def evaluate(label,cp,kind):
 out=ROOT/'evaluations'/label
 run('eval-'+label,list(map(str,[ROOT/'code/evaluate.py','--checkpoint',cp,'--kind',kind,'--out',out,'--headless','--kit_args',KIT])))
 a=read(out/'result.json');assert a['passed'] and a['checkpoint_sha256']==sha(cp) and a['training_history_updates']==0 and a['model_weights_unchanged']
 assert {k:a[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}==read(ROOT/f'evaluation_identity_{kind}.json')
 return a

def attempt(arm,target,source=None,smoke=False):
 # Completed successful segments are immutable. New attempts always use new names.
 prefix=f'{arm}-seed{SEED}-to{target:05d}'
 successful=[]
 for p in (ROOT/'runs').glob(prefix+'-a*/result.json'):
  x=read(p)
  if x.get('passed') and x['completed_updates']==target:successful.append((p,x))
 if successful:
  assert len(successful)==1;return outside(successful[0][1]['checkpoint'])
 previous=sorted((ROOT/'runs').glob(prefix+'-a*'))
 # After an inspected failure, restart from the latest atomically saved paired checkpoint;
 # fixed_prepare validates phase, actor-update count, feedback boundary and immutable hashes.
 candidates=[]
 for out in previous:
  for cp in (out/'checkpoints').glob('update_*.pt'):
   u=int(cp.stem.split('_')[1])
   if 0<u<target:candidates.append((u,cp))
 if candidates:source=max(candidates,key=lambda x:x[0])[1]
 reserved=list((ROOT/'attempts').glob(prefix+'-a*.json'))
 index=max([int(p.stem.rsplit('-a',1)[1]) for p in reserved]+[int(p.name.rsplit('-a',1)[1]) for p in previous]+[0])+1
 label=prefix+f'-a{index:02d}';out=ROOT/'runs'/label
 write(ROOT/'attempts'/f'{label}.json',dict(arm=arm,target=target,source=str(source),unix=time.time()))
 proxy=ROOT/'teachers'/arm/'proxy.pt'
 if source:
  proxy=ROOT/'resume_inputs'/f'{label}.pt'
  cpu('pair-'+label,[ROOT/'code/fixed_prepare.py','restore','--arm',arm,'--checkpoint',source,'--out',proxy])
 else:
  # A fresh-arm restart must also restore empty Teacher memory, never use an advanced current.pt.
  assert not previous,'No completed boundary: reset from initial.pt explicitly after auditing the failed attempt'
 args=[ROOT/'code/train_cl1.py','--out',out,'--dataset',EXPERIENCE/label,'--mode','transfer','--target',target,
       '--student-seed',SEED,'--teacher-in',proxy,'--transfer-kind','uniform' if arm in ('uniform','probe') else 'meta',
       '--branch','cl1','--headless','--kit_args',KIT]
 if source:args+=['--resume',source]
 if smoke:args+=['--smoke']
 def tick(process):
  for req in sorted((out/'evaluation_requests').glob('*.json')):
   if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():continue
   d=read(req);cp=outside(d['checkpoint']);assert cp.parent==out/'actors' and sha(cp)==d['checkpoint_sha256']
   result=evaluate(label+'-'+req.stem,cp,d['kind']);assert result['update']==d['update'];write(req.with_suffix('.done.json'),result)
  for req in sorted((out/'teacher_requests').glob('*.pt')):
   if req.name.endswith(('.done.pt','.teacher_after.pt')) or req.with_suffix('.done.pt').exists():continue
   cpu('rpc-'+label+'-'+req.stem,[ROOT/'code/frozen_ops.py','request','--stage',arm,'--request',req])
 status('training',arm=arm,target=target,label=label,source=str(source),seed=SEED)
 run(label,list(map(str,args)),on_tick=tick)
 assert not (out/'fault.txt').exists(),str(out/'fault.txt')
 a=read(out/'result.json');assert a['passed'] and a['completed_updates']==target
 cp=outside(a['checkpoint'])
 cpu('verify-'+label,[ROOT/'code/fixed_prepare.py','verify','--arm',arm])
 write(ROOT/'segments'/f'{label}.json',dict(passed=True,arm=arm,target=target,source=str(source),checkpoint=str(cp),checkpoint_sha256=sha(cp),run=str(out)))
 return cp

def main(smoke):
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 binding=read(ROOT/'CODE_BINDING.json')
 for rel,digest in binding['files'].items():assert sha(ROOT/rel)==digest, 'Runtime source drift: '+rel
 running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not running,'GPU busy: '+running
 if smoke:
  cp=attempt('probe',1410,OLD/'runs/uniform-seed70159-to01500/checkpoints/update_01400.pt',smoke=True)
  cpu('validate-physical-resume',[ROOT/'code/verify_probe.py','--checkpoint',cp])
  status('repair_smoke_complete',updates=10);return
 assert read(ROOT/'PHYSICAL_RESUME_ACCEPTANCE.json')['passed']
 assert (ROOT/'RECOVERY_V2_LOCK.json').exists(),'Corrected Recovery has not been locked yet'
 for arm in ('uniform','frozen-base','frozen-recovery'):
  cp=None
  schedule=[500,1000,1400]+list(range(1500,10001,500))
  for target in schedule:cp=attempt(arm,target,cp)
  status('arm_complete',arm=arm,updates=10000)
 cpu('final-report',[ROOT/'code/report_fixed.py'])
 status('complete',seed=SEED,logical_student_updates=30000)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args()
 try:main(a.smoke_only)
 except BaseException:
  write(ROOT/'FAULT.json',dict(unix=time.time(),traceback=traceback.format_exc()));status('fault');raise
