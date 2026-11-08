"""Fresh, old-Meta and new-Meta seed48 from-zero queue on YangLab."""
import json,time,traceback
from pathlib import Path
from host import ROOT,EXPERIENCE,KIT,run,inside,outside,write

def read(p):return json.loads(Path(p).read_text())
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),**kw))
def sha(p):
 import hashlib
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def evaluate(label,cp,kind):
 out=ROOT/'evaluations'/label
 run('eval-'+label,[ROOT/'code/evaluate.py','--checkpoint',cp,'--kind',kind,'--out',out,'--headless','--kit_args',KIT])
 a=read(out/'result.json');assert a['passed'] and a['checkpoint_sha256']==sha(cp) and a['training_history_updates']==0 and a['model_weights_unchanged']
 identity={k:a[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}
 assert read(ROOT/f'evaluation_identity_{kind}.json')==identity
 return a

def main():
 status('cpu_preflight');run('cpu-preflight',[ROOT/'code/prepare_split.py'],gpu=False)
 assert read(ROOT/'PREFLIGHT.json')['passed']
 for arm in ('fresh','old-meta','new-meta'):
  label='transfer-seed48-'+arm;out=ROOT/'runs'/label
  def tick(process):
   for req in sorted((out/'evaluation_requests').glob('*.json')):
    if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():continue
    request=read(req);cp=outside(request['checkpoint']);assert cp.parent==out/'actors'
    reply=evaluate(label+'-'+req.stem,cp,request['kind']);assert reply['update']==request['update']
    write(req.with_suffix('.done.json'),reply)
  kind='fresh' if arm=='fresh' else 'meta'
  status('training',student_seed=48,arm=arm,target=10000,teacher_frozen=True)
  run(label,[ROOT/'code/train_v02.py','--mode','transfer','--out',out,'--dataset',EXPERIENCE/label,'--target','10000','--student-seed','48','--transfer-kind',kind,'--teacher-in',ROOT/'teachers'/f'{arm}.pt','--headless','--kit_args',KIT],on_tick=tick)
  assert not (out/'fault.txt').exists()
  result=read(out/'result.json');assert result['passed'] and result['completed_updates']==10000 and result['teacher_frozen']
  assert read(out/'initialization.json')==read(ROOT/'runs/student_seed48_initialization.json')
  write(ROOT/'completed'/f'{arm}.json',dict(passed=True,completed_updates=10000,result=result))
 status('complete',student_seed=48,completed_arms=['fresh','old-meta','new-meta'],updates=30000,uniform_host='ubuntu-robot')
if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'FAULT.txt').write_text(traceback.format_exc());status('fault',detail=traceback.format_exc());raise
