"""YangLab serial continuation: Meta24 real experience, then frozen49 Meta16/24."""
import json,shutil,time,traceback,subprocess
from pathlib import Path
from host import ROOT,EXPERIENCE,KIT,run,write,outside

BASE=Path('/data/go2_piper_nsgs_v02_seed49_yanglab_20261108')
def read(p):return json.loads(Path(p).read_text())
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),student_seed=49,**kw))
def sha(p):
 import hashlib
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def evaluate(label,cp,kind='development'):
 out=ROOT/'evaluations'/label
 run('eval-'+label,[ROOT/'code/evaluate.py','--checkpoint',cp,'--kind',kind,'--out',out,'--headless','--kit_args',KIT])
 a=read(out/'result.json')
 assert a['passed'] and a['checkpoint_sha256']==sha(cp) and a['training_history_updates']==0 and a['model_weights_unchanged']
 identity={k:a[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}
 assert read(ROOT/f'evaluation_identity_{kind}.json')==identity
 return out/'result.json'

def train(label,target,seed,mode='branch',**options):
 out=ROOT/'runs'/label
 args=[ROOT/'code/train_v02.py','--out',out,'--dataset',EXPERIENCE/label,'--mode',mode,'--target',str(target),'--student-seed',str(seed),'--headless','--kit_args',KIT]
 for k,v in options.items():
  args.append('--'+k.replace('_','-'))
  if v is not True:args.append(v)
 def tick(process):
  for req in sorted((out/'evaluation_requests').glob('*.json')):
   if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():continue
   request=read(req);cp=outside(request['checkpoint'])
   assert cp.parent==out/'actors' and sha(cp)==request['checkpoint_sha256']
   result=evaluate(label+'-'+req.stem,cp,request['kind']);reply=read(result);assert reply['update']==request['update']
   write(req.with_suffix('.done.json'),reply)
 status('training_'+mode,label=label,target=target,source_student_seed=seed)
 run(label,args,on_tick=tick if mode=='transfer' else None)
 assert not (out/'fault.txt').exists()
 result=read(out/'result.json');assert result['passed'] and result['completed_updates']==target
 if mode=='transfer':
  assert result['environment_transitions']==983040000 and result['teacher_frozen']
  assert read(out/'initialization.json')==read(BASE/'runs/student_seed49_initialization.json')
  tests=[read(p) for p in sorted((ROOT/'evaluations').glob(label+'-*-test/result.json'))]
  assert [x['update'] for x in tests]==list(range(1000,10001,1000))
  write(ROOT/'completed'/(label+'.json'),dict(passed=True,result=result,curve=tests))
 return result

def main():
 status('cpu_preflight')
 run('cpu-tests',['-m','unittest','discover','-s',ROOT/'tests','-v'],gpu=False)
 run('inherit-meta16',[ROOT/'code/prepare_meta24.py'],gpu=False)
 run('placement-preflight',[ROOT/'code/placement_preflight.py'],gpu=False)
 while not (BASE/'completed/meta8.json').exists():
  for f in ('FAULT.txt','META8_FAULT.txt'):
   if (BASE/f).exists():raise RuntimeError('Prior seed49 queue fault: '+str(BASE/f))
  status('waiting_yanglab_meta8',gpu_jobs_started=False,queued_teacher_new_updates=2400,queued_student_arms=['meta16','meta24'])
  time.sleep(15)
 assert read(BASE/'completed/meta8.json')['passed']
 while True:
  active=subprocess.check_output(['/usr/bin/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if not active:break
  status('waiting_gpu_resource',active_compute_pids=active);time.sleep(15)
 teacher=ROOT/'teachers/meta16.pt'
 sources=read(ROOT/'sources.json')['sources']
 for index,s in enumerate(sources):
  u,seed=s['updates'],s['seed'];label=f'p{index:02d}-seed{seed}-cp{u:05d}';start=outside(s['path'])
  status('prepare_counterfactual',index=index,source_student_seed=seed,source_update=u)
  run('prepare-'+label,[ROOT/'code/prepare_meta24.py','--source-index',str(index)],gpu=False)
  before=evaluate(label+'-before',start)
  point=EXPERIENCE/'meta-eight-starts'/f'point_{index:02d}'
  run('decision-'+label,[ROOT/'code/teacher_ops.py','decision','--out',point,'--teacher',teacher,'--source',start,'--before',before,'--version',str(16+index)],gpu=False)
  ev={}
  for arm in ('uniform','plus','minus'):
   result=train(label+'-'+arm,u+100,seed,source=start,probability=point/f'{arm}.npy',branch=arm,decision_source=point/'decision.pt')
   assert result['new_updates']==100
   ev[arm]=evaluate(label+'-'+arm+'-after',outside(result['actor_checkpoint']))
  run('audit-'+label,[ROOT/'code/audit_point.py','--index',str(index)],gpu=False)
  run('update-'+label,[ROOT/'code/teacher_ops.py','update','--out',point,'--teacher',teacher,*[v for arm in ('uniform','plus','minus') for v in ('--'+arm,ev[arm])]],gpu=False)
  teacher=point/'teacher_after.pt'
  status('teacher_experience_added',new_points=index+1,new_student_updates=(index+1)*300,teacher_total_updates=17+index)
 shutil.copy2(teacher,ROOT/'teachers/meta24.pt')
 run('verify-meta24',[ROOT/'code/verify_meta24.py'],gpu=False)
 for arm in ('meta16','meta24'):
  train('transfer-seed49-'+arm,10000,49,mode='transfer',transfer_kind='meta',teacher_in=ROOT/'teachers'/(arm+'.pt'))
 run('verify-student-pair',[ROOT/'code/verify_student_pair.py'],gpu=False)
 status('complete',teacher_updates=24,new_real_counterfactual_updates=2400,completed_student_arms=['meta16','meta24'],student_updates_each=10000)

if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'FAULT.txt').write_text(traceback.format_exc());status('fault',detail=traceback.format_exc());raise
