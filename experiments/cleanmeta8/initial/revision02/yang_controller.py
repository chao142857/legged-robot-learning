import json,time,traceback,subprocess,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from host import ROOT,EXPERIENCE,KIT,run,outside,write
BASE=Path('/data/go2_piper_nsgs_v02_seed49_yanglab_20261108')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),seed=49,**kw))
def evaluate(label,cp,kind):
 out=ROOT/'evaluations'/label
 run('eval-'+label,[ROOT/'code/evaluate.py','--checkpoint',cp,'--kind',kind,'--out',out,'--headless','--kit_args',KIT])
 r=read(out/'result.json');assert r['passed'] and r['training_history_updates']==0 and r['model_weights_unchanged'] and r['checkpoint_sha256']==sha(cp)
 identity={k:r[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}
 assert identity==read(BASE/f'evaluation_identity_{kind}.json')
 return r
try:
 status('waiting_CleanMeta8',new_student_budget=10000,reuse=['uniform','fresh'],legacy_meta8_stopped_at=2632)
 subprocess.run(['python3',str(ROOT/'revision02/bridge.py'),'receive'],check=True)
 run('install-CleanMeta8',[ROOT/'code/compact_teacher.py','install'],gpu=False)
 assert read(ROOT/'IMPORT_ACCEPTANCE.json')['passed']
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():status('waiting_gpu');time.sleep(30)
 label='transfer-seed49-clean-meta8';out=ROOT/'runs'/label
 def tick(process):
  for p in sorted((out/'evaluation_requests').glob('*.json')):
   if p.name.endswith('.done.json') or p.with_suffix('.done.json').exists():continue
   req=read(p);cp=outside(req['checkpoint']);assert cp.parent==out/'actors' and sha(cp)==req['checkpoint_sha256']
   answer=evaluate(label+'-'+p.stem,cp,req['kind']);assert answer['update']==req['update'];write(p.with_suffix('.done.json'),answer)
 status('training_CleanMeta8',target=10000,teacher_frozen=True)
 run(label,[ROOT/'code/train_v02.py','--mode','transfer','--out',out,'--dataset',EXPERIENCE/label,'--target','10000','--student-seed','49','--branch','clean-meta8','--transfer-kind','meta','--teacher-in',ROOT/'teachers/CleanMeta8.pt','--headless','--kit_args',KIT],on_tick=tick)
 assert not (out/'fault.txt').exists();r=read(out/'result.json');assert r['passed'] and r['new_updates']==10000 and r['teacher_frozen'] and r['environment_transitions']==983040000
 assert read(out/'initialization.json')==read(BASE/'runs/student_seed49_initialization.json')
 run('compare-existing-baselines',[ROOT/'code/comparison_reuse.py','--seed','49'],gpu=False)
 status('complete',new_student_updates=10000,retrained_baselines=False)
except BaseException:
 (ROOT/'FAULT.txt').write_text(traceback.format_exc());status('fault',detail=traceback.format_exc());raise
