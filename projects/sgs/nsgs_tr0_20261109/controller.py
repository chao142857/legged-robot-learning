"""Persistent Ubuntu TR0 queue; no12k run without a distinct real intervention."""
import json,shutil,subprocess,time,traceback,sys
from pathlib import Path
from host import ROOT,run,KIT,write
sys.path.insert(0,str(ROOT/'code'))
from experiment_spec import SOURCES,SPEC

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 import hashlib
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),host='ubuntu-robot',**kw))
def idle():
 while True:
  p=subprocess.run(['/usr/bin/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  if not p.stdout.strip():return
  status('waiting_existing_gpu_job',active_pids=p.stdout.strip());time.sleep(15)

def evaluate(label,checkpoint,idle_first=True):
 if idle_first:idle()
 out=ROOT/'evaluations'/label
 run('eval-'+label,[str(ROOT/'code/evaluate.py'),'--checkpoint',str(checkpoint),'--kind','development','--out',str(out),'--headless','--kit_args',KIT])
 result=read(out/'result.json');assert result['passed'] and result['kind']=='development' and result['training_history_updates']==0 and result['model_weights_unchanged']
 assert result['checkpoint_sha256']==sha(checkpoint)
 identity={k:result[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}
 reference=ROOT/'evaluation_identity_development.json'
 if reference.exists():assert read(reference)==identity
 else:write(reference,identity)
 return out/'result.json'

def training(label,source,target,teacher=None,arm=None,smoke=False,repeat_seed=None):
 idle();out=ROOT/'runs'/label;dataset=ROOT/'experience'/label;u=source['update']
 start=ROOT/'starts'/source['id'];cp=start/'prepared.pt'
 args=[str(ROOT/'code/train_tr0.py' if teacher else ROOT/'code/train_v02.py'),
  '--out',str(out),'--dataset',str(dataset),'--mode','transfer' if teacher else 'branch',
  '--source',str(cp),'--target',str(target),'--student-seed',str(source['seed']),
  '--branch',arm or 'old','--headless','--kit_args',KIT]
 if teacher:
  args+=['--tr0','--transfer-kind','meta','--teacher-in',str(teacher),'--source-row',str(start/'progress_row.json'),'--baseline-dev',str(start/'baseline_dev.json')]
  if repeat_seed is not None:args+=['--repeat-seed',str(repeat_seed)]
 else:
  point=ROOT/'experience/meta-eight-starts/point_00'
  args+=['--probability',str(point/(arm+'.npy')),'--decision-source',str(point/'decision.pt')]
 if smoke:args+=['--smoke']
 def tick(process):
  for req in sorted((out/'evaluation_requests').glob('*.json')):
   if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():continue
   request=read(req);cp=Path(request['checkpoint']);assert cp.parent==out/'actors' and sha(cp)==request['checkpoint_sha256']
   reply_path=evaluate(label+'-'+req.stem,cp,idle_first=False);reply=read(reply_path);assert reply['update']==request['update']
   write(req.with_suffix('.done.json'),reply)
 status('gpu_smoke' if smoke else 'validation_training' if teacher else 'fresh_direction_training',label=label,
   source_student_seed=source['seed'],source_update=u,new_updates=target-u,teacher=arm,teacher_frozen=True)
 run(label,args,on_tick=tick if teacher else None)
 assert not (out/'fault.txt').exists();result=read(out/'result.json')
 assert result['passed'] and result['completed_updates']==target and result['new_updates']==target-u
 assert result['new_environment_transitions']==(target-u)*4096*24
 if teacher:assert result['teacher_frozen']
 return result

def main():
 started=time.time();git=read(ROOT/'TR0_GIT_RECEIPT.json')
 assert git['push_verified'] and git['branch']=='research/nsgs-tr0-20261109'
 assert read(ROOT/'CPU_ACCEPTANCE.json')['passed'];assert len(read(ROOT/'SOURCE_RECEIPTS.json'))==5
 status('resource_check',git_commit=git['commit'])
 idle();resource=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True)
 import shutil
 assert shutil.disk_usage(ROOT/'runs').free>40*1024**3
 write(ROOT/'RESOURCE_ACCEPTANCE.json',dict(passed=True,unix=started,gpu=resource.strip(),data_disk_free_bytes=shutil.disk_usage(ROOT/'runs').free,
   no_existing_compute_jobs=True,existing_yanglab_queue_untouched=True))
 gradient=SOURCES[0];cp=ROOT/'starts'/gradient['id']/'prepared.pt';prefix='p00-seed43-cp01000'
 status('development_baseline',source=gradient['id'])
 baseline=evaluate(prefix+'-before',cp);shutil.copyfile(baseline,cp.parent/'baseline_dev.json')
 repeated=evaluate(prefix+'-before-repeat',cp)
 write(ROOT/'BASELINE_RESTORE_NOISE.json',dict(first=str(baseline),repeat=str(repeated),
  continuous_difference=read(repeated)['continuous_joint_score']-read(baseline)['continuous_joint_score'],
  scope='One fixed-policy repeat; not a variance estimate or a tuning criterion'))
 # The only smoke PPO budget: two local updates, excluded from real gradients.
 smoke=training('engineering-smoke-old',gradient,1002,ROOT/'teachers/meta8.pt',arm='old',smoke=True)
 assert smoke['new_updates']==2
 write(ROOT/'GPU_SMOKE_ACCEPTANCE.json',dict(passed=True,student_updates=2,environment_transitions=196608,
  checkpoint_sha256=smoke['checkpoint_sha256'],teacher_frozen=True,native_ppo_unchanged=True))
 point=ROOT/'experience/meta-eight-starts/point_00'
 run('new-real-decision',[str(ROOT/'code/teacher_ops.py'),'decision','--out',str(point),'--teacher',str(ROOT/'teachers/meta8.pt'),
  '--source',str(cp),'--before',str(baseline),'--version','8'],gpu=False)
 for arm in ('uniform','plus','minus'):
  result=training(prefix+'-'+arm,gradient,1100,arm=arm)
  evaluate(prefix+'-'+arm+'-after',Path(result['actor_checkpoint']))
 # One after-policy repeat quantifies fixed-evaluation noise; first outcomes alone
 # construct the real gradient. It never chooses the sign, step, radius or state.
 after=read(ROOT/'runs'/(prefix+'-plus')/'result.json')
 evaluate(prefix+'-plus-after-repeat',Path(after['actor_checkpoint']))
 for source in SOURCES[1:]:
  status('validation_source_baseline',source=source['id'])
  cp=ROOT/'starts'/source['id']/'prepared.pt'
  baseline=evaluate(source['id']+'-before',cp);shutil.copyfile(baseline,cp.parent/'baseline_dev.json')
 run('fixed-reference-panel',[str(ROOT/'code/prepare_tr0.py'),'--actual-panel'],gpu=False)
 run('new-real-feedback-audit',[str(ROOT/'code/audit_point.py'),'--index','0'],gpu=False)
 run('construct-original-and-sampling-tr',[str(ROOT/'code/make_candidates.py')],gpu=False)
 candidate=read(ROOT/'CANDIDATE_RESULT.json')
 if not candidate['launch_12k']:
  reason=candidate['stop_reason']
  status('no_intervention_no_12k' if not candidate['constraint_intervention'] else 'stopped_invalid_controls',
    reason=reason,original_max_tv=candidate['original_max_tv'],tv_limit=.02,
    formal_validation_updates=0,fresh_direction_updates=300,engineering_updates=2,wall_seconds=time.time()-started)
  run('final-report',[str(ROOT/'code/report_tr0.py')],gpu=False)
  return
 status('formal_queue_start',formal_validation_updates=12000,source_seeds=[44,46,47,48],paired_repeats=2,
   original_max_tv=candidate['original_max_tv'],accepted_max_tv=candidate['max_accepted_tv'])
 for source in SOURCES[1:]:
  for repeat in range(2):
   repeat_seed=None if repeat==0 else 42000000+source['seed']*100+source['update']
   order=('old','original','sampling-tr') if repeat==0 else ('sampling-tr','original','old')
   for arm in order:
    label=source['id']+'-r%d-'%repeat+arm
    training(label,source,source['update']+500,ROOT/'teachers'/(arm+'.pt'),arm=arm,repeat_seed=repeat_seed)
   run('audit-'+source['id']+'-r%d'%repeat,[str(ROOT/'code/report_tr0.py'),'--audit-pair',source['id'],'--repeat',str(repeat)],gpu=False)
 run('final-report',[str(ROOT/'code/report_tr0.py')],gpu=False)
 assert read(ROOT/'RESULTS.json')['formal_validation_updates']==12000
 status('complete',formal_validation_updates=12000,fresh_direction_updates=300,engineering_updates=2,
   primary_endpoint='+500 development continuous joint-score gain',wall_seconds=time.time()-started)

if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'FAULT.txt').write_text(traceback.format_exc());status('fault',detail=traceback.format_exc());raise
