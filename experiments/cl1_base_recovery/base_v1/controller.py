"""Durable CL1 orchestration. GPU work is serial; Teacher updates are CPU-only."""
import hashlib,json,shutil,time,traceback,subprocess
from pathlib import Path
from host import ROOT,EXPERIENCE,KIT,run,write,outside

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),**kw))
def ops(label,op,stage,**kw):
 args=[ROOT/'code/cl1_ops.py',op,'--stage',stage]
 for k,v in kw.items():args+=['--'+k.replace('_','-'),str(v)]
 run('cpu-'+label,list(map(str,args)),gpu=False)
def evaluate(label,cp,kind):
 out=ROOT/'evaluations'/label
 run('eval-'+label,list(map(str,[ROOT/'code/evaluate.py','--checkpoint',cp,'--kind',kind,'--out',out,'--headless','--kit_args',KIT])))
 a=read(out/'result.json');assert a['passed'] and a['checkpoint_sha256']==sha(cp)
 assert a['training_history_updates']==0 and a['model_weights_unchanged']
 ident={k:a[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}
 ref=ROOT/f'evaluation_identity_{kind}.json'
 if ref.exists():assert read(ref)==ident
 else:write(ref,ident)
 return a

def train(stage,seed,target,resume=None,label=None):
 label=label or f'{stage}-seed{seed}-to{target:05d}'
 out=ROOT/'runs'/label
 args=[ROOT/'code/train_cl1.py','--out',out,'--dataset',EXPERIENCE/label,'--mode','transfer','--target',target,
       '--student-seed',seed,'--teacher-in',ROOT/'teachers'/stage/'proxy.pt','--transfer-kind','meta','--branch','cl1',
       '--headless','--kit_args',KIT]
 if resume:args+=['--resume',resume]
 def tick(process):
  for req in sorted((out/'evaluation_requests').glob('*.json')):
   if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():continue
   a=read(req);cp=outside(a['checkpoint']);assert cp.parent==out/'actors' and sha(cp)==a['checkpoint_sha256']
   result=evaluate(label+'-'+req.stem,cp,a['kind']);assert result['update']==a['update']
   write(req.with_suffix('.done.json'),result)
  for req in sorted((out/'teacher_requests').glob('*.pt')):
   if req.name.endswith(('.done.pt','.teacher_after.pt')) or req.with_suffix('.done.pt').exists():continue
   ops(label+'-'+req.stem,'request',stage,request=req)
   assert req.with_suffix('.done.pt').exists()
 status('training',stage=stage,student_seed=seed,target=target,label=label)
 run(label,list(map(str,args)),on_tick=tick)
 assert not (out/'fault.txt').exists(),(out/'fault.txt').read_text() if (out/'fault.txt').exists() else ''
 a=read(out/'result.json');assert a['passed'] and a['completed_updates']==target
 return outside(a['checkpoint'])

def recovery_plan():
 """Past/current source development only. No formal-test input or hindsight return ranking."""
 import random
 rng=random.Random(20261109);selected={};catalog={}
 for seed in (43,44):
  rows=[]
  for p in sorted((ROOT/'evaluations').glob(f'base-seed{seed}-*-development/result.json')):
   row=read(p);rows.append((row['update'],row,p.parent.name))
  rows.sort();events=[];normal=[]
  for i in range(4,len(rows)):
   u,r,label=rows[i]
   if not 500<=u<=9500:continue
   prior=[z[1] for z in rows[i-4:i-1]];recent=[z[1] for z in rows[i-1:i+1]]
   drop=sum(z['continuous_joint_score'] for z in recent)/2-sum(z['continuous_joint_score'] for z in prior)/3
   ee=sum(z['pooled_rmse']['ee_position_m'] for z in recent)/2-sum(z['pooled_rmse']['ee_position_m'] for z in prior)/3
   item=dict(seed=seed,update=u,delta_j=drop,delta_ee=ee,eval_label=label)
   (events if drop<=-.03 and ee>=.003 else normal).append(item)
  chosen=[]
  for e in events:
   if all(abs(e['update']-z['update'])>=500 for z in chosen):chosen.append(e)
   if len(chosen)==2:break
  pairs=[];used=[z['update'] for z in chosen]
  for e in chosen:
   candidates=[n for n in normal if all(abs(n['update']-u)>=500 for u in used)]
   if not candidates:break
   distance=min(abs(n['update']-e['update']) for n in candidates)
   n=rng.choice([n for n in candidates if abs(n['update']-e['update'])==distance]);used.append(n['update'])
   pairs.append([dict(e,condition='degradation'),dict(n,condition='normal')])
  selected[seed]=pairs;catalog[seed]=dict(eligible_events=events,normal_candidates=normal)
 batches=min(len(selected[s]) for s in (43,44));plan=[]
 for b in range(batches):
  plan.append([item for seed in (43,44) for item in selected[seed][b]])
 for batch in plan:
  for item in batch:
   # Every 100-step source checkpoint and its exact Teacher memory are retained.
   u=item['update'];seed=item['seed'];upper=((u+499)//500)*500
   runroot=ROOT/'runs'/f'base-seed{seed}-to{upper:05d}'
   item['checkpoint']=str(runroot/'checkpoints'/f'update_{u:05d}.pt')
   item['memory']=str(runroot/'teacher_requests'/f'{u//100:04d}-complete.teacher_after.pt')
   assert Path(item['checkpoint']).exists() and Path(item['memory']).exists(),item
 write(ROOT/'RECOVERY_PLAN.json',dict(rule='last2 J vs prior3 <= -0.03 AND EE >= +0.003m; past/current only',
    seed=20261109,batches=plan,selection_catalog=catalog,actual_student_budget=batches*2000,max_budget=4000,
    insufficient_events=batches<2))
 return plan

def costs():
 jobs=[read(p) for p in (ROOT/'jobs').glob('*/result.json')]
 evaluations=[read(p) for p in (ROOT/'evaluations').glob('*/result.json')]
 runs=[read(p) for p in (ROOT/'runs').glob('*/result.json')]
 # new_updates in the inherited runner is cumulative since original initialization;
 # each isolated segment has one JSONL row per actual new PPO update.
 updates={}
 for p in (ROOT/'runs').glob('*/updates.jsonl'):
  updates[p.parent.name]=sum(1 for _ in p.open())
 write(ROOT/'COSTS.json',dict(actual_updates_by_segment=updates,actual_updates=sum(updates.values()),
   training_transitions=sum(updates.values())*4096*24,
   evaluation_transitions=sum(x['executed_environment_transitions'] for x in evaluations),
   job_wall_seconds=sum(x['wall_seconds'] for x in jobs),
   gpu_job_wall_seconds=sum(x['wall_seconds'] for x in jobs if x['gpu']),
   note='GPU parent wall includes evaluation waits; summing nested GPU walls is allocated job time, not disjoint device elapsed time.',
   evaluations=len(evaluations),completed_segments=len(runs)))
def main():
 run('cpu-tests',[str(ROOT/'tests/test_cl1.py')],gpu=False)
 run('cpu-adam-regression',[str(ROOT/'tests/test_adam_runtime.py')],gpu=False)
 # Unit tests are necessary, but physics smoke additionally covers restored third macrostep.
 ops('smoke-init','init','smoke');sources={}
 for seed in (43,44):sources[seed]=train('smoke',seed,200)
 ops('smoke-update','update','smoke',batch=0,length=2)
 # Resume one student for a real macrostep under new Teacher, checking physical and memory restoration.
 train('smoke',43,300,resume=sources[43])
 write(ROOT/'SMOKE_ACCEPTANCE.json',dict(passed=True,student_updates=500,macro_samples_outer_update=4,
      real_two_students=True,outer_update=True,physical_resume=True,performance_gate=False))
 ops('base-init','init','base');sources={43:None,44:None}
 for batch in range(20):
  for seed in (43,44):sources[seed]=train('base',seed,(batch+1)*500,resume=sources[seed])
  ops(f'base-update-{batch:02d}','update','base',batch=batch)
  costs();status('base_batch_complete',batch=batch+1,source_updates_each=(batch+1)*500)
 shutil.copy2(ROOT/'teachers/base/current.pt',ROOT/'teachers/base/Base.pt')
 write(ROOT/'BASE_ACCEPTANCE.json',dict(passed=True,updates_each=10000,teacher_transitions=200,outer_batches=20,
    teacher_sha256=sha(ROOT/'teachers/base/Base.pt')))
 plan=recovery_plan()
 if plan:
  ops('recovery-init','init','recovery')
  for batch,fragments in enumerate(plan):
   for i,item in enumerate(fragments):
    label=f'recovery-b{batch}-f{i}-seed{item["seed"]}-from{item["update"]:05d}'
    ops(label+'-memory','fork-memory','recovery',seed=item['seed'],source_memory=item['memory'])
    train('recovery',item['seed'],item['update']+500,resume=item['checkpoint'],label=label)
    ops(label+'-finish','finish-fragment','recovery',seed=item['seed'])
   ops(f'recovery-update-{batch}','update','recovery',batch=batch)
  shutil.copy2(ROOT/'teachers/recovery/current.pt',ROOT/'teachers/recovery/Recovery.pt')
 costs();status('complete',base_student_updates=20000,recovery_updates=len(plan)*2000,recovery_coverage_complete=len(plan)==2)
if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'FAULT.txt').write_text(traceback.format_exc());costs();status('fault',detail=traceback.format_exc());raise
