"""Fixed 2200-update queue. No automatic long-training continuation."""
import json,time,hashlib,fcntl,subprocess,traceback
from pathlib import Path
from host import ROOT,EXPERIENCE,KIT,run,write,outside
P=json.loads((ROOT/'PROTOCOL.json').read_text())
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),**kw))
def cpu(label,op,**kwargs):
 args=[str(ROOT/'code/experiment_ops.py'),op]
 for k,v in kwargs.items():args+=['--'+k,str(v)]
 run(label,args,gpu=False)
def train(item,stage,label,collect):
 cp=Path(item['checkpoint']);assert sha(cp)==item['checkpoint_sha256'];target=item['start']+item['updates'];out=ROOT/'runs'/label
 args=[ROOT/'code/train_cl1.py','--out',out,'--dataset',EXPERIENCE/label,'--mode','transfer','--target',target,'--student-seed',item['seed'],'--teacher-in',ROOT/'teachers'/stage/'proxy.pt','--transfer-kind','meta','--branch','cl1','--resume',cp,'--headless','--kit_args',KIT]
 def tick(process):
  for req in sorted((out/'evaluation_requests').glob('*.json')):
   if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():continue
   d=read(req);assert d['kind']=='development','Formal test is forbidden in this calibration'
   checkpoint=outside(d['checkpoint']);assert sha(checkpoint)==d['checkpoint_sha256'];dest=ROOT/'evaluations'/(label+'-'+req.stem)
   run('eval-'+label+'-'+req.stem,list(map(str,[ROOT/'code/evaluate.py','--checkpoint',checkpoint,'--kind','development','--out',dest,'--headless','--kit_args',KIT])))
   a=read(dest/'result.json');assert a['passed'] and a['model_weights_unchanged'] and a['training_history_updates']==0
   assert {k:a[k] for k in ['initial_sha256','pool_sha256','seed','episodes','protocol_sha256']}==read(ROOT/'evaluation_identity_development.json')
   write(req.with_suffix('.done.json'),a)
  for req in sorted((out/'teacher_requests').glob('*.pt')):
   if req.name.endswith(('.done.pt','.teacher_after.pt')) or req.with_suffix('.done.pt').exists():continue
   script='collection_rpc.py' if collect else 'frozen_ops.py'
   run('rpc-'+label+'-'+req.stem,[str(ROOT/'code'/script),'request','--stage',stage,'--request',str(req)],gpu=False)
   if collect and req.stem=='0051-complete':
    receipt=read(out/'restore_receipt.json');assert receipt['passed'] and receipt['next_4096_sampler_draws_equal']
    write(ROOT/'engineering'/f'{label}-first100.json',dict(passed=True,real_student_updates=100,included_in_2200_budget=True,restoration=receipt,teacher_complete_reply=str(req.with_suffix('.done.pt'))))
 status('training',stage=stage,label=label,seed=item['seed'],from_update=item['start'],target=target)
 run(label,list(map(str,args)),on_tick=tick)
 a=read(out/'result.json');assert a['passed'] and a['completed_updates']==target
 assert sum(1 for _ in (out/'updates.jsonl').open())==item['updates']
 if not collect:cpu('verify-'+label,'verify',stage=stage,seed=item['seed'])
 write(ROOT/'segments'/f'{label}.json',dict(passed=True,updates=item['updates'],checkpoint=a['checkpoint']))
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 for rel,digest in read(ROOT/'CODE_BINDING.json')['files'].items():assert sha(ROOT/rel)==digest,rel
 run('cpu-epoch-tests',[str(ROOT/'code/test_epochs.py')],gpu=False)
 cpu('prepare','prepare')
 for item in P['collection']:
  train(item,'collect',f'collect-seed{item["seed"]}',True);cpu(f'finish-seed{item["seed"]}','finish',seed=item['seed'])
 cpu('paired-teacher-updates','update')
 for item in P['validation']:
  for arm in P['arms']:train(item,f'{arm}-seed{item["seed"]}',f'validate-{arm}-seed{item["seed"]}',False)
 cpu('final-report','report');status('complete',student_updates=2200,no_long_training_queued=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/'FAULT.json',dict(unix=time.time(),traceback=traceback.format_exc()));status('fault');raise
