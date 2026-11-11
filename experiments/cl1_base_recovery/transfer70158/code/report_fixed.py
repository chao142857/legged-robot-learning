"""Logical learning curves are separate from retries, discarded work and nested job costs."""
import json,time
from pathlib import Path
from common import ROOT,write
OLD=Path('/data/go2_piper_nsgs_cl1_gen_seed70158_20261110')
ARMS=('uniform','frozen-base','frozen-recovery')
def read(p):return json.loads(p.read_text())
def main():
 result={}
 for arm in ARMS:
  data={}
  if arm=='uniform':
   for p in (OLD/'evaluations').glob('uniform-*/result.json'):
    a=read(p)
    if a['update']<=1400:data[(a['kind'],a['update'])]=a
  for p in (ROOT/'evaluations').glob(arm+'-*/result.json'):
   a=read(p);key=(a['kind'],a['update'])
   # Fault attempts are not silently mixed into final curves.
   label=p.parent.name.rsplit('-',2)[0]
   if not (ROOT/'segments'/f'{label}.json').exists():continue
   assert key not in data,'Duplicate evaluation requires explicit attempt adjudication'
   data[key]=a
  tests=[a for (k,u),a in sorted(data.items()) if k=='test']
  assert tests and tests[-1]['update']==10000 and len(tests)==10
  result[arm]=dict(final_test={k:tests[-1][k] for k in ('update','joint_success_rate','continuous_joint_score','pooled_rmse','native_failure_rate')},
   curve=[{k:a[k] for k in ('kind','update','joint_success_rate','continuous_joint_score','pooled_rmse')} for _,a in sorted(data.items())])
 executed={}
 for r in (OLD,ROOT):
  for p in (r/'runs').glob('*/updates.jsonl'):executed[str(p.parent)]=sum(1 for _ in p.open())
 actual=sum(executed.values());logical=30000
 jobs=[read(p) for p in (ROOT/'jobs').glob('*/result.json')]
 write(ROOT/'COMPARISON.json',dict(seed=70158,arms=result,descriptive_single_seed=True,
    recovery='V2: normal requires both indicators below degradation thresholds',old_recovery_not_tested=True))
 write(ROOT/'COSTS_FINAL.json',dict(unix=time.time(),logical_updates=logical,executed_student_and_probe_updates=actual,
    extra_or_discarded_updates=actual-logical,training_transitions=actual*4096*24,executed_by_attempt=executed,
    development_transitions=sum(read(p)['executed_environment_transitions'] for r in (OLD,ROOT) for p in (r/'evaluations').glob('*/result.json') if read(p)['kind']=='development'),
    test_transitions=sum(read(p)['executed_environment_transitions'] for r in (OLD,ROOT) for p in (r/'evaluations').glob('*/result.json') if read(p)['kind']=='test'),
    job_records=[dict(label=x['label'],wall_seconds=x['wall_seconds'],gpu=x['gpu']) for x in jobs],
    note='Parent Student job wall includes nested evaluations and CPU RPC waits; do not add these to derive elapsed GPU time. Source meta-training, old/new Recovery and initial500 smoke costs remain separately recorded in their roots.'))
if __name__=='__main__':main()
