"""Read-only accounting; parent GPU waits and nested eval time are not double-counted."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def report():
 groups={k:dict(student_updates=0,training_transitions=0,development_transitions=0,test_transitions=0,
               development_job_seconds=0.,test_job_seconds=0.,student_job_seconds=0.,
               startup_to_ready_seconds=0.,exposed_restore_seconds=0.,segments=0) for k in ('smoke','base','recovery')}
 for run in (ROOT/'runs').iterdir():
  if not run.is_dir():continue
  stage=run.name.split('-')[0]
  if stage not in groups:continue
  g=groups[stage];p=run/'updates.jsonl'
  n=sum(1 for _ in p.open()) if p.exists() else 0
  g['student_updates']+=n;g['training_transitions']+=n*4096*24;g['segments']+=1
  p=ROOT/'jobs'/run.name/'result.json'
  if p.exists():g['student_job_seconds']+=read(p)['wall_seconds']
  p=run/'startup_timing.json'
  if p.exists():g['startup_to_ready_seconds']+=read(p)['train_entry_to_ready_seconds']-read(p)['evaluation_seconds_before_learning']
  p=run/'restore_receipt.json'
  if p.exists():g['exposed_restore_seconds']+=read(p).get('exposed_state_restore_seconds',0.)
 for p in (ROOT/'evaluations').glob('*/result.json'):
  e=read(p);stage=p.parent.name.split('-')[0]
  if stage not in groups:continue
  kind=e['kind'];g=groups[stage];g[kind+'_transitions']+=e['executed_environment_transitions']
  job=ROOT/'jobs'/('eval-'+p.parent.name)/'result.json'
  if job.exists():g[kind+'_job_seconds']+=read(job)['wall_seconds']
 result=dict(groups=groups,notes=[
  'Student job wall includes nested dev/test and Teacher RPC waits; never add nested evaluation walls to it as disjoint GPU device time.',
  'Startup-to-ready excludes initial dev evaluation and includes restoration; exposed restore seconds are a subset, not additional cost.',
  'First smoke segment predates startup timing instrumentation; its complete parent job wall and raw console remain available.',
  'PhysX hidden cache cannot be serialized; restore equality refers to captured exposed state.'])
 p=ROOT/'COST_BREAKDOWN.json';p.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':report()
