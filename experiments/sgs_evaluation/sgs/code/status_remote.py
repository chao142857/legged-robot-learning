import json,pathlib,subprocess,datetime
r=pathlib.Path('/data/go2_piper_sgs_20261102')
print('CHECKED_BEIJING',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
print('STATUS', (r/'status.json').read_text())
for p in (r/'PIPELINE_FAULT.json',r/'COMPLETED.json'):
    if p.exists():print(p.name,p.read_text())
for p in sorted((r/'runs').glob('*/latest.json')):
    d=json.loads(p.read_text());s=d['summary']
    print('RUN',p.parent.name,dict(completed_updates=d['completed_updates'],environment_transitions=d['environment_transitions'],feedback=s['feedback_accepted'],histories=s['nonempty_histories'],probability_min=s['probability_min'],probability_max=s['probability_max']))
for p in sorted((r/'runs').glob('*/result.json')):
    d=json.loads(p.read_text());print('PASS',p.parent.name,{k:d[k] for k in ('passed','completed_updates','training_transitions_per_second_excluding_eval_wait','paused_evaluation_seconds','rng_roundtrip','restoration')})
print(subprocess.check_output(['docker','ps','--format','{{.Names}} {{.Status}}'],text=True))
