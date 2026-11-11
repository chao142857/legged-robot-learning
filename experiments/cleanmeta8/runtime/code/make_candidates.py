"""The only new Teacher update consumes newly collected TR0 branch outcomes."""
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,write,file_sha
from cpu_hash import tree_hash
from teacher_v02 import Teacher,state_hash
from trust_region import construct_teachers,frozen_input
from experiment_spec import META8_SHA

point=ROOT/'experience/meta-eight-starts/point_00'
d=torch.load(point/'decision.pt',map_location='cpu',weights_only=False)
assert d['version']==8 and d['teacher_updates']==8 and not d['source_preparation'].get('new_teacher_gradient_data',True)
assert d['source_original']['seed']==43 and d['source_original']['updates']==1000
assert file_sha(ROOT/'teachers/meta8.pt')==META8_SHA
engineering=json.loads((point/'engineering.json').read_text());assert engineering['passed']
results={arm:json.loads((ROOT/'evaluations'/('p00-seed43-cp01000-'+arm+'-after')/'result.json').read_text()) for arm in ('uniform','plus','minus')}
before=d['before_development']
for x in results.values():
 assert x['kind']=='development' and x['update']==1100 and x['training_history_updates']==0
 assert x['initial_sha256']==before['initial_sha256'] and x['pool_sha256']==before['pool_sha256']
manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
t=Teacher(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False),manifest['binding']);t.load(torch.load(ROOT/'teachers/meta8.pt',map_location='cpu',weights_only=False)['teacher'])
assert tree_hash(t.encoder.state_dict())==tree_hash(d['teacher']['encoder'])
panel=torch.load(ROOT/'panels/reference12.pt',map_location='cpu',weights_only=False)
h=frozen_input(t,'actual-gradient-decision',d,d['global_state'],dict(decision_path=str(point/'decision.pt'),decision_sha256=file_sha(point/'decision.pt'),formal_test_used=False))
assert h['basis_sha256']==d['basis_sha256'];panel.append(h);assert len(panel)==13
if (ROOT/'panels/fixed13.pt').exists():
 assert tree_hash(panel)==tree_hash(torch.load(ROOT/'panels/fixed13.pt',map_location='cpu',weights_only=False)),'Frozen inputs changed'
else:
 torch.save(panel,ROOT/'panels/fixed13.pt')
 write(ROOT/'panels/fixed13.json',[{k:v for k,v in x.items() if k not in ('basis','global_state','dynamic','progress','progress_observed','partial')} for x in panel])
scores=[before['continuous_joint_score'],*[results[a]['continuous_joint_score'] for a in ('uniform','plus','minus')]]
old,original,tr,report=construct_teachers(t,d,scores,panel,evidence_dir=ROOT/'candidates/new_real')
for name,teacher in (('old',old),('original',original),('sampling-tr',tr)):
 path=ROOT/'teachers'/(name+'.pt');assert not path.exists()
 torch.save(dict(teacher=teacher.state(),tr0=dict(source_meta8_sha256=META8_SHA,new_real_decision_sha256=file_sha(point/'decision.pt'),
   source_version=8,formal_test_used=False,kind=name,tv_limit=.02)),path)
report.update(fresh_direction_updates=300,fresh_direction_scores=dict(before=scores[0],uniform=scores[1],plus=scores[2],minus=scores[3]),
 real_decision_sha256=file_sha(point/'decision.pt'),fixed_panel_sha256=file_sha(ROOT/'panels/fixed13.pt'),
 fresh_feedback_only=True,historical_replay_for_learning=False,formal_test_used=False,
 teacher_networks={name:state_hash(x.net.state_dict()) for name,x in (('old',old),('original',original),('sampling-tr',tr))})
distinct=len(set(report['teacher_networks'].values()))==3
report['three_distinct_teacher_parameters']=distinct
report['launch_12k']=report['constraint_intervention'] and report['accepted'] and distinct
report['stop_reason']=None if report['launch_12k'] else ('new_bound_not_binding_B_equals_C' if not report['constraint_intervention'] else 'three_distinct_controls_not_formed')
write(ROOT/'CANDIDATE_RESULT.json',report)
print('TR0_REAL_CANDIDATE '+json.dumps({k:report[k] for k in ('original_max_tv','max_accepted_tv','constraint_intervention','three_distinct_teacher_parameters','launch_12k','stop_reason')}),flush=True)
