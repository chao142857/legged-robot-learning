import argparse,json
import numpy as np
import torch
from common import ROOT,file_sha,write
from cpu_hash import tree_hash
from teacher_v02 import state_hash
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args()
a=torch.load(ROOT/'teachers/old-meta.pt',map_location='cpu',weights_only=False)['teacher'];b=torch.load(ROOT/'teachers/new-meta.pt',map_location='cpu',weights_only=False)['teacher']
assert a['updates']==8 and b['updates']==16
assert set(b['used_versions'])==set(str(i) for i in range(16))
assert a['config']==b['config'] and a['binding']==b['binding'] and tree_hash(a['encoder'])==tree_hash(b['encoder'])
network_changed=state_hash(a['network'])!=state_hash(b['network'])
assert file_sha(ROOT/'teachers/old-meta.pt')=='8929d0a13c4aaa7b98c70acf584d04390711bbc9ea6df8b61f3bcb556cdbd3b9'
updates=[]
for i in range(8):
 point=ROOT/'experience/meta-eight-starts'/f'point_{i:02d}'
 d=json.loads((point/'decision.json').read_text());e=json.loads((point/'engineering.json').read_text());u=json.loads((point/'update.json').read_text())
 assert d['version']==i+8 and not d['formal_test_inputs'] and e['passed'] and u['formal_test_used']==False
 updates.append(u)
result=dict(passed=True,student_seed=48,old_teacher_updates=8,new_teacher_updates=16,new_real_counterfactual_updates=2400,old_checkpoint_sha256=file_sha(ROOT/'teachers/old-meta.pt'),new_checkpoint_sha256=file_sha(ROOT/'teachers/new-meta.pt'),no_old_on_policy_replay=True,network_changed=network_changed,unchanged_encoder=True,unchanged_teacher_config=True,updates=updates)
if args.final:
 rows={};initial=None
 for arm in ('new-meta',):
  run=ROOT/'runs'/('transfer-seed48-'+arm);r=json.loads((run/'result.json').read_text());i=json.loads((run/'initialization.json').read_text())
  assert r['passed'] and r['completed_updates']==10000 and r['environment_transitions']==983040000 and r['teacher_frozen']
  if initial is None:initial=i
  else:assert i==initial
  rows[arm]=dict(result=r,formal_test_points=[])
  for path in sorted((ROOT/'evaluations').glob('transfer-seed48-'+arm+'-*-test/result.json')):
   e=json.loads(path.read_text());rows[arm]['formal_test_points'].append({k:e[k] for k in ('update','joint_success_rate','continuous_joint_score','pooled_rmse','native_failure_rate')})
  assert [e['update'] for e in rows[arm]['formal_test_points']]==list(range(1000,10001,1000))
 result['transfer']=rows
 result['comparison_initialization_reference']='Existing seed48 Uniform on the same ubuntu-robot GPU; exact actor/critic/empty-optimizer check before update1.'
write(ROOT/'results'/('final_summary.json' if args.final else 'extended_teacher.json'),result)
print('EXTENDED_TEACHER_AUDIT_PASS',args.final,flush=True)
