import json
from common import ROOT,write
reference=json.loads((ROOT/'runs/student_seed49_initialization.json').read_text());rows={}
for arm in ('meta16','meta24'):
 p=ROOT/'runs'/('transfer-seed49-'+arm)
 r=json.loads((p/'result.json').read_text());i=json.loads((p/'initialization.json').read_text())
 assert r['passed'] and r['completed_updates']==10000 and r['environment_transitions']==983040000 and r['teacher_frozen'] and i==reference
 tests=[json.loads(q.read_text()) for q in sorted((ROOT/'evaluations').glob('transfer-seed49-'+arm+'-*-test/result.json'))]
 assert [x['update'] for x in tests]==list(range(1000,10001,1000))
 rows[arm]=dict(result=r,curve=[{k:x[k] for k in ('update','joint_success_rate','continuous_joint_score','pooled_rmse','native_failure_rate')} for x in tests])
write(ROOT/'results/seed49_meta_pair.json',dict(passed=True,student_seed=49,initialization_matches_yanglab_uniform=True,hardware='All five seed49 arms run on YangLab RTX3060; initial actor/critic/empty optimizer match Uniform',teacher24_new_counterfactual_updates=2400,student_arms=rows))
print('SEED49_META16_META24_PAIR_COMPLETE',flush=True)
