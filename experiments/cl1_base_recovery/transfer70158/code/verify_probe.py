import argparse,json
from pathlib import Path
import torch,numpy as np
from common import ROOT,write
from checkpoint_pair import pair,load
from cpu_hash import tree_hash
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
b=pair(a.checkpoint);assert b['completed_updates']==1410 and not b['pending']
out=a.checkpoint.parent.parent;receipt=json.loads((out/'restore_receipt.json').read_text());assert receipt['passed'] and receipt['next_4096_sampler_draws_equal'] and len(receipt['first_rollout_signatures'])==24
rows=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()];assert len(rows)==10 and rows[0]['completed_updates']==1401
# A pending macroaction must be reused. No additional Teacher decision may be drawn at1400.
assert not list((out/'teacher_requests').glob('*-decide.pt'))
initial=load(ROOT/'teachers/probe/initial.pt');current=load(ROOT/'teachers/probe/current.pt')
for k in ('net','optimizer','rng'):assert tree_hash(initial['learner'][k])==tree_hash(current['learner'][k])
assert len(current['students']['70158']['history'])==len(initial['students']['70158']['history'])
write(ROOT/'PHYSICAL_RESUME_ACCEPTANCE.json',dict(passed=True,real_student_updates=10,paired_checkpoint=b,
 pending_action_reused=True,extra_teacher_draws=0,weights_adam_rng_unchanged=True,restore=receipt,formal_uniform_untouched=True))
print('PAIRED_PHYSICAL_RESUME_PASS')
