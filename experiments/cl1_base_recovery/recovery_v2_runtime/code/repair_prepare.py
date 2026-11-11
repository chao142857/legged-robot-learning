from pathlib import Path
import json,copy,hashlib
import torch
from common import ROOT,write
from checkpoint_pair import pair,sha
from selection_fix import select,category
OLD=Path('/data/go2_piper_nsgs_cl1_gen_seed70158_20261110')
BASE=Path('/data/go2_piper_nsgs_cl1_20261109')
def main():
 cp=OLD/'runs/uniform-seed70158-to01500/checkpoints/update_01400.pt'
 bundle=pair(cp)
 write(ROOT/'UNIFORM_RESUME_PAIR.json',bundle)
 old=json.loads((BASE/'RECOVERY_PLAN.json').read_text());rows=select(old)
 for item in rows:
  u,seed=item['update'],item['seed'];upper=((u+499)//500)*500
  run=BASE/'runs'/f'base-seed{seed}-to{upper:05d}'
  item['checkpoint']=str(run/'checkpoints'/f'update_{u:05d}.pt')
  item['memory']=str(run/'teacher_requests'/f'{u//100:04d}-complete.teacher_after.pt')
  item['checkpoint_sha256']=sha(item['checkpoint']);item['memory_sha256']=sha(item['memory'])
  info=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)['infos']
  memory=torch.load(item['memory'],map_location='cpu',weights_only=False)['students'][str(seed)]
  assert info['completed_updates']==u and memory['pending'] is None
  assert memory['transitions'][-1]['completed_updates']==u
  assert category(item)==item['condition']
 write(ROOT/'RECOVERY_V2_PLAN.json',dict(version='normal-both-indicators-v2',
  rule=dict(degradation='delta_J <= -0.03 AND delta_EE >= 0.003m',normal='delta_J > -0.03 AND delta_EE < 0.003m',ambiguous='only one indicator triggers: excluded'),
  selection='Keep original degraded starts; nearest eligible nonoverlapping stage; original RNG seed20261109 breaks ties; no future performance input',
  from_teacher=str(BASE/'teachers/base/Base.pt'),base_sha256=sha(BASE/'teachers/base/Base.pt'),
  new_student_updates=2000,legacy_recovery_retained=True,legacy_recovery_updates=2000,fragments=rows))
 write(ROOT/'PREFLIGHT.json',dict(passed=True,uniform_recoverable_at=1400,new_recovery_budget=2000,
  source_fixed_runtime_sha256=sha(ROOT/'code/teacher_v02.py')))
 print(json.dumps(dict(passed=True,uniform_resume=bundle,fragments=[(x['seed'],x['update'],x['condition']) for x in rows])))
if __name__=='__main__':main()
