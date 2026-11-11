"""Read-only V0/V0.1 per-cycle Adam transition audit and source inventory."""
import json,hashlib
from pathlib import Path
import torch
from cpu_hash import tree_hash
ROOT=Path('/home/alex/dog');OUT=ROOT/'teacher_adam_audit_20261109/evidence'
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for folder in ('nsgs_teacher_experience_v0','nsgs_teacher_experience_v01_validation'):
 for p in sorted((ROOT/folder).glob('*/cycle_*/teacher_after.pt')):
  d=torch.load(p.parent/'decision.pt',weights_only=False,map_location='cpu');a=torch.load(p,weights_only=False,map_location='cpu')
  d=d.get('decision',d);b=d['teacher'];a=a.get('teacher',a)
  if 'optimizer' not in a:
   raise ValueError((str(p),list(a)))
  info=json.loads((p.parent/'learning.json').read_text())
  sb=sorted(set(int(v['step']) for v in b['optimizer']['state'].values()));sa=sorted(set(int(v['step']) for v in a['optimizer']['state'].values()))
  rejected=info['step_rejected'];frozen=info.get('teacher_frozen',False);expected=sb if rejected or frozen else [n+1 for n in sb] if sb else [1]
  rows.append(dict(path=str(p),checkpoint_sha256=sha(p),logical_before=b['updates'],logical_after=a['updates'],
   before_steps=sb,after_steps=sa,rejected=rejected,teacher_frozen=frozen,step_transition_correct=sa==expected,
   frozen_full_state_equal=tree_hash(a)==tree_hash(b) if frozen else None,
   rejected_full_optimizer_restored=tree_hash(a['optimizer'])==tree_hash(b['optimizer']) if rejected else None))
sources=[]
for r in ROOT.glob('go2_piper_nsgs*'):
 for name in ('teacher_v02.py','teacher_ops.py','prepare_continual.py','prepare_matched.py','prepare_meta24.py','neural_core.py','v0_core.py','neural_v01.py'):
  p=r/'code'/name
  if p.exists():sources.append(dict(path=str(p),sha256=sha(p)))
result=dict(cycles=rows,all_transitions_correct=all(x['step_transition_correct'] for x in rows),sources=sources)
(OUT/'V0_V01.json').write_text(json.dumps(result,indent=2));print(json.dumps({'cycles':len(rows),'all_transitions_correct':result['all_transitions_correct'],'rejected':sum(x['rejected'] for x in rows),'source_files':len(sources)}))
