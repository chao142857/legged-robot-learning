"""Validate Student + matching Teacher memory as one recoverable transaction."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from cpu_hash import tree_hash

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def pair(checkpoint, expected_seed=70159, expected_network=None):
 cp=Path(checkpoint); student=load(cp); info=student['infos'];u=int(info['completed_updates'])
 assert u>0 and info['student_seed']==expected_seed
 assert cp.stem==f'update_{u:05d}', 'Checkpoint filename/update identity mismatch'
 assert info['optimizer_calls']==u*20 and info['environment_snapshot_available']
 assert info['phase'] in ('ready','complete')
 pending=info['phase']=='ready'
 ref=info.get('cl1_teacher_snapshot')
 if ref:
  teacher=Path(ref['path'])
  if str(teacher).startswith('/workspace/'):teacher=cp.parents[3]/str(teacher)[len('/workspace/'):]
  assert sha(teacher)==ref['sha256'],'Teacher snapshot checksum differs'
 else:
  # Legacy checkpoints saved a pending decision at internal100 boundaries and
  # a completed decision at the block end. Pair the appropriate immutable RPC state.
  v=info['decision']['version'];kind='decide' if pending else 'complete'
  teacher=cp.parent.parent/'teacher_requests'/f'{v:04d}-{kind}.teacher_after.pt'
 state=load(teacher);s=state['students'][str(expected_seed)]
 assert bool(s['pending'] is not None)==pending,'Student / Teacher phase mismatch'
 assert len(s['history'])==len(s['transitions'])+int(pending)
 assert s['transitions'][-1]['completed_updates']==u,'Teacher feedback is ahead of or behind Student'
 net=tree_hash(state['learner']['net'])
 assert net==info['decision']['teacher_sha256'],'Teacher weights differ'
 if expected_network:assert net==expected_network
 if pending:
  t=s['pending'];assert t['macro_version']==info['decision']['version']
  assert t['version']==info['decision']['teacher_version']
  np.testing.assert_array_equal(t['action'],info['decision']['preference'])
  assert abs(t['x'][0]-u/10000)<1e-6
 q=info['decision']['probability'];assert np.isfinite(q).all() and abs(q.sum()-1)<1e-12
 if info['transfer_kind']=='uniform':np.testing.assert_array_equal(q,np.full(len(q),1/len(q)))
 return dict(student_path=str(cp),student_sha256=sha(cp),teacher_path=str(teacher),teacher_sha256=sha(teacher),
   completed_updates=u,pending=pending,seed=expected_seed,transfer_kind=info['transfer_kind'],teacher_network_sha256=net,
   preserved_pending_decision=pending,passed=True)
