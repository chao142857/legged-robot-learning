"""One real Meta8 update, with an optional trust region on mixed sequence q.

No simulator/PPO gradient or historical on-policy replay is introduced. The
Original-step keeps the V0.2 gradient and KL rule, with immutable Adam loads.
Sampling-TR reuses exactly
its pre-clipping gradient and Adam direction, restoring the same old Adam state
before every candidate. The declared base LR is restored after an accepted step.
"""
import copy,math,contextlib
from pathlib import Path
import numpy as np
import torch
from teacher_v02 import Teacher,array_hash,state_hash
from cpu_hash import tree_hash

def distribution_change(old,new):
 old=np.asarray(old,dtype=np.float64);new=np.asarray(new,dtype=np.float64)
 assert old.shape==new.shape and old.ndim==1
 assert np.isfinite(old).all() and np.isfinite(new).all() and min(old.min(),new.min())>0
 assert abs(old.sum()-1)<1e-12 and abs(new.sum()-1)<1e-12
 ratio=new/old;n=len(old);k=int(np.ceil(n*.1))
 return dict(tv=float(np.abs(new-old).sum()/2),
  forward_kl_old_new=float(np.sum(old*np.log(old/new))),reverse_kl_new_old=float(np.sum(new*np.log(new/old))),
  ratio_quantiles=dict(zip(('min','p01','p10','p50','p90','p99','max'),np.quantile(ratio,[0,.01,.1,.5,.9,.99,1]).tolist())),
  top10_mass_old=float(np.sort(old)[-k:].sum()),top10_mass_new=float(np.sort(new)[-k:].sum()),
  old_probability_sha256=array_hash(old),new_probability_sha256=array_hash(new))

def panel_probabilities(teacher,panel):
 assert panel
 encoder_hash=tree_hash(teacher.encoder.state_dict());rows=[]
 with torch.no_grad():
  for h in panel:
   assert h['encoder_sha256']==encoder_hash,'Frozen panel normalization differs'
   assert h['basis_sha256']==array_hash(h['basis'])
   assert h['global_state_sha256']==array_hash(h['global_state'])
   rows.append(teacher.probability_from_preference(h['basis'],teacher.mean(h['global_state']).numpy()))
 return np.stack(rows)

def frozen_input(teacher,identifier,context,global_state,source):
 """Fit nothing: encode actual source feedback with fixed inherited statistics."""
 encoded=teacher.encoder.encode(context['dynamic'],context['progress'],context['progress_observed'],context['partial'])
 g=np.asarray(global_state,dtype=np.float64).copy()
 assert g.shape==(10,) and np.isfinite(g).all()
 return dict(id=identifier,basis=encoded['basis'],basis_sha256=encoded['basis_sha256'],global_state=g,
  global_state_sha256=array_hash(g),encoder_sha256=tree_hash(teacher.encoder.state_dict()),source=copy.deepcopy(source),
  dynamic=encoded['dynamic'],progress=encoded['progress'],progress_observed=encoded['progress_observed'],partial=encoded['partial'])

def clone(teacher,state=None):
 result=copy.deepcopy(teacher)
 if state is not None:result.load(copy.deepcopy(state))
 return result

@contextlib.contextmanager
def isolated_adam_loads(optimizer):
 """CPU Optimizer.load_state_dict may alias snapshot tensors. Never reuse them.

 The hook is local to this one isolated Teacher optimizer and removed before
 cloning/checkpointing. It changes neither the gradient nor the first suggested
 Adam direction, but prevents failed backtracks from advancing the old moments.
 Historical source modules/checkpoints are not edited.
 """
 had_override='load_state_dict' in optimizer.__dict__
 prior_override=optimizer.__dict__.get('load_state_dict')
 native_load=optimizer.load_state_dict
 def load_independent_snapshot(state):return native_load(copy.deepcopy(state))
 optimizer.load_state_dict=load_independent_snapshot
 try:yield
 finally:
  if had_override:optimizer.load_state_dict=prior_override
  else:optimizer.__dict__.pop('load_state_dict',None)

def construct_teachers(meta8,decision,scores,panel,tv_limit=.02,evidence_dir=None):
 """Construct A/B/C from one on-policy decision and its new real branch gains.

 Consuming the real decision advances version bookkeeping even if rejected;
 rejected parameters, Adam, encoder and private RNG remain at the pre-step state.
 B/C are bit-identical in full state when the extra bound does not bind.
 """
 assert 0<tv_limit<=1 and len(scores)==4
 assert meta8.updates==decision['teacher_updates']
 assert state_hash(meta8.net.state_dict())==decision['teacher_sha256']
 assert tree_hash(meta8.optimizer.state_dict())==tree_hash(decision['teacher']['optimizer'])
 assert tree_hash(meta8.encoder.state_dict())==tree_hash(decision['teacher']['encoder'])
 # All three alternatives begin after the one shared exploration draw. Old
 # keeps Meta8 weights/Adam; it does not retain a different private RNG cursor.
 old=clone(meta8,decision['teacher']);original=clone(meta8,decision['teacher'])
 with isolated_adam_loads(original.optimizer):
  original_info=original.learn_pair(decision,*scores)
 old_adam_steps={str(k):int(v['step']) for k,v in decision['teacher']['optimizer']['state'].items()}
 new_adam_steps={str(k):int(v['step']) for k,v in original.optimizer.state_dict()['state'].items()}
 if original_info['step_rejected'] or original_info['zero_signal']:
  assert tree_hash(original.optimizer.state_dict())==tree_hash(decision['teacher']['optimizer'])
 else:assert all(new_adam_steps[k]==v+1 for k,v in old_adam_steps.items())
 old_q=panel_probabilities(old,panel);original_q=panel_probabilities(original,panel)
 original_changes=[dict(id=h['id'],**distribution_change(a,b)) for h,a,b in zip(panel,old_q,original_q)]
 maximum=max(x['tv'] for x in original_changes)
 root=Path(evidence_dir) if evidence_dir is not None else None
 if root is not None:
  root.mkdir(parents=True,exist_ok=False)
  np.savez_compressed(root/'old_and_original_full_probabilities.npz',old=old_q,original=original_q)
  np.save(root/'shared_real_gradient.npy',original_info['gradient_vector'],allow_pickle=False)
 # Audit every original KL-backtracking candidate, including rejected ones.
 # These are detached CPU reconstructions of the same one step, never PPO forks.
 original_candidate_audit=[]
 if original_info['step_backtracking']:
  audit=clone(meta8,decision['teacher']);base_state=audit.state()
  base_lrs=[x['lr'] for x in base_state['optimizer']['param_groups']]
  for index,record in enumerate(original_info['step_backtracking']):
   audit.load(copy.deepcopy(base_state));audit.optimizer.zero_grad(set_to_none=True);cursor=0
   for parameter in audit.net.parameters():
    size=parameter.numel();parameter.grad=torch.as_tensor(original_info['gradient_vector'][cursor:cursor+size].copy(),dtype=parameter.dtype).reshape(parameter.shape);cursor+=size
   torch.nn.utils.clip_grad_norm_(audit.net.parameters(),audit.config['gradient_clip'])
   for group,base_lr in zip(audit.optimizer.param_groups,base_lrs):group['lr']=base_lr*record['learning_rate_multiplier']
   audit.optimizer.step()
   if not all(bool(torch.isfinite(x).all()) for x in audit.net.parameters()):
    original_candidate_audit.append(dict(index=index,finite=False));continue
   q=panel_probabilities(audit,panel)
   with torch.no_grad():mu=audit.mean(decision['global_state']).numpy()
   kl=float(np.sum((mu-decision['mean'])**2)/(2*decision['sigma']**2))
   assert math.isclose(kl,record['preference_kl'],rel_tol=1e-12,abs_tol=1e-12)
   changes=[dict(id=h['id'],**distribution_change(a,b)) for h,a,b in zip(panel,old_q,q)]
   original_candidate_audit.append(dict(index=index,finite=True,preference_kl=kl,changes=changes,
    learning_rate_multiplier=record['learning_rate_multiplier'],moments_sha256=tree_hash(audit.optimizer.state_dict()['state'])))
   if root is not None:np.savez_compressed(root/f'original_candidate_{index:02d}_full_probabilities.npz',probability=q)
  if not original_info['step_rejected']:
   assert state_hash(audit.net.state_dict())==state_hash(original.net.state_dict())
 def summarize(attempts,intervention,tr,final_q,accepted):
  changes=[dict(id=h['id'],**distribution_change(a,b)) for h,a,b in zip(panel,old_q,final_q)]
  if root is not None:np.savez_compressed(root/'accepted_full_probabilities.npz',old=old_q,original=original_q,sampling_tr=final_q)
  original_optimizer=original.optimizer.state_dict();tr_optimizer=tr.optimizer.state_dict()
  return dict(tv_limit=tv_limit,mixed_uniform_mass=.2,reference_input_ids=[h['id'] for h in panel],
   original_max_tv=maximum,original_changes=original_changes,original_step=original_info,original_candidate_audit=original_candidate_audit,
   adam_snapshot_isolation_fixed=True,old_adam_steps=old_adam_steps,new_adam_steps=new_adam_steps,
   constraint_intervention=intervention,original_sampling_tr_network_identical=state_hash(original.net.state_dict())==state_hash(tr.net.state_dict()),
   original_sampling_tr_full_state_identical=tree_hash(original.state())==tree_hash(tr.state()),
   accepted=accepted,attempts=attempts,accepted_changes=changes,max_accepted_tv=max(x['tv'] for x in changes),
   accepted_learning_rate_multiplier=next((a['learning_rate_multiplier'] for a in reversed(attempts) if a.get('accepted')),original_info['accepted_learning_rate_multiplier'] if not intervention else 0.),
   adam_moments_equal_to_original_single_step=(tree_hash(original_optimizer['state'])==tree_hash(tr_optimizer['state'])) if accepted else None,
   cumulative_drift_reference='Old Meta8; one new update only',
   existing_protection='Squared mean-preference change / (2*sigma^2), at the decision global state; equal-covariance Gaussian KL. Not Student PPO desired_kl.',
   candidate_direction='Same pre-clipping gradient and same Adam state; no reward or sigma change; no PPO per backtrack')
 if maximum<=tv_limit:
  tr=clone(original)
  result=summarize([],False,tr,original_q,not original_info['step_rejected'] and not original_info['zero_signal'])
  assert result['original_sampling_tr_full_state_identical']
  return old,original,tr,result

 tr=clone(meta8,decision['teacher']);before=tr.state()
 before_network=copy.deepcopy(tr.net.state_dict());before_optimizer=copy.deepcopy(tr.optimizer.state_dict())
 parameters=list(tr.net.parameters());gradient=np.asarray(original_info['gradient_vector']);cursor=0
 tr.optimizer.zero_grad(set_to_none=True)
 for parameter in parameters:
  size=parameter.numel();parameter.grad=torch.as_tensor(gradient[cursor:cursor+size].copy(),dtype=parameter.dtype).reshape(parameter.shape)
  cursor+=size
 assert cursor==len(gradient)
 norm=float(torch.nn.utils.clip_grad_norm_(parameters,tr.config['gradient_clip']))
 assert math.isclose(norm,original_info['gradient_norm_before_clip'],rel_tol=1e-12,abs_tol=1e-12)
 base_lrs=[group['lr'] for group in before_optimizer['param_groups']]
 attempts=[];accepted=False;final_q=old_q
 sigma=float(decision['sigma'])
 for backtrack in range(tr.config['max_step_backtracks']+1):
  tr.net.load_state_dict(before_network);tr.optimizer.load_state_dict(copy.deepcopy(before_optimizer))
  multiplier=2.**(-backtrack)
  for group,base_lr in zip(tr.optimizer.param_groups,base_lrs):group['lr']=base_lr*multiplier
  tr.optimizer.step()
  finite=all(bool(torch.isfinite(p).all()) for p in parameters)
  candidate_kl=None;changes=[];maximum_tv=None
  if finite:
   with torch.no_grad():mu=tr.mean(decision['global_state']).numpy()
   candidate_kl=float(np.sum((mu-decision['mean'])**2)/(2*sigma*sigma))
   candidate_q=panel_probabilities(tr,panel)
   changes=[dict(id=h['id'],**distribution_change(a,b)) for h,a,b in zip(panel,old_q,candidate_q)]
   maximum_tv=max(x['tv'] for x in changes)
   if root is not None:np.savez_compressed(root/f'candidate_{backtrack:02d}_full_probabilities.npz',probability=candidate_q)
  okay=finite and math.isfinite(candidate_kl) and candidate_kl<=tr.config['max_preference_step_kl'] and maximum_tv<=tv_limit
  attempts.append(dict(backtrack=backtrack,learning_rate_multiplier=multiplier,attempted_learning_rates=[lr*multiplier for lr in base_lrs],
   finite=finite,preference_kl=candidate_kl,max_reference_tv=maximum_tv,changes=changes,accepted=bool(okay),
   starting_optimizer_sha256=tree_hash(before_optimizer),candidate_optimizer_sha256=tree_hash(tr.optimizer.state_dict()),
   moments_sha256=tree_hash(tr.optimizer.state_dict()['state'])))
  if okay:
   accepted=True;final_q=candidate_q
   for group,base_lr in zip(tr.optimizer.param_groups,base_lrs):group['lr']=base_lr
   break
 if not accepted:
  tr.load(before)
  assert tree_hash(tr.net.state_dict())==tree_hash(before_network) and tree_hash(tr.optimizer.state_dict())==tree_hash(before_optimizer)
 else:
  assert tree_hash(tr.optimizer.state_dict()['state'])==tree_hash(original.optimizer.state_dict()['state']), 'Adam backtracks accumulated steps or changed the shared direction'
 assert tree_hash(tr.encoder.state_dict())==tree_hash(before['encoder'])
 assert tree_hash(tr.rng.bit_generator.state)==tree_hash(before['rng'])
 tr.updates+=1;tr.used_versions.add(str(decision['version']))
 result=summarize(attempts,True,tr,final_q,accepted)
 assert result['max_accepted_tv']<=tv_limit
 return old,original,tr,result
