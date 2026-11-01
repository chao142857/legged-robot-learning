
import pathlib,sys,os,json,hashlib,time,fcntl,traceback,torch,numpy as np
p=pathlib.Path('/data/go2_piper_wbc_stage3b_dynamic_x_20261101');sys.path.insert(0,str(p/'active'))
from gpu_budget import Budget,read,write
from stage3_pipeline import wait
b=Budget(p);torch.set_num_threads(1);assert not torch.cuda.is_initialized()
write(p/'REPAIR3V2_PROCESS.json',dict(pid=os.getpid(),scope='Two frozen single-cell causal probes; never candidate policy/training'))
try:
 with(p/'learning.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not b.meter()['active'] and not(p/'FROZEN_SELECTION.json').exists()
  assert read(p/'REPAIR2_DONE.json')['hypothesis_supported']==False
  nodes=read(p/'CANDIDATE_NODES.json');bad=next(n for n in nodes if n['label']=='legKL002-n400-seed22');native=pathlib.Path(read(p/'EVAL_INDEX.json')['legKL002-n400-seed22-A-6181031']);cfg=read(native/'config.json');original=pathlib.Path(bad['checkpoint']['checkpoint']);zero=p/'diagnostic_only'/'HEAD_ZERO_ACTOR_ONLY_V2_NOT_FOR_TRAINING.pt';zero.parent.mkdir(exist_ok=True);assert not zero.exists()
  loaded=torch.load(original,map_location='cpu',weights_only=False);cp=dict(actor_state_dict=loaded['actor_state_dict'],infos=dict(stage3_actor_recipe=loaded['infos']['stage3_actor_recipe']));del loaded;assert hashlib.sha256(original.read_bytes()).hexdigest()==bad['checkpoint']['checkpoint_sha256'];cp['actor_state_dict']['synergy_head.weight']=torch.zeros_like(cp['actor_state_dict']['synergy_head.weight']);cp['actor_state_dict']['synergy_head.bias']=torch.zeros_like(cp['actor_state_dict']['synergy_head.bias']);cp['infos']['diagnostic_only_never_train']=True;cp['infos']['diagnostic_origin_checkpoint_sha256']=bad['checkpoint']['checkpoint_sha256'];torch.save(cp,zero);del cp
  import zipfile
  with zipfile.ZipFile(zero) as archive:
   key=next(n for n in archive.namelist() if n.endswith('/data.pkl'));assert b'numpy' not in archive.read(key),'Actor-only diagnostics cannot retain NumPy RNG pickle objects'
  parent=read(p/'CHECKPOINTS.json')['parent'];plan=dict(current=bad['checkpoint'],native_control_run=str(native),head_zero_checkpoint_sha256=hashlib.sha256(zero.read_bytes()).hexdigest(),modes=['head_zero','parent_all'],prefix_s=1.,blend_s=.25,training_updates=0,scope='head_zero: only rank1arm residual disabled, all current MLP functions retained. parent_all: original CP1800 full function. Diagnostic composition, not learned policy success.')
  write(p/'REPAIR3V2_PLAN_FROZEN.json',plan);runs=[]
  for mode in plan['modes']:
   diagnostic=dict(checkpoint=str(zero),checkpoint_sha256=plan['head_zero_checkpoint_sha256']) if mode=='head_zero' else parent
   config=dict(cfg,mode='probe',entry='component_probe_entry.py',phase='P2',resume=False,diagnostic_parent_checkpoint=diagnostic['checkpoint'],diagnostic_parent_sha256=diagnostic['checkpoint_sha256'],component_probe=dict(mode='parent_arm' if mode=='head_zero' else 'parent_all',start_s=1.,blend_s=.25,first_known_failure_s=18.7))
   run=b.launch(config,'repair3v2-seed22-'+mode,timeout=300);write(p/('REPAIR3V2_'+mode+'_RUN.json'),dict(run=str(run)));wait(b,run);s=read(run/'evaluation/summary.json');d=read(run/'evaluation/dynamic_metrics.json');row=dict(mode=mode,run=str(run),strict=s['safe_count'],complete=s['complete_count'],hard=s['joint_limit_violation_count'],nonfoot=s['nonfoot_contact_count'],EE_after1s_m=d['windows']['after1s']['EE_absolute_rmse_m']);runs.append(row);write(p/'REPAIR3V2_STATE.json',dict(rows=runs,clock=b.meter()))
   # Do not continue the same dangerous diagnostic cell sequence after gross failures.
   if row['hard']+row['nonfoot']>=4:break
  baseline=pathlib.Path(read(p/'REPAIR2_current_RUN.json')['run']);a=np.load(baseline/'evaluation/trajectories.npz',allow_pickle=False);checks=[]
  for row in runs:
   run=pathlib.Path(row['run']);z=np.load(run/'evaluation/trajectories.npz',allow_pickle=False);fields=['diag_actor_obs','component_current_mean','control_processed_target','control_joint_position','control_joint_velocity','ee_cmd','base_cmd','kin_link0_pos','kin_link0_quat','component_contacts_history_w'];prefix={k:bool(np.array_equal(a[k][:49],z[k][:49])) for k in fields};realized=read(baseline/'evaluation/component_realized_start.json')==read(run/'evaluation/component_realized_start.json');checks.append(dict(mode=row['mode'],equal_prefix=prefix,equal_realization=realized,removes_contact=bool(row['hard']==row['nonfoot']==0 and row['complete']==64 and row['strict']>=61 and row['EE_after1s_m']<=read(p/'REPAIR2_DONE.json')['rows'][0]['EE_after1s_m']+.01 and realized and all(prefix.values()))));z.close()
  a.close();write(p/'REPAIR3V2_DONE.json',dict(rows=runs,checks=checks,diagnostic_only_not_learned_policy=True,clock=b.meter()));assert not torch.cuda.is_initialized()
except BaseException as error:write(p/'REPAIR3V2_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),clock=b.meter()));raise
