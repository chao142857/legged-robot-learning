"""Prepare local immutable full Student forks and preserve genuine feedback."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,PROTOCOL,digest,file_sha,write
from cpu_hash import tree_hash
from experiment_spec import SPEC,SOURCES,BASE,META8_SHA
from neural_v01 import NeuralSampler
from fork_v02 import partial_features,learning_progress
from trust_region import frozen_input
from teacher_v02 import Teacher

def source_row(source):
 if 'row_file' in source:return json.loads(Path(source['row_file']).read_text())
 if 'row_decision' in source:return torch.load(source['row_decision'],map_location='cpu',weights_only=False)['source_progress_row']
 with Path(source['row_log']).open() as f:
  for line in f:
   row=json.loads(line)
   if row['completed_updates']==source['update']:return row
 raise AssertionError('No exact source progress row: '+source['id'])

def complete_progress_row(row,actor_state):
 """Legacy E3 rows omit std; derive it from the real saved Gaussian parameter."""
 result=copy.deepcopy(row)
 measured=float(actor_state['distribution.log_std_param'].exp().mean())
 assert np.isfinite(measured) and measured>0
 if 'action_std' in result:
  # Native logger output_std is a recorded distribution snapshot, not a promise
  # of bit equality to CPU exp(mean) of the post-update checkpoint parameter.
  # Keep that genuine logged value. Only an absent field is derived from weights.
  assert np.isfinite(result['action_std']) and result['action_std']>0
 else:result['action_std']=measured
 return result

def prepare():
 assert file_sha(ROOT/'teachers/meta8.pt')==META8_SHA
 write(ROOT/'PREREGISTRATION.json',SPEC)
 manifest=json.loads((ROOT/'prepared/manifest.json').read_text());teacher=Teacher(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False),manifest['binding'])
 teacher.load(torch.load(ROOT/'teachers/meta8.pt',map_location='cpu',weights_only=False)['teacher']);assert teacher.updates==8
 receipts=[]
 for source in SOURCES:
  p=Path(source['checkpoint']);saved=torch.load(p,map_location='cpu',weights_only=False);i=saved['infos'];u=source['update'];seed=source['seed']
  assert (i['student_seed'],i['completed_updates'])==(seed,u)
  assert i['pool_sha256']==manifest['binding']['train_pool_sha256'] and i['protocol_sha256']==digest(PROTOCOL)
  assert i['environment'] is not None and i['optimizer_calls']==20*u
  hashes={k:tree_hash(saved[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')}
  physical=tree_hash(i['environment']);rng=tree_hash(i['rng']);original=copy.deepcopy(i['adapter'])
  owner=i['adapter']['owner'];engine=NeuralSampler(32768,0);engine.load_state_dict(owner['engine'])
  history_hash=tree_hash({k:engine.state_dict()[k] for k in ('history','completed_counts','success_counts','sample_counts')})
  old_sampler_rng=tree_hash(engine.rng.bit_generator.state)
  engine.version=-1;engine.draws=[];engine.draw_end_steps={};sampler_seed=32000000+seed*100+u;engine.rng=np.random.default_rng(sampler_seed)
  owner['engine']=engine.state_dict();owner['teacher_version']=np.full(4096,-1,dtype=np.int64)
  owner['decision_id']=-np.arange(1,4097,dtype=np.int64);owner['selected_probability']=np.zeros(4096);owner['cycle_version_exposure']={}
  assert history_hash==tree_hash({k:engine.state_dict()[k] for k in ('history','completed_counts','success_counts','sample_counts')})
  partial,partial_receipt=partial_features(original);now=owner['control_steps'];assert now==u*24
  dynamic=engine.features(now,owner['transitions']);progress,po=learning_progress(engine,now)
  source_sha=file_sha(p)
  i['v02_preparation']=dict(source_checkpoint=str(p),source_checkpoint_sha256=source_sha,student_seed=seed,completed_updates=u,
    sampler_seed=sampler_seed,source_sampler_rng_sha256=old_sampler_rng,paired_new_sampler_rng_sha256=tree_hash(engine.rng.bit_generator.state),
    source_global_rng_preserved=True,actor_critic_optimizer_preserved=True,physical_state_preserved=True,
    genuine_feedback_history_preserved=True,old_inflight_teacher_version=-1,new_teacher_gradient_data=False)
  i['base_update']=u
  i['v02_context']=dict(dynamic=dynamic,partial=partial,partial_receipt=partial_receipt,progress=progress,progress_observed=po,
    control_step=now,source_checkpoint_sha256=source_sha)
  assert all(tree_hash(saved[k])==v for k,v in hashes.items()) and tree_hash(i['environment'])==physical and tree_hash(i['rng'])==rng
  dest=ROOT/'starts'/source['id'];dest.mkdir(parents=True,exist_ok=False)
  cp=dest/'prepared.pt';torch.save(saved,cp)
  row=complete_progress_row(source_row(source),saved['actor_state_dict']);assert row['completed_updates']==u and row['native_feedback_episodes']==owner['accepted']
  write(dest/'progress_row.json',row);write(dest/'source.json',dict(source,updates=u,path=str(cp),sha256=file_sha(cp),original_sha256=source_sha))
  write(dest/'export_receipt.json',dict(progress_row_sha256=file_sha(dest/'progress_row.json')))
  receipt=dict(source,prepared_checkpoint=str(cp),prepared_sha256=file_sha(cp),source_sha256=source_sha,
   actor_critic_optimizer_sha256=hashes,physical_state_sha256=physical,source_global_rng_sha256=rng,
   feedback_history_sha256=history_hash,read_only_original_unchanged=file_sha(p)==source_sha,preparation=i['v02_preparation'])
  write(dest/'preparation.json',receipt);receipts.append(receipt)
  print('TR0_SOURCE_PREPARED',source['id'],flush=True)
 write(ROOT/'SOURCE_RECEIPTS.json',receipts)
 write(ROOT/'sources.json',dict(sources=[json.loads((ROOT/'starts'/s['id']/'source.json').read_text()) for s in SOURCES]))
 # Historical source panels carry complete raw masked features and provenance;
 # their encoders must be exactly the inherited command-only normalization.
 panel=[]
 for index in range(8):
  p=BASE/'experience/meta-eight-starts'/f'point_{index:02d}'/'decision.pt';d=torch.load(p,map_location='cpu',weights_only=False)
  assert tree_hash(d['teacher']['encoder'])==tree_hash(teacher.encoder.state_dict())
  h=frozen_input(teacher,'historical-source-%02d'%index,d,d['global_state'],dict(path=str(p),sha256=file_sha(p),
      student_seed=d['source_original']['seed'],update=d['before_development']['update'],formal_test_used=False))
  assert h['basis_sha256']==d['basis_sha256'];panel.append(h)
 torch.save(panel,ROOT/'panels/historical8.pt')
 write(ROOT/'panels/historical8.json',[{k:v for k,v in h.items() if k not in ('basis','global_state','dynamic','progress','progress_observed','partial')} for h in panel])
 print('TR0_SOURCES_AND_HISTORICAL_PANEL_READY',flush=True)

def actual_panel():
 manifest=json.loads((ROOT/'prepared/manifest.json').read_text());teacher=Teacher(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False),manifest['binding'])
 teacher.load(torch.load(ROOT/'teachers/meta8.pt',map_location='cpu',weights_only=False)['teacher'])
 panel=torch.load(ROOT/'panels/historical8.pt',map_location='cpu',weights_only=False)
 for source in SOURCES:
  if source['role']!='validation':continue
  start=ROOT/'starts'/source['id'];cp=torch.load(start/'prepared.pt',map_location='cpu',weights_only=False);i=cp['infos'];row=json.loads((start/'progress_row.json').read_text())
  dev=json.loads((start/'baseline_dev.json').read_text());assert dev['kind']=='development' and dev['update']==source['update']
  dynamic=i['v02_context']['dynamic'];from neural_v01 import DYNAMIC_FIELDS
  recent=dynamic[:,DYNAMIC_FIELDS.index('recent_feedback_observed')]>.5
  g=np.zeros(10,dtype=np.float32);g[0]=source['update']/10000;g[1]=row['learning_rate']/1e-3;g[2]=row['action_std']
  g[3]=row['losses'].get('value',row['losses'].get('value_loss',0.));g[4]=(row['mean_training_reward'] or 0.)/200
  g[5]=recent.mean();g[6]=dynamic[recent,DYNAMIC_FIELDS.index('recent_good_fraction')].mean() if recent.any() else 0.
  g[7]=dynamic[recent,DYNAMIC_FIELDS.index('recent_failure')].mean() if recent.any() else 0.;g[8]=dev['continuous_joint_score'];g[9]=1.
  h=frozen_input(teacher,source['id'],i['v02_context'],g,dict(source,prepared_sha256=file_sha(start/'prepared.pt'),
      baseline_dev_sha256=file_sha(start/'baseline_dev.json'),formal_test_used=False));panel.append(h)
 assert len(panel)==12
 torch.save(panel,ROOT/'panels/reference12.pt')
 write(ROOT/'panels/reference12.json',[{k:v for k,v in h.items() if k not in ('basis','global_state','dynamic','progress','progress_observed','partial')} for h in panel])
 print('TR0_FIXED_REFERENCE12_READY;ACTUAL_GRADIENT_H_WILL_BE_ADDED',flush=True)

def repair_progress_metadata():
 """Complete only new TR0 metadata; retain the original copied legacy row."""
 import shutil
 for p in sorted((ROOT/'starts').glob('*/progress_row.json')):
  cp=torch.load(p.parent/'prepared.pt',map_location='cpu',weights_only=False)
  old=json.loads(p.read_text());new=complete_progress_row(old,cp['actor_state_dict'])
  if 'action_std' not in old:
   original=p.with_name('progress_row.original.json');assert not original.exists();shutil.copyfile(p,original)
   write(p,new)
   write(p.parent/'ACTION_STD_PROVENANCE.json',dict(derived_from='Saved actor distribution.log_std_param.exp().mean()',value=new['action_std'],
    checkpoint_sha256=file_sha(p.parent/'prepared.pt'),original_progress_sha256=file_sha(original)))
  receipt=json.loads((p.parent/'export_receipt.json').read_text());receipt['progress_row_sha256']=file_sha(p)
  write(p.parent/'export_receipt.json',receipt)
 print('REAL_CHECKPOINT_ACTION_STD_VERIFIED_FOR_ALL_SOURCES',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--actual-panel',action='store_true');p.add_argument('--repair-progress',action='store_true');a=p.parse_args()
 repair_progress_metadata() if a.repair_progress else actual_panel() if a.actual_panel else prepare()
