"""Verify immutable sources and package lightweight evidence; never runs PPO."""
from pathlib import Path
import json,hashlib,shutil,zipfile,subprocess
import numpy as np
import torch
from common import ROOT,write,file_sha
from cpu_hash import tree_hash
from experiment_spec import META8_SHA

assert json.loads((ROOT/'STATUS.json').read_text())['phase']=='no_intervention_no_12k'
result=json.loads((ROOT/'RESULTS.json').read_text());assert result['passed'] and result['formal_validation_updates']==0 and result['all_new_student_updates']==302
candidate=json.loads((ROOT/'CANDIDATE_RESULT.json').read_text());assert not candidate['launch_12k']
teachers={k:torch.load(ROOT/'teachers'/(k+'.pt'),map_location='cpu',weights_only=False)['teacher'] for k in ('old','original','sampling-tr')}
assert tree_hash(teachers['original'])==tree_hash(teachers['sampling-tr'])
assert tree_hash(teachers['old']['network'])!=tree_hash(teachers['original']['network'])
assert len({tree_hash(t['rng']) for t in teachers.values()})==1
canonical=torch.load(ROOT/'teachers/meta8.pt',map_location='cpu',weights_only=False)['teacher']
assert file_sha(ROOT/'teachers/meta8.pt')==META8_SHA
for field in ('network','optimizer','encoder'):assert tree_hash(canonical[field])==tree_hash(teachers['old'][field])
old_steps={str(k):int(v['step']) for k,v in teachers['old']['optimizer']['state'].items()}
new_steps={str(k):int(v['step']) for k,v in teachers['original']['optimizer']['state'].items()}
assert all(new_steps[k]==v+1 for k,v in old_steps.items())
sources=json.loads((ROOT/'SOURCE_RECEIPTS.json').read_text())
for s in sources:
 assert file_sha(s['checkpoint'])==s['source_sha256']
 assert file_sha(s['prepared_checkpoint'])==s['prepared_sha256']
verify=dict(passed=True,all_original_student_checkpoints_unchanged=True,prepared_full_student_states_hashes_match=True,
 original_sampling_tr_complete_teacher_state_equal=True,old_original_networks_differ=True,
 canonical_meta8_network_adam_encoder_unchanged=True,all_controls_share_recorded_private_rng=True,
 old_adam_steps=old_steps,new_adam_steps=new_steps,adam_step_increments_exactly_one=True,
 formal_500_update_runs=0,native_upstream_commit='4e12109d1eccb1d9da2e4a35f189fabea3f13677')
write(ROOT/'FINAL_VERIFICATION.json',verify)

# Owned outputs only; shared assets/pools/dependency trees are never scanned.
owned=[]
for name in ('code','tests','runs','experience','starts','panels','teachers','candidates','jobs','evaluations','attempts','figures'):
 for p in sorted((ROOT/name).rglob('*')):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
   owned.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=file_sha(p)))
write(ROOT/'ARTIFACT_MANIFEST.json',owned)
delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True);stage=delivery/'TR0_report_and_evidence';stage.mkdir(exist_ok=False)
def copy(p,relative):
 if p.is_file():
  dest=stage/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
for name in ('REPORT.md','RESULTS.json','STATUS.json','CANDIDATE_RESULT.json','PREREGISTRATION.json','SOURCE_RECEIPTS.json',
 'TR0_GIT_RECEIPT.json','CODE_BINDING.json','CPU_ACCEPTANCE.json','GPU_SMOKE_ACCEPTANCE.json','RESOURCE_ACCEPTANCE.json',
 'FINAL_VERIFICATION.json','ARTIFACT_MANIFEST.json','BASELINE_RESTORE_NOISE.json','cpu_tests_final.log','cpu_tests_rng_cursor.log'):
 copy(ROOT/name,name)
for name in ('ADAM_SNAPSHOT_DIAGNOSTIC.json','ADAM_FIX_ACCEPTANCE.json','cpu_tests_adam_snapshot.log'):copy(ROOT/name,name)
copy(ROOT/'panels/fixed13.json','fixed_input_identities.json')
for p in (ROOT/'figures').glob('*'):copy(p,Path('figures')/p.name)
for p in (ROOT/'code').glob('*.py'):
 if p.name in ('trust_region.py','experiment_spec.py','report_tr0.py','plot_tr0.py','prepare_tr0.py','make_candidates.py','finalize_tr0.py','build_tr0_trainer.py'):
  copy(p,Path('implementation')/p.name)
for p in (ROOT/'tests').glob('test_*tr*.py'):copy(p,Path('tests')/p.name)
copy(ROOT/'tests/test_progress_row.py','tests/test_progress_row.py')
for p in (ROOT/'runs').glob('*/*'):
 if p.is_file() and p.name in ('updates.jsonl','sampling_cycles.jsonl','result.json','restore_receipt.json','configuration.json','native_comparison.json','initialization.json','starting_student_state.json'):
  copy(p,Path('training')/p.relative_to(ROOT/'runs'))
for p in (ROOT/'evaluations').glob('*/*.json'):
 if p.name in ('result.json','episode_results.json'):copy(p,Path('development')/p.relative_to(ROOT/'evaluations'))
for name in ('decision.json','engineering.json'):
 copy(ROOT/'experience/meta-eight-starts/point_00'/name,Path('fresh_direction')/name)

# All distinct candidate probability arrays, exact float64. B/C and the accepted
# second KL candidate are aliases of one matrix, not duplicated or quantized.
with np.load(ROOT/'candidates/new_real/accepted_full_probabilities.npz',allow_pickle=False) as a:
 old=a['old'].copy();accepted=a['original'].copy();assert np.array_equal(accepted,a['sampling_tr'])
with np.load(ROOT/'candidates/new_real/original_candidate_00_full_probabilities.npz',allow_pickle=False) as a:raw=a['probability'].copy()
with np.load(ROOT/'candidates/new_real/original_candidate_01_full_probabilities.npz',allow_pickle=False) as a:assert np.array_equal(a['probability'],accepted)
assert old.shape==accepted.shape==raw.shape==(13,32768) and old.dtype==accepted.dtype==raw.dtype==np.float64
np.testing.assert_allclose(np.abs(accepted-old).sum(1)/2,[x['tv'] for x in candidate['original_changes']],rtol=0,atol=1e-15)
probability_all=stage/'probability_all_distinct.npz'
np.savez_compressed(probability_all,old_meta8=old,original_and_sampling_tr=accepted,
 raw_lr001_rejected_by_existing_kl=raw,reference_input_ids=np.array(candidate['reference_input_ids']),sequence_ids=np.arange(32768,dtype=np.int32))
contents=dict(exact_probability_dtype='float64',full_sequences=32768,frozen_inputs=13,
 sampling_tr_probability_alias='original_and_sampling_tr',accepted_lr005_probability_alias='original_and_sampling_tr',
 includes_all_distinct_candidate_distributions=True,includes_per_reset_development_metrics=True,
 complete_raw_inputs_and_teacher_student_checkpoints='Retained on execution host; paths and SHA256 in ARTIFACT_MANIFEST.json',
 no_formal_500_update_results=True)
write(stage/'CONTENTS.json',contents)
(stage/'READ_ME.txt').write_text('本包是TR0实际结果与证据。新增2% TV没有改变既有KL接受步，B/C完整Teacher状态一致，因此未启动12k。\nfull_sequence_probability_evidence.npz保留13份固定H×32768序列的旧参数和最终接受概率，原float64、不量化；B/C及已接受的第二候选共用同一数组。原始全步候选和逐回合明细是否装入见CONTENTS.json，未装入项给出服务器路径。\n完整模型、完整H快照和原始逐步训练反馈保留服务器，文件路径/哈希见ARTIFACT_MANIFEST.json。\n')
archive=delivery/'nSGS_TR0_report_and_evidence.zip'
include_episodes=True
probability_source=probability_all
def pack():
 selected=[]
 for p in sorted(stage.rglob('*')):
  if not p.is_file() or p==stage/'MANIFEST.json':continue
  if p.name.startswith('probability_') and p!=probability_source:continue
  if not include_episodes and p.name=='episode_results.json' and stage/'development' in p.parents:continue
  relative='full_sequence_probability_evidence.npz' if p==probability_source else str(p.relative_to(stage))
  selected.append((p,relative))
 manifest=[dict(path=rel,bytes=p.stat().st_size,sha256=file_sha(p)) for p,rel in selected]
 write(stage/'MANIFEST.json',manifest)
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p,rel in selected:z.write(p,'TR0_report_and_evidence/'+rel)
  z.write(stage/'MANIFEST.json','TR0_report_and_evidence/MANIFEST.json')
pack()
if archive.stat().st_size>=10_000_000:
 include_episodes=False
 contents['includes_per_reset_development_metrics']=False;contents['per_reset_evidence_on_host']=True;write(stage/'CONTENTS.json',contents);pack()
if archive.stat().st_size>=10_000_000:
 probability_source=stage/'probability_old_and_accepted.npz'
 np.savez_compressed(probability_source,old_meta8=old,original_and_sampling_tr=accepted,
  reference_input_ids=np.array(candidate['reference_input_ids']),sequence_ids=np.arange(32768,dtype=np.int32))
 contents['includes_all_distinct_candidate_distributions']=False;contents['raw_lr001_distribution_on_host']=str(ROOT/'candidates/new_real/original_candidate_00_full_probabilities.npz');write(stage/'CONTENTS.json',contents);pack()
assert archive.stat().st_size<10_000_000,archive.stat().st_size
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
receipt=dict(passed=True,path=str(archive),bytes=archive.stat().st_size,sha256=file_sha(archive),contents=contents,final_verification=verify)
write(ROOT/'DELIVERY_RECEIPT.json',receipt)
print('TR0_DELIVERY_READY '+json.dumps({k:receipt[k] for k in ('path','bytes','sha256')}),flush=True)
