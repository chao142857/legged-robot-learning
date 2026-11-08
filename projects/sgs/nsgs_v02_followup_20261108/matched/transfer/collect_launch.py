from pathlib import Path
import json,hashlib,subprocess,time
r=Path('/home/alex/dog/go2_piper_nsgs_v02_matched_seed48_20261108')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(r/'source_manifest.json');assert all(sha(r/p)==v for p,v in manifest['files'].items())
smoke=read(r/'GPU_SMOKE_ACCEPTANCE.json');assert smoke['passed'] and smoke['resume_verified']
inherit=read(r/'teachers/inheritance.json');assert inherit['passed'] and inherit['old_updates']==8
source=read(r/'starts/seed43-cp01000/import_acceptance.json');assert source['passed']
p=r/'runs/p00-seed43-cp01000-uniform/updates.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()];assert rows and rows[0]['completed_updates']==1001 and rows[0]['new_updates']==1
restore=read(r/'runs/p00-seed43-cp01000-uniform/restore_receipt.json');assert restore['passed']
log=(r/'jobs/cpu-tests/console.log').read_text();assert 'Ran 45 tests' in log and log.rstrip().endswith('OK')
session=subprocess.check_output(['bash',str(r/'tmux.sh'),'display-message','-p','-t','nsgs-matched-seed48','#{pane_pid} #{session_created}'],text=True).split()
result=dict(passed=True,collected_unix=time.time(),host='ubuntu-robot',gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip(),tmux_session='nsgs-matched-seed48',pane_pid=int(session[0]),session_created_unix=int(session[1]),initial_results_published_commit='41a6be8',branch='research/nsgs-v02-matched-seed48-20261108',source_manifest_sha256=sha(r/'source_manifest.json'),all_runtime_source_hashes_verified=True,cpu_tests=45,gpu_smoke=smoke,teacher_inheritance=inherit,first_source=source,first_real_branch_update=rows[0],first_real_branch_latest=rows[-1],first_restore=restore,status=read(r/'STATUS.json'),logical_root=str(r),data_root='/media/alex/Sharedata/go2_piper_nsgs_v02_matched_seed48_20261108',planned_new_counterfactual_ppo_updates=2400,planned_frozen_teacher_student_seed=48,planned_fresh_student_ppo_updates=10000,full_queue_automatic=True,raw_old_results_preserved=True,comparison_baselines_sha256=sha(r/'comparison_baselines.json'),new_student_initialization_reference=read(r/'runs/student_seed48_initialization.json'))
progress=[]
for path in sorted((r/'runs').glob('p*-seed*-cp*-*/updates.jsonl')):
    items=[json.loads(line) for line in path.read_text().splitlines()]
    if items:progress.append(dict(label=path.parent.name,new_updates=items[-1]['new_updates']))
result['counterfactual_progress']=progress
result['completed_new_counterfactual_updates']=sum(x['new_updates'] for x in progress)
points=[]
for path in sorted((r/'experience/meta-eight-starts').glob('point_*/update.json')):
    response=read(path);points.append(dict(point=path.parent.name,version=response['version'],teacher_checkpoint_sha256=response['teacher_checkpoint_sha256']))
result['new_teacher_updates']=points
transfer=r/'PEER_TRANSFER_LAUNCH.json'
if transfer.exists():
    pid=read(transfer)['pid'];proc=Path('/proc')/str(pid)/'status'
    result['peer_receiver_parent_pid']=int(proc.read_text().split('PPid:')[1].splitlines()[0].strip()) if proc.exists() else None
for name in ('IMPORT_COMPLETE.json','PEER_TRANSFER_LAUNCH.json','PEER_TRANSFER_COMPLETE.json'):
 if (r/name).exists():result[name]=read(r/name)
for name in ('FAULT.txt','IMPORT_FAULT.txt','PEER_TRANSFER_FAULT.txt'):assert not (r/name).exists(),name
(r/'LAUNCH_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
