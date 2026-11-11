"""Per-stage integrity and strictly separate implementation-fix comparison."""
import argparse,json,copy
from pathlib import Path
import torch
from common import ROOT,write,file_sha
from cpu_hash import tree_hash
BASE=Path('/home/alex/dog/go2_piper_nsgs_v02_seed48_20261107')
def read(p):return json.loads(p.read_text())
def normalize(x):
 if isinstance(x,dict):return {k:normalize(v) for k,v in x.items() if k!='log_dir'}
 if isinstance(x,list):return [normalize(v) for v in x]
 if isinstance(x,str):
  for root in (str(ROOT),str(BASE)):x=x.replace(root,'<EXPERIMENT>')
 return x

def teacher():
 fresh=torch.load(ROOT/'teachers/Fresh.pt',weights_only=False,map_location='cpu')['teacher']
 final=torch.load(ROOT/'teachers/CleanMeta8.pt',weights_only=False,map_location='cpu')['teacher']
 assert final['updates']==8 and set(final['used_versions'])==set(map(str,range(8)))
 accepted=0;rows=[]
 for index in range(8):
  p=ROOT/'experience/meta-eight-starts'/f'point_{index:02d}';a=read(p/'adam_acceptance.json');assert a['passed'];accepted+=int(a['accepted'])
  assert a['adam_steps_after']==([accepted] if accepted else [])
  assert file_sha(p/'adam_full_trace.pt')==a['full_trace_sha256'];rows.append(a)
 assert all(int(v['step'])==accepted for v in final['optimizer']['state'].values())
 for k in ('config','binding','encoder'):assert tree_hash(final[k])==tree_hash(fresh[k])
 # Existing parent models are retained and cannot be rewritten as a "clean" fix.
 assert file_sha(ROOT/'teachers/OldMeta8.pt')=='8929d0a13c4aaa7b98c70acf584d04390711bbc9ea6df8b61f3bcb556cdbd3b9'
 write(ROOT/'CLEANMETA8_ACCEPTANCE.json',dict(passed=True,logical_updates=8,accepted_optimizer_updates=accepted,
  checkpoint_sha256=file_sha(ROOT/'teachers/CleanMeta8.pt'),teacher_state_sha256=tree_hash(final),updates=rows,
  no_tv_constraint=True,original_config_unchanged=True,all_new_real_branch_feedback=True))
def student(arm):
 out=ROOT/'runs'/('transfer-seed48-'+arm);r=read(out/'result.json')
 assert r['passed'] and r['new_updates']==10000 and r['new_environment_transitions']==983040000 and r['teacher_frozen']
 assert file_sha(Path(r['checkpoint']))==r['checkpoint_sha256']
 assert read(out/'initialization.json')==read(ROOT/'runs/transfer-seed48-old-meta8/initialization.json')
 cfg=read(out/'configuration.json');ref=read(ROOT/'runs/transfer-seed48-old-meta8/configuration.json')
 for k in ('agent','native_environment','protocol','binding','student_seed','num_envs','rollout'):
  assert normalize(cfg[k])==normalize(ref[k]),k
 ev=[read(p) for p in sorted((ROOT/'evaluations').glob('transfer-seed48-'+arm+'-*-test/result.json'))]
 assert [x['update'] for x in ev]==list(range(1000,10001,1000))
 write(ROOT/('ACCEPTANCE_'+arm+'.json'),dict(passed=True,curve=[{k:x[k] for k in ('update','joint_success_rate','continuous_joint_score','pooled_rmse','safety')} for x in ev]))
def uniform():
 old=BASE/'runs/transfer-seed48-uniform';new=ROOT/'runs/transfer-seed48-old-meta8';a,b=read(old/'configuration.json'),read(new/'configuration.json');checks={}
 for k in ('agent','native_environment','protocol','binding','student_seed','num_envs','rollout'):checks[k]=normalize(a[k])==normalize(b[k])
 checks['initialization']=read(old/'initialization.json')==read(new/'initialization.json')
 r=read(old/'result.json');checks['complete']=r['passed'] and r['new_updates']==10000 and r['teacher_frozen']
 checks['checkpoint_sha']=file_sha(Path(r['checkpoint']))==r['checkpoint_sha256']
 oa=read(BASE/'evaluation_identity_test.json');na=read(ROOT/'evaluation_identity_test.json');checks['evaluation_identity']=oa==na
 write(ROOT/'UNIFORM_REUSE.json',dict(matched=all(checks.values()),checks=checks,source=str(old),label='Original Uniform; unaffected by Teacher Adam bug'))
def report():
 u=read(ROOT/'UNIFORM_REUSE.json');table={arm:read(ROOT/('ACCEPTANCE_'+arm+'.json'))['curve'] for arm in ('old-meta8','clean-meta8')}
 ur=BASE if u['matched'] else ROOT
 vals=[read(p) for p in sorted((ur/'evaluations').glob('transfer-seed48-uniform-*-test/result.json'))]
 assert [x['update'] for x in vals]==list(range(1000,10001,1000))
 table['uniform']=[{k:x[k] for k in ('update','joint_success_rate','continuous_joint_score','pooled_rmse','safety')} for x in vals]
 write(ROOT/'COMPARISON.json',dict(seed=48,curves=table,uniform_reuse=u,interpretation='Correct Adam implementation versus legacy implementation, NOT a new algorithm; single student seed; PhysX not bit deterministic.'))
 jobs=[read(p) for p in (ROOT/'jobs').glob('*/result.json')]
 write(ROOT/'COST.json',dict(meta_student_updates=2400,transfer_student_updates=20000+(0 if u['matched'] else 10000),
  summed_job_wall_seconds=sum(x['wall_seconds'] for x in jobs),note='Summed jobs include nested transfer/evaluation overlap; not total GPU hours. Per-job launch/end and GPU telemetry retained.'))
 lines=['# CleanMeta8：正确实现复现与学生匹配比较','','仅比较 Adam 修复；未添加 TV 或 Meta24。','','| Update | Uniform | OldMeta8（旧实现） | CleanMeta8 |','|---:|---:|---:|---:|']
 for i in range(10):lines.append('| %d | %.2f%% | %.2f%% | %.2f%% |'%((i+1)*1000,100*table['uniform'][i]['joint_success_rate'],100*table['old-meta8'][i]['joint_success_rate'],100*table['clean-meta8'][i]['joint_success_rate']))
 lines+=['','同机seed48、相同初始化与原生PPO配置；不能将单种子结果解释为统计显著算法提升。完整物理误差、安全指标和逐点结果见COMPARISON.json。']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['teacher','student','uniform','report']);p.add_argument('--arm');a=p.parse_args();student(a.arm) if a.action=='student' else globals()[a.action]()

if __name__=="__main__" and a.action=="teacher":
 import runpy
 runpy.run_path(str(ROOT/"revision02/handoff.py"),run_name="__main__")
