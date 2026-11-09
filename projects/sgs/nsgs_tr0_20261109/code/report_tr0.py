"""Predeclared endpoints, paired integrity checks, and non-double-counted costs."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,write
from cpu_hash import tree_hash
from experiment_spec import SOURCES

def read(p):return json.loads(Path(p).read_text())
def audit_pair(identifier,repeat):
 source=next(s for s in SOURCES if s['id']==identifier);rows=[];reference=None;sampler_rng=None;first_action=None
 for arm in ('old','original','sampling-tr'):
  label=identifier+'-r%d-'%repeat+arm;out=ROOT/'runs'/label
  result=read(out/'result.json');assert result['passed'] and result['new_updates']==500 and result['new_environment_transitions']==49152000
  start=read(out/'starting_student_state.json');receipt=read(out/'restore_receipt.json')
  assert receipt['passed'] and receipt['next_4096_sampler_draws_equal']
  if reference is None:reference=start;sampler_rng=receipt['installed_sampler_rng_sha256'];first_action=receipt['first_action_sha256']
  else:
   assert start==reference,'Different initial Student/optimizer/global RNG within paired repeat'
   assert sampler_rng==receipt['installed_sampler_rng_sha256'],'Different paired sampler RNG'
   assert first_action==receipt['first_action_sha256'],'Different very first Student action in paired repeat'
  teacher=torch.load(ROOT/'teachers'/(arm+'.pt'),map_location='cpu',weights_only=False)['teacher']
  assert result['teacher_frozen'] and result['teacher_state_sha256']==tree_hash(teacher)
  baseline=read(ROOT/'starts'/identifier/'baseline_dev.json');points=[]
  for offset in (100,300,500):
   e=read(ROOT/'evaluations'/(label+'-%05d-development'%(source['update']+offset))/'result.json')
   assert e['passed'] and e['kind']=='development' and e['training_history_updates']==0 and e['model_weights_unchanged']
   points.append(dict(offset=offset,score=e['continuous_joint_score'],gain=e['continuous_joint_score']-baseline['continuous_joint_score'],
     strict_success=e['joint_success_rate'],rmse=e['pooled_rmse'],native_failure_rate=e['native_failure_rate'],safety=e['safety'],
     scores=e['scores'],longest_joint_good_seconds_mean=e['longest_joint_good_seconds_mean']))
  rows.append(dict(source=identifier,seed=source['seed'],source_update=source['update'],repeat=repeat,teacher=arm,
   baseline=baseline['continuous_joint_score'],points=points,result=result,
   first_rollout_signatures=receipt['first_rollout_signatures'],baseline_restore_noise=read(out/'baseline_restore_noise.json')))
 result=dict(passed=True,source=identifier,repeat=repeat,starting_student_state=reference,paired_sampler_rng_sha256=sampler_rng,
  first_action_sha256=first_action,arms=rows,physx_hidden_caches_not_bitwise=True)
 write(ROOT/'pair_audits'/(identifier+'-r%d.json'%repeat),result);return result

def final_report():
 candidate=read(ROOT/'CANDIDATE_RESULT.json');pairs=[]
 if candidate['launch_12k']:
  for source in SOURCES[1:]:
   for repeat in range(2):pairs.append(audit_pair(source['id'],repeat))
 jobs=[read(p) for p in (ROOT/'jobs').glob('*/result.json')]
 trains=[read(p) for p in (ROOT/'runs').glob('*/result.json')]
 formal=[r for r in trains if r['new_updates']==500]
 all_eval=[read(p) for p in (ROOT/'evaluations').glob('*/result.json')]
 attempt_evaluations=[read(p) for p in (ROOT/'attempts').glob('*/evaluations/*/result.json')]
 all_eval+=attempt_evaluations
 attempt_jobs=[read(p) for p in (ROOT/'attempts').glob('*/jobs/*/result.json')]
 contrasts=[]
 for pair in pairs:
  a={r['teacher']:r for r in pair['arms']}
  contrasts.append(dict(source=pair['source'],repeat=pair['repeat'],
    gains={k:v['points'][-1]['gain'] for k,v in a.items()},
    sampling_tr_minus_old=a['sampling-tr']['points'][-1]['gain']-a['old']['points'][-1]['gain'],
    sampling_tr_minus_original=a['sampling-tr']['points'][-1]['gain']-a['original']['points'][-1]['gain'],
    original_minus_old=a['original']['points'][-1]['gain']-a['old']['points'][-1]['gain']))
 started=read(ROOT/'RESOURCE_ACCEPTANCE.json').get('unix')
 result=dict(passed=True,formal_validation_updates=sum(r['new_updates'] for r in formal),
  all_new_student_updates=sum(r['new_updates'] for r in trains),all_new_training_transitions=sum(r['new_environment_transitions'] for r in trains),
  fresh_candidate_updates=300,smoke_updates=2,validation_branch_count=len(formal),source_seeds=[44,46,47,48],
  evaluation_environment_transitions=sum(e['executed_environment_transitions'] for e in all_eval),
  evaluation_job_wall_seconds=sum(j['wall_seconds'] for j in jobs+attempt_jobs if j['label'].startswith('eval-')),
  failed_attempt_job_wall_seconds=sum(j['wall_seconds'] for j in attempt_jobs),
  archived_attempt_evaluations=len(attempt_evaluations),
  student_job_wall_excluding_wait_for_evaluations=sum(max(0,r['wall_seconds']-r['evaluation_seconds']) for r in trains),
  total_pipeline_wall_seconds=time.time()-started if started else None,
  cost_scope='Training updates exclude historical source updates. Student evaluation waiting is subtracted before adding separate eval jobs; bootstrap/startup is retained in job wall time. No claim of exact GPU-active time.',
  candidate={k:v for k,v in candidate.items() if k not in ('original_step','original_candidate_audit')},
  primary_endpoint='+500 development continuous joint-score gain; no best-checkpoint selection',paired_contrasts=contrasts,
  no_formal_test_inputs=True,random_seed_exploration_only=True,statistical_significance_claim=False)
 if candidate['launch_12k']:assert result['formal_validation_updates']==12000 and len(formal)==24 and len(contrasts)==8
 else:assert not formal and result['formal_validation_updates']==0
 write(ROOT/'RESULTS.json',result)
 lines=['# nSGS-TR0 实验结果','',
  '执行主机：ubuntu-robot / RTX5060 Laptop。Git记录见 TR0_GIT_RECEIPT.json。',
  '','## 实际候选与限幅','',
  'Original-step 在固定13份输入上的最大序列TV：%.6f%%。'%(100*candidate['original_max_tv']),
  '2%% TV限制是否改变更新：%s。'%candidate['constraint_intervention'],
  'Original-step 与 Sampling-TR 完整Teacher状态是否一致：%s。'%candidate['original_sampling_tr_full_state_identical'],
  '完整概率、参考输入身份、KL、概率比、Adam状态与回溯记录均保留在 candidates/ 与 panels/。','']
 if not candidate['launch_12k']:
  lines+=['没有启动12k验证。原因：'+str(candidate['stop_reason'])+'。',
   '若新限制未触发，B/C相同，这次没有形成更新幅度干预，不能据此判定信赖域有效或无效，也不能检验“大步危害”。',
   '四个验证源的完整起点及基线已保留；没有生成或冒充逐起点PPO对照结果。','']
 else:
  lines+=['## 逐起点与重复的 +500 主指标','',
   '|起点|重复|Old增量|Original增量|Sampling-TR增量|TR−Old|TR−Original|',
   '|---|---:|---:|---:|---:|---:|---:|']
  for x in contrasts:
   g=x['gains'];lines.append('|%s|%d|%.6f|%.6f|%.6f|%.6f|%.6f|'%(x['source'],x['repeat'],g['old'],g['original'],g['sampling-tr'],x['sampling_tr_minus_old'],x['sampling_tr_minus_original']))
  lines+=['','所有起点、重复和+100/+300/+500数据均在pair_audits/。四个来源的阶段与先前采样器不同，配对只在各固定起点内部成立。PhysX隐藏状态不保证逐位确定，不宣称统计显著性。','']
 lines+=['## 算力与证据','',
  '正式验证新增PPO更新：%d。全部新增学生更新（含方向采集和smoke）：%d。'%(result['formal_validation_updates'],result['all_new_student_updates']),
  '新增训练环境转移：%d。额外评测环境转移：%d。'%(result['all_new_training_transitions'],result['evaluation_environment_transitions']),
  '学生作业墙钟（扣除等待评测）：%.1fs；独立评测作业墙钟：%.1fs。'%(result['student_job_wall_excluding_wait_for_evaluations'],result['evaluation_job_wall_seconds']),
  '原有E1–E3、nSGS V0–V0.2和其他主机队列没有修改。PPO、奖励、观测、18维动作、命令池及严格5cm/80%定义保持原样。','']
 (ROOT/'REPORT.md').write_text('\n'.join(lines))
 print('TR0_REPORT_READY '+json.dumps({k:result[k] for k in ('formal_validation_updates','all_new_student_updates','all_new_training_transitions','evaluation_environment_transitions')}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit-pair');p.add_argument('--repeat',type=int);a=p.parse_args()
 audit_pair(a.audit_pair,a.repeat) if a.audit_pair else final_report()
