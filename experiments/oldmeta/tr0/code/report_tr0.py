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
 gpu_intervals=sorted((j['started_unix'],j['finished_unix']) for j in jobs+attempt_jobs if j.get('gpu') and j.get('finished_unix'))
 merged=[]
 for start,end in gpu_intervals:
  if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
  else:merged.append([start,end])
 occupied_gpu_job_seconds=sum(end-start for start,end in merged)
 job_by_label={j['label']:j for j in jobs}
 contrasts=[]
 for pair in pairs:
  a={r['teacher']:r for r in pair['arms']}
  contrasts.append(dict(source=pair['source'],repeat=pair['repeat'],
    gains={k:v['points'][-1]['gain'] for k,v in a.items()},
    sampling_tr_minus_old=a['sampling-tr']['points'][-1]['gain']-a['old']['points'][-1]['gain'],
    sampling_tr_minus_original=a['sampling-tr']['points'][-1]['gain']-a['original']['points'][-1]['gain'],
    original_minus_old=a['original']['points'][-1]['gain']-a['old']['points'][-1]['gain']))
 started=read(ROOT/'RESOURCE_ACCEPTANCE.json').get('unix')
 finished=max(j['finished_unix'] for j in jobs+attempt_jobs if j.get('finished_unix'))
 adam_fix=read(ROOT/'ADAM_FIX_ACCEPTANCE.json') if (ROOT/'ADAM_FIX_ACCEPTANCE.json').exists() else None
 adam_diagnostic=read(ROOT/'ADAM_SNAPSHOT_DIAGNOSTIC.json') if (ROOT/'ADAM_SNAPSHOT_DIAGNOSTIC.json').exists() else None
 if adam_fix:finished=max(finished,(ROOT/'ADAM_FIX_ACCEPTANCE.json').stat().st_mtime)
 validation_baselines=[]
 for source in SOURCES[1:]:
  path=ROOT/'starts'/source['id']/'baseline_dev.json'
  if path.exists():
   b=read(path);validation_baselines.append(dict(source=source['id'],seed=source['seed'],source_update=source['update'],
    continuous_joint_score=b['continuous_joint_score'],strict_joint_success=b['joint_success_rate'],rmse=b['pooled_rmse'],
    native_failure_rate=b['native_failure_rate'],validation_500_update_results=None if not formal else 'See pair_audits'))
 fixed_policy_repeats={}
 for key,a,b in [('source_before','p00-seed43-cp01000-before','p00-seed43-cp01000-before-repeat'),('plus_after','p00-seed43-cp01000-plus-after','p00-seed43-cp01000-plus-after-repeat')]:
  first=read(ROOT/'evaluations'/a/'result.json');second=read(ROOT/'evaluations'/b/'result.json')
  aa=np.asarray(first['scores']);bb=np.asarray(second['scores'])
  fixed_policy_repeats[key]=dict(mean_difference=float((bb-aa).mean()),mean_absolute_per_reset_difference=float(abs(bb-aa).mean()),
    max_absolute_per_reset_difference=float(abs(bb-aa).max()),score_vectors_equal=bool(np.array_equal(aa,bb)),
    scope='Fixed-policy development reset repeat only, not same-policy PPO training/restore variance')
 result=dict(passed=True,formal_validation_updates=sum(r['new_updates'] for r in formal),
  all_new_student_updates=sum(r['new_updates'] for r in trains),all_new_training_transitions=sum(r['new_environment_transitions'] for r in trains),
  fresh_candidate_updates=300,smoke_updates=2,validation_branch_count=len(formal),source_seeds=[44,46,47,48],
  evaluation_environment_transitions=sum(e['executed_environment_transitions'] for e in all_eval),
  evaluation_job_wall_seconds=sum(j['wall_seconds'] for j in jobs+attempt_jobs if j['label'].startswith('eval-')),
  failed_attempt_job_wall_seconds=sum(j['wall_seconds'] for j in attempt_jobs),
  archived_attempt_evaluations=len(attempt_evaluations),
  student_job_wall_excluding_wait_for_evaluations=sum(max(0,job_by_label[Path(r['experience']).name]['wall_seconds']-r['evaluation_seconds']) for r in trains),
  gpu_job_window_union_seconds=occupied_gpu_job_seconds,gpu_job_window_hours=occupied_gpu_job_seconds/3600,
  gpu_job_windows=merged,experiment_finished_unix=finished,total_pipeline_wall_seconds=finished-started if started else None,
  cost_scope='New updates exclude historical sources. GPU job window union avoids nested Student/dev double counting and includes startup/serialization/failed smoke; it is reserved job wall time, not utilization-integrated active GPU time. End-to-end wall includes schema repair and post-run Adam audit/repair intervals.',
  fixed_policy_repeats=fixed_policy_repeats,validation_source_baselines=validation_baselines,
  adam_snapshot_fix=adam_fix,legacy_adam_snapshot_diagnostic=adam_diagnostic,
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
 if adam_fix:
  lines+=['## Adam快照审计与修复','',
   '独立核验发现旧CPU load_state_dict共享状态Tensor：逻辑Meta8的实际Adam步数为15，旧中间候选经过两次KL尝试推进到17。该中间候选与原报告、图、权重已封存到attempts/01-adam-snapshot-alias，不作为本报告结论。',
   '本轮保留Meta8原始参数/Adam资产不变，只在TR0的B/C构造中让每次load使用独立深拷贝。修复后实际接受步为15→16；新增回归覆盖多次KL回溯和全拒绝。',
   '同一新反馈从原Meta8重构，梯度和第一次完整Adam方向的全概率逐位不变，固定13份H身份不变；未再运行PPO，也未从错误的9步Teacher继续学习。',
   '因此这里Original-step指保留原梯度与KL规则、但修复Adam快照隔离的版本；不能把它当作旧别名错误的复刻。历史Meta8/16及既有实验结果没有回写。','']
 lines+=['原规则尝试的实际候选（正确隔离Adam快照，先应用原有KL，再判断新增TV是否改变结果）：','',
  '|学习率系数|实际LR|偏好KL|13输入最大序列TV|原规则结果|','|---:|---:|---:|---:|---|']
 for a in candidate['original_candidate_audit']:
  if not a.get('finite'):continue
  lines.append('|%.4g|%.6g|%.8f|%.6f%%|%s|'%(a['learning_rate_multiplier'],.01*a['learning_rate_multiplier'],a['preference_kl'],
    100*max(x['tv'] for x in a['changes']),'KL通过' if a['preference_kl']<=.05 else '已被原有KL拒绝'))
 lines+=['','全学习率原始候选的TV可能高于2%，但它已被原有KL拒绝；新增TR没有进一步改变最终接受步。接受后下一次基准LR仍为0.01。','']
 lines+=['## 新真实方向反馈（不是12k验证结果）','',
  '|分叉|新增PPO更新|开发集连续联合达标|相对固定起点增量|','|---|---:|---:|---:|']
 scores=candidate['fresh_direction_scores']
 for arm in ('uniform','plus','minus'):
  lines.append('|%s|100|%.6f%%|%+.6f个百分点|'%(arm,100*scores[arm],100*(scores[arm]-scores['before'])))
 lines+=['','固定起点连续指标为%.6f%%；三臂均从seed43 CP1000完整状态开始，首次结果用于同一个真实梯度，没有挑选重复结果或放大梯度。'%(100*scores['before']),
  '这是单个随机扰动对。q+表现更好不能证明均值Teacher更新会产生迁移收益；后者需要有效、不同的后续干预对照。','']
 if not candidate['launch_12k']:
  lines+=['没有启动12k验证。原因：'+str(candidate['stop_reason'])+'。',
   '若新限制未触发，B/C相同，这次没有形成更新幅度干预，不能据此判定信赖域有效或无效，也不能检验“大步危害”。',
   '四个验证源的完整起点及基线已保留；没有生成或冒充逐起点PPO对照结果。','']
  lines+=['## 四起点仅完成的固定开发基线','',
   '|Student seed|源更新数|连续联合达标|严格联合成功率|EE位置RMSE|+500三组对照|','|---:|---:|---:|---:|---:|---|']
  for b in validation_baselines:
   lines.append('|%d|%d|%.4f%%|%.4f%%|%.4fcm|未运行|'%(b['seed'],b['source_update'],100*b['continuous_joint_score'],100*b['strict_joint_success'],100*b['rmse']['ee_position_m']))
  lines+=['','这些是独立开发集256序列×2固定重置的基线，不是正式1024测试池结果，也不是Teacher对照训练后的表现。','']
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
  'GPU作业占用窗口并集：%.1fs（%.4f小时，含启动/序列化和失败smoke，不是GPU利用率积分）。'%(occupied_gpu_job_seconds,occupied_gpu_job_seconds/3600),
  '首次GPU检查至正确候选核验结束的墙钟：%.1fs，包含字段修复及Adam独立审计修复期间的空闲。'%(result['total_pipeline_wall_seconds'] or 0),
  '固定策略的两组评测重复已记录；它们不能替代同策略PPO训练恢复方差测量。',
  '原有E1–E3、nSGS V0–V0.2和其他主机队列没有修改。PPO、奖励、观测、18维动作、命令池及严格5cm/80%定义保持原样。','']
 (ROOT/'REPORT.md').write_text('\n'.join(lines))
 print('TR0_REPORT_READY '+json.dumps({k:result[k] for k in ('formal_validation_updates','all_new_student_updates','all_new_training_transitions','evaluation_environment_transitions')}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit-pair');p.add_argument('--repeat',type=int);a=p.parse_args()
 audit_pair(a.audit_pair,a.repeat) if a.audit_pair else final_report()
