import json,argparse
from pathlib import Path
from common import ROOT,write

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();label=f'transfer-seed{a.seed}-clean-meta8';run=ROOT/'runs'/label
r=read(run/'result.json');assert r['passed'] and r['new_updates']==10000 and r['teacher_frozen']
catalog=read(ROOT/'baseline_catalog.json');assert catalog['seed']==a.seed
initial=read(run/'initialization.json')
for k in ('actor_sha256','critic_sha256','optimizer_empty','seed','pool_sha256','initial_sampler_uniform'):assert initial[k]==catalog['initialization'][k],k
cfg=read(run/'configuration.json')
for k in ('agent','protocol','binding','num_envs','rollout','student_seed'):assert cfg[k]==catalog['configuration'][k],k
curve=[read(x) for x in sorted((ROOT/'evaluations').glob(label+'-*-test/result.json'))];assert [x['update'] for x in curve]==list(range(1000,10001,1000))
clean=[{k:x[k] for k in ('update','joint_success_rate','continuous_joint_score','pooled_rmse')} for x in curve]
write(ROOT/'COMPARISON.json',dict(seed=a.seed,cleanmeta8=clean,existing_baselines=catalog['curves'],baseline_provenance=catalog['provenance'],initialization_and_ppo_verified=True,
 caveat='Historical baselines reused by current explicit instruction; old Meta curves are legacy implementation and may be incomplete. Single seed, not statistically significant; no new algorithm or TV effect claim.'))
lines=[f'# seed{a.seed} CleanMeta8 对照','','已有基线复用，没有重跑Uniform、Fresh或旧Meta8。','', '| Update | CleanMeta8 | Uniform | Fresh | 旧Meta8（已有点） |','|---:|---:|---:|---:|---:|']
for row in clean:
 vals=[]
 for arm in ('uniform','fresh','old-meta8'):
  match=[x for x in catalog['curves'].get(arm,[]) if x['update']==row['update']];vals.append(f"{match[0]['joint_success_rate']*100:.2f}%" if match else '')
 lines.append('| %d | %.2f%% | %s | %s | %s |'%(row['update'],100*row['joint_success_rate'],*vals))
lines+=['','旧Meta缺失点留空，不外推；本轮只比较正确实现复现，不宣称新算法改进。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
