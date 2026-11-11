"""Read-only per-update reconstruction. Never emits a reusable trained Teacher."""
import sys, json, hashlib, copy, argparse
from pathlib import Path
import torch, numpy as np
from teacher_v02 import Teacher as Old
from teacher_fixed import Teacher as Fixed
from cpu_hash import tree_hash

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def steps(s): return sorted(set(int(v['step']) for v in s['optimizer']['state'].values()))
def distance(a,b): return float(sum(torch.sum((a[k]-b[k])**2) for k in a)**.5)
def run(root, name, experience=None):
 root=Path(root); static=np.load(root/'prepared/command_features.npy')
 rows=[]
 for point in sorted((Path(experience) if experience else root/'experience/meta-eight-starts').glob('point_*')):
  if not (point/'teacher_after.pt').exists(): continue
  dp=point/'decision.pt';ap=point/'teacher_after.pt'
  d=torch.load(dp,map_location='cpu',weights_only=False);out=torch.load(ap,map_location='cpu',weights_only=False)
  s=d['teacher'];after=out['teacher'];u=json.loads((point/'update.json').read_text())
  scores=[u[k] for k in ('before_dev','uniform_after','plus_after','minus_after')]
  candidates={};info={};probabilities={}
  for label,cls in [('old',Old),('fixed',Fixed)]:
   t=cls(static,s['binding'],config=s['config']);t.load(copy.deepcopy(s));i=t.learn_pair(copy.deepcopy(d),*scores)
   candidates[label]=t.state();info[label]=i
   probabilities[label]=t.probability_from_preference(d['basis'],t.mean(d['global_state']).detach().numpy())
  a,b=candidates['old'],candidates['fixed']
  row=dict(name=name,point=point.name,version=d['version'],logical_before=s['updates'],logical_after=after['updates'],
   step_before=steps(s),step_after=steps(after),corrected_local_step_after=steps(b),
   attempts=len(u['step_backtracking']),rejected=u['step_rejected'],zero_signal=u['zero_signal'],
   original_attempts=u['step_backtracking'],fixed_attempts=info['fixed']['step_backtracking'],
   corrected_local_rejected=info['fixed']['step_rejected'],
   saved_old_network_reproduces=tree_hash(a['network'])==tree_hash(after['network']),
   saved_old_optimizer_reproduces=tree_hash(a['optimizer'])==tree_hash(after['optimizer']),
   old_fixed_network_equal=tree_hash(a['network'])==tree_hash(b['network']),
   old_fixed_full_state_equal=tree_hash(a)==tree_hash(b),
   saved_old_full_state_reproduces=tree_hash(a)==tree_hash(after),
   local_mixed_probability_tv=float(np.abs(probabilities['old']-probabilities['fixed']).sum()/2),
   old_fixed_optimizer_equal=tree_hash(a['optimizer'])==tree_hash(b['optimizer']),
   network_difference_l2=distance(a['network'],b['network']),
   decision_sha256=sha(dp),checkpoint_sha256=sha(ap),checkpoint=str(ap),
   before_network_sha256=tree_hash(s['network']),after_network_sha256=tree_hash(after['network']),
   before_optimizer_sha256=tree_hash(s['optimizer']),after_optimizer_sha256=tree_hash(after['optimizer']),
   note='One isolated update from its actual historical state. NOT a corrected causal retraining chain.')
  rows.append(row)
 result=dict(name=name,root=str(root),rows=rows,source_code_sha256=sha(root/'code/teacher_v02.py'))
 return result
if __name__=='__main__':
 torch.set_num_threads(1)
 p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--name');p.add_argument('--out');p.add_argument('--experience');args=p.parse_args()
 r=run(args.root,args.name,args.experience);Path(args.out).write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='rows'}));print(json.dumps([{k:r[k] for k in ('point','step_before','step_after','corrected_local_step_after','attempts','saved_old_network_reproduces','saved_old_optimizer_reproduces','old_fixed_network_equal')} for r in r['rows']]))
