"""Lock selection before new training: ambiguous single-indicator declines are neither class."""
import copy,random

def category(row):
 bad_j=row['delta_j']<=-.03;bad_ee=row['delta_ee']>=.003
 return 'degradation' if bad_j and bad_ee else 'normal' if not bad_j and not bad_ee else 'ambiguous'
def select(old):
 rng=random.Random(20261109);result=[]
 for seed in (43,44):
  # Keep the original degraded source, not a retrospectively best/worst recovery.
  event=next(x for x in old['batches'][0] if x['seed']==seed and x['condition']=='degradation')
  assert category(event)=='degradation'
  candidates=[x for x in old['selection_catalog'][str(seed)]['normal_candidates'] if category(x)=='normal' and abs(x['update']-event['update'])>=500]
  distance=min(abs(x['update']-event['update']) for x in candidates)
  tied=sorted([x for x in candidates if abs(x['update']-event['update'])==distance],key=lambda x:x['update'])
  normal=copy.deepcopy(rng.choice(tied));normal['condition']='normal'
  result.extend([copy.deepcopy(event),normal])
 return result
