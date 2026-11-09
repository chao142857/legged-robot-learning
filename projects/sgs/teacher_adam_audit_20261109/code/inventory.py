import json,hashlib,sys
from pathlib import Path
import torch
from cpu_hash import tree_hash
base=Path(sys.argv[1]);out=Path(sys.argv[2]);torch.set_num_threads(1)
rows=[]
for r in sorted(base.glob('go2_piper_nsgs_v02*')):
 for p in sorted((r/'teachers').glob('*.pt')):
  v=torch.load(p,map_location='cpu',weights_only=False);t=v.get('teacher',v)
  if not isinstance(t,dict) or 'optimizer' not in t:continue
  rows.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),logical_updates=t['updates'],
   adam_steps=sorted(set(int(s['step']) for s in t['optimizer']['state'].values())),
   network_hash=tree_hash(t['network']),optimizer_hash=tree_hash(t['optimizer'])))
out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
