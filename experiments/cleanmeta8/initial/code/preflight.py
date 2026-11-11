"""Strict source preservation and original Fresh identity; no training."""
import json,sys
from pathlib import Path
import numpy as np,torch
from common import ROOT,file_sha,write
from cpu_hash import tree_hash
from teacher_v02 import Teacher,V02_CONFIG
from matched_sources import EXPECTED,verify_required,source_progress

def run():
 binding=json.loads((ROOT/'CODE_BINDING.json').read_text())
 for p,h in binding['files'].items():assert file_sha(ROOT/p)==h,p
 old=json.loads((ROOT/'original_fresh_identity.json').read_text())
 state=torch.load(ROOT/'teachers/Fresh.pt',weights_only=False,map_location='cpu')['teacher']
 assert tree_hash(state)==old['state_sha256'];assert state['updates']==0 and not state['used_versions'] and not state['optimizer']['state']
 t=Teacher(np.load(ROOT/'prepared/command_features.npy'),state['binding']);t.load(state)
 assert tree_hash(t.state())==old['state_sha256'] and t.config==V02_CONFIG
 sources=json.loads((ROOT/'sources.json').read_text())['sources'];assert [(s['seed'],s['updates']) for s in sources]==EXPECTED
 checks=[]
 for s in sources:
  p=Path(s['path']);assert file_sha(p)==s['sha256'];cp=torch.load(p,map_location='cpu',weights_only=False);receipt=json.loads((p.parent/'export_receipt.json').read_text());hashes=verify_required(cp,receipt)
  source_progress(p,cp['infos']);assert cp['infos']['adapter']['owner']['engine']['version']==-1
  checks.append(dict(seed=s['seed'],update=s['updates'],checkpoint_sha256=s['sha256'],unchanged_required_state_hashes=hashes))
 assert file_sha(ROOT/'teachers/OldMeta8.pt')=='8929d0a13c4aaa7b98c70acf584d04390711bbc9ea6df8b61f3bcb556cdbd3b9'
 write(ROOT/'PREFLIGHT.json',dict(passed=True,original_fresh_full_state_identical=True,fresh_identity=old,sources=checks,
  teacher_source=str(__import__('teacher_v02').__file__),teacher_source_sha256=file_sha(Path(__import__('teacher_v02').__file__)),
  settings=V02_CONFIG,no_tv_constraint=True,no_old_feedback_reused=True))
 print('CLEANMETA8_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':run()
