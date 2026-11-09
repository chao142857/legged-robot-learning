"""Build a NEW inactive runtime; keep historical inputs byte-identical.
No launch, weights, pools or environment install. All V0.2 callers import the
fixed canonical teacher_v02 module, including teacher_ops/continual/resume.
"""
from pathlib import Path
import argparse,hashlib,json,shutil

def build(source,destination):
 source=Path(source);destination=Path(destination)
 assert not destination.exists();destination.mkdir(parents=True)
 manifest=[]
 for p in sorted(source.glob('*.py')):
  text=p.read_text();new=text
  if p.name=='teacher_v02.py':new=Path(__file__).with_name('teacher_fixed.py').read_text()
  elif p.name in ('v0_core.py','neural_core.py'):
   # Single-attempt V0 rollback already has a separate snapshot before step.
   # Own the caller snapshot too, including when restoring from a checkpoint.
   new=new.replace('self.optimizer.load_state_dict(state["optimizer"])','self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))')
   new=new.replace('self.optimizer.load_state_dict(optimizer_before)','self.optimizer.load_state_dict(copy.deepcopy(optimizer_before))')
  (destination/p.name).write_text(new)
  manifest.append(dict(name=p.name,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
   new_sha256=hashlib.sha256(new.encode()).hexdigest(),changed=text!=new))
 (destination/'ADAM_FIX_MANIFEST.json').write_text(json.dumps(dict(source=str(source),files=manifest,
  no_history_repair_claim=True,existing_checkpoint_adam_steps_preserved=True,launch_performed=False),indent=2))
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(json.dumps(build(a.source,a.destination)))
