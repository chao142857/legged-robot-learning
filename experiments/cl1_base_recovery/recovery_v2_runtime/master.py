"""Engineering smoke -> corrected Recovery2k -> validated original three-arm continuation."""
import json,subprocess,time,traceback,shutil
from pathlib import Path
from host import ROOT,run,write
from controller import train,ops,sha,costs
GEN=Path('/data/go2_piper_nsgs_cl1_gen_seed70158_fixed_20261110')
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),**kw))
def child(script,*args):subprocess.run(['python3','-u',str(script),*args],cwd=script.parent,check=True)
def main():
 status('paired_physical_smoke')
 child(GEN/'gen_fixed_controller.py','--smoke-only')
 assert json.loads((GEN/'PHYSICAL_RESUME_ACCEPTANCE.json').read_text())['passed']
 status('corrected_recovery_meta_training',student_updates_budget=2000)
 ops('recovery-v2-init','init','recovery')
 plan=json.loads((ROOT/'RECOVERY_V2_PLAN.json').read_text())
 for i,item in enumerate(plan['fragments']):
  assert sha(Path(item['checkpoint']))==item['checkpoint_sha256'] and sha(Path(item['memory']))==item['memory_sha256']
  label=f'recovery-v2-f{i}-seed{item["seed"]}-from{item["update"]:05d}'
  ops(label+'-memory','fork-memory','recovery',seed=item['seed'],source_memory=item['memory'])
  train('recovery',item['seed'],item['update']+500,resume=item['checkpoint'],label=label)
  ops(label+'-finish','finish-fragment','recovery',seed=item['seed'])
  status('corrected_recovery_fragment_complete',fragments=i+1,student_updates=(i+1)*500)
 ops('recovery-v2-update','update','recovery',batch=0)
 result=json.loads((ROOT/'teachers/recovery/batch_000/result.json').read_text());assert result['macro_samples']==20 and result['version']==21
 final=ROOT/'teachers/recovery/RecoveryV2.pt';shutil.copy2(ROOT/'teachers/recovery/current.pt',final)
 executed=sum(sum(1 for _ in p.open()) for p in (ROOT/'runs').glob('*/updates.jsonl'));assert executed==2000
 write(ROOT/'REPAIR_RECOVERY_ACCEPTANCE.json',dict(passed=True,new_student_updates=2000,teacher_sha256=sha(final),
    base_source_sha256=plan['base_sha256'],normal_and_degraded_each=2,legacy_recovery_untouched=True,outer_update=result))
 costs()
 # This separate process resolves host.ROOT against the fixed generalization directory.
 child(GEN/'install_recovery.py')
 status('fixed_generalization_running',gen_root=str(GEN))
 child(GEN/'gen_fixed_controller.py')
 status('complete',new_recovery_updates=2000,generalization_updates=30000)
if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'MASTER_FAULT.txt').write_text(traceback.format_exc());status('fault',detail=traceback.format_exc());raise
