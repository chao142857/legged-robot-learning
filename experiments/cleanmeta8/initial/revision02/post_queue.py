import sys,time,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import controller as c
try:
 while not (ROOT/'HANDOFF_COMPLETE.json').exists():time.sleep(1)
 assert c.read(ROOT/'CLEANMETA8_ACCEPTANCE.json')['passed']
 c.train('transfer-seed48-clean-meta8',10000,48,mode='transfer',branch='clean-meta8',transfer_kind='meta',teacher_in=ROOT/'teachers/CleanMeta8.pt')
 c.run('seed48-existing-baselines-report',[ROOT/'code/comparison_reuse.py','--seed','48'],gpu=False)
 c.status('complete',teacher='CleanMeta8',meta_training_student_updates=2400,student_seed=48,new_student_updates=10000,baselines_reused=True)
except BaseException:
 (ROOT/'POST_QUEUE_FAULT.txt').write_text(traceback.format_exc());raise
