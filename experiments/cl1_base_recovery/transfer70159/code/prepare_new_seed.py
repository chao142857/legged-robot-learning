import copy,json
from pathlib import Path
from common import ROOT,write
from frozen_ops import load
from fixed_prepare import install
from checkpoint_pair import sha
S=Path('/data/go2_piper_nsgs_cl1_gen_seed70158_fixed_20261110')
locks={}
for arm in ('uniform','frozen-base','frozen-recovery'):
 src=S/'teachers'/arm/'initial.pt';s=load(src)
 s['students']={'70159':dict(history=[],pending=None,previous=None,transitions=[],cursor=0)};s['batches']=0
 locks[arm]=dict(source=str(src),source_sha256=sha(src),**install(arm,s))
write(ROOT/'INITIAL_LOCK.json',dict(seed=70159,arms=locks,all_student_memory_empty=True))
write(ROOT/'RECOVERY_V2_LOCK.json',json.loads((S/'RECOVERY_V2_LOCK.json').read_text()))
print('FRESH_SEED_INITIALIZATION_PASSED',flush=True)
