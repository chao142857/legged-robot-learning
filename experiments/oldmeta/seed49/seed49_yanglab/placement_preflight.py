"""CPU-only path, cached-source and unchanged Student implementation check."""
import json
from pathlib import Path
from common import ROOT,file_sha,write
BASE=Path('/data/go2_piper_nsgs_v02_seed49_yanglab_20261108')
CORE=['train_v02.py','teacher_v02.py','commands.py','metrics.py','neural_adapter.py','neural_owner.py','neural_core.py','neural_v01.py','sampling.py','vendor_sampler.py','common.py','evaluate.py','validation.py','command_check_math.py']
for name in CORE:assert file_sha(ROOT/'code'/name)==file_sha(BASE/'code'/name),name
assert file_sha(ROOT/'teachers/meta16.pt')=='a2e81e5a672628ad9f5463c61b6318a0f6bde6c80b714ab252a3ad0bd8f640e6'
for s in json.loads((ROOT/'sources.json').read_text())['sources']:
 assert file_sha(s['path'])==s['sha256']
assert json.loads((ROOT/'runs/student_seed49_initialization.json').read_text())==json.loads((BASE/'runs/student_seed49_initialization.json').read_text())
for kind in ('test','development'):
 assert json.loads((ROOT/f'evaluation_identity_{kind}.json').read_text())==json.loads((BASE/f'evaluation_identity_{kind}.json').read_text())
write(ROOT/'PLACEMENT_PREFLIGHT.json',dict(passed=True,student_seed=49,core_sources_identical_to_running_yanglab_seed49=CORE,
 cached_eight_source_hashes_match=True,parent_meta16_exact=True,initialization_reference_equal=True,
 evaluation_identity_equal=True,gpu_training_not_started=True,teacher_lineage='8 -> matched43/44 16 -> newly learned24'))
print('YANG_META16_24_PLACEMENT_PREFLIGHT_PASS',flush=True)
