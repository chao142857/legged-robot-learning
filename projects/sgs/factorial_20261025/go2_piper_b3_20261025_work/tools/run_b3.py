"""B3 delegates to B2's single common native entry; no private sampler/control."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT.parent/'go2_piper_b2_20261025_work/tools/run_b2.py'
if '--method' in sys.argv:
    raise SystemExit('B3 wrapper fixes --method B3; use common entry directly for B2')
sys.argv=[str(ENTRY),'--method','B3',*sys.argv[1:]]
runpy.run_path(str(ENTRY),run_name='__main__')
