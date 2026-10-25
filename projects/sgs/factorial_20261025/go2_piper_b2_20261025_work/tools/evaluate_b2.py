"""Thin B2 alias for the single shared B2/B3 frozen evaluator owned by B3."""
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT.parent/'go2_piper_b3_20261025_work/tools/evaluate_b3.py'
if __name__=='__main__':
    args=sys.argv[1:]
    if '--method' not in args:args=['--method','B2',*args]
    sys.argv=[str(TARGET),*args]
    runpy.run_path(str(TARGET),run_name='__main__')
