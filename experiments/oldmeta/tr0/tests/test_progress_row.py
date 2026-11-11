import sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
archive=ROOT.parent/'nsgs_v02_20261106/code'
if archive.exists():sys.path.append(str(archive))
from prepare_tr0 import complete_progress_row

class ProgressRowTests(unittest.TestCase):
 def test_legacy_missing_std_is_measured_from_checkpoint_not_imputed(self):
  row={'completed_updates':1000,'losses':{'value':.05}}
  state={'distribution.log_std_param':torch.linspace(-.8,.3,18)}
  result=complete_progress_row(row,state)
  self.assertNotIn('action_std',row)
  self.assertEqual(result['action_std'],float(state['distribution.log_std_param'].exp().mean()))
  self.assertEqual(result['losses'],row['losses'])
 def test_existing_valid_native_snapshot_is_preserved(self):
  state={'distribution.log_std_param':torch.zeros(18)}
  self.assertEqual(complete_progress_row({'action_std':.999998},state)['action_std'],.999998)
 def test_invalid_recorded_std_rejected(self):
  state={'distribution.log_std_param':torch.zeros(18)}
  with self.assertRaises(AssertionError):complete_progress_row({'action_std':-1.},state)

if __name__=='__main__':unittest.main()
