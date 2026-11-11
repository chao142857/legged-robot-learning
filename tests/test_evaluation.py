import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from nsgs.evaluation.metrics import Metrics

class StrictProtocol(unittest.TestCase):
 def setUp(self):
  self.env=SimpleNamespace(num_envs=6,device='cpu',step_dt=.02,
    reset_terminated=torch.tensor([False,False,False,True,False,False]),
    reset_time_outs=torch.ones(6,dtype=torch.bool))
  self.m=Metrics(self.env,'wbc')
 def test_joint_fraction_rmse_failure_and_minimum_are_all_required(self):
  m=self.m;m.scored[:]=100;m.good[:]=80;m.sq[:]=m.tol.square()*100*.5
  m.good[1]=79;m.sq[2,2]=(.051**2)*100;m.scored[4]=99;m.good[4]=99
  m.sq[5]=m.tol.square()*100
  self.assertEqual(m.finish(torch.arange(6))[:,6].tolist(),[1,0,0,0,0,1])
 def test_settling_resets_and_simultaneous_compliance(self):
  m=self.m;m.age[:]=99
  errors=torch.zeros(6,4,dtype=torch.float64);errors[0,2]=.051
  with patch('nsgs.evaluation.metrics.physical_errors',return_value=errors):m.capture()
  self.assertEqual(m.scored.tolist(),[1]*6);self.assertEqual(m.good.tolist(),[0,1,1,1,1,1])
  m.command('ee',torch.arange(6))
  with patch('nsgs.evaluation.metrics.physical_errors',return_value=errors):m.capture()
  self.assertEqual(m.scored.tolist(),[1]*6)
 def test_nonfinite_measurements_rejected(self):
  with patch('nsgs.evaluation.metrics.physical_errors',return_value=torch.full((6,4),float('nan'))):
   with self.assertRaises(FloatingPointError):self.m.capture()
