import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from audit_point import paired


class PairedNoiseAuditTests(unittest.TestCase):
    def test_keeps_two_resets_in_same_sequence_block(self):
        # Within each sequence +0.1/-0.1 cancels; splitting resets independently
        # would incorrectly produce a nonzero uncertainty interval.
        baseline=np.full(512,.5)
        other=np.r_[np.full(256,.6),np.full(256,.4)]
        result=paired(other,baseline,42)
        self.assertAlmostEqual(result['mean_difference'],0.)
        self.assertTrue(np.allclose(result['bootstrap95_sequence_interval'],[0.,0.]))
        self.assertAlmostEqual(result['repeat_reset_mean_absolute_difference'],.1)
        self.assertEqual(result['changed_reset_count'],512)

    def test_exact_repeat_has_zero_difference(self):
        a=np.linspace(0,1,512)
        result=paired(a,a,1)
        self.assertEqual(result['mean_difference'],0.)
        self.assertEqual(result['changed_reset_count'],0)
        self.assertEqual(result['bootstrap95_sequence_interval'],[0.,0.])

    def test_requires_fixed_complete_development_denominator(self):
        with self.assertRaises(AssertionError): paired(np.zeros(511),np.zeros(511),1)


if __name__=='__main__': unittest.main()
