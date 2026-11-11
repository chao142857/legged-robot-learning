"""Regression: fresh Isaac environments have no post-step buffers at update zero."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from checkpoint_v02 import capture_environment


class InitialCaptureTests(unittest.TestCase):
    def test_zero_update_does_not_read_lazy_buffers_or_call_capture(self):
        class Pristine:
            def __getattr__(self, name):
                raise AssertionError('Pristine environment must not be queried: '+name)

        def forbidden_capture(raw, logger):
            raise AssertionError('Full capture before first native step is invalid')

        self.assertIsNone(capture_environment(0, Pristine(), Pristine(), forbidden_capture))

    def test_positive_update_invokes_full_capture_with_original_objects(self):
        raw, logger, expected = object(), object(), {'environment': 'full-real-state'}
        calls = []

        def capture(actual_raw, actual_logger):
            calls.append((actual_raw, actual_logger))
            return expected

        actual = capture_environment(100, raw, logger, capture)
        self.assertIs(actual, expected)
        self.assertEqual(calls, [(raw, logger)])

    def test_positive_capture_failure_is_not_silenced(self):
        def broken_capture(raw, logger):
            raise AttributeError('Unexpected missing post-step state')

        with self.assertRaisesRegex(AttributeError, 'Unexpected missing'):
            capture_environment(1, object(), object(), broken_capture)

    def test_negative_update_is_rejected_before_capture(self):
        def forbidden_capture(raw, logger):
            raise AssertionError('Invalid update must be rejected before capture')

        with self.assertRaises((AssertionError, ValueError)):
            capture_environment(-1, object(), object(), forbidden_capture)


if __name__ == '__main__':
    unittest.main()
