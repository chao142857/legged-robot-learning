import sys,unittest
from common import ROOT,write
s=unittest.defaultTestLoader.discover('/workspace/tests')
r=unittest.TextTestRunner(verbosity=2).run(s)
write(ROOT/'artifacts/tests_result.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors)))
raise SystemExit(not r.wasSuccessful())
