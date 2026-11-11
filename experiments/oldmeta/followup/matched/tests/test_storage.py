import json,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from storage import prepare_output
class StorageTests(unittest.TestCase):
 def test_transfer_links_and_exact_content(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'root';root.mkdir();dest=Path(d)/'data'
   (root/'STORAGE.json').write_text(json.dumps(dict(data_root=str(dest))))
   a=SimpleNamespace(mode='transfer',out=root/'runs/seed48',dataset=root/'experience/seed48')
   prepare_output(a,root);self.assertTrue(a.out.is_symlink());a.dataset.mkdir(exist_ok=False)
   (a.out/'checkpoint.pt').write_bytes(b'unchanged checkpoint bytes')
   self.assertEqual((dest/'runs/seed48/checkpoint.pt').read_bytes(),b'unchanged checkpoint bytes')
   self.assertTrue((root/'experience/seed48').is_dir())
   with self.assertRaises(AssertionError):prepare_output(a,root)
 def test_branch_does_not_relocate(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=SimpleNamespace(mode='branch',out=root/'runs/smoke',dataset=root/'experience/smoke')
   prepare_output(a,root);self.assertTrue(a.out.is_dir());self.assertFalse(a.out.is_symlink())
if __name__=='__main__':unittest.main()
