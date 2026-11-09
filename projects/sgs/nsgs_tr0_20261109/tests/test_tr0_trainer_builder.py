import ast,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from build_tr0_trainer import build

class BuilderTests(unittest.TestCase):
 def test_only_continuation_lifecycle_changes_native_ppo_unchanged(self):
  source=ROOT/'code/train_v02.py'
  if not source.exists():source=ROOT.parent/'nsgs_v02_20261106/code/train_v02.py'
  self.assertTrue(source.exists())
  with tempfile.TemporaryDirectory() as d:
   target=Path(d)/'train_tr0.py';receipt=build(source,target)
   old=ast.parse(source.read_text());new=ast.parse(target.read_text())
   def named(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
   for name in ('Runner','counted_update','optimizer_step','global_state','new_decision'):
    self.assertEqual(ast.dump(named(old,name),include_attributes=False),ast.dump(named(new,name),include_attributes=False),name)
   text=target.read_text();self.assertIn('not args.tr0 and (completed % 1000',text)
   self.assertIn('args.target-completed == 500',text)
   self.assertIn('completed-base_update in (100,300,500)',text)
   self.assertIn('torch.cuda.manual_seed_all(args.repeat_seed)',text)
   self.assertGreater(len(receipt['replacements']),3)
   with self.assertRaises(AssertionError):build(source,target)

if __name__=='__main__':unittest.main()
