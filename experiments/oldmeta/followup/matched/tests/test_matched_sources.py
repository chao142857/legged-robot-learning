import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from matched_sources import EXPECTED,source_progress,verify_required,import_ready
from common import file_sha
from cpu_hash import tree_hash

class MatchedSourceTests(unittest.TestCase):
    def test_partial_file_never_ready_and_marker_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'prepared.pt';path.write_bytes(b'partial')
            self.assertFalse(import_ready(path))
            marker=root/'IMPORT_READY.json';marker.write_text(json.dumps(dict(passed=True,checkpoint_sha256=file_sha(path))))
            self.assertTrue(import_ready(path))
            path.write_bytes(b'changed')
            with self.assertRaises(AssertionError):import_ready(path)
    def test_source_grid_exact_order(self):
        self.assertEqual(EXPECTED,[(43,1000),(44,1000),(43,3000),(44,3000),(43,6000),(44,6000),(43,9000),(44,9000)])
    def test_progress_snapshot_rejects_future_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row=dict(completed_updates=1000,native_feedback_episodes=300,learning_rate=.001)
            path=root/'progress_row.json';path.write_text(json.dumps(row))
            (root/'export_receipt.json').write_text(json.dumps(dict(progress_row_sha256=file_sha(path))))
            info=dict(completed_updates=1000,adapter=dict(owner=dict(accepted=300)),adaptive_learning_rate=.001)
            self.assertEqual(source_progress(root/'prepared.pt',info)[0],row)
            info['completed_updates']=2000
            with self.assertRaises(AssertionError):source_progress(root/'prepared.pt',info)
            path.write_text('{}')
            with self.assertRaises(AssertionError):source_progress(root/'prepared.pt',info)
    def test_required_state_hashes_reject_optimizer_or_rng_change(self):
        saved={k:dict(value=1) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')}
        saved['infos']={k:dict(value=1) for k in ('adapter','environment','rng','v02_context')}
        hashes={k:tree_hash(v) for k,v in saved.items() if k!='infos'}
        hashes.update({k:tree_hash(v) for k,v in saved['infos'].items()})
        receipt=dict(unchanged_required_state_hashes=hashes)
        self.assertEqual(verify_required(saved,receipt),hashes)
        saved['optimizer_state_dict']['value']=2
        with self.assertRaises(AssertionError):verify_required(saved,receipt)
        saved['optimizer_state_dict']['value']=1;saved['infos']['rng']['value']=2
        with self.assertRaises(AssertionError):verify_required(saved,receipt)
if __name__=='__main__':unittest.main()
