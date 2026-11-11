"""Independent synthetic provenance audit; fixtures never enter training data."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from sampling import Ownership
from neural_owner import NeuralOwner
from neural_v01 import NeuralSampler
from fork_v02 import convert_adapter, partial_features, learning_progress
from teacher_v02 import learning_progress_from_records


class ForkAuditTests(unittest.TestCase):
    def fixture(self, root):
        pool=SimpleNamespace(n=4, identity='synthetic:test-only')
        owner=Ownership(pool,2,'uniform',42)
        ids=np.arange(2); owner.assign(ids)
        batches=[]; journal=[]
        for index in range(3):
            for _ in range(24): owner.advance()
            token=owner.token(ids)
            successes=[index % 2 == 0, index % 2 != 0]
            rows=[[120,100 if s else 50,.1,.2,.04,.1,int(s),0,1] for s in successes]
            batches.append(dict(control_step=owner.control_steps,task_ids=token['task_ids'].tolist(),
                env_ids=ids.tolist(),generation=token['generations'].tolist(),rows=rows))
            journal.extend([[int(seq),int(ok),owner.control_steps,int(env),int(gen)]
                for seq,ok,env,gen in zip(token['task_ids'],successes,ids,token['generations'])])
            owner.commit(token,successes); owner.assign(ids)
        state=owner.state_dict(); state['engine']['episode_journal']=np.asarray(journal,dtype=np.int32)
        source=dict(owner=state,metrics=dict(scored=torch.tensor([10,0]),good=torch.tensor([5,0]),
            sq=torch.tensor([[.1,.4,.025,.1],[0.,0.,0.,0.]],dtype=torch.float64)),
            summary=dict(episodes=6,successes=3,native_failures=0,effective_commands=dict(
                command_draws={'ee':dict(count=5,minimum=[0.]*7,maximum=[1.]*7),
                    'velocity':dict(count=4,minimum=[0.]*3,maximum=[1.]*3)},
                standing_draws=1,velocity_draws=4)))
        prefix=root/'prefix.jsonl'; later=root/'later.jsonl'
        prefix.write_text(''.join(json.dumps(x)+'\n' for x in batches[:2]))
        future=copy.deepcopy(batches[-1]); future['control_step']=73
        later.write_text(json.dumps(batches[-1])+'\n'+json.dumps(future)+'\n')
        return pool,source,[prefix,later],batches

    def test_real_feedback_counters_and_future_cutoff(self):
        with tempfile.TemporaryDirectory() as td:
            pool,source,files,_=self.fixture(Path(td))
            converted,receipt=convert_adapter(source,files,2048)
            self.assertEqual(receipt['committed_terminal_records'],6)
            self.assertEqual(receipt['recent_records_imported'],6)
            self.assertEqual(receipt['last_allowed_control_step'],72)
            engine=NeuralSampler(pool.n,0); engine.load_state_dict(converted['owner']['engine'])
            self.assertEqual(int(engine.completed_counts.sum()),6)
            self.assertEqual(sum(len(x) for x in engine.recent_records.values()),6)
            self.assertLessEqual(max(t for rows in engine.recent_records.values() for t,r in rows),72)
            self.assertTrue(np.array_equal(engine.completed_counts,source['owner']['engine']['completed_counts']))

    def test_owner_restore_active_episodes_and_sampler_rng(self):
        with tempfile.TemporaryDirectory() as td:
            pool,source,files,_=self.fixture(Path(td))
            converted,_=convert_adapter(source,files,2048)
            owner=NeuralOwner(pool,2,99); owner.load_state_dict(converted['owner']); owner.check()
            self.assertTrue(np.array_equal(owner.task,source['owner']['task']))
            self.assertTrue(np.array_equal(owner.generation,source['owner']['generation']))
            self.assertTrue(np.array_equal(owner.teacher_version,[-1,-1]))
            self.assertEqual(len(np.unique(owner.decision_id)),2)
            self.assertTrue((owner.decision_id<0).all())
            owner.engine.install(np.full(pool.n,1/pool.n),0)
            clone=NeuralOwner(pool,2,31); clone.load_state_dict(owner.state_dict()); clone.check()
            self.assertTrue(np.array_equal(owner.engine.sample(4096),clone.engine.sample(4096)))

    def test_wrong_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            _,source,files,batches=self.fixture(Path(td))
            batches[0]['task_ids'][0]=(batches[0]['task_ids'][0]+1)%4
            files[0].write_text(''.join(json.dumps(x)+'\n' for x in batches[:2]))
            with self.assertRaisesRegex(AssertionError,'Log does not match'):
                convert_adapter(source,files,2048)

    def test_duplicate_or_missing_batch_rejected(self):
        for duplicate in (True,False):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as td:
                _,source,files,batches=self.fixture(Path(td))
                selected=[batches[0],batches[0],batches[1]] if duplicate else [batches[1]]
                files[0].write_text(''.join(json.dumps(x)+'\n' for x in selected))
                with self.assertRaises(AssertionError): convert_adapter(source,files,2048)

    def test_partial_measurements_not_terminal_history(self):
        with tempfile.TemporaryDirectory() as td:
            _,source,_,_=self.fixture(Path(td))
            source['owner']['task']=np.array([0,1]); source['owner']['steps']=np.array([20,10])
            original=source['owner']['accepted']
            f,r=partial_features(source,n=4)
            self.assertTrue(np.array_equal(f[:,0],[1,1,0,0]))
            self.assertTrue(np.array_equal(f[:,1],[1,0,0,0]))
            self.assertAlmostEqual(float(f[0,3]),.5)
            self.assertEqual(r['contribution_to_completed_history'],0)
            self.assertEqual(source['owner']['accepted'],original)

    def test_progress_definition_matches_transfer(self):
        engine=NeuralSampler(4,42)
        for step,score in ((100,30),(12000,50),(20000,80),(23000,90)):
            engine.observe([0],np.array([[100,score,.1,.1,.04,.1,0,0,1]]),step)
        a,am=learning_progress(engine,24000)
        b,bm=learning_progress_from_records(engine.recent_records,engine.n,24000)
        self.assertTrue(np.array_equal(am,bm))
        self.assertTrue(np.allclose(a,b))
        self.assertAlmostEqual(float(a[0]),.45,places=6)


if __name__=='__main__': unittest.main()
