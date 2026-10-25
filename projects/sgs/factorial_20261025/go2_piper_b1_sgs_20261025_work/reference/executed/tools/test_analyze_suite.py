"""Scientific reporting regressions; no simulator, checkpoints, or third-party dependencies."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from analyze_suite import compare_pair, load_run, run_row, training_row


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='suite-analysis-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root/'pre'
        self.path.mkdir()
        (self.path/'candidate_identity.json').write_text(json.dumps(dict(bindings=dict(sources={'eval':'a'}, assets={'robot':'b'}))))
        (self.path/'transfer.json').write_text(json.dumps(dict(strict_load=True,bitwise_equal={'actor':True,'critic':True},observation_dims={'policy':[210],'critic':[102]})))
        (self.path/'reset_state.json').write_text(json.dumps(dict(root_state=[[0]*13]*4, joint_pos=[[0]*18]*4)))
        self.run = dict(name='pre', path=self.path, error=None, updates=[], parse_errors=[],
            protocol=dict(task=dict(family='T0',terrain='plane'), mode='eval', admission_key='k', seed=7101,
                          observation_noise=False, evaluator_sha256='e', telemetry_sha256='t',
                          success_thresholds=dict(xy_rmse=.15), checkpoint_sha256='cp3000'),
            summary=dict(complete_episode_evaluation=True, actor_unchanged=True, critic_unchanged=True,
                         num_episodes=4, requested_steps=1000, max_steps=1000, stochastic=False),
            episodes=[dict(env_id=i, failure=i==0, complete=i!=0, joint_success_10cm=i!=0,
                           xy_rmse=.1+i*.01) for i in range(4)])

    def post(self):
        run=copy.deepcopy(self.run)
        run['name']='post'
        run['path']=self.root/'post'
        run['path'].mkdir()
        for name in ('reset_state.json','candidate_identity.json','transfer.json'):
            (run['path']/name).write_text((self.path/name).read_text())
        run['protocol']['checkpoint_sha256']='new'
        for row in run['episodes']:
            row['xy_rmse']-=.01
        return run

    def test_smoke_has_no_full_task_rates(self):
        self.run['summary']['complete_episode_evaluation']=False
        row=run_row(self.run)
        self.assertEqual(row['status'],'censored_smoke')
        self.assertIsNone(row['joint_success_10cm_rate'])
        self.assertIsNone(row['failure_rate'])
        self.assertEqual(row['failure_observed_count'],1)

    def test_early_failure_remains_in_full_horizon_denominator(self):
        row=run_row(self.run)
        self.assertEqual(row['failure_rate'],.25)
        self.assertEqual(row['joint_success_10cm_rate'],.75)

    def test_paired_deltas_keep_failed_env(self):
        header, rows=compare_pair(self.run,self.post())
        self.assertTrue(header['comparable'])
        xy=next(row for row in rows if row['metric']=='xy_rmse')
        self.assertEqual(xy['n'],4)
        self.assertAlmostEqual(xy['post_minus_pre'],-.01)
        self.assertAlmostEqual(xy['bootstrap95_low'],-.01)
        self.assertTrue(xy['degenerate_bootstrap'])
        self.assertIn('does not establish certainty',xy['interval_warning'])

    def test_changed_evaluator_or_initial_state_rejected(self):
        post=self.post()
        post['protocol']['telemetry_sha256']='different'
        header,rows=compare_pair(self.run,post)
        self.assertFalse(header['comparable'])
        self.assertIn('telemetry_sha256',header['rejected_checks'])
        self.assertFalse(rows)
        post['protocol']['telemetry_sha256']='t'
        (post['path']/'reset_state.json').write_text(json.dumps(dict(root_state=[[1]*13]*4,joint_pos=[[0]*18]*4)))
        self.assertFalse(compare_pair(self.run,post)[0]['comparable'])

    def test_incomplete_population_cannot_have_success_rate(self):
        self.run['episodes'].pop()
        row=run_row(self.run)
        self.assertFalse(row['evaluation_population_complete'])
        self.assertIsNone(row['joint_success_10cm_rate'])
        self.assertEqual(row['status'],'incomplete_evaluation_records')

    def test_changed_asset_or_input_rejected(self):
        post=self.post()
        (post['path']/'candidate_identity.json').write_text(json.dumps(dict(bindings=dict(sources={'eval':'a'},assets={'robot':'changed'}))))
        header,_=compare_pair(self.run,post)
        self.assertIn('source_and_asset_bindings_equal',header['rejected_checks'])
        (post['path']/'candidate_identity.json').write_text((self.path/'candidate_identity.json').read_text())
        transfer=json.loads((post['path']/'transfer.json').read_text())
        transfer['observation_dims']['policy']=[211]
        (post['path']/'transfer.json').write_text(json.dumps(transfer))
        header,_=compare_pair(self.run,post)
        self.assertIn('verified_strict_transfer_same_observation_dims',header['rejected_checks'])

    def test_route_endpoint_is_separate_from_ee_and_safety(self):
        p=copy.deepcopy(self.run['protocol'])
        p['task']['family']='T2'
        p['success_thresholds']['T2_final_goal_distance']=.3
        (self.path/'protocol.json').write_text(json.dumps(p))
        (self.path/'summary.json').write_text(json.dumps(self.run['summary']))
        rows=[dict(env_id=i,complete=i!=3,safe=False,joint_success_10cm=False,goal_distance_final_m=d)
              for i,d in enumerate((.2,.29,.3,.01))]
        with (self.path/'episodes.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        row=run_row(load_run(self.path,self.root))
        self.assertEqual(row['route_endpoint_rate'],.5)
        self.assertEqual(row['joint_success_10cm_rate'],0.)
        self.assertEqual(row['safe_rate'],0.)

    def test_smoke_pair_rejected(self):
        post=self.post()
        self.run['summary']['complete_episode_evaluation']=False
        post['summary']['complete_episode_evaluation']=False
        self.assertFalse(compare_pair(self.run,post)[0]['comparable'])

    def test_norm_is_preclip_and_kl_streak_detected(self):
        self.run['updates']=[dict(completed_updates=i, env_steps=i*24*64,
            kl_mean=.11,kl_max=.2,actor_grad_norm_mean=3.,actor_grad_norm_max=5.,
            critic_grad_norm_mean=4.,critic_grad_norm_max=6.,learning_rate=.0005,
            action_std=[.3]*18,metrics=dict(xy_mse=.02)) for i in range(1,6)]
        row=training_row(self.run)
        self.assertEqual(row['actor_grad_norm_max_max'],5.)
        self.assertIn('unclipped',row['gradient_convention'])
        self.assertEqual(row['kl_mean_above_0_1_max_consecutive_updates'],5)
        self.assertEqual(row['kl_mean_above_0_1_update_count'],5)


if __name__=='__main__':
    unittest.main()
