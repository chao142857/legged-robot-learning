"""Synthetic admission-gate tests; fixtures are never physical evidence."""
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_admission_v2 import build, file_sha256, task_identity


FIELDS = ('complete', 'failure', 'safe', 'contact_safe', 'overspeed', 'joint_limit_violation',
          'nonfoot_contact', 'ee_success_10cm', 'ee_success_5cm', 'movement_success',
          'joint_success_10cm', 'joint_success_5cm')


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic-admission-test-')
        self.root = Path(self.tmp.name)
        self.evaluation = self.root/'evaluation'
        self.evaluation.mkdir()
        profiles = json.loads((Path(__file__).resolve().parents[1]/'task_suite/profiles.json').read_text())
        self.p = profiles['easy']
        source_paths = {}
        for name in ('task_suite/profiles.json', 'tools/audit_geometry_runtime.py', 'tools/run_suite.py', 'tools/telemetry.py'):
            path = self.root/'sources'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(profiles) if name.endswith('.json') else '# synthetic fixture\n')
            source_paths[name] = str(path)
        usd = self.root/'fixture_robot.usd'; usd.write_text('synthetic fixture, not USD\n')
        task = dict(family='T0', terrain='plane', difficulty='easy', ee_mode='smooth', params=self.p)
        key, _ = task_identity(task)
        self.identity = dict(task=task, admission_key=key, source_paths=source_paths,
                             asset_paths={'robot_usd': str(usd)},
                             bindings={'sources': {k:file_sha256(v) for k,v in source_paths.items()},
                                       'assets': {'robot_usd':file_sha256(usd)}})
        self.write('candidate_identity.json', self.identity)
        np.savez_compressed(self.evaluation/'geometry_samples.npz', fixture=np.ones(2))
        self.geometry = dict(task=task, geometry_only_checks_pass=True,
            source_sha256=self.identity['bindings']['sources']['tools/audit_geometry_runtime.py'],
            unique_origins_tested=1, all_unique_origins_count=1, layouts=[{}],
            checks={name:True for name in ('all_sampled_mesh_rays_finite', 'reset_foot_body_centers_above_ground',
                'foot_collision_bounds_available', 'reset_collision_clearance_finite',
                'reset_collision_aabb_clearance_ge_minus_2mm', 'start_goal_rays_finite')},
            samples={'path':str(self.evaluation/'geometry_samples.npz'),
                     'sha256':file_sha256(self.evaluation/'geometry_samples.npz')})
        self.write('geometry.json', self.geometry)
        self.summary = dict(admission_key=key, family='T0', mode='eval', complete_episode_evaluation=True,
                            num_episodes=2, max_steps=3, requested_steps=3, actor_unchanged=True,
                            critic_unchanged=True, terminations={'time_out':2, 'nonfoot_contact':0})
        self.summary.update({name+'_count':2 if name in ('complete','overspeed') else 0 for name in FIELDS})
        self.write('summary.json', self.summary)
        self.write('initial_runtime.json', dict(num_envs=2, max_episode_length=3, step_dt=.02,
                  terminations={'time_out':{'time_out':True}, 'nonfoot_contact':{'time_out':False}}))
        self.write('transfer.json', dict(strict_load=True, bitwise_equal={'actor':True, 'critic':True},
                  optimizer_state_entries=0, iteration=0, observation_dims={'policy':[210], 'critic':[102]}))
        self.write('protocol.json', dict(admission_key=key, mode='eval', task=task,
                  evaluator_sha256=self.identity['bindings']['sources']['tools/run_suite.py'],
                  telemetry_sha256=self.identity['bindings']['sources']['tools/telemetry.py']))
        self.write('reset_state.json', {'root_state':[[0,0,.35]]})
        self.write('final_runtime.json', {'synthetic_fixture':True})
        for name in ('env.yaml','agent.yaml'):
            (self.evaluation/name).write_text('synthetic_fixture: true\n')
        fields = ['env_id','steps','duration_s',*FIELDS]
        with (self.evaluation/'episodes.csv').open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for index in range(2):
                row=dict(env_id=index,steps=3,duration_s=.06)
                row.update({name:name in ('complete','overspeed') for name in FIELDS}); writer.writerow(row)
        self.arrays = {name:np.zeros((3,2)) for name in ('actions','joint_position','joint_velocity','base_pos_world',
            'base_cmd','base_actual','ee_cmd','ee_position_error','base_clearance','joint_velocity_limit_ratio',
            'joint_position_limit_excess')}
        self.arrays.update(valid=np.ones((3,2),dtype=bool), age_steps=np.tile(np.arange(1,4)[:,None],(1,2)),
                           termination_time_out=np.array([[False,False],[False,False],[True,True]]),
                           termination_nonfoot_contact=np.zeros((3,2),dtype=bool))
        self.save_trajectory()
        rows=[]
        for height in (.24,.28,.35):
            for roll,pitch in ((0.,0.),(.12,0.),(-.12,0.),(0.,.12),(0.,-.12)):
                for mode in ('fixed','smooth'):
                    for phase in np.linspace(0,2*math.pi,17 if mode=='smooth' else 1):
                        offset=np.array(self.p['ee_amplitude'])*[math.sin(phase),math.cos(phase),-math.sin(phase)]
                        xyz=np.array(self.p['ee_center'])+(offset if mode=='smooth' else 0.)
                        r20,r21,r22=-math.sin(pitch),math.cos(pitch)*math.sin(roll),math.cos(pitch)*math.cos(roll)
                        local=[xyz[0],xyz[1],(.15-r20*xyz[0]-r21*xyz[1])/r22]
                        rows.append(dict(mode=mode,phase=float(phase),base_height=height,base_roll=roll,base_pitch=pitch,
                                         command_xyz=xyz.tolist(),local_target=local,exact=True,
                                         position_residual_m=0.,orientation_residual_rad=0.))
        self.ik = dict(parameters={'center':self.p['ee_center'],'amplitude':self.p['ee_amplitude'],
            'period':self.p['ee_period_s'],'pitch_offset':self.p['ee_orientation_offset_rpy'][1],'samples':17},
            all_sampled_poses_pass=True,total=len(rows),exact_count=len(rows),rows=rows,
            assets={'fixture_robot.usd':{'expected':file_sha256(usd),'actual':file_sha256(usd)}},
            max_position_residual_m=0.,max_orientation_residual_rad=0.)
        self.ik_path=self.root/'synthetic_reachability.json'
        self.ik_path.write_text(json.dumps(self.ik))
        self.manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self,name,value):
        (self.evaluation/name).write_text(json.dumps(value))

    def manifest(self):
        self.write('manifest.json',{path.name:{'sha256':file_sha256(path),'bytes':path.stat().st_size}
                   for path in self.evaluation.iterdir() if path.is_file() and path.name!='manifest.json'})

    def save_trajectory(self):
        np.savez_compressed(self.evaluation/'trajectories.npz',**self.arrays)

    def call(self):
        return build(self.evaluation,self.ik_path)

    def test_stable_but_unsafe_can_enter_bounded_ppo(self):
        result=self.call()
        self.assertTrue(result['physics_pass'])
        self.assertFalse(result['strict_safety_pass'])
        self.assertEqual(result['strict_evaluation_results']['overspeed_count'],2)
        self.assertFalse(result['learning_demonstrated'])

    def test_reset_failure_rejected(self):
        self.geometry['checks']['reset_collision_clearance_finite']=False
        self.write('geometry.json',self.geometry); self.manifest()
        with self.assertRaisesRegex(ValueError,'failed check'): self.call()

    def test_smoke_rejected(self):
        self.summary['complete_episode_evaluation']=False
        self.write('summary.json',self.summary); self.manifest()
        with self.assertRaisesRegex(ValueError,'Censored smoke'): self.call()

    def test_stale_source_rejected(self):
        Path(self.identity['source_paths']['tools/run_suite.py']).write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'Stale source'): self.call()

    def test_mismatched_ik_rejected(self):
        self.ik['parameters']['pitch_offset']=0.
        self.ik_path.write_text(json.dumps(self.ik))
        with self.assertRaisesRegex(ValueError,'pitch_offset'): self.call()

    def test_modified_evidence_rejected(self):
        (self.evaluation/'episodes.csv').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'manifest mismatch'): self.call()

    def test_nonfinite_trajectory_rejected(self):
        self.arrays['joint_velocity'][0,0]=np.nan
        self.save_trajectory(); self.manifest()
        with self.assertRaisesRegex(ValueError,'Nonfinite physical'): self.call()

    def test_postreset_samples_rejected(self):
        self.arrays['valid'][1,0]=False
        self.save_trajectory(); self.manifest()
        with self.assertRaisesRegex(ValueError,'post-reset'): self.call()

    def test_uncovered_stone_bank_rejected(self):
        self.identity['task']['terrain']='stepping_stones'
        self.identity['admission_key']=task_identity(self.identity['task'])[0]
        self.write('candidate_identity.json',self.identity)
        with self.assertRaisesRegex(ValueError,'bank coverage'): self.call()

    def test_nonexact_easy_override_rejected(self):
        self.identity['task']['params']['ee_period_s']=10.
        self.identity['admission_key']=task_identity(self.identity['task'])[0]
        self.write('candidate_identity.json',self.identity)
        with self.assertRaisesRegex(ValueError,'exact current easy'): self.call()

    def test_less_than_half_complete_rejected(self):
        self.arrays['valid'][2,:]=False
        self.arrays['termination_time_out'][:,:]=False
        self.arrays['termination_nonfoot_contact'][1,:]=True
        self.save_trajectory()
        self.summary.update(complete_count=0,failure_count=2,nonfoot_contact_count=2,
                            terminations={'time_out':0,'nonfoot_contact':2})
        self.write('summary.json',self.summary)
        fields=['env_id','steps','duration_s',*FIELDS]
        with (self.evaluation/'episodes.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for index in range(2):
                row=dict(env_id=index,steps=2,duration_s=.04)
                row.update({name:name in ('failure','overspeed','nonfoot_contact') for name in FIELDS})
                writer.writerow(row)
        self.manifest()
        with self.assertRaisesRegex(ValueError,'Fewer than half'): self.call()


if __name__=='__main__':
    unittest.main()
