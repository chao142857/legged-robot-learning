"""CPU-only guard tests; all subprocess calls are mocked, no simulation runs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import ppo_smoke_gate as gate
import run_bounded_ppo as launcher


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,allow_nan=True)+'\n')


def update(i,kl):
    return dict(completed_updates=i,kl_mean=kl,kl_max=kl*2,actor_grad_norm_mean=1.,actor_grad_norm_max=2.,
        critic_grad_norm_mean=1.,critic_grad_norm_max=2.,learning_rate=.0005,losses={'value':.1},
        metrics={'ee_mse':.01},action_std=[.5]*18)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ppo-gate-fixture-')
        self.base=Path(self.temp.name); self.root=self.base/'repo/task_suite_experiment'
        self.smoke=self.base/'output/train_t0_smoke_v1';self.smoke.mkdir(parents=True)
        self.admission=self.base/'output/audit/admission_t0_v1.json'
        deps=gate.current_dependencies(self.root)
        for paths in deps.values():
            for name,path in paths.items():
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture '+name)
        params={'ee_center':[.45,0,.5]};write(self.root/'task_suite/profiles.json',{'easy':params})
        self.cp=self.base/'native_fixture.pt';self.cp.write_bytes(b'fixture checkpoint, not a real model')
        self.native_patch=patch.object(gate,'NATIVE_CP3000_SHA256',gate.file_sha256(self.cp));self.native_patch.start()
        task=dict(family='T0',terrain='plane',difficulty='easy',ee_mode='smooth',params=params)
        key,_=gate.task_identity(task)
        bindings={group:{name:gate.file_sha256(path) for name,path in paths.items()} for group,paths in deps.items()}
        evidence=self.base/'physical_fixture.txt';evidence.write_text('fixture only')
        admission=dict(admission_key=key,task=task,bindings=bindings,geometry_pass=True,kinematics_pass=True,physics_pass=True,
            exact_profile_only=True,interval_or_bank_admission=False,evidence=[{'path':str(evidence),'sha256':gate.file_sha256(evidence)}])
        write(self.admission,admission)
        write(self.smoke/'candidate_identity.json',dict(admission_key=key,task=task,bindings=bindings))
        write(self.smoke/'protocol.json',dict(admission_key=key,task=task,admission=admission,checkpoint=str(self.cp),
            checkpoint_sha256=gate.file_sha256(self.cp),mode='train',requested_updates=2,seed=21,ppo_schedule='adaptive',
            learning_rate=5e-4,initialization='strict complete actor/critic, fresh optimizer, iteration zero'))
        write(self.smoke/'summary.json',dict(admission_key=key,completed_updates=2,stopped_reason='bounded run completed'))
        write(self.smoke/'transfer.json',dict(strict_load=True,bitwise_equal={'actor':True,'critic':True},optimizer_state_entries=0,
            iteration=0,actor_sha256=gate.NATIVE_ACTOR_SHA256,critic_sha256=gate.NATIVE_CRITIC_SHA256,
            observation_dims={'policy':[210],'critic':[102]}))
        final=self.smoke/'cp_final.pt';final.write_bytes(b'final fixture')
        write(self.smoke/'cp_final.json',dict(completed_updates=2,bytes=final.stat().st_size,sha256=gate.file_sha256(final)))
        self.rows=[update(1,.01),update(2,.005)];self.save_rows()
    def tearDown(self): self.native_patch.stop();self.temp.cleanup()
    def manifest(self):
        write(self.smoke/'manifest.json',{name:{'bytes':(self.smoke/name).stat().st_size,'sha256':gate.file_sha256(self.smoke/name)} for name in gate.REQUIRED_FILES})
    def save_rows(self):
        (self.smoke/'updates.jsonl').write_text(''.join(json.dumps(row,allow_nan=True)+'\n' for row in self.rows));self.manifest()
    def result(self): return gate.inspect_smoke(self.smoke,self.admission,self.root,self.cp,'t0')
    def test_complete_good_fixture_passes(self): self.assertTrue(self.result()['passed'],self.result()['errors'])
    def test_first_update_spike_rejected_even_when_last_good(self):
        self.rows[0]['kl_mean']=14.67;self.save_rows()
        self.assertTrue(any('updates:1:kl_mean_out_of_bounds' in e for e in self.result()['errors']))
    def test_threshold_inclusive_and_nan_anywhere(self):
        self.rows[0]['kl_mean']=.1;self.rows[1]['metrics']['ee_mse']=float('nan');self.save_rows()
        errors=self.result()['errors'];self.assertTrue(any('kl_mean_out_of_bounds' in e for e in errors));self.assertTrue(any('nonfinite:' in e for e in errors))
    def test_extra_update_rejected(self):
        self.rows.append(update(3,.001));self.save_rows();self.assertIn('updates:expected_exactly_2_rows',self.result()['errors'])
    def test_manifest_and_current_source_changes_rejected(self):
        (self.smoke/'updates.jsonl').write_text((self.smoke/'updates.jsonl').read_text()+'\n')
        (self.root/'task_suite/mdp.py').write_text('changed')
        errors=self.result()['errors'];self.assertIn('manifest:hash_or_size_mismatch:updates.jsonl',errors)
        self.assertIn('identity:current_dependency_mismatch:sources/task_suite/mdp.py',errors)
    def test_nonfresh_optimizer_rejected(self):
        p=self.smoke/'transfer.json';d=json.loads(p.read_text());d['optimizer_state_entries']=1;write(p,d);self.manifest()
        self.assertIn('transfer:not_fresh_optimizer_iteration_zero',self.result()['errors'])
    def test_all_four_preflight_cases_collected(self):
        with patch.object(gate,'inspect_smoke',side_effect=lambda *a:dict(case=a[-1],passed=False,errors=['fixture refusal'])) as mock:
            result=gate.preflight_all(self.base,self.root,self.cp,list(gate.CASES));self.assertEqual(mock.call_count,4)
            self.assertFalse(result['passed']);self.assertEqual(len(result['cases']),4)
    def test_rejected_stage_never_launches_and_is_failed(self):
        out=self.base/'launcher';write(out/'audit/admission_t0_v1.json',{})
        with patch.object(sys,'argv',['launcher','--phase','short','--out',str(out),'--cases','t0']),\
             patch.object(launcher,'preflight_all',return_value={'passed':False,'cases':[]}) as preflight,\
             patch.object(launcher.subprocess,'run',side_effect=AssertionError('must not launch')) as run:
            with self.assertRaises(RuntimeError): launcher.main()
            run.assert_not_called();self.assertEqual(preflight.call_args.args[3],list(gate.CASES))
        status=json.loads((out/'ppo_short_v1.json').read_text());self.assertEqual(status['state'],'failed')
        self.assertFalse(status['training_started']);self.assertEqual(status['jobs'],[])
    def test_child_timeout_marks_job_and_stage_failed(self):
        out=self.base/'timeout';write(out/'audit/admission_t0_v1.json',{})
        with patch.object(sys,'argv',['launcher','--phase','smoke','--out',str(out),'--cases','t0']),\
             patch.object(launcher.subprocess,'run',side_effect=subprocess.TimeoutExpired('fixture',1800)):
            with self.assertRaises(subprocess.TimeoutExpired): launcher.main()
        status=json.loads((out/'ppo_smoke_v1.json').read_text());self.assertEqual(status['state'],'failed')
        self.assertEqual(status['jobs'][0]['state'],'failed')


if __name__=='__main__': unittest.main()
