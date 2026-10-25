"""Only tests directly associated with the previous control/mapping risks."""
import copy
import ast
import json
from pathlib import Path
import types
import unittest
import torch
from shared_ik import ArmOnlyIK, NativeActionMap, PolicyObservationDecoder, LEG_NAMES, ARM_NAMES

ROOT = Path(__file__).resolve().parents[1]


from shared_ik.model import KINEMATICS

@unittest.skipUnless(KINEMATICS.is_file(), "Authorized model geometry excluded; set SHARED_IK_MODEL for asset-dependent tests")
class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.ik = ArmOnlyIK(4, dtype=torch.float64)
        self.legs = torch.tensor([[-.1,.8,-1.5,.1,.8,-1.5,-.1,1.,-1.5,.1,1.,-1.5]], dtype=torch.float64).repeat(4,1)
        self.arm = torch.zeros((4,6), dtype=torch.float64).clamp(self.ik.lower, self.ik.upper)
        self.gravity = torch.tensor([[0.,0.,-1.]], dtype=torch.float64).repeat(4,1)
        self.cmd = torch.tensor([[.45,0.,.5,1.,0.,0.,0.]], dtype=torch.float64).repeat(4,1)

    def command(self, ik=None, **kw):
        return (ik or self.ik).command(self.legs, self.arm, self.gravity, self.cmd, .02, **kw)

    def install_fixed_solution(self, ik):
        target = (ik.lower*.5+ik.upper*.5).expand(4,-1).clone()
        # Drive joint1 away from the fixed measurement without leaving limits.
        target[:,0] = min(float(ik.upper[0])-.1, float(self.arm[0,0])+.8)
        def solve(this, *a, **kw):
            z = torch.zeros(4,dtype=torch.float64)
            return target.clone(), dict(position_residual_m=z.clone(), orientation_residual_rad=z.clone(), converged=torch.ones(4,dtype=torch.bool))
        ik.solve = types.MethodType(solve, ik)
        return target

    def test_reference_accumulates_while_legacy_recenters(self):
        legacy = ArmOnlyIK(4, dtype=torch.float64, slew_mode='measured_q_legacy')
        self.install_fixed_solution(self.ik); self.install_fixed_solution(legacy)
        for _ in range(10):
            new, d = self.command(); old, _ = self.command(legacy)
        self.assertAlmostEqual(float(new[0,0]-self.arm[0,0]), .4, places=9)
        self.assertAlmostEqual(float(old[0,0]-self.arm[0,0]), .04, places=9)
        self.assertTrue(bool((d['q_ref_step'].abs() <= d['slew_budget']+1e-12).all()))

    def test_per_joint_speed_and_variable_dt(self):
        self.install_fixed_solution(self.ik)
        speed = torch.tensor([.1,.2,.3,.4,.5,.6], dtype=torch.float64).repeat(4,1)
        self.ik.reset(q_arm=self.arm)
        for dt in (.01,.035,.02):
            before=self.ik.q_ref.clone()
            q,d=self.ik.command(self.legs,self.arm,self.gravity,self.cmd,dt,velocity_limits=speed)
            self.assertTrue(bool(((q-before).abs() <= speed*dt+1e-12).all()))

    def test_partial_reset_and_replay_do_not_mix_environments(self):
        self.install_fixed_solution(self.ik)
        self.command(); self.command()
        before=self.ik.state_dict()
        reset_q=((self.ik.lower+self.ik.upper)/2).repeat(2,1)
        self.ik.reset([1,3], reset_q)
        self.assertTrue(torch.equal(self.ik.q_ref[[0,2]],before['q_ref'][[0,2]]))
        self.assertTrue(torch.equal(self.ik.q_ref[[1,3]],reset_q))
        state=self.ik.state_dict(); clone=ArmOnlyIK(4,dtype=torch.float64)
        self.install_fixed_solution(clone); clone.load_state_dict(state)
        a,_=self.command(); b,_=self.command(clone)
        self.assertTrue(torch.equal(a,b))
        clone.reset([0]); self.assertFalse(bool(clone.initialized[0]))
        self.assertTrue(torch.equal(clone.q_ref[1:],a[1:]))
        bad=copy.deepcopy(state);bad['slew_mode']='measured_q_legacy'
        with self.assertRaises(ValueError): clone.load_state_dict(bad)

    def test_unreachable_goal_remains_bounded_and_not_converged(self):
        p=torch.tensor([[4.,4.,4.]],dtype=torch.float64).repeat(4,1)
        rot=torch.eye(3,dtype=torch.float64).repeat(4,1,1)
        q,d=self.ik.solve(self.arm,p,rot)
        self.assertTrue(bool(torch.isfinite(q).all()))
        self.assertTrue(bool(((q >= self.ik.lower)&(q <= self.ik.upper)).all()))
        self.assertFalse(bool(d['converged'].any()))

    def test_reachable_pose_and_analytic_jacobian(self):
        q=self.arm.clone();q[:,0]+=.1;q[:,3]-=.1
        t,j=self.ik.fk(q,True)
        found,d=self.ik.solve(q+.005,t[:,:3,3],t[:,:3,:3],iterations=30)
        self.assertTrue(bool(d['converged'].all()))
        h=1e-6
        for axis in range(6):
            dq=torch.zeros_like(q);dq[:,axis]=h
            numeric=(self.ik.fk(q+dq)[:,:3,3]-self.ik.fk(q-dq)[:,:3,3])/(2*h)
            self.assertTrue(torch.allclose(numeric,j[:,:3,axis],atol=1e-7,rtol=1e-5))

    def test_limits_recovery_and_empty_intersection(self):
        self.install_fixed_solution(self.ik)
        self.ik.reset(q_arm=self.arm)
        self.ik.q_ref[1,0]=self.ik.upper[0]+.4
        self.arm[2,0]=self.ik.lower[0]-.2
        q,d=self.command()
        self.assertTrue(bool(d['reference_limit_recovery'][1]))
        self.assertTrue(bool(d['measured_limit_violation'][2]))
        self.assertTrue(bool(((q>=self.ik.lower)&(q<=self.ik.upper)).all()))
        with self.assertRaises(ValueError): self.command(lower=self.ik.upper+.1)
        with self.assertRaises(ValueError): self.command(velocity_limits=torch.zeros(6))

    def test_initial_soft_limit_recovery_is_explicit(self):
        upper=self.ik.upper.expand(4,-1).clone();upper[:,2]=-.15
        _,d=self.command(upper=upper)
        self.assertTrue(bool(d['initialization_limit_recovery'].all()))
        self.assertTrue(bool(d['measured_limit_violation'].all()))
        self.assertTrue(bool((self.ik.q_ref[:,2] <= -.15).all()))

    def test_bad_rows_hold_and_do_not_contaminate_good_rows(self):
        self.command();state=self.ik.state_dict(); clean=ArmOnlyIK(4,dtype=torch.float64)
        clean.load_state_dict(state);expected,_=self.command(clean)
        self.cmd[1,0]=float('nan');self.cmd[2,3:]=0;self.gravity[3]=0
        q,d=self.command()
        self.assertTrue(torch.equal(q[0],expected[0]))
        self.assertTrue(torch.equal(q[1:],state['q_ref'][1:]))
        self.assertEqual(d['input_valid'].tolist(),[True,False,False,False])
        self.assertFalse(bool(d['converged'][1:].any()))
        self.assertTrue(bool(torch.isfinite(q).all()))

    def test_feedback_inf_frame_invalid_and_bad_dt(self):
        self.arm[1,0]=float('inf');self.gravity[2]=torch.tensor([-1.,0.,0.])
        q,d=self.command()
        self.assertFalse(bool(d['input_valid'][1]));self.assertFalse(bool(d['input_valid'][2]))
        self.assertTrue(bool(torch.isfinite(q).all()))
        for dt in (0.,-.1,float('nan')):
            with self.assertRaises(ValueError): self.ik.command(self.legs,self.arm,self.gravity,self.cmd,dt)


class MappingTests(unittest.TestCase):
    def test_exact_installed_native_process_function(self):
        path=ROOT/'frozen/native_joint_actions.py.txt'
        tree=ast.parse(path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='JointAction')
        function=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='process_actions')
        namespace={'torch':torch}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        names=LEG_NAMES+ARM_NAMES;scale=torch.linspace(.1,.5,18);offset=torch.linspace(-.1,.1,18)
        clip=torch.tensor([[-1.,1.]]).repeat(2,18,1)
        mapper=NativeActionMap(names,names,scale,offset,clip)
        raw=torch.arange(24).float().reshape(2,12)/10.;arm=torch.zeros(2,6)
        action,expected=mapper.compose(raw,arm)
        native=types.SimpleNamespace(_raw_actions=torch.zeros(2,18),_scale=scale,_offset=offset,
            _clip=clip,cfg=types.SimpleNamespace(clip={'.*':(-1.,1.)}))
        namespace['process_actions'](native,action)
        self.assertTrue(torch.equal(native._raw_actions,action))
        self.assertTrue(torch.equal(native._processed_actions,expected))

    def test_real_interleaved_articulation_and_native_affine(self):
        fixture=json.loads((ROOT/'frozen/native_binding_fixture.json').read_text())
        names=fixture['articulation_joint_names'];action_names=fixture['action_joint_names']
        mapper=NativeActionMap(names,action_names,fixture['scale'],fixture['action_offset'])
        legs=torch.arange(24,dtype=torch.float32).reshape(2,12)*.01
        arm=torch.arange(12,dtype=torch.float32).reshape(2,6)*.02
        a,p=mapper.compose(legs,arm)
        self.assertEqual(mapper.joint_ids,fixture['action_joint_ids'])
        self.assertTrue(torch.equal(a[:,mapper.leg_slots],legs))
        self.assertTrue(torch.allclose(p[:,mapper.arm_slots],arm))
        joint_target=mapper.articulation_targets(p)
        for i,name in enumerate(ARM_NAMES):
            self.assertTrue(torch.allclose(joint_target[:,names.index(name)],arm[:,i]))
        arm2=arm+.1;a2,p2=mapper.compose(legs,arm2)
        self.assertTrue(torch.equal(a[:,mapper.leg_slots],a2[:,mapper.leg_slots]))
        self.assertGreater(float(((a2-a)**2).sum(-1).min()),0.)

    def test_permuted_action_slots_and_heterogeneous_affine(self):
        names=LEG_NAMES+ARM_NAMES;order=names[::2]+names[1::2]
        scale=torch.linspace(.1,.7,18);scale[3]=-.5;offset=torch.linspace(-.5,.5,18)
        mapper=NativeActionMap(names,order,scale,offset)
        legs=torch.randn(3,12);arm=torch.randn(3,6)
        action,processed=mapper.compose(legs,arm)
        self.assertTrue(torch.equal(action[:,mapper.leg_slots],legs))
        self.assertTrue(torch.allclose(processed[:,mapper.arm_slots],arm,atol=1e-6))
        with self.assertRaises(ValueError):NativeActionMap(names,order,0.,offset)
        clipping=torch.tensor([[-.01,.01]]).repeat(18,1)
        clipped=NativeActionMap(names,order,scale,offset,clipping)
        with self.assertRaises(ValueError):clipped.compose(legs,torch.ones(3,6))

    def test_newest_noisy_observation_and_distinct_joint_order(self):
        fixture=json.loads((ROOT/'frozen/native_binding_fixture.json').read_text())
        names=fixture['articulation_joint_names']
        widths=[3,3,18,18,18,3,7];terms=['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','velocity_commands','pos_commands']
        shapes=[(w*3,) for w in widths];defaults=torch.tensor(fixture['articulation_default']).repeat(2,1)
        dec=PolicyObservationDecoder(terms,shapes,names,defaults)
        obs=torch.zeros(2,210);rel=torch.arange(18).float()*.001
        start=0
        for name,w in zip(terms,widths):
            if name=='joint_pos':obs[:,start:start+3*w].reshape(2,3,w)[:,-1]=rel
            if name=='projected_gravity':obs[:,start:start+3*w].reshape(2,3,w)[:,-1]=torch.tensor([.01,.02,-.99])
            if name=='pos_commands':obs[:,start:start+3*w].reshape(2,3,w)[:,-1]=torch.tensor([.45,.03,.5,1,0,0,0])
            start+=3*w
        state=dec.decode(obs)
        expected=defaults+rel
        self.assertTrue(torch.equal(state['q_arm'],expected[:,[names.index(n) for n in ARM_NAMES]]))
        self.assertTrue(torch.equal(state['q_leg'],expected[:,[names.index(n) for n in LEG_NAMES]]))
        self.assertEqual(float(state['gravity_b'][0,0]),float(obs[0,15]))


if __name__=='__main__': unittest.main()
