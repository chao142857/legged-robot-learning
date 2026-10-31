"""Frozen component intervention for causal diagnosis ONLY; not a trained policy.

Both frozen networks receive the same actual210D history. No inference IK,
PhysX state restore, reward change or optimizer update. Inference concatenation
is labelled diagnostic and is never accepted as single-policy WBC success.
"""
import copy,json,pathlib
import torch
from torch import nn
from rsl_rl.models import MLPModel
from telemetry import plain,write_json,sha,model_sha
from response_tap import ResponseTap

class ComponentProbe:
    def __init__(self,env,runner,observer,core,out):
        self.raw=env.unwrapped;self.current=runner.alg.actor;self.out=pathlib.Path(out)
        self.cfg=json.loads(pathlib.Path('/stage2-config.json').read_text());self.c=self.cfg['component_probe']
        assert self.c['mode'] in ('current','parent_legs','parent_arm','parent_all')
        assert self.c['start_s']>=.5 and self.c['blend_s']>=.1
        if self.c.get('first_known_failure_s') is not None:
            assert self.c['start_s']+self.c['blend_s']+.2<self.c['first_known_failure_s'], 'Intervene before the measured failure, never after it'
        assert self.current.obs_dim==210 and isinstance(self.current.obs_normalizer,nn.Identity)
        parent_path=pathlib.Path('/diagnostic_parent/checkpoint.pt');assert sha(parent_path)==self.cfg['diagnostic_parent_sha256']
        checkpoint=torch.load(parent_path,map_location='cpu',weights_only=False);state=checkpoint['actor_state_dict']
        if 'synergy_head.weight' in state:
            self.parent=copy.deepcopy(self.current)
            assert hasattr(self.parent,'synergy_head')
        else:
            # Reassemble the exact installed native class from verified modules;
            # strict loading covers all actual native names, tensors and shapes.
            self.parent=MLPModel.__new__(MLPModel);nn.Module.__init__(self.parent)
            for name in ('obs_groups','obs_dim','obs_normalization'):setattr(self.parent,name,copy.deepcopy(getattr(self.current,name)))
            for name in ('obs_normalizer','distribution','mlp'):setattr(self.parent,name,copy.deepcopy(getattr(self.current,name)))
        self.parent.load_state_dict(state,strict=True)
        assert self.parent.state_dict().keys()==state.keys()
        assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in self.parent.state_dict().items())
        self.parent.eval().requires_grad_(False);self.parent_hash=model_sha(self.parent)
        self.tap=ResponseTap(self.raw,runner,observer,out);self.old_step=env.step;env.step=self.step
        # The reused tap records computed PD torque. Additionally measure actual
        # clipped/applied effort at all4substeps; do not call computed effort
        # a measured applied torque or infer saturation from requests alone.
        for name,act in self.raw.scene['robot'].actuators.items():
            self.tap.rows[name]['applied']=[]
            original=act.compute;self.tap.old.append((act,'compute',original))
            def applied(*args,_original=original,_act=act,_name=name,**kwargs):
                result=_original(*args,**kwargs)
                self.tap.rows[_name]['applied'].append(_act.applied_effort.detach().clone())
                return result
            act.compute=applied
        self.original_physical=core.physical_suite;self.last={};self.steps=0
        def observed(raw):
            p=self.original_physical(raw)
            # Native core queries support_height once for initial metadata,
            # before any control step. No4substep actuator sample exists yet.
            if self.tap.input is None:return p
            p.update(self.tap.physical());p.update(self.last)
            p['component_contacts_history_w']=raw.scene['contact_forces'].data.net_forces_w_history.clone()
            robot=raw.scene['robot'];effort=torch.empty((raw.num_envs,4,len(robot.joint_names)),device=raw.device)
            limits=torch.empty((raw.num_envs,len(robot.joint_names)),device=raw.device)
            for name,act in robot.actuators.items():
                values=self.tap.rows[name]['applied'];assert len(values)==4
                effort[:,:,act.joint_indices]=torch.stack(values,dim=1)
                limits[:,act.joint_indices]=act.effort_limit
            ids=self.tap.ids
            p['component_applied_torque_substeps']=effort[:,:,ids]
            p['component_effort_limits']=limits[:,ids]
            p['component_base_angular_velocity_b']=robot.data.root_ang_vel_b.clone()
            return p
        core.physical_suite=observed
        robot=self.raw.scene['robot'];ids=self.raw.action_manager.get_term('joint_pos')._joint_ids
        realized=dict(root_state=plain(robot.data.root_state_w),joint_q=plain(robot.data.joint_pos),joint_qd=plain(robot.data.joint_vel),
            action_joint_names=self.raw.action_manager.get_term('joint_pos')._joint_names,body_names=robot.body_names,
            contact_sensor_body_names=getattr(self.raw.scene['contact_forces'],'body_names','API_UNAVAILABLE_NOT_INFERRED'),
            hard_joint_limits=plain(robot.data.joint_pos_limits[:,ids]),actuators={n:dict(stiffness=plain(a.stiffness),damping=plain(a.damping),effort_limit=plain(a.effort_limit),delay=plain(a.positions_delay_buffer.time_lags)) for n,a in robot.actuators.items()})
        for name in ('get_masses','get_coms','get_inertias','get_material_properties'):
            if hasattr(robot.root_physx_view,name):
                try:realized[name]=plain(getattr(robot.root_physx_view,name)())
                except (AttributeError,NotImplementedError) as error:realized[name]='NOT_MEASURED:'+type(error).__name__
            else:realized[name]='API_UNAVAILABLE_NOT_MEASURED'
        write_json(self.out/'component_realized_start.json',realized)
        write_json(self.out/'component_contract.json',dict(mode=self.c['mode'],parent_checkpoint_sha256=self.cfg['diagnostic_parent_sha256'],parent_strict_load=True,
            parent_normalization=str(self.parent.obs_normalizer),actor_observations=210,critic_observations=102,actions=18,
            start_s=self.c['start_s'],blend_s=self.c['blend_s'],policy_success_claim=False,optimizer_updates=0,
            causal_scope='Independent cold starts with actual identical prefix; never hidden PhysX-state recovery. Both networks observe actual mixed action history.',
            inference_composition_diagnostic_only=True,all_original_physics_rewards_commands_and_strict_metrics_unchanged=True))
    def capture_input(self,obs):
        self.tap.capture_input(obs)
        with torch.no_grad():
            self.cur=self.current(obs,stochastic_output=False).detach().clone()
            self.prior=self.parent(obs,stochastic_output=False).detach().clone()
        self.last=dict(component_current_mean=self.cur,component_parent_mean=self.prior,
            component_applied_mean=torch.zeros_like(self.cur),component_delta=torch.zeros_like(self.cur),
            component_input_time_s=(self.raw.episode_length_buf*self.raw.step_dt).clone())
    def step(self,actions):
        assert torch.equal(actions,self.cur),'Diagnostic must see the actual deterministic native mean'
        applied=actions.clone();age=self.raw.episode_length_buf*self.raw.step_dt
        alpha=((age-float(self.c['start_s']))/float(self.c['blend_s'])).clamp(0,1)
        alpha=alpha.square()*(3-2*alpha)
        if self.c['mode']!='current':
            cols={'parent_legs':slice(0,12),'parent_arm':slice(12,18),'parent_all':slice(0,18)}[self.c['mode']]
            applied[:,cols]=actions[:,cols]+alpha[:,None]*(self.prior[:,cols]-actions[:,cols])
        before=age<float(self.c['start_s'])
        assert torch.equal(applied[before],actions[before]),'No pre-start clipping or history change'
        self.last['component_applied_mean']=applied.clone();self.last['component_delta']=(applied-actions).clone()
        result=self.old_step(applied);self.steps+=1;return result
    def finish(self,data,summary):
        assert model_sha(self.parent)==self.parent_hash and summary['actor_unchanged'] and summary['critic_unchanged']
        write_json(self.out/'component_result.json',dict(complete_control_steps=self.steps,mode=self.c['mode'],parent_unchanged=True,current_unchanged=True,
            native_strict_safe=summary['safe_count'],native_complete=summary['complete_count'],diagnostic_only=True,trained_policy_success=False))
        self.tap.close()

def install(env,runner,observer,core,out):return ComponentProbe(env,runner,observer,core,out)
