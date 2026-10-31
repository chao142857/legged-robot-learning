"""Reused frozen-only actuator/input taps; no command generator or controller."""
import math
import torch
from scratch_observer import command_observations
from telemetry import plain,write_json

class ResponseTap:
    def __init__(self,raw,runner,observer,out):
        self.raw,self.runner,self.observer=raw,runner,observer
        self.robot=raw.scene['robot']; self.action=raw.action_manager.get_term('joint_pos')
        self.ids=self.action._joint_ids
        self.input=None; self.network_input=None; self.rows={};self.old=[];self.max_pd_error=0.
        layers=[(n,m) for n,m in runner.alg.actor.named_modules() if isinstance(m,torch.nn.Linear) and m.in_features==210]
        assert len(layers)==1,[(n,m.in_features) for n,m in layers]
        self.hook=layers[0][1].register_forward_pre_hook(self.capture_network)
        for name,act in self.robot.actuators.items():
            assert type(act).__name__=='DelayedPDActuator'
            self.rows[name]={k:[] for k in ['position','velocity','effort','q','qd','torque','delay']}
            for kind,attr in [('position','positions_delay_buffer'),('velocity','velocities_delay_buffer'),('effort','efforts_delay_buffer')]:
                buf=getattr(act,attr); original=buf.compute; self.old.append((buf,'compute',original))
                def tapped(*a,_original=original,_name=name,_kind=kind,**kw):
                    result=_original(*a,**kw)
                    self.rows[_name][_kind].append(result.detach().clone())
                    return result
                buf.compute=tapped
            original=act.compute;self.old.append((act,'compute',original))
            def computed(*a,_original=original,_name=name,_act=act,**kw):
                qp=kw.get('joint_pos',a[1] if len(a)>1 else None).detach().clone()
                qv=kw.get('joint_vel',a[2] if len(a)>2 else None).detach().clone()
                result=_original(*a,**kw)
                r=self.rows[_name]
                expected=_act.stiffness*(r['position'][-1]-qp)+_act.damping*(r['velocity'][-1]-qv)+r['effort'][-1]
                err=float((expected-_act.computed_effort).abs().max());self.max_pd_error=max(self.max_pd_error,err)
                assert err<1e-4,('PD tap mismatch',_name,err)
                r['q'].append(qp);r['qd'].append(qv);r['torque'].append(_act.computed_effort.detach().clone())
                r['delay'].append(_act.positions_delay_buffer.time_lags[:,None].expand_as(qp).clone())
                return result
            act.compute=computed
        manager=raw.observation_manager
        write_json(out/'response_layout.json',dict(action_joint_names=self.action._joint_names,articulation_joint_names=self.robot.joint_names,
            actor_terms=manager.active_terms['policy'],actor_term_dims=manager.group_obs_term_dim['policy'],actor_normalizer=str(runner.alg.actor.obs_normalizer) if hasattr(runner.alg.actor,'obs_normalizer') else str(runner.alg.actor),
            network_input_layer=layers[0][0],timing='input t=(age_steps-1)*.02; actual q/EE t=age_steps*.02; delay values at t_input+[0,.005,.010,.015]',
            actuators={n:dict(joint_indices=plain(a.joint_indices),class_name=type(a).__name__) for n,a in self.robot.actuators.items()}))
    def capture_network(self,module,args):
        self.network_input=args[0].detach().clone()
    def capture_input(self,obs):
        for row in self.rows.values():
            for values in row.values():values.clear()
        self.observer.capture_input(obs)
        self.input=obs['policy'].detach().clone()
        self.input_cmd=command_observations(self.raw,obs)['policy/pos_commands'].detach().clone()
        self.input_vel=command_observations(self.raw,obs)['policy/velocity_commands'].detach().clone()
        self.q_pre=self.robot.data.joint_pos[:,self.ids].detach().clone()
        self.qd_pre=self.robot.data.joint_vel[:,self.ids].detach().clone()
        self.network_input=None
    def physical(self):
        if self.input is None:return {}
        assert self.network_input is not None
        fields=dict(diag_actor_obs=self.input,diag_actor_network_input=self.network_input,diag_cmd_input=self.input_cmd,
                    diag_velocity_input=self.input_vel,diag_q_pre=self.q_pre,diag_qd_pre=self.qd_pre)
        for kind in ('position','velocity','effort','q','qd','torque','delay'):
            result=torch.empty((self.raw.num_envs,4,len(self.robot.joint_names)),device=self.raw.device)
            for name,act in self.robot.actuators.items():
                values=self.rows[name][kind]
                assert len(values)==4,(name,kind,len(values))
                result[:,:,act.joint_indices]=torch.stack(values,dim=1)
            fields['diag_delay_'+kind]=result[:,:,self.ids]
        return fields
    def close(self):
        self.hook.remove()
        for obj,name,original in reversed(self.old):setattr(obj,name,original)
