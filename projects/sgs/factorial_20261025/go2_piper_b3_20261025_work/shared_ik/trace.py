"""Optional read-only diagnostic capture, called pre-reset after native reward.

Truth captured here never feeds command(), compose(), or the actor. Keep full
logs on the execution host. Actuator substep instrumentation is separate.
"""
import json


class ControlTrace:
    def __init__(self, wrapper, path, env_ids=(0,1,2,3), physical_metrics=None):
        self.wrapper=wrapper;self.raw=wrapper.unwrapped
        self.ids=list(env_ids);self.physical_metrics=physical_metrics
        self.file=open(path,'x');self.steps=0
        self.original=self.raw.reward_manager.compute
        self.raw.reward_manager.compute=self.compute

    def compute(self,*args,**kwargs):
        reward=self.original(*args,**kwargs)
        raw=self.raw;r=raw.scene['robot'];c=self.wrapper.last_control
        if c is None:raise RuntimeError('Trace requires the shared controller wrapper')
        term=raw.action_manager.get_term('joint_pos')
        row={'control_step':self.steps,'env_ids':self.ids,
             'articulation_joint_names':r.joint_names,
             'action_joint_names':self.wrapper.mapper.action_names}
        tensors=dict(c)
        tensors.update(native_action18=raw.action_manager.action,
            native_processed_target18=term.processed_actions,
            actual_joint_pos_target=r.data.joint_pos_target,
            measured_q=r.data.joint_pos,measured_dq=r.data.joint_vel,
            computed_torque=r.data.computed_torque,applied_torque=r.data.applied_torque,
            episode_age_steps=raw.episode_length_buf,
            terminated=raw.reset_terminated,timeout=raw.reset_time_outs,
            actual_ee_body_position_world=r.data.body_pos_w[:,r.body_names.index('end_effector')],
            actual_link0_position_world=r.data.body_pos_w[:,r.body_names.index('link0')])
        if self.physical_metrics is not None:
            tensors.update({'evaluation_'+k:v for k,v in self.physical_metrics(raw).items()})
        for key,value in tensors.items():
            if hasattr(value,'detach'):
                row[key]=value[self.ids].detach().cpu().tolist()
        self.file.write(json.dumps(row,allow_nan=False)+'\n');self.steps+=1
        return reward

    def close(self):
        self.raw.reward_manager.compute=self.original
        self.file.close()
