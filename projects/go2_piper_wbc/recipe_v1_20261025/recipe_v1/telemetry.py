"""Observe native PPO and the actual actor input; no extra observation reads."""
import torch
from scratch_observer import InterfaceObserver,ScratchTelemetry
from telemetry import TrainingTelemetry,physical_suite
from .control import control_diagnostics,low_height_penalty,raw_action_tail_penalty


class RecipeInterface(InterfaceObserver):
    def reward(self,*a,**kw):
        result=self.original_reward(*a,**kw)
        r=self.raw
        self.last={'done':r.reset_buf.clone(),'timeout':r.reset_time_outs.clone()}
        for name in r.termination_manager.active_terms:
            count=r.termination_manager.get_term(name).sum()
            self.termination_counts[name]=self.termination_counts.get(name,torch.zeros_like(count))+count
        if self.steps<128 or self.steps%240==0:
            for name in self.before:
                self.equal('reward_command/'+name,r.command_manager.get_command(name),self.before[name],0.)
            p=physical_suite(r)
            lin=r.reward_manager.get_term_cfg('track_lin_vel_xy_exp')
            yaw=r.reward_manager.get_term_cfg('track_ang_vel_z_exp')
            expected={'end_effector_position_tracking_exp':torch.exp(-p['ee_position_error'].square()/.1),
                      'end_effector_orientation_tracking':p['ee_orientation_reward_error'],
                      'track_lin_vel_xy_exp':torch.exp(-p['xy_error_sq']/lin.params['std']**2),
                      'track_ang_vel_z_exp':torch.exp(-p['yaw_error_sq']/yaw.params['std']**2),
                      'low_height_l2':low_height_penalty(r),
                      'raw_action_tail_l2':raw_action_tail_penalty(r)}
            for name,value in expected.items():
                cfg=r.reward_manager.get_term_cfg(name)
                i=r.reward_manager.active_terms.index(name)
                self.equal('reward_value/'+name,r.reward_manager._step_reward[:,i],value*cfg.weight)
            self.equal('termination_or_timeout',r.reset_buf.float(),(r.reset_terminated|r.reset_time_outs).float(),0.)
        if not bool(torch.isfinite(result).all()):raise RuntimeError('Nonfinite reward')
        return result


class RecipeTelemetry(ScratchTelemetry):
    checkpoints={200,500,800,1200}

    def __init__(self,runner,out,monitor):
        self.actuator_monitor=monitor
        super().__init__(runner,out)

    def reward(self,*a,**kw):
        result=TrainingTelemetry.reward(self,*a,**kw)
        with torch.no_grad():
            r=self.env;d=r.scene['robot'].data;p=physical_suite(r)
            term=r.action_manager.get_term('joint_pos');diag=control_diagnostics(r)
            row=self.steps[-1]
            row['target_clipping_fraction']=diag['target_was_clipped'].float().mean()
            row['target_limit_excess_max_rad']=diag['target_limit_excess'].max()
            row['arm_pd_error_abs_mean_rad']=diag['pd_target_error'][:,12:].abs().mean()
            row['actual_soft_limit_fraction']=(diag['soft_limit_excess']>.001).float().mean()
            row['raw_action_abs_max']=diag['raw_action'].abs().max()
            row['actuator_saturation_fraction']=self.actuator_monitor.fraction[:,term._joint_ids].mean()
            row['low_height_fraction']=(p['base_clearance']<.16).float().mean()
            cmd=p['base_cmd'];actual=p['base_actual']
            desired=r.command_manager.get_term('base_velocity').desired
            plateau=(cmd-desired).abs().amax(-1)<1e-5
            for name,axis,sign in [('forward',0,1),('backward',0,-1),('left',2,1),('right',2,-1)]:
                w=(plateau&(desired[:,axis]*sign>.1)).float()
                row['direction/'+name+'/samples']=w.sum()
                row['direction/'+name+'/command_sq_sum']=(w*cmd[:,axis].square()).sum()
                row['direction/'+name+'/cross_sum']=(w*cmd[:,axis]*actual[:,axis]).sum()
                row['direction/'+name+'/error_sq_sum']=(w*(actual[:,axis]-cmd[:,axis]).square()).sum()
                row['direction/'+name+'/right_sign_sum']=(w*(actual[:,axis]*sign>0).float()).sum()
        return result
