"""Tracking metrics on real pre-reset samples; counterfactuals are labelled math."""
import numpy as np


def tracking_metrics(data, params, dt=.02, family='T1'):
    valid=data['valid'].astype(bool)
    target=data['ee_cmd'][...,:3]
    actual=data['ee_actual_mixed']
    center=np.asarray(params['ee_center'])
    base_command=data['base_cmd'];base_actual=data['base_actual']
    age=(data['age_steps']-1)*dt
    env_ids=np.arange(valid.shape[1])[None,:]
    desired_group=(env_ids+(age/params['base_segment_s']).astype(int))%4
    def ee(mask):
        if not mask.any():return {'samples':0}
        t=target[mask];a=actual[mask]
        sq=(a-t)**2;reference=(t-center)**2
        mse=float(sq.sum(-1).mean());ref=float(reference.sum(-1).mean())
        axes={}
        for j,axis in enumerate('xyz'):
            x=t[:,j]-center[j];y=a[:,j]-center[j]
            energy=float((x*x).mean());error=float(sq[:,j].mean())
            covariance=float(np.mean((t[:,j]-t[:,j].mean())*(a[:,j]-a[:,j].mean())))
            variance=float(np.var(t[:,j])*np.var(a[:,j]))
            axes[axis]={'target_mean_m':float(t[:,j].mean()),'actual_mean_m':float(a[:,j].mean()),
                'target_std_m':float(t[:,j].std()),'actual_std_m':float(a[:,j].std()),
                'rmse_m':error**.5,'center_counterfactual_rmse_m':energy**.5,
                'mse_improvement_vs_center':1-error/energy if energy>1e-12 else None,
                'response_gain_about_center':float((x*y).mean()/energy) if energy>1e-12 else None,
                'correlation':covariance/variance**.5 if variance>1e-16 else None}
        return {'samples':int(mask.sum()),'position_rmse_m':mse**.5,
            'center_counterfactual_rmse_m':ref**.5,'mse_improvement_vs_center':1-mse/ref if ref>1e-12 else None,
            'axes':axes,'counterfactual_kind':'mathematical fixed center evaluated on identical valid target samples; not a robot rollout'}
    def direction(mask,axis,sign):
        if not mask.any():return {'samples':0}
        c=base_command[...,axis][mask];a=base_actual[...,axis][mask]
        energy=float(np.mean(c*c));mse=float(np.mean((a-c)**2))
        return {'samples':int(mask.sum()),'command_mean':float(c.mean()),'actual_mean':float(a.mean()),
            'rmse':mse**.5,'gain':float(np.mean(a*c)/energy) if energy>1e-12 else None,
            'correct_sign_fraction':float(np.mean(a*sign>0)),
            'zero_velocity_counterfactual_rmse':energy**.5,
            'mse_improvement_vs_zero':1-mse/energy if energy>1e-12 else None,
            'counterfactual_kind':'mathematical zero velocity; not a robot rollout'}
    output={'schema':1,'scope':'valid first-episode pre-reset samples; failure samples retained','windows':{}}
    for window,mask in [('all',valid),('after_1s',valid&(data['age_steps']*dt>1.))]:
        entry={'ee':ee(mask),'directions':{}}
        if family=='T1':
            for g,name in enumerate(('forward','backward','left','right')):
                axis=0 if g<2 else 2;sign=1 if g in (0,2) else -1
                desired=sign*(params['base_linear_speed'] if g<2 else params['base_yaw_rate'])
                selection=mask&(desired_group==g)
                plateau=selection&(np.abs(base_command[...,axis]-desired)<1e-5)
                entry['directions'][name]={'including_transition':direction(selection,axis,sign),
                    'command_plateau':direction(plateau,axis,sign),'unit':'m/s' if g<2 else 'rad/s'}
        output['windows'][window]=entry
    return output
