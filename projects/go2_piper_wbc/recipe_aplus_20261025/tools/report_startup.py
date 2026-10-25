"""Full20s frozen CP1200 safety and first-target jumps; no excluded startup window."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from report_recipe_a import capability


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1];v1=json.loads((source/'recipe_v1/spec.json').read_text());spec=json.loads((source/'recipe_aplus/spec.json').read_text())
    state=json.loads((root/'startup_study.json').read_text());assert state['state']=='complete' and len(state['stages'])==15
    rows=[]
    for item in state['stages']:
        r=capability(item,v1['A_to_B_gate']);r['tau_s']=item['tau_s'];d=Path(item['out']);z=np.load(d/'trajectories.npz');valid=z['valid'];over=valid&(z['joint_velocity_limit_ratio']>1.0001)
        r.update(overspeed_episodes=int(over.any(0).sum()),strict_safe_episodes=r['legacy_safe_count'],overspeed_fraction=float(over.sum()/valid.sum()),overspeed_duration_mean_s=float(over.sum(0).mean()*.02),overspeed_duration_max_s=float(over.sum(0).max()*.02),velocity_ratio_max=float(z['joint_velocity_limit_ratio'][valid].max()),target_limit_excess_max_rad=float(z['control_target_limit_excess'][valid].max()),requested_jump_first_rad_mean=np.abs(z['startup_requested_target'][0]-z['startup_initial_actual_q'][0]).mean(0).tolist(),executed_jump_first_rad_mean=np.abs(z['control_processed_target'][0]-z['startup_initial_actual_q'][0]).mean(0).tolist(),actual_initial_q_mean_rad=z['startup_initial_actual_q'][0].mean(0).tolist())
        rows.append(r)
    candidates=[];g=spec['startup_study']['candidate_gate']
    for tau in [.1,.2,.3,.5]:
        rr=[r for r in rows if r['tau_s']==tau];checks=[]
        for r in rr:
            base=next(b for b in rows if b['tau_s']==0 and b['seed']==r['seed'])
            c=dict(overspeed_episode_reduction=r['overspeed_episodes']<=base['overspeed_episodes']*(1-g['overspeed_episode_reduction_min']),completion=r['complete_count']>=base['complete_count'],EE=r['ee_rmse_m']<=base['ee_rmse_m']+g['ee_after1s_rmse_increase_max_m'],target_bounds=r['target_limit_excess_max_rad']<=g['processed_target_excess_max_rad'],actual_hard_limits=r['actual_joint_hard_limit_violation_fraction']<=g['hard_joint_violation_fraction_max'])
            for direction in ['forward','backward','left','right']:c[direction]=abs(r['directions'][direction]['gain']-base['directions'][direction]['gain'])<=g['direction_gain_absolute_change_max']
            checks.append(dict(seed=r['seed'],checks=c,passed=all(c.values())))
        candidates.append(dict(tau_s=tau,passed=all(c['passed'] for c in checks),seeds=checks))
    selected=next((c['tau_s'] for c in candidates if c['passed']),None)
    decision=dict(candidate_tau_s=selected,candidates=candidates,applied_to_Aplus=False,full_strict_safety_unchanged=True,requires_frozen_revalidation_on_latest_model_before_B=True)
    (out/'physical_results.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'candidate_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        keys=['tau_s','seed','complete_count','strict_safe_episodes','overspeed_episodes','overspeed_fraction','overspeed_duration_mean_s','overspeed_duration_max_s','velocity_ratio_max','ee_rmse_m','target_limit_excess_max_rad'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for tau in [0.,.1,.2,.3,.5]:
        item=next(s for s in state['stages'] if s['tau_s']==tau and s['seed']==8101);z=np.load(Path(item['out'])/'trajectories.npz');t=np.arange(len(z['valid']))*.02;m=z['valid'][:,0]&(t<1.)
        for col,j in enumerate([13,16]):
            axs[0,col].plot(t[m],z['control_processed_target'][m,0,j],label=f'tau={tau}')
            axs[1,col].plot(t[m],z['control_joint_position'][m,0,j],label=f'tau={tau}')
            axs[2,col].plot(t[m],z['control_joint_velocity'][m,0,j]/5.,label=f'tau={tau}')
    for col,name in enumerate(['joint2','joint5']):
        for row,kind in enumerate(['executed target(rad)','actual q(rad)','velocity / actual5rad/s limit']):axs[row,col].set_title(name+' '+kind);axs[row,col].grid(alpha=.25)
        axs[2,col].axhline(1.,ls='--',c='k');axs[2,col].axhline(-1.,ls='--',c='k');axs[2,col].set_xlabel('Time(s)')
    axs[0,0].legend(fontsize=8);fig.suptitle('Frozen CP1200 measured startup traces, seed8101 env0; safety metrics still cover full20s');fig.savefig(out/'startup_targets_and_velocity.png',dpi=150);plt.close(fig)
    (out/'REPORT_ZH.md').write_text('# 冻结CP1200启动过渡对照\n\n完整20秒旧严格安全门未改；没有忽略最初.3秒。候选tau='+str(selected)+'。候选未应用于A+，进入后续训练前还须对最新模型冻结复核。\n\n逐seed安全数、超速时长/峰值见summary.csv，所有首目标跳变见physical_results.json。joint2默认0与最小安全目标.177rad的差异仍计入首次执行跳变。\n')
    print(json.dumps(decision))


if __name__=='__main__':main()
