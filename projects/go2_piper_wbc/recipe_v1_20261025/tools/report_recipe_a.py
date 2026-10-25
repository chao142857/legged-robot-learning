"""CPU-only Recipe A physical capability report; never substitutes reward for gates."""
from publication_paths import research_path
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

DIRECTIONS=('forward','backward','left','right')
NODES=(0,200,500,800,1200)


def read(path):
    return json.loads(Path(path).read_text())


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def stat(x,operation='mean'):
    a=np.asarray(x)
    if not a.size:return None
    assert np.isfinite(a).all(),'Nonfinite evaluation data'
    if operation=='p05':return float(np.quantile(a,.05))
    if operation=='p95':return float(np.quantile(a,.95))
    if operation=='median':return float(np.median(a))
    return float(a.mean())


def capability(item,gate):
    directory=Path(item['out']);summary=read(directory/'summary.json')
    tracking=read(directory/'tracking_metrics.json')['windows']['after_1s']
    z=np.load(directory/'trajectories.npz',allow_pickle=False)
    valid=z['valid'].astype(bool);mask=valid&(z['age_steps']*.02>=1.)
    count=summary['num_episodes']; assert count==64
    if item['protocol'] in ('A','T1'):
        assert set(DIRECTIONS)<=set(tracking['directions']),'Missing movement directions in A/T1 evaluation'
    empty_direction=dict(samples=0,command_mean=None,actual_mean=None,rmse=None,gain=None,correct_sign_fraction=None)
    directions={d:tracking['directions'][d]['command_plateau'] if d in tracking['directions'] else dict(empty_direction) for d in DIRECTIONS}
    r={k:item[k] for k in ('label','protocol','seed','out')}
    r.update(complete_count=summary['complete_count'],episodes=count,
        completion_fraction=summary['complete_count']/count,
        legacy_safe_count=summary['safe_count'],legacy_contact_safe_count=summary['contact_safe_count'],
        legacy_overspeed_count=summary['overspeed_count'],
        terminations=summary['terminations'],nonfoot_failure_fraction=summary['terminations'].get('nonfoot_contact',0)/count,
        ee_rmse_m=tracking['ee']['position_rmse_m'],
        mathematical_center_mse_improvement=tracking['ee']['mse_improvement_vs_center'],
        ee_axes=tracking['ee']['axes'],
        directions=directions,
        valid_after_1s_samples=int(mask.sum()),
        low_height_time_fraction=stat(z['low_height'][mask]),
        low_height_duration_s_mean=float((z['low_height']&valid).sum()/count*.02),
        low_height_duration_s_max=float((z['low_height']&valid).sum(0).max()*.02),
        base_clearance_p05_m=stat(z['base_clearance'][mask],'p05'),
        base_clearance_median_m=stat(z['base_clearance'][mask],'median'),
        body_abs_roll_pitch_p95_rad=stat(np.maximum(np.abs(z['base_roll'][mask]),np.abs(z['base_pitch'][mask])),'p95'),
        arm_pd_abs_error_mean_rad=stat(np.abs(z['control_pd_target_error'][mask][:,12:])),
        actual_joint_hard_limit_violation_fraction=stat(z['control_hard_limit_excess'][mask]>.001),
        target_clipping_element_fraction=stat(z['control_target_was_clipped'][mask]),
        actuator_saturation_element_fraction=stat(z['control_actuator_saturation_substep_fraction'][mask]),
        action_delta_mse=summary['action_delta_mse'],
        actor_unchanged=summary['actor_unchanged'],critic_unchanged=summary['critic_unchanged'])
    assert r['actor_unchanged'] and r['critic_unchanged']
    assert np.all(z['control_actuator_substep_count'][valid]==4),'Incomplete actuator monitoring'
    rates=z['reward_terms'][valid]
    r['reward_rate_mean']={name:float(rates[:,i].mean()) for i,name in enumerate(summary['reward_names'])}
    r['reward_return_mean']={name:float(rates[:,i].sum()*.02/count) for i,name in enumerate(summary['reward_names'])}
    checks={}
    def le(name,value,limit):checks[name]=value is not None and value<=limit
    def ge(name,value,limit):checks[name]=value is not None and value>=limit
    for d,m in r['directions'].items():
        checks[d+'/coverage']=m['samples']>0
        ge(d+'/gain_min',m['gain'],gate['direction_gain_min'])
        le(d+'/gain_max',m['gain'],gate['direction_gain_max'])
        ge(d+'/correct_sign',m['correct_sign_fraction'],gate['direction_correct_sign_min'])
        le(d+'/rmse',m['rmse'],gate['linear_rmse_max_mps'] if d in DIRECTIONS[:2] else gate['yaw_rmse_max_radps'])
    le('ee_rmse',r['ee_rmse_m'],gate['ee_after_1s_rmse_max_m'])
    ge('completion',r['completion_fraction'],gate['complete_fraction_min'])
    for name,key in [('low_height_time_fraction','low_height_time_fraction_max'),('nonfoot_failure_fraction','nonfoot_failure_fraction_max'),
        ('actual_joint_hard_limit_violation_fraction','actual_joint_hard_limit_violation_fraction_max'),
        ('target_clipping_element_fraction','target_clipping_element_fraction_max'),
        ('actuator_saturation_element_fraction','actuator_saturation_element_fraction_max'),
        ('body_abs_roll_pitch_p95_rad','body_abs_roll_pitch_p95_max_rad'),
        ('arm_pd_abs_error_mean_rad','arm_pd_abs_error_mean_max_rad')]:le(name,r[name],gate[key])
    ge('base_clearance_p05_m',r['base_clearance_p05_m'],gate['base_clearance_p05_min_m'])
    ge('base_clearance_median_m',r['base_clearance_median_m'],gate['base_clearance_median_min_m'])
    r['checks']=checks;r['gate_pass']=all(checks.values()) if item['protocol']!='T0' else None
    r['failed_checks']=[k for k,v in checks.items() if not v] if item['protocol']!='T0' else []
    r['trajectory_sha256']=digest(directory/'trajectories.npz')
    return r


def validate_training(root):
    train=root/'pilot_A_seed21';summary=read(train/'summary.json');init=read(train/'initialization.json')
    assert summary['completed'] and summary['completed_updates']==1200 and summary['environment_steps']==117964800
    assert summary['load_calls']==init['load_calls']==0 and init['optimizer_state_entries']==0
    assert init['action_dim']==18 and init['observation_dims']=={'policy':[210],'critic':[102]}
    candidates=list(train.glob('*.jsonl'))
    logs=[]
    for path in candidates:
        with path.open() as f:
            first=f.readline()
        if first and 'completed_updates' in json.loads(first):logs.append(path)
    assert len(logs)==1,logs
    rows=[json.loads(line) for line in logs[0].read_text().splitlines() if line]
    assert [r['completed_updates'] for r in rows]==list(range(1,1201))
    assert all(r['env_steps']==r['completed_updates']*4096*24 for r in rows)
    audits=[]
    for n in NODES:
        cp=train/f'cp_{n}.pt';audit=read(train/f'cp_{n}_cpu_audit.json')
        assert audit['finite'] and audit['completed_updates']==n and audit['sha256']==digest(cp)
        assert audit['adam_step_min']==audit['adam_step_max']==n*20
        audits.append(audit)
    native=Path(research_path('go2_piper_upstream_wbc_20261023/formal_seed21/cp_3000.pt'))
    assert digest(native)=='600a2dab9caf18289bb6ec558f86c4e6e86c352b621f9a6d83ff73f209cc08b6'
    return rows,dict(completed_updates=1200,transitions=117964800,load_calls=0,
        checkpoint_audits=audits,continuous_log=str(logs[0]),log_sha256=digest(logs[0]),
        native_checkpoint_sha256=digest(native),initialization=init['initialization'])


def make_plots(out,training,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    xs=np.array([r['completed_updates'] for r in training])
    def smooth(a):
        a=np.array(a,dtype=float);n=20
        return np.convolve(a,np.ones(n)/n,'valid') if len(a)>=n else a
    fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    fields=[('kl_mean','KL mean'),('learning_rate','Adaptive learning rate'),
        ('actor_grad_norm_mean','Actor pre-clip gradient norm'),('critic_grad_norm_mean','Critic pre-clip gradient norm')]
    for ax,(key,title) in zip(axes.flat,fields):
        a=[r[key] for r in training];ax.plot(xs,a,alpha=.25);ax.plot(xs[len(xs)-len(smooth(a)):],smooth(a));ax.set_title(title)
    ax=axes[2,0]
    for key in ('low_height_fraction','target_clipping_fraction','actuator_saturation_fraction'):
        a=[r['metrics'].get(key,np.nan) for r in training];ax.plot(xs[len(xs)-len(smooth(a)):],smooth(a),label=key)
    ax.set_title('Training physical fractions');ax.legend(fontsize=7)
    ax=axes[2,1];std=np.array([r['action_std'] for r in training]);ax.plot(xs,std.mean(1),label='mean std');ax.fill_between(xs,std.min(1),std.max(1),alpha=.2);ax.set_title('Action distribution std')
    for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('Completed updates')
    fig.savefig(out/'ppo_diagnostics.png',dpi=150);plt.close(fig)
    for protocol in ('A','T1'):
        selected=[r for r in results if r['protocol']==protocol]
        fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
        series=[(d+' gain',lambda r,d=d:r['directions'][d]['gain']) for d in DIRECTIONS]
        series.extend([('EE RMSE (m)',lambda r:r['ee_rmse_m']),('20s completion',lambda r:r['completion_fraction'])])
        for ax,(title,fn) in zip(axes.flat,series):
            for seed in (8101,8102,8103):
                rr=[next(r for r in selected if r['label']==f'cp_{n}' and r['seed']==seed) for n in NODES]
                ax.plot(NODES,[fn(r) for r in rr],'-o',label=f'seed {seed}')
            native=[fn(r) for r in selected if r['label']=='native_cp3000'];native=[v for v in native if v is not None]
            if native:ax.axhline(np.mean(native),color='k',ls='--',label='Native CP3000 mean')
            ax.set_title(title);ax.set_xlabel('Completed updates');ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=7);fig.suptitle(f'Frozen {protocol} physical evaluation (3 seeds)')
        fig.savefig(out/f'capability_{protocol}.png',dpi=150);plt.close(fig)
    # Actual first-episode traces. Each direction gets its own fixed environment.
    for label in ('cp_1200','native_cp3000'):
        for protocol in ('A','T1','T0'):
            row=next(r for r in results if r['label']==label and r['protocol']==protocol and r['seed']==8101)
            z=np.load(Path(row['out'])/'trajectories.npz',allow_pickle=False);t=np.arange(len(z['valid']))*.02
            fig,axes=plt.subplots(5,4,figsize=(16,12),constrained_layout=True)
            for e in range(4):
                m=z['valid'][:,e];tt=t[m]
                for axid,axis in enumerate((0,2)):
                    axes[axid,e].plot(tt,z['base_cmd'][m,e,axis],label='target');axes[axid,e].plot(tt,z['base_actual'][m,e,axis],label='actual')
                    axes[axid,e].set_ylabel(('vx (m/s)','yaw (rad/s)')[axid])
                for j,axis in enumerate('xyz'):
                    axes[j+2,e].plot(tt,z['ee_cmd'][m,e,j],label='target');axes[j+2,e].plot(tt,z['ee_actual_mixed'][m,e,j],label='actual');axes[j+2,e].set_ylabel(f'EE {axis} (m)')
                axes[0,e].set_title(f'env {e}, initial {DIRECTIONS[e]}, {tt[-1]+.02:.2f}s valid')
                for ax in axes[:,e]:ax.grid(alpha=.25);ax.set_xlim(0,20)
                axes[-1,e].set_xlabel('First-episode time (s)')
            axes[0,0].legend(fontsize=7);fig.suptitle(f'{label} {protocol}: measured headless physics, seed8101; no post-reset samples')
            fig.savefig(out/f'trajectory_{label}_{protocol}.png',dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    spec=read(Path(__file__).resolve().parents[1]/'recipe_v1/spec.json');gate=spec['A_to_B_gate']
    training,verified=validate_training(root)
    stages=read(root/'posteval_status.json');assert stages['state']=='complete' and len(stages['stages'])==42
    assert all(item['state']=='complete' for item in stages['stages'])
    results=[capability(item,gate) for item in stages['stages']]
    nodes={n:all(r['gate_pass'] for r in results if r['label']==f'cp_{n}' and r['protocol'] in ('A','T1')) for n in NODES}
    passed_pairs=[list(pair) for pair in zip(NODES[:-1],NODES[1:]) if all(nodes[n] for n in pair)]
    decision=dict(frozen_gate=gate,node_pass=nodes,passed_consecutive_pairs=passed_pairs,
        evidence_for_B=bool(passed_pairs),automatic_B_C_started=False,training_seeds=1,
        reproducibility_proven=False,rule='all 3 evaluation seeds and both A/T1, at two consecutive saved nodes; missing directions fail')
    write(out/'gate_decision.json',decision);write(out/'physical_results.json',results);write(out/'training_verified.json',verified)
    flat=[]
    for r in results:
        row={k:r[k] for k in ('label','protocol','seed','complete_count','episodes','ee_rmse_m','completion_fraction',
            'legacy_safe_count','low_height_time_fraction','low_height_duration_s_mean','low_height_duration_s_max',
            'base_clearance_p05_m','base_clearance_median_m','body_abs_roll_pitch_p95_rad','arm_pd_abs_error_mean_rad',
            'target_clipping_element_fraction','actuator_saturation_element_fraction','actual_joint_hard_limit_violation_fraction','gate_pass')}
        for d in DIRECTIONS:
            for k in ('gain','rmse','correct_sign_fraction','samples'):row[d+'_'+k]=r['directions'][d][k]
        row['failed_checks']=';'.join(r['failed_checks']);flat.append(row)
    with (out/'physical_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    make_plots(out,training,results)
    text=['# Recipe V1 阶段A：物理能力验收','',
        '随机初始化pilot达到1200 updates（117,964,800 transitions）。以下结论由冻结物理轨迹及预先保存的门槛产生；reward上涨不能替代能力验收。',
        '',f"是否具备进入B的证据：**{'具备预设门槛证据' if passed_pairs else '尚不具备'}**。本轮未启动B/C。单训练seed不能证明跨seed复现。",'',
        '|协议/策略|20秒完成|EE RMSE均值(cm)|前进gain|后退gain|左转gain|右转gain|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for protocol in ('A','T1','T0'):
        for label in [f'cp_{n}' for n in NODES]+['native_cp3000']:
            rr=[r for r in results if r['protocol']==protocol and r['label']==label]
            if not rr:continue
            gains=[np.mean([r['directions'][d]['gain'] for r in rr if r['directions'][d]['gain'] is not None]) if any(r['directions'][d]['gain'] is not None for r in rr) else None for d in DIRECTIONS]
            text.append('|'+protocol+'/'+label+'|'+str(sum(r['complete_count'] for r in rr))+'/'+str(sum(r['episodes'] for r in rr))+'|'+('--' if not any(r['ee_rmse_m'] is not None for r in rr) else f"{np.mean([r['ee_rmse_m'] for r in rr if r['ee_rmse_m'] is not None])*100:.2f}")+'|'+'|'.join('--' if v is None else f'{v:.3f}' for v in gains)+'|')
    text.extend(['','连续节点门槛：'+json.dumps(nodes,ensure_ascii=False),
        '', '逐seed失败项、低趴时长、目标裁剪、四substep饱和、接触、原始安全门和reward分项见physical_results.json/CSV。T0无方向覆盖，不能用来宣布移动能力。',
        '', '轨迹图为真实headless物理测量，不是RTX视频。数学中心保持与零速度对照只存在于指标计算，均非机器人rollout。',
        '', 'PPO、reward和任务在1200更新过程中均未自动改变。CP0/200/500/800/1200、逐步NPZ留在执行主机，见delivery_manifest.json。'])
    (out/'REPORT_ZH.md').write_text('\n'.join(text)+'\n')
    manifest=dict(root=str(root),source=str(Path(__file__).resolve().parents[1]),training=verified,
        raw_trajectories=[dict(path=str(Path(r['out'])/'trajectories.npz'),sha256=r['trajectory_sha256'],label=r['label'],protocol=r['protocol'],seed=r['seed']) for r in results],
        light_files=[dict(path=str(x),bytes=x.stat().st_size,sha256=digest(x)) for x in out.iterdir() if x.is_file()])
    write(out/'delivery_manifest.json',manifest)
    print(json.dumps(dict(report=str(out),evidence_for_B=bool(passed_pairs),nodes=nodes)))


if __name__=='__main__':main()
