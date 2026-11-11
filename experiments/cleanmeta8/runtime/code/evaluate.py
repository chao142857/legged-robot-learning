"""Isolated frozen evaluations. Official test metrics never reach the teacher."""
import argparse
import json
from pathlib import Path
import time
import traceback
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--kind",choices=("development","test"),required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app


def evaluate():
    import gymnasium as gym
    import numpy as np
    import torch
    import LeggedManip_Lab.tasks
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from adapter import SequenceAdapter
    from commands import SequenceVelocity,SequenceWBC
    from common import ROOT,PROTOCOL,Pool,digest,file_sha,model_cfg,tensor_hash,write
    from neural_core import development_score
    from protocol import CONFIG,TRAIN_POOL_SHA,TEST_POOL_SHA

    start=time.monotonic()
    manifest=json.loads((ROOT/"prepared/manifest.json").read_text())
    if args.kind=="development":
        path=ROOT/"prepared/wbc_development.npz"
        data=np.load(path,allow_pickle=False)
        pool=Pool.__new__(Pool)
        pool.kind,pool.split,pool.path="wbc","development",path
        pool.sha=file_sha(path)
        assert pool.sha==manifest["binding"]["development_pool_sha256"]
        selected=np.tile(np.arange(256),2)
        pool.ee,pool.velocity=data["ee"][selected],data["velocity"][selected]
        pool.n=len(selected)
        pool.identity="wbc:development:"+pool.sha+":two-fixed-resets"
        seed=CONFIG["dev_reset_seed"]
        n_sequences=256
    else:
        pool=Pool("wbc","evaluation")
        assert pool.sha==TEST_POOL_SHA and pool.n==1024
        selected=np.arange(1024)
        seed=CONFIG["test_seed"]
        n_sequences=1024
    cfg,agent=model_cfg("GO2-PIPER-WBC")
    cfg.scene.num_envs,cfg.seed,cfg.sim.device=pool.n,seed,"cuda:0"
    cfg.commands.ee_pose.class_type=SequenceWBC
    cfg.commands.base_velocity.class_type=SequenceVelocity
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark=False
    cp_sha=file_sha(args.checkpoint)
    cp=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    assert cp["infos"]["pool_sha256"]==TRAIN_POOL_SHA
    assert cp["infos"]["protocol_sha256"]==digest(PROTOCOL)
    update=cp["infos"]["completed_updates"]
    env=gym.make("GO2-PIPER-WBC",cfg=cfg)
    try:
        raw=env.unwrapped
        adapter=SequenceAdapter(raw,"wbc","evaluation",pool,args.out)
        # Observe strict-good streaks at exactly the same pre-reset capture point.
        current=torch.zeros(pool.n,device=raw.device,dtype=torch.long)
        longest=torch.zeros_like(current)
        first_longest={}
        safety_steps=torch.zeros(pool.n,device=raw.device,dtype=torch.long)
        limit_steps=torch.zeros_like(safety_steps)
        tilt_steps=torch.zeros_like(safety_steps)
        max_limit=torch.zeros(pool.n,device=raw.device)
        max_tilt=torch.zeros_like(max_limit)
        first_safety={}
        original_capture,original_finish,original_reset=adapter.metrics.capture,adapter.metrics.finish,adapter.metrics.reset
        def capture():
            robot=raw.scene['robot'].data
            limits=robot.soft_joint_pos_limits
            excess=torch.maximum(limits[...,0]-robot.joint_pos,robot.joint_pos-limits[...,1]).clamp_min(0).amax(-1)
            tilt=torch.acos((-robot.projected_gravity_b[:,2]).clamp(-1,1))
            safety_steps.add_(1)
            limit_steps.add_((excess>1e-6).long())
            tilt_steps.add_((tilt>np.pi/6).long())
            max_limit.copy_(torch.maximum(max_limit,excess))
            max_tilt.copy_(torch.maximum(max_tilt,tilt))
            before=adapter.metrics.good.clone()
            original_capture()
            good=adapter.metrics.good>before
            current.copy_(torch.where(good,current+1,0))
            longest.copy_(torch.maximum(longest,current))
        def finish(ids):
            for env_id,steps in zip(ids.cpu().tolist(),longest[ids].cpu().tolist()):
                if adapter.owner.generation[env_id]==1:
                    first_longest[env_id]=steps
                    count=int(safety_steps[env_id])
                    first_safety[env_id]=dict(control_steps=count,
                        joint_soft_limit_step_fraction=float(limit_steps[env_id])/max(1,count),
                        max_joint_soft_limit_excess_rad=float(max_limit[env_id]),
                        body_tilt_over_30deg_step_fraction=float(tilt_steps[env_id])/max(1,count),
                        max_body_tilt_rad=float(max_tilt[env_id]))
            return original_finish(ids)
        def reset(ids):
            original_reset(ids)
            current[ids]=0
            longest[ids]=0
            for value in (safety_steps,limit_steps,tilt_steps,max_limit,max_tilt): value[ids]=0
        adapter.metrics.capture,adapter.metrics.finish,adapter.metrics.reset=capture,finish,reset
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        assert wrapped.num_actions==18 and raw.max_episode_length==1000
        runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=agent.device)
        actor=runner.alg.actor
        actor.load_state_dict(cp["actor_state_dict"],strict=True)
        actor_sha=tensor_hash(actor.state_dict())
        del cp
        policy=runner.get_inference_policy(device=raw.device)
        obs=wrapped.get_observations()
        initial_sha=tensor_hash(dict(robot=raw.scene["robot"].data.root_state_w,
            joints=raw.scene["robot"].data.joint_pos,**{f"obs/{k}":v for k,v in obs.items()}))
        with torch.inference_mode():
            for step in range(1,raw.max_episode_length+3):
                obs,_,dones,_=wrapped.step(policy(obs))
                policy.reset(dones)
                if len(adapter.first_episode_rows)==pool.n:
                    break
        adapter.owner.check()
        assert adapter.owner.accepted==0
        rows=sorted(adapter.first_episode_rows,key=lambda r:r["env_id"])
        assert len(rows)==pool.n
        for i,row in enumerate(rows):
            assert row["env_id"]==row["sequence_id"]==i
            row.update(sequence_id=int(selected[i]),replicate=i//n_sequences,
                       longest_joint_good_seconds=first_longest[i]*raw.step_dt,
                       safety=first_safety[i])
        values=np.array([r["values"] for r in rows])
        n=values[:,0]
        score=development_score(rows)
        rmse=np.sqrt((values[:,2:6]**2*n[:,None]).sum(0)/max(1,n.sum()))
        assert file_sha(args.checkpoint)==cp_sha and tensor_hash(actor.state_dict())==actor_sha
        write(args.out/"episode_results.json",rows)
        result=dict(passed=True,kind=args.kind,update=update,checkpoint_sha256=cp_sha,actor_sha256=actor_sha,
            pool_sha256=pool.sha,initial_sha256=initial_sha,seed=seed,episodes=pool.n,
            protocol_sha256=digest(PROTOCOL),training_history_updates=0,model_weights_unchanged=True,
            joint_success_rate=float(values[:,6].mean()),native_failure_rate=float(values[:,7].mean()),
            no_valid_window_rate=float((n<100).mean()),continuous_joint_score=float(score.mean()),
            longest_joint_good_seconds_mean=float(np.mean([r["longest_joint_good_seconds"] for r in rows])),
            loco_success_rate=float(((n>=100)&(values[:,7]==0)&(values[:,2]<=.2)&(values[:,3]<=.3)).mean()),
            manipulation_rmse_pass_rate=float(((n>=100)&(values[:,7]==0)&(values[:,4]<=.05)&(values[:,5]<=np.pi/12)).mean()),
            pooled_rmse=dict(zip(("velocity_m_s","yaw_rad_s","ee_position_m","ee_orientation_rad"),rmse.tolist())),
            scores=score.tolist(),wall_seconds=time.monotonic()-start,
            executed_environment_transitions=step*pool.n,first_episode_transitions=sum(r["steps"] for r in rows),
            episode_results_sha256=file_sha(args.out/"episode_results.json"))
        result['safety']=dict(native_failure_rate=result['native_failure_rate'],
            joint_soft_limit_step_fraction_mean=float(np.mean([r['safety']['joint_soft_limit_step_fraction'] for r in rows])),
            body_tilt_over_30deg_step_fraction_mean=float(np.mean([r['safety']['body_tilt_over_30deg_step_fraction'] for r in rows])),
            max_joint_soft_limit_excess_rad=float(max(r['safety']['max_joint_soft_limit_excess_rad'] for r in rows)),
            max_body_tilt_rad=float(max(r['safety']['max_body_tilt_rad'] for r in rows)),
            scope='Read-only pre-reset monitoring; does not change native termination, reward or success criteria.')
        if args.kind=="development":
            pair=score.reshape(2,256)
            result.update(reset_repeat_mean_gap=float(abs(pair[0]-pair[1]).mean()),
                          sequence_mean_standard_error=float(pair.mean(0).std(ddof=1)/np.sqrt(256)))
        write(args.out/"result.json",result)
        print("NSGS_EVAL_PASS "+json.dumps({k:v for k,v in result.items() if k!="scores"}),flush=True)
    finally:
        env.close()


args.out.mkdir(parents=True,exist_ok=False)
try:
    evaluate()
except BaseException:
    (args.out/"fault.txt").write_text(traceback.format_exc())
    raise
finally:
    app.close()
