"""One immutable checkpoint, same latest T1/T0 protocol; optional uncut video."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import traceback

os.environ.pop('NATIVE_WBC_DIAGNOSTICS',None)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--out',required=True)
parser.add_argument('--family',choices=['T0','T1'],default='T1')
parser.add_argument('--seed',type=int,default=7101)
parser.add_argument('--num_envs',type=int,default=64)
parser.add_argument('--video',action='store_true')
parser.add_argument('--stochastic',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.steps=0;args.mode='eval';args.terrain='plane'
if args.video:
    if args.num_envs!=4:parser.error('Video uses all four initial groups in a four-environment scene')
    args.enable_cameras=True
out=Path(args.out)
if out.exists():parser.error('Refusing existing output directory')
if not Path(args.checkpoint).is_file():parser.error('Checkpoint does not exist')
app=AppLauncher(args).app

import json
import numpy as np
import torch
import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks
from task_suite.config import build_cfg
from task_identity import task_identity
from telemetry import plain,write_json,sha,snapshot,model_sha
from scratch_observer import InterfaceObserver
from scratch_evaluation_core import evaluate
from t1_tracking_metrics import tracking_metrics


def main():
    out.mkdir(parents=True,exist_ok=False)
    cfg=build_cfg(args.family,num_envs=args.num_envs,seed=args.seed,ee_mode='smooth',evaluation=True)
    cfg.sim.device=args.device or 'cuda:0';cfg.log_dir=str(out)
    if args.video:
        cfg.viewer.resolution=(1280,720)
        cfg.viewer.eye=(6.,6.,5.);cfg.viewer.lookat=(0.,0.,.25)
        cfg.viewer.origin_type='world'
    agent=load_cfg_from_registry('GO2-PIPER-WBC','rsl_rl_cfg_entry_point')
    agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
    agent.seed=args.seed;agent.device=cfg.sim.device;agent.resume=False
    dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
    source=Path(__file__).resolve().parents[1]
    native_root=Path(LeggedManip_Lab.__file__).resolve().parent
    bindings={n:sha(source/n) for n in ['task_suite/config.py','task_suite/mdp.py','task_suite/profiles.json',
        'tools/evaluate_t1_scratch.py','tools/scratch_evaluation_core.py','tools/t1_tracking_metrics.py','tools/telemetry.py','tools/scratch_observer.py']}
    key,spec=task_identity(plain(cfg.task_suite))
    protocol=dict(task=spec,admission_key=key,checkpoint=args.checkpoint,checkpoint_sha256=sha(args.checkpoint),
        mode='eval',seed=args.seed,stochastic=args.stochastic,num_envs=args.num_envs,version='t1_scratch_comparison_v1',
        observation_noise=cfg.observations.policy.enable_corruption,source_hashes=bindings,
        horizon_s=20,control_dt=.02,initial_groups='env_id modulo 4: forward/backward/left/right',
        historical_metrics='unchanged thresholds from task suite; weak 5cm/yaw gates are not proof of tracking',
        additional_metrics='direction response gain/RMSE/sign; per-axis EE and MSE improvement versus mathematical center holding',
        video=dict(enabled=args.video,uncut=True,first_episode_only_metrics=True,
            postreset_visuals='shown with explicit ENDED/RESET label; excluded from metrics'),
        evaluator_sha256=sha(__file__),telemetry_sha256=sha(source/'tools/telemetry.py'))
    write_json(out/'protocol.json',protocol)
    env=RslRlVecEnvWrapper(gym.make('GO2-PIPER-WBC',cfg=cfg,render_mode='rgb_array' if args.video else None),clip_actions=agent.clip_actions)
    raw=env.unwrapped
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=agent.device)
    runner.load(args.checkpoint,load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),strict=True,map_location=agent.device)
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    equality={name:all(torch.equal(v.cpu(),cp[name+'_state_dict'][k]) for k,v in getattr(runner.alg,name).state_dict().items()) for name in ('actor','critic')}
    assert all(equality.values())
    del cp
    write_json(out/'transfer.json',dict(strict_load=True,bitwise_equal=equality,optimizer_state_entries=len(runner.alg.optimizer.state),
        actor_sha256=model_sha(runner.alg.actor),critic_sha256=model_sha(runner.alg.critic)))
    write_json(out/'initial_runtime.json',snapshot(raw,runner))
    from audit_geometry_runtime import audit
    audit(raw,out)
    interface=InterfaceObserver(env)
    writer=None;frame_count=0
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw,ImageFont
        writer=imageio.get_writer(str(out/'full_rollout.mp4'),fps=50,codec='libx264',quality=7,
            macro_block_size=2,ffmpeg_params=['-pix_fmt','yuv420p'])
        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        except OSError:font=ImageFont.load_default()
        origins=raw.scene.env_origins.detach().cpu().numpy()
        center=origins.mean(0);center[2]=.25
        raw.sim.set_camera_view(eye=tuple(center+np.array([6,6,4.75])),target=tuple(center))
        def frame(index,live,values):
            nonlocal frame_count
            pixels=raw.render(recompute=True)
            image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,1280,145),fill=(12,18,24))
            draw.text((12,8),f'{args.family} easy/smooth | {Path(args.checkpoint).name} | seed {args.seed} | t={(index+1)*.02:.2f}s',font=font,fill='white')
            live_np=live.detach().cpu().numpy()
            for i,name in enumerate(('forward','backward','left','right')):
                c=values['base_cmd'][i];a=values['base_actual'][i]
                error=values['ee_position_error'][i]
                status='FIRST EPISODE' if live_np[i] else 'ENDED / RESET (excluded from metrics)'
                draw.text((12,35+24*i),f'env{i} initial={name:8s} | vx {a[0]:+.3f}/{c[0]:+.3f} | yaw {a[2]:+.3f}/{c[2]:+.3f} | EE {error:.3f}m | {status}',font=font,fill=(225,235,245))
            writer.append_data(np.asarray(image));frame_count+=1
        callback=frame
    else:callback=None
    try:
        data,summary=evaluate(env,runner,out,protocol,args,frame_callback=callback,input_callback=interface.capture_input)
        write_json(out/'tracking_metrics.json',tracking_metrics(data,cfg.task_suite['params'],family=args.family))
        write_json(out/'interface.json',interface.report())
    finally:
        if writer is not None:writer.close()
        interface.close();env.close()
    if args.video:
        write_json(out/'video.json',dict(path=str(out/'full_rollout.mp4'),frames=frame_count,fps=50,
            duration_s=frame_count/50,sha256=sha(out/'full_rollout.mp4'),bytes=(out/'full_rollout.mp4').stat().st_size,
            uncut=True,real_simulator_render=True,postreset_frames_labelled=True))
    write_json(out/'manifest.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file()})
    print('SCRATCH_EVAL_COMPLETE',str(out),flush=True)


try:main()
except BaseException as exc:
    if out.exists():write_json(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    raise
finally:app.close()
