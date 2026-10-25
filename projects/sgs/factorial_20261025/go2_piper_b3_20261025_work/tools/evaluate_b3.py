"""Single prepared frozen evaluator for B2/B3. Default plan imports no simulator."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
B2=ROOT.parent/'go2_piper_b2_20261025_work'
sys.path.insert(0,str(B2));sys.path.insert(0,str(B2/'tools'));sys.path.insert(0,str(ROOT))
from b2.dependencies import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('plan','eval'),default='plan')
    parser.add_argument('--checkpoint');parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--out');parser.add_argument('--admission')
    parser.add_argument('--method',choices=('B2','B3'),default='B3')
    parser.add_argument('--family',choices=('T0','T1'),default='T1')
    parser.add_argument('--seed',type=int,default=7101);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    protocol_path=B2/'protocol/common_protocol.json';protocol=json.loads(protocol_path.read_text())
    if args.mode=='plan':
        print(json.dumps(dict(method=args.method,evaluation=protocol['evaluation'],family=args.family,
            simulation_started=False,optimizer_updates=0),indent=2));return
    for name in ('checkpoint','checkpoint_sha256','out','admission'):
        if not getattr(args,name):parser.error('--'+name.replace('_','-')+' required for future evaluation')
    if args.seed not in protocol['evaluation']['seeds']:parser.error('Use a common registered evaluation seed')
    if not Path(args.checkpoint).is_file() or Path(args.out).exists():parser.error('Require an existing checkpoint and new output directory')
    if sha(args.checkpoint)!=args.checkpoint_sha256:raise RuntimeError('Frozen checkpoint SHA256 mismatch')
    from run_b2 import admission,bind_native_checkout
    record=admission(args.admission)
    bind_native_checkout(record) # Bind independent native source before AppLauncher.
    from b2.runtime import common_inputs
    _,pool=common_inputs()
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device=args.device).app
    try:
        import torch
        import gymnasium as gym
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.io import dump_yaml
        import LeggedManip_Lab.tasks
        from b2.runtime import configure
        from task_suite.config import build_cfg
        from shared_ik.runtime import LegPPOArmIKWrapper
        from shared_ik.trace import ControlTrace
        from telemetry import write_json,physical_suite,snapshot
        from scratch_observer import InterfaceObserver
        from scratch_evaluation_core import evaluate
        from t1_tracking_metrics import tracking_metrics
        n=protocol['evaluation']['num_envs'];out=Path(args.out)
        if args.family=='T1':cfg=configure(pool,[i%4 for i in range(n)],n,args.seed,args.device,out)
        else:
            cfg=build_cfg('T0',num_envs=n,seed=args.seed,ee_mode='smooth',evaluation=True)
            cfg.sim.device=args.device;cfg.log_dir=str(out)
        agent=copy.deepcopy(protocol['recipe']);agent['device']=args.device;agent['seed']=args.seed;agent['resume']=False
        out.mkdir(parents=True,exist_ok=False)
        dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
        env=LegPPOArmIKWrapper(gym.make('GO2-PIPER-WBC',cfg=cfg),clip_actions=agent['clip_actions'])
        runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=args.device)
        runner.load(args.checkpoint,load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),strict=True,map_location=args.device)
        cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if cp.get('infos',{}).get('protocol_sha256')!=sha(protocol_path):raise RuntimeError('Checkpoint protocol mismatch')
        for name in ('actor','critic'):
            if not all(torch.equal(v.cpu(),cp[name+'_state_dict'][k]) for k,v in getattr(runner.alg,name).state_dict().items()):
                raise RuntimeError('Frozen model mismatch: '+name)
        del cp
        if len(runner.alg.optimizer.state):raise RuntimeError('Evaluation loaded optimizer state')
        args.steps=0;args.stochastic=False;args.mode='eval';args.terrain='plane';args.num_envs=n
        fixed=dict(admission_key=cfg.task_suite_signature,common_protocol_sha256=sha(protocol_path),
            support_sha256=pool.support_sha256,checkpoint_sha256=args.checkpoint_sha256,
            method=args.method,family=args.family,seed=args.seed,binding=env.binding(),
            first_episode_only=True,observation_noise=True,fixed_phase_assignment='env_id modulo4 for T1',
            training_sampler_calls=0,training_feedback_calls=0)
        write_json(out/'protocol.json',fixed);write_json(out/'initial_runtime.json',snapshot(env.unwrapped,runner))
        interface=InterfaceObserver(env)
        trace=ControlTrace(env,out/'control_chain_first4.jsonl',physical_metrics=physical_suite)
        def capture(obs):
            env.capture_input(obs);interface.capture_input(obs)
        try:
            data,summary=evaluate(env,runner,out,fixed,args,input_callback=capture)
            write_json(out/'tracking_metrics.json',tracking_metrics(data,cfg.task_suite['params'],family=args.family))
            write_json(out/'interface.json',interface.report())
        finally:
            trace.close();interface.close();env.close()
        write_json(out/'manifest.json',{str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file()})
    finally:app.close()


if __name__=='__main__':main()
