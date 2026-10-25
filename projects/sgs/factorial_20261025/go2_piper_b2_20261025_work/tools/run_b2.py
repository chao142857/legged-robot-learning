"""B2/B3 prepared entry; plan is CPU-only, train requires a bound later admission.

This round prepares this file but never invokes train or imports Isaac.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from b2.dependencies import verify,sha

def admission(path):
    r=json.loads(Path(path).read_text())
    for key in ('baseline_review_bound','phase4_and_label_bound','native_boundary_verified','formal_run_authorized','shared_ik_mapping_verified'):
        if r.get(key) is not True:raise RuntimeError('Pending common binding: '+key)
    if r['common_protocol_sha256']!=sha(ROOT/'protocol/common_protocol.json'):raise RuntimeError('Admission protocol changed')
    if r['dependency_lock_sha256']!=sha(ROOT/'protocol/dependencies.lock.json'):raise RuntimeError('Admission dependencies changed')
    for item in r.get('source_files',[]):
        if sha(item['path']) != item['sha256']:raise RuntimeError('Admission source changed: '+item['path'])
    verify();return r

def bind_native_checkout(record):
    native_root=Path(os.environ['B2_NATIVE_WORKTREE']).resolve()
    expected_commit=record['native_source_commit']
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=native_root,text=True).strip()!=expected_commit:
        raise RuntimeError('Independent native source commit changed')
    if subprocess.check_output(['git','diff','4e12109d1eccb1d9da2e4a35f189fabea3f13677','--','source','scripts'],cwd=native_root):
        raise RuntimeError('Native controller/asset source differs from common baseline')
    sys.path.insert(0,str(native_root/'source/LeggedManip_Lab'))
    os.environ.pop('NATIVE_WBC_DIAGNOSTICS',None)
    return native_root

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('plan','train'),default='plan')
    parser.add_argument('--method',choices=('B2','B3'),default='B2')
    parser.add_argument('--out')
    parser.add_argument('--run-id')
    parser.add_argument('--sampler-seed',type=int,default=9021)
    parser.add_argument('--admission')
    parser.add_argument('--pair-initialization')
    parser.add_argument('--pair-sha256')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.mode=='plan':
        p=json.loads((ROOT/'protocol/common_protocol.json').read_text())
        print(json.dumps(dict(method=args.method,mode='preparation_only',budget=p['budget'],
          required_before_training=p['pending_bindings'],simulation_started=False,optimizer_updates=0),ensure_ascii=False,indent=2));return
    for name in ('out','run_id','admission','pair_initialization','pair_sha256'):
        if not getattr(args,name):parser.error('--'+name.replace('_','-')+' required for future training')
    record=admission(args.admission) # Always before importing a simulator.
    if Path(args.out).exists():parser.error('Refuse existing training output')
    if sha(args.pair_initialization)!=args.pair_sha256:parser.error('Untrained pair SHA256 mismatch')
    if record.get('pair_sha256')!=args.pair_sha256:parser.error('Pair artifact not bound in common admission')
    bind_native_checkout(record)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device=args.device).app
    try:
        from b2.runtime import assemble_training,make_telemetry
        a=assemble_training(method=args.method,run_id=args.run_id,sampler_seed=args.sampler_seed,device=args.device,
             out=args.out,pair_path=args.pair_initialization,pair_sha256=args.pair_sha256)
        from telemetry import write_json,snapshot
        from scratch_observer import InterfaceObserver
        runner,env=a['runner'],a['env'];out=Path(args.out)
        interface=InterfaceObserver(env,runner)
        telemetry=make_telemetry(a,out)
        write_json(out/'initialization.json',dict(native=snapshot(env.unwrapped,runner),pair=a['pair'],
            binding=env.binding(),method=args.method,run_id=args.run_id,dependency_lock_sha256=sha(ROOT/'protocol/dependencies.lock.json')))
        write_json(out/'launch.json',dict(method=args.method,run_id=args.run_id,seed=21,sampler_seed=args.sampler_seed,
            protocol_sha256=sha(ROOT/'protocol/common_protocol.json'),pair_sha256=args.pair_sha256,budget=a['protocol']['budget'],
            support_version=a['pool'].support_version,support_sha256=a['pool'].support_sha256))
        try:
            telemetry.save('cp_0.pt')
            runner.learn(num_learning_iterations=a['protocol']['budget']['updates'],init_at_random_ep_len=False)
            if telemetry.update_count!=5000 or sum(a['session'].transitions)!=491520000:raise RuntimeError('Budget mismatch')
            for row in a['session'].censor_active('training_cutoff'):a['boundary'].emit('censored',**row)
            write_json(out/'sampling_final.json',a['session'].summary())
            write_json(out/'summary.json',dict(completed=True,completed_updates=5000,environment_steps=491520000))
        finally:
            telemetry.close();interface.close();a['boundary'].close();a['log'].close();env.close()
    finally:app.close()
if __name__=='__main__':main()
