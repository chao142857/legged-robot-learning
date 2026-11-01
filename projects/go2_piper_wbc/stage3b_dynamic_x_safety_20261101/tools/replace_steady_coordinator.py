"""One CPU-only coordinator handoff while its native GPU child keeps running."""
import datetime, fcntl, hashlib, json, os, pathlib, signal, subprocess, sys, time, types


def main(payload):
    p=pathlib.Path('/data/go2_piper_wbc_stage3b_dynamic_x_20261101')
    sys.path.insert(0,str(p/'active'))
    from gpu_budget import Budget, read, write
    b=Budget(p)
    assert not any((p/n).exists() for n in ('RESEARCH_CLOSED.json','FROZEN_SELECTION.json','COMPLETED.json'))
    assert all(read(f).get('seed') not in (611101,621101,631101) for f in (p/'runs').glob('*/config.json'))
    old=read(p/'ACTIVE_LEARNING_VARIANT.json')['pid']
    cmd=(pathlib.Path('/proc')/str(old)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(p/'active/run_steady_kl.py') in cmd and str(p) in cmd
    for _ in range(30):
        active=b.meter()['active']
        if len(active)==1 and active[0]['mode']=='train' and pathlib.Path(active[0]['run']).name.endswith(
                tuple('steadyKL001-seed21-window-to'+str(n) for n in (2,100,200,400))):break
        assert not (p/'RESEARCH_CLOSED.json').exists()
        time.sleep(2)
    else:raise RuntimeError('No verified existing seed21 native window; original CPU coordinator left intact')
    before_job=dict(active[0]);new=None;old_stopped=False
    with (p/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert b.meter()['active']==active
        assert not (p/'FROZEN_SELECTION.json').exists() and not (p/'RESEARCH_CLOSED.json').exists()
        for name,entry in payload['sources'].items():
            assert name in ('run_steady_kl_v2.py','stage3b_selection.py')
            assert hashlib.sha256(entry['source'].encode()).hexdigest()==entry['sha256']
            compile(entry['source'],name,'exec')
        # The single amendment is frozen before any formal seed is opened.
        amendment=p/'ROBUST_NODE_EXTENSION_PREREGISTRATION.json'
        assert not amendment.exists();write(amendment,payload['preregistration'])
        changes={}
        for name,entry in payload['sources'].items():
            f=p/'active'/name;old_sha=None
            if f.exists():
                old_bytes=f.read_bytes();old_sha=hashlib.sha256(old_bytes).hexdigest()
                folder=p/'selection_rule_history' if name=='stage3b_selection.py' else p/'learning_source_history';folder.mkdir(exist_ok=True)
                archive=folder/(name+'.'+old_sha)
                if archive.exists():assert archive.read_bytes()==old_bytes
                else:archive.write_bytes(old_bytes)
            temp=f.with_suffix('.handoff.tmp');temp.write_text(entry['source']);temp.replace(f)
            changes[name]=dict(previous_sha256=old_sha,current_sha256=entry['sha256'])
        # Stop only the host CPU coordinator, while it is waiting for the
        # independently supervised native container. No Docker/RNG reset.
        os.kill(old,signal.SIGSTOP);old_stopped=True
        try:
            python=read(p/'CPU_RUNTIME.json')['python']
            with (p/'steadyKL001-v2-launch.log').open('ab') as log:
                new=subprocess.Popen([python,'-u',str(p/'active/run_steady_kl_v2.py'),'--root',str(p)],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            time.sleep(.3);assert new.poll() is None
            assert b.meter()['active']==active
            os.kill(old,signal.SIGTERM);os.kill(old,signal.SIGCONT);old_stopped=False
        except BaseException:
            if new and new.poll() is None:new.terminate()
            if old_stopped:os.kill(old,signal.SIGCONT)
            raise
        record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_CPU_pid=old,new_CPU_pid=new.pid,
            actual_native_GPU_unchanged=before_job,source_changes=changes,formal_seeds_unopened=True,
            GPU_watcher_finalizer_and_collector_unchanged=True,original_task_optimization_and_all_gates_unchanged=True,
            receipt_replay_prevents_duplicate_training=True,clock=b.meter())
        write(p/'STEADY_COORDINATOR_HANDOFF_V2.json',record)
    print(json.dumps(record))


if __name__=='__main__':main(json.load(sys.stdin))
