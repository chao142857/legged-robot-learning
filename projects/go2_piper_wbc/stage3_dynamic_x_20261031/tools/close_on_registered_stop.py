"""Finish Stage3 formal acceptance after a registered seed22 branch stop.

No GPU starts, checkpoint modification, threshold change or new-budget launch.
The existing independent finalizer performs all formal tests and full restore.
"""
import argparse,fcntl,json,os,pathlib,time,traceback
from gpu_budget import Budget,read,write

def main(b):
    write(b.p/'REGISTERED_STOP_CLOSER_PROCESS.json',dict(pid=os.getpid(),CPU_only=True))
    while not (b.p/'COMPLETED.json').exists() and not (b.p/'RESEARCH_CLOSED.json').exists():
        if (b.p/'REPLICATION_ERROR.json').exists():
            write(b.p/'REGISTERED_STOP_CLOSER_STATE.json',dict(state='engineering_error_requires_repair',not_reclassified_as_research_failure=True));return
        marker=b.p/'REPLICATION_REPAIR_NEEDED.json'
        if not marker.exists():time.sleep(5);continue
        flag=read(marker);nodes=read(b.p/'SEED22_NODES.json');latest=nodes[-1]
        assert latest['label'].endswith('-seed22')
        reference=flag.get('node') or flag['nodes'][-1]
        assert reference['checkpoint']['checkpoint_sha256']==latest['checkpoint']['checkpoint_sha256']
        rows=[r for r in read(b.p/'PHYSICAL_CURVE.json')['rows'] if r['label'].startswith(latest['label']+'-')]
        assert len(rows)==8 and all({r['protocol'] for r in rows if r['seed']==s}=={'A','T0','T1','X'} for s in (6181031,6281031))
        assert all(r['checkpoint_sha256']==latest['checkpoint']['checkpoint_sha256'] for r in rows)
        c=b.meter()
        if c['active']:time.sleep(3);continue
        # The marker is written immediately before coordinator return. Never
        # race a live controller at a window boundary or kill it to proceed.
        pid=read(b.p/'SEED22_PROCESS.json')['pid']
        stat=pathlib.Path('/proc')/str(pid)/'stat'
        if stat.exists() and stat.read_text().split()[2]!='Z':time.sleep(3);continue
        write(b.p/'RESEARCH_CLOSED.json',dict(
            reason='Registered seed22 stopping condition reached after full development matrix; finish original formal tests/restore before independent Stage3b.',
            original_stopping_marker=flag,latest_development_node=latest,clock=c,
            unsafe_checkpoint_will_not_be_resumed=True,new_Stage3b_not_started=True,
            original21600_cap_and3600_final_reserve_unchanged=True))
        write(b.p/'REGISTERED_STOP_CLOSER_STATE.json',dict(state='formal_finalizer_released',clock=c));return

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();b=Budget(a.root)
    with (b.p/'registered_stop_closer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(b)
        except BaseException as e:
            write(b.p/'REGISTERED_STOP_CLOSER_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
