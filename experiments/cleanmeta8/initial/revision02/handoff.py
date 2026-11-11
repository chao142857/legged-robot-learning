"""Execute at the completed Teacher boundary, never during a PPO job."""
import os,json,time,signal,subprocess
from pathlib import Path
ROOT=Path('/home/alex/dog/go2_piper_cleanmeta8_20261109')
def main():
 assert json.loads((ROOT/'CLEANMETA8_ACCEPTANCE.json').read_text())['passed']
 parent=os.getppid();p=Path('/proc')/str(parent)
 assert p.joinpath('cwd').resolve()==ROOT
 assert b'controller.py' in p.joinpath('cmdline').read_bytes()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 record=dict(parent_pid=parent,unix=time.time(),reason='User removed duplicate OldMeta8/Uniform jobs',new_session='cleanmeta8-seed48-only',no_student_job_interrupted=True)
 (ROOT/'HANDOFF_REQUEST.json').write_text(json.dumps(record,indent=2))
 subprocess.run(['bash',str(ROOT/'tmux.sh'),'new-session','-d','-s','cleanmeta8-seed48-only',f'cd {ROOT} && python3 revision02/post_queue.py > post_queue.log 2>&1'],check=True)
 # The old controller is waiting only for THIS completed CPU acceptance job.
 # SIGTERM stops that obsolete schedule; independent GPU workers do not exist.
 os.kill(parent,signal.SIGTERM)
 (ROOT/'HANDOFF_COMPLETE.json').write_text(json.dumps(record,indent=2))
 print('CONTROLLED_QUEUE_HANDOFF_COMPLETE',flush=True)
if __name__=='__main__':main()
