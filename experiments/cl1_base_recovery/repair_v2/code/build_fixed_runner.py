"""Apply count-checked recovery/provenance fixes to the frozen-transfer runner only."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[1];s=(r/'code/train_cl1_legacy.py').read_text();changes=[]
def sub(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new);changes.append(old)
sub("teacher.load(snapshot['teacher'])","teacher.load(snapshot['teacher'])\n        teacher.checkpoint_source = snapshot.get('resume_teacher_snapshot')")
sub("phase = 'initialized'","phase = 'initialized'\n    resume_pending = False")
sub("latest_row = info.get('latest_row')","latest_row = info.get('latest_row')\n                resume_pending = info['phase'] == 'ready'")
sub("        if args.resume:\n            decision = new_decision(latest_row)","        if args.resume and not resume_pending:\n            decision = new_decision(latest_row)")
sub("source_checkpoint_sha256=source_hash, latest_row=latest_row, teacher=teacher.state() if teacher else None,", "source_checkpoint_sha256=source_hash, latest_row=latest_row, teacher=teacher.state() if teacher else None,\n                    cl1_teacher_snapshot=teacher.checkpoint_reference() if teacher else None,")
(r/'code/train_cl1.py').write_text(s)
p=(r/'code/cl1_proxy_legacy.py').read_text()
p=p.replace('self.dev=None','self.dev=None;self.last_snapshot=None;self.checkpoint_source=None')
p=p.replace("        return torch.load(reply,map_location='cpu',weights_only=False)","        self.last_snapshot = p.with_suffix('.teacher_after.pt')\n        return torch.load(reply,map_location='cpu',weights_only=False)")
p+='''\n    def checkpoint_reference(self):
        from common import file_sha
        if self.last_snapshot is not None:
            return dict(path=str(self.last_snapshot),sha256=file_sha(self.last_snapshot))
        assert self.checkpoint_source is not None, 'Missing paired Teacher checkpoint'
        assert file_sha(self.checkpoint_source['path'])==self.checkpoint_source['sha256']
        return self.checkpoint_source
'''
(r/'code/cl1_proxy.py').write_text(p)
(r/'FIXED_ADAPTATION.json').write_text(json.dumps(dict(changes=changes,runner_sha256=hashlib.sha256(s.encode()).hexdigest(),proxy_sha256=hashlib.sha256(p.encode()).hexdigest()),indent=2))
