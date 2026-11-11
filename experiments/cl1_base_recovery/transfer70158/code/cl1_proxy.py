"""Filesystem RPC: physics runner never imports or mutates the outer optimizer."""
import copy,time
from pathlib import Path
import torch
class Teacher:
    def __init__(self, static, binding, out, seed):
        self.out=Path(out);self.seed=seed;self.binding=copy.deepcopy(binding);self.dev=None;self.last_snapshot=None;self.checkpoint_source=None
    def state(self):return dict(cl1_proxy=1,binding=self.binding)
    def load(self,s):assert s==self.state()
    def rpc(self,kind,version,**data):
        folder=self.out/'teacher_requests';folder.mkdir(exist_ok=True)
        p=folder/f'{version:04d}-{kind}.pt';assert not p.exists()
        temp=p.with_suffix('.tmp');torch.save(dict(kind=kind,seed=self.seed,version=version,**data),temp);temp.replace(p)
        reply=p.with_suffix('.done.pt');start=time.monotonic()
        while not reply.exists():
            if time.monotonic()-start>1800:raise TimeoutError(str(p))
            time.sleep(.25)
        self.last_snapshot = p.with_suffix('.teacher_after.pt')
        return torch.load(reply,map_location='cpu',weights_only=False)
    def decide(self,dynamic,global_state,version,**kw):
        return self.rpc('decide',version,dynamic=dynamic,global_state=global_state,dev=self.dev,**kw)
    def complete(self,decision,dynamic,global_state,dev,checkpoint_update):
        return self.rpc('complete',decision['version'],dynamic=dynamic,global_state=global_state,
                        dev=dev,checkpoint_update=checkpoint_update)

    def checkpoint_reference(self):
        from common import file_sha
        if self.last_snapshot is not None:
            return dict(path=str(self.last_snapshot),sha256=file_sha(self.last_snapshot))
        assert self.checkpoint_source is not None, 'Missing paired Teacher checkpoint'
        assert file_sha(self.checkpoint_source['path'])==self.checkpoint_source['sha256']
        return self.checkpoint_source
