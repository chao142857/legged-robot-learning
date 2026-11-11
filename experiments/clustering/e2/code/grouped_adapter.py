"""One integration for Random Group and Command Cluster; only the mapping differs."""
import copy
import numpy as np
import torch
from adapter import SequenceAdapter
from grouped_sampling import GroupedOwnership
from common import digest

class GroupedAdapter(SequenceAdapter):
    def __init__(self,env,kind,mode,pool,out,mapping):
        assert kind=='wbc' and mode=='sgs'
        super().__init__(env,kind,mode,pool,out)
        self.mapping=np.asarray(mapping,dtype=np.int64)
        assert len(self.mapping)==32768 and np.array_equal(np.bincount(self.mapping),np.full(4096,8))
        self.owner=GroupedOwnership(pool,env.num_envs,self.mapping,42)

    def summary(self):
        s=super().summary();s.update(self.owner.engine.diagnostics(self.owner.control_steps))
        s['history_fields_scope']='history_samples and nonempty_histories refer to statistical groups; probabilities refer to original sequences'
        return s

    def restoration_check(self,state=None):
        state=self.state_dict() if state is None else state
        other=GroupedOwnership(self.pool,self.env.num_envs,self.mapping,42);other.load_state_dict(state['owner'])
        a=copy.deepcopy(self.owner.engine);b=copy.deepcopy(other.engine)
        assert a.sample(4096)==b.sample(4096) and digest(a.state_dict())==digest(b.state_dict())
        other.check();self.owner.check()
        assert np.array_equal(other.task,self.owner.task) and np.array_equal(other.generation,self.owner.generation)
        for k,c in self.cursor.items():assert torch.equal(c.cpu(),state['cursor'][k])
        return dict(passed=True,sampler_rng_draws_equal=4096,shared_group_windows_equal=True,ownership_equal=True,mapping_sha256=self.owner.engine.mapping_sha256)
