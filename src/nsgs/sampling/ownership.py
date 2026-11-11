"""Cache acceleration around the unchanged projects/sgs TaskSampler core.
Only the algebraic probability calculation is vectorized; history, empty prior,
Python RNG, weighted choices, counts and serialization are inherited verbatim.
"""
import copy
import numpy as np
from nsgs.sampling.vendor_sampler import TaskSampler
from nsgs.common import PARAMETERS

class CachedSampler(TaskSampler):
    def __init__(self,n,seed,mode='uniform'):
        super().__init__(n,seed,mode=mode,**PARAMETERS)
        self.cached_rates=np.full(n,.5,dtype=np.float64)
        self.uniform=[1./n]*n
    def probabilities(self):
        if self.mode!='sgs':return self.uniform
        p=self.cached_rates;k=self.concentration;t=self.target;e=self.epsilon
        w=np.power(p+e,k*t)*np.power(1-p+e,k*(1-t))
        logits=np.log(np.maximum(w,e)+e)/self.temperature
        w=np.exp(logits-logits.max());return (w/w.sum()).tolist()
    def record(self,ids,successes):
        super().record(ids,successes)
        for i in set(ids):self.cached_rates[i]=sum(self.outcomes[i])/len(self.outcomes[i])
    def load_state_dict(self,s):
        super().load_state_dict(s)
        self.cached_rates=np.asarray(super().rates(),dtype=np.float64)

class Ownership:
    def __init__(self,pool,num_envs,mode,seed=42):
        self.pool=pool;self.engine=CachedSampler(pool.n,seed,mode)
        self.mode=mode;self.task=np.full(num_envs,-1,dtype=np.int64);self.generation=np.zeros(num_envs,dtype=np.int64)
        self.steps=np.zeros(num_envs,dtype=np.int64);self.closed_generation=np.full(num_envs,-1,dtype=np.int64)
        self.transitions=np.zeros(pool.n,dtype=np.int64);self.censored=np.zeros(pool.n,dtype=np.int64)
        self.accepted=0;self.duplicate_attempts=0;self.evaluation_ignored=0;self.control_steps=0
    def assign(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        assert len(np.unique(ids))==len(ids) and np.all(self.task[ids]<0),'Assignment over an active old episode'
        self.task[ids]=self.engine.sample(len(ids));self.generation[ids]+=1;self.steps[ids]=0
        return self.task[ids].copy()
    def advance(self):
        active=self.task>=0;self.steps[active]+=1;np.add.at(self.transitions,self.task[active],1);self.control_steps+=1
    def token(self,ids):return dict(pool_identity=self.pool.identity,env_ids=np.array(ids),task_ids=self.task[ids].copy(),generations=self.generation[ids].copy(),steps=self.steps[ids].copy())
    def commit(self,token,successes):
        assert token['pool_identity']==self.pool.identity,'Foreign task/pool feedback'
        ids=np.asarray(token['env_ids']);assert len(np.unique(ids))==len(ids),'Duplicate terminal batch'
        assert np.array_equal(self.task[ids],token['task_ids']) and np.array_equal(self.generation[ids],token['generations']),'Stale/foreign task ownership'
        assert np.array_equal(self.steps[ids],token['steps']) and np.all(self.steps[ids]>0),'Duration mismatch'
        assert np.all(self.closed_generation[ids]!=self.generation[ids]),'Duplicate feedback'
        if self.mode!='evaluation':
            self.engine.record(self.task[ids].tolist(),[bool(s) for s in successes]);self.accepted+=len(ids)
        else:self.evaluation_ignored+=len(ids)
        self.closed_generation[ids]=self.generation[ids];self.task[ids]=-1
    def censor(self,ids):
        ids=np.asarray(ids);old=self.task[ids];valid=old>=0
        np.add.at(self.censored,old[valid],1);self.task[ids]=-1
    def state_dict(self):
        return copy.deepcopy(dict(pool_identity=self.pool.identity,engine=self.engine.state_dict(),task=self.task,generation=self.generation,steps=self.steps,closed_generation=self.closed_generation,transitions=self.transitions,censored=self.censored,accepted=self.accepted,evaluation_ignored=self.evaluation_ignored,control_steps=self.control_steps))
    def load_state_dict(self,s):
        assert s['pool_identity']==self.pool.identity
        self.engine.load_state_dict(s['engine'])
        for k in ('task','generation','steps','closed_generation','transitions','censored'):setattr(self,k,np.array(s[k],copy=True))
        for k in ('accepted','evaluation_ignored','control_steps'):setattr(self,k,s[k])
    def check(self):
        assert sum(self.engine.completed_counts)==self.accepted
        assert self.control_steps*len(self.task)==int(self.transitions.sum())
        if self.mode=='evaluation':assert not any(self.engine.completed_counts) and not any(self.engine.outcomes)
        else:assert sum(self.engine.sample_counts)==self.accepted+int(self.censored.sum())+int((self.task>=0).sum())
