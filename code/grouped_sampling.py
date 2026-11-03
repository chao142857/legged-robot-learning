"""E2: same strict SGS kernel, a last-100 *episode* window per statistical group.

The sequence pool is unchanged. P(group k)=|Gk|*f(pk)/sum_j |Gj|*f(pj),
then a member is drawn uniformly. Thus P(sequence i)=f(p_group(i))/Z,
including unequal group sizes. The sole experimental input is the mapping.
"""
import copy, hashlib, random
import numpy as np
from sampling import Ownership
from common import PARAMETERS

def mapping_hash(mapping):
    return hashlib.sha256(np.asarray(mapping,dtype='<i8').tobytes()).hexdigest()

class GroupedSampler:
    def __init__(self, mapping, pool_identity, seed=42, **parameters):
        raw=np.asarray(mapping)
        if raw.ndim!=1 or not len(raw) or not np.issubdtype(raw.dtype,np.integer):raise ValueError('Mapping must be a nonempty integer vector')
        self.mapping=raw.astype(np.int64,copy=True);self.n=len(raw)
        if self.mapping.min()<0:raise ValueError('Negative group')
        self.groups=int(self.mapping.max())+1
        if set(self.mapping.tolist())!=set(range(self.groups)):raise ValueError('Group IDs must be contiguous and nonempty')
        self.members=[np.flatnonzero(self.mapping==k).tolist() for k in range(self.groups)]
        self.sizes=np.asarray([len(a) for a in self.members],dtype=np.int64)
        self.pool_identity=str(pool_identity);self.mapping_sha256=mapping_hash(self.mapping);self.mode='sgs'
        cfg=dict(PARAMETERS);cfg.update(parameters)
        self.history=int(cfg['history']);self.target=float(cfg['target']);self.concentration=float(cfg['concentration']);self.temperature=float(cfg['temperature']);self.epsilon=float(cfg['epsilon'])
        if self.history<1 or self.temperature<1 or not 0<=self.target<=1 or self.concentration<0 or self.epsilon<=0:raise ValueError('Invalid SGS parameters')
        self.rng=random.Random(seed);self.outcomes=[[] for _ in self.members]
        self.window_steps=[[] for _ in self.members];self.window_members=[[] for _ in self.members];self.window_events=[[] for _ in self.members]
        self.window_successes=np.zeros(self.groups,dtype=np.int64)
        self.sample_counts=[0]*self.n;self.completed_counts=[0]*self.n;self.success_counts=[0]*self.n
        self.group_completed=np.zeros(self.groups,dtype=np.int64);self.group_successes=np.zeros(self.groups,dtype=np.int64)
        self.last_generations={};self.last_control_step=0;self.eval_cursor=0

    def group_rates(self):
        n=np.fromiter((len(h) for h in self.outcomes),dtype=np.int64,count=self.groups)
        return np.divide(self.window_successes,n,out=np.full(self.groups,.5),where=n>0)

    def _factors(self):
        p=self.group_rates();e=self.epsilon
        kernel=(p+e)**(self.concentration*self.target)*(1-p+e)**(self.concentration*(1-self.target))
        logits=np.log(np.maximum(kernel,e)+e)/self.temperature
        return np.exp(logits-logits.max())

    def group_probabilities(self):
        mass=self.sizes*self._factors();return mass/mass.sum()

    def probabilities(self):
        f=self._factors();return (f[self.mapping]/np.dot(self.sizes,f)).tolist()

    def rates(self):return self.group_rates()[self.mapping].tolist()

    def sample(self,count):
        if not isinstance(count,int) or count<0:raise ValueError('Invalid count')
        groups=self.rng.choices(range(self.groups),weights=self.group_probabilities().tolist(),k=count)
        result=[self.members[k][self.rng.randrange(len(self.members[k]))] for k in groups]
        for i in result:self.sample_counts[i]+=1
        return result

    def record(self,ids,successes,*,control_step=None,event_ids=None):
        ids=list(ids);successes=list(successes)
        if len(ids)!=len(successes):raise ValueError('Feedback length mismatch')
        tick=self.last_control_step+1 if control_step is None else int(control_step)
        if tick<self.last_control_step:raise ValueError('Feedback clock went backwards')
        if event_ids is not None and len(event_ids)!=len(ids):raise ValueError('Event length mismatch')
        next_generations=dict(self.last_generations)
        # Validate the entire batch before changing any count, history or clock.
        for j,(i,y) in enumerate(zip(ids,successes)):
            if not isinstance(i,(int,np.integer)) or not 0<=int(i)<self.n or y not in (False,True,0,1):raise ValueError('Invalid sequence or binary label')
            if event_ids is not None:
                env,generation=map(int,event_ids[j])
                if env<0 or generation<1 or generation<=next_generations.get(env,0):raise ValueError('Repeated or stale episode')
                next_generations[env]=generation
        for j,(i,y) in enumerate(zip(ids,successes)):
            i=int(i);y=int(y);k=int(self.mapping[i]);h=self.outcomes[k]
            if len(h)==self.history:
                self.window_successes[k]-=h.pop(0)
                self.window_steps[k].pop(0);self.window_members[k].pop(0);self.window_events[k].pop(0)
            h.append(y);self.window_successes[k]+=y
            self.window_steps[k].append(tick);self.window_members[k].append(i)
            self.window_events[k].append(tuple(map(int,event_ids[j])) if event_ids is not None else None)
            self.completed_counts[i]+=1;self.success_counts[i]+=y
            self.group_completed[k]+=1;self.group_successes[k]+=y
        self.last_generations=next_generations;self.last_control_step=tick

    def state_dict(self):
        return copy.deepcopy(dict(version=1,n=self.n,mode=self.mode,groups=self.groups,pool_identity=self.pool_identity,mapping=self.mapping,
            mapping_sha256=self.mapping_sha256,history=self.history,target=self.target,concentration=self.concentration,temperature=self.temperature,epsilon=self.epsilon,
            empty_history_prior=.5,outcomes=self.outcomes,window_steps=self.window_steps,window_members=self.window_members,window_events=self.window_events,
            window_successes=self.window_successes,sample_counts=self.sample_counts,completed_counts=self.completed_counts,success_counts=self.success_counts,
            group_completed=self.group_completed,group_successes=self.group_successes,last_generations=self.last_generations,last_control_step=self.last_control_step,
            eval_cursor=self.eval_cursor,rng_state=self.rng.getstate()))

    def load_state_dict(self,state):
        state=copy.deepcopy(state);expected=self.state_dict()
        for k in ('version','n','mode','groups','pool_identity','mapping_sha256','history','target','concentration','temperature','epsilon','empty_history_prior'):
            if state[k]!=expected[k]:raise ValueError('Incompatible checkpoint: '+k)
        if not np.array_equal(state['mapping'],self.mapping) or mapping_hash(state['mapping'])!=self.mapping_sha256:raise ValueError('Mapping content mismatch')
        for k in ('outcomes','window_steps','window_members','window_events'):
            if len(state[k])!=self.groups:raise ValueError('Group state shape mismatch')
        for g,h in enumerate(state['outcomes']):
            if len(h)>self.history or any(y not in (0,1) for y in h):raise ValueError('Invalid history')
            if any(len(state[k][g])!=len(h) for k in ('window_steps','window_members','window_events')):raise ValueError('Window provenance mismatch')
            if int(state['window_successes'][g])!=sum(h):raise ValueError('Success counter mismatch')
            if any(self.mapping[i]!=g for i in state['window_members'][g]):raise ValueError('Foreign sequence inside group history')
            if any(a>b for a,b in zip(state['window_steps'][g],state['window_steps'][g][1:])):raise ValueError('History time order mismatch')
        for k in ('sample_counts','completed_counts','success_counts'):
            if len(state[k])!=self.n:raise ValueError('Sequence count shape mismatch')
        grouped=np.bincount(self.mapping,weights=np.array(state['completed_counts']),minlength=self.groups).astype(np.int64)
        if not np.array_equal(grouped,state['group_completed']):raise ValueError('Group completion counter mismatch')
        grouped_success=np.bincount(self.mapping,weights=np.array(state['success_counts']),minlength=self.groups).astype(np.int64)
        if not np.array_equal(grouped_success,state['group_successes']):raise ValueError('Group success counter mismatch')
        if any(a<0 or b<0 or b>a for a,b in zip(state['completed_counts'],state['success_counts'])):raise ValueError('Invalid sequence totals')
        def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,(tuple,list)) else v
        restored_rng=random.Random();restored_rng.setstate(tuples(state['rng_state']))
        for k in ('window_successes','group_completed','group_successes'):state[k]=np.asarray(state[k],dtype=np.int64).copy()
        state['last_generations']={int(k):int(v) for k,v in state['last_generations'].items()}
        for k in ('outcomes','window_steps','window_members','window_events','window_successes','sample_counts','completed_counts','success_counts','group_completed','group_successes','last_generations','last_control_step','eval_cursor'):setattr(self,k,state[k])
        self.rng=restored_rng

    def diagnostics(self,now=None):
        now=self.last_control_step if now is None else int(now)
        ages=np.asarray([now-t for ts in self.window_steps for t in ts],dtype=np.float64)
        diversity=np.asarray([len(set(ids)) for ids in self.window_members]);q=self.group_probabilities();p=np.asarray(self.probabilities())
        return dict(statistical_groups=self.groups,real_sequences=self.n,mapping_sha256=self.mapping_sha256,
            group_sizes_min=int(self.sizes.min()),group_sizes_max=int(self.sizes.max()),window='Last100 ended episodes per group, chronological native terminal order; no per-sequence100x8 expansion.',
            empty_groups=sum(not h for h in self.outcomes),full_groups=sum(len(h)==self.history for h in self.outcomes),
            median_members_represented=float(np.median(diversity)),mean_feedback_age_control_steps=float(ages.mean()) if len(ages) else None,
            effective_groups=float(1/np.dot(q,q)),effective_sequences=float(1/np.dot(p,p)),feedback_total=int(self.group_completed.sum()))

class GroupedOwnership(Ownership):
    def __init__(self,pool,num_envs,mapping,seed=42):
        super().__init__(pool,num_envs,'sgs',seed)
        self.engine=GroupedSampler(mapping,pool.identity,seed)
        if self.engine.n!=pool.n:raise ValueError('Pool/mapping size mismatch')

    def commit(self,token,successes):
        assert token['pool_identity']==self.pool.identity,'Foreign pool feedback'
        ids=np.asarray(token['env_ids']);assert len(np.unique(ids))==len(ids),'Duplicate terminal batch'
        assert np.array_equal(self.task[ids],token['task_ids']) and np.array_equal(self.generation[ids],token['generations']),'Stale ownership'
        assert np.array_equal(self.steps[ids],token['steps']) and np.all(self.steps[ids]>0),'Invalid terminal duration'
        assert np.all(self.closed_generation[ids]!=self.generation[ids]),'Duplicate feedback'
        events=list(zip(ids.tolist(),self.generation[ids].tolist()))
        self.engine.record(self.task[ids].tolist(),successes,control_step=self.control_steps,event_ids=events)
        self.accepted+=len(ids);self.closed_generation[ids]=self.generation[ids];self.task[ids]=-1
