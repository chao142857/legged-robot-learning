import copy,hashlib,json,numpy as np,torch
from common import Pool,plain,emit,write,digest,PROTOCOL
from sampling import Ownership,CachedSampler
from metrics import Metrics

class SequenceAdapter:
    def __init__(self,env,kind,mode,pool,out):
        self.env,self.kind,self.mode,self.pool,self.out=env,kind,mode,pool.to(env.device),out
        self.owner=Ownership(pool,env.num_envs,mode,42)
        self.metrics=Metrics(env,kind);self.pending=None;self.episodes=0;self.successes=0;self.failures=0
        self.first_episode_rows=[];self.initial_offset=None
        self.task_tensor=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        self.cursor={k:torch.zeros_like(self.task_tensor) for k in ('ee','velocity')}
        self.command_stats={k:dict(count=torch.zeros((),device=env.device,dtype=torch.long),minimum=torch.full((7 if k=='ee' else 3,),float('inf'),device=env.device),maximum=torch.full((7 if k=='ee' else 3,),-float('inf'),device=env.device)) for k in ('ee','velocity')}
        self.standing_draws=torch.zeros((),device=env.device,dtype=torch.long);self.velocity_draws=torch.zeros_like(self.standing_draws)
        env.sequence_adapter=self
        self.original_reward=env.reward_manager.compute;self.original_reset=env._reset_idx
        env.reward_manager.compute=self.reward;env._reset_idx=self.reset
        self.feedback_file=out/'terminal_feedback.jsonl';self.assign_file=out/'assignments.jsonl'
        self.feedback_file.touch();self.assign_file.touch()
    def inputs(self,kind,ids):
        idx=self.cursor[kind][ids]
        assert bool((idx<4).all()),'Fixed sequence exhausted before native episode ended'
        table=self.pool.ee_tensor if kind=='ee' else self.pool.vel_tensor
        result=table[self.task_tensor[ids],idx];self.cursor[kind][ids]+=1
        return result
    def on_command(self,kind,ids,command):
        self.metrics.command(kind,ids);s=self.command_stats[kind];s['count']+=len(ids)
        s['minimum']=torch.minimum(s['minimum'],command.min(0).values);s['maximum']=torch.maximum(s['maximum'],command.max(0).values)
        if kind=='velocity':
            self.velocity_draws+=len(ids);self.standing_draws+=self.env.command_manager.get_term('base_velocity').is_standing_env[ids].sum()
    def reward(self,*args,**kwargs):
        assert self.pending is None,'Unconsumed previous terminal batch'
        result=self.original_reward(*args,**kwargs)
        self.owner.advance();self.metrics.capture()
        ids=self.env.reset_buf.nonzero(as_tuple=False).flatten()
        if len(ids):
            cpu_ids=ids.cpu().numpy();token=self.owner.token(cpu_ids);rows=self.metrics.finish(ids)
            self.pending=(cpu_ids,token,rows)
        return result
    def reset(self,ids):
        cpu_ids=ids.detach().cpu().numpy() if hasattr(ids,'detach') else np.asarray(ids,dtype=np.int64)
        if len(cpu_ids)==0:return self.original_reset(ids)
        if self.pending is not None:
            expected,token,rows=self.pending
            assert np.array_equal(cpu_ids,expected),'Native reset differs from captured terminal batch'
            self.owner.commit(token,rows[:,6].astype(bool).tolist())
            self.episodes+=len(rows);self.successes+=int(rows[:,6].sum());self.failures+=int(rows[:,7].sum())
            entry=dict(control_step=self.owner.control_steps,pool_identity=self.pool.identity,env_ids=cpu_ids.tolist(),task_ids=token['task_ids'].tolist(),generation=token['generations'].tolist(),physical_steps=token['steps'].tolist(),columns=['scored_steps','joint_good_steps','velocity_xy_rmse_m_s','yaw_rate_rmse_rad_s','ee_position_rmse_m','ee_orientation_rmse_rad','success','native_failure','timeout'],rows=rows.tolist())
            if self.mode=='evaluation':
                for i,row in enumerate(rows):
                    if token['generations'][i]==1:self.first_episode_rows.append(dict(env_id=int(cpu_ids[i]),sequence_id=int(token['task_ids'][i]),steps=int(token['steps'][i]),values=row.tolist()))
            emit(self.feedback_file,entry);self.pending=None
        else:self.owner.censor(cpu_ids)
        selected=self.owner.assign(cpu_ids);self.task_tensor[ids]=torch.from_numpy(selected).to(self.env.device)
        for c in self.cursor.values():c[ids]=0
        self.metrics.reset(ids)
        emit(self.assign_file,dict(control_step=self.owner.control_steps,pool_identity=self.pool.identity,env_ids=cpu_ids.tolist(),task_ids=selected.tolist(),generation=self.owner.generation[cpu_ids].tolist()))
        return self.original_reset(ids)
    def effective(self):
        e=self.env.command_manager.get_term('ee_pose').cfg;v=self.env.command_manager.get_term('base_velocity').cfg
        return dict(ee_ranges=plain(e.ranges),ee_limit_ranges=plain(e.limit_ranges),ee_curriculum=e.curriculum_enabled,ee_resampling=e.resampling_time_range,velocity_ranges=plain(v.ranges),velocity_limit_ranges=plain(v.limit_ranges),velocity_curriculum=v.curriculum_enabled,velocity_resampling=v.resampling_time_range,rel_standing_envs=v.rel_standing_envs,heading_command=v.heading_command,command_draws={k:plain(s) for k,s in self.command_stats.items()},standing_draws=int(self.standing_draws),velocity_draws=int(self.velocity_draws),pool_standing_prior=.1,observed_standing_draw_fraction=float(self.standing_draws/max(1,int(self.velocity_draws))))
    def summary(self):
        p=np.asarray(self.owner.engine.probabilities())
        return dict(pool_identity=self.pool.identity,mode=self.mode,episodes=self.episodes,successes=self.successes,native_failures=self.failures,control_steps=self.owner.control_steps,environment_transitions=self.owner.control_steps*self.env.num_envs,feedback_accepted=self.owner.accepted,active=int((self.owner.task>=0).sum()),history_samples=sum(len(h) for h in self.owner.engine.outcomes),nonempty_histories=sum(bool(h) for h in self.owner.engine.outcomes),probability_min=float(p.min()),probability_max=float(p.max()),effective_sequences=float(1/(p*p).sum()),effective_commands=self.effective())
    def state_dict(self):
        assert self.pending is None
        terms={}
        for name in ('ee_pose','base_velocity'):
            term=self.env.command_manager.get_term(name)
            terms[name]={k:getattr(term,k).detach().clone().cpu() for k in ('time_left','command_counter','pose_command','pose_command_b','pose_command_w','vel_command_b','is_standing_env') if hasattr(term,k)}
        return dict(version=1,kind=self.kind,protocol_sha=digest(PROTOCOL),owner=self.owner.state_dict(),metrics=self.metrics.state_dict(),task_tensor=self.task_tensor.cpu().clone(),cursor={k:c.cpu().clone() for k,c in self.cursor.items()},terms=terms,summary=self.summary(),restore_scope='Sampler/metrics/command/RNG serialized. No assertion of PhysX solver bit-exact restore.')
    def restoration_check(self,state=None):
        state=self.state_dict() if state is None else state;restored=Ownership(self.pool,self.env.num_envs,self.mode,42);restored.load_state_dict(state['owner'])
        probe=CachedSampler(self.pool.n,42,self.mode);probe.load_state_dict(state['owner']['engine'])
        expected=copy.deepcopy(self.owner.engine)
        assert probe.sample(4096)==expected.sample(4096)
        assert np.array_equal(restored.task,self.owner.task) and np.array_equal(restored.generation,self.owner.generation)
        restored.check();self.owner.check()
        for k,c in self.cursor.items():assert torch.equal(c.cpu(),state['cursor'][k])
        return dict(passed=True,sampler_rng_draws_equal=4096,task_ownership_equal=True,feedback_counter_consistent=True,command_cursors_equal=True,probability_span=float(max(self.owner.engine.probabilities())-min(self.owner.engine.probabilities())),restore_scope=state['restore_scope'])
