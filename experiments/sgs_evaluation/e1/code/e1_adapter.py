import copy
import numpy as np
import torch
from adapter import SequenceAdapter
from common import emit, write, digest
from e1_sampling import E1Ownership, stage_for
from e1_metrics import E1Metrics

class E1Adapter(SequenceAdapter):
    def __init__(self, env, kind, mode, pool, out):
        assert mode == 'sgs'
        super().__init__(env, kind, mode, pool, out)
        self.owner = E1Ownership(pool, env.num_envs)
        self.metrics = E1Metrics(env, kind)
        self.rollout_update = 1
        self.assigned_stage = np.full(env.num_envs, -1, dtype=np.int64)
        self.assigned_update = np.zeros(env.num_envs, dtype=np.int64)
        self.stage_crossing_episodes = 0
        self.stage_transitions = 0

    def before_rollout(self, update):
        new = stage_for(update); old = self.owner.engine.active_stage
        self.rollout_update = update
        if new != old:
            assert new == old+1
            e = self.owner.engine
            before = dict(task=self.owner.task.copy(), generation=self.owner.generation.copy(), accepted=self.owner.accepted,
                          labels=digest(e.labels), ids=digest(e.episode_ids), rng=e.rng.getstate())
            snap = dict(rollout_update=update, control_step=self.owner.control_steps, active_episodes=int((self.owner.task>=0).sum()),
                        old=e.mechanism(old,True), new=e.mechanism(new,True), same_history_labels_sha256=before['labels'], same_episode_ids_sha256=before['ids'])
            e.set_stage(new)
            np.testing.assert_allclose(e.probabilities(),snap['new']['probabilities'],rtol=1e-14,atol=1e-16)
            assert before['rng'] == e.rng.getstate() and before['accepted'] == self.owner.accepted
            assert np.array_equal(before['task'],self.owner.task) and np.array_equal(before['generation'],self.owner.generation)
            assert before['labels'] == digest(e.labels) and before['ids'] == digest(e.episode_ids)
            snap.update(no_forced_reset=True, rng_unchanged=True, history_unchanged=True, probabilities_refreshed_before_next_request=True)
            write(self.out/'switches'/('before_rollout_'+str(update)+'.json'),snap)
            self.stage_transitions += 1
            print('E1_STAGE_SWITCH '+str(update)+' '+str(old+1)+' -> '+str(new+1),flush=True)

    def reset(self, ids):
        cpu_ids = ids.detach().cpu().numpy() if hasattr(ids,'detach') else np.asarray(ids,dtype=np.int64)
        if len(cpu_ids) == 0: return self.original_reset(ids)
        if self.pending is not None:
            expected, token, rows = self.pending
            assert np.array_equal(cpu_ids, expected)
            tri = self.metrics.last_finished
            labels, good = tri['labels'], tri['good']
            assert np.array_equal(labels[:,2], rows[:,6].astype(bool))
            self.owner.commit(token,labels)
            self.episodes += len(rows); self.successes += int(rows[:,6].sum()); self.failures += int(rows[:,7].sum())
            crossed = self.assigned_stage[cpu_ids] != self.owner.engine.active_stage
            self.stage_crossing_episodes += int(crossed.sum())
            entry = dict(control_step=self.owner.control_steps, rollout_update=self.rollout_update, active_stage=self.owner.engine.active_stage+1,
                         pool_identity=self.pool.identity, env_ids=cpu_ids.tolist(), task_ids=token['task_ids'].tolist(),
                         generation=token['generations'].tolist(), physical_steps=token['steps'].tolist(),
                         assignment_stage=(self.assigned_stage[cpu_ids]+1).tolist(), assignment_rollout_update=self.assigned_update[cpu_ids].tolist(),
                         labels_loose_medium_strict=labels.astype(int).tolist(), joint_good_steps_loose_medium_strict=good.tolist(),
                         columns=['scored_steps','joint_good_steps','velocity_xy_rmse_m_s','yaw_rate_rmse_rad_s','ee_position_rmse_m','ee_orientation_rmse_rad','success','native_failure','timeout'], rows=rows.tolist())
            emit(self.feedback_file,entry); self.pending=None; self.metrics.last_finished=None
        else: self.owner.censor(cpu_ids)
        selected=self.owner.assign(cpu_ids); self.task_tensor[ids]=torch.from_numpy(selected).to(self.env.device)
        self.assigned_stage[cpu_ids]=self.owner.engine.active_stage; self.assigned_update[cpu_ids]=self.rollout_update
        for c in self.cursor.values(): c[ids]=0
        self.metrics.reset(ids)
        emit(self.assign_file,dict(control_step=self.owner.control_steps, rollout_update=self.rollout_update, stage=self.owner.engine.active_stage+1,
                                  pool_identity=self.pool.identity,env_ids=cpu_ids.tolist(),task_ids=selected.tolist(),generation=self.owner.generation[cpu_ids].tolist()))
        return self.original_reset(ids)

    def summary(self):
        s=super().summary()
        s.update(active_stage=self.owner.engine.active_stage+1, rollout_update=self.rollout_update,
                 strict_shadow_compared_episodes=self.metrics.strict_compared_episodes, stage_crossing_episodes=self.stage_crossing_episodes,
                 stage_transitions=self.stage_transitions, successes_field='Strict shadow successes, not the current sampling label')
        return s

    def state_dict(self):
        s=super().state_dict()
        s['e1']=dict(rollout_update=self.rollout_update,assigned_stage=self.assigned_stage.copy(),assigned_update=self.assigned_update.copy(),
                     stage_crossing_episodes=self.stage_crossing_episodes,stage_transitions=self.stage_transitions)
        return s

    def restoration_check(self,state=None):
        state=self.state_dict() if state is None else state
        restored=E1Ownership(self.pool,self.env.num_envs); restored.load_state_dict(state['owner'])
        a=copy.deepcopy(self.owner.engine); b=copy.deepcopy(restored.engine)
        assert a.sample(4096)==b.sample(4096)
        assert digest(a.state_dict())==digest(b.state_dict())
        restored_metrics=E1Metrics(self.env,self.kind);restored_metrics.load_state_dict(state['metrics'])
        assert digest(restored_metrics.state_dict())==digest(state['metrics'])
        restored.check(); self.owner.check()
        assert np.array_equal(restored.generation,self.owner.generation) and np.array_equal(restored.steps,self.owner.steps)
        assert state['e1']['rollout_update']==self.rollout_update
        assert np.array_equal(state['e1']['assigned_stage'],self.assigned_stage) and np.array_equal(state['e1']['assigned_update'],self.assigned_update)
        for k,c in self.cursor.items(): assert torch.equal(c.cpu(),state['cursor'][k])
        assert self.metrics.strict_compared_episodes == self.episodes == self.owner.accepted
        return dict(passed=True,sampler_rng_draws_equal=4096,shared_three_label_window_equal=True,
                    task_ownership_equal=np.array_equal(restored.task,self.owner.task),strict_shadow_compared_episodes=self.episodes,
                    stage_crossing_episodes=self.stage_crossing_episodes,stage_transitions=self.stage_transitions,
                    restore_scope=state['restore_scope'])
