"""Isolated synthetic tests. No object here is reused by a training sampler."""
import copy
import numpy as np
import torch
from common import PARAMETERS, digest, rng_state, restore_rng
from vendor_sampler import TaskSampler
from e1_sampling import E1Sampler, E1Ownership, stage_for, rollout_update

def run_checks(device):
    before=rng_state()
    try:
        assert [stage_for(u) for u in (1,4000,4001,8000,8001,10000)] == [0,0,1,1,2,2]
        assert [rollout_update(u-1,True) for u in (50,51,150,151)] == [4000,4001,8000,8001]
        rng=np.random.default_rng(1984)
        s=E1Sampler(19); core=TaskSampler(19,42,mode='sgs',**PARAMETERS)
        assert s.outcomes==[[] for _ in range(19)] and not s.labels.any() and np.all(s.cached_rates==.5)
        for block in range(7):
            ids=rng.integers(0,19,500).tolist(); severity=rng.integers(0,4,500)
            y=(severity[:,None]>np.arange(3)).astype(np.uint8)
            eids=np.arange(block*500+1,(block+1)*500+1,dtype=np.uint64)
            s.record_multi(ids,y,eids); core.record(ids,y[:,2].tolist())
        original=digest(dict(labels=s.labels,ids=s.episode_ids,head=s.head)); old_rng=s.rng.getstate()
        for stage in (1,2,0,2):
            s.set_stage(stage)
            assert digest(dict(labels=s.labels,ids=s.episode_ids,head=s.head))==original and s.rng.getstate()==old_rng
            assert s.outcomes==[s.window(i)[0][:,stage].tolist() for i in range(s.n)]
        np.testing.assert_allclose(s.probabilities(),core.probabilities(),rtol=1e-13,atol=1e-15)
        assert s.sample(4096)==core.sample(4096)
        for stage in (0,1,2):
            s.set_stage(stage); state=s.state_dict(); restored=E1Sampler(19); restored.load_state_dict(state)
            assert digest(state)==digest(restored.state_dict())
            assert s.sample(4096)==restored.sample(4096)
        class Pool: n=19; identity='synthetic-e1-only'
        owner=E1Ownership(Pool(),6); ids=np.arange(6); owner.assign(ids); owner.advance()
        token=owner.token(ids); owner.engine.set_stage(1); owner.commit(token,[[1,0,0]]*6)
        assert owner.accepted==6
        rejected=[]
        for label,tok in [('duplicate',token),('foreign',dict(token,pool_identity='another-task'))]:
            try: owner.commit(tok,[[1,0,0]]*6)
            except AssertionError: rejected.append(label)
        assert rejected==['duplicate','foreign']
        owner.assign(ids); owner.check()
        state=owner.state_dict(); other=E1Ownership(Pool(),6); other.load_state_dict(state)
        assert digest(other.state_dict())==digest(state)

        # Replay known raw per-step physical errors through both original and E1 counters.
        import metrics as original_metrics, e1_metrics
        class Env:
            num_envs=9; step_dt=.02
            def __init__(self):
                self.device=device
                self.reset_terminated=torch.zeros(9,device=device,dtype=torch.bool);self.reset_terminated[4]=True
                self.reset_time_outs=torch.ones(9,device=device,dtype=torch.bool)
        env=Env(); raw=torch.zeros(9,4,device=device,dtype=torch.float32)
        native=original_metrics.physical_errors; patched=e1_metrics.physical_errors
        def errors(e,k): assert e is env; return raw
        original_metrics.physical_errors=errors; e1_metrics.physical_errors=errors
        try:
            m=e1_metrics.E1Metrics(env,'wbc')
            for step in range(300):
                raw.zero_();raw[:,2]=.03;raw[1,2]=.08;raw[2,2]=.06
                if step>=99:
                    i=step-99
                    # Two 60% marginal successes overlap on only 20% of scored steps.
                    raw[6,2]=.03 if i<121 else .11
                    raw[6,1]=0 if i>=80 else .31
                    # Time fraction passes loose, but position RMSE must veto it.
                    raw[7,2]=.03 if i<121 else .30
                m.command('ee',torch.tensor([5],device=device))
                m.capture()
            rows=m.finish(torch.arange(9,device=device)); labels=m.last_finished['labels'].astype(int).tolist()
            assert labels==[[1,1,1],[1,0,0],[1,1,0],[1,1,1],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,1,1]],labels
            saved=m.state_dict(); restored=e1_metrics.E1Metrics(env,'wbc');restored.load_state_dict(saved)
            assert digest(saved)==digest(restored.state_dict())
            assert np.array_equal(rows,restored.finish(torch.arange(9,device=device)))
        finally:
            original_metrics.physical_errors=native; e1_metrics.physical_errors=patched
        return dict(passed=True,synthetic_objects_discarded=True,synthetic_data_entered_training=False,
                    schedule_boundaries=[4000,4001,8000,8001],shared_window_and_rng_restore=True,
                    strict_core_probability_and_4096_draw_equivalence=True,phase_switch_preserves_window_and_rng=True,
                    failure_and_insufficient_window_veto=True,three_label_monotonicity=True,
                    simultaneous_four_way_time_check=True,position_rmse_veto=True,
                    strict_trajectory_oracle_equal=True,metric_state_restore=True,duplicate_and_foreign_feedback_rejected=True,
                    test_labels=labels)
    finally: restore_rng(before)
