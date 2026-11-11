import numpy as np
import torch
from adapter import SequenceAdapter
from common import emit, plain
from neural_owner import NeuralOwner
from neural_core import distribution_stats
from protocol import CONFIG


class NeuralAdapter(SequenceAdapter):
    def __init__(self, env, pool, out, decision):
        super().__init__(env, "wbc", "uniform", pool, out)
        self.mode = "neural"
        self.owner = NeuralOwner(pool, env.num_envs, CONFIG["sampler_seed"])
        self.owner.engine.install(decision["probability"], decision["version"])

    def reset(self, ids):
        cpu_ids = ids.detach().cpu().numpy() if hasattr(ids,"detach") else np.asarray(ids,dtype=np.int64)
        if not len(cpu_ids):
            return self.original_reset(ids)
        if self.pending is not None:
            expected, token, rows = self.pending
            assert np.array_equal(cpu_ids, expected)
            self.owner.commit_rows(token, rows)
            self.episodes += len(rows)
            self.successes += int(rows[:,6].sum())
            self.failures += int(rows[:,7].sum())
            emit(self.feedback_file, dict(control_step=self.owner.control_steps,
                **token, rows=rows,
                columns=["scored_steps","joint_good_steps","velocity_xy_rmse_m_s","yaw_rmse_rad_s",
                         "ee_position_rmse_m","ee_orientation_rmse_rad","success","native_failure","timeout"]))
            self.pending = None
        else:
            self.owner.censor(cpu_ids)
        selected = self.owner.assign(cpu_ids)
        self.task_tensor[ids] = torch.from_numpy(selected).to(self.env.device)
        for c in self.cursor.values():
            c[ids] = 0
        self.metrics.reset(ids)
        emit(self.assign_file, dict(control_step=self.owner.control_steps, pool_identity=self.pool.identity,
            env_ids=cpu_ids, task_ids=selected, generation=self.owner.generation[cpu_ids],
            teacher_version=self.owner.teacher_version[cpu_ids], decision_ids=self.owner.decision_id[cpu_ids],
            selected_probability=self.owner.selected_probability[cpu_ids]))
        return self.original_reset(ids)

    def summary(self):
        return dict(pool_identity=self.pool.identity, control_steps=self.owner.control_steps,
                    accepted=self.owner.accepted, episodes=self.episodes, successes=self.successes,
                    failures=self.failures, teacher_version=self.owner.engine.version,
                    distribution=distribution_stats(self.owner.engine.probability),
                    observed_sequences=int((self.owner.engine.last_feedback_step>=0).sum()),
                    effective_commands=self.effective())

    def state_dict(self):
        state = super().state_dict()
        state["command_stats"] = {k:{f:v.detach().cpu().clone() for f,v in s.items()} for k,s in self.command_stats.items()}
        state["counts"] = dict(episodes=self.episodes, successes=self.successes, failures=self.failures,
                              standing_draws=int(self.standing_draws), velocity_draws=int(self.velocity_draws))
        return state

    def restore(self, state):
        self.owner.load_state_dict(state["owner"])
        self.metrics.load_state_dict(state["metrics"])
        self.task_tensor.copy_(state["task_tensor"].to(self.env.device))
        for k,v in state["cursor"].items():
            self.cursor[k].copy_(v.to(self.env.device))
        for name, values in state["terms"].items():
            term=self.env.command_manager.get_term(name)
            for key,value in values.items():
                getattr(term,key).copy_(value.to(self.env.device))
        for k,values in state["command_stats"].items():
            for name,value in values.items():
                self.command_stats[k][name].copy_(value.to(self.env.device))
        for name,value in state["counts"].items():
            target=getattr(self,name)
            if isinstance(target,torch.Tensor):
                target.fill_(value)
            else:
                setattr(self,name,value)
        self.owner.check()
