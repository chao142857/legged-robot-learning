"""Native pre-autoreset boundary adapter; imports no simulator at module import.

Uses B2's single raw-metric accumulator and command hook. No controller, reward,
termination, observation getter, network, optimizer or scheduler is replaced.
"""
from dataclasses import asdict
import json

from shared_sgs import TerminalFeedback, T1_RULE_VERSION, t1_episode_labels


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, 'item'):
        return x.item()
    return x


class PreResetObserver:
    def __init__(self, env, session, metrics, physical, *, log=None):
        self.raw, self.session, self.metrics, self.physical = env.unwrapped, session, metrics, physical
        self.log = log
        self.original_reward = self.raw.reward_manager.compute
        self.original_reset = self.raw._reset_idx
        self.pending = []
        self.steps = 0
        self.raw.reward_manager.compute = self.reward
        self.raw._reset_idx = self.reset

    def emit(self, kind, **data):
        if self.log is not None:
            self.log.write(json.dumps(dict(kind=kind, **plain(data)), allow_nan=False)+'\n')

    def reward(self, *args, **kwargs):
        if self.pending:
            raise RuntimeError('Previous terminal batch was not reset')
        result = self.original_reward(*args, **kwargs)
        raw = self.raw
        p = self.physical(raw)
        def cpu(name):
            return p[name].detach().cpu().numpy()
        self.metrics.capture(range(raw.num_envs), base_command=cpu('base_cmd'), base_actual=cpu('base_actual'),
            ee_position_error=cpu('ee_position_error'), ee_orientation_error=cpu('ee_orientation_error'),
            base_clearance=cpu('base_clearance'), joint_velocity_limit_ratio=cpu('joint_velocity_limit_ratio'),
            joint_position_limit_excess=cpu('joint_position_limit_excess'), nonfoot_max_force=cpu('nonfoot_max_force'),
            failure=raw.reset_terminated.detach().cpu().numpy())
        self.session.advance_all()
        self.steps += 1
        ids = raw.reset_buf.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for i in sorted(ids):
            token = self.session.assignment(i)
            row = plain(self.metrics.finish(i))
            if row['steps'] != int(raw.episode_length_buf[i]) or row['steps'] != self.session.episode_steps(i):
                raise RuntimeError('Pre-reset metric age does not match old episode')
            terminated, timeout = bool(raw.reset_terminated[i]), bool(raw.reset_time_outs[i])
            labels = t1_episode_labels(row, terminated=terminated, timeout=timeout, max_steps=raw.max_episode_length)
            row.update(labels)
            reasons = tuple(n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[i]))
            feedback = TerminalFeedback(token, terminated, timeout, labels['joint_success_5cm'], row['steps'],
                                        T1_RULE_VERSION, row, reasons)
            self.pending.append(feedback)
        return result

    def reset(self, env_ids):
        ids = sorted(env_ids.detach().cpu().tolist() if hasattr(env_ids, 'detach') else list(env_ids))
        if not ids:
            return self.original_reset(env_ids)
        if self.pending:
            if ids != sorted(r.assignment.env_id for r in self.pending):
                raise RuntimeError('Native reset batch differs from snapshotted terminals')
            self.session.observe_terminal(self.pending)
            for row in self.pending:
                self.emit('terminal', **asdict(row), truncated=row.timeout)
            self.pending = []
        else:
            for row in self.session.censor_active('manual_reset', ids):
                self.emit('censored', **row)
        probabilities = self.session.probabilities()
        assigned = self.session.assign(ids)
        self.emit('reset_batch', env_ids=ids, probabilities=dict(zip(self.session.pool.ids, probabilities)))
        phases = [self.session.pool.by_id[a.task_id].content['base_sequence_phase_offset'] for a in assigned]
        self.raw.command_manager.get_term('base_velocity').bind_phases(ids, phases)
        for a in assigned:
            self.emit('assignment', **asdict(a), probability=probabilities[self.session.pool.index[a.task_id]])
        # Original method resets states, command age, action histories and buffers.
        # Its command reset sees the already-bound new phase, after old feedback.
        result = self.original_reset(env_ids)
        self.metrics.reset(ids)
        return result

    def state_dict(self):
        if self.pending:
            raise RuntimeError('Checkpoint requires a completed native step boundary')
        return dict(version=1, steps=self.steps, sampler=self.session.state_dict(), metrics=self.metrics.state_dict(),
                    restore_scope='module state only; simulator/command/action state or explicit censor-and-reset required')

    def close(self):
        self.raw.reward_manager.compute = self.original_reward
        self.raw._reset_idx = self.original_reset
