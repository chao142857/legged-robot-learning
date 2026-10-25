"""CPU native PPO replay tests; synthetic states only, never writes checkpoints."""
import copy
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ppo_update_microscope import (
    calibration, evaluate, native_gae, norm, normal, one_step, restore, select_window,
    tensor_hash, vector,
)


def fixture(n=16, seed=703):
    """Native MLP/Adam topology with small hidden layers and nonzero moments."""
    torch.manual_seed(seed)
    model_cfg = dict(class_name='MLPModel', hidden_dims=[8, 6, 4], activation='elu',
                     obs_normalization=False,
                     distribution_cfg=dict(class_name='GaussianDistribution',
                                           init_std=1., std_type='log'))
    config = dict(actor=copy.deepcopy(model_cfg), critic=copy.deepcopy(model_cfg),
                  algorithm=dict(class_name='PPO', num_learning_epochs=5,
                                 num_mini_batches=4, clip_param=.2, gamma=.99,
                                 lam=.95, value_loss_coef=1., entropy_coef=.005,
                                 learning_rate=.0005, max_grad_norm=1.,
                                 optimizer='adam', use_clipped_value_loss=True,
                                 schedule='adaptive', desired_kl=.01,
                                 normalize_advantage_per_mini_batch=False,
                                 share_cnn_encoders=False, rnd_cfg=None,
                                 symmetry_cfg=None))
    obs = TensorDict(dict(policy=torch.randn(n, 4), critic=torch.randn(n, 3)), batch_size=[n])
    actors = []
    for role, group, size in [('actor', 'policy', 18), ('critic', 'critic', 1)]:
        cfg = copy.deepcopy(model_cfg); cfg.pop('class_name')
        actors.append(MLPModel(obs, {role: [group]}, role, size, **cfg))
    actor, critic = actors
    saved_lr = 1. / 3000
    opt = torch.optim.Adam(itertools.chain(actor.parameters(), critic.parameters()), lr=saved_lr)
    # A synthetic prior update gives every used parameter a real Adam state.
    warm = actor(obs).square().mean() + critic(obs).square().mean() - .01 * actor.distribution.log_std_param.sum()
    opt.zero_grad(); warm.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
    opt.step()
    with torch.no_grad():
        mu = actor(obs); std = actor.distribution.log_std_param.exp().expand_as(mu)
        d = torch.distributions.Normal(mu, std)
        actions = d.sample(); lp = d.log_prob(actions).sum(-1)
        value = critic(obs)
        advantage = norm(torch.randn(n, 1))
        targets = value + torch.linspace(-2., 2., n).unsqueeze(-1)
    batch = RolloutStorage.Batch(observations=obs, actions=actions, values=value,
                                 advantages=advantage, returns=targets,
                                 old_actions_log_prob=lp.unsqueeze(-1),
                                 old_distribution_params=(mu, std))
    cp = copy.deepcopy(dict(actor_state_dict=actor.state_dict(),
                            critic_state_dict=critic.state_dict(),
                            optimizer_state_dict=opt.state_dict(),
                            infos=dict(learning_rate=saved_lr)))
    return cp, config, batch


def manual_step(cp, config, batch, scale=1.):
    """Independent native-formula loss, schedule, clipping and Adam step."""
    alg = restore(cp, config, batch, scale)
    mu = alg.actor(batch.observations)
    std = alg.actor.distribution.log_std_param.exp().expand_as(mu)
    d = torch.distributions.Normal(mu, std)
    with torch.no_grad():
        kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(*batch.old_distribution_params), d).sum(-1).mean()
        if kl > .02:
            alg.learning_rate = max(1e-5, alg.learning_rate / 1.5)
        elif 0 < kl < .005:
            alg.learning_rate = min(1e-2, alg.learning_rate * 1.5)
        for g in alg.optimizer.param_groups:
            g['lr'] = alg.learning_rate
    ratio = (d.log_prob(batch.actions).sum(-1) - batch.old_actions_log_prob.squeeze(-1)).exp()
    a = batch.advantages.squeeze(-1)
    pi = torch.max(-a * ratio, -a * torch.clamp(ratio, .8, 1.2)).mean()
    values = alg.critic(batch.observations)
    clipped_values = batch.values + torch.clamp(values - batch.values, -.2, .2)
    vl = torch.max((values - batch.returns).square(), (clipped_values - batch.returns).square()).mean()
    entropy = d.entropy().sum(-1).mean()
    loss = pi + vl - .005 * entropy
    alg.optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(alg.actor.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(alg.critic.parameters(), 1.)
    alg.optimizer.step()
    return alg


def batch_with_kl(cp, config, batch, target):
    b = copy.deepcopy(batch)
    a = restore(cp, config, b).actor
    with torch.no_grad():
        mu = a(b.observations); std = a.distribution.log_std_param.exp().expand_as(mu)
        old_mu = mu + math.sqrt(2. * target / 18.) * std
        b.old_distribution_params = (old_mu, std)
        b.old_actions_log_prob = torch.distributions.Normal(old_mu, std).log_prob(b.actions).sum(-1, keepdim=True)
    return b


class MicroscopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gae_matches_native_process_and_compute_returns(self):
        cp, config, batch = fixture(n=4)
        alg = restore(cp, config, batch)
        t, n = 6, 4
        st = RolloutStorage('rl', n, t, batch.observations, [18], 'cpu')
        alg.storage = st
        torch.manual_seed(711)
        rewards = torch.randn(t, n, 1)
        values = torch.randn(t, n, 1)
        dones = torch.zeros(t, n, dtype=torch.bool)
        timeouts = torch.zeros_like(dones)
        dones[1, 0] = True                  # true termination followed by a new episode
        dones[3, 1] = True; timeouts[3, 1] = True
        dones[-1, 2] = True                # terminal final step
        dones[-1, 3] = True; timeouts[-1, 3] = True
        for i in range(t):
            tr = alg.transition
            tr.observations = batch.observations
            tr.actions = batch.actions
            tr.values = values[i].clone()
            tr.actions_log_prob = batch.old_actions_log_prob.squeeze(-1)
            tr.distribution_params = batch.old_distribution_params
            alg.process_env_step(batch.observations, rewards[i, :, 0], dones[i], {'time_outs': timeouts[i]})
        expected_rewards = rewards + .99 * values * timeouts.unsqueeze(-1)
        torch.testing.assert_close(st.rewards, expected_rewards, rtol=0, atol=0)
        with torch.no_grad():
            last = alg.critic(batch.observations)
        raw, ret = native_gae(rewards, values, last, dones, timeouts)
        alg.compute_returns(batch.observations)
        torch.testing.assert_close(ret, st.returns, rtol=0, atol=0)
        torch.testing.assert_close(norm(raw), st.advantages, rtol=0, atol=0)
        finite, _ = native_gae(rewards, values, last, dones, timeouts, timeout_current=False)
        self.assertGreater(float((raw-finite).abs().max()), .01)

    def test_missing_bootstrap_refused_except_final_done(self):
        r = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 10
        v = torch.ones_like(r)
        done = torch.zeros(3, 2, dtype=torch.bool); timeout = torch.zeros_like(done)
        with self.assertRaisesRegex(ValueError, 'Missing nonterminal bootstrap'):
            native_gae(r, v, None, done, timeout)
        done[-1] = True; timeout[-1, 0] = True
        got, _ = native_gae(r, v, None, done, timeout)
        expected, _ = native_gae(r, v, torch.full((2, 2), 9876.), done, timeout)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        timeout[0, 1] = True
        with self.assertRaisesRegex(ValueError, 'Timeout must be done'):
            native_gae(r, v, None, done, timeout)

    def test_component_gae_sum_is_scalar_gae_with_summed_baseline(self):
        torch.manual_seed(712)
        r = torch.randn(5, 3, 3); v = torch.randn_like(r); last = torch.randn(3, 3)
        done = torch.zeros(5, 3, dtype=torch.bool); to = torch.zeros_like(done)
        done[2, 0] = True; done[-1, 1] = True; to[-1, 1] = True
        for convention in (True, False):
            pieces, _ = native_gae(r, v, last, done, to, timeout_current=convention)
            total, _ = native_gae(r.sum(-1, keepdim=True), v.sum(-1, keepdim=True),
                                  last.sum(-1, keepdim=True), done, to, timeout_current=convention)
            torch.testing.assert_close(pieces.sum(-1), total.squeeze(-1), rtol=3e-6, atol=2e-6)

    def test_window_selection_preserves_bootstrap_and_first_episode_boundary(self):
        valid = torch.ones(50, 4, dtype=torch.bool)
        end = torch.zeros_like(valid)
        valid[8:, 1] = False; end[7, 1] = True
        valid[24:, 2] = False; end[23, 2] = True
        valid[24:, 3] = False              # no valid next state AND no terminal evidence
        raw = dict(valid=valid, episode_end=end)
        ids, info = select_window(raw, 0)
        self.assertEqual(ids.tolist(), [0, 2])
        self.assertEqual(info['excluded_envs'], 2)
        with self.assertRaisesRegex(ValueError, 'No eligible complete window'):
            select_window(raw, 26)
        end[-1, 0] = True
        ids, info = select_window(raw, 26)
        self.assertEqual(ids.tolist(), [0])
        self.assertEqual(info['terminal_last_without_next'], 1)
        self.assertTrue(info['survivor_conditioned'])
        with self.assertRaisesRegex(ValueError, 'Incomplete rollout window'):
            select_window(raw, 48)

    def test_normalization_decomposition_and_native_reference(self):
        torch.manual_seed(713)
        t, n = 48, 5
        r = torch.randn(t, n, 3); value = torch.randn(t, n)
        valid = torch.ones(t, n, dtype=torch.bool); end = torch.zeros_like(valid); to = torch.zeros_like(valid)
        end[23, :2] = True; to[23, 0] = True; valid[24:, :2] = False
        end[-1, 2:] = True; to[-1, 2:] = True
        raw = dict(component_rewards=r, total_reward=r.sum(-1), native_value=value,
                   valid=valid, episode_end=end, timeout=to)
        probe = torch.randn(t, n, 3)
        ids, an, ret, pc, pt, info = calibration(raw, probe, 0)
        self.assertEqual(ids.tolist(), list(range(n)))
        self.assertEqual(info['native_reference_max_error'], 0.)
        self.assertLess(info['decomposition_max_error'], 2e-5)
        sigma = an.std() + 1e-8
        center = lambda x: x-x.mean()
        baseline = center(pc.sum(-1)-an)/sigma
        timeout = center(pt.sum(-1)-pc.sum(-1))/sigma
        scale = norm(pt.sum(-1)) - center(pt.sum(-1))/sigma
        torch.testing.assert_close(norm(pt.sum(-1))-norm(an), baseline+timeout+scale, rtol=2e-5, atol=2e-5)
        self.assertGreater(info['raw_timeout_difference_rms'], 0.)
        # The terminal-only last window needs no fabricated observation at t=48.
        ids, _, _, _, _, tail = calibration(raw, probe, 24)
        self.assertEqual(ids.tolist(), [2, 3, 4])
        self.assertEqual(tail['terminal_last_without_next'], 3)

    def test_native_one_step_matches_independent_manual_adam(self):
        cp, cfg, b = fixture()
        cp_hash = tensor_hash(cp)
        native, delta, info, _, _ = one_step(cp, cfg, b, 1., 714)
        manual = manual_step(cp, cfg, b)
        for role in ('actor', 'critic'):
            for actual, expected in zip(getattr(native, role).parameters(), getattr(manual, role).parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tensor_hash(native.optimizer.state_dict()), tensor_hash(manual.optimizer.state_dict()))
        self.assertEqual(cp_hash, tensor_hash(cp))
        self.assertEqual(info['optimizer_steps'], 1)
        self.assertTrue(info['unused_critic_std_grad_none'])
        self.assertGreater(float(delta.norm()), 0.)
        unused = native.critic.distribution.log_std_param
        self.assertNotIn(unused, native.optimizer.state)

    def test_paired_tenth_lr_same_moments_and_scaled_delta(self):
        cp, cfg, b = fixture()
        cp_hash = tensor_hash(cp)
        full, df, fi, _, _ = one_step(cp, cfg, b, 1., 715)
        small, ds, si, _, _ = one_step(cp, cfg, b, .1, 715)
        self.assertEqual(fi['initial_model_moments_sha256'], si['initial_model_moments_sha256'])
        self.assertAlmostEqual(si['lr_actual']/fi['lr_actual'], .1, places=12)
        torch.testing.assert_close(ds, df*.1, rtol=2e-3, atol=3e-8)
        self.assertEqual(tensor_hash(full.optimizer.state_dict()['state']), tensor_hash(small.optimizer.state_dict()['state']))
        self.assertEqual(cp_hash, tensor_hash(cp))
        # Restored moment tensors must not alias each other or original saved state.
        a = restore(cp, cfg, b); c = restore(cp, cfg, b)
        pa = next(p for p in a.actor.parameters() if p in a.optimizer.state)
        pc = next(p for p in c.actor.parameters() if p in c.optimizer.state)
        a.optimizer.state[pa]['exp_avg'].add_(7)
        self.assertFalse(torch.equal(a.optimizer.state[pa]['exp_avg'], c.optimizer.state[pc]['exp_avg']))
        self.assertEqual(cp_hash, tensor_hash(cp))

    def test_native_adaptive_lr_zero_low_middle_high_and_bounds(self):
        cp, cfg, b = fixture()
        saved = cp['infos']['learning_rate']
        for target, factor in [(0., 1.), (.001, 1.5), (.01, 1.), (.1, 1/1.5)]:
            bb = batch_with_kl(cp, cfg, b, target)
            _, _, info, _, _ = one_step(cp, cfg, bb, 1., 716)
            self.assertAlmostEqual(info['lr_actual'], saved*factor, places=12)
        for lr, target, expected in [(1e-5, .1, 1e-5), (.009, .001, .01)]:
            cc = copy.deepcopy(cp)
            cc['infos']['learning_rate'] = lr
            cc['optimizer_state_dict']['param_groups'][0]['lr'] = lr
            bb = batch_with_kl(cc, cfg, b, target)
            _, _, info, _, _ = one_step(cc, cfg, bb, 1., 717)
            self.assertAlmostEqual(info['lr_actual'], expected, places=12)

    def test_evaluate_accumulates_diagnostic_losses_in_float64(self):
        cp, cfg, b = fixture(n=16, seed=718)
        alg = restore(cp, cfg, b)
        with torch.no_grad():
            mu = alg.actor(b.observations)
            std = alg.actor.distribution.log_std_param.exp().expand_as(mu)
            d = torch.distributions.Normal(mu, std)
            lp = d.log_prob(b.actions).sum(-1)
            # Deliberately exercise both clipping sides with deterministic ratios.
            desired = torch.linspace(.6, 1.4, len(b.actions))
            b.old_actions_log_prob = (lp-desired.log()).unsqueeze(-1)
            ratio = (lp-b.old_actions_log_prob.squeeze(-1)).exp().double()
            raw = torch.randn(len(b.actions), 4, 3)*.05 + 10000.
            centered = (raw-raw.mean(0))/.7
        result, samples = evaluate(alg.actor, alg.critic, b, raw, centered)
        for name, targets in [('raw', raw), ('centered_native_std', centered)]:
            expected = -ratio[:, None, None]*targets.double()
            clipped = torch.maximum(expected, -ratio.clamp(.8, 1.2)[:, None, None]*targets.double())
            self.assertEqual(samples[name].dtype, torch.float64)
            torch.testing.assert_close(samples[name], expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.tensor(result[name]['unclipped_loss'], dtype=torch.float64),
                                       expected.mean(0), rtol=0, atol=0)
            torch.testing.assert_close(torch.tensor(result[name]['separate_clipped_loss'], dtype=torch.float64),
                                       clipped.mean(0), rtol=0, atol=0)
        advantage = b.advantages.squeeze(-1).double()
        native = torch.maximum(-ratio*advantage, -ratio.clamp(.8, 1.2)*advantage)
        self.assertEqual(samples['native'].dtype, torch.float64)
        torch.testing.assert_close(samples['native'], native, rtol=0, atol=0)
        self.assertEqual(result['native_surrogate'], float(native.mean()))
        self.assertEqual(result['native_surrogate'], float(samples['native'].mean()))

    def test_ambiguous_checkpoint_lr_fails_closed(self):
        cp, cfg, b = fixture()
        cp['infos']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'Ambiguous checkpoint learning rate'):
            restore(cp, cfg, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
