"""CPU-only proof of a proposed appended terrain observation migration.

Loads the verified native CP3000 read-only, creates expanded MLPs in memory,
checks exact parameter mappings and random forward equivalence, and writes only
a JSON proof. It does not install dependencies, start Isaac Sim, write a model,
modify task observations, or perform optimization. Run with the existing native
Python environment and the native resolved agent.yaml.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import torch
from tensordict import TensorDict
import yaml
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization


CP3000_SHA = '600a2dab9caf18289bb6ec558f86c4e6e86c352b621f9a6d83ff73f209cc08b6'
NATIVE_LAYOUT = {
    'actor': [('base_ang_vel', 0, 9), ('projected_gravity', 9, 18),
              ('joint_pos', 18, 72), ('joint_vel', 72, 126), ('actions', 126, 180),
              ('velocity_commands', 180, 189), ('pos_commands', 189, 210)],
    'critic': [('base_lin_vel', 0, 3), ('base_ang_vel', 3, 6),
               ('projected_gravity', 6, 9), ('velocity_commands', 9, 12),
               ('pos_commands', 12, 19), ('joint_pos', 19, 37), ('joint_vel', 37, 55),
               ('actions', 55, 73), ('joint_torques', 73, 91), ('feet_contact', 91, 95),
               ('ee_link0_rel_pose', 95, 102)],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def norm_metadata(model):
    norm = model.obs_normalizer
    result = {'class': type(norm).__module__+'.'+type(norm).__name__,
              'obs_normalization': model.obs_normalization}
    for name in ('eps', 'until'):
        if hasattr(norm, name):
            result[name] = getattr(norm, name)
    if hasattr(norm, 'count'):
        result['count'] = int(norm.count)
    return result


def model_kwargs(agent_cfg, name):
    kwargs = copy.deepcopy(agent_cfg[name])
    if kwargs.pop('class_name') != 'MLPModel':
        raise ValueError('Only the native MLPModel is supported')
    if kwargs['hidden_dims'] != [512, 256, 128] or kwargs['activation'] != 'elu':
        raise ValueError('Unexpected native network architecture')
    return kwargs


def migrate_model(old, new, old_dim):
    """Append scanner columns without reordering the old observation groups."""
    old_state, new_state = old.state_dict(), new.state_dict()
    first_name, first_layer = next((name, layer) for name, layer in old.named_modules()
                                   if isinstance(layer, torch.nn.Linear))
    first_weight = first_name+'.weight'
    if first_layer.in_features != old_dim:
        raise ValueError('Native first layer input mismatch')
    if set(old_state) != set(new_state):
        raise ValueError('Unexpected state keys; do not drop saved model state')
    copied, extended = [], []
    for key, value in old_state.items():
        if key == first_weight:
            new_state[key].zero_()
            new_state[key][:, :old_dim].copy_(value)
            extended.append(key)
        elif key.startswith('obs_normalizer.') and value.ndim == 2 and value.shape[-1] == old_dim:
            # The verified native configuration uses Identity, so this branch
            # remains available only for an explicitly normalized future source.
            if key.endswith('._mean'):
                new_state[key].zero_()
            elif key.endswith(('._std', '._var')):
                new_state[key].fill_(1.)
            else:
                raise ValueError('Unknown per-feature normalization statistic '+key)
            new_state[key][..., :old_dim].copy_(value)
            extended.append(key)
        else:
            if new_state[key].shape != value.shape:
                raise ValueError('Unexpected changed parameter shape '+key)
            new_state[key].copy_(value)
            copied.append(key)
    new.load_state_dict(new_state, strict=True)
    current = new.state_dict()
    assert torch.equal(current[first_weight][:, :old_dim], old_state[first_weight])
    assert torch.count_nonzero(current[first_weight][:, old_dim:]).item() == 0
    assert all(torch.equal(current[key], old_state[key]) for key in copied)
    if isinstance(old.obs_normalizer, EmpiricalNormalization):
        # eps/until are Python attributes, not entries in the state dictionary.
        new.obs_normalizer.eps = old.obs_normalizer.eps
        new.obs_normalizer.until = old.obs_normalizer.until
    assert norm_metadata(old) == norm_metadata(new)
    return {'first_weight': first_weight, 'first_shape_old': list(old_state[first_weight].shape),
            'first_shape_new': list(current[first_weight].shape),
            'copied_state_keys': copied, 'extended_state_keys': extended,
            'old_columns_bitwise_equal': True, 'new_columns_exact_zero': True,
            'remaining_state_bitwise_equal': True, 'normalizer_metadata': norm_metadata(new)}


def forward_checks(old, new, group, old_dim, ray_count, generator):
    rows = []
    for dtype in (torch.float32, torch.float64):
        old_test, new_test = copy.deepcopy(old).to(dtype), copy.deepcopy(new).to(dtype)
        old_test.eval(); new_test.eval()
        # A different GEMM reduction shape can introduce fp32 roundoff despite
        # exact column identity. Require tight numeric agreement and report
        # bitwise equality separately; fp64 detects mapping errors independently.
        tolerance = 2e-5 if dtype == torch.float32 else 1e-12
        for batch in (1, 17, 257):
            for scale in (.1, 1., 3.):
                native = torch.randn(batch, old_dim, generator=generator, dtype=dtype)*scale
                old_obs = TensorDict({group: native}, batch_size=[batch])
                for ray_mode in ('zeros', 'random'):
                    rays = torch.zeros(batch, ray_count, dtype=dtype) if ray_mode == 'zeros' else torch.randn(
                        batch, ray_count, generator=generator, dtype=dtype)*scale
                    new_obs = TensorDict({group: native, 'terrain_scan': rays}, batch_size=[batch])
                    with torch.no_grad():
                        expected = old_test(old_obs)
                        actual = new_test(new_obs)
                        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance,
                                                   equal_nan=False, check_dtype=True)
                        # Distribution std and mean must be preserved too. Use
                        # update directly, avoiding stochastic action draws.
                        old_raw = old_test.mlp(old_test.get_latent(old_obs))
                        new_raw = new_test.mlp(new_test.get_latent(new_obs))
                        distribution_pass = None
                        if old_test.distribution is not None:
                            old_test.distribution.update(old_raw)
                            new_test.distribution.update(new_raw)
                            torch.testing.assert_close(new_test.output_mean, old_test.output_mean,
                                                       rtol=tolerance, atol=tolerance)
                            torch.testing.assert_close(new_test.output_std, old_test.output_std,
                                                       rtol=0., atol=0.)
                            distribution_pass = True
                    rows.append({'dtype': str(dtype), 'batch': batch, 'input_scale': scale,
                                 'ray_mode': ray_mode, 'max_abs_error': float((actual-expected).abs().max()),
                                 'bitwise_equal': torch.equal(actual, expected),
                                 'atol': tolerance, 'rtol': tolerance,
                                 'distribution_mean_std_pass': distribution_pass, 'pass': True})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--agent-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-sha256', default=CP3000_SHA)
    parser.add_argument('--ray-count', type=int, default=63)
    parser.add_argument('--seed', type=int, default=7101)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Refusing to overwrite an existing proof')
    if not 1 <= args.ray_count <= 4096:
        parser.error('Invalid ray count')
    before_sha = sha(args.checkpoint)
    if before_sha != args.expected_sha256:
        raise ValueError('Checkpoint hash does not match the verified native CP3000')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    agent_cfg = yaml.safe_load(args.agent_config.read_text())
    dims = {'actor': 210, 'critic': 102}
    groups = {'actor': ['policy'], 'critic': ['critic']}
    groups_new = {name: old+['terrain_scan'] for name, old in groups.items()}
    obs = TensorDict({'policy': torch.zeros(1, 210), 'critic': torch.zeros(1, 102)}, batch_size=[1])
    new_obs = obs.clone()
    new_obs['terrain_scan'] = torch.zeros(1, args.ray_count)
    report = {'kind': 'in_memory_input_migration_proof', 'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': before_sha, 'checkpoint_iteration': cp.get('iter'),
              'script_sha256': sha(__file__), 'agent_config_sha256': sha(args.agent_config),
              'torch_version': torch.__version__, 'rsl_rl_version': importlib.metadata.version('rsl-rl-lib'),
              'model_source': inspect.getfile(MLPModel), 'model_source_sha256': sha(inspect.getfile(MLPModel)),
              'device': 'cpu', 'seed': args.seed, 'ray_count': args.ray_count,
              'observation_groups_old': groups, 'observation_groups_proposed': groups_new,
              'native_history_layout': 'Actor term-major, each term has 3 history frames; critic has one frame.',
              'new_scan_layout': 'Separate terrain_scan group, one frame, appended after the intact native group.',
              'models': {}, 'optimizer_rule': 'Fresh optimizer required; old first-layer Adam tensors have incompatible shape.',
              'checkpoint_written': False, 'task_observations_changed': False,
              'physical_scan_interface_verified_here': False,
              'limits': 'Random-vector algebra proof only; it does not validate ray order, ray validity, deployable sensing or terrain learning.'}
    for name in ('actor', 'critic'):
        kwargs = model_kwargs(agent_cfg, name)
        old = MLPModel(obs, groups, name, 18 if name == 'actor' else 1, **copy.deepcopy(kwargs)).cpu().eval()
        old.load_state_dict(cp[name+'_state_dict'], strict=True)
        assert all(torch.equal(old.state_dict()[key], value) for key, value in cp[name+'_state_dict'].items())
        new = MLPModel(new_obs, groups_new, name, 18 if name == 'actor' else 1,
                       **copy.deepcopy(kwargs)).cpu().eval()
        model_report = migrate_model(old, new, dims[name])
        model_report['native_term_column_ranges_half_open'] = NATIVE_LAYOUT[name]
        model_report['new_scan_column_range_half_open'] = [dims[name], dims[name]+args.ray_count]
        model_report['model_config_preserved'] = kwargs
        model_report['forward_checks'] = forward_checks(old, new, groups[name][0], dims[name], args.ray_count, generator)
        report['models'][name] = model_report
    report['checkpoint_unchanged_after'] = sha(args.checkpoint) == before_sha
    assert report['checkpoint_unchanged_after']
    report['passed'] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'out': str(args.out), 'sha256': sha(args.out),
                      'checks': {name: len(report['models'][name]['forward_checks']) for name in dims},
                      'maximum_errors': {name: max(row['max_abs_error'] for row in report['models'][name]['forward_checks'])
                                         for name in dims}}))


if __name__ == '__main__':
    main()
