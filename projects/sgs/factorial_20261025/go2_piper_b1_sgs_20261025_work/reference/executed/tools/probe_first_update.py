"""Explicitly instrumented T0 2-update/128-env/seed-21 native-PPO diagnostic.

Reads the frozen run_suite.py and injects a telemetry subclass after its imports.
No on-disk suite source, PPO loss, LR, optimizer state, observation or reward is
changed. The diagnostic is labelled as instrumented in protocol.json and has a
separate probe_manifest.json. This file does not launch anything when imported.
"""
from __future__ import annotations
from publication_paths import research_path

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

NATIVE_CP = research_path('go2_piper_upstream_wbc_20261023/formal_seed21/cp_3000.pt')
NATIVE_SHA = '600a2dab9caf18289bb6ec558f86c4e6e86c352b621f9a6d83ff73f209cc08b6'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def instrument(base_class, metadata):
    """Called only after AppLauncher and the frozen suite's imports complete."""
    import inspect
    import torch
    from telemetry import model_sha, plain, write_json

    class FirstUpdateProbe(base_class):
        def __init__(self, runner, out):
            self.probe_log = (Path(out)/'first_update_probe.jsonl').open('x', buffering=1)
            self.probe_minibatch = 0
            self.probe_optimizer_steps = 0
            self.probe_update_index = 0
            self.probe_metadata = dict(metadata)
            self.probe_metadata['runtime_sources'] = {}
            for name, obj in [('ppo', type(runner.alg)), ('actor', type(runner.alg.actor)),
                              ('distribution', type(runner.alg.actor.distribution))]:
                path = inspect.getfile(obj)
                self.probe_metadata['runtime_sources'][name] = dict(path=path, sha256=sha(path))
            self.probe_metadata['normalization'] = {
                name:dict(enabled=getattr(getattr(runner.alg,name),'obs_normalization',None),
                          class_name=type(getattr(runner.alg,name).obs_normalizer).__name__)
                for name in ('actor','critic')}
            self.probe_metadata['distribution'] = dict(class_name=type(runner.alg.actor.distribution).__name__,
                std_type=getattr(runner.alg.actor.distribution,'std_type',None))
            self.probe_metadata['optimizer_initial_state_entries'] = len(runner.alg.optimizer.state)
            self.probe_metadata['actor_initial_sha256'] = model_sha(runner.alg.actor)
            self.probe_metadata['critic_initial_sha256'] = model_sha(runner.alg.critic)
            write_json(Path(out)/'probe_manifest.json', self.probe_metadata)
            protocol_path = Path(out)/'protocol.json'
            protocol = json.loads(protocol_path.read_text())
            protocol['diagnostic_instrumentation'] = self.probe_metadata
            write_json(protocol_path, protocol)
            self.original_optimizer_step = runner.alg.optimizer.step
            super().__init__(runner, out)
            runner.alg.optimizer.step = self.optimizer_step
            if runner.alg.actor.obs_dim != 210 or runner.alg.critic.obs_dim != 102:
                raise RuntimeError('Probe accepts only unchanged native 210D/102D observations')

        def emit(self, event, **values):
            # Preserve the numerical evidence even when it contains non-finite
            # diagnostics, leaving native failure handling in charge of stopping.
            def clean(x):
                if isinstance(x, float) and not math.isfinite(x):
                    return str(x)
                if isinstance(x, dict):
                    return {k:clean(v) for k,v in x.items()}
                if isinstance(x, (tuple,list)):
                    return [clean(v) for v in x]
                return x
            self.probe_log.write(json.dumps(clean(dict(event=event, monotonic_s=time.monotonic(),
                                  update=self.probe_update_index, **values)), allow_nan=False)+'\n')

        def adam_steps(self):
            steps=[]
            for state in self.runner.alg.optimizer.state.values():
                if 'step' in state:
                    step=state['step']
                    steps.append(float(step.item()) if isinstance(step,torch.Tensor) else float(step))
            return dict(state_entries=len(self.runner.alg.optimizer.state), parameter_step_count=len(steps),
                        minimum=min(steps) if steps else None, maximum=max(steps) if steps else None)

        def kl(self, *args, **kwargs):
            result = super().kl(*args, **kwargs)
            old = args[0] if args else kwargs['old_params']
            new = args[1] if len(args)>1 else kwargs['new_params']
            with torch.no_grad():
                old_mean, old_std = old
                new_mean, new_std = new
                # Independent analytic KL(old || new), computed in fp64 from
                # native inputs. No forward, sampling, normalization or update.
                om, os = old_mean.double(), old_std.double()
                nm, ns = new_mean.double(), new_std.double()
                mean_part = .5*((om-nm)/ns).square().sum(-1)
                variance_ratio = (os/ns).square()
                variance_part = .5*(variance_ratio-1.-variance_ratio.log()).sum(-1)
                analytic = mean_part+variance_part
                self.emit('minibatch_kl_before_native_adaptation', minibatch=self.probe_minibatch,
                    learning_rate_entering=float(self.runner.alg.learning_rate),
                    optimizer_group_lrs_entering=[group['lr'] for group in self.runner.alg.optimizer.param_groups],
                    native_kl_mean=float(result.mean()), native_kl_max=float(result.max()),
                    analytic_kl_mean_fp64=float(analytic.mean()),
                    native_vs_analytic_max_abs=float((result.double()-analytic).abs().max()),
                    kl_mean_shift_component=float(mean_part.mean()), kl_variance_component=float(variance_part.mean()),
                    old_mean_abs_mean=float(om.abs().mean()), new_mean_abs_mean=float(nm.abs().mean()),
                    mean_change_abs_max=float((om-nm).abs().max()),
                    mean_change_rms_per_action=plain((om-nm).square().mean(0).sqrt()),
                    old_std_min=float(os.min()), old_std_max=float(os.max()),
                    new_std_min=float(ns.min()), new_std_max=float(ns.max()),
                    old_std_first_sample=plain(os[0]), new_std_first_sample=plain(ns[0]),
                    std_change_abs_max=float((os-ns).abs().max()),
                    adam_before=self.adam_steps())
            self.probe_minibatch += 1
            return result

        def optimizer_step(self, *args, **kwargs):
            before = self.adam_steps()
            lrs = [group['lr'] for group in self.runner.alg.optimizer.param_groups]
            # Only inspect the first two steps in each update at parameter level;
            # per-minibatch KL already records the full distribution trajectory.
            old = [p.detach().clone() for p in self.runner.alg.actor.parameters()] if self.probe_optimizer_steps < 2 else None
            result = self.original_optimizer_step(*args, **kwargs)
            delta = None
            if old is not None:
                with torch.no_grad():
                    diffs=[p.detach()-q for p,q in zip(self.runner.alg.actor.parameters(),old)]
                    delta=dict(l2=float(torch.stack([v.square().sum() for v in diffs]).sum().sqrt()),
                               abs_max=float(torch.stack([v.abs().max() for v in diffs]).max()))
            self.emit('optimizer_step_completed', optimizer_step_in_update=self.probe_optimizer_steps,
                      learning_rates_used=lrs, adam_before=before, adam_after=self.adam_steps(),
                      actor_parameter_change=delta)
            self.probe_optimizer_steps += 1
            return result

        def update(self, *args, **kwargs):
            if self.update_count >= 2:
                raise RuntimeError('Probe refuses a third PPO update')
            self.probe_update_index = self.update_count+1
            self.probe_minibatch = 0
            self.probe_optimizer_steps = 0
            self.emit('update_begin', actor_sha256=model_sha(self.runner.alg.actor),
                      critic_sha256=model_sha(self.runner.alg.critic), adam=self.adam_steps(),
                      learning_rate=float(self.runner.alg.learning_rate))
            try:
                return super().update(*args, **kwargs)
            finally:
                # Keep the update index stable across the base counter increment.
                self.emit('update_end', completed_updates=self.update_count,
                          minibatch_calls=self.probe_minibatch, optimizer_step_calls=self.probe_optimizer_steps,
                          actor_sha256=model_sha(self.runner.alg.actor), critic_sha256=model_sha(self.runner.alg.critic),
                          adam=self.adam_steps(), learning_rate=float(self.runner.alg.learning_rate))

        def close(self):
            self.runner.alg.optimizer.step = self.original_optimizer_step
            try:
                super().close()
            finally:
                self.probe_log.close()

    return FirstUpdateProbe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path(NATIVE_CP))
    parser.add_argument('--num_envs', type=int, choices=[128], default=128)
    parser.add_argument('--updates', type=int, choices=[2], default=2)
    parser.add_argument('--family', choices=['T0'], default='T0')
    parser.add_argument('--seed', type=int, choices=[21], default=21)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action='store_true')
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Refusing an existing output directory')
    if not args.admission.is_file() or not args.checkpoint.is_file():
        parser.error('Existing exact admission and native checkpoint are required')
    if sha(args.checkpoint) != NATIVE_SHA:
        parser.error('Probe requires the verified original CP3000, not a smoke checkpoint')
    suite=Path(__file__).resolve().with_name('run_suite.py')
    source=suite.read_text()
    marker='\n\ndef evaluate('
    if source.count(marker)!=1:
        parser.error('Frozen suite injection point is ambiguous')
    injection='\nfrom probe_first_update import instrument as _instrument_probe\nTrainingTelemetry = _instrument_probe(TrainingTelemetry, _PROBE_METADATA)\n'
    injected=source.replace(marker,injection+marker,1)
    metadata=dict(kind='instrumented_diagnostic_not_pristine_suite', read_only_observer=True,
        wrapper_path=str(Path(__file__).resolve()),wrapper_sha256=sha(__file__),
        original_suite_path=str(suite),original_suite_sha256=sha(suite),
        instrumented_source_sha256=hashlib.sha256(injected.encode()).hexdigest(),
        injection=injection, checkpoint_sha256=sha(args.checkpoint),
        admission_path=str(args.admission.resolve()),admission_sha256=sha(args.admission),
        limits=dict(family='T0',terrain='plane',ee_mode='smooth',updates=2,num_envs=128,seed=21),
        observation='Native minibatch KL inputs, optimizer step counters, LR before/after native adaptation, and model-state hashes only.',
        optimizer_lr_or_network_changed=False,additional_policy_forward_or_rng_draw=False,
        scope='Diagnostic replay only; never grants stable-recipe, safety or long-training approval.')
    sys.argv=[str(suite),'--mode','train','--family','T0','--terrain','plane','--ee_mode','smooth',
              '--num_envs','128','--updates','2','--seed','21','--checkpoint',str(args.checkpoint),
              '--admission',str(args.admission),'--out',str(args.out),'--device',args.device]
    if args.headless:
        sys.argv.append('--headless')
    scope={'__name__':'__main__','__file__':str(suite),'_PROBE_METADATA':metadata}
    exec(compile(injected,str(suite),'exec'),scope)


if __name__=='__main__':
    main()
