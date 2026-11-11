"""Engineering and paired-noise audit; development outcomes never gate expansion."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT, file_sha, write
from teacher_v02 import array_hash

EXPERIENCE=Path('/experience')


def read(path): return json.loads(Path(path).read_text())


def paired(a,b,seed):
    """Resample sequence blocks, keeping its two fixed resets together."""
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    assert a.shape==b.shape==(512,) and np.isfinite(a).all() and np.isfinite(b).all()
    difference=(a-b).reshape(2,256).mean(0)
    rng=np.random.default_rng(seed)
    means=difference[rng.integers(0,256,size=(2000,256))].mean(1)
    return dict(mean_difference=float(difference.mean()),
        sequence_paired_mean_absolute_difference=float(abs(difference).mean()),
        sequence_paired_rms_difference=float(np.sqrt(np.square(difference).mean())),
        bootstrap95_sequence_interval=np.quantile(means,[.025,.975]).tolist(),
        repeat_reset_mean_absolute_difference=float(abs(a-b).mean()),
        changed_reset_count=int((abs(a-b)>1e-12).sum()),
        scope='Finite development-sequence uncertainty only; does not quantify training-seed or repeated-PPO variance.')


def evaluation(label,expected_update):
    path=ROOT/'evaluations'/label/'result.json'; result=read(path)
    assert result['passed'] and result['kind']=='development'
    assert result['update']==expected_update and result['episodes']==512
    assert result['training_history_updates']==0 and result['model_weights_unchanged']
    assert len(result['scores'])==512 and np.isfinite(result['scores']).all()
    assert abs(float(np.mean(result['scores']))-result['continuous_joint_score'])<1e-12
    return result,dict(path=str(path),sha256=file_sha(path))


def audit(index):
    source=read(ROOT/'sources.json')['sources'][index]
    u,seed=source['updates'],source['seed']
    label=f'p{index:02d}-seed{seed}-cp{u:05d}'
    point=EXPERIENCE/'meta-eight-starts'/f'point_{index:02d}'
    pair=torch.load(point/'decision.pt',map_location='cpu',weights_only=False)
    assert file_sha(Path(source['path']))==source['sha256']
    assert pair['source_original']['sha256']==source['sha256']
    before,before_file=evaluation(label+'-before',u)
    assert pair['before_development_sha256']==before_file['sha256']
    identity={k:before[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}
    basis=np.asarray(pair['basis']); assert basis.shape==(32768,8)
    assert array_hash(basis)==pair['basis_sha256']
    arms=['uniform','plus','minus']+(['uniform-repeat'] if index<2 else [])
    summaries={}; results={}; receipts={}; counts_by_arm={}; exposure_by_arm={}
    common_restore_fields=['source_checkpoint_sha256','actor_sha256','critic_sha256',
        'optimizer_sha256','environment_sha256','global_rng_sha256',
        'adapter_restored_sha256','installed_sampler_rng_sha256','completed_updates','active_episodes']
    common_restore=None
    for arm in arms:
        arm_label=label+'-'+arm; out=ROOT/'runs'/arm_label
        run=read(out/'result.json'); receipt=read(out/'restore_receipt.json')
        assert run['passed'] and run['new_updates']==100 and run['completed_updates']==u+100
        assert run['new_environment_transitions']==100*4096*24
        assert run['optimizer_calls']==(u+100)*20
        assert receipt['passed'] and receipt['next_4096_sampler_draws_equal']
        assert receipt['source_checkpoint_sha256']==pair['source_checkpoint_sha256']
        restored={k:receipt[k] for k in common_restore_fields}
        if common_restore is None: common_restore=restored
        else: assert restored==common_restore, f'Different restorable branch input: {arm}'
        assert read(out/'native_comparison.json')['passed']
        cycle=EXPERIENCE/arm_label/'cycle_0000'
        outcome=read(cycle/'outcome.json'); committed=read(cycle/'COMMITTED.json')
        assert committed['passed'] and committed['completed_updates']==u+100
        for name,h in committed['files'].items(): assert file_sha(cycle/name)==h
        assert committed['checkpoint_sha256']==run['checkpoint_sha256']
        target='uniform' if arm=='uniform-repeat' else arm
        q=np.load(point/f'{target}.npy',allow_pickle=False)
        assert q.shape==(32768,) and np.isfinite(q).all() and np.min(q)>=.2/32768-1e-15
        assert abs(q.sum()-1)<1e-12
        if target=='uniform': assert np.array_equal(q,np.ones(32768)/32768)
        else: assert np.array_equal(q,pair[target]['probability'])
        with np.load(cycle/'real_draws.npz',allow_pickle=False) as d:
            ids=d['sequence_id']; serial=d['decision_id']; probabilities=d['probability']; counts=d['counts']
            n=len(ids); assert n>0 and len(serial)==n and len(np.unique(serial))==n
            assert np.all((ids>=0)&(ids<32768))
            assert np.array_equal(counts,np.bincount(ids,minlength=32768)) and counts.sum()==n
            assert np.allclose(probabilities,q[ids],atol=1e-15,rtol=0)
            assert np.all(d['teacher_version']==0)
            assert np.all((d['assigned_control_step']>=u*24)&(d['assigned_control_step']<=(u+100)*24))
            durations=d['exposure_steps_until_boundary']; assert np.all(durations>=0)
            exposure=d['executed_transitions_per_sequence']; assert np.all(exposure>=0)
            assert exposure.shape==(32768,) and int(exposure.sum())==100*4096*24
            assert int(durations.sum())==int(outcome['cycle_version_exposure'].get('0',0))
            assert sum(int(v) for v in outcome['cycle_version_exposure'].values())==100*4096*24
            assert n==outcome['sampled_decisions']
            empirical=counts/n; realized_mean=empirical@basis; expected_mean=q@basis
            counts_by_arm[arm]=counts.copy(); exposure_by_arm[arm]=exposure.copy()
            sampling=dict(real_decisions=n,unique_selected_sequences=int((counts>0).sum()),
                planned_probability_sha256=array_hash(q),planned_effective_sequences=float(1/np.square(q).sum()),
                empirical_sequence_tv_from_planned=float(abs(empirical-q).sum()/2),
                empirical_basis_mean=realized_mean.tolist(),expected_basis_mean=expected_mean.tolist(),
                basis_mean_absolute_deviation=float(abs(realized_mean-expected_mean).mean()),
                new_draw_exposure_steps=int(durations.sum()),
                conditioned_old_episode_exposure_steps=int(outcome['cycle_version_exposure'].get('-1',0)),
                zero_exposure_end_boundary_draws=int((durations==0).sum()),
                assignment_probability_exact=True,all_transitions_accounted=True,
                real_draws_sha256=file_sha(cycle/'real_draws.npz'))
        result,result_file=evaluation(arm_label+'-after',u+100)
        assert {k:result[k] for k in identity}==identity
        actor_path=Path(run['actor_checkpoint'].replace('/workspace',str(ROOT),1))
        assert result['checkpoint_sha256']==file_sha(actor_path)
        results[arm]=result; receipts[arm]=receipt
        summaries[arm]=dict(run_result_sha256=file_sha(out/'result.json'),restore=restored,
            sampling=sampling,development=result_file,continuous_score=result['continuous_joint_score'],
            strict_success=result['joint_success_rate'],physical_errors=result['pooled_rmse'],safety=result['safety'])
    assert not np.array_equal(counts_by_arm['plus'],counts_by_arm['minus']), 'Interventions produced identical sampled counts'
    assert not np.array_equal(exposure_by_arm['plus'],exposure_by_arm['minus']), 'Interventions produced identical physical exposure'
    assert pair['pair_tv']>.02
    comparisons={name:paired(results[a]['scores'],results[b]['scores'],700000+index*10+j)
        for j,(name,a,b) in enumerate([('plus_minus','plus','minus'),('plus_uniform','plus','uniform'),('minus_uniform','minus','uniform')])}
    noise={}
    if index<2:
        before_repeat,before_repeat_file=evaluation(label+'-before-repeat',u)
        assert {k:before_repeat[k] for k in identity}==identity
        assert before_repeat['checkpoint_sha256']==before['checkpoint_sha256']
        noise['before_fixed_policy_repeat']=paired(before_repeat['scores'],before['scores'],710000+index)
        noise['before_repeat_file']=before_repeat_file
        noise['branch_after_fixed_policy_repeats']={}
        for j,arm in enumerate(('uniform','plus','minus')):
            repeat,path=evaluation(label+'-'+arm+'-after-repeat',u+100)
            assert {k:repeat[k] for k in identity}==identity
            assert repeat['checkpoint_sha256']==results[arm]['checkpoint_sha256']
            noise['branch_after_fixed_policy_repeats'][arm]=paired(repeat['scores'],results[arm]['scores'],720000+index*10+j)|dict(result_file=path)
        noise['same_uniform_restore_and_100update_repeat']=paired(results['uniform-repeat']['scores'],results['uniform']['scores'],730000+index)
        q1=counts_by_arm['uniform']/counts_by_arm['uniform'].sum()
        q2=counts_by_arm['uniform-repeat']/counts_by_arm['uniform-repeat'].sum()
        noise['same_uniform_empirical_sequence_tv']=float(abs(q1-q2).sum()/2)
        noise['interpretation']='Only one independent repeat of the same Uniform branch per pilot state; a diagnostic noise observation, not a precise variance estimate.'
    warnings=[]
    if index<2:
        signal=abs(comparisons['plus_minus']['mean_difference'])
        repeat=abs(noise['same_uniform_restore_and_100update_repeat']['mean_difference'])
        if signal<=repeat: warnings.append('Paired treatment contrast no larger than the observed same-policy training-repeat difference.')
        frozen_repeat=max(abs(v['mean_difference']) for v in noise['branch_after_fixed_policy_repeats'].values())
        frozen_repeat=max(frozen_repeat,abs(noise['before_fixed_policy_repeat']['mean_difference']))
        if signal<=frozen_repeat:
            warnings.append('Treatment contrast is no larger than observed fixed-policy reevaluation variation; one pair cannot establish a reliable directional learning effect.')
    result=dict(passed=True,index=index,seed=seed,start_update=u,source_checkpoint_sha256=source['sha256'],
        prepared_checkpoint_sha256=pair['source_checkpoint_sha256'],decision_sha256=file_sha(point/'decision.pt'),
        restorable_branch_inputs_equal=True,original_assets_unchanged=True,all_sampler_probabilities_verified=True,
        physically_executed_exposure_differs=True,planned_plus_minus_tv=pair['pair_tv'],
        arms=summaries,paired_development_comparisons=comparisons,noise=noise,warnings=warnings,
        acceptance='Only engineering correctness. No success, reward, improvement or statistical significance criterion.',
        interpretation='Fixed source states were selected by seed and update, not outcomes. PhysX hidden caches remain non-bitwise. Empirical full-ID TV is sparse-sample biased; projected exposure and Uniform repeats are reported. Bootstrap intervals do not establish cross-student significance.')
    assert not (point/'engineering.json').exists()
    write(point/'engineering.json',result)
    print('V02_POINT_ENGINEERING_PASS '+json.dumps(dict(index=index,seed=seed,start_update=u,planned_tv=pair['pair_tv'],warnings=warnings)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--index',type=int,required=True)
    audit(parser.parse_args().index)
