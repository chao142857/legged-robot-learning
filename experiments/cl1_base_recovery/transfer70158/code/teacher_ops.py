"""Append-only CPU teacher decisions and updates from isolated development results."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT, file_sha, write
from neural_v01 import DYNAMIC_FIELDS
from teacher_v02 import Teacher, V02_CONFIG, array_hash


def read(path): return json.loads(Path(path).read_text())


def load_teacher(path):
    manifest=read(ROOT/'prepared/manifest.json')
    teacher=Teacher(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False),manifest['binding'])
    teacher.load(torch.load(path,map_location='cpu',weights_only=False)['teacher'])
    return teacher


def dev_result(path, checkpoint=None):
    result=read(path)
    assert result['passed'] and result['kind']=='development'
    assert result['training_history_updates']==0 and result['model_weights_unchanged']
    manifest=read(ROOT/'prepared/manifest.json')
    assert result['pool_sha256']==manifest['binding']['development_pool_sha256']
    assert result['episodes']==512 and len(result['scores'])==512
    if checkpoint is not None: assert result['checkpoint_sha256']==file_sha(checkpoint)
    return result


def decision(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    cp=torch.load(args.source,map_location='cpu',weights_only=False)
    info=cp['infos']; context=info['v02_context']
    source=read(args.source.parent/'source.json')
    dev=dev_result(args.before,args.source)
    progress_path=Path(source['path']).parent.parent/'updates.jsonl'
    exact=[]
    with progress_path.open() as f:
        for line in f:
            row=json.loads(line)
            if row['completed_updates']==info['completed_updates']: exact.append(row)
    assert len(exact)==1
    row=exact[0]
    assert row['native_feedback_episodes']==info['adapter']['owner']['accepted']
    assert row['learning_rate']==info['adaptive_learning_rate']
    dynamic=context['dynamic']
    recent=dynamic[:,DYNAMIC_FIELDS.index('recent_feedback_observed')]>0.5
    action_std=float(cp['actor_state_dict']['distribution.log_std_param'].exp().mean())
    global_state=np.array([info['completed_updates']/10000,info['adaptive_learning_rate']/1e-3,
        action_std,row['losses']['value'],row['mean_training_reward']/200,recent.mean(),
        dynamic[recent,DYNAMIC_FIELDS.index('recent_good_fraction')].mean() if recent.any() else 0,
        dynamic[recent,DYNAMIC_FIELDS.index('recent_failure')].mean() if recent.any() else 0,
        dev['continuous_joint_score'],1.],dtype=np.float64)
    teacher=load_teacher(args.teacher)
    pair=teacher.sample_pair(dynamic,global_state,args.version,context['progress'],context['progress_observed'],context['partial'])
    pair.update(source_checkpoint=str(args.source),source_checkpoint_sha256=file_sha(args.source),
        source_original=source,source_progress_row=row,source_progress_log=str(progress_path),
        source_preparation=info['v02_preparation'],before_development=dev,
        before_development_sha256=file_sha(args.before),teacher_input_checkpoint=str(args.teacher),
        teacher_input_checkpoint_sha256=file_sha(args.teacher))
    torch.save(pair,out/'decision.pt')
    probability=np.ones(32768,dtype=np.float64)/32768
    np.save(out/'uniform.npy',probability,allow_pickle=False)
    for name in ('plus','minus'):
        np.save(out/f'{name}.npy',pair[name]['probability'],allow_pickle=False)
        # Probability geometry alone: no development or formal-test performance tuning.
        p=pair[name]['probability']
        assert 1/np.square(p).sum()/len(p)>.15,'Exploration concentrated beyond predeclared engineering bound'
        assert np.sort(p)[-int(np.ceil(len(p)*.01)):].sum()<.20
    assert pair['pair_tv']>.02,'Exploration distributions are numerically indistinguishable'
    write(out/'decision.json',dict(version=args.version,student_seed=source['seed'],start_update=source['updates'],
        source_checkpoint=source['path'],source_checkpoint_sha256=source['sha256'],
        prepared_checkpoint_sha256=file_sha(args.source),decision_sha256=file_sha(out/'decision.pt'),
        teacher_sha256=pair['teacher_sha256'],global_state=global_state,config=V02_CONFIG,
        plus=pair['plus']['distribution'],minus=pair['minus']['distribution'],
        preference_mean=pair['mean'],epsilon=pair['epsilon'],plus_preference=pair['plus']['preference'],
        minus_preference=pair['minus']['preference'],plus_log_probability=pair['plus']['log_probability'],
        minus_log_probability=pair['minus']['log_probability'],pair_tv=pair['pair_tv'],
        before_development_score=dev['continuous_joint_score'],
        feedback_coverage=float(recent.mean()),learning_progress_coverage=float(context['progress_observed'].mean()),
        future_feedback_excluded=True,formal_test_inputs=False))
    print('V02_DECISION '+json.dumps(read(out/'decision.json')),flush=True)


def update(args):
    pair=torch.load(args.out/'decision.pt',map_location='cpu',weights_only=False)
    teacher=load_teacher(args.teacher)
    assert file_sha(args.teacher)==pair['teacher_input_checkpoint_sha256']
    assert teacher.updates==pair['teacher_updates']
    # The decision checkpoint contains the RNG state immediately after exploration.
    teacher.load(pair['teacher'])
    before=pair['before_development']
    results={name:dev_result(getattr(args,name)) for name in ('uniform','plus','minus')}
    for result in results.values():
        assert result['initial_sha256']==before['initial_sha256']
        assert result['update']==before['update']+100
    response=teacher.learn_pair(pair,before['continuous_joint_score'],
        *[results[n]['continuous_joint_score'] for n in ('uniform','plus','minus')])
    assert not (args.out/'teacher_after.pt').exists()
    torch.save(dict(teacher=teacher.state(),update=response,decision_sha256=file_sha(args.out/'decision.pt'),
        results=results),args.out/'teacher_after.pt')
    np.save(args.out/'gradient.npy',response['gradient_vector'],allow_pickle=False)
    summary={k:v for k,v in response.items() if k!='gradient_vector'}
    summary.update(teacher_checkpoint_sha256=file_sha(args.out/'teacher_after.pt'),
        development_results={n:dict(path=str(getattr(args,n)),sha256=file_sha(getattr(args,n)),
            score=r['continuous_joint_score'],strict=r['joint_success_rate'],rmse=r['pooled_rmse'],
            safety=r['safety']) for n,r in results.items()},
        student_seed=pair['source_original']['seed'],start_update=before['update'],
        gradient_sha256=file_sha(args.out/'gradient.npy'),formal_test_used=False,
        causal_interpretation='Paired restorable state and RNG; hidden PhysX nondeterminism remains. One pair is noisy.')
    write(args.out/'update.json',summary)
    print('V02_TEACHER_UPDATE '+json.dumps({k:v for k,v in summary.items() if k not in ('development_results','mean_gradient_ascent')}),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['decision','update'])
    p.add_argument('--out',type=Path,required=True); p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--source',type=Path); p.add_argument('--before',type=Path); p.add_argument('--version',type=int)
    for name in ('uniform','plus','minus'): p.add_argument('--'+name,type=Path)
    args=p.parse_args(); globals()[args.action](args)


if __name__=='__main__':main()
