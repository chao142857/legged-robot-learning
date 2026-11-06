"""Persistent V0.2: engineering pilot, eight counterfactual starts, frozen transfer."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback
from host import ROOT, EXPERIENCE, KIT, run, write, inside, outside

META=EXPERIENCE/'meta-eight-starts'
STUDENT_SEED=47
RESUME_PREFLIGHT=False
TRANSFER_ONLY=False


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),**kw))


def evaluate(label,checkpoint,kind='development'):
    out=ROOT/'evaluations'/label
    if not (RESUME_PREFLIGHT and (out/'result.json').exists()):
        run('eval-'+label,[ROOT/'code/evaluate.py','--checkpoint',checkpoint,'--kind',kind,
            '--out',out,'--headless','--kit_args',KIT])
    result=read(out/'result.json')
    assert result['passed'] and result['checkpoint_sha256']==sha(checkpoint)
    assert result['kind']==kind and result['training_history_updates']==0 and result['model_weights_unchanged']
    identity={k:result[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}
    reference=ROOT/f'evaluation_identity_{kind}.json'
    if reference.exists():assert read(reference)==identity
    else:write(reference,identity)
    return out/'result.json'


def training(label,target,seed,mode='branch',**options):
    out=ROOT/'runs'/label
    arguments=[ROOT/'code/train_v02.py','--mode',mode,'--out',out,'--dataset',EXPERIENCE/label,
        '--target',str(target),'--student-seed',str(seed),'--headless','--kit_args',KIT]
    for key,value in options.items():
        arguments.append('--'+key.replace('_','-'))
        if value is not True:arguments.append(str(value))
    def tick(process):
        for req in sorted((out/'evaluation_requests').glob('*.json')):
            if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():continue
            request=read(req)
            cp=outside(request['checkpoint'])
            assert cp.parent==out/'actors' and sha(cp)==request['checkpoint_sha256']
            result_path=evaluate(label+'-'+req.stem,cp,request['kind'])
            result=read(result_path)
            assert result['update']==request['update']
            write(req.with_suffix('.done.json'),result)
    status('running_'+label,run=str(out),target=target,student_seed=seed,mode=mode)
    run(label,arguments,on_tick=tick if mode=='transfer' else None)
    if (out/'fault.txt').exists():
        raise RuntimeError('Simulator wrote a fault even if its launcher returned zero: '+(out/'fault.txt').read_text())
    result=read(out/'result.json')
    assert result['passed'] and result['completed_updates']==target
    return out,result


def main():
    manifest=read(ROOT/'source_manifest.json')
    for name,expected in manifest['files'].items():assert sha(ROOT/name)==expected,name
    sources=read(ROOT/'sources.json')['sources']
    assert [(s['seed'],s['updates']) for s in sources]==[(s,u) for u in (1000,3000,6000,9000) for s in (43,44)]
    if TRANSFER_ONLY:
        assert not RESUME_PREFLIGHT
        assert read(ROOT/'PILOT_ACCEPTANCE.json')['passed']
        assert read(ROOT/'results/meta_summary.json')['complete']
        for index in range(8):
            point=META/f'point_{index:02d}'
            assert read(point/'engineering.json')['passed']
            update=read(point/'update.json')
            assert sha(point/'teacher_after.pt')==update['teacher_checkpoint_sha256']
        assert sha(ROOT/'teachers/meta-final.pt')==sha(META/'point_07/teacher_after.pt')
        assert sha(ROOT/'teachers/fresh.pt')==read(ROOT/'teachers/initial.json')['sha256']
        assert not list((ROOT/'runs').glob('transfer-*/updates.jsonl')), 'Transfer restart requires all three groups still untrained'
        status('transfer_startup_regression_tests')
        run('cpu-tests-'+manifest['revision'],['-m','unittest','discover','-s','/workspace/tests','-v'],gpu=False)
        return transfer_queue(sources)
    META.mkdir(parents=True,exist_ok=RESUME_PREFLIGHT)
    if RESUME_PREFLIGHT:
        assert not list(META.glob('point_*/update.json')), 'Only pre-training engineering retry is supported'
        assert not list((ROOT/'runs').glob('p*-*/result.json')), 'Existing counterfactual branches cannot be overwritten'
    protocol=dict(version='nsgs-v02-counterfactual-20261106',host='yanglab',
        source_seeds=[43,44],source_updates=[1000,3000,6000,9000],source_selection='Fixed stages, not selected by performance; B/prefix have full restorable state.',
        source_order=[dict(seed=s['seed'],updates=s['updates'],sha256=s['sha256']) for s in sources],
        branch_order=['uniform','plus','minus'],branch_updates=100,meta_updates=2400,
        first_two_uniform_noise_repeat_updates=200,gpu_restore_smoke_updates=4,
        total_pre_transfer_new_student_updates=2604,noise_repeated_development_all_first_two_arms=True,
        uniform_mix=.2,teacher_preference_dim=8,teacher_update='Antithetic Gaussian score function from real paired development gains; no simulator/PPO backpropagation.',
        development='Fixed256 sequences x2 resets, deterministic mean action. Joint eligible-time fraction; failure/invalid window scores zero.',
        formal_test_excluded_from_meta=True,strict_definition='Original5cm/80% with all four tolerances and native-failure exclusion.',
        acceptance='Engineering only; no positive performance gate. Same-policy noise is reported, not assumed zero.',
        transfer_seed=STUDENT_SEED,transfer_arms=['uniform','fresh','meta'],transfer_updates_each=10000,
        transfer_teacher_parameters_frozen=True,transfer_teacher_exploration=False,
        transfer_initial_active_episodes='Identical Uniform reset before actor initialization and development0; intervention affects subsequent real episode assignments.',
        transfer_distribution_update_interval=100,transfer_development_interval=100,formal_test_interval=1000,
        transfer_checkpoint_selection='Every1000 and final10k; no selection using final test.',
        user_confirmed_transfer_mode='Frozen Fresh and pretrained Teachers, same from-zero10k student budget.',
        training_envs=4096,rollout=24,transitions_per_update=98304,preserved_old_experiments=True)
    if (ROOT/'PROTOCOL.json').exists():assert read(ROOT/'PROTOCOL.json')==protocol
    else:write(ROOT/'PROTOCOL.json',protocol)
    status('cpu_tests')
    run('cpu-tests-'+manifest.get('revision','initial'),['-m','unittest','discover','-s','/workspace/tests','-v'],gpu=False)
    if not RESUME_PREFLIGHT:run('prepare',[ROOT/'code/prepare_v02.py'],gpu=False)
    else:
        assert read(ROOT/'jobs/prepare/result.json')['exit_code']==0
        assert sha(ROOT/'teachers/fresh.pt')==read(ROOT/'teachers/initial.json')['sha256']
    current_teacher=ROOT/'teachers/fresh.pt'
    for index,s in enumerate(sources):
        label=f'p{index:02d}-seed{s["seed"]}-cp{s["updates"]:05d}'
        status('preparing_'+label,point=index,source=s['path'])
        start=ROOT/'starts'/f'seed{s["seed"]}-cp{s["updates"]:05d}'/'prepared.pt'
        if not (RESUME_PREFLIGHT and index==0):
            run('prepare-'+label,[ROOT/'code/prepare_v02.py','--source-index',str(index)],gpu=False)
        else:
            receipt=read(start.parent/'preparation.json')
            assert sha(start)==receipt['prepared_checkpoint_sha256'] and sha(s['path'])==receipt['source_checkpoint_sha256']
        before=evaluate(label+'-before',start)
        if index<2:evaluate(label+'-before-repeat',start)
        point=META/f'point_{index:02d}'
        if not (RESUME_PREFLIGHT and index==0):
            run('decision-'+label,[ROOT/'code/teacher_ops.py','decision','--out',point,
                '--teacher',current_teacher,'--source',start,'--before',before,'--version',str(index)],gpu=False)
        else:
            decision=read(point/'decision.json')
            assert sha(point/'decision.pt')==decision['decision_sha256']
            assert sha(start)==decision['prepared_checkpoint_sha256']
        if index==0:
            smoke,_=training('smoke1',s['updates']+2,s['seed'],source=start,
                probability=point/'uniform.npy',branch='smoke',smoke=True,decision_source=point/'decision.pt')
            resumed,_=training('smoke2',s['updates']+4,s['seed'],resume=smoke/'checkpoints'/f'update_{s["updates"]+2:05d}.pt',
                probability=point/'uniform.npy',branch='smoke',smoke=True,decision_source=point/'decision.pt')
            assert read(resumed/'restore_receipt.json')['passed']
            write(ROOT/'GPU_SMOKE_ACCEPTANCE.json',dict(passed=True,cpu_log=str(ROOT/'jobs'/('cpu-tests-'+manifest.get('revision','initial'))/'console.log'),
                source_preparation=str(start.parent/'preparation.json'),restore=read(resumed/'restore_receipt.json'),new_student_updates=4))
        evaluations={}
        for arm in ('uniform','plus','minus'):
            arm_label=label+'-'+arm
            out,result=training(arm_label,s['updates']+100,s['seed'],source=start,
                probability=point/f'{arm}.npy',branch=arm,decision_source=point/'decision.pt')
            assert result['new_updates']==100
            actor=outside(result['actor_checkpoint'])
            evaluations[arm]=evaluate(arm_label+'-after',actor)
            if index<2:evaluate(arm_label+'-after-repeat',actor)
        if index<2:
            out,result=training(label+'-uniform-repeat',s['updates']+100,s['seed'],source=start,
                probability=point/'uniform.npy',branch='uniform',decision_source=point/'decision.pt')
            evaluate(label+'-uniform-repeat-after',outside(result['actor_checkpoint']))
        run('audit-'+label,[ROOT/'code/audit_point.py','--index',str(index)],gpu=False)
        run('update-'+label,[ROOT/'code/teacher_ops.py','update','--out',point,'--teacher',current_teacher,
            *[value for arm in ('uniform','plus','minus') for value in ('--'+arm,evaluations[arm])]],gpu=False)
        current_teacher=point/'teacher_after.pt'
        status('counterfactual_point_complete',point=index,teacher=str(current_teacher),total_points=8)
        if index==1:
            write(ROOT/'PILOT_ACCEPTANCE.json',dict(passed=True,points=2,three_arm_updates=600,
                uniform_repeat_updates=200,gpu_smoke_updates=4,engineering_only=True,
                point_audits=[read(META/f'point_{i:02d}'/'engineering.json') for i in (0,1)],
                no_performance_gate=True,automatic_expansion_to_eight=True))
    shutil.copy2(current_teacher,ROOT/'teachers/meta-final.pt')
    run('meta-report',[ROOT/'code/report_v02.py','--stage','meta'],gpu=False)


    transfer_queue(sources)


def transfer_queue(sources):
    for arm in ('uniform','fresh','meta'):
        teacher=ROOT/'teachers'/('meta-final.pt' if arm=='meta' else 'fresh.pt')
        training('transfer-seed47-'+arm,10000,STUDENT_SEED,mode='transfer',
            transfer_kind=arm,teacher_in=teacher)
    run('final-report',[ROOT/'code/report_v02.py','--stage','final'],gpu=False)
    for s in sources:assert sha(s['path'])==s['sha256']
    status('complete',report=str(ROOT/'results/final_summary.json'),student_seed=STUDENT_SEED,
        meta_student_updates=2604,transfer_student_updates=30000,teacher=str(ROOT/'teachers/meta-final.pt'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume-preflight',action='store_true')
    parser.add_argument('--transfer-only',action='store_true')
    args=parser.parse_args(); RESUME_PREFLIGHT=args.resume_preflight; TRANSFER_ONLY=args.transfer_only
    try:main()
    except BaseException as error:
        status('failed',error=repr(error),traceback=traceback.format_exc())
        raise
