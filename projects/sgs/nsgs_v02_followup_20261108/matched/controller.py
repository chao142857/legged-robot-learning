"""Matched-source continuation of the old Meta, then one fresh seed48 Student."""
import json,shutil,time,traceback
from pathlib import Path
from host import ROOT,EXPERIENCE,KIT,run,write

def read(p):return json.loads(Path(p).read_text())
def status(phase,**kw):write(ROOT/'STATUS.json',dict(phase=phase,unix=time.time(),**kw))
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def evaluate(label,cp,kind='development'):
    out=ROOT/'evaluations'/label
    run('eval-'+label,[str(ROOT/'code/evaluate.py'),'--checkpoint',str(cp),'--kind',kind,'--out',str(out),'--headless','--kit_args',KIT])
    a=read(out/'result.json');assert a['passed'] and a['checkpoint_sha256']==sha(cp) and a['training_history_updates']==0 and a['model_weights_unchanged']
    identity={k:a[k] for k in ('initial_sha256','pool_sha256','seed','episodes','protocol_sha256')}
    reference=ROOT/f'evaluation_identity_{kind}.json'
    if reference.exists():assert read(reference)==identity
    else:write(reference,identity)
    return out/'result.json'

def train(label,target,seed,mode='branch',**options):
    out=ROOT/'runs'/label
    args=[ROOT/'code/train_v02.py','--out',out,'--dataset',EXPERIENCE/label,'--mode',mode,'--target',str(target),'--student-seed',str(seed),'--headless','--kit_args',KIT]
    for k,v in options.items():
        args.append('--'+k.replace('_','-'))
        if v is not True:args.append(v)
    def tick(process):
        for req in sorted((out/'evaluation_requests').glob('*.json')):
            if req.name.endswith('.done.json') or req.with_suffix('.done.json').exists():continue
            a=read(req);cp=Path(a['checkpoint']);assert cp.parent==out/'actors' and sha(cp)==a['checkpoint_sha256']
            result=evaluate(label+'-'+req.stem,cp,a['kind']);reply=read(result);assert reply['update']==a['update']
            write(req.with_suffix('.done.json'),reply)
    status('training_'+mode,label=label,target=target,student_seed=seed)
    run(label,[str(v) for v in args],on_tick=tick if mode=='transfer' else None)
    assert not (out/'fault.txt').exists(), (out/'fault.txt').read_text() if (out/'fault.txt').exists() else ''
    a=read(out/'result.json');assert a['passed'] and a['completed_updates']==target
    return a

def main():
    sources=read(ROOT/'sources.json')['sources'];assert [(s['seed'],s['updates']) for s in sources]==[(seed,u) for u in (1000,3000,6000,9000) for seed in (43,44)]
    status('cpu_tests');run('cpu-tests',['-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],gpu=False)
    run('inherit-teacher',[str(ROOT/'code/prepare_matched.py')],gpu=False)
    teacher=ROOT/'teachers/old-meta.pt'
    for index,s in enumerate(sources):
        u=s['updates'];seed=s['seed'];label=f'p{index:02d}-seed{seed}-cp{u:05d}';start=Path(s['path'])
        status('prepare_counterfactual',index=index,source_update=u)
        run('prepare-'+label,[str(ROOT/'code/prepare_matched.py'),'--source-index',str(index)],gpu=False)
        before=evaluate(label+'-before',start)
        point=EXPERIENCE/'meta-eight-starts'/f'point_{index:02d}'
        run('decision-'+label,[str(ROOT/'code/teacher_ops.py'),'decision','--out',str(point),'--teacher',str(teacher),'--source',str(start),'--before',str(before),'--version',str(8+index)],gpu=False)
        if index==0:
            smoke=train('matched-smoke',u+2,seed,source=start,probability=point/'uniform.npy',branch='uniform',smoke=True,decision_source=point/'decision.pt')
            assert smoke['new_updates']==2
            smoke2=train('matched-smoke-resume',u+4,seed,resume=Path(smoke['checkpoint']),probability=point/'uniform.npy',branch='uniform',smoke=True,decision_source=point/'decision.pt')
            assert smoke2['new_updates']==4
            write(ROOT/'GPU_SMOKE_ACCEPTANCE.json',dict(passed=True,new_updates=4,resume_verified=True,short_checkpoints_not_used_for_meta_or_transfer=True))
        ev={}
        for arm in ('uniform','plus','minus'):
            r=train(label+'-'+arm,u+100,seed,source=start,probability=point/f'{arm}.npy',branch=arm,decision_source=point/'decision.pt')
            assert r['new_updates']==100
            ev[arm]=evaluate(label+'-'+arm+'-after',Path(r['actor_checkpoint']))
        run('audit-'+label,[str(ROOT/'code/audit_point.py'),'--index',str(index)],gpu=False)
        run('update-'+label,[str(ROOT/'code/teacher_ops.py'),'update','--out',str(point),'--teacher',str(teacher),*[str(v) for arm in ('uniform','plus','minus') for v in ('--'+arm,ev[arm])]],gpu=False)
        teacher=point/'teacher_after.pt'
        status('teacher_experience_added',completed_new_points=index+1,new_student_updates=(index+1)*300,teacher_total_updates=9+index)
    shutil.copy2(teacher,ROOT/'teachers/new-meta.pt')
    run('verify-extended-teacher',[str(ROOT/'code/verify_extended.py')],gpu=False)
    for arm in ('new-meta',):
        cp=ROOT/'teachers'/('fresh.pt' if arm in ('uniform','fresh') else arm+'.pt')
        kind=arm if arm in ('uniform','fresh') else 'meta'
        train('transfer-seed48-'+arm,10000,48,mode='transfer',transfer_kind=kind,teacher_in=cp)
    run('final-audit',[str(ROOT/'code/verify_extended.py'),'--final'],gpu=False)
    run('compare-existing-results',[str(ROOT/'code/compare_existing.py')],gpu=False)
    status('complete',student_seed=48,new_counterfactual_updates=2400,engineering_updates=4,transfer_student_updates=10000,arms=['matched-meta'],source_seeds=[43,44],source_updates=[1000,3000,6000,9000])

if __name__=='__main__':
    try:main()
    except BaseException:
        (ROOT/'FAULT.txt').write_text(traceback.format_exc());status('fault',detail=traceback.format_exc());raise
