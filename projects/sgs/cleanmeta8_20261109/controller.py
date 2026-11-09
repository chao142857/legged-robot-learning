"""Clean V0.2 reproduction, then hardware-matched frozen Teacher transfers."""
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
    run('preflight',[str(ROOT/'code/preflight.py')],gpu=False)
    teacher=ROOT/'teachers/Fresh.pt'
    sources=read(ROOT/'sources.json')['sources']
    for index,s in enumerate(sources):
        u,seed=s['updates'],s['seed'];label=f'p{index:02d}-seed{seed}-cp{u:05d}';start=Path(s['path'])
        assert sha(start)==s['sha256']
        before=evaluate(label+'-before',start)
        point=EXPERIENCE/'meta-eight-starts'/f'point_{index:02d}'
        run('decision-'+label,[ROOT/'code/teacher_ops.py','decision','--out',point,'--teacher',teacher,'--source',start,'--before',before,'--version',str(index)],gpu=False)
        ev={}
        for arm in ('uniform','plus','minus'):
            r=train(label+'-'+arm,u+100,seed,source=start,probability=point/f'{arm}.npy',branch=arm,decision_source=point/'decision.pt')
            assert r['new_updates']==100
            ev[arm]=evaluate(label+'-'+arm+'-after',Path(r['actor_checkpoint']))
        run('audit-'+label,[ROOT/'code/audit_point.py','--index',str(index)],gpu=False)
        run('update-'+label,[ROOT/'code/traced_teacher_ops.py','update','--out',point,'--teacher',teacher,*[v for arm in ('uniform','plus','minus') for v in ('--'+arm,ev[arm])]],gpu=False)
        a=read(point/'adam_acceptance.json');assert a['passed'] and a['logical_after']==index+1
        teacher=point/'teacher_after.pt'
        status('clean_meta_point_complete',completed_points=index+1,new_student_updates=(index+1)*300,adam_steps=a['adam_steps_after'])
    assert not (ROOT/'teachers/CleanMeta8.pt').exists()
    shutil.copy2(teacher,ROOT/'teachers/CleanMeta8.pt')
    run('clean-meta-acceptance',[ROOT/'code/final_checks.py','teacher'],gpu=False)
    status('CleanMeta8_complete',teacher=str(ROOT/'teachers/CleanMeta8.pt'),new_student_updates=2400)
    raise RuntimeError('Obsolete transfer schedule disabled: use revision02 boundary handoff and CleanMeta8-only queue.')

if __name__=='__main__':
    try:main()
    except BaseException:
        (ROOT/'FAULT.txt').write_text(traceback.format_exc());status('fault',detail=traceback.format_exc());raise
