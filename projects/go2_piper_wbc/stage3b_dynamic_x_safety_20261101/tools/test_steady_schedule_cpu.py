"""Resume/one-window contract fixtures; no GPU, native data or study writes."""
import hashlib, json, pathlib, sys, tempfile, types
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]/'stage3_dynamic_x_20261031/tools'))
import run_steady_kl_v2 as controller
from stage3_pipeline import decision


def exercise(unsafe_extra=False, enough=True, near=True):
    with tempfile.TemporaryDirectory(prefix='CPU-only-steady-schedule-') as td:
        p=pathlib.Path(td)
        def write(f,d):f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(d))
        def read(f):return json.loads(f.read_text())
        parent=dict(checkpoint='/CPU_FIXTURE/parent.pt',checkpoint_sha256='fixture-parent')
        for name,value in {
            'ROBUST_NODE_EXTENSION_PREREGISTRATION.json':dict(extra_updates=100,training_seed=22,adjacent_amplitude_trigger_m=.0085),
            'STEADY_KL001_PLAN_FROZEN.json':dict(leg_weight=.02,steady_mean_weight=.00419,training_seeds=[22,21],recipe_id='steady',
                only_changed_learning_factor='add_observable_steady_goal_full_mean_preservation',common_parent_sha256='fixture-parent'),
            'REPAIR3V2_DONE.json':dict(checks=[dict(mode='parent_all',removes_contact=True)]),
            'LEARNING_REPAIR_NEEDED.json':dict(preserved=True),'CHECKPOINTS.json':dict(parent=parent),
            'P1_BASE_CONFIG.json':{},'NATIVE_TRAIN_CONFIG.json':{},'SYNERGY_RECIPE.json':{},'CANDIDATE_NODES.json':[]}.items():write(p/name,value)
        used=[0.];b=types.SimpleNamespace(p=p,cap=28800,reserve=3600,meter=lambda:dict(GPU_container_SUM_seconds=used[0],active=[]))
        new_calls=[];evals=[]
        def train(b,cfg,label):
            receipt=p/(label+'_RUN.json');run=p/'runs'/label
            if not receipt.exists():
                new_calls.append((cfg['training_seed'],label,cfg['updates'],bool(cfg.get('resume'))))
                target=int(label.split('-to')[-1]);cp=run/'ppo/checkpoints'/('cp_'+str(1800+target)+'.pt');cp.parent.mkdir(parents=True)
                cp.write_bytes(label.encode())
                if target==400:
                    middle=cp.with_name('cp_2100.pt');middle.write_bytes(label.encode()+b'-saved300')
                    write(middle.with_suffix('.json'),dict(completed_updates=2100,sha256=hashlib.sha256(middle.read_bytes()).hexdigest()))
                for f in ('restore-evidence.json','zero-mean-attachment.json'):write(run/'ppo'/f,dict(CPU_fixture_only=True))
                write(run/'ppo/result.json',dict(completed_updates=1800+target,time_stopped=False))
                write(receipt,dict(run=str(run),checkpoint=str(cp)))
            cp=pathlib.Path(read(receipt)['checkpoint'])
            return run,dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
        def tests(b,base,cp,label,phase):
            evals.append(label);n=int(label.split('-n')[1].split('-')[0]);gain={200:.30,300:.40,400:.45,500:.52}[n]
            if label=='steadyKL001-n400-seed21' and not enough:used[0]=24000.
            contact=int(unsafe_extra and n==500)
            amp=(.00801 if near else .009) if n==300 else gain*.021
            return [dict(label=label+'-'+task+'-'+str(seed),protocol=task,seed=seed,n=64,complete=64-contact,
                strict_safe=64-contact,nonfoot=contact,joint_limit=0,X_after1s=dict(synchronous_gain=gain,
                actual_fitted_amplitude_m=amp,correlation=.8),absolute_X_gate=dict(gain=gain>=.35),
                relative_X_gate=dict(original=True),retention_checks=dict(original=True))
                for seed in (6181031,6281031) for task in ('A','T0','T1','X')]
        controller.read=read;controller.write=write;controller.train=train;controller.tests=tests;controller.decision=decision
        controller.main(b)
        expected=[(s,w) for s in (22,21) for w in (2,98,100,200)]
        did_extra=enough and near
        assert [(s,w) for s,_,w,_ in new_calls]==expected+([(22,100)] if did_extra else [])
        assert all(resume==(w!=2) for _,_,w,resume in new_calls)
        assert read(p/'REPAIR3_STOP_BEFORE_STEADYKL001.json')==dict(preserved=True)
        assert read(p/'FINAL_RECIPE_ADMISSION.json')['recipe_ids']==['steady']
        assert (p/'RESEARCH_CLOSED.json').exists()
        if unsafe_extra and did_extra:assert (p/'steadyKL001-seed22-PHYSICAL_STOP.json').exists()
        before=[(f.name,f.read_bytes()) for f in p.glob('*FIRST_COVERAGE.json')]
        # Simulate a CPU coordinator restart before closure; receipts and nodes
        # are reused, including an unsafe extra, and no native window repeats.
        (p/'RESEARCH_CLOSED.json').unlink();calls=len(new_calls);old_evals=len(evals)
        controller.main(b)
        assert len(new_calls)==calls and len(evals)==old_evals
        assert all((p/name).read_bytes()==content for name,content in before)
        try:controller.main(b)
        except AssertionError as error:assert 'Closed research' in str(error)
        else:raise AssertionError('Closed research was reopened')


exercise();exercise(unsafe_extra=True);exercise(enough=False);exercise(near=False)
print('CPU steady coordinator PASS: exact reset cuts, receipt-only restart, one optional100 window, unsafe extra excluded, reserve and closed grant protected; no physics claim')
