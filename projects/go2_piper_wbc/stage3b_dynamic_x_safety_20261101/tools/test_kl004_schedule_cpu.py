"""CPU coordinator contract fixtures; never Docker, training or real study I/O."""
import hashlib, json, pathlib, sys, tempfile, types
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]/'stage3_dynamic_x_20261031/tools'))
import run_leg_kl004 as controller
from stage3_pipeline import decision


def exercise(unsafe300=False):
    with tempfile.TemporaryDirectory(prefix='CPU-only004-schedule-') as td:
        p=pathlib.Path(td)
        def write(f,d):f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(d))
        def read(f):return json.loads(f.read_text())
        parent=dict(checkpoint='/CPU_FIXTURE/parent.pt',checkpoint_sha256='fixture-parent')
        for name,value in {'LEG_KL004_PLAN_FROZEN.json':dict(weight=.04,training_seeds=[22,21],recipe_id='arm_synergy_fixed_legKL_004',
            only_changed_learning_factor='fixed_leg_Gaussian_KL_coefficient_0.02_to_0.04',common_parent_sha256='fixture-parent'),
            'REPAIR2_DONE.json':dict(hypothesis_supported=True),'LEARNING_REPAIR_NEEDED.json':dict(preserved=True),
            'CHECKPOINTS.json':dict(parent=parent),'P1_BASE_CONFIG.json':{},'NATIVE_TRAIN_CONFIG.json':{},
            'SYNERGY_RECIPE.json':{},'CANDIDATE_NODES.json':[]}.items():write(p/name,value)
        b=types.SimpleNamespace(p=p,cap=28800,reserve=3600,meter=lambda:dict(GPU_container_SUM_seconds=0.,active=[]))
        calls=[];evals=[]
        def train(b,cfg,label):
            calls.append((cfg['training_seed'],label,cfg['updates'],cfg.get('resume')))
            target=int(label.split('-to')[-1]);run=p/'runs'/label
            cp=run/'ppo/checkpoints'/('cp_'+str(1800+target)+'.pt');cp.parent.mkdir(parents=True)
            cp.write_bytes(label.encode())
            if target==400:
                middle=cp.with_name('cp_2100.pt');middle.write_bytes(b'CPU-only-native-saved300')
                write(middle.with_suffix('.json'),dict(completed_updates=2100,sha256=hashlib.sha256(middle.read_bytes()).hexdigest()))
            for f in ('restore-evidence.json','zero-mean-attachment.json'):write(run/'ppo'/f,dict(CPU_fixture_only=True))
            write(run/'ppo/result.json',dict(completed_updates=1800+target,time_stopped=False))
            return run,dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
        def tests(b,base,cp,label,phase):
            evals.append(label);n=int(label.split('-n')[1].split('-')[0]);gain={200:.30,300:.40,400:.49}[n]
            contact=int(unsafe300 and n==300)
            return [dict(label=label+'-'+task+'-'+str(seed),protocol=task,seed=seed,n=64,complete=64-contact,
                strict_safe=64-contact,nonfoot=contact,joint_limit=0,X_after1s=dict(synchronous_gain=gain,
                actual_fitted_amplitude_m=gain*.021,correlation=.8),absolute_X_gate=dict(gain=gain>=.35),
                relative_X_gate=dict(original=True),retention_checks=dict(original=True))
                for seed in (6181031,6281031) for task in ('A','T0','T1','X')]
        controller.read=read;controller.write=write;controller.train=train;controller.tests=tests;controller.decision=decision
        controller.main(b)
        assert [(seed,window) for seed,_,window,_ in calls]==[(s,w) for s in (22,21) for w in (2,98,100,200)]
        assert all(resume is (window!=2) for _,_,window,resume in calls)
        assert read(p/'REPAIR2_STOP_BEFORE_LEGKL004.json')==dict(preserved=True)
        nodes=read(p/'CANDIDATE_NODES.json')
        if unsafe300:
            assert not(p/'RESEARCH_CLOSED.json').exists() and not(p/'FINAL_RECIPE_ADMISSION.json').exists()
            assert (p/'LEARNING_REPAIR_NEEDED.json').exists()
            assert [n['additional_actual_updates'] for n in nodes]==[200,300,200,300]
            assert all('-n400-' not in label for label in evals)
        else:
            assert read(p/'FINAL_RECIPE_ADMISSION.json')['recipe_ids']==['arm_synergy_fixed_legKL_004']
            assert (p/'RESEARCH_CLOSED.json').exists()
            assert [n['additional_actual_updates'] for n in nodes]==[200,300,400,200,300,400]
            assert all(n['actual_checkpoint_iteration']==1800+n['additional_actual_updates'] for n in nodes)
            # Saved300 is read from to400, not trained from a synthetic300 reset.
            assert all('window-to400' in n['run'] for n in nodes if n['additional_actual_updates']==300)


exercise();exercise(unsafe300=True)
print('CPU004 orchestration PASS: exact native reset schedule, both independent seeds, real saved-node iteration, safety stop and archived marker; no GPU/physics claim')
