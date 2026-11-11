import dataclasses,hashlib,json,pathlib,random,time
import numpy as np

ROOT=pathlib.Path('/workspace')
NATIVE=pathlib.Path('/native')
PARAMETERS=dict(history=100,target=.5,concentration=1.,temperature=2.,epsilon=1e-4)
PROTOCOL=dict(version='native-sequence-joint-v1',settle_seconds=2.0,min_scored_steps=100,
              joint_fraction=.8,thresholds=dict(velocity_xy_m_s=.20,yaw_rate_rad_s=.30,ee_position_m=.05,ee_orientation_rad=float(np.pi/12)),
              rule='No native failure termination; at least 100 joint-eligible control steps; >=80% simultaneous four-way tolerance compliance AND each eligible RMSE within its tolerance. Timeout alone is insufficient.',
              scoring='Both velocity and EE command ages >=2 seconds; ages restart at every native command resample; terminal transition is included before autoreset.',
              scope='Flat link0 pose; WBC native mixed xy-in-link0/z-in-world pose; native body-frame base velocity. Training reward unchanged.')

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if dataclasses.is_dataclass(x):return plain(x.to_dict() if hasattr(x,'to_dict') else dataclasses.asdict(x))
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if x is dataclasses.MISSING:return '__MISSING__'
    if callable(x):return x.__module__+'.'+getattr(x,'__qualname__',x.__class__.__qualname__)
    if hasattr(x,'detach'):return x.detach().cpu().tolist()
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    return str(x)

def canonical(x):return json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n');tmp.replace(p)
def emit(p,x):
    with pathlib.Path(p).open('a') as f:f.write(json.dumps(plain(x),allow_nan=False)+'\n')
def rng_state():
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    import torch
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu']);torch.cuda.set_rng_state_all(s['torch_cuda'])
def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Pool:
    def __init__(self,kind,split):
        self.kind,self.split=kind,split
        self.path=ROOT/'pools'/(kind+'_'+split+'.npz');d=np.load(self.path)
        self.ee=d['ee'];self.velocity=d['velocity'];self.n=len(self.ee);self.sha=file_sha(self.path)
        self.identity=kind+':'+split+':'+self.sha
    def to(self,device):
        import torch
        self.ee_tensor=torch.from_numpy(self.ee).to(device);self.vel_tensor=torch.from_numpy(self.velocity).to(device)
        return self

def model_cfg(task):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    import importlib.metadata as md
    env=load_cfg_from_registry(task,'env_cfg_entry_point');agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    agent=handle_deprecated_rsl_rl_cfg(agent,md.version('rsl-rl-lib'))
    assert agent.seed==42 and agent.num_steps_per_env==24 and agent.max_iterations==10000 and not agent.resume
    return env,agent
