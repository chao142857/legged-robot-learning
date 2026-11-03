"""Command-only features. No feedback/checkpoint/policy/evaluation inputs.

Rotations are embedded as rotation matrices; squared chord distance is
8*sin(theta/2)^2, a sign-invariant monotone SO(3) angular distance. Exact
geodesic distances are also used in the offline geometry audit.
"""
import numpy as np

DT=np.float32(.02)
HORIZON=1000
SAMPLE_STEPS=np.arange(12,1000,25,dtype=np.int64) #40 fixed0.5s bins

def euler_matrix(e):
    r,p,y=np.moveaxis(np.asarray(e),-1,0);cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr),-1).reshape(e.shape[:-1]+(3,3))

def quat_matrix(q):
    q=np.asarray(q,dtype=np.float64);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))

def native_reference_rotation(position,offset,reference):
    R=quat_matrix(np.asarray(reference['quaternion']));height=float(reference['height'])
    pos_b=np.zeros_like(position);pos_b[...,:2]=position[...,:2]
    delta_w=pos_b@R.T;delta_w[...,2]=position[...,2]-height
    delta_b=delta_w@R
    aim=np.zeros_like(offset);aim[...,1]=-np.arctan2(delta_b[...,2],np.linalg.norm(delta_b[...,:2],axis=-1));aim[...,2]=np.arctan2(delta_b[...,1],delta_b[...,0])
    return euler_matrix(np.clip(aim+offset,-3.14/4,3.14/3))

def timeline(ee,velocity,reset_compute=True):
    """Exact float32 subtract timers; pre-action slot at each control step.

    Regular auto-reset executes compute(dt) on the reset frame, before its
    first physical step. Explicit initial reset does not. Terminal reset
    precedes the old sequence's command compute; old inputs at step1000 do
    not execute.
    """
    n=len(ee);ee=np.asarray(ee,dtype=np.float32);velocity=np.asarray(velocity,dtype=np.float32)
    slots=np.zeros((n,2),dtype=np.int64)
    left=np.stack((np.float32(8)+np.float32(2)*ee[:,0,6],np.full(n,10,dtype=np.float32)),1)
    if reset_compute:left-=DT
    trace=np.zeros((n,HORIZON,2),dtype=np.uint8);starts=np.full((n,2,4),HORIZON,dtype=np.int32);starts[:,:,0]=0
    for step in range(HORIZON):
        trace[:,step]=slots
        if step==HORIZON-1:break
        left-=DT
        for term in (0,1):
            ids=np.flatnonzero(left[:,term]<=0)
            if len(ids):
                slots[ids,term]+=1
                assert slots[ids,term].max()<4
                s=slots[ids,term];left[ids,term]=np.float32(8)+np.float32(2)*ee[ids,s,6] if term==0 else np.float32(10)
                starts[ids,term,s]=step+1
    return trace,starts

def features(ee,velocity,reference,return_content=False):
    ee=np.asarray(ee,dtype=np.float32);v=np.asarray(velocity,dtype=np.float32);n=len(ee)
    assert ee.shape==(n,4,7) and v.shape==(n,4,5)
    trace,starts=timeline(ee,v,True);initial_trace,_=timeline(ee[:min(16,n)],v[:min(16,n)],False)
    pos=ee[:,:3,:3].astype(np.float64)*np.array([.25,.7,.7])+np.array([.45,-.35,.1])
    offsets=ee[:,:3,3:6].astype(np.float64)*np.array([2*3.14/3,2*3.14/4,2*3.14/6])+np.array([-3.14/3,-3.14/4,-3.14/6])
    # Roll's native clamp is state independent; values below its lower bound are identical commands.
    offsets[...,0]=np.clip(offsets[...,0],-3.14/4,3.14/3)
    rotations=native_reference_rotation(pos,offsets,reference)
    perturb_rotations=euler_matrix(offsets)
    speeds=v[:,:2,:3].astype(np.float64)*np.array([2,1.2,2])+np.array([-1,-.6,-1])
    standing=v[:,:2,3]<=.1;effective_speed=speeds.copy();effective_speed[standing]=0
    index=np.arange(n)[:,None];ee_slot=trace[:,SAMPLE_STEPS,0].astype(int);vel_slot=trace[:,SAMPLE_STEPS,1].astype(int);T=len(SAMPLE_STEPS)
    blocks={
      'position_time':((pos[index,ee_slot]-np.array([.575,0,.45]))/np.array([.25,.7,.7])/np.sqrt(3*T)).reshape(n,-1),
      'velocity_time':(effective_speed[index,vel_slot]/np.array([1,.6,1])/np.sqrt(3*T)).reshape(n,-1),
      'standing_time':standing[index,vel_slot].astype(float)/np.sqrt(T),
      'reference_SO3_time':(rotations[index,ee_slot]/np.sqrt(16*T)).reshape(n,-1),
      'perturbation_SO3_time':(perturb_rotations[index,ee_slot]/np.sqrt(16*T)).reshape(n,-1),
      'EE_switch_times':starts[:,0,1:3]/HORIZON/np.sqrt(2),
      'initial_reset_standing_pulse':(speeds[:,0]*standing[:,0,None]/np.array([1,.6,1])/np.sqrt(3*HORIZON)),
    }
    # Exact dwell descriptors retain tiny final segments missed by the fixed time grid.
    duration=np.stack([(trace[:,:,0]==j).sum(1) for j in range(3)],1)/HORIZON
    event=np.concatenate(((pos-np.array([.575,0,.45]))/np.array([.25,.7,.7])/np.sqrt(3),rotations.reshape(n,3,9)/4,perturb_rotations.reshape(n,3,9)/4),2)
    blocks['EE_exact_dwell']=(event*np.sqrt(duration)[...,None]/np.sqrt(3)).reshape(n,-1)
    X=np.concatenate(list(blocks.values()),1).astype(np.float32)
    # The raw standing velocity is observable for one initial-reset step only; explicit provenance, not the regular episode descriptor.
    metadata=dict(dimensions=X.shape[1],block_dimensions={k:v.shape[1] for k,v in blocks.items()},normalization='Fixed native physical ranges; equal time-averaged position/velocity/standing blocks; SO3 native-reference and perturbation rotations split total orientation weight.',reference=reference,
        primary_lifecycle='Regular native autoreset, including one compute on reset frame',initial_reset_sensitivity='One-control-step offset; explicit initial reset raw standing velocity lasts one step, regular autoreset applies zero mask before first action.',
        excluded=['all EE slot3 fields','all velocity slot2/3 fields','EE slot2 timer','all velocity timer inputs (constant10s)','raw standing uniforms beyond their boolean marker','raw velocity values when standing in regular episodes (initial-reset slot0 pulse is retained separately)'],
        rotation_limitation='Reference SO3 target follows the exact native function at a fixed pre-policy reset state. Later state-dependent target rotations are not precomputed; perturbation SO3 descriptors retain their input dependence. No claim of full moving-state orientation equivalence.')
    content=dict(position=pos,rotation=rotations,perturb_rotation=perturb_rotations,velocity=effective_speed,standing=standing,trace=trace,starts=starts,duration=duration)
    return (X,metadata,content) if return_content else (X,metadata)

def balanced_partition(X,capacity=8,seed=20261104,power_steps=8):
    """Recursive deterministic principal-axis median split, exact capacity at every leaf.

    This is not unconstrained KMeans plus arbitrary leftovers. At each node,
    a command-only principal axis divides an even population in half; stable
    sequence-ID tie breaking makes the final32768/8 leaves reproducible.
    """
    n,d=X.shape
    if n%capacity or ((n//capacity)&(n//capacity-1)):raise ValueError('Requires a power-of-two number of equal leaves')
    rng=np.random.default_rng(seed);mapping=np.full(n,-1,dtype=np.int64);leaf=0
    def split(ids):
        nonlocal leaf
        if len(ids)==capacity:mapping[ids]=leaf;leaf+=1;return
        A=X[ids].astype(np.float64);A-=A.mean(0)
        if len(ids)<=64:
            G=A@A.T;v=rng.normal(size=len(ids))
            for _ in range(power_steps):v=G@v;v/=max(np.linalg.norm(v),1e-30)
            projection=v
        else:
            v=rng.normal(size=d)
            for _ in range(power_steps):v=A.T@(A@v);v/=max(np.linalg.norm(v),1e-30)
            projection=A@v
        if projection[np.argmax(np.abs(projection))]<0:projection=-projection
        order=np.lexsort((ids,projection));m=len(ids)//2;split(ids[order[:m]]);split(ids[order[m:]])
    split(np.arange(n));assert leaf==n//capacity and np.all(np.bincount(mapping)==capacity)
    return mapping
