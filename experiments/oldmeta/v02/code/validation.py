"""Physical command substitution check with inspectable diagnostics."""
import torch
from commands import UniformPoseWBCCommand,_native_wbc_fixed


def command_equivalence(term):
    ids=torch.arange(min(term.num_envs,128),device=term.device)
    u=term._env.sequence_adapter.pool.ee_tensor[torch.arange(len(ids),device=term.device),0]
    before=term.pose_command.clone()
    saved_inputs=term._sequence_inputs
    original_rand=torch.rand
    calls=[]
    def fixed_rand(*args,**kwargs):
        assert tuple(args)==(len(ids),3)
        value=u[:,:3] if len(calls)==0 else u[:,3:6]
        calls.append(1)
        return value
    try:
        torch.rand=fixed_rand
        UniformPoseWBCCommand._resample_command(term,ids)
        native=term.pose_command[ids].clone()
    finally:
        torch.rand=original_rand
        term.pose_command.copy_(before)
    try:
        term._sequence_inputs=u
        _native_wbc_fixed(term,ids)
        actual=term.pose_command[ids].clone()
    finally:
        term._sequence_inputs=saved_inputs
        term.pose_command.copy_(before)
    difference=(native-actual).abs()
    # Use physical rotation distance, allowing FP32/JIT arithmetic roundoff.
    q=native[:,3:].double();q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    r=actual[:,3:].double();r=r/torch.linalg.vector_norm(r,dim=-1,keepdim=True)
    r=r*torch.where((q*r).sum(-1,keepdim=True)>=0,1.,-1.)
    angle=4*torch.atan2(torch.linalg.vector_norm(q-r,dim=-1),torch.linalg.vector_norm(q+r,dim=-1))
    return dict(passed=len(calls)==2 and bool(torch.isfinite(difference).all())
                and float(difference[:,:3].max())<1e-7 and float(angle.max())<1e-5,
                position_absolute_tolerance_m=1e-7,rotation_tolerance_rad=1e-5,
                max_rotation_difference_rad=float(angle.max()),
                calls=len(calls),native_finite=bool(torch.isfinite(native).all()),actual_finite=bool(torch.isfinite(actual).all()),
                per_column_max=torch.nan_to_num(difference,nan=1e30).max(0).values.cpu().tolist(),
                native_first=native[:3].cpu().tolist(),actual_first=actual[:3].cpu().tolist())
