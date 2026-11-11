"""Read-only command comparison. Original physical tolerances are unchanged."""
import torch
POSITION_ATOL_M=1e-7
ROTATION_ATOL_RAD=1e-5

def compare_commands(native,actual,calls=2):
 assert native.shape==actual.shape and native.ndim==2 and native.shape[1]==7 and len(native)>0
 difference=(native-actual).abs()
 finite=bool(torch.isfinite(native).all() and torch.isfinite(actual).all())
 if finite:
  q=native[:,3:].double();q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
  r=actual[:,3:].double();r=r/torch.linalg.vector_norm(r,dim=-1,keepdim=True)
  r=r*torch.where((q*r).sum(-1,keepdim=True)>=0,1.,-1.)
  angle=4*torch.atan2(torch.linalg.vector_norm(q-r,dim=-1),torch.linalg.vector_norm(q+r,dim=-1))
  rotation=float(angle.max()) if bool(torch.isfinite(angle).all()) else None
 else:rotation=None
 position=float(difference[:,:3].max()) if finite else None
 return dict(passed=calls==2 and finite and position is not None and position<POSITION_ATOL_M and rotation is not None and rotation<ROTATION_ATOL_RAD,
  position_absolute_tolerance_m=POSITION_ATOL_M,rotation_tolerance_rad=ROTATION_ATOL_RAD,
  max_position_difference_m=position,max_rotation_difference_rad=rotation,calls=calls,
  native_finite=bool(torch.isfinite(native).all()),actual_finite=bool(torch.isfinite(actual).all()),
  per_column_max=torch.nan_to_num(difference,nan=1e30,posinf=1e30,neginf=1e30).max(0).values.cpu().tolist())
