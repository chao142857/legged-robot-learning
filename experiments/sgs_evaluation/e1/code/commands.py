"""Native command classes with only their random inputs replaced by a fixed sequence."""
import inspect,textwrap
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils.math import quat_from_euler_xyz,quat_unique
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_b import UniformBodyPoseCommand
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_wbc import UniformPoseWBCCommand

def mapped(u,bounds):return bounds[0]+u*(bounds[1]-bounds[0])

class SequenceMixin:
    sequence_kind=None
    def _resample(self,env_ids):
        adapter=getattr(self._env,'sequence_adapter',None)
        if adapter is None:return super()._resample(env_ids)
        if len(env_ids)==0:return
        self._sequence_inputs=adapter.inputs(self.sequence_kind,env_ids)
        interval=self.cfg.resampling_time_range
        self.time_left[env_ids]=mapped(self._sequence_inputs[:,-1],interval)
        self._resample_command(env_ids)
        self.command_counter[env_ids]+=1
        adapter.on_command(self.sequence_kind,env_ids,self.command[env_ids])

class SequenceFlat(SequenceMixin,UniformBodyPoseCommand):
    sequence_kind='ee'
    def _resample_command(self,env_ids):
        if not hasattr(self,'_sequence_inputs'):return super()._resample_command(env_ids)
        u=self._sequence_inputs
        for j,k in enumerate(('pos_x','pos_y','pos_z')):self.pose_command_b[env_ids,j]=mapped(u[:,j],getattr(self.cfg.ranges,k))
        e=torch.stack([mapped(u[:,j+3],getattr(self.cfg.ranges,k)) for j,k in enumerate(('roll','pitch','yaw'))],dim=1)
        q=quat_from_euler_xyz(e[:,0],e[:,1],e[:,2]);self.pose_command_b[env_ids,3:]=quat_unique(q) if self.cfg.make_quat_unique else q

# Execute the upstream WBC function verbatim, substituting just the two uniform
# draws. This preserves its state-dependent pitch/yaw, mixed frame and clamping.
_src=textwrap.dedent(inspect.getsource(UniformPoseWBCCommand._resample_command))
_old_pos='rand_pos = torch.rand(num_resample, 3, device=self.device)'
_old_euler='rand_euler = torch.rand(num_resample, 3, device=self.device)'
assert _src.count(_old_pos)==1 and _src.count(_old_euler)==1
_src=_src.replace(_old_pos,'rand_pos = self._sequence_inputs[:, :3]').replace(_old_euler,'rand_euler = self._sequence_inputs[:, 3:6]')
_namespace={};exec(compile(_src,'<native_wbc_fixed_random_inputs>','exec'),UniformPoseWBCCommand._resample_command.__globals__.copy(),_namespace)
_native_wbc_fixed=_namespace['_resample_command']

class SequenceWBC(SequenceMixin,UniformPoseWBCCommand):
    sequence_kind='ee'
    def _resample_command(self,env_ids):
        if not hasattr(self,'_sequence_inputs'):return super()._resample_command(env_ids)
        return _native_wbc_fixed(self,env_ids)

class SequenceVelocity(SequenceMixin,UniformVelocityCommand):
    sequence_kind='velocity'
    def _resample_command(self,env_ids):
        if not hasattr(self,'_sequence_inputs'):return super()._resample_command(env_ids)
        assert not self.cfg.heading_command,'This frozen native pair has heading_command=False'
        u=self._sequence_inputs
        for j,k in enumerate(('lin_vel_x','lin_vel_y','ang_vel_z')):self.vel_command_b[env_ids,j]=mapped(u[:,j],getattr(self.cfg.ranges,k))
        self.is_standing_env[env_ids]=u[:,3]<=self.cfg.rel_standing_envs
        # Native _update_command applies the standing zero mask at its original
        # lifecycle position, including the native first step following reset.

def wbc_equivalence(term):
    if not isinstance(term,SequenceWBC):return dict(applicable=False)
    ids=torch.arange(min(term.num_envs,128),device=term.device)
    u=term._env.sequence_adapter.pool.ee_tensor[torch.arange(len(ids),device=term.device),0]
    before=term.pose_command.clone();original_rand=torch.rand;calls=[]
    def fixed_rand(*args,**kwargs):
        assert tuple(args)==(len(ids),3)
        value=u[:,:3] if len(calls)==0 else u[:,3:6];calls.append(1);return value
    try:
        torch.rand=fixed_rand
        UniformPoseWBCCommand._resample_command(term,ids)
        native=term.pose_command[ids].clone()
    finally:torch.rand=original_rand;term.pose_command.copy_(before)
    old_inputs=term._sequence_inputs;term._sequence_inputs=u
    try:_native_wbc_fixed(term,ids);actual=term.pose_command[ids].clone()
    finally:term.pose_command.copy_(before);term._sequence_inputs=old_inputs
    error=float((native-actual).abs().max());assert len(calls)==2 and error<1e-6
    return dict(applicable=True,passed=True,samples=len(ids),max_absolute_error=error,native_state_dependent_math_unchanged=True)
