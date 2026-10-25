"""Read the actual USD joint frames; validate FK before bounded IK screening."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

_MODEL_CACHE={}


def rotation_wxyz(q):
    q = np.asarray(q)
    return Rotation.from_quat(q[..., [1,2,3,0]])


def export_arm_chain(asset_path):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(asset_path)
    edges = {}
    def frame(pos, quat):
        t = np.eye(4)
        t[:3,3] = np.array(pos or (0.,0.,0.))
        if quat:
            t[:3,:3] = Rotation.from_quat([*quat.GetImaginary(),quat.GetReal()]).as_matrix()
        return t.tolist()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint): continue
        joint=UsdPhysics.Joint(prim)
        b0=joint.GetBody0Rel().GetTargets();b1=joint.GetBody1Rel().GetTargets()
        if not b0 or not b1:continue
        parent=str(b0[0]).split('/')[-1];child=str(b1[0]).split('/')[-1]
        row=dict(name=prim.GetName(),parent=parent,child=child,
                 frame0=frame(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get()),
                 frame1=frame(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get()))
        if prim.IsA(UsdPhysics.RevoluteJoint):row['axis']=UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
        elif not prim.IsA(UsdPhysics.FixedJoint):continue
        edges.setdefault(parent,[]).append(row)
    def find(body,seen):
        if body=='end_effector':return []
        for e in edges.get(body,[]):
            if e['child'] in seen:continue
            result=find(e['child'],seen|{e['child']})
            if result is not None:return [e]+result
        return None
    chain=find('link0',{'link0'})
    assert chain and sorted(e['name'] for e in chain if 'axis' in e)==[f'joint{i}' for i in range(1,7)],chain
    return chain


def fk(chain,q):
    if id(chain) not in _MODEL_CACHE:
        frames=[]
        for e in chain:
            axis=np.eye(3)['XYZ'.index(e['axis'])] if 'axis' in e else None
            index=int(e['name'][5:])-1 if axis is not None else None
            frames.append((np.array(e['frame0']),np.linalg.inv(np.array(e['frame1'])),axis,index))
        _MODEL_CACHE[id(chain)]=(chain,frames)
    t=np.eye(4)
    for f0,f1,axis,index in _MODEL_CACHE[id(chain)][1]:
        motion=np.eye(4)
        if axis is not None:
            motion[:3,:3]=Rotation.from_rotvec(axis*q[index]).as_matrix()
        t=t@f0@motion@f1
    return t


def validate_dynamic_fk(chain,data):
    rows=[]
    for step in range(0,len(data['valid']),50):
        for env in range(min(4,data['valid'].shape[1])):
            if not data['valid'][step,env]:continue
            q=data['control_joint_position'][step,env,12:]
            t=fk(chain,q)
            root=rotation_wxyz(data['kin_link0_quat'][step,env])
            pred=root.apply(t[:3,3])+data['kin_link0_pos'][step,env]
            pos_error=np.linalg.norm(pred-data['kin_ee_pos'][step,env])
            rel=root.inv()*rotation_wxyz(data['kin_ee_quat'][step,env])
            rot_error=(Rotation.from_matrix(t[:3,:3]).inv()*rel).magnitude()
            rows.append((pos_error,rot_error))
    assert rows
    result=dict(samples=len(rows),position_error_max_m=float(max(x[0] for x in rows)),
                orientation_error_max_rad=float(max(x[1] for x in rows)))
    result['passed']=result['position_error_max_m']<1e-4 and result['orientation_error_max_rad']<1e-3
    return result


def screen_cloud(chain,cloud,soft_bounds,link0_height_offset,link0_quat):
    bounds=np.array(soft_bounds)+np.array([.02,-.02])
    root_rot=rotation_wxyz(link0_quat).as_matrix()
    assert abs(root_rot[2,2])>.9,'Mixed Z conversion needs explicit handling of this link0 mounting'
    rows=[]
    for center in cloud:
        tests=[]
        for height in (.25,.28,.32):
            x,y,z=center;link_z=height+link0_height_offset
            local=np.array([x,y,(z-link_z-root_rot[2,0]*x-root_rot[2,1]*y)/root_rot[2,2]])
            pitch=-np.arctan2(z-link_z,np.hypot(x,y))+.6;yaw=np.arctan2(y,x)
            desired=Rotation.from_euler('xyz',[0.,pitch,yaw])
            def residual(q):
                t=fk(chain,q)
                return np.r_[(t[:3,3]-local)/.01,(desired.inv()*Rotation.from_matrix(t[:3,:3])).as_rotvec()/.1]
            candidates=[]
            for initial in ([0.,.8,-1.3,0.,.5,0.],[0.,1.4,-2.,0.,.8,0.],[0.,.5,-.8,0.,-.3,0.]):
                initial=np.clip(initial,bounds[:,0]+1e-5,bounds[:,1]-1e-5)
                result=least_squares(residual,initial,bounds=(bounds[:,0],bounds[:,1]),max_nfev=160,
                                     ftol=1e-8,xtol=1e-8,gtol=1e-8)
                t=fk(chain,result.x)
                pe=float(np.linalg.norm(t[:3,3]-local));re=float((desired.inv()*Rotation.from_matrix(t[:3,:3])).magnitude())
                candidates.append((pe+re*.1,pe,re,result.x))
                if pe<.0005 and re<.005:break
            _,pe,re,q=min(candidates,key=lambda x:x[0])
            tests.append(dict(base_height_m=height,position_error_m=pe,orientation_error_rad=re,
                joint_angles_rad=q.tolist(),margin_inside_original_soft_rad=float(np.minimum(q-np.array(soft_bounds)[:,0],np.array(soft_bounds)[:,1]-q).min()),
                passed=pe<.003 and re<.03))
        rows.append(dict(target_m=list(center),tests=tests,kinematically_admitted=all(t['passed'] for t in tests)))
    return rows
