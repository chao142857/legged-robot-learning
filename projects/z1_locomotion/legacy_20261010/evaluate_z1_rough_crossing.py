"""Use existing Z1 PPO/contact/video evaluator on upstream Rough terrain entrance-to-exit."""
from pathlib import Path
source=Path(__file__).with_name('evaluate_z1_stairs.py').read_text()
source=source.replace("add_launcher_args(parser)","parser.add_argument('--terrain-type',required=True,choices=['random_rough','boxes','hf_pyramid_slope'])\nparser.add_argument('--difficulty',type=float,default=.25)\nadd_launcher_args(parser)")
start=source.index('cfg.terrain.terrain_generator = TerrainGeneratorCfg(')
end=source.index('cfg.terrain.max_init_terrain_level = 0',start)
source=source[:start]+"""native_generator=cfg.terrain.terrain_generator.copy()
sub=native_generator.sub_terrains[args.terrain_type].copy();sub.proportion=1.
native_generator.num_rows=1;native_generator.num_cols=1;native_generator.curriculum=False
native_generator.difficulty_range=(args.difficulty,args.difficulty)
native_generator.sub_terrains={args.terrain_type:sub};native_generator.seed=args.seed
native_generator.use_cache=False;native_generator.border_width=2.
cfg.terrain.terrain_generator=native_generator
"""+source[end:]
source=source.replace('state[:,:3] += self._terrain.env_origins[ids]',"""state[:,:3] += self._terrain.env_origins[ids]
        state[:,0] -= 3.4
        # Height from the actual generated triangular surface, not center height.
        from pxr import UsdGeom
        import numpy as np
        mesh=UsdGeom.Mesh(self.scene.stage.GetPrimAtPath(self._terrain.terrain_prim_paths[0]+'/mesh'))
        vertices=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int).reshape(-1,3)
        transform=np.asarray(UsdGeom.Xformable(mesh.GetPrim()).ComputeLocalToWorldTransform(0),dtype=float)
        vertices=np.concatenate([vertices,np.ones((len(vertices),1))],axis=1)@transform
        triangles=vertices[indices,:3]
        def surface(x,y):
            v0=triangles[:,1,:2]-triangles[:,0,:2];v1=triangles[:,2,:2]-triangles[:,0,:2];v2=np.array([x,y])-triangles[:,0,:2]
            den=v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0];good=np.abs(den)>1e-12
            u=np.zeros(len(den));v=u.copy()
            u[good]=(v2[good,0]*v1[good,1]-v2[good,1]*v1[good,0])/den[good]
            v[good]=(v0[good,0]*v2[good,1]-v0[good,1]*v2[good,0])/den[good]
            hits=good&(u>=-1e-6)&(v>=-1e-6)&(u+v<=1+1e-6)
            assert hits.any(),'Spawn point outside generated terrain'
            return float((triangles[:,0,2]+u*(triangles[:,1,2]-triangles[:,0,2])+v*(triangles[:,2,2]-triangles[:,0,2]))[hits].max())
        state[:,2]=torch.tensor([surface(float(p[0]),float(p[1]))+self.cfg.robot.init_state.pos[2] for p in state],device=self.device)
""")
source=source.replace('goal_progress = 1.8+8*0.30+0.4-0.9','goal_progress = 3.4')
source=source.replace('reached = (pos[:,0]>=goal_progress)&upright&supported&correct_height',"reached = (pos[:,0]>=goal_progress)&upright&supported&((snap['foot_pos']-origin[:,None,:])[:,:,0].amin(dim=1)>2.9)")
source=source.replace("'direction':'down' if args.descending else 'up','risers':8,'tread_m':0.30", "'direction':'entrance_to_exit','terrain_type':args.terrain_type,'difficulty':args.difficulty,'entrance_relative_x_m':-3.4,'exit_relative_x_m':3.4,'risers':None,'tread_m':None")
source=source.replace('Cross all eight risers plus 0.4m landing, upright, foot support and matching final elevation before first fall/reset/lateral exit','Traverse generated8m tile from entry x-3.4 to exit x+3.4, allfourfootcenters x>2.9, upright/foot support before first fall/reset/lateralexit')
source=source.replace('Eight-riser first-trial traversal','Entrance-to-exit first-trial traversal')
source=source.replace("'height_m':args.step_height","'height_m':None")
exec(compile(source,str(Path(__file__).with_name('evaluate_z1_stairs.py')),'exec'))
