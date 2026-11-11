import pathlib,json,csv,time,hashlib,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write,file_sha
from command_features import features,timeline,balanced_partition,SAMPLE_STEPS
from grouped_sampling import mapping_hash

start=time.time();OUT=ROOT/'artifacts/groups';OUT.mkdir(parents=True,exist_ok=True)
protocol=json.loads((ROOT/'protocol/E2_protocol_frozen.json').read_text())
pool=ROOT/'pools/wbc_train.npz';raw=np.load(pool);ee=raw['ee'];velocity=raw['velocity'];assert len(ee)==32768
refs=json.loads((ROOT/'artifacts/structure_v2/reset_reference_states.json').read_text());reference=dict(height=refs['link0_positions'][0][2],quaternion=refs['link0_quaternions'][0])
trace0,starts0=timeline(ee[:16],velocity[:16],False);trace1,starts1=timeline(ee[:16],velocity[:16],True)
events=json.loads((ROOT/'artifacts/structure_v2/resample_events.json').read_text())
for e in events:
    starts=starts0 if e['generation']==1 else starts1
    assert int(starts[e['sequence_id'],0 if e['kind']=='ee' else 1,e['slot']])==e['episode_step'],e
samples=json.loads((ROOT/'artifacts/structure_v2/control_samples.json').read_text())
for s in samples:
    trace=trace0 if s['generation']==1 else trace1
    assert list(trace[s['env_id'],s['episode_step']])==[s['ee_slot'],s['velocity_slot']]
print('NATIVE_TRACE_TIMELINE_MATCH '+str(len(samples)),flush=True)
X,meta,content=features(ee,velocity,reference,True)
write(OUT/'feature_definition.json',dict(**meta,inputs=[str(pool),str(ROOT/'artifacts/structure_v2/reset_reference_states.json')],no_feedback_or_eval_inputs=True,pool_sha256=file_sha(pool)))
np.save(OUT/'command_features.npy',X);np.savez_compressed(OUT/'effective_content.npz',**content)
print('FEATURES '+str(X.shape),flush=True)
maps={}
for seed in protocol['random_group_seeds']:
    permutation=np.random.default_rng(seed).permutation(len(X));m=np.empty(len(X),dtype=np.int64);m[permutation]=np.repeat(np.arange(4096),8);maps['random_'+str(seed)]=m
maps['command']=balanced_partition(X,8,protocol['command_cluster_seed'])
metadata={}
for name,m in maps.items():
    assert np.all(np.bincount(m)==8);np.save(OUT/('mapping_'+name+'.npy'),m)
    metadata[name]=dict(seed=protocol['command_cluster_seed'] if name=='command' else int(name.split('_')[-1]),groups=4096,capacity=8,mapping_sha256=mapping_hash(m))
with (OUT/'sequence_to_group_all_mappings.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['sequence_id']+list(maps))
    for i in range(len(X)):w.writerow([i]+[int(m[i]) for m in maps.values()])
write(OUT/'mapping_manifest.json',dict(mappings=metadata,pool_sha256=file_sha(pool),features_sha256=file_sha(OUT/'command_features.npy'),train_only=True,evaluation_pool_not_loaded=True,
    algorithm='Recursive principal-axis median partition; power_steps8; capacity8; command features normalized by native physical ranges; stable ID ties.',protocol_sha256=file_sha(ROOT/'protocol/E2_protocol_frozen.json')))
pair_indices=np.array(list(itertools.combinations(range(8),2)));pair_dist={};geometry=[];per_group=[]
for name,m in maps.items():
    members=np.argsort(m,kind='stable').reshape(4096,8);pairs=members[:,pair_indices].reshape(-1,2);D=[];angles=[];positions=[]
    for chunk in np.array_split(pairs,64):
        a,b=chunk.T;D.extend(np.linalg.norm(X[a]-X[b],axis=1).tolist())
        ia=np.arange(len(a))[:,None];sa=content['trace'][a][:,SAMPLE_STEPS,0].astype(int);sb=content['trace'][b][:,SAMPLE_STEPS,0].astype(int)
        Ra=content['rotation'][a][ia,sa];Rb=content['rotation'][b][ia,sb]
        theta=np.arccos(np.clip(((Ra*Rb).sum((-1,-2))-1)/2,-1,1));angles.extend((theta.mean(1)*180/np.pi).tolist())
        pa=content['position'][a][ia,sa];pb=content['position'][b][ia,sb];positions.extend(np.sqrt(((pa-pb)/np.array([.25,.7,.7]))**2).mean((1,2)).tolist())
    dist=np.array(D);pair_dist[name]=dist;group_means=dist.reshape(4096,28).mean(1)
    row=dict(mapping=name,pairs=len(dist),mean_feature_distance=float(dist.mean()),median_feature_distance=float(np.median(dist)),p95_feature_distance=float(np.quantile(dist,.95)),mean_reference_SO3_geodesic_deg=float(np.mean(angles)),mean_normalized_position_abs_difference=float(np.mean(positions)))
    geometry.append(row)
    for g,d in enumerate(group_means):per_group.append(dict(mapping=name,group_id=g,mean_pair_feature_distance=float(d)))
    print('GEOMETRY '+json.dumps(row),flush=True)
for name,rows in [('geometry_comparison.csv',geometry),('within_group_command_distances.csv',per_group)]:
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
usage=dict(EE_slot_exposure_steps=[int((content['trace'][:,:,0]==j).sum()) for j in range(4)],velocity_slot_exposure_steps=[int((content['trace'][:,:,1]==j).sum()) for j in range(4)],
    standing_fraction_effective_slots=content['standing'].mean(0).tolist(),EE_switch_seconds_quantiles=[np.quantile(content['starts'][:,0,j]*.02,[0,.25,.5,.75,1]).tolist() for j in (1,2)],
    native_trace_matched_samples=len(samples),normal_full_episode_slots=dict(EE=[0,1,2],velocity_initial=[0,1],velocity_autoreset=[0,1,2]),velocity_slot2_autoreset_steps=1)
write(OUT/'geometry_summary.json',dict(passed=True,rows=geometry,slot_usage=usage,wall_seconds=time.time()-start,reference_rotation_limitation=meta['rotation_limitation']))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for name,d in pair_dist.items():axes[0].hist(d,bins=55,density=True,histtype='step',lw=1.5,label=name)
axes[0].set_xlabel('Normalized effective-command distance');axes[0].set_ylabel('Density');axes[0].legend(fontsize=7);axes[0].set_title('Within-group command distances')
for j in range(3):axes[1].plot(np.arange(0,1000,25)*.02,[(content['trace'][:,t,0]==j).mean() for t in range(0,1000,25)],label='EE slot'+str(j))
axes[1].set_xlabel('Episode time (s)');axes[1].set_ylabel('Sequence share');axes[1].set_title('Effective EE command order');axes[1].legend(fontsize=8)
axes[2].bar([x['mapping'] for x in geometry],[x['mean_reference_SO3_geodesic_deg'] for x in geometry]);axes[2].tick_params(axis='x',rotation=25);axes[2].set_ylabel('Mean SO(3) angle (deg)');axes[2].set_title('Native targets at fixed reset reference')
fig.tight_layout();fig.savefig(OUT/'command_geometry.png',dpi=180);fig.savefig(OUT/'command_geometry.svg');plt.close(fig)
print('BUILD_GROUPS_PASS '+str(time.time()-start),flush=True)
