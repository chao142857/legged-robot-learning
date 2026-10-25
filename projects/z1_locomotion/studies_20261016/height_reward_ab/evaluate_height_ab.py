"""Reuse archived first-trial evaluator for fixed stand/flat/stair inference, without PPO updates."""
import argparse, hashlib, json, sys
from pathlib import Path
from local_gate import validate
validate()
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--test-mode', choices=['stand','flat','stairs','rough','boxes','slopes'], required=True)
mode, remaining = p.parse_known_args()
original = Path(__file__).with_name('evaluate_z1_stairs.py')
source = original.read_text()
if mode.test_mode in ('rough','boxes','slopes'):
    # Execute only the existing rough evaluator's source transformations, reusing its native generator/protocol.
    template=Path(__file__).with_name('evaluate_z1_rough_crossing.py').read_text()
    transformations=template[template.index('source=source.replace'):template.rindex('exec(compile')]
    context={'source':source}
    exec(transformations,globals(),context)
    source=context['source']
    remaining += ['--terrain-type',{'rough':'random_rough','boxes':'boxes','slopes':'hf_pyramid_slope'}[mode.test_mode],'--difficulty','.25']
source = source.replace('or args.speed <= 0:', 'or args.speed < 0:')
source = source.replace('obs,reward,dones,info = wrapper.step(policy(obs))', '''action = policy(obs)
            assert bool(torch.isfinite(action).all()), 'Nonfinite inference action'
            obs,reward,dones,info = wrapper.step(action)
            assert bool(torch.isfinite(obs['policy']).all()) and bool(torch.isfinite(obs['critic']).all())''')
source = source.replace("obs = wrapper.get_observations()", """obs = wrapper.get_observations()
    assert obs['policy'].shape[-1] == 260 and obs['critic'].shape[-1] == 321
    assert wrapper.num_actions == 12
    assert e.cfg.history_length == 5 and not getattr(e.cfg,'use_vision',False)
    before_actor = {k:v.detach().clone() for k,v in runner.alg.actor.state_dict().items()}
    inference_contract = {'actor_dim':260,'critic_dim':321,'action_dim':12,
        'history_length':e.cfg.history_length,'history_order':'oldest to newest',
        'action_scale':e.cfg.action_scale,'filtered':e.cfg.use_filter_actions,
        'action_filter':'0.8 current clipped raw action + 0.2 previous clipped raw action',
        'joint_order':[e._robot.joint_names[i] for i in e._ids_joints_order],
        'default_q_policy_order':t(e._robot.data.default_joint_pos)[0,e._ids_joints_order].tolist(),
        'actor_config':agent.actor.to_dict() if hasattr(agent,'actor') else None,
        'initial_action_history_max':float(e._previous_actions.abs().max()),
        'normalization_note':'Runner created from archived native config; checkpoint loaded without overriding normalization'}""")
source = source.replace("'contact_profile':cfg.z1_contact_profile", "'inference_contract':inference_contract,'contact_profile':cfg.z1_contact_profile")
source = source.replace("    temporary = args.output", """    assert all(torch.equal(v,before_actor[k]) for k,v in runner.alg.actor.state_dict().items()), 'Inference modified actor'
    assert all(bool(torch.isfinite(v).all()) for v in runner.alg.actor.state_dict().values())
    result['weights_unchanged'] = True
    temporary = args.output""", 1)
if mode.test_mode in ('flat','stand'):
    marker = "cfg.terrain.max_init_terrain_level = 0"
    source = source.replace(marker, """cfg.terrain.terrain_type = 'generator'
cfg.terrain.terrain_generator = TerrainGeneratorCfg(size=(12,6),num_rows=1,num_cols=1,
    border_width=2,seed=args.seed,use_cache=False,sub_terrains={
    'closed_flat':StaircaseCfg(step_height=0.,num_steps=0)})
""" + marker)
    source = source.replace('goal_progress = 1.8+8*0.30+0.4-0.9', 'goal_progress = args.speed*args.seconds*.7')
    source = source.replace('end_ground_delta = -8*args.step_height if args.descending else 8*args.step_height', 'end_ground_delta = 0')
    if mode.test_mode == 'stand':
        source = source.replace('reached = (pos[:,0]>=goal_progress)&upright&supported&correct_height',
            'reached = ((step+1)*e.step_dt >= args.seconds-.001)&(pos[:,:2].norm(dim=1)<.3)&upright&supported&correct_height')
else:
    source = source.replace('reached = (pos[:,0]>=goal_progress)&upright&supported&correct_height',
        'reached = (pos[:,0]>=goal_progress)&((snap[\'foot_pos\']-origin[:,None,:])[:,:,0].amin(dim=1)>3.3)&upright&supported&correct_height')
sys.argv = [str(original)] + remaining
exec(compile(source, str(original), 'exec'), {'__name__':'__main__','__file__':str(original)})
output = Path(remaining[remaining.index('--output')+1])
result = json.loads(output.read_text())
result.update(test_mode=mode.test_mode, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              no_weight_updates=True, no_external_body_force=True, no_video=True)
if mode.test_mode in ('stand','flat'):
    result.update(height_m=0,risers=0,direction=None,
        success_definition=('Fixed zero body command, upright/supported and within0.3m for full duration before reset/fall' if mode.test_mode=='stand'
            else 'Fixed low-speed body command, crosses70% of commanded distance upright/supported before first reset/fall/lateral exit'))
output.write_text(json.dumps(result,indent=2))

if mode.test_mode in ('stand','flat'):
    result['strict_safe_definition']=result['success_definition']+'; zero sampled nonfoot contacts above5N, no sampled torque>120.001Nm or speed>20.01rad/s'
    output.write_text(json.dumps(result,indent=2))
