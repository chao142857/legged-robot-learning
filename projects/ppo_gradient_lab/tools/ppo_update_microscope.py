"""CPU-only native RSL PPO single-step replay on frozen diagnostic rollouts.

Never imports a simulator or writes a checkpoint. Each candidate restores a fresh
CP100 + Adam copy. OneBatchStorage limits native PPO.update to exactly one step.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from diagnose_go2_piper_gradients import sha256, complete_mc
from analyze_go2_gradient_probe import ridge_fit, ridge_predict, value_metrics, parameter_scopes

SEEDS=(8201,8202,8203)
UPDATE_STARTS=(0,120,240,360,376)
ESTIMATORS=('mc','probe_mc_baseline','probe_gae_finite','probe_gae_native_timeout')


def tensor_hash(tree):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(tree);return h.hexdigest()


def native_gae(rewards,values,last_values,dones,timeouts,gamma=.99,lam=.95,timeout_current=True):
    """Scalar or component [T,N,K] GAE with native current-V timeout rule.

    Return rounding mirrors RSL: returns=A+V then raw_adv=returns-V.
    last_values is required unless every final transition is done.
    """
    if rewards.shape!=values.shape or rewards.ndim!=3:raise ValueError('Require matching [T,N,K]')
    if dones.shape!=rewards.shape[:2] or timeouts.shape!=dones.shape:raise ValueError('Bad boundary shape')
    if torch.any(timeouts & ~dones):raise ValueError('Timeout must be done')
    if last_values is None:
        if not dones[-1].all():raise ValueError('Missing nonterminal bootstrap')
        last_values=torch.zeros_like(values[-1])  # Never used: final done mask is zero.
    if last_values.shape!=values.shape[1:]:raise ValueError('Bad bootstrap shape')
    rr=rewards+gamma*values*timeouts.unsqueeze(-1) if timeout_current else rewards
    returns=torch.empty_like(values);tail=torch.zeros_like(values[0])
    for t in reversed(range(len(values))):
        nxt=last_values if t==len(values)-1 else values[t+1]
        alive=(~dones[t]).unsqueeze(-1)
        delta=rr[t]+alive*gamma*nxt-values[t]
        tail=delta+alive*gamma*lam*tail
        returns[t]=tail+values[t]
    return returns-values,returns


def select_window(raw,start,length=24):
    end=start+length;T,N=raw['valid'].shape
    if start<0 or end>T:raise ValueError('Incomplete rollout window')
    mask=raw['valid'][start:end].all(0)
    if end==T:mask &= raw['episode_end'][end-1]
    else:mask &= raw['episode_end'][end-1] | raw['valid'][end]
    ids=mask.nonzero().flatten()
    if len(ids)==0:raise ValueError('No eligible complete window')
    return ids,dict(start=start,end=end,eligible_envs=len(ids),excluded_envs=N-len(ids),
        survivor_conditioned=start>0 or len(ids)<N,overlapping_boundary_probe=start==376,
        terminal_last_without_next=int(raw['episode_end'][end-1,ids].sum()) if end==T else 0)


def norm(a):return (a-a.mean())/(a.std()+1e-8)

def vector(model):return torch.cat([p.detach().flatten() for p in model.parameters()])

def flat_grad(loss,model,retain_graph=False):
    parts=torch.autograd.grad(loss,list(model.parameters()),retain_graph=retain_graph,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(model.parameters(),parts)])

def cosine(a,b):
    den=a.double().norm()*b.double().norm()
    return float(a.double().dot(b.double())/den) if den>1e-20 else None


def normal(actor,obs):
    mu=actor(obs);std=actor.distribution.log_std_param.exp().expand_as(mu)
    return torch.distributions.Normal(mu,std)


def calibration(raw,probe,start):
    ids,info=select_window(raw,start);end=start+24
    r=raw['component_rewards'][start:end,ids];v=raw['native_value'][start:end,ids,None]
    vp=probe[start:end,ids];done=raw['episode_end'][start:end,ids];to=raw['timeout'][start:end,ids]
    last=raw['native_value'][end,ids,None] if end<len(raw['valid']) else None
    plast=probe[end,ids] if end<len(raw['valid']) else None
    an,ret=native_gae(raw['total_reward'][start:end,ids,None],v,last,done,to)
    pc,_=native_gae(r,vp,plast,done,to)
    pt,_=native_gae(r,vp,plast,done,to,timeout_current=False)
    # Independent reference executes the installed native compute_returns method.
    st=SimpleNamespace(num_transitions_per_env=24,values=v,
        rewards=raw['total_reward'][start:end,ids,None]+.99*v*to.unsqueeze(-1),
        dones=done.unsqueeze(-1),returns=torch.zeros_like(v))
    holder=SimpleNamespace(storage=st,critic=lambda _:last if last is not None else torch.zeros_like(v[-1]),
        gamma=.99,lam=.95,normalize_advantage_per_mini_batch=False)
    PPO.compute_returns(holder,None)
    torch.testing.assert_close(st.returns,ret,rtol=0,atol=0)
    torch.testing.assert_close(st.advantages,norm(an),rtol=0,atol=0)
    anf,_=native_gae(raw['total_reward'][start:end,ids,None],v,last,done,to,timeout_current=False)
    baseline=pc.sum(-1)-an.squeeze(-1);timeout=pt.sum(-1)-pc.sum(-1)
    sigma=an.std()+1e-8
    center=lambda x:x-x.mean()
    bn=center(baseline)/sigma;tn=center(timeout)/sigma
    normalizer=norm(pt.sum(-1))-center(pt.sum(-1))/sigma
    residual=norm(pt.sum(-1))-norm(an).squeeze(-1)-bn-tn-normalizer
    assert float(residual.abs().max())<2e-5
    rms=lambda x:float(x.double().square().mean().sqrt())
    info.update(timeouts=int(to.sum()),native_adv_mean=float(an.mean()),native_adv_std=float(an.std()),
        component_sum_adv_std=float(pt.sum(-1).std()),native_vs_probe_current_corr=cosine(center(an.flatten()),center(pc.sum(-1).flatten())),
        native_vs_probe_finite_corr=cosine(norm(an).flatten(),norm(pt.sum(-1)).flatten()),
        raw_baseline_difference_rms=rms(baseline),raw_timeout_difference_rms=rms(timeout),
        finite_baseline_difference_rms=rms(pt.sum(-1)-anf.squeeze(-1)),native_timeout_removal_rms=rms(anf-an),
        timeout_baseline_interaction_rms=rms(baseline-(pt.sum(-1)-anf.squeeze(-1))),
        standardized_baseline_rms=rms(bn),standardized_timeout_rms=rms(tn),normalizer_change_rms=rms(normalizer),
        normalized_difference_rms=rms(norm(pt.sum(-1))-norm(an).squeeze(-1)),decomposition_max_error=float(residual.abs().max()),
        native_reference_max_error=float((st.returns-ret).abs().max()))
    return ids,an.squeeze(-1),ret.squeeze(-1),pc,pt,info


def make_window(raw,probe,mc,start):
    ids,an,ret,pc,pt,info=calibration(raw,probe,start);end=start+24
    obs=TensorDict({'policy':raw['actor_obs'][start:end,ids].flatten(0,1),
        'critic':raw['critic_obs'][start:end,ids].flatten(0,1)},batch_size=[24*len(ids)])
    targets=torch.stack((mc[start:end,ids],mc[start:end,ids]-probe[start:end,ids],pt,pc),2).flatten(0,1)
    b=RolloutStorage.Batch(observations=obs,actions=raw['actions'][start:end,ids].flatten(0,1),
        values=raw['native_value'][start:end,ids].reshape(-1,1),advantages=norm(an).reshape(-1,1),returns=ret.reshape(-1,1),
        old_actions_log_prob=raw['old_logp'][start:end,ids].reshape(-1,1),
        old_distribution_params=(raw['old_mu'][start:end,ids].flatten(0,1),raw['old_std'][start:end,ids].flatten(0,1)))
    # Each estimator's components centered on the whole window, using the SAME native sigma.
    centered=(targets-targets.mean(0))/(an.std()+1e-8)
    return dict(batch=b,raw_targets=targets,centered_targets=centered,ids=ids,info=info,normalizer=float(an.std()+1e-8))


def slice_batch(b,ids):
    return RolloutStorage.Batch(observations=b.observations[ids],actions=b.actions[ids],values=b.values[ids],
        advantages=b.advantages[ids],returns=b.returns[ids],old_actions_log_prob=b.old_actions_log_prob[ids],
        old_distribution_params=tuple(x[ids] for x in b.old_distribution_params))


class OneBatchStorage:
    def __init__(self,batch):self.batch=batch;self.yielded=0;self.cleared=False
    def mini_batch_generator(self,num_mini_batches,num_epochs):
        if self.yielded:raise RuntimeError('Single-step storage cannot be reused')
        self.yielded+=1;yield self.batch
    def clear(self):self.cleared=True


def restore(cp,config,batch,lr_scale=1.):
    required=dict(clip_param=.2,entropy_coef=.005,value_loss_coef=1.,max_grad_norm=1.,gamma=.99,lam=.95,
        normalize_advantage_per_mini_batch=False,use_clipped_value_loss=True,schedule="adaptive",desired_kl=.01,
        rnd_cfg=None,symmetry_cfg=None,optimizer="adam",num_learning_epochs=5,num_mini_batches=4,share_cnn_encoders=False)
    for key,value in required.items():
        if config["algorithm"].get(key)!=value:raise ValueError("Unsupported native config: "+key)
    models=[]
    for role,group,out in [('actor','policy',18),('critic','critic',1)]:
        cfg=copy.deepcopy(config[role]);cfg.pop('class_name')
        model=MLPModel(batch.observations,{role:[group]},role,out,**cfg)
        model.load_state_dict(cp[role+'_state_dict'],strict=True);models.append(model)
    cfg=copy.deepcopy(config['algorithm']);cfg.pop('class_name');cfg.pop('share_cnn_encoders')
    alg=PPO(*models,OneBatchStorage(batch),device='cpu',**cfg)
    alg.optimizer.load_state_dict(copy.deepcopy(cp['optimizer_state_dict']))
    groups=alg.optimizer.param_groups
    if len(groups)!=1:raise ValueError('Expected one native optimizer group')
    saved_lr=cp['infos']['learning_rate']
    if abs(groups[0]['lr']-saved_lr)>1e-14:raise ValueError('Ambiguous checkpoint learning rate')
    alg.learning_rate=saved_lr*lr_scale;groups[0]['lr']=alg.learning_rate
    params=list(alg.actor.parameters())+list(alg.critic.parameters())
    if len(params)!=len(groups[0]['params']):raise ValueError('Optimizer parameter mapping changed')
    for p,loaded in zip(params,groups[0]['params']):
        if p is not loaded:raise ValueError('Optimizer order mismatch')
        state=alg.optimizer.state.get(p,{})
        for key in ('exp_avg','exp_avg_sq'):
            if key in state and state[key].shape!=p.shape:raise ValueError('Optimizer state shape mismatch')
    return alg


def losses(actor,critic,batch):
    d=normal(actor,batch.observations);lp=d.log_prob(batch.actions).sum(-1)
    ratio=(lp-batch.old_actions_log_prob.squeeze(-1)).exp();a=batch.advantages.squeeze(-1)
    pi=torch.maximum(-a*ratio,-a*ratio.clamp(.8,1.2)).mean()
    entropy=d.entropy().sum(-1).mean()
    vv=critic(batch.observations);vc=batch.values+(vv-batch.values).clamp(-.2,.2)
    vl=torch.maximum((vv-batch.returns).square(),(vc-batch.returns).square()).mean()
    return pi,vl,entropy


def component_gradients(actor,batch,targets):
    ratio=(normal(actor,batch.observations).log_prob(batch.actions).sum(-1)-batch.old_actions_log_prob.squeeze(-1)).exp()
    flat=targets.reshape(len(targets),-1);gs=[]
    for i in range(flat.shape[1]):gs.append(flat_grad(-(ratio*flat[:,i]).mean(),actor,retain_graph=i<flat.shape[1]-1))
    return torch.stack(gs).reshape(*targets.shape[1:],-1)


def native_gradients(actor,critic,batch):
    pi,vl,ent=losses(actor,critic,batch)
    gp=flat_grad(pi,actor,retain_graph=True);ga=flat_grad(pi-.005*ent,actor,retain_graph=True)
    gv=flat_grad(vl,critic)
    return gp,ga,gv,dict(surrogate=float(pi.detach()),value=float(vl.detach()),entropy=float(ent.detach()))


def evaluate(actor,critic,batch,raw_targets,centered_targets):
    with torch.no_grad():
        d=normal(actor,batch.observations);lp=d.log_prob(batch.actions).sum(-1)
        ratio=(lp-batch.old_actions_log_prob.squeeze(-1)).exp();adv=batch.advantages.squeeze(-1)
        old=torch.distributions.Normal(*batch.old_distribution_params)
        kl=torch.distributions.kl_divergence(old,d).sum(-1)
        pi,vl,ent=losses(actor,critic,batch)
        result=dict(kl_mean=float(kl.mean()),kl_max=float(kl.max()),clip_fraction=float(((ratio-1).abs()>.2).float().mean()),
            active_clip_fraction=float((((adv>0)&(ratio>1.2))|((adv<0)&(ratio<.8))).float().mean()),
            native_surrogate=float(torch.maximum(-ratio.double()*adv.double(),-ratio.double().clamp(.8,1.2)*adv.double()).mean()),
            native_surrogate_float32=float(pi),native_value_loss=float(vl),entropy=float(ent))
        samples={}
        for name,aa in [('raw',raw_targets),('centered_native_std',centered_targets)]:
            unclip=-ratio.double()[:,None,None]*aa.double()
            separate=torch.maximum(unclip,-ratio.double().clamp(.8,1.2)[:,None,None]*aa.double())
            result[name]=dict(unclipped_loss=unclip.mean(0).tolist(),separate_clipped_loss=separate.mean(0).tolist())
            samples[name]=unclip
        samples['native']=torch.maximum(-ratio.double()*adv.double(),-ratio.double().clamp(.8,1.2)*adv.double())
    return result,samples


def one_step(cp,config,batch,scale,rng_seed):
    alg=restore(cp,config,batch,scale);before_a=vector(alg.actor);before_v=vector(alg.critic)
    initial_hash=tensor_hash(dict(actor=alg.actor.state_dict(),critic=alg.critic.state_dict(),state=alg.optimizer.state_dict()['state']))
    gp,ga,gv,preloss=native_gradients(alg.actor,alg.critic,batch)
    with torch.no_grad():
        update_kl_pre=float(torch.distributions.kl_divergence(torch.distributions.Normal(*batch.old_distribution_params),normal(alg.actor,batch.observations)).sum(-1).mean())
    steps_before={id(p):float(s['step']) for p,s in alg.optimizer.state.items()}
    captured={}
    def hook(opt,args,kwargs):
        captured['actor_grad']=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten().detach().clone() for p in alg.actor.parameters()])
        captured['critic_grad']=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten().detach().clone() for p in alg.critic.parameters()])
        captured['lr']=opt.param_groups[0]['lr']
    handle=alg.optimizer.register_step_pre_hook(hook)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(rng_seed);native_log=alg.update()
    handle.remove()
    if alg.storage.yielded!=1 or not alg.storage.cleared:raise RuntimeError('Not a single native step')
    for p,s in alg.optimizer.state.items():
        expected=steps_before.get(id(p),0)+(p.grad is not None)
        if float(s['step'])!=expected:raise RuntimeError('Adam step count mismatch')
    torch.testing.assert_close(captured['actor_grad'],ga*min(1.,1./(float(ga.norm())+1e-6)),rtol=2e-5,atol=3e-7)
    torch.testing.assert_close(captured['critic_grad'],gv*min(1.,1./(float(gv.norm())+1e-6)),rtol=2e-5,atol=3e-7)
    da=vector(alg.actor)-before_a;dv=vector(alg.critic)-before_v
    postp,posta,_,postloss=native_gradients(alg.actor,alg.critic,batch)
    info=dict(lr_scale=scale,lr_before=cp['infos']['learning_rate']*scale,lr_actual=captured['lr'],
        initial_model_moments_sha256=initial_hash,optimizer_steps=1,update_kl_pre=update_kl_pre,actor_delta_norm=float(da.norm()),critic_delta_norm=float(dv.norm()),
        native_policy_grad_norm=float(gp.norm()),native_actor_loss_grad_norm=float(ga.norm()),critic_grad_norm=float(gv.norm()),
        cosine_delta_negative_policy_grad=cosine(da,-gp),cosine_delta_negative_actor_loss_grad=cosine(da,-ga),
        native_policy_g_dot_delta=float(gp.double().dot(da.double())),actor_loss_g_dot_delta=float(ga.double().dot(da.double())),
        policy_gradient_pre_post_cosine=cosine(gp,postp),actor_loss_gradient_pre_post_cosine=cosine(ga,posta),
        losses_before=preloss,losses_after=postloss,native_returned_losses_divided_by_20=native_log,
        unused_critic_std_grad_none=alg.critic.distribution.log_std_param.grad is None)
    return alg,da,info,gp,ga


def target_gradients(actor,window,batch=None,indices=None):
    batch=window['batch'] if batch is None else batch
    aa=window['raw_targets'] if indices is None else window['raw_targets'][indices]
    cols=torch.cat((aa.reshape(len(aa),12),torch.ones(len(aa),1)),1)
    gg=component_gradients(actor,batch,cols);raw=gg[:12].reshape(4,3,-1)
    centered=(raw-window['raw_targets'].mean(0).unsqueeze(-1)*gg[-1])/window['normalizer']
    return {'raw':raw,'centered_native_std':centered}


def loss_changes(before,after,grads,delta):
    result={}
    for mode in ('raw','centered_native_std'):
        pre=torch.tensor(before[mode]['unclipped_loss'],dtype=torch.float64)
        post=torch.tensor(after[mode]['unclipped_loss'],dtype=torch.float64)
        predicted=torch.einsum('ekp,p->ek',grads[mode].double(),delta.double())
        actual=post-pre
        separate=torch.tensor(after[mode]['separate_clipped_loss'])-torch.tensor(before[mode]['separate_clipped_loss'])
        result[mode]=dict(actual_loss_change=actual.tolist(),first_order_loss_change=predicted.tolist(),
            nonlinear_remainder=(actual-predicted).tolist(),change_per_delta_norm=(actual/delta.double().norm()).tolist(),
            separate_clipped_loss_change=separate.tolist())
    return result


def episode_bootstrap(samples_before,samples_after,ids,draws=500):
    """Conditional on candidate, resample env episodes within each fixed task.

    Do not treat 24 transitions as independent. Across candidates/folds these
    intervals are dependent; only 3 collection seeds, no training-seed CI.
    """
    gen=torch.Generator().manual_seed(24680)
    delta=torch.cat(((samples_after['native']-samples_before['native'])[:,None],
        (samples_after['raw'][:,2]-samples_before['raw'][:,2])),1).reshape(24,len(ids),4).mean(0).double()
    counts=torch.zeros(draws,len(ids),dtype=torch.float64)
    tasks=ids.remainder(90)
    for t in tasks.unique().tolist():
        eligible=(tasks==t).nonzero().flatten();n=len(eligible)
        chosen=eligible[torch.randint(n,(draws,n),generator=gen)]
        counts.scatter_add_(1,chosen,torch.ones_like(chosen,dtype=torch.float64))
    means=counts@delta/len(ids)
    return dict(labels=['native_clipped','L_probe_finite','M_probe_finite','S_probe_finite'],
        ci95=torch.quantile(means,torch.tensor([.025,.975],dtype=torch.float64),dim=0).T.tolist(),
        scope='conditional candidate, task-stratified heldout env/episode bootstrap; 500 draws')


def run(args,out):
    torch.set_num_threads(args.threads)
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only replay must not initialize CUDA')
    repo=Path(__file__).resolve().parents[1];root=Path(args.root);cp_path=Path(args.checkpoint);cp_sha=sha256(cp_path)
    cp=torch.load(cp_path,map_location='cpu',weights_only=False);cp_memory_sha=tensor_hash(cp)
    config=json.loads((repo/'benchmarks/go2_piper_baselines_v1/agent.json').read_text())
    import rsl_rl.algorithms.ppo as ppo_module
    expected=json.loads((repo/'docs/go2_piper_gradient_audit/AUDIT_EVIDENCE.json').read_text())['runtime_sources']
    actual_ppo=Path(ppo_module.__file__)
    if sha256(actual_ppo)!=expected[str(actual_ppo)]:raise ValueError('Native PPO source changed since audit')
    raws={};mcs={};features={};inputs=[]
    for seed in SEEDS:
        folder=root/f'cp100_s{seed}';receipt=json.loads((folder/'collection.json').read_text())
        path=folder/'raw_rollout.pt'
        if receipt['status']!='PASS' or sha256(path)!=receipt['raw_sha256']:raise ValueError('Rollout receipt failed')
        rr=torch.load(path,map_location='cpu',weights_only=True);meta=rr['metadata']
        if meta['behavior_checkpoint_sha256']!=cp_sha or not meta['on_policy_stochastic'] or meta['control_structure']!='B0_18D':raise ValueError('Behavior provenance failed')
        torch.testing.assert_close(rr['component_rewards'].sum(-1),rr['total_reward'],rtol=1e-5,atol=1e-6)
        raws[seed]=rr;mcs[seed]=complete_mc(rr['component_rewards'],rr['episode_end'],rr['valid'])
        features[seed]=torch.cat((rr['critic_obs'],rr['age'].float().unsqueeze(-1)/400),-1)
        inputs.append(dict(seed=seed,path=str(path),sha256=receipt['raw_sha256']))
    metadata=dict(checkpoint=str(cp_path),checkpoint_sha256=cp_sha,inputs=inputs,device='cpu',formal_training=False,
        starts=list(UPDATE_STARTS),calibration_starts=list(range(0,384,24))+[376],excluded_incomplete_start=384,
        roles=[dict(update=SEEDS[i],fit=SEEDS[(i+1)%3],heldout=SEEDS[(i+2)%3]) for i in range(3)],
        estimators=list(ESTIMATORS),source_sha256=sha256(__file__),native_ppo_sha256=sha256(actual_ppo),
        original_optimizer_lr=cp['infos']['learning_rate'],single_step_candidates=0,
        scope='CP100 fresh diagnostic distribution, native algorithm replay; not historical minibatches or full 20-step update')
    (out/'SOURCE_USED.py').write_text(Path(__file__).read_text())
    (out/'METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[];cal=[];probes=[];crossseed={};start_time=time.monotonic()
    for fold,roles in enumerate(metadata['roles']):
        update,fit_seed,held=roles['update'],roles['fit'],roles['heldout']
        assert len(set(roles.values()))==3
        fit=ridge_fit(features[fit_seed][raws[fit_seed]['valid']],mcs[fit_seed][raws[fit_seed]['valid']])
        pred={s:ridge_predict(fit,features[s]) for s in (update,held)}
        for s in (update,held):
            probes.append(dict(fold=fold,fit_seed=fit_seed,evaluated_seed=s,metrics=value_metrics(pred[s][raws[s]['valid']],mcs[s][raws[s]['valid']])))
            for ws in metadata['calibration_starts']:
                *_,info=calibration(raws[s],pred[s],ws)
                cal.append(dict(fold=fold,seed=s,fit_seed=fit_seed,**info))
        for ws in UPDATE_STARTS:
            train=make_window(raws[update],pred[update],mcs[update],ws)
            hold=make_window(raws[held],pred[held],mcs[held],ws)
            base=restore(cp,config,train['batch'])
            hg=target_gradients(base.actor,hold);hgp,hga,_,_=native_gradients(base.actor,base.critic,hold['batch'])
            tg_full=target_gradients(base.actor,train);tgp,tga,_,_=native_gradients(base.actor,base.critic,train['batch'])
            crossseed[(ws,update)]=dict(native_policy=tgp,native_actor=tga,probe=tg_full['raw'][2])
            hb,hs_before=evaluate(base.actor,base.critic,hold['batch'],hold['raw_targets'],hold['centered_targets'])
            n=len(train['batch'].actions);mini=n//4
            gen=torch.Generator().manual_seed(9100+ws+fold)
            perm=torch.randperm(4*mini,generator=gen)
            # All paired alternatives use identical indices, A, old policy, moments and RNG.
            vectors=dict(heldout_native_policy=hgp,heldout_native_actor=hga,heldout_component=hg,
                update_window_native_policy=tgp,update_window_native_actor=tga,update_window_component=tg_full,
                minibatches=[],candidates=[])
            for part in range(4):
                ids=perm[part*mini:(part+1)*mini];batch=slice_batch(train['batch'],ids)
                tg=target_gradients(base.actor,train,batch,ids)
                paired=[]
                for scale in (1.,.1):
                    alg,delta,info,gp,ga=one_step(cp,config,batch,scale,31000+ws+fold*100+part)
                    after,hs_after=evaluate(alg.actor,alg.critic,hold['batch'],hold['raw_targets'],hold['centered_targets'])
                    changes=loss_changes(hb,after,hg,delta)
                    postgp,postga,_,_=native_gradients(alg.actor,alg.critic,hold['batch'])
                    postcomp=component_gradients(alg.actor,hold['batch'],hold['raw_targets'][:,2])
                    posttraincomp=component_gradients(alg.actor,batch,train['raw_targets'][ids,2])
                    native_actual=after['native_surrogate']-hb['native_surrogate']
                    native_pred=float(hgp.double().dot(delta.double()))
                    scopes=parameter_scopes(alg.actor)
                    scope_rows={key:dict(delta_norm=float(delta[ix].norm()),cos_delta_negative_actor_grad=cosine(delta[ix],-ga[ix]),
                        heldout_primary_g_dot_delta=(hg['raw'][2][:,ix].double()@delta[ix].double()).tolist()) for key,ix in scopes.items()}
                    row=dict(fold=fold,roles=roles,window_start=ws,partition=part,minibatch_n=mini,heldout_n=len(hold['batch'].actions),
                        train_window=train['info'],heldout_window=hold['info'],minibatch_indices_sha256=tensor_hash(ids),**info,
                        heldout_before=hb,heldout_after=after,heldout_changes=changes,
                        heldout_native_actual_loss_change=native_actual,heldout_native_first_order_loss_change=native_pred,
                        heldout_native_nonlinear_remainder=native_actual-native_pred,
                        heldout_native_change_per_delta_norm=native_actual/float(delta.norm()),
                        update_component_g_dot_delta={mode:torch.einsum('ekp,p->ek',gg.double(),delta.double()).tolist() for mode,gg in tg.items()},
                        heldout_policy_gradient_pre_post_cosine=cosine(hgp,postgp),heldout_actor_gradient_pre_post_cosine=cosine(hga,postga),
                        heldout_primary_gradient_pre_post_cosines=[cosine(hg['raw'][2,k],postcomp[k]) for k in range(3)],
                        update_primary_gradient_pre_post_cosines=[cosine(tg['raw'][2,k],posttraincomp[k]) for k in range(3)],
                        heldout_uncertainty=episode_bootstrap(hs_before,hs_after,hold['ids']),parameter_scopes=scope_rows)
                    rows.append(row);paired.append((row,delta))
                    vectors['candidates'].append(dict(partition=part,scale=scale,actor_delta=delta,
                        native_policy_gradient=gp,native_actor_gradient=ga,heldout_post_policy=postgp,heldout_post_actor=postga,
                        heldout_post_primary=postcomp,update_post_primary=posttraincomp))
                    metadata['single_step_candidates']+=1
                    del alg
                if paired[0][0]['initial_model_moments_sha256']!=paired[1][0]['initial_model_moments_sha256']:raise RuntimeError('Pair initialized differently')
                actual_scale=float(paired[1][1].norm()/paired[0][1].norm())
                for row,_ in paired:row.update(paired_delta_norm_ratio=actual_scale,paired_delta_cosine=cosine(paired[0][1],paired[1][1]),
                    paired_actual_lr_ratio=paired[1][0]['lr_actual']/paired[0][0]['lr_actual'])
                vectors['minibatches'].append(dict(partition=part,indices=ids,component_gradients=tg))
            torch.save(vectors,out/f'directions_fold{fold}_window{ws}.pt')
            (out/'results_partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(fold=fold,window=ws,candidates=len(rows),elapsed_s=time.monotonic()-start_time)),flush=True)
    consistency=[]
    for ws in UPDATE_STARTS:
        for i,s in enumerate(SEEDS):
            for t in SEEDS[i+1:]:
                a,b=crossseed[ws,s],crossseed[ws,t]
                consistency.append(dict(window=ws,seeds=[s,t],native_policy_cosine=cosine(a['native_policy'],b['native_policy']),
                    native_actor_cosine=cosine(a['native_actor'],b['native_actor']),
                    component_probe_cosines=[cosine(a['probe'][k],b['probe'][k]) for k in range(3)],
                    component_caveat='Different disjoint fit seeds; includes baseline-fit sensitivity'))
    if sha256(cp_path)!=cp_sha or tensor_hash(cp)!=cp_memory_sha:raise RuntimeError('Original checkpoint mutated')
    for inp in inputs:
        if sha256(inp['path'])!=inp['sha256']:raise RuntimeError('Original rollout mutated')
    metadata.update(checkpoint_and_raw_hashes_unchanged=True,cuda_initialized=torch.cuda.is_initialized(),elapsed_s=time.monotonic()-start_time,
        completed=True,artifact_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in out.glob('directions_*.pt')])
    for name,value in [('METADATA',metadata),('RESULTS',rows),('CALIBRATION',cal),('VALUE_PROBES',probes),('CROSS_SEED',consistency)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',candidates=len(rows),elapsed_s=metadata['elapsed_s'],cuda_initialized=metadata['cuda_initialized'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--threads',type=int,default=1);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    try:run(args,out)
    except BaseException:
        (out/'FAILURE.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
