"""CPU analysis of complete frozen-policy episodes; ridge is diagnostic only.

Accumulate exact episode score-gradient sums; bootstrap task-balanced replicate
blocks via Gram matrices, without repeatedly running backprop. No optimizer.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import torch
from diagnose_go2_piper_gradients import FrozenMLP,complete_mc,component_gae,sha256,pair_stats

ESTIMATORS=('mc','crossfit_lambda1','crossfit_lambda095')


def ridge_fit(x,y,ridge=.001):
    x=x.double();y=y.double()
    mean=x.mean(0);std=x.std(0).clamp_min(1e-6)
    z=torch.cat(((x-mean)/std,torch.ones(len(x),1,dtype=torch.float64)),1)
    penalty=torch.eye(z.shape[1],dtype=z.dtype)*ridge;penalty[-1,-1]=0
    coef=torch.linalg.solve(z.T@z/len(z)+penalty,z.T@y/len(z))
    return dict(mean=mean,std=std,coef=coef,ridge=ridge)


def ridge_predict(fit,x):
    z=(x.double()-fit['mean'])/fit['std']
    return (z@fit['coef'][:-1]+fit['coef'][-1]).float()


def value_metrics(pred,target):
    e=(pred-target).double();var=target.double().var(0,unbiased=False)
    return dict(mse=e.square().mean(0).tolist(),target_variance=var.tolist(),
        explained_variance=[float(1-e[:,i].var(unbiased=False)/v) if v>1e-12 else None for i,v in enumerate(var)])


def score_sums(actor,obs,actions,old_lp,advantages):
    """One complete episode: column-wise -sum(A * ratio) gradients."""
    lp=actor.normal(obs).log_prob(actions).sum(-1)
    ratio=(lp-old_lp).exp()
    pars=list(actor.parameters());gs=[]
    for i in range(advantages.shape[1]):
        parts=torch.autograd.grad(ratio,pars,grad_outputs=-advantages[:,i],retain_graph=i+1<advantages.shape[1])
        gs.append(torch.cat([g.detach().flatten() for g in parts]))
    return torch.stack(gs),float((lp.detach()-old_lp).abs().max())


def parameter_scopes(model):
    sections={};offset=0
    for name,p in model.named_parameters():
        sections[name]=torch.arange(offset,offset+p.numel());offset+=p.numel()
    result={'all':torch.arange(offset),'trunk':torch.cat([v for k,v in sections.items() if k.startswith(('mlp.0.','mlp.2.','mlp.4.'))])}
    for layer in ('mlp.0','mlp.2','mlp.4','mlp.6','distribution'):
        result[layer]=torch.cat([v for k,v in sections.items() if k.startswith(layer+'.')])
    for label,start,end in [('leg_rows',0,12),('arm_rows',12,18)]:
        result[label]=torch.cat((sections['mlp.6.weight'].reshape(18,-1)[start:end].flatten(),sections['mlp.6.bias'][start:end]))
    return result


def coefficients(counts,n,advsum,advcross,mode,reference=None):
    """Coefficients on [block, objective0/1/2/constant-score] gradient sums."""
    counts=counts.double();den=counts@n.double()
    mean=(counts@advsum.double())/den[:,None]
    if reference is not None:mean=reference[0].double().expand(len(counts),-1)
    c=torch.zeros(len(counts),3,len(n),4,dtype=torch.float64)
    for i in range(3):
        c[:,i,:,i]=counts/den[:,None]
        if mode!='raw':c[:,i,:,3]=-mean[:,i,None]*counts/den[:,None]
    if mode!='raw':
        second=torch.einsum('rb,bij->rij',counts,advcross.double())
        cov=(second-den[:,None,None]*mean[:,:,None]*mean[:,None,:])/(den-1)[:,None,None]
        if reference is not None:cov=reference[1].double().expand(len(counts),-1,-1)
        scale=(cov.sum((1,2)).clamp_min(0).sqrt()[:,None].expand(-1,3) if mode=='shared_total_std'
               else cov.diagonal(dim1=1,dim2=2).clamp_min(0).sqrt())+1e-8
        c=c/scale[:,:,None,None]
    return c.flatten(2)


def pooled_moments(n,sums,cross):
    total=n.sum();mean=sums.sum(0)/total
    cov=(cross.sum(0)-total*mean[:,None]*mean[None,:])/(total-1)
    return mean,cov


def gram_statistics(gram,c):
    dot=torch.einsum('rip,pq,rjq->rij',c,gram.double(),c)
    norms=dot.diagonal(dim1=1,dim2=2).clamp_min(0).sqrt()
    cos=dot/(norms[:,:,None]*norms[:,None,:]).clamp_min(1e-24)
    return dot,norms,cos


def bootstrap_counts(seeds,reps,draws=1000,seed=1907):
    gen=torch.Generator().manual_seed(seed);unique=sorted(set(seeds));result=[]
    for _ in range(draws):
        counts=torch.zeros(len(seeds),dtype=torch.float64)
        for s in torch.randint(len(unique),(len(unique),),generator=gen).tolist():
            ids=[i for i,x in enumerate(seeds) if x==unique[s]]
            for j in torch.randint(len(ids),(len(ids),),generator=gen).tolist():counts[ids[j]]+=1
        result.append(counts)
    return torch.stack(result)


def summarize_basis(gs,n,sums,cross,seeds,reps,scopes,draws=1000):
    """gs [B,10,P]; 3 objectives per estimator plus common constant score."""
    counts=torch.cat((torch.ones(1,len(n),dtype=torch.float64),bootstrap_counts(seeds,reps,draws)))
    output=[]
    for e,ename in enumerate(ESTIMATORS):
        cols=[3*e,3*e+1,3*e+2,9]
        reference=pooled_moments(n,sums[:,e],cross[:,e])
        for scope,indices in scopes.items():
            g=gs[:,cols][:,:,indices].double().reshape(len(n)*4,-1)
            gram=g@g.T
            for mode in ('raw','shared_total_std','per_objective_std'):
                c=coefficients(counts,n,sums[:,e],cross[:,e],mode)
                dot,norm,cos=gram_statistics(gram,c)
                for i,j in itertools.combinations(range(3),2):
                    ci=torch.quantile(cos[1:,i,j],torch.tensor([.025,.975],dtype=torch.float64)).tolist()
                    # Conflict across disjoint complete-episode blocks, not timesteps.
                    cb=coefficients(torch.eye(len(n),dtype=torch.float64),n,sums[:,e],cross[:,e],mode,reference)
                    db,nb,xb=gram_statistics(gram,cb)
                    split=[]
                    for parity in (0,1):split.append(torch.tensor([int(r%2==parity) for r in reps],dtype=torch.float64))
                    sc=coefficients(torch.stack(split),n,sums[:,e],cross[:,e],mode,reference)
                    consistency=[]
                    for k in range(3):
                        v=sc[:,k];num=(v[0]@gram@v[1]);den=((v[0]@gram@v[0])*(v[1]@gram@v[1])).clamp_min(1e-24).sqrt()
                        consistency.append(float(num/den))
                    seed_counts=torch.tensor([[int(x==seed) for x in seeds] for seed in sorted(set(seeds))],dtype=torch.float64)
                    seed_c=coefficients(seed_counts,n,sums[:,e],cross[:,e],mode,reference)
                    _,_,seed_cos=gram_statistics(gram,seed_c)
                    cross_half_dot=float((sc[0,i]@gram@sc[1,j]+sc[1,i]@gram@sc[0,j])/2)
                    output.append(dict(estimator=ename,scope=scope,normalization=mode,pair=[i,j],
                        cosine=float(cos[0,i,j]),cosine_ci95=ci,dot=float(dot[0,i,j]),
                        norms=norm[0].tolist(),norm_ratio=float(norm[0,i]/norm[0,j].clamp_min(1e-12)),
                        dot_i_total=float(dot[0,i].sum()),dot_j_total=float(dot[0,j].sum()),
                        block_conflict_rate=float((xb[:,i,j]<-1e-8).double().mean()),blocks=len(n),
                        same_objective_split_half_cosines=consistency,
                        cross_half_pair_dot=cross_half_dot,per_seed_cosines=seed_cos[:,i,j].tolist(),
                        bootstrap_negative_fraction=float((cos[1:,i,j]<0).double().mean())))
    return output


def summarize_tasks(task_g,task_n,task_sums,task_cross,scopes,task_metadata):
    rows=[]
    # Task/level estimates are descriptive; only primary trunk pairs, no 90-way significance search.
    subsets=[('task='+str(i),[i]) for i in range(90)]
    for key in ('base_level','ee_level','category'):
        for value in sorted({x[key] for x in task_metadata}):
            subsets.append((key+'='+str(value),[i for i,x in enumerate(task_metadata) if x[key]==value]))
    references=[pooled_moments(task_n,task_sums[:,e],task_cross[:,e]) for e in range(3)]
    for label,ids in subsets:
        n=task_n[ids].sum().reshape(1);sg=task_g[ids].sum(0,keepdim=True)
        for e,ename in enumerate(ESTIMATORS):
            g=sg[:,[e*3,e*3+1,e*3+2,9]][:,:,scopes['trunk']].double().reshape(4,-1)
            for mode in ('raw','shared_total_std'):
                c=coefficients(torch.ones(1,1),n,task_sums[ids,e].sum(0,keepdim=True),task_cross[ids,e].sum(0,keepdim=True),mode,references[e])
                d,nn,cs=gram_statistics(g@g.T,c)
                rows.append(dict(stratum=label,estimator=ename,normalization=mode,scope='trunk',n=int(n),
                    cosine_loco_manip=float(cs[0,0,1]),norms=nn[0].tolist(),dot_loco_manip=float(d[0,0,1])))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--out',required=True);p.add_argument('--checkpoints',nargs='+',type=int,default=[100,250]);p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads)
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    repo=Path(__file__).resolve().parents[1]
    taskmeta=json.loads((repo/'benchmarks/go2_piper_wbc_v1/task_pool.json').read_text())['train']['tasks']
    for cp_step in a.checkpoints:
        cp_path=Path(a.checkpoint_root)/f'cp_{cp_step}.pt';cp_sha=sha256(cp_path)
        cp=torch.load(cp_path,map_location='cpu',weights_only=False);actor=FrozenMLP(cp['actor_state_dict'])
        sc=parameter_scopes(actor);P=sum(x.numel() for x in actor.parameters())
        raw=[];features=[];returns=[];provenance=[]
        for seed in (8201,8202,8203):
            folder=root/f'cp{cp_step}_s{seed}';receipt=json.loads((folder/'collection.json').read_text())
            assert receipt['status']=='PASS' and receipt['metadata']['behavior_checkpoint_sha256']==cp_sha
            rr=torch.load(folder/'raw_rollout.pt',map_location='cpu',weights_only=True)
            assert sha256(folder/'raw_rollout.pt')==receipt['raw_sha256']
            raw.append(rr);provenance.append(dict(rollout_id=rr['metadata']['rollout_id'],raw_sha256=receipt['raw_sha256']))
            features.append(torch.cat((rr['critic_obs'],rr['age'].float().unsqueeze(-1)/400),-1))
            returns.append(complete_mc(rr['component_rewards'],rr['episode_end'],rr['valid']))
        block_g=[];block_n=[];block_s=[];block_cross=[];block_seeds=[];block_reps=[];vmetrics=[]
        task_g=torch.zeros(90,10,P,dtype=torch.float32);task_n=torch.zeros(90,dtype=torch.float64)
        task_sums=torch.zeros(90,3,3,dtype=torch.float64);task_cross=torch.zeros(90,3,3,3,dtype=torch.float64)
        max_lp=0.;same_theta_clip=0;max_mu=0.;max_std=0.;max_kl=0.;max_ratio_delta=0.
        for f,rr in enumerate(raw):
            train=[i for i in range(3) if i!=f]
            fit=ridge_fit(torch.cat([features[i][raw[i]['valid']] for i in train]),torch.cat([returns[i][raw[i]['valid']] for i in train]))
            vv=ridge_predict(fit,features[f]);vv=vv*rr['valid'].unsqueeze(-1)
            nxt=torch.cat((vv[1:],torch.zeros_like(vv[:1])))
            # Finite-horizon: every first-episode endpoint, including timeout, is terminal.
            r=rr['component_rewards']*rr['valid'].unsqueeze(-1)
            a1=component_gae(r,vv,nxt,rr['episode_end'],rr['episode_end'],lam=1.)
            a95=component_gae(r,vv,nxt,rr['episode_end'],rr['episode_end'],lam=.95)
            torch.testing.assert_close(a1[rr['valid']],(returns[f]-vv)[rr['valid']],rtol=2e-4,atol=5e-5)
            val=value_metrics(vv[rr['valid']],returns[f][rr['valid']]);val.update(rollout_id=rr['metadata']['rollout_id'],
                fitted_on=[raw[i]['metadata']['rollout_id'] for i in train],ridge=.001,features='native critic 102 + episode age',
                native_total_mc_metrics=value_metrics(rr['native_value'][rr['valid']].unsqueeze(-1),returns[f][rr['valid']].sum(-1,keepdim=True)))
            vmetrics.append(val)
            adv=torch.stack((returns[f],a1,a95),2)
            for rep in range(4):
                bg=torch.zeros(10,P,dtype=torch.float64);bn=0;bs=torch.zeros(3,3,dtype=torch.float64);bc=torch.zeros(3,3,3,dtype=torch.float64)
                for task in range(90):
                    env_id=rep*90+task;mask=rr['valid'][:,env_id];n=int(mask.sum())
                    aa=adv[mask,env_id].reshape(n,9);aa=torch.cat((aa,torch.ones(n,1)),1)
                    obs=rr['actor_obs'][mask,env_id];acts=rr['actions'][mask,env_id];old=rr['old_logp'][mask,env_id]
                    with torch.no_grad():
                        dd=actor.normal(obs);max_mu=max(max_mu,float((dd.mean-rr['old_mu'][mask,env_id]).abs().max()))
                        max_std=max(max_std,float((dd.stddev-rr['old_std'][mask,env_id]).abs().max()))
                        ratio=(dd.log_prob(acts).sum(-1)-old).exp();same_theta_clip+=int(((ratio-1).abs()>.2).sum())
                        max_ratio_delta=max(max_ratio_delta,float((ratio-1).abs().max()))
                        kl=torch.distributions.kl_divergence(torch.distributions.Normal(rr['old_mu'][mask,env_id],rr['old_std'][mask,env_id]),dd).sum(-1)
                        max_kl=max(max_kl,float(kl.max()))
                    gg,lp=score_sums(actor,obs,acts,old,aa);max_lp=max(max_lp,lp)
                    if max_lp>.002 or max_mu>.0001 or max_std>1e-5:raise RuntimeError('Behavior checkpoint mismatch')
                    bg+=gg.double();bn+=n;task_g[task]+=gg;task_n[task]+=n
                    av=adv[mask,env_id].double();ss=av.sum(0);cc=torch.einsum('nek,nel->ekl',av,av)
                    bs+=ss;bc+=cc;task_sums[task]+=ss;task_cross[task]+=cc
                block_g.append(bg);block_n.append(bn);block_s.append(bs);block_cross.append(bc);block_seeds.append(rr['metadata']['seed']);block_reps.append(rep)
                print(json.dumps(dict(cp=cp_step,seed=rr['metadata']['seed'],replicate=rep,transitions=bn,elapsed_s=time.monotonic()-start)),flush=True)
        gs=torch.stack(block_g);ns=torch.tensor(block_n,dtype=torch.float64);sums=torch.stack(block_s);cr=torch.stack(block_cross)
        basis=dict(gradient_sums=gs,n=ns,advantage_sums=sums,advantage_cross=cr,seeds=block_seeds,replicates=block_reps,
            provenance=provenance,scopes=sc,checkpoint_sha256=cp_sha)
        torch.save(basis,out/f'cp{cp_step}_gradient_basis.pt')
        rows=summarize_basis(gs,ns,sums,cr,block_seeds,block_reps,sc)
        strata=summarize_tasks(task_g,task_n,task_sums,task_cross,sc,taskmeta)
        result=dict(checkpoint=cp_step,checkpoint_sha256=cp_sha,provenance=provenance,optimizer_updates=0,device='cpu',
            episodes=1080,valid_transitions=int(ns.sum()),behavior_validation=dict(max_logp_error=max_lp,max_mu_error=max_mu,
                max_std_error=max_std,clip_count=same_theta_clip,max_kl=max_kl,max_ratio_delta=max_ratio_delta),
            primary_parameter_scope='trunk',normalization_scope='whole_checkpoint_collection; fixed for blocks, halves, seeds and strata; refitted within bootstrap draws',
            estimand='uniform valid transitions from equally allocated first episodes; finite-horizon timeout terminal',
            bootstrap='1000 hierarchical seed then task-balanced replicate-block resamples; conditional ridge fits, only 3 seeds; exploratory CI',
            statistics=rows,strata=strata,value_probes=vmetrics,elapsed_s=time.monotonic()-start)
        assert sha256(cp_path)==cp_sha
        (out/f'cp{cp_step}_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(cp=cp_step,status='ANALYSIS_COMPLETE',optimizer_updates=0,elapsed_s=time.monotonic()-start)),flush=True)

if __name__=='__main__':main()
