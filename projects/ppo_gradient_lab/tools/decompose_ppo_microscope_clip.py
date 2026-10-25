"""Read-only postprocessing of saved deltas; no optimizer steps or simulation."""
import argparse,json
from pathlib import Path
import torch
from ppo_update_microscope import restore,make_window,normal,vector,sha256


def run(out):
    torch.set_num_threads(1)
    meta=json.loads((out/'METADATA.json').read_text())
    if not meta.get('completed'):raise ValueError('Require completed microscope')
    cp=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False)
    config=json.loads((Path(__file__).resolve().parents[1]/'benchmarks/go2_piper_baselines_v1/agent.json').read_text())
    raw={x['seed']:torch.load(x['path'],map_location='cpu',weights_only=True) for x in meta['inputs']}
    results=json.loads((out/'RESULTS.json').read_text());lookup={(r['fold'],r['window_start'],r['partition'],r['lr_scale']):r for r in results}
    rows=[]
    for fold,roles in enumerate(meta['roles']):
        rr=raw[roles['heldout']]
        for ws in meta['starts']:
            # Only native A is used. Zero placeholders do not define component targets here.
            zeros=torch.zeros_like(rr['component_rewards']);w=make_window(rr,zeros,zeros,ws);b=w['batch']
            alg=restore(cp,config,b);before=vector(alg.actor)
            saved=torch.load(out/f'directions_fold{fold}_window{ws}.pt',map_location='cpu',weights_only=True)
            def evaluate():
                with torch.no_grad():
                    d=normal(alg.actor,b.observations);ratio=(d.log_prob(b.actions).sum(-1)-b.old_actions_log_prob.squeeze(-1)).exp().double()
                    a=b.advantages.squeeze(-1).double()
                    unclip=-ratio*a;clip=torch.maximum(unclip,-ratio.clamp(.8,1.2)*a)
                    kl=torch.distributions.kl_divergence(torch.distributions.Normal(*b.old_distribution_params),d).sum(-1).mean()
                return unclip,clip,float(kl)
            u0,c0,k0=evaluate()
            for candidate in saved['candidates']:
                delta=candidate['actor_delta'];target=before+delta
                offset=0
                with torch.no_grad():
                    for p in alg.actor.parameters():
                        p.copy_(target[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
                if not torch.equal(vector(alg.actor)-before,delta):raise RuntimeError('Delta reconstruction is not exact')
                u1,c1,k1=evaluate();record=lookup[fold,ws,candidate['partition'],candidate['scale']]
                if abs(k1-record['heldout_after']['kl_mean'])>1e-12:raise RuntimeError('Reconstructed KL mismatch')
                if abs(float(c1.mean())-record['heldout_after']['native_surrogate'])>1e-12:raise RuntimeError('Reconstructed loss mismatch')
                actual_unclip=float((u1-u0).mean());actual_clip=float((c1-c0).mean());penalty=float(((c1-u1)-(c0-u0)).mean())
                predicted=record['heldout_native_first_order_loss_change']
                if abs(actual_clip-actual_unclip-penalty)>1e-12:raise RuntimeError('Penalty decomposition failed')
                rows.append(dict(fold=fold,window=ws,partition=candidate['partition'],scale=candidate['scale'],
                    unclipped_loss_change=actual_unclip,clipped_loss_change=actual_clip,clipping_penalty_change=penalty,
                    first_order_loss_change=predicted,unclipped_nonlinear_remainder=actual_unclip-predicted,
                    candidate_reconstruction_exact=True,old_policy_ratio_error_note='CPU/GPU logp rounding retained'))
    if sha256(meta['checkpoint'])!=meta['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    result=dict(rows=rows,optimizer_steps=0,cuda_initialized=torch.cuda.is_initialized(),source_sha256=sha256(__file__))
    path=out/'CLIP_DECOMPOSITION.json'
    with path.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',reconstructed_candidates=len(rows),optimizer_steps=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out');a=p.parse_args();run(Path(a.out))
