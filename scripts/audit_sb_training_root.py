"""Inspect training-time loss/gradient allocation without a new training run.

Profiles training-cache examples and existing EMA checkpoints. The three-update
endpoint-only control checks the full optimizer/EMA path, not model quality.
"""
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from swarmbridge.bridges.audit import AuditDenoiser, VarianceClock
from swarmbridge.channel import aggregate, sample_params, transmit
from swarmbridge.data.real import CachedLatents
from swarmbridge.experiments.q2_progressive.run import EVAL_CFG, random_subset
from swarmbridge.experiments.q2_retx.run import to_blocks, to_cells
from swarmbridge.experiments.sb_audit.run import confusion, confusion_metrics, gpu_lock
from swarmbridge.teacher.model import IGNORE

OUT = ROOT / 'runs/sb_training_root_audit'
RUN = ROOT / 'runs/sb_audit_ve_v1'


def materialize(ds):
    idx = torch.randperm(len(ds.train['z']), generator=torch.Generator().manual_seed(447))[:32]
    clean = ds.train['z'][idx].cuda().float()
    fov = ds.train['fov48'][idx].cuda().float()
    labels = ds.train['y'][idx].cuda().long()
    parameters = sample_params(EVAL_CFG, len(clean), 1, 'cuda', torch.Generator(device='cuda').manual_seed(511), None)
    selected = random_subset(to_blocks(fov)>0, .15, torch.Generator(device='cuda').manual_seed(512))
    rx = transmit(clean[:, None], torch.zeros_like(clean[:, None]), (fov*to_cells(selected))[:, None],
                  parameters, EVAL_CFG, torch.Generator(device='cuda').manual_seed(513))
    source, reliability = aggregate(*rx)
    noise = torch.randn(clean.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(514))
    return clean, fov, labels, source, reliability, noise


@torch.no_grad()
def profile(model, ds, batch, times):
    clean, fov, labels, source, reliability, noise = batch
    rows = []
    for time in times:
        t = torch.full((len(clean),), time, device='cuda')
        state = model.clock.marginal(clean, source, t, noise)
        totals = dict(mse=0., ce_sum=0., valid=0, missing_error=0., missing_count=0,
                      input_missing_error=0., copied_input_missing_error=0.)
        matrix = torch.zeros(3, 3, dtype=torch.long, device='cuda')
        a, b, variance = model.clock.coefficients(t[:1])
        for offset in range(0, len(clean), 8):
            sl = slice(offset, offset+8)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, _ = model(state[sl], source[sl], reliability[sl], t[sl])
                logits = ds.dec(prediction.float()*ds.sd+ds.mu)
            error = (prediction.float()-clean[sl]).square().mean(1, keepdim=True)
            missing = (reliability[sl] == 0) & (fov[sl] > 0)
            count = int(missing.sum())
            totals['missing_error'] += error[missing].sum().item()
            totals['missing_count'] += count
            totals['mse'] += error.sum().item()/error.numel()
            totals['ce_sum'] += F.cross_entropy(logits.float(), labels[sl], ignore_index=IGNORE, reduction='sum').item()
            totals['valid'] += int((labels[sl] != IGNORE).sum())
            # Missing region y=0: a plain copying operation already sees part of clean.
            totals['copied_input_missing_error'] += (state[sl]-clean[sl]).square().mean(1,keepdim=True)[missing].sum().item()
            if time < 1:
                oracle_scale_copy = state[sl]/a.view(1,1,1,1)
                totals['input_missing_error'] += (oracle_scale_copy-clean[sl]).square().mean(1,keepdim=True)[missing].sum().item()
            matrix += confusion(logits.argmax(1), labels[sl])
        rows.append(dict(t=time, mse=totals['mse']/(len(clean)//8),
                         ce=totals['ce_sum']/totals['valid'], **confusion_metrics(matrix),
                         missing_mse=totals['missing_error']/totals['missing_count'],
                         untrained_copy_missing_mse=totals['copied_input_missing_error']/totals['missing_count'],
                         rescaled_copy_missing_mse=(totals['input_missing_error']/totals['missing_count'] if time<1 else None),
                         clean_coefficient=a.item(), bridge_variance=variance.item(),
                         clean_signal_to_bridge_noise=(a.square()/variance).item() if variance.item()>0 else None))
    return rows


def gradient(model, ds, batch, sl, endpoint, seed):
    clean, fov, labels, source, reliability, _ = batch
    clean, labels, source, reliability = clean[sl], labels[sl], source[sl], reliability[sl]
    generator = torch.Generator(device='cuda').manual_seed(seed)
    t = torch.ones(len(clean),device='cuda') if endpoint else 1e-4+(1-1e-4)*torch.rand(len(clean),device='cuda',generator=generator)
    noise = torch.randn(clean.shape,device='cuda',generator=generator)
    state = model.clock.marginal(clean,source,t,noise)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction,_ = model(state,source,reliability,t)
        mse = (prediction.float()-clean).square().mean()
        ce = F.cross_entropy(ds.dec(prediction.float()*ds.sd+ds.mu).float(),labels,ignore_index=IGNORE)
        loss = mse+ce
    values = torch.autograd.grad(loss,tuple(model.parameters()))
    vector = torch.cat([v.detach().float().flatten() for v in values])
    return vector, dict(mse=mse.item(),ce=ce.item(),loss=loss.item(),norm=vector.norm().item())


def gradients(model,ds,batch):
    rows=[]
    for index in range(2):
        endpoint,e=gradient(model,ds,batch,slice(16*index,16*(index+1)),True,610+index)
        interior,i=gradient(model,ds,batch,slice(16*index,16*(index+1)),False,610+index)
        cosine=F.cosine_similarity(endpoint,interior,dim=0).item()
        row=dict(batch=index,endpoint=e,interior=i,cosine=cosine,mixtures={})
        for fraction in (.1,.5,1.):
            combined=fraction*endpoint+(1-fraction)*interior
            row['mixtures'][str(fraction)]=dict(norm=combined.norm().item(),
                cosine_with_endpoint=F.cosine_similarity(combined,endpoint,dim=0).item(),
                endpoint_directional_dot=combined.dot(endpoint).item(),
                weighted_endpoint_to_interior_norm_ratio=(fraction*e['norm']/((1-fraction)*i['norm']) if fraction<1 else None))
        rows.append(row)
    return rows


def endpoint_control(ds,batch,deterministic=False):
    old_benchmark=torch.backends.cudnn.benchmark
    old_deterministic=torch.are_deterministic_algorithms_enabled()
    if deterministic:
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(938)
    mapping=AuditDenoiser(16,bridge=False,width=64).cuda().train()
    endpoint=AuditDenoiser(16,bridge=True,width=64,endpoint_fraction=1.).cuda().train()
    endpoint.load_state_dict(mapping.state_dict())
    twin=AuditDenoiser(16,bridge=False,width=64).cuda().train()
    twin.load_state_dict(mapping.state_dict())
    models=[mapping,endpoint,twin]
    optimizers=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01) for m in models]
    averages=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
    clean,_,labels,source,reliability,_=batch
    losses=[]
    first_predictions=[]
    first_gradients=[]
    for step in range(3):
        for j,(m,o) in enumerate(zip(models,optimizers)):
            sl=slice(8*(step%4),8*(step%4+1))
            o.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                mse=m.loss(source[sl],reliability[sl],clean[sl],generator=torch.Generator(device='cuda').manual_seed(800+step))
                logits=ds.dec(m.last_pred.float()*ds.sd+ds.mu).float()
                if deterministic:
                    # Same CE objective, deterministic reduction for this reference control.
                    ce=F.cross_entropy(logits,labels[sl],ignore_index=IGNORE,reduction='none').sum()/(labels[sl]!=IGNORE).sum()
                else:
                    ce=F.cross_entropy(logits,labels[sl],ignore_index=IGNORE)
                loss=mse+ce
            loss.backward()
            if step==0:
                first_predictions.append(m.last_pred.detach().float().clone())
                first_gradients.append(torch.cat([p.grad.detach().flatten().float() for p in m.parameters()]))
            torch.nn.utils.clip_grad_norm_(m.parameters(),10.,error_if_nonfinite=True)
            o.step()
            with torch.no_grad():
                for key,value in m.state_dict().items(): averages[j][key].lerp_(value,.001)
            losses.append(dict(step=step,model=j,loss=loss.item()))
    raw=max((a-b).abs().max().item() for a,b in zip(mapping.state_dict().values(),endpoint.state_dict().values()))
    ema=max((averages[0][k]-averages[1][k]).abs().max().item() for k in averages[0])
    twin_raw=max((a-b).abs().max().item() for a,b in zip(mapping.state_dict().values(),twin.state_dict().values()))
    twin_ema=max((averages[0][k]-averages[2][k]).abs().max().item() for k in averages[0])
    result=dict(updates=3,deterministic=deterministic,raw_max_difference=raw,ema_max_difference=ema,losses=losses,
                mapping_twin_raw_max_difference=twin_raw,mapping_twin_ema_max_difference=twin_ema,
                first_prediction_max_difference=[(p-first_predictions[0]).abs().max().item() for p in first_predictions],
                first_gradient_max_difference=[(p-first_gradients[0]).abs().max().item() for p in first_gradients])
    torch.backends.cudnn.benchmark=old_benchmark
    torch.use_deterministic_algorithms(old_deterministic)
    return result


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    OUT.mkdir(parents=True,exist_ok=True)
    with gpu_lock():
        ds=CachedLatents(str(ROOT/'runs/teacher'),'cuda',eval_split='val')
        batch=materialize(ds)
        result=dict(scope='Training-cache diagnostic; existing EMA checkpoints, seed 0, 32 fixed examples, 15% selected send budget. No quality-training run.',
                    allocation=dict(endpoint_sample_probability=.1,batch=32,expected_endpoint_samples_per_update=3.2,
                                    probability_of_any_endpoint_in_update=1-.9**32),profiles={},gradients={})
        result['endpoint_only_training_control']=endpoint_control(ds,batch)
        print('ENDPOINT_CONTROL',result['endpoint_only_training_control']['raw_max_difference'],flush=True)
        for step in (6000,15000,30000):
            model=AuditDenoiser(16,bridge=True,width=64,endpoint_fraction=.1).cuda().eval()
            model.load_state_dict(torch.load(RUN/f'sb_ve_xt_endpoint_s0/ema_{step}.pt',map_location='cuda',weights_only=True),strict=True)
            result['profiles'][str(step)]=profile(model,ds,batch,[0.,.25,.5,.75,.9,.95,.99,1.])
            result['gradients'][str(step)]=gradients(model,ds,batch)
            (OUT/'results.json').write_text(json.dumps(result,indent=2))
            print('PROFILED',step,flush=True)
            del model
            torch.cuda.empty_cache()
        mapping=AuditDenoiser(16,bridge=False,width=64).cuda().eval()
        mapping.load_state_dict(torch.load(RUN/'mapping_xt_s0/ema_30000.pt',map_location='cuda',weights_only=True),strict=True)
        result['mapping_30000_endpoint']=profile(mapping,ds,batch,[1.])[0]
        (OUT/'results.json').write_text(json.dumps(result,indent=2))
        print('COMPLETED',flush=True)


if __name__=='__main__':
    main()
