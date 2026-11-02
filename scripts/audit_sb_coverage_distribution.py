"""CPU Monte Carlo of train vs 15%-send received spatial coverage."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent.parent
torch.set_num_threads(4)
f=torch.load(ROOT/'runs/teacher/latents_train.pt',map_location='cpu',weights_only=False)['fov48'].float()
w=F.avg_pool2d(f,4)[:,0].flatten(1)
g=torch.Generator().manual_seed(723)
n=20000
weights=w[torch.randint(len(w),(n,),generator=g)]
valid=weights>0
denom=weights.sum(1)
p_train=.9*torch.rand(n,generator=g)
keep_train=torch.rand(weights.shape,generator=g)>=p_train[:,None]
train=(keep_train*weights).sum(1)/denom
p_eval=.6*torch.rand(n,generator=g)
score=torch.rand(weights.shape,generator=g).masked_fill(~valid,-1)
order=score.argsort(1,descending=True)
k=(.15*valid.sum(1)).round().long()
selected=torch.zeros_like(valid).scatter_(1,order,torch.arange(weights.shape[1])[None]<k[:,None])
keep_eval=torch.rand(weights.shape,generator=g)>=p_eval[:,None]
eval15=(selected*keep_eval*weights).sum(1)/denom
result=dict(scope='CPU Monte Carlo on training footprint geometry; no model training or test evaluation',
            n=n,train_erasure_probability=[0,.9],eval15_selected_fraction=.15,
            eval_physical_erasure_probability=[0,.6],train_mean_received=train.mean().item(),
            eval15_mean_received=eval15.mean().item(),thresholds={})
for threshold in (.10,.15,.20):
    a=(train<=threshold).float().mean().item()
    b=(eval15<=threshold).float().mean().item()
    result['thresholds'][str(threshold)]=dict(train_fraction=a,eval15_fraction=b,
        train_joint_exact_endpoint10=.1*a,train_joint_exact_endpoint50=.5*a)
path=ROOT/'runs/sb_training_root_audit/coverage_distribution.json'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
