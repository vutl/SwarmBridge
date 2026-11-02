"""Execute official Rethinking backbones, a bridge sampler and backward passes.

Random initialization is explicitly recorded. This is a code smoke test, not
an enhancement-quality experiment or a reproduction of published metrics.
"""
import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'pdf/GithubModels/Dahan-Wang__Rethinking-Flow-and-Diffusion-Bridge-Models-for-Speech-Enhancement'
sys.path.insert(0, str(SOURCE))


def main():
    torch.set_num_threads(4)
    # PESQ is imported only by unused training/evaluation helpers in this test.
    if importlib.util.find_spec('pesq') is None:
        optional = types.ModuleType('pesq')
        optional.__spec__ = importlib.machinery.ModuleSpec('pesq', loader=None)
        def unsupported(*args, **kwargs):
            raise RuntimeError('PESQ is not measured by this code smoke test')
        optional.pesq = unsupported
        sys.modules['pesq'] = optional
    cpp_extension = importlib.import_module('torch.utils.cpp_extension')
    original = cpp_extension.load
    cpp_extension.load = lambda *args, **kwargs: None
    try:
        from fdbm.backbones import BackboneRegistry
    finally:
        cpp_extension.load = original
    fir = importlib.import_module('fdbm.backbones.ncsnpp_utils.op.upfirdn2d')
    scope = {}
    exec('def pure(input,kernel,up=1,down=1,pad=(0,0)):\n'
         ' return upfirdn2d_native(input,kernel,up,up,down,down,pad[0],pad[1],pad[0],pad[1])', scope)
    fir.upfirdn2d.__code__ = scope['pure'].__code__
    from fdbm.bridge import Bridge
    torch.manual_seed(20261031)
    clean = torch.complex(torch.randn(1, 1, 257, 16), torch.randn(1, 1, 257, 16)).cuda() * .1
    noisy = clean + .1 * torch.randn_like(clean)
    t = torch.tensor([.5], device='cuda')
    bridge = Bridge(path='sb', noise_schedule='bb', N=5, sampler_type='sde_ei')
    mean, std = bridge.probability_path(clean, noisy, t)
    state = mean + std[:, None, None, None] * torch.randn_like(clean)
    records = []
    for name in ('tfgridnet_5l32c100', 'tfgridnet_5l32c100_predictive'):
        model = BackboneRegistry.get_by_name(name)().cuda()
        model.train()
        output = model(state, noisy, t) if 'predictive' not in name else model(noisy)
        loss = (output - clean).abs().square().mean()
        loss.backward()
        gradient = sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None) ** .5
        row = dict(backbone=name, parameters=sum(p.numel() for p in model.parameters()),
                   output_shape=list(output.shape), loss=float(loss.detach()), gradient_norm=gradient)
        if 'predictive' not in name:
            model.eval()
            sampled = bridge.sampler(lambda x, y, times: model(x, y, times), noisy)
            row['sampler_finite'] = bool(torch.isfinite(sampled).all())
        records.append(row)
        print(json.dumps(row), flush=True)
        del model
    result = dict(status='code_smoke_only', pretrained=False, note='Random initialization, synthetic complex inputs; no quality claim.',
                  results=records)
    (ROOT / 'runs/sb_paper_reproductions/rethinking_code_probe.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
