"""Inference-only adapter for Sony's official SBCTM checkpoint.

Use upstream native PyTorch FIR/activation implementations on GPU instead of
compiling CUDA extensions on Windows. Training-only imports are blocked; no
training loss or quality metric is fabricated. The model/sampler are upstream.
"""
import importlib
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def unavailable(*args, **kwargs):
    raise RuntimeError('This optional training/metric dependency is unavailable in the inference adapter')


def load_sony(checkpoint):
    import pytorch_lightning  # resolve optional metric availability before inference-only stubs
    sys.path.insert(0, str(ROOT / 'pdf/GithubModels/sony__sbctm'))
    sys.modules['wandb'] = types.ModuleType('wandb')
    if importlib.util.find_spec('pesq') is None:
        metric = types.ModuleType('pesq')
        metric.__spec__ = importlib.machinery.ModuleSpec('pesq', loader=None)
        metric.pesq = unavailable
        sys.modules['pesq'] = metric
    losses = types.ModuleType('asteroid.losses')
    losses.singlesrc_neg_sisdr = unavailable
    sys.modules['asteroid'] = types.ModuleType('asteroid')
    sys.modules['asteroid.losses'] = losses
    import torch.utils.cpp_extension
    original_load = torch.utils.cpp_extension.load
    def native_only(name, *args, **kwargs):
        if name not in ('fused', 'upfirdn2d'):
            raise RuntimeError('Unexpected CUDA extension: ' + name)
        return None
    torch.utils.cpp_extension.load = native_only
    try:
        from sgmse.model import CTModel
    finally:
        torch.utils.cpp_extension.load = original_load
    fir = importlib.import_module('sgmse.backbones.ncsnpp_utils.op.upfirdn2d')
    activation = importlib.import_module('sgmse.backbones.ncsnpp_utils.op.fused_act')
    # Keep function objects intact because other upstream modules imported aliases.
    scope = {}
    exec('def pure(input,kernel,up=1,down=1,pad=(0,0)):\n'
         ' return upfirdn2d_native(input,kernel,up,up,down,down,pad[0],pad[1],pad[0],pad[1])', scope)
    fir.upfirdn2d.__code__ = scope['pure'].__code__
    exec('def pure(input,bias,negative_slope=0.2,scale=2**0.5):\n'
         ' shape=[1]*(input.ndim-bias.ndim-1)\n'
         ' return F.leaky_relu(input+bias.view(1,bias.shape[0],*shape),negative_slope=negative_slope)*scale', scope)
    activation.fused_leaky_relu.__code__ = scope['pure'].__code__
    # PyTorch 2.5 loads the trusted author checkpoint, including original class metadata.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hyperparameters = saved['hyper_parameters'].copy()
    print('SONY_HYPERPARAMETERS', {k: str(v) for k, v in hyperparameters.items()}, flush=True)
    model = CTModel(**hyperparameters)
    # Published inference creates CTModel without its distillation teacher and uses
    # strict=False. Here exclude only that unused training teacher, then require
    # strict loading of every student/target/loss tensor used by this instance.
    state = saved['state_dict']
    excluded = [key for key in state if key.startswith('teacher.')]
    state = {key: value for key, value in state.items() if not key.startswith('teacher.')}
    loaded = model.load_state_dict(state, strict=True)
    print('SONY_UNUSED_TRAINING_TEACHER_TENSORS', len(excluded), flush=True)
    del saved
    model = model.eval().cuda()
    print('SONY_STRICT_LOAD', loaded, flush=True)
    from sgmse.util.other import pad_spec
    def enhance(waveform, mode, steps):
        signal = waveform.reshape(1, -1)
        factor = signal.abs().max()
        spectrum = model._forward_transform(model._stft(signal / factor)).unsqueeze(0)
        spectrum = pad_spec(spectrum, mode='reflection')
        sample = model.get_ctm_sample(spectrum, steps, model.dnn_tgt)
        return model.to_audio(sample.squeeze(), signal.shape[-1]) * factor
    return model, enhance
