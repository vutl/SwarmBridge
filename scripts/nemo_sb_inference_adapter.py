"""Run the official NeMo numerical modules without its ASR training framework.

Only framework type metadata/decorators are bypassed. Estimator, transforms,
noise schedule and SDE/PF-ODE sampler are executed from unchanged NeMo 2.0.0
source. Strict state loading verifies all checkpoint tensors. This is an
inference adapter, not a claim of a native NeMo training reproduction.
"""
import ast
import importlib.util
import logging
import shutil
import sys
import tarfile
import types
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'pdf/GithubModels/NVIDIA__NeMo_audio_2.0.0'
INSTALLED = ROOT / '.venv-sb-papers/Lib/site-packages/nemo'


def module(name):
    value = types.ModuleType(name)
    sys.modules[name] = value
    return value


def load(name, relative):
    source = SOURCE / relative
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(INSTALLED / relative, source)
    spec = importlib.util.spec_from_file_location(name, source)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def official_modules():
    # Bypass eager import of ASR training, datasets, tokenizers and distributed runtimes.
    core = module('nemo.core.classes')
    core.NeuralModule = torch.nn.Module
    core.typecheck = lambda *args, **kwargs: (lambda function: function)
    neural = module('nemo.core.neural_types')
    class Metadata:
        def __init__(self, *args, **kwargs):
            pass
    for name in ('AudioSignal', 'LengthsType', 'NeuralType', 'SpectrogramType',
                 'FloatType', 'VoidType'):
        setattr(neural, name, Metadata)
    module('nemo.utils').logging = logging.getLogger('official_nemo_sb')
    load('nemo.collections.common.parts.utils', 'collections/common/parts/utils.py')
    # Execute the original sequence-mask helper, avoiding its ASR-heavy parent module.
    path = 'collections/asr/parts/preprocessing/features.py'
    target = SOURCE / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(INSTALLED / path, target)
    tree = ast.parse(target.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_seq_mask_like')
    scope = module('nemo.collections.asr.parts.preprocessing.features')
    scope.torch = torch
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(target), 'exec'), scope.__dict__)
    transforms = load('official_nemo_transforms', 'collections/audio/modules/transforms.py')
    network = load('official_nemo_ncsnpp', 'collections/audio/parts/submodules/ncsnpp.py')
    bridge = load('official_nemo_bridge', 'collections/audio/parts/submodules/schroedinger_bridge.py')
    return transforms, network, bridge


class OfficialNemoSB(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        transforms, network, bridge = official_modules()
        with tarfile.open(checkpoint) as archive:
            config = yaml.safe_load(archive.extractfile('./model_config.yaml'))
            weights = torch.load(archive.extractfile('./model_weights.ckpt'), map_location='cpu', weights_only=False)
        def parameters(name):
            return {k: v for k, v in config[name].items() if k != '_target_'}
        self.encoder = transforms.AudioToSpectrogram(**parameters('encoder'))
        self.decoder = transforms.SpectrogramToAudio(**parameters('decoder'))
        self.estimator = network.SpectrogramNoiseConditionalScoreNetworkPlusPlus(**parameters('estimator'))
        self.noise_schedule = bridge.SBNoiseScheduleVE(**parameters('noise_schedule'))
        self.sampler = bridge.SBSampler(noise_schedule=self.noise_schedule, estimator=self.estimator,
                                       estimator_output=config['estimator_output'], **parameters('sampler'))
        self.load_state_dict(weights, strict=True)
        self.config = config
        self.sample_rate = config['sample_rate']

    @torch.inference_mode()
    def enhance(self, waveform, process='ode', steps=10):
        self.sampler.process = process
        self.sampler.num_steps = steps
        scale = waveform.abs().amax((-1, -2), keepdim=True)
        encoded, lengths = self.encoder(input=waveform / (scale + 1e-8))
        generated, lengths = self.sampler(prior_mean=encoded, estimator_condition=encoded, state_length=lengths)
        output, _ = self.decoder(input=generated, input_length=lengths)
        output = output * scale
        target_length = waveform.shape[-1]
        return torch.nn.functional.pad(output[..., :target_length], (0, max(0, target_length - output.shape[-1])))
