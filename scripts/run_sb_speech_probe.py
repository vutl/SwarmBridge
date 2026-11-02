"""Small, paired pretrained speech inference probe; not a full paper reproduction."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from pystoi import stoi

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'runs/sb_paper_reproductions'


def metrics(reference, signal):
    reference = reference.astype(np.float64) - reference.mean()
    signal = signal.astype(np.float64) - signal.mean()
    projection = np.dot(signal, reference) / max(np.dot(reference, reference), 1e-12) * reference
    sisdr = 10 * np.log10(max(np.dot(projection, projection), 1e-12) /
                         max(np.dot(signal - projection, signal - projection), 1e-12))
    return dict(si_sdr_db=float(sisdr), estoi=float(stoi(reference, signal, 16000, extended=True)))


def load_wave(path):
    audio, sample_rate = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sample_rate != 16000:
        from math import gcd
        divisor = gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('nvidia', 'sony'), default='nvidia')
    parser.add_argument('--files', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261031)
    destination = OUT / args.engine
    destination.mkdir(parents=True, exist_ok=True)
    if args.engine == 'nvidia':
        from nemo_sb_inference_adapter import OfficialNemoSB
        checkpoint = OUT / 'checkpoints/se_den_sb_16k_small.nemo'
        model = OfficialNemoSB(checkpoint).eval().cuda()
        modes = [('ode', 1), ('ode', 10), ('sde', 10), ('sde', 50)]
        enhance = lambda x, mode, steps: model.enhance(x, process=mode, steps=steps)
    else:
        from sony_sb_inference_adapter import load_sony
        checkpoint = OUT / 'downloads/sony_sbctm.ckpt'
        model, enhance = load_sony(checkpoint)
        modes = [('ctm', 1), ('ctm', 4)]
    print('MODEL_LOADED', args.engine, sum(p.numel() for p in model.parameters()), flush=True)
    rows = []
    files = sorted((ROOT / 'data/sb_speech_test/clean').glob('*.wav'))[:args.files]
    if not files:
        raise ValueError('No paired test files')
    manifest = dict(engine=args.engine, n=len(files), checkpoint=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    subset='First 12 archive members; one speaker, small inference probe, not full test benchmark.',
                    resampling='scipy.signal.resample_poly, official 48 kHz test audio to 16 kHz',
                    scope='Official numerical code and pretrained weights with documented Windows inference adapters; no retraining.')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for path in files:
        clean = load_wave(path)
        noisy = load_wave(ROOT / 'data/sb_speech_test/noisy' / path.name)
        length = min(len(clean), len(noisy))
        clean, noisy = clean[:length], noisy[:length]
        input_metrics = metrics(clean, noisy)
        sf.write(destination / ('clean_' + path.name), clean, 16000, subtype='FLOAT')
        sf.write(destination / ('noisy_' + path.name), noisy, 16000, subtype='FLOAT')
        signal = torch.from_numpy(noisy.copy())[None, None].cuda()
        for process, steps in modes:
            label = f'{process}_{steps}'
            target = destination / f'{label}_{path.name}'
            torch.manual_seed(20261031)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                output = enhance(signal, process, steps).reshape(-1).cpu().numpy()[:length]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            if len(output) != length or not np.isfinite(output).all():
                raise ValueError('Invalid reconstruction')
            sf.write(target, output, 16000, subtype='FLOAT')
            row = dict(file=path.name, mode=label, input=input_metrics, output=metrics(clean, output),
                       seconds=seconds, duration=length / 16000)
            rows.append(row)
            (destination / 'results.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    from statistics import mean
    summary = {f'{p}_{n}': {k: mean(r['output'][k] for r in rows if r['mode'] == f'{p}_{n}')
                          for k in ('si_sdr_db', 'estoi')} for p, n in modes}
    summary['input'] = {k: mean(r['input'][k] for r in rows if r['mode'] == rows[0]['mode'])
                        for k in ('si_sdr_db', 'estoi')}
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
