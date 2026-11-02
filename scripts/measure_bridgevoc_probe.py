"""Measure actual BridgeVoC output; this vocoder task differs from denoising."""
import json
import sys
from pathlib import Path
from statistics import mean

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from pystoi import stoi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'pdf/GithubModels/Andong-Li-speech__BridgeVoC'))
from div.data_module import mel_spectrogram

torch.set_num_threads(4)
rows = []
settings = dict(n_fft=1024, num_mels=100, sampling_rate=24000, hop_size=256,
                win_size=1024, fmin=0, fmax=12000)
for path in sorted((ROOT / 'runs/sb_paper_reproductions/bridgevoc_NFE1').glob('*.wav')):
    reference, _ = librosa.load(ROOT / 'data/sb_speech_test/clean' / path.name, sr=24000, mono=True)
    output, rate = sf.read(path, dtype='float32')
    assert rate == 24000 and len(reference) == len(output) and np.isfinite(output).all()
    normalization = np.max(np.abs(reference)) + 1e-6
    with torch.no_grad():
        clean_mel = mel_spectrogram(torch.from_numpy(reference / normalization)[None], **settings)
        output_mel = mel_spectrogram(torch.from_numpy(output / normalization)[None], **settings)
    rows.append(dict(file=path.name, estoi=float(stoi(resample_poly(reference, 2, 3),
                 resample_poly(output, 2, 3), 16000, extended=True)),
                 log_mel_l1=float((clean_mel-output_mel).abs().mean()), duration=len(output)/24000))
result = dict(task='Vocoder: clean waveform -> 100-band mel -> range-space prior -> pretrained SB -> waveform',
              variant='Extended BridgeVoC single-step, Andong-Li-speech/BridgeVoC; not original IJCAI checkpoint',
              n=len(rows), checkpoint_training='LibriTTS', test_subset='12 VoiceBank test utterances, one speaker',
              metrics=dict(estoi=mean(r['estoi'] for r in rows), log_mel_l1=mean(r['log_mel_l1'] for r in rows)),
              no_claim='No matched predictive vocoder baseline evaluated; no comparison with denoising task scores.',
              results=rows)
(ROOT / 'runs/sb_paper_reproductions/bridgevoc_results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result['metrics']), flush=True)
