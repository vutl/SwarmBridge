"""Numerical checks of the framework/Windows inference adapters."""
import importlib.util
import json
from pathlib import Path

import torch
from nemo_sb_inference_adapter import OfficialNemoSB

ROOT = Path(__file__).resolve().parent.parent


def main():
    torch.set_num_threads(4)
    model = OfficialNemoSB(ROOT / 'runs/sb_paper_reproductions/checkpoints/se_den_sb_16k_small.nemo').cuda().eval()
    signal = torch.randn(2, 1, 32000, device='cuda') * .1
    encoded, length = model.encoder(input=signal)
    restored, _ = model.decoder(input=encoded, input_length=length)
    roundtrip_error = (signal - restored[..., :signal.shape[-1]]).abs().max().item()
    assert roundtrip_error < 1e-5
    t = torch.linspace(0, 1, 101, device='cuda')
    expected = .4 * torch.expm1(2 * torch.log(torch.tensor(2.6, device='cuda')) * t) / (2 * torch.log(torch.tensor(2.6, device='cuda')))
    clock_error = (model.noise_schedule.sigma(t).square() - expected).abs().max().item()
    assert clock_error < 1e-6
    # Confirm our independent SI-SDR calculation against the standard TorchMetrics implementation.
    from run_sb_speech_probe import metrics, load_wave
    from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
    clean = load_wave(ROOT / 'data/sb_speech_test/clean/p232_001.wav')
    noisy = load_wave(ROOT / 'data/sb_speech_test/noisy/p232_001.wav')
    value = metrics(clean, noisy)['si_sdr_db']
    reference = scale_invariant_signal_distortion_ratio(torch.from_numpy(noisy).double(),
                                                       torch.from_numpy(clean).double(), zero_mean=True).item()
    assert abs(value - reference) < 1e-6
    record = dict(nvidia_checkpoint_strict_load=True, stft_roundtrip_max_error=roundtrip_error,
                  ve_clock_vs_closed_form_max_error=clock_error, sisdr_vs_torchmetrics_error=abs(value-reference))
    (ROOT / 'runs/sb_paper_reproductions/compatibility_checks.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
