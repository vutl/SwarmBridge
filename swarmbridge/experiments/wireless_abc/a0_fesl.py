"""A0: smoke reproduction of the Fesl et al. (WCL 2024) DM channel estimator with the shipped checkpoints.

Runs the authors' DMCE code (repo pdf/GithubModels/benediktfesl__Diffusion_channel_est, BSD-3) on at most N_TEST test
channels of the published dataset (Zenodo 20737830, `bin/`), with the authors' Tester (SNR -10..40 dB, deterministic
reverse process, SNR-matched start step) and compares the NMSE with the shipped results/<model>/test_results.json
(computed by the authors on 10 000 test channels). Differences from the paper run: N_TEST <= 1024 samples, our torch
2.12 / numpy 2.5 env instead of torch 2.1, CPU or one GPU.
Usage: python -I a0_fesl.py REPO BIN_DIR OUT [--device cpu] [--model 3gpp|quadriga_LOS] [--n 1024]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("bin"); ap.add_argument("out")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--model", default="3gpp"); ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    import DMCE  # noqa: E402
    from DMCE.utils import cmplx2real  # noqa: E402
    n_rx, n_tx = 64, 16
    if args.model == "3gpp":
        f = Path(args.bin) / f"3gpp_path=3_dimrx={n_rx}_dimtx={n_tx}_samp=10000_test.npy"
        data = np.load(f)
        data = np.reshape(data, (-1, n_rx, n_tx), "F")[:args.n]
        mdir = Path(args.repo) / "results/best_models_dm_paper/3gpp_path=3"
    else:
        import h5py
        f = Path(args.bin) / f"120000UEs_500m_1x{n_rx}BS_1x{n_tx}MS_1carr_1symb_diff=0_quadriga_LOS.mat"
        ch = h5py.File(f, "r")["H_all"]
        ch = np.transpose(np.array(ch[:, :, 110000:110000 + args.n]["real"]) + 1j * np.array(ch[:, :, 110000:110000 + args.n]["imag"]))
        data = ch
        mdir = Path(args.repo) / "results/best_models_dm_paper/quadriga_LOS"
    x = torch.from_numpy(np.asarray(data[:, None, :]))
    x = cmplx2real(x, dim=1, new_dim=False).float()
    sim = DMCE.utils.load_params(str(mdir / "sim_params"))
    cnn_dict, dm_dict = sim["unet_dict"], sim["diff_model_dict"]
    cnn_dict["device"] = args.device
    dm = DMCE.DiffusionModel(DMCE.CNN(**cnn_dict), **dm_dict)
    mp = mdir / "train_models"
    dm.load_state_dict(torch.load(mp / sorted(os.listdir(mp))[-1], map_location=args.device)["model"])
    dm.reverse_add_random = False
    tester = DMCE.Tester(dm, data=x, batch_size=512, criteria=["nmse"], complex_data=False, return_all_timesteps=False,
                         fft_pre=True, mode="2D")
    torch.manual_seed(0)
    t0 = time.time()
    res = tester.test()["nmse"]
    secs = time.time() - t0
    ref = json.loads((mdir / "test_results.json").read_text())["nmse"]
    rmap = dict(zip(ref["SNRs"], ref["NMSEs_total_power"]))
    rows = []
    for s, v in zip(res["SNRs"], res["NMSEs_total_power"]):
        r = rmap.get(s)
        rows.append(dict(snr_db=s, nmse=float(v), nmse_db=float(10 * np.log10(v)),
                         ref_nmse_db=None if r is None else float(10 * np.log10(r)),
                         diff_db=None if r is None else float(10 * np.log10(v) - 10 * np.log10(r))))
        print(f"SNR {s:+5.0f} dB  NMSE {10 * np.log10(v):7.2f} dB   shipped {10 * np.log10(r) if r else float('nan'):7.2f} dB", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"a0_{args.model}.json").write_text(json.dumps(dict(model=args.model, n_test=len(data), device=args.device,
                                                                secs=round(secs, 1), data_file=f.name, rows=rows), indent=1))


if __name__ == "__main__":
    main()
