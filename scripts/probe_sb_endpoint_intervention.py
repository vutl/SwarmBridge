"""Paired endpoint-allocation intervention from one existing 30k checkpoint.

Two 500-update continuations share starting raw weights, AdamW, EMA and data/RNG
streams. Only endpoint_fraction changes. This is a local mechanism probe, not
the proposed from-scratch 6k pilot or a budget-matched victory over mapping.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from swarmbridge.experiments.sb_audit.run import (BevData, atomic_json, evaluate_bev, gpu_lock,
                                                make_model, now, train_until)

ORIGINAL = ROOT / 'runs/sb_audit_ve_v1'
OUT = ROOT / 'runs/sb_endpoint_intervention'


class OneStepView:
    bridge = False  # evaluate_bev selects only its one-step evaluation mode

    def __init__(self, model):
        self.model = model

    def sample(self, source, reliability, **kwargs):
        return self.model.sample(source, reliability, **kwargs)


def raw_evaluation(path, data):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = make_model('sb_ve_xt_endpoint', 16, 64, .1).cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    result = evaluate_bev(OneStepView(model), data, 0, 200, 32)
    del model, saved
    torch.cuda.empty_cache()
    return result


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    source = ORIGINAL / 'sb_ve_xt_endpoint_s0/last.pt'
    source_sha = hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()
    atomic_json(OUT/'protocol.json', dict(start_step=30000, additional_updates=500, seed=0,
                varied_factor='endpoint sample probability: 0.1 vs 0.5',
                same_initial_model_optimizer_ema_and_rng=True, source_checkpoint=str(source),
                source_sha256=source_sha, primary='one_step validation at fixed final update 30500',
                limitation='One-seed local continuation, not from-scratch pilot or matched comparison with mapping.'))
    settings = json.loads((ORIGINAL/'manifest.json').read_text())['settings']
    status = dict(state='starting', pid=os.getpid(), started_at=now())
    with gpu_lock():
        data = BevData('runs/teacher', 'cuda')
        initial_path = OUT/'initial_raw_evaluation.json'
        if not initial_path.exists():
            atomic_json(initial_path, raw_evaluation(source, data))
        for probability, label in ((.1, 'endpoint10'), (.5, 'endpoint50')):
            folder = OUT / label
            checkpoint = folder/'sb_ve_xt_endpoint_s0/last.pt'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.exists():
                shutil.copyfile(source, checkpoint)
            config = settings.copy()
            config.update(out=str(folder), suite='bev', endpoint_fraction=probability,
                          log_every=100, save_every=100)
            args = argparse.Namespace(**config)
            status.update(state='running', branch=label, updated_at=now())
            atomic_json(OUT/'status.json', status)
            train_until('sb_ve_xt_endpoint', data, 0, 30500, args, status)
            atomic_json(folder/'final_raw_evaluation.json', raw_evaluation(checkpoint, data))
        status.update(state='completed', updated_at=now())
        atomic_json(OUT/'status.json', status)
        print('ENDPOINT INTERVENTION COMPLETED', flush=True)


if __name__ == '__main__':
    main()
