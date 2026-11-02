"""Diagnose existing BEV checkpoints; no training and no new recipe selection.

Uses the original validation channels/masks. Regional scores describe where
outputs fail, not independent causes: the decoder has a spatial receptive field.
Latent swaps are exploratory interventions, not trained deployment policies.
"""
import json
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from swarmbridge.channel import aggregate, sample_params, transmit
from swarmbridge.data.real import CachedLatents
from swarmbridge.experiments.q2_progressive.run import EVAL_CFG, random_subset
from swarmbridge.experiments.q2_retx.run import to_blocks, to_cells
from swarmbridge.experiments.sb_audit.run import make_model, confusion_metrics
from swarmbridge.teacher.model import IGNORE

RUN = ROOT / 'runs/sb_audit_ve_v1'
OUT = ROOT / 'runs/sb_bev_region_diagnostic'
FRACTIONS = (.15, .5, 1.)
REGIONS = ('all_valid', 'no_evidence', 'weak_evidence', 'strong_evidence')
MODES = ('mapping', 'sb', 'mapping_received_sb_missing', 'sb_received_mapping_missing')


def scores(matrix):
    values = confusion_metrics(matrix)
    n = int(matrix.sum())
    values.update(pixels=n, pixel_error_rate=1-float(matrix.diag().sum())/max(n, 1),
                  vehicle_recall=float(matrix[2, 2])/max(int(matrix[2].sum()), 1),
                  vehicle_precision=float(matrix[2, 2])/max(int(matrix[:, 2].sum()), 1),
                  confusion=matrix.cpu().tolist())
    return values


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    ds = CachedLatents(str(ROOT/'runs/teacher'), 'cuda', eval_split='val')
    clean = ds.eval['z'].cuda().float()
    fov = ds.eval['fov48'].cuda().float()
    labels = ds.eval['y'].cuda().long()
    results = []
    for seed in (0, 1, 2):
        models = {}
        for label, name in [('mapping', 'mapping_xt'), ('sb', 'sb_ve_xt_endpoint')]:
            model = make_model(name, ds.C, 64, .1).cuda().eval()
            model.load_state_dict(torch.load(RUN/f'{name}_s{seed}/ema_30000.pt', map_location='cuda', weights_only=True), strict=True)
            models[label] = model
        parameters = sample_params(EVAL_CFG, len(clean), 1, 'cuda', torch.Generator(device='cuda').manual_seed(3000+seed), None)
        for fraction in FRACTIONS:
            selected = random_subset(to_blocks(fov)>0, fraction, torch.Generator(device='cuda').manual_seed(21+seed))
            received = transmit(clean[:, None], torch.zeros_like(clean[:, None]), (fov*to_cells(selected))[:, None],
                                parameters, EVAL_CFG, torch.Generator(device='cuda').manual_seed(31+seed))
            source, reliability = aggregate(*received)
            matrices = {mode: {region: torch.zeros(3, 3, device='cuda', dtype=torch.long) for region in REGIONS} for mode in MODES}
            mse_sums = {name: {region: [0., 0] for region in REGIONS[1:]} for name in ('mapping', 'sb')}
            for offset in range(0, len(clean), 16):
                sl = slice(offset, offset+16)
                evidence = reliability[sl] > 0
                masks = dict(no_evidence=~evidence, weak_evidence=evidence & (reliability[sl] < .5),
                             strong_evidence=reliability[sl] >= .5)
                target = labels[sl]
                valid = target != IGNORE
                label_masks = {'all_valid': valid}
                label_masks.update({key: valid & F.interpolate(value.float(), size=target.shape[-2:], mode='nearest')[:, 0].bool()
                                    for key, value in masks.items()})
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = {name: model.sample(source[sl], reliability[sl], nfe=1, stochastic=False)
                               for name, model in models.items()}
                    outputs['mapping_received_sb_missing'] = torch.where(evidence, outputs['mapping'], outputs['sb'])
                    outputs['sb_received_mapping_missing'] = torch.where(evidence, outputs['sb'], outputs['mapping'])
                    for name, output in outputs.items():
                        prediction = ds.dec(output.float()*ds.sd+ds.mu).argmax(1)
                        for region, mask in label_masks.items():
                            encoded = (3*target[mask]+prediction[mask]).long()
                            matrices[name][region] += torch.bincount(encoded, minlength=9).reshape(3, 3)
                for name in ('mapping', 'sb'):
                    error = (outputs[name].float()-clean[sl]).square().mean(1, keepdim=True)
                    for region, mask in masks.items():
                        mask = mask & (fov[sl] > 0)
                        mse_sums[name][region][0] += error[mask].sum().item()
                        mse_sums[name][region][1] += int(mask.sum())
            row = dict(seed=seed, fraction=fraction,
                       modes={name: {region: scores(value) for region, value in regions.items()} for name, regions in matrices.items()},
                       latent_mse={name: {region: value/max(count, 1) for region, (value, count) in groups.items()} for name, groups in mse_sums.items()})
            # Verify baseline reconstruction against the original audit.
            differences = {}
            for name, original in [('mapping', 'mapping_xt'), ('sb', 'sb_ve_xt_endpoint')]:
                previous = json.loads((RUN/f'{original}_s{seed}/evaluation_30000.json').read_text())['fractions'][str(fraction)]['one_step']['miou']
                differences[name] = abs(row['modes'][name]['all_valid']['miou']-previous)
            row['reproduction_abs_difference'] = differences
            if max(differences.values()) > .001:
                raise RuntimeError('Existing validation result did not reproduce closely: '+str(differences))
            results.append(row)
            (OUT/'results.json').write_text(json.dumps(results, indent=2))
            print('DONE', seed, fraction, {name: round(row['modes'][name]['all_valid']['miou'], 5) for name in MODES}, flush=True)
        del models
        torch.cuda.empty_cache()
    summary = {}
    for fraction in FRACTIONS:
        subset = [row for row in results if row['fraction'] == fraction]
        summary[str(fraction)] = {
            'modes': {name: {region: {key: mean(row['modes'][name][region][key] for row in subset)
                                    for key in ('miou','vehicle_iou','pixel_error_rate','vehicle_recall','vehicle_precision','pixels')}
                            for region in REGIONS} for name in MODES},
            'latent_mse': {name: {region: mean(row['latent_mse'][name][region] for row in subset) for region in REGIONS[1:]}
                           for name in ('mapping', 'sb')}}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETED', str(OUT), flush=True)


if __name__ == '__main__':
    main()
