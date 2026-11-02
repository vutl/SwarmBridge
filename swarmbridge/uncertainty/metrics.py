"""Q1b probes: does bridge uncertainty predict task errors *beyond* channel reliability r?

auroc(U, err)             : U as an error detector (all pixels)
auroc(1 - r, err)         : zero-cost channel baseline
binned_auroc(U, err, r)   : AUROC of U *within* bins of r, averaged. ~0.5 means U carries
                            nothing beyond r (the circularity the Q1 review warned about).
"""
import torch


def auroc(score, label, jitter=1e-6):
    s = score.flatten().float()
    y = label.flatten().bool()
    n1, n0 = y.sum().item(), (~y).sum().item()
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = s + jitter * torch.rand_like(s)  # break ties (1 - r has many exact ties)
    ranks = torch.empty_like(s)
    ranks[s.argsort()] = torch.arange(1, s.numel() + 1, device=s.device, dtype=s.dtype)
    return ((ranks[y].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binned_auroc(score, label, r, nbins=5, min_count=200):
    s, y, rr = score.flatten(), label.flatten(), r.flatten()
    bins = (rr * nbins).long().clamp(0, nbins - 1)
    total, acc = 0, 0.0
    for b in range(nbins):
        m = bins == b
        if m.sum() < min_count:
            continue
        a = auroc(s[m], y[m])
        if a == a:  # not NaN
            acc += a * m.sum().item()
            total += m.sum().item()
    return acc / total if total else float("nan")
