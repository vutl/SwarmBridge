"""Reference-process geometry Lambda(c) for the bridge (the object Q1 is about).

The reference is dZ = sqrt(beta0 * Lambda(c)) dW with Lambda diagonal and constant in
generative time, so the endpoint-conditioned bridge stays a per-coordinate Brownian bridge.

  R0 'iso'         : Lambda = 1                                  (standard isotropic)
  R1 'reliability' : Lambda = lam_min + (1 - r) (lam_max - lam_min) (hand-crafted)
  R2 'empirical'   : Lambda[c, bin(r)] = Var[z* - z_p | r-bin]     (fitted, per channel;
                     closest to PRISM's "noise spectrum ~ destroyed information")

R1/R2 are rescaled so their data-average equals 1, i.e. same total noise budget as R0:
only the *geometry* differs, not the amount.
"""
import torch
import torch.nn as nn

KINDS = ("iso", "reliability", "empirical")


class Reference(nn.Module):
    def __init__(self, kind, C, lam_min=0.2, lam_max=1.8, nbins=8):
        super().__init__()
        assert kind in KINDS, kind
        self.kind, self.lam_min, self.lam_max, self.nbins = kind, lam_min, lam_max, nbins
        self.register_buffer("table", torch.ones(C, nbins))
        self.register_buffer("scale", torch.tensor(1.0))

    def _bin(self, r):
        return (r * self.nbins).long().clamp(0, self.nbins - 1)

    def raw(self, r):
        """r: (B,1,H,W) -> Lambda: (B,1,H,W) or (B,C,H,W)."""
        if self.kind == "iso":
            return torch.ones_like(r)
        if self.kind == "reliability":
            return self.lam_min + (1 - r) * (self.lam_max - self.lam_min)
        return self.table.t()[self._bin(r).squeeze(1)].permute(0, 3, 1, 2)

    def forward(self, r):
        return (self.raw(r) * self.scale).clamp_min(1e-3)

    @torch.no_grad()
    def fit(self, batches):
        """batches: iterable of dicts with z_star, z_p, r (training distribution)."""
        if self.kind == "iso":
            return
        C = self.table.shape[0]
        sumsq = torch.zeros(C, self.nbins, device=self.table.device)
        count = torch.zeros(self.nbins, device=self.table.device)
        rs = []
        for b in batches:
            bins = self._bin(b["r"]).squeeze(1)  # (B,H,W)
            res2 = (b["z_star"] - b["z_p"]).pow(2)  # (B,C,H,W)
            onehot = nn.functional.one_hot(bins, self.nbins).float()  # (B,H,W,nb)
            sumsq += torch.einsum("bchw,bhwk->ck", res2, onehot)
            count += onehot.sum((0, 1, 2))
            rs.append(b["r"])
        if self.kind == "empirical":
            table = sumsq / count.clamp_min(1)
            fill = (sumsq.sum(1) / count.sum()).unsqueeze(1).expand_as(table)
            self.table.copy_(torch.where(count > 0, table, fill))
        self.scale.fill_(1.0)
        self.scale.fill_(1.0 / self.raw(torch.cat(rs)).mean().item())
