"""The Q1 control grid. Each row isolates one mechanism:

            net sees CSI?   noise/reference depends on r?   bridge?
  M0        no              no                              no
  M1        yes             no                              no
  M2        no              yes (augmentation)              no
  M3        yes             yes (augmentation)              no     <- must beat this
  M4        yes             iso augmentation                no     (is it shape or just noise?)
  F1        yes             -                               flow matching
  S0        no              iso reference                   yes
  S1        yes             iso reference                   yes    <- and this
  S2        no              R1 reliability reference        yes
  S3        yes             R1 reliability reference        yes    (main hypothesis)
  S4        yes             R2 empirical reference          yes    (PRISM-like matched)
  H1        yes             -                               no     mapping + variance head (Q1b baseline)
"""
from ...bridges import FlowMatching, HeteroMapping, Mapping, Reference, SchrodingerBridge

GRID = {
    "M0": dict(kind="map", csi=False, aug="none"),
    "M1": dict(kind="map", csi=True, aug="none"),
    "M2": dict(kind="map", csi=False, aug="aniso"),
    "M3": dict(kind="map", csi=True, aug="aniso"),
    "M4": dict(kind="map", csi=True, aug="iso"),
    "F1": dict(kind="fm", csi=True),
    "S0": dict(kind="sb", csi=False, ref="iso"),
    "S1": dict(kind="sb", csi=True, ref="iso"),
    "S2": dict(kind="sb", csi=False, ref="reliability"),
    "S3": dict(kind="sb", csi=True, ref="reliability"),
    "S4": dict(kind="sb", csi=True, ref="empirical"),
    "H1": dict(kind="hetero", csi=True),
}


def build(name, C, width, fit_batches, beta0=1.0, t1_frac=0.0):
    spec = GRID[name]
    device = fit_batches[0]["r"].device
    if spec["kind"] == "map":
        ref = None
        if spec["aug"] == "aniso":
            ref = Reference("reliability", C).to(device)
            ref.fit(fit_batches)
        return Mapping(C, spec["csi"], spec["aug"], ref, width=width)
    if spec["kind"] == "fm":
        return FlowMatching(C, spec["csi"], width=width)
    if spec["kind"] == "hetero":
        return HeteroMapping(C, spec["csi"], width=width)
    ref = Reference(spec["ref"], C).to(device)
    ref.fit(fit_batches)
    return SchrodingerBridge(C, ref, spec["csi"], beta0=beta0, width=width, t1_frac=t1_frac)
