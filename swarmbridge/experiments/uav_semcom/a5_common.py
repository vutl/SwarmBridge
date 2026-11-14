"""A5 shared definitions: data paths, mode set, seeds, operating regimes, table loading.

Reference regime = the A4 configuration (K = 3 UAVs, M = 2 channels, 1 MHz, survey at 20 m/s, `predicted` mode info,
w_T = 0.3, deadlines U(4, 12) s, Qmin U(22, 30) dB, urban air-to-ground channel, Shannon rate), now on the A5 image
pools. Every other regime changes ONE group of parameters (one-dimensional sweeps, no Cartesian product), each with
a physical reason (report reports/archive/uav-semcom-a/uav-semcom-a5.md section 3):
  K2/K4/K5        : number of UAVs sharing M = 2 channels (contention; K2 = no contention)
  v10/v15         : survey speed (frame rate per UAV, 0.4 / 0.6 / 0.8 frames/s)
  env_*           : air-to-ground propagation environment, Al-Hourani et al. IEEE WCL 2014 (a, b, eta_LoS, eta_NLoS)
                    suburban / dense urban / high-rise urban (reference: urban)
  fade_k3/fade_k0 : weaker LoS (Rician K factor 3 / 0 dB instead of 10 dB): stronger small-scale fading
  dl_tight/loose  : per-image deadline U(2, 6) / U(8, 20) s
  q_low/q_high    : per-image quality floor U(18, 26) / U(28, 36) dB
  wt0/wt0.1/wt1   : weight of mission time in the reward (objective change, reported as a Pareto sweep)
  phy_*           : PHY abstraction (env.UavSemcomEnv.link_rate): 3 dB SNR gap, LTE-CQI MCS with genie link
                    adaptation, LTE-CQI MCS with interference-unaware link adaptation (block fails if SINR < threshold)
Seeds: tuning / training episodes 0..29 (train pool), validation 3000..3049 (val pool; PPO checkpoint selection uses
3000..3019), final test 7000..7099 (test pool and test-dev pool). None overlaps the A4 seed sets.
"""
import json
from pathlib import Path

import env as E

REPO = Path(__file__).resolve().parents[3]
A5 = REPO / "runs" / "uav_semcom" / "a5"
MANIFEST = A5 / "manifest.json"
PROFILES_VAL = [A5 / "profile" / "a11_val" / "profile.jsonl", A5 / "profile" / "a12_val" / "profile.jsonl"]
PROFILES_TD = [A5 / "profile" / "a11_testdev" / "profile.jsonl", A5 / "profile" / "a12_testdev" / "profile.jsonl"]
CPU_FILE = A5 / "profile" / "a11_val" / "cpu_learned.jsonl"
MODES = ["cap_t2i", "thumb64_up", "cap_thumb64_i2i_t0.6", "cap_thumb128_i2i_t0.3", "avif_5", "avif_20", "avif_50",
         "avif_80", "mbt2018-mean_q1", "mbt2018-mean_q3", "mbt2018-mean_q5", "mbt2018-mean_q8", "jpeg2000_100",
         "jpeg2000_25"]                                        # the A4 mode set (run_a4.sh)

TRAIN_SEEDS = list(range(30))
VAL_SEEDS = list(range(3000, 3050))
PPO_VAL_SEEDS = list(range(3000, 3020))
TEST_SEEDS = list(range(7000, 7100))
A4_SEEDS = set(range(30)) | set(range(1000, 1030)) | set(range(2000, 2010)) | set(range(5000, 5100))
assert not (set(VAL_SEEDS) | set(TEST_SEEDS)) & A4_SEEDS

REF = dict(mode_info="predicted", v_max=20.0, w_time=0.3)
URBAN = dict(los_a=9.61, los_b=0.16, eta_los_db=1.0, eta_nlos_db=20.0)       # = env.DEFAULT (reference)
REGIMES = {
    "ref": {},
    "K2": dict(K=2), "K4": dict(K=4), "K5": dict(K=5),
    "v10": dict(v_max=10.0), "v15": dict(v_max=15.0),
    "env_suburban": dict(los_a=4.88, los_b=0.43, eta_los_db=0.1, eta_nlos_db=21.0),
    "env_dense": dict(los_a=12.08, los_b=0.11, eta_los_db=1.6, eta_nlos_db=23.0),
    "env_highrise": dict(los_a=27.23, los_b=0.08, eta_los_db=2.3, eta_nlos_db=34.0),
    "fade_k3": dict(rician_k_db=3.0), "fade_k0": dict(rician_k_db=0.0),
    "dl_tight": dict(deadline=(2.0, 6.0)), "dl_loose": dict(deadline=(8.0, 20.0)),
    "q_low": dict(qmin=(18.0, 26.0)), "q_high": dict(qmin=(28.0, 36.0)),
    "wt0": dict(w_time=0.0), "wt0.1": dict(w_time=0.1), "wt1": dict(w_time=1.0),
    "phy_gap3": dict(phy="gap", phy_gap_db=3.0), "phy_mcs": dict(phy="mcs"), "phy_mcs_blind": dict(phy="mcs_blind"),
    # A8 mismatch ladder (reports/archive/a7-a8-b1/swarmbridge-a8-b1-preregistration.md): link adaptation sees lambda x interference
    "phy_mcs_l75": dict(phy="mcs_partial", phy_lambda=0.75), "phy_mcs_l50": dict(phy="mcs_partial", phy_lambda=0.5),
    "phy_mcs_l25": dict(phy="mcs_partial", phy_lambda=0.25),
}
for _k, _v in URBAN.items():
    assert E.DEFAULT[_k] == _v


def regime_cfg(name, **extra):
    """Full cfg overrides of a regime ('NAME' or 'NAME@oracle' etc. for a mode-info variant)."""
    base, _, info = name.partition("@")
    cfg = dict(REF, **REGIMES[base], **extra)
    if info:
        cfg["mode_info"] = info
    return cfg


def load_table(testdev=False):
    """The A5 mode table: val-pool profiles (+ test-dev profiles). Encode / decode latencies are always the means over
    the VAL-pool images (+ CPU timing of learned codecs), so adding the test-dev rows does not change them; the
    controller-side knowledge is fitted on the train split only."""
    tv = E.ModeTable([str(p) for p in PROFILES_VAL], MODES, str(CPU_FILE))
    if not testdev:
        pools = tv.split_manifest(str(MANIFEST))
        return tv, pools
    t = E.ModeTable([str(p) for p in PROFILES_VAL + PROFILES_TD], MODES, str(CPU_FILE))
    t.enc_s, t.dec_s = tv.enc_s.copy(), tv.dec_s.copy()
    pools = t.split_manifest(str(MANIFEST))
    return t, pools


def manifest():
    return json.loads(MANIFEST.read_text())
