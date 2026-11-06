"""Simplified Huang-inspired SAGIN model (step 0): mode table, channels, SCE.

Source: Huang et al., "DRL-Based Resource Allocation for Hybrid Bit and Generative Semantic Communications in
SAGINs", IEEE JSAC 43(12), 2025 (arXiv 2412.05647). Everything marked PAPER is taken from the text or Fig. 5;
everything marked ASSUMPTION is not specified in the paper and is chosen here (and varied where it matters).
This is NOT a reproduction of the paper's simulator.
"""
import numpy as np
from scipy.special import j0

C = 3e8
KB = 1.380649e-23
R_E = 6371e3

# ---- PAPER (Sec. V, Fig. 5) ----
K_LEO, N_UAV, M_USER = 5, 3, 8
H_LEO = 750e3
V_LEO = 7.8e3
TAU_MIN = np.deg2rad(40.0)
F_LEO = 25e9
GAIN_LEO = 10 ** (40 / 10)
P_LEO, P_UAV = 1.0, 0.2
CHI = 354.81                      # thermal noise temperature [K] (used for ISL, eq. 4)
NOISE_PAPER = 10 ** ((-130 - 30) / 10)   # -130 dBm at UAVs and users
BW = 10e6
KAPPA_L, KAPPA_N = 2.0, 2.6
UAV_ALT = 50.0
V_UAV = 10.0
G_TASKS = 100
DMAX_RANGE = (2.0, 11.0)
QMIN_RANGE = (10.0, 30.0)
XI_Q = XI_D = 0.5

# Fig. 5: PSNR [dB] and size of transmitted information [kbit] (512x512 image); compute delay Sec. V
MODES = ["M1_jpeg2000", "M2_text", "M3_1", "M3_2", "M3_3"]
PSNR = np.array([32.2, 11.26, 16.3, 18.8, 23.2])
KBIT = np.array([314.0, 0.376, 0.897, 8.575, 33.151])
COMPUTE_S = np.array([0.1323, 1.33, 1.33, 1.33, 1.33])
ORIGINAL_IMAGE_BITS = 3.5e6 * 8    # "size of an original image is 3.5 MB" (used by the M1 payload variant)

# ---- ASSUMPTIONS ----
RICIAN_MU = 10.0                   # Rician factor (linear) for UAV-user links
AREA = 500.0                       # each of the N areas is AREA x AREA m, areas side by side
ISL_GAIN = GAIN_LEO                # peak ISL antenna gain eta_max
ISL_FREQ = F_LEO
Q_NORM = (PSNR.min(), PSNR.max())  # SCE normalisation of PSNR (min-max over the mode table)
D_NORM = DMAX_RANGE[1]             # SCE normalisation of delay (largest deadline)


def noise_power(kind):
    """'paper' = -130 dBm as stated; 'thermal' = k*chi*B with the paper's own chi (about -103 dBm)."""
    return NOISE_PAPER if kind == "paper" else KB * CHI * BW


def mode_bits(m1_payload="fig5"):
    bits = KBIT * 1e3
    if m1_payload == "original":
        bits = bits.copy()
        bits[0] = ORIGINAL_IMAGE_BITS
    return bits


# ---- geometry of an overhead LEO pass ----
OMEGA = V_LEO / (R_E + H_LEO)      # orbital angular rate [rad/s]


def leo_elev_dist(phi):
    """Central angle phi (rad) between sub-satellite point and the area -> (elevation rad, slant range m)."""
    rs = R_E + H_LEO
    d = np.sqrt(R_E ** 2 + rs ** 2 - 2 * R_E * rs * np.cos(phi))
    elev = np.arctan2(rs * np.cos(phi) - R_E, rs * np.abs(np.sin(phi)))
    return elev, d


PHI_VIS = np.arccos(R_E * np.cos(TAU_MIN) / (R_E + H_LEO)) - TAU_MIN   # |phi| <= PHI_VIS  <=> elevation >= tau


def access_time():
    """Eq. (7)-(8): full pass duration for an overhead pass."""
    rk = 2 * (R_E + H_LEO) * PHI_VIS
    return rk / V_LEO


def outdated_csi_delta(d):
    """Eq. (2): delta = J0(2 pi f_D T) with f_D = v f / c and T = d / c."""
    fd = V_LEO * F_LEO / C
    return float(j0(2 * np.pi * fd * d / C))


def leo_uav_snr_mean(d, noise):
    """Mean SNR of eq. (1)-(3); with the eq. (2) outdated-CSI model the gain is |h|^2 = |delta h + sqrt(1-delta^2) g|^2."""
    lam = C / F_LEO
    return P_LEO * GAIN_LEO * (lam / (4 * np.pi * d)) ** 2 / noise


def leo_uav_rate(d, noise, fade=1.0):
    """fade = |h|^2 / |h_hat|^2 (exp(1) draw for Rayleigh-dominated outdated CSI; 1.0 for the mean)."""
    return BW * np.log2(1 + leo_uav_snr_mean(d, noise) * fade)


def expected_log_rate(snr_mean, n=4000, seed=0):
    g = np.random.default_rng(seed).exponential(1.0, n)
    return BW * float(np.mean(np.log2(1 + snr_mean * g)))


def uav_user_gain(dist, nlos_draw):
    """Eq. (5): |sqrt(mu/(mu+1)) gbar d^-kL/2 + sqrt(1/(mu+1)) ghat d^-kN/2|^2, gbar unit modulus (phase 0)."""
    mu = RICIAN_MU
    los = np.sqrt(mu / (mu + 1)) * dist ** (-KAPPA_L / 2)
    nl = np.sqrt(1 / (mu + 1)) * nlos_draw * dist ** (-KAPPA_N / 2)
    return np.abs(los + nl) ** 2


def uav_user_rate(dist, nlos, interferer_gains, noise):
    """Eq. (6) with interference from the UAVs that are transmitting at the same time."""
    s = P_UAV * uav_user_gain(dist, nlos)
    i = P_UAV * float(np.sum(interferer_gains))
    return BW * np.log2(1 + s / (i + noise))


def isl_rate(d):
    """Eq. (4)."""
    fspl = (4 * np.pi * d * ISL_FREQ / C) ** 2
    return BW * np.log2(1 + P_LEO * ISL_GAIN / (KB * CHI * BW * fspl))


def norm_q(q):
    lo, hi = Q_NORM
    return (q - lo) / (hi - lo)


def sce(q, d, qmin, dmax):
    """Eq. (14) with min-max normalised PSNR and delay normalised by the largest deadline."""
    return XI_Q * (norm_q(q) - norm_q(qmin)) + XI_D * (dmax - d) / D_NORM


def utility(q, d, qmin, dmax):
    """Per-task score: SCE, minus 1 if a constraint (15h)/(15i) is violated (the eq. 18 penalty, per task)."""
    ok = (d <= dmax) and (q >= qmin)
    return sce(q, d, qmin, dmax) - (0.0 if ok else 1.0), ok
