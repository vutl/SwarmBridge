"""Core linear-Gaussian tools for the wireless A/B/C pilot (numpy only).

Measurement model (downlink beam training, BS N antennas, UAV 1 antenna):
    y_k = sqrt(Ep) * w_k^H h + n_k,  k = 1..Np,  n_k ~ CN(0, noise_var),  ||w_k|| = 1
written y = A h + n with A = sqrt(Ep) * W^H, W = [w_1 .. w_Np] (N x Np).
Conventions: complex vectors are 1-D numpy complex arrays; CN(m, C) means E[(h-m)(h-m)^H] = C.
"""
import numpy as np


def c2r(x):
    x = np.asarray(x)
    return np.concatenate([x.real, x.imag], axis=-1)


def r2c(x):
    x = np.asarray(x)
    n = x.shape[-1] // 2
    return x[..., :n] + 1j * x[..., n:]


def crandn(rng, *shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def sample_cn(rng, mean, cov, n):
    """n samples of CN(mean, cov) -> (n, N)."""
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(mean)))
    return mean[None] + crandn(rng, n, len(mean)) @ L.T


def dft_codebook(n):
    """Unitary DFT codebook, columns = beams (ULA, half-wavelength): F[m, b] = exp(-j 2 pi m b / n) / sqrt(n)."""
    m = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(m, m) / n) / np.sqrt(n)


def ula_steering(n, sin_theta):
    m = np.arange(n)
    return np.exp(1j * np.pi * m * sin_theta) / np.sqrt(n)


def pilot_beams(n, n_p, kind, rng=None):
    """N x Np matrix of unit-norm pilot beams. kind 'dft': Np evenly spaced DFT beams (partial sweep);
    'random': random unit-modulus (phases uniform), normalised to unit norm."""
    if kind == "dft":
        idx = np.round(np.linspace(0, n, n_p, endpoint=False)).astype(int)
        return dft_codebook(n)[:, idx]
    if kind == "random":
        return np.exp(2j * np.pi * rng.random((n, n_p))) / np.sqrt(n)
    raise ValueError(kind)


def measure(h, W, Ep, noise_var, rng):
    """h: (..., N). Returns y: (..., Np)."""
    A = np.sqrt(Ep) * W.conj().T
    y = h @ A.T
    return y + np.sqrt(noise_var) * crandn(rng, *y.shape)


def lmmse(y, W, Ep, noise_var, mu, C):
    """Gaussian posterior of h ~ CN(mu, C) given y = A h + n. Returns (mean, cov) for one y (Np,)."""
    A = np.sqrt(Ep) * W.conj().T
    S = A @ C @ A.conj().T + noise_var * np.eye(A.shape[0])
    K = C @ A.conj().T @ np.linalg.inv(S)
    mean = mu + K @ (y - A @ mu)
    cov = C - K @ A @ C
    return mean, 0.5 * (cov + cov.conj().T)


def log_cn_pdf(y, mean, cov):
    d = y - mean
    sign, logdet = np.linalg.slogdet(cov)
    return float(-len(y) * np.log(np.pi) - logdet.real - (d.conj() @ np.linalg.solve(cov, d)).real)


def gmm_posterior(y, W, Ep, noise_var, weights, means, covs):
    """Posterior of h under a GMM prior sum_k pi_k CN(m_k, C_k): returns (post_weights, post_means, post_covs).
    Exact for a known GMM prior (each component is conjugate)."""
    A = np.sqrt(Ep) * W.conj().T
    logw, pm, pc = [], [], []
    for pi, m, C in zip(weights, means, covs):
        S = A @ C @ A.conj().T + noise_var * np.eye(A.shape[0])
        logw.append(np.log(pi) + log_cn_pdf(y, A @ m, S))
        mk, ck = lmmse(y, W, Ep, noise_var, m, C)
        pm.append(mk)
        pc.append(ck)
    logw = np.array(logw)
    w = np.exp(logw - logw.max())
    return w / w.sum(), np.array(pm), np.array(pc)


def second_moment(post_w, post_m, post_c):
    """E[h h^H | y] for a Gaussian mixture posterior."""
    return sum(p * (c + np.outer(m, m.conj())) for p, m, c in zip(post_w, post_m, post_c))


def expected_gain(codebook, R):
    """E|w^H h|^2 = w^H R w for each codebook column."""
    return np.real(np.einsum("nb,nm,mb->b", codebook.conj(), R, codebook))


def beam_gain(codebook, h):
    """|w^H h|^2 per beam. h: (..., N) -> (..., B)."""
    return np.abs(h @ codebook.conj()) ** 2


def rate(gain, snr):
    return np.log2(1 + snr * gain)
