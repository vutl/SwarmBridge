"""Unit tests for linear.py. Run from this directory: python -m pytest -q -p no:cacheprovider"""
import numpy as np
import pytest
from linear import (c2r, r2c, crandn, sample_cn, dft_codebook, ula_steering, pilot_beams, measure, lmmse, gmm_posterior,
                    second_moment, expected_gain, beam_gain, log_cn_pdf)

N = 16


def rand_cov(rng, n, rank=None):
    B = crandn(rng, n, rank or n)
    return B @ B.conj().T / (rank or n) + 1e-3 * np.eye(n)


def test_complex_real_roundtrip():
    rng = np.random.default_rng(0)
    x = crandn(rng, 5, N)
    assert c2r(x).shape == (5, 2 * N)
    assert np.allclose(r2c(c2r(x)), x)


def test_crandn_unit_power():
    rng = np.random.default_rng(1)
    x = crandn(rng, 200000)
    assert abs(np.mean(np.abs(x) ** 2) - 1) < 0.01
    assert abs(np.mean(x.real ** 2) - 0.5) < 0.01


def test_dft_unitary_and_steering():
    F = dft_codebook(N)
    assert np.allclose(F.conj().T @ F, np.eye(N))
    # beam b collects all power of a steering vector with sin(theta) = -2b/N (wrapped to [-1, 1))
    for b in (0, 3, 9):
        s = ((-2 * b / N + 1) % 2) - 1
        g = beam_gain(F, ula_steering(N, s))
        assert np.argmax(g) == b and abs(g[b] - 1) < 1e-9


@pytest.mark.parametrize("kind", ["dft", "random"])
def test_pilot_beams_unit_norm(kind):
    W = pilot_beams(N, 8, kind, np.random.default_rng(2))
    assert W.shape == (N, 8)
    assert np.allclose(np.linalg.norm(W, axis=0), 1)


def test_noise_variance_matches_snr():
    rng = np.random.default_rng(3)
    W = pilot_beams(N, 4, "dft")
    y = measure(np.zeros((100000, N), complex), W, Ep=1.0, noise_var=0.1, rng=rng)
    assert abs(np.mean(np.abs(y) ** 2) - 0.1) < 0.002


def test_signal_power_per_pilot():
    """E|sqrt(Ep) w^H h|^2 = Ep for h ~ CN(0, I) and unit-norm w (pilot energy fixed, not normalised per sample)."""
    rng = np.random.default_rng(31)
    W = pilot_beams(N, 4, "random", rng)
    y = measure(crandn(rng, 100000, N), W, Ep=3.0, noise_var=0.0, rng=rng)
    assert np.allclose(np.mean(np.abs(y) ** 2, 0), 3.0, rtol=0.02)


def test_full_pilots_noiseless_pseudoinverse_recovers_h():
    rng = np.random.default_rng(4)
    W = pilot_beams(N, N, "dft")
    h = crandn(rng, N)
    y = measure(h, W, Ep=2.0, noise_var=0.0, rng=rng)
    A = np.sqrt(2.0) * W.conj().T
    assert np.allclose(np.linalg.pinv(A) @ y, h)


def test_lmmse_posterior_matches_monte_carlo():
    """Posterior cov = E[(h - mean)(h - mean)^H] over joint samples; 95% marginal coverage of Re parts."""
    rng = np.random.default_rng(5)
    mu, C = crandn(rng, N) * 0.3, rand_cov(rng, N, rank=4)
    W = pilot_beams(N, 4, "random", rng)
    H = sample_cn(rng, mu, C, 20000)
    Y = measure(H, W, Ep=1.0, noise_var=0.05, rng=rng)
    P = lmmse(Y[0], W, 1.0, 0.05, mu, C)[1]          # posterior cov does not depend on y
    sd = np.sqrt(np.real(np.diag(P)) / 2)
    err, cover = [], []
    for h, y in zip(H, Y):
        e = h - lmmse(y, W, 1.0, 0.05, mu, C)[0]
        err.append(np.outer(e, e.conj()))
        cover.append(np.abs(e.real) <= 1.96 * sd)
    assert np.allclose(np.mean(err, 0), P, atol=0.02 * np.abs(P).max() + 5e-3)
    assert abs(np.mean(cover) - 0.95) < 0.01


def test_gmm_single_component_equals_lmmse():
    rng = np.random.default_rng(6)
    mu, C = crandn(rng, N), rand_cov(rng, N)
    W = pilot_beams(N, 8, "dft")
    y = crandn(rng, 8)
    w, m, P = gmm_posterior(y, W, 1.0, 0.1, [1.0], [mu], [C])
    m0, P0 = lmmse(y, W, 1.0, 0.1, mu, C)
    assert np.allclose(w, [1.0]) and np.allclose(m[0], m0) and np.allclose(P[0], P0)


def test_gmm_posterior_weights_identify_state():
    """Two-component prior (two blockage states / dominant paths): posterior weights pick the true state and the
    mixture mean beats committing to the wrong component."""
    rng = np.random.default_rng(7)
    m1, m2 = ula_steering(N, 0.3) * 4, ula_steering(N, -0.5) * 4
    C1, C2 = 0.05 * np.eye(N), 0.05 * np.eye(N)
    W = pilot_beams(N, 4, "random", rng)
    n = 4000
    z = rng.random(n) < 0.5
    H = np.where(z[:, None], sample_cn(rng, m1, C1, n), sample_cn(rng, m2, C2, n))
    Y = measure(H, W, 1.0, 0.2, rng)
    correct, mse_mix, mse_wrong = 0, 0.0, 0.0
    for zi, h, y in zip(z, H, Y):
        w, m, P = gmm_posterior(y, W, 1.0, 0.2, [0.5, 0.5], [m1, m2], [C1, C2])
        correct += (w[0] > 0.5) == zi
        mse_mix += np.sum(np.abs(h - w @ m) ** 2)
        mse_wrong += np.sum(np.abs(h - m[1 if zi else 0]) ** 2)
    assert correct / n > 0.95
    assert mse_mix < mse_wrong


def test_gmm_weights_exact_bayes():
    """Posterior weights equal pi_k p(y|k) / sum, with p(y|k) = CN(A m_k, A C_k A^H + s I) evaluated independently."""
    rng = np.random.default_rng(9)
    ms = [crandn(rng, N), crandn(rng, N)]
    Cs = [rand_cov(rng, N), rand_cov(rng, N, rank=2)]
    W = pilot_beams(N, 4, "random", rng)
    y = crandn(rng, 4)
    A = W.conj().T
    lik = np.array([np.exp(log_cn_pdf(y, A @ m, A @ C @ A.conj().T + 0.3 * np.eye(4))) for m, C in zip(ms, Cs)])
    ref = np.array([0.2, 0.8]) * lik
    w, _, _ = gmm_posterior(y, W, 1.0, 0.3, [0.2, 0.8], ms, Cs)
    assert np.allclose(w, ref / ref.sum())


def test_expected_gain_matches_monte_carlo():
    rng = np.random.default_rng(8)
    mu, C = crandn(rng, N) * 0.5, rand_cov(rng, N, rank=3)
    F = dft_codebook(N)
    H = sample_cn(rng, mu, C, 100000)
    mc = beam_gain(F, H).mean(0)
    an = expected_gain(F, C + np.outer(mu, mu.conj()))
    assert np.allclose(mc, an, rtol=0.03, atol=1e-3)
    R = second_moment([0.3, 0.7], [mu, -mu], [C, C])
    assert np.allclose(R, C + np.outer(mu, mu.conj()))


def test_log_cn_pdf_normalised_1d():
    s = 0.7
    g = np.linspace(-6, 6, 601)
    X, Y = np.meshgrid(g, g)
    z = (X + 1j * Y).ravel()
    p = np.exp([log_cn_pdf(np.array([v]), np.zeros(1), np.array([[s]])) for v in z])
    assert abs(p.sum() * (g[1] - g[0]) ** 2 - 1) < 1e-3
