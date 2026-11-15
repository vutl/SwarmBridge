"""Synthetic tests for B-E3 (b2_pipeline.py): codec shapes, VICReg behaviour, Rep round trip with random weights."""
import numpy as np
import torch

import b2_pipeline as P


def test_codec_shapes():
    enc, dec = P.make_codec(32)
    x = torch.randn(2, P.DIM, P.GH, P.GW)
    z = enc(x)
    assert z.shape == (2, 32, P.GH, P.GW) and dec(z).shape == x.shape


def test_vicreg_penalises_collapse():
    collapsed = torch.zeros(4, 8, P.GH, P.GW) + 0.01 * torch.randn(4, 8, P.GH, P.GW)
    spread = torch.randn(4, 8, P.GH, P.GW)
    assert P.vicreg(collapsed)[0] > P.vicreg(spread)[0]


def test_rep_codes_tokens_round_trip(tmp_path):
    d = 16
    enc, dec = P.make_codec(d)
    ck = dict(enc=enc.state_dict(), dec=dec.state_dict(), pred=None, zmu=np.zeros(d, np.float32),
              zsd=np.ones(d, np.float32), mu=np.zeros(P.DIM, np.float32), sd_tok=np.ones(P.DIM, np.float32))
    torch.save(ck, tmp_path / "m.pt")
    rep = P.Rep(tmp_path / "m.pt", d)
    X = np.random.default_rng(0).normal(size=(3, P.N_TOK, P.DIM)).astype(np.float32)
    Z = rep.codes(X)
    assert Z.shape == (3, P.N_TOK, d)
    with torch.no_grad():
        ref = dec(enc(torch.from_numpy(X.reshape(3, P.GH, P.GW, P.DIM).transpose(0, 3, 1, 2).copy()))).numpy()
    assert np.allclose(rep.tokens(Z), ref.transpose(0, 2, 3, 1).reshape(3, P.N_TOK, P.DIM), atol=1e-4)


def test_rate_surrogate_tracks_innovation_size():
    z = torch.randn(4, 8, P.GH, P.GW)
    small = P.rate_surrogate(z, z + 0.01 * torch.randn_like(z))
    large = P.rate_surrogate(z, z + 1.0 * torch.randn_like(z))
    assert 0 <= float(small) < float(large)


def test_change_masks():
    T = 3
    Y = np.zeros((T, P.N_TOK, 4), np.uint8)
    Y[1, 5, 0] = 1                       # object appears at patch 5 in frame 1
    Y[2, 5, 0] = 1
    M = np.ones(T * P.N_TOK, bool)
    ch, ev = P.change_masks(Y.reshape(-1, 4), M, T)
    ch = ch.reshape(T, P.N_TOK); ev = ev.reshape(T, P.N_TOK)
    assert ch[1, 5] and ch.sum() == 1 and ev[1].all() and not ev[0].any() and not ev[2].any()
