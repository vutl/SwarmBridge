"""Tile codec for BEV feature enhancement messages (hybrid object-feature pilot, mode bank M2/M3).

A helper's BEV feature map [C, G, G] (G = 256 cells of 0.8 m, C = 80) is cut into T x T tiles (T = 8 -> 32 x 32
tiles of 6.4 m). The sender ranks tiles with a causal local score (its own detector objectness, or feature norm as a
control) and encodes the top-n_planned tiles in rank order. Byte layout (little endian):

    header   HDR  (version, sender, grid_id, t_sense_ms, msg_id, n_planned, C, T)        17 bytes
    qparams  lo[C] fp16, step[C] fp16  (per-channel uniform 8-bit quantiser over the n_planned tiles)  4C bytes
    patch_k  tile index u16 + C*T*T uint8 values, k = 0 .. n_planned-1 in rank order        2 + C*T*T bytes each

Every prefix that ends on a patch boundary is itself a valid message, so M2 (16 tiles) is literally the first bytes
of M3 (64 tiles) and an enhancement only sends the remaining patches. The decoder uses only COMPLETE patches of
whatever bytes arrived (a truncated last patch is ignored), and only data read from the bytes.
numpy only (runs in the legacy UAV3D env and in the wireless env).
"""
import struct

import numpy as np

VERSION = 1
G, T = 256, 8
NT = G // T                      # tiles per side
HDR = struct.Struct("<BBHIIHHB")  # version, sender, grid_id, t_sense_ms, msg_id, n_planned, C, T


def patch_bytes(c=80, t=T):
    return 2 + c * t * t


def header_bytes(c=80):
    return HDR.size + 4 * c


def message_bytes(n_tiles, c=80, t=T):
    return header_bytes(c) + n_tiles * patch_bytes(c, t)


def tiles_of(feat, t=T):
    """[C, G, G] -> [NT*NT, C, t, t] with tile id = row * NT + col (row = first spatial axis)."""
    c, g, _ = feat.shape
    n = g // t
    return feat.reshape(c, n, t, n, t).transpose(1, 3, 0, 2, 4).reshape(n * n, c, t, t)


def tile_scores(cell_score, t=T):
    """[G', G'] per-cell score at any resolution dividing the tile grid -> [NT*NT] max-pooled tile score."""
    g = cell_score.shape[0]
    f = g // NT
    return cell_score.reshape(NT, f, NT, f).max(axis=(1, 3)).reshape(-1)


def encode(tiles, idx, sender=0, t_sense_ms=0, msg_id=0, grid_id=0):
    """tiles [n, C, T, T] float (already in rank order), idx [n] tile ids -> bytes (all n patches)."""
    tiles = np.asarray(tiles, np.float32)
    n, c, t, _ = tiles.shape
    lo = tiles.min(axis=(0, 2, 3)).astype(np.float16)
    hi = tiles.max(axis=(0, 2, 3))
    step = ((hi - lo.astype(np.float32)) / 255.0).astype(np.float16)
    lo32, st32 = lo.astype(np.float32), step.astype(np.float32)
    safe = np.where(st32 > 0, st32, 1.0)
    q = np.where(st32[None, :, None, None] > 0,
                 np.rint((tiles - lo32[None, :, None, None]) / safe[None, :, None, None]), 0)
    q = np.clip(q, 0, 255).astype(np.uint8)
    out = [HDR.pack(VERSION, sender, grid_id, t_sense_ms, msg_id, n, c, t), lo.tobytes(), step.tobytes()]
    for k in range(n):
        out.append(struct.pack("<H", int(idx[k])))
        out.append(q[k].tobytes())
    return b"".join(out)


def prefix(msg, k):
    """Bytes of the first k patches (a valid message on its own)."""
    _, _, _, _, _, n, c, t = HDR.unpack_from(msg, 0)
    return msg[:message_bytes(min(k, n), c, t)]


def decode(buf):
    """bytes -> (meta dict, idx [k], tiles [k, C, T, T] float32) from the COMPLETE patches present in buf."""
    if len(buf) < HDR.size:
        return None, np.zeros(0, int), None
    ver, sender, grid_id, t_ms, msg_id, n, c, t = HDR.unpack_from(buf, 0)
    if ver != VERSION:
        raise ValueError(f"unknown version {ver}")
    meta = dict(sender=sender, grid_id=grid_id, t_sense_ms=t_ms, msg_id=msg_id, n_planned=n, C=c, T=t)
    if len(buf) < header_bytes(c):
        return meta, np.zeros(0, int), np.zeros((0, c, t, t), np.float32)
    o = HDR.size
    lo = np.frombuffer(buf, np.float16, c, o).astype(np.float32); o += 2 * c
    st = np.frombuffer(buf, np.float16, c, o).astype(np.float32); o += 2 * c
    pb = patch_bytes(c, t)
    k = min(n, (len(buf) - o) // pb)
    idx = np.zeros(k, int)
    q = np.zeros((k, c, t, t), np.uint8)
    for i in range(k):
        idx[i] = struct.unpack_from("<H", buf, o)[0]
        q[i] = np.frombuffer(buf, np.uint8, c * t * t, o + 2).reshape(c, t, t)
        o += pb
    tiles = q.astype(np.float32) * st[None, :, None, None] + lo[None, :, None, None]
    return meta, idx, tiles


def scatter(idx, tiles, c=80, g=G, t=T):
    """Place decoded tiles into a dense [C, G, G] map and a [G, G] validity mask (float32)."""
    feat = np.zeros((c, g, g), np.float32)
    mask = np.zeros((g, g), np.float32)
    n = g // t
    for i, tile in zip(idx, tiles):
        r, col = divmod(int(i), n)
        feat[:, r * t:(r + 1) * t, col * t:(col + 1) * t] = tile
        mask[r * t:(r + 1) * t, col * t:(col + 1) * t] = 1.0
    return feat, mask
