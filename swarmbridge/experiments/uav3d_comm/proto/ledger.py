"""Causal packet ledger (protocol v1, section 2; Codex review point 1, option b).

Model
-----
Every helper j in 1..4 *sends* one packet per frame s (0-based index inside a scene, 1 frame = 0.2 s simulated time).
The packet is lost with probability p (Bernoulli, independent per link and frame); otherwise it *arrives* at frame
s + d, with d drawn from the scenario's delay distribution (independent per link and frame). Delay and loss come
from separate random streams keyed by (seed, scene, sender), so changing p never changes the delays and vice versa.

Receiver rule at frame t (never looks at anything with arrival > t):
  * mode "latest":       among the packets of sender j that arrived by t (arrival <= t), use the one with the newest
                         send frame (out-of-order arrivals of older packets are ignored), if its age t - s <= max_age;
                         otherwise no message from j.  The cache is reset at every scene start, so frames before the
                         first arrival have no message (cold start).
  * mode "arrival_only": use only a packet that arrives exactly at t (newest send frame if several); no holding.

Ledger file = scenario definition + one record per sent packet + the per-(scene, frame, sender) usage table, with a
sha256 over its canonical JSON so that every method can be shown to have used the same packets.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCENARIOS = {
    # name: (delay distribution {d: prob}, loss p)
    "sync": ({0: 1.0}, 0.0),
    "fix1": ({1: 1.0}, 0.0),
    "fix2": ({2: 1.0}, 0.0),
    "rand": ({0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}, 0.0),
    "loss0.2": ({0: 1.0}, 0.2),
    "loss0.5": ({0: 1.0}, 0.5),
    "rand_loss0.2": ({0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}, 0.2),
}
STOCHASTIC = {"rand", "loss0.2", "loss0.5", "rand_loss0.2"}
SENDERS = (1, 2, 3, 4)


def _rng(seed, scene, sender, stream):
    key = hashlib.sha256(f"{seed}|{scene}|{sender}|{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little"))


def make_packets(scene, n_frames, delay_dist, loss, seed):
    ds = np.array(sorted(delay_dist), int)
    ps = np.array([delay_dist[d] for d in ds], float)
    assert abs(ps.sum() - 1) < 1e-9
    out = []
    for j in SENDERS:
        rd, rl = _rng(seed, scene, j, "delay"), _rng(seed, scene, j, "loss")
        delays = rd.choice(ds, size=n_frames, p=ps)
        drops = rl.random(n_frames) < loss
        for s in range(n_frames):
            out.append({"scene": scene, "sender": j, "send_frame": s, "seq": s, "drop": bool(drops[s]),
                        "delay": int(delays[s]), "arrival_frame": None if drops[s] else int(s + delays[s])})
    return out


def usage(packets, n_frames, mode="latest", max_age=3):
    """-> {sender: [ (send_frame or None, age or None) for t in 0..n_frames-1 ]} for one scene."""
    by = {j: [p for p in packets if p["sender"] == j and not p["drop"]] for j in SENDERS}
    out = {}
    for j, ps in by.items():
        row = []
        for t in range(n_frames):
            if mode == "latest":
                cand = [p["send_frame"] for p in ps if p["arrival_frame"] <= t]
            elif mode == "arrival_only":
                cand = [p["send_frame"] for p in ps if p["arrival_frame"] == t]
            else:
                raise ValueError(mode)
            s = max(cand) if cand else None
            if s is not None and t - s > max_age:
                s = None
            row.append((s, None if s is None else t - s))
        out[j] = row
    return out


def build(scene_frames, scenario, seed, mode="latest", max_age=3):
    """scene_frames: {scene_name: [token_0, ..., token_{T-1}]} in time order."""
    dist, loss = SCENARIOS[scenario]
    led = {"scenario": scenario, "delay_dist": {str(k): v for k, v in dist.items()}, "loss": loss, "seed": seed,
           "mode": mode, "max_age": max_age, "frame_dt_s": 0.2, "scenes": {}}
    for sc in sorted(scene_frames):
        toks = scene_frames[sc]
        pk = make_packets(sc, len(toks), dist, loss, seed)
        use = usage(pk, len(toks), mode, max_age)
        led["scenes"][sc] = {
            "tokens": toks, "packets": pk,
            # per receiver frame t: for each sender, the token whose message is used (or null) and its age
            "use": [{str(j): {"src_token": None if use[j][t][0] is None else toks[use[j][t][0]],
                              "send_frame": use[j][t][0], "age": use[j][t][1]} for j in SENDERS}
                    for t in range(len(toks))],
        }
    led["sha256"] = digest(led)
    return led


def digest(led):
    body = {k: v for k, v in led.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path):
    led = json.loads(Path(path).read_text())
    assert led["sha256"] == digest(led), f"ledger hash mismatch: {path}"
    return led


def lookup(led):
    """-> {receiver_token: {sender: (src_token or None, age or None)}}."""
    out = {}
    for sc in led["scenes"].values():
        for t, tok in enumerate(sc["tokens"]):
            out[tok] = {int(j): (u["src_token"], u["age"]) for j, u in sc["use"][t].items()}
    return out


def tx_packets(led):
    """-> {(scene, sender, send_frame): dropped?} : every packet that was *transmitted* (bits are spent even if lost)."""
    return {(p["scene"], p["sender"], p["send_frame"]): p["drop"]
            for sc in led["scenes"].values() for p in sc["packets"]}


def scene_frames_from_db(db, scene_names):
    """MiniDB (evaldet) -> {scene: [tokens in time order]}."""
    out = {}
    for sc_tok, sc in db.scenes.items():
        if sc["name"] not in set(scene_names):
            continue
        toks, t = [], sc["first_sample_token"]
        while t:
            toks.append(t)
            t = db.samples[t]["next"]
        out[sc["name"]] = toks
    return out
