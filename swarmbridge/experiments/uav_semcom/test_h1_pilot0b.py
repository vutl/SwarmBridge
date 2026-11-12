"""Synthetic tests for the 0B-TUNE pilot: MDMT adapter (split guard, rights guard, GT isolation, frame indexing),
crop geometry, candidates, token quantisation, budget packing, T2 oracle rules and costs, BS fusion, scorers,
screening measurements and an end-to-end toy. No real dataset image is used."""
import io
import itertools
import zipfile

import numpy as np
import pytest
from PIL import Image

import h1_gate0a as G
import h1_gate0a_audit as AU
import h1_pilot0b as P
import h1_pilot0b_eval as E
import mdmt_adapter as M

ROOT = "Multi-Drone-Multi-Object-Detection-and-Tracking"


# ================================================================ adapter
def _jpg(color):
    b = io.BytesIO(); Image.new("RGB", (32, 16), color).save(b, "JPEG", quality=95); return b.getvalue()


def _xml(tracks):
    s = '<annotations count="1">\n'
    for tid, lab, boxes in tracks:
        s += f'\t<track id="{tid}" label="{lab}">\n'
        for f, (x1, y1, x2, y2), out in boxes:
            s += (f'\t\t<box frame="{f}" occluded="0" outside="{out}" xbr="{x2}" xtl="{x1}" ybr="{y2}" '
                  f'ytl="{y1}" />\n')
        s += '\t</track>\n'
    return s + '</annotations>\n'


@pytest.fixture
def fakezip(tmp_path):
    p = tmp_path / "mdmt.zip"
    with zipfile.ZipFile(p, "w") as z:
        # test seq 26: both views, 3 frames; image k colour encodes the file index
        for v in (1, 2):
            for k in (1, 2, 3):
                z.writestr(f"{ROOT}/test/{v}/26-{v}/{k:08d}.jpg", _jpg((40 * k, 0, 0)))
            z.writestr(f"{ROOT}/new_xml/{v}/26-{v}.xml",
                       _xml([(0, "car", [(0, (1, 2, 5, 6), 0), (1, (1, 2, 5, 6), 1)]),
                             (3, "person", [(2, (0, 0, 4, 9), 0)])]))
        # test seq 52: files numbered from 191 (labels still 0-based by rank); view 2 colours differ
        for v in (1, 2):
            for k in (191, 192, 193):
                z.writestr(f"{ROOT}/test/{v}/52-{v}/{k:08d}.jpg", _jpg((40 * (k - 190), 60 * v, 0)))
            z.writestr(f"{ROOT}/new_xml/{v}/52-{v}.xml",
                       _xml([(0, "car", [(f, (1, 2, 5, 6), 0) for f in range(3)])]))
        # test seq 55: view 1 has one image more than label frames -> not aligned
        for v in (1, 2):
            for k in (550, 551):
                z.writestr(f"{ROOT}/test/{v}/55-{v}/{k:08d}.jpg", _jpg((0, 0, 0)))
            z.writestr(f"{ROOT}/new_xml/{v}/55-{v}.xml",
                       _xml([(0, "car", [(f, (1, 2, 5, 6), 0) for f in range(1 if v == 1 else 2)])]))
        # test seq 71: drone 1 only
        z.writestr(f"{ROOT}/test/1/71-1/00000001.jpg", _jpg((0, 0, 0)))
        for v in (1, 2):
            z.writestr(f"{ROOT}/new_xml/{v}/71-{v}.xml", _xml([]))
        # val seq 22: both views (must stay closed)
        for v in (1, 2):
            z.writestr(f"{ROOT}/val/{v}/22-{v}/00000001.jpg", _jpg((0, 0, 0)))
            z.writestr(f"{ROOT}/new_xml/{v}/22-{v}.xml", _xml([(5, "car", [(0, (1, 1, 3, 3), 0)])]))
        z.writestr(f"{ROOT}/checkpoints/model.pth", b"never-read")
        z.writestr(f"{ROOT}/configs/evil.py", b"raise SystemExit('executed')")
    return p


def test_adapter_pairs_and_frames(fakezip):
    ds = M.MDMT(fakezip, rights_note="synthetic")
    assert ds.pairs("test") == [26, 52, 55] and ds.pairs("val") == [22] and ds.pairs("train") == []
    assert ds.frames(26) == [0, 1, 2] and ds.frames(52) == [0, 1, 2]
    assert ds.epochs(26, step=2, max_epochs=60) == [0, 2]
    assert ds.aligned(26) and ds.aligned(52) and not ds.aligned(55)
    assert sorted(ds.pilot_order("test")) == [26, 52]


def test_adapter_xml_frame_maps_to_file_rank(fakezip):
    ds = M.MDMT(fakezip, rights_note="synthetic")
    for f in (0, 1, 2):
        assert abs(int(ds.image(26, 1, f)[0, 0, 0]) - 40 * (f + 1)) <= 4
        im1, im2 = ds.image(52, 1, f), ds.image(52, 2, f)            # file 191 + f, not file f + 1
        assert abs(int(im1[0, 0, 0]) - 40 * (f + 1)) <= 4 and abs(int(im2[0, 0, 1]) - 120) <= 4
    assert sorted(ds.labels(52, 1)) == [0, 1, 2]


def test_adapter_parse_skips_outside_and_keeps_ids(fakezip):
    g = M.MDMT(fakezip).labels(26, 2)
    assert g[0] == {0: (1.0, 2.0, 5.0, 6.0, M.CLASSES.index("car"))}
    assert 1 not in g                                         # outside=1 dropped
    assert g[2] == {3: (0.0, 0.0, 4.0, 9.0, M.CLASSES.index("person"))}


def test_adapter_unknown_label_raises():
    with pytest.raises(ValueError):
        M.parse_cvat(_xml([(0, "tank", [(0, (0, 0, 1, 1), 0)])]))


def test_adapter_val_closed_and_rights_required(fakezip):
    ds = M.MDMT(fakezip)                                      # no rights note
    with pytest.raises(M.AccessError):
        ds.image(26, 1, 0)                                    # pixels need the rights note
    ds.labels(26, 1)                                          # labels of an opened split are fine
    with pytest.raises(M.AccessError):
        ds.labels(22, 1)                                      # val is closed
    ds2 = M.MDMT(fakezip, rights_note="x")
    with pytest.raises(M.AccessError):
        ds2.image(22, 1, 0)
    assert ds2.provenance()["rights_note"] == "x"


def test_adapter_never_opens_non_image_members(fakezip, monkeypatch):
    ds = M.MDMT(fakezip, rights_note="synthetic")
    opened = []
    orig = zipfile.ZipFile.read
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda self, n, *a: opened.append(n) or orig(self, n, *a))
    ds.image(26, 1, 0); ds.labels(26, 1); ds.pairs("test"); ds.frames(26)
    assert opened and all(n.endswith((".jpg", ".xml")) for n in opened)


def test_hold_rule_is_registered_hash():
    from hashlib import blake2b
    for s in range(1, 100):
        assert M.is_hold(s) == (int(blake2b(str(s).encode(), digest_size=8).hexdigest(), 16) % 4 == 0)


# ================================================================ geometry and candidates
def test_crop_window_scale_min_and_clip():
    assert P.crop_window((100, 100, 140, 120), 1920, 1080) == (88, 78, 152, 142)   # 60 x 30 -> 64 x 64
    assert P.crop_window((100, 100, 200, 120), 1920, 1080) == (75, 78, 225, 142)   # 150 x 30 -> 150 x 64
    w = P.crop_window((0, 0, 10, 10), 1920, 1080)
    assert w == (0, 0, 64, 64)                                                       # shifted inside
    w = P.crop_window((1900, 1070, 1920, 1080), 1920, 1080)
    assert w[2] == 1920 and w[3] == 1080 and w[2] - w[0] >= 64 and w[3] - w[1] >= 64
    w = P.crop_window((0, 0, 50, 50), 40, 30)                                       # image smaller than min
    assert w == (0, 0, 40, 30)


def test_crop_window_contains_box_and_scales():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x1, y1 = rng.uniform(0, 1800), rng.uniform(0, 1000)
        b = (x1, y1, x1 + rng.uniform(2, 110), y1 + rng.uniform(2, 70))
        w = P.crop_window(b, 1920, 1080)
        assert w[0] <= b[0] and w[1] <= b[1] and w[2] >= min(b[2], 1920) and w[3] >= min(b[3], 1080)
        assert 0 <= w[0] < w[2] <= 1920 and 0 <= w[1] < w[3] <= 1080


def test_merge_windows():
    wins, confs = P.merge_windows([(0, 0, 100, 100), (10, 0, 110, 100), (500, 500, 600, 600)], [0.1, 0.2, 0.15])
    assert sorted(wins) == [(0, 0, 110, 100), (500, 500, 600, 600)]
    assert dict(zip(wins, confs))[(0, 0, 110, 100)] == 0.2
    w2, _ = P.merge_windows([(0, 0, 100, 100), (60, 0, 160, 100)], [0.1, 0.1])     # IoU 0.25: kept
    assert len(w2) == 2


def test_p1_candidates_only_uncertain_and_sorted():
    d = P.dets([(100, 100, 120, 120), (400, 400, 420, 420), (800, 800, 820, 820), (900, 100, 920, 120)],
               [1, 1, 0, 2], [0.9, 0.2, 0.04, 0.1])
    c = P.p1_candidates(d, 1920, 1080)
    assert [x["conf"] for x in c] == [0.2, 0.1]
    assert c[0]["win"] == P.crop_window(d["b"][1], 1920, 1080)


def test_p2_tiles_cover_image_and_select_uncertain():
    t = P.tiles(1920, 1080)
    assert t[-1][2] == 1920 and t[-1][3] == 1080 and all(x[2] - x[0] == 320 for x in t)
    d = P.dets([(10, 10, 20, 20), (1000, 500, 1010, 510)], [1, 1], [0.1, 0.9])
    c = P.p2_candidates(d, 1920, 1080)
    assert len(c) == 1 and c[0]["win"] == (0, 0, 320, 320)


def test_to_global_roundtrip():
    d = P.dets([(1, 2, 3, 4)], [0], [0.5])
    g = P.to_global(d, (100, 50, 200, 150))
    assert np.allclose(g["b"], [[101, 52, 103, 54]])


def test_map_visdrone():
    d = P.map_visdrone([(0, 0, 1, 1)] * 4, [1, 4, 7, 9], [0.5, 0.6, 0.7, 0.8])
    assert d["c"].tolist() == [0, 1, 3] and d["s"].tolist() == [0.5, 0.6, 0.8]


# ================================================================ tokens and packing
def test_token_quantisation_error_bounded():
    rng = np.random.default_rng(1)
    b = rng.uniform(0, 1000, (100, 2)); b = np.concatenate([b, b + rng.uniform(5, 80, (100, 2))], 1)
    d = P.dets(b, np.zeros(100), rng.uniform(0.05, 1, 100))
    q = P.quant_tokens(d, 1920, 1080)
    assert np.abs(q["b"] - d["b"]).max() <= 1.5 * 1920 / P.QGRID
    assert np.all(np.abs(q["s"] - d["s"]) <= 1 / 16 + 1e-9)


def test_pack_headers_and_skip():
    one = P.msg_cost(100)
    assert one == 100 + G.MSG_HDR + G.APP_HDR
    sel, pay = P.pack([100, 5000, 50], one + 50)
    assert sel == [0, 2] and pay == 150
    assert P.pack([100], one - 1) == ([], 0)
    big = [G.MTU] * 3                                          # multi-MTU payload pays one APP header per MTU
    sel, pay = P.pack(big, 10 ** 9)
    assert P.msg_cost(pay) == pay + G.MSG_HDR + G.APP_HDR * -(-(G.MSG_HDR + pay) // G.MTU)


def _cand(win, conf, bits=1000, bs=None):
    return dict(win=win, conf=conf, bits={35: bits, 50: 2 * bits}, enc_ms={35: 5.0, 50: 6.0},
                bs={35: bs if bs is not None else P.dets(), 50: bs if bs is not None else P.dets()})


def _rec(det, p1=(), p2=(), frame_bits=10 ** 6, frame_bs=None, W=1920, H=1080):
    return dict(W=W, H=H, det=det, det_ms=50.0, cands=dict(P1=list(p1), P2=list(p2)),
                frame=dict(bits=frame_bits, enc_ms=30.0, bs=frame_bs if frame_bs is not None else P.dets()))


def test_condition_item_orders():
    d = P.dets([(0, 0, 10, 10)] * 4, [0] * 4, [0.3, 0.9, 0.1, 0.2])
    r = _rec(d, p1=[_cand((0, 0, 64, 64), 0.2)], p2=[_cand((0, 0, 320, 320), 0.2)])
    assert [k for _, k, _ in P.uav_items(r, "T0", 35)] == [1, 0]
    assert [(t, k) for t, k, _ in P.uav_items(r, "T0+", 35)] == [("tok", 1), ("tok", 0), ("tok", 3), ("tok", 2)]
    assert [t for t, _, _ in P.uav_items(r, "T1", 35)] == ["tok", "tok", "P1"]
    assert P.uav_items(r, "T1", 50)[-1][2] == 2000
    assert [t for t, _, _ in P.uav_items(r, "T1P2", 35)] == ["tok", "tok", "P2"]


def test_F_sends_frame_only_if_it_fits_else_T0():
    d = P.dets([(0, 0, 10, 10)], [0], [0.9])
    r = _rec(d, frame_bits=5000)
    s = P.send(r, "F", 35, P.msg_cost(5000))
    assert s["frame"] and s["items"] == [("frame", 0, 5000)]
    s = P.send(r, "F", 35, P.msg_cost(5000) - 1)
    assert not s["frame"] and s["items"] == [("tok", 0, P.TOK_BITS)]


# ================================================================ T2 oracle
def _gt(*boxes):
    return {i: (*b, 1) for i, b in boxes}


def test_t2_drop_rules():
    # view 1: object 7 proposed as candidate; object 8 carried by a strong token
    g1 = _gt((7, (100, 100, 120, 120)), (8, (500, 500, 520, 520)))
    r1 = _rec(P.dets([(500, 500, 520, 520)], [1], [0.9]), p1=[_cand(P.crop_window((100, 100, 120, 120), 1920, 1080), .2)])
    # view 2: candidates on 7 (drop: proposed), 8 (drop: token), 9 (keep: only view 2), and an FP window (keep)
    g2 = _gt((7, (300, 300, 320, 320)), (8, (700, 700, 720, 720)), (9, (900, 100, 920, 120)))
    w = [P.crop_window(b, 1920, 1080) for b in [(300, 300, 320, 320), (700, 700, 720, 720), (900, 100, 920, 120)]]
    r2 = _rec(P.dets(), p1=[_cand(w[0], .2), _cand(w[1], .2), _cand(w[2], .2), _cand((1500, 900, 1564, 964), .2)])
    assert P.t2_drop(r1, r2, g1, g2) == frozenset({0, 1})


def test_t2_weak_token_does_not_count_and_class_mismatch_does_not_count():
    g1 = _gt((8, (500, 500, 520, 520)))
    r1 = _rec(P.dets([(500, 500, 520, 520)], [1], [0.2]))                  # weak: not a strong token
    g2 = _gt((8, (700, 700, 720, 720)))
    r2 = _rec(P.dets(), p1=[_cand(P.crop_window((700, 700, 720, 720), 1920, 1080), .2)])
    assert P.t2_drop(r1, r2, g1, g2) == frozenset()
    r1b = _rec(P.dets([(500, 500, 520, 520)], [0], [0.9]))                 # strong but wrong class
    assert P.t2_drop(r1b, r2, g1, g2) == frozenset()


def test_t2_charges_both_manifests_and_bitmap():
    r1 = _rec(P.dets([(0, 0, 9, 9), (20, 20, 29, 29)], [1, 1], [0.9, 0.1]), p1=[_cand((0, 0, 64, 64), .1)])
    r2 = _rec(P.dets(), p1=[_cand((0, 0, 64, 64), .1)] * 3)
    ul, dl = P.t2_coord_costs(r1, r2)
    assert (ul, dl) == AU.coord_costs(2, 3, P.MAN_BITS, bitmaps_for=("B",))
    assert ul == P.msg_cost(48) + P.msg_cost(72) and dl == 3 + G.DL_HDR
    e = P.run_epoch(r1, r2, {}, {}, "T2", 35, 10 ** 7, 0.5)
    assert e["ul"] == e["sent"][0]["ul"] + e["sent"][1]["ul"] + ul and e["dl"] == dl
    e = P.run_epoch(r1, r2, {}, {}, "T2", 35, ul + dl, 0.5)                 # nothing left for payload
    assert e["sent"][0]["items"] == [] and e["sent"][1]["items"] == []


def test_t2_with_ample_budget_never_lowers_unique_tp():
    rng = np.random.default_rng(3)
    for _ in range(20):
        boxes = {i: (x, y, x + 20, y + 20) for i, (x, y) in enumerate(rng.uniform(0, 1500, (6, 2)))}
        g1 = {i: (*b, 1) for i, b in boxes.items() if rng.random() < .7}
        g2 = {i: (*b, 1) for i, b in boxes.items() if rng.random() < .7}
        recs = []
        for g in (g1, g2):
            p1 = [_cand(P.crop_window(b[:4], 1920, 1080), .2, bs=P.dets([b[:4]], [1], [.9])) for b in g.values()]
            recs.append(_rec(P.dets(), p1=p1))
        u = {}
        for cond in ("T1", "T2"):
            e = P.run_epoch(recs[0], recs[1], g1, g2, cond, 35, 10 ** 8, .5)
            v = [P.view_counts(e["decl"][k], g, .05) for k, g in enumerate((g1, g2))]
            u[cond] = P.unique_counts(v[0], v[1], g1, g2)["tp"]
        assert u["T2"] == u["T1"] == len(set(g1) | set(g2))


# ================================================================ BS fusion and scoring
def test_nms_classwise():
    d = P.dets([(0, 0, 10, 10), (1, 0, 11, 10), (0, 0, 10, 10)], [1, 1, 2], [0.5, 0.9, 0.3])
    k = P.nms_classwise(d)
    assert sorted(zip(k["c"].tolist(), k["s"].tolist())) == [(1, 0.9), (2, 0.3)]


def test_bs_decode_fuses_token_and_crop_dets():
    d = P.dets([(100, 100, 120, 120)], [1], [0.9])
    crop_det = P.dets([(101, 100, 121, 120), (300, 300, 320, 320)], [1, 0], [0.6, 0.4])
    r = _rec(d, p1=[_cand((0, 0, 400, 400), .2, bs=crop_det)])
    s = P.send(r, "T1", 35, 10 ** 6)
    out = P.bs_decode(r, s, 35)
    assert len(out["s"]) == 2 and set(out["c"].tolist()) == {0, 1}


def _brute_match(d, gt, thr):
    ids = sorted(gt); best = 0
    n = len(d["s"])
    for perm in itertools.permutations(range(max(n, len(ids))), len(ids)):
        k = sum(1 for g, di in zip(ids, perm) if di < n and d["c"][di] == gt[g][4]
                and P.iou_matrix([d["b"][di]], [gt[g][:4]])[0, 0] >= thr)
        best = max(best, k)
    return best


@pytest.mark.parametrize("seed", range(15))
def test_match_is_maximum_cardinality(seed):
    rng = np.random.default_rng(seed)
    gt = {i: (x, y, x + 20, y + 20, int(rng.integers(0, 2))) for i, (x, y) in enumerate(rng.uniform(0, 60, (4, 2)))}
    b = rng.uniform(0, 60, (5, 2)); b = np.concatenate([b, b + 20], 1)
    d = P.dets(b, rng.integers(0, 2, 5), rng.uniform(0, 1, 5))
    assert len(P.match(d, gt)) == _brute_match(d, gt, P.MATCH_IOU)


def test_unique_counts_object_in_two_views_counts_once():
    g1 = _gt((1, (0, 0, 10, 10)), (2, (50, 50, 60, 60)))
    g2 = _gt((1, (100, 100, 110, 110)), (3, (200, 200, 210, 210)))
    d1 = P.dets([(0, 0, 10, 10), (300, 300, 310, 310)], [1, 1], [.9, .9])   # TP id1 + FP
    d2 = P.dets([(100, 100, 110, 110)], [1], [.9])                         # TP id1 again
    v1, v2 = P.view_counts(d1, g1, .5), P.view_counts(d2, g2, .5)
    assert (v1["tp"], v1["fp"], v1["fn"]) == (1, 1, 1) and (v2["tp"], v2["fp"], v2["fn"]) == (1, 0, 1)
    u = P.unique_counts(v1, v2, g1, g2)
    assert u == dict(tp=1, fp=1, fn=2)
    assert P.f1(u) == pytest.approx(2 / 5)


def test_view_counts_threshold_and_class_agnostic():
    g = _gt((1, (0, 0, 10, 10)))
    d = P.dets([(0, 0, 10, 10)], [0], [0.3])                               # wrong class
    assert P.view_counts(d, g, .25)["tp"] == 0 and P.view_counts(d, g, .25, False)["tp"] == 1
    assert P.view_counts(d, g, .4, False)["tp"] == 0


# ================================================================ screening
def test_c1_c2_and_false_candidates():
    g = _gt((1, (100, 100, 120, 120)), (2, (500, 500, 520, 520)), (3, (900, 900, 920, 920)))
    d = P.dets([(100, 100, 120, 120), (500, 500, 520, 520)], [1, 1], [0.9, 0.1])
    r = _rec(d, p1=[_cand(P.crop_window((500, 500, 520, 520), 1920, 1080), .1, bits=700),
                    _cand((1500, 100, 1564, 164), .08, bits=300)])
    c = P.c1_c2(r, g)
    assert c == dict(n_gt=3, missed=2, missed_covered=1, n_cand=2, false_cand=1, false_bits=300)


def test_c4_redundancy_both_directions():
    g1 = _gt((1, (100, 100, 120, 120)), (2, (300, 300, 320, 320)))
    g2 = _gt((1, (700, 700, 720, 720)), (2, (900, 300, 920, 320)))
    r1 = _rec(P.dets(), p1=[_cand(P.crop_window(g1[1][:4], 1920, 1080), .1),
                            _cand(P.crop_window(g1[2][:4], 1920, 1080), .1)])
    r2 = _rec(P.dets([g2[2][:4]], [1], [0.9]), p1=[_cand(P.crop_window(g2[1][:4], 1920, 1080), .1)])
    c = P.c4_redundancy(r1, r2, g1, g2)
    assert c == dict(matched=3, redundant=3)


def test_cpu_ms_all_vs_sent():
    r = _rec(P.dets([(0, 0, 9, 9)], [0], [0.1]), p1=[_cand((0, 0, 64, 64), .1), _cand((100, 0, 164, 64), .1)])
    assert P.cpu_ms(r, 35) == 60.0
    s = P.send(r, "T1", 35, P.msg_cost(1000))
    assert P.cpu_ms(r, 35, s) == 55.0


def test_budget_levels_rule():
    r = _rec(P.dets([(0, 0, 9, 9)], [0], [0.9]), p1=[_cand((0, 0, 64, 64), .1, bits=945)], frame_bits=50000)
    lv = P.budget_levels([(r, r, {}, {})])
    per = P.msg_cost(P.TOK_BITS + 945)
    assert lv["stream_mean"] == 2 * per and lv["levels"] == [round(f * 2 * per) for f in (.25, .5, 1.0)]
    assert lv["frame_level"] == 2 * P.msg_cost(50000)


# ================================================================ end-to-end toy
def _toy_seq(n=4):
    """each view: object A detected strongly, object B only weakly (missed by tokens); the BS detector on B's crop
    recovers it. A crop over empty road is a false candidate that yields an FP at the BS."""
    pairs = []
    for t in range(n):
        A, Bx = (100 + t, 100, 130 + t, 120), (600, 400 + t, 620, 420 + t)
        g = {1: (*A, 1), 2: (*Bx, 0)}
        det = P.dets([A, Bx, (1500, 900, 1510, 910)], [1, 0, 1], [0.9, 0.12, 0.06])
        cands = P.p1_candidates(det, 1920, 1080)
        for c in cands:
            hit = c["win"][0] <= 610 < c["win"][2]
            bs = P.dets([Bx], [0], [0.7]) if hit else P.dets([(1500, 900, 1510, 910)], [1], [0.3])
            c.update(bits={35: 3000, 50: 6000}, enc_ms={35: 4.0, 50: 5.0}, bs={35: bs, 50: bs})
        r = dict(W=1920, H=1080, det=det, det_ms=40.0, cands=dict(P1=cands, P2=[]),
                 frame=dict(bits=400000, enc_ms=80.0, bs=P.dets([A, Bx], [1, 0], [.8, .8])))
        pairs.append((r, r, g, g))
    return pairs


def test_end_to_end_toy_crop_recovers_missed_object():
    res = E.evaluate({1: _toy_seq(), 2: _toy_seq()}, qs=(35,), procs=1)
    get = lambda c, q, B: next(r for r in res["table"] if r["cond"] == c and r["q"] == q and r["B"] == B)
    Bmax = res["budget"]["levels"][-1]
    assert get("T0", None, Bmax)["r_view"] == pytest.approx(0.5)
    assert get("T1", 35, Bmax)["f1_view"] > get("T0", None, Bmax)["f1_view"]
    assert get("T1", 35, Bmax)["f1_view"] > get("T0+", None, Bmax)["f1_view"]
    assert get("F", None, res["budget"]["frame_level"])["f1_view"] == pytest.approx(1.0)
    cr = res["criteria"]
    assert cr["C1_pass"] and cr["C2_pass"] and cr["C3"]["pass"] and cr["C5_pass"]
    assert res["screening"]["C1"]["miss_rate"] == pytest.approx(0.5)
    assert res["screening"]["C2"]["false_cand_per_view_epoch"] == pytest.approx(1.0)
    # same scene in both views -> every matched candidate is redundant; T2 drops UAV2's copy
    assert res["screening"]["C4_redundancy"]["frac"] == pytest.approx(1.0)


def test_pipeline_uav_side_never_sees_gt():
    import inspect
    import h1_pilot0b_precompute as PC
    src = inspect.getsource(PC)
    assert "labels(" not in src and "gt" not in inspect.signature(PC.Pipeline.record).parameters


@pytest.mark.parametrize("seed", range(10))
def test_vectorised_cover_matches_loop(seed):
    rng = np.random.default_rng(seed)
    gt = {int(i): (x, y, x + w, y + h, 1) for i, (x, y, w, h) in
          zip(rng.permutation(50)[:12], np.c_[rng.uniform(0, 500, (12, 2)), rng.uniform(5, 60, (12, 2))])}
    wins = [P.crop_window((x, y, x + 20, y + 20), 640, 640) for x, y in rng.uniform(0, 520, (15, 2))]
    for w, g in zip(wins, P.cand_gt_all(wins, gt)):
        cov = []
        for i, (x1, y1, x2, y2, _) in gt.items():
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            v = P.iou_matrix([w], [(x1, y1, x2, y2)])[0, 0]
            if (w[0] <= cx < w[2] and w[1] <= cy < w[3]) or v >= P.PROP_IOU:
                cov.append((-v, i))
        assert g == (min(cov)[1] if cov else None)
        assert P.covered_ids(w, gt) == {i for _, i in cov}
