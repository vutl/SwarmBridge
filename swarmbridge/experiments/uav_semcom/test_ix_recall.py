"""UAVDT vehicle-recall definitions (prereg §3), synthetic fixtures only (no detector, no UAVDT image)."""
import numpy as np

import a5_fidelity as AF
import ix_profile as XP

CAR, VAN, TRUCK, BUS, PED = 3, 4, 5, 8, 0          # VisDrone detector classes


def dets(*items):
    """items: (box, score, cls) in any order -> arrays sorted by descending score (as a5_fidelity.detect)."""
    items = sorted(items, key=lambda x: -x[1])
    return (np.array([b for b, _, _ in items], float).reshape(-1, 4), np.array([s for _, s, _ in items], float),
            np.array([c for _, _, c in items], int))


G = np.array([[0, 0, 10, 10], [100, 100, 120, 110], [200, 200, 230, 215]], float)
GC = np.array([0, 1, 2])                           # car, truck, bus
NO_IGN = np.zeros((0, 4))


def test_exact_matches_all_variants():
    r = XP.recall_variants(*dets(([0, 0, 10, 10], .9, CAR), ([100, 100, 120, 110], .8, TRUCK),
                                 ([200, 200, 230, 215], .7, BUS)), G, GC, NO_IGN)
    assert r == {"veh": (3, 3, 3), "cls_vancar": (3, 3, 3), "cls_car": (3, 3, 3)}


def test_one_to_one_matching():
    """two detections on one GT box count once; the second is a false positive."""
    r = XP.recall_variants(*dets(([0, 0, 10, 10], .9, CAR), ([0, 0, 10, 9], .8, CAR)), G, GC, NO_IGN)
    assert r["veh"] == (1, 3, 2)


def test_van_on_car_gt():
    """a van prediction on a car box: matched class-agnostic and with van->car, missed with car-only."""
    r = XP.recall_variants(*dets(([0, 0, 10, 10], .9, VAN)), G, GC, NO_IGN)
    assert r["veh"][0] == 1 and r["cls_vancar"][0] == 1 and r["cls_car"] == (0, 3, 0)


def test_wrong_vehicle_class():
    """a car prediction on a truck box: class-agnostic hit, class-aware miss."""
    r = XP.recall_variants(*dets(([100, 100, 120, 110], .9, CAR)), G, GC, NO_IGN)
    assert r["veh"][0] == 1 and r["cls_vancar"][0] == 0 and r["cls_car"][0] == 0


def test_non_vehicle_predictions_discarded():
    r = XP.recall_variants(*dets(([0, 0, 10, 10], .9, PED)), G, GC, NO_IGN)
    assert all(v == (0, 3, 0) for v in r.values())


def test_confidence_threshold():
    r = XP.recall_variants(*dets(([0, 0, 10, 10], AF.CONF - 1e-3, CAR)), G, GC, NO_IGN)
    assert r["veh"] == (0, 3, 0)
    r = XP.recall_variants(*dets(([0, 0, 10, 10], AF.CONF, CAR)), G, GC, NO_IGN)
    assert r["veh"] == (1, 3, 1)


def test_iou_threshold():
    """IoU 0.5 matches, below does not (box shifted by half its width: IoU 1/3)."""
    r = XP.recall_variants(*dets(([5, 0, 15, 10], .9, CAR)), G, GC, NO_IGN)
    assert r["veh"][0] == 0


def test_ignored_detections_removed():
    """a detection >= 50 % inside an ignore region is removed (cannot match a GT box outside it)."""
    ign = np.array([[0, 0, 12, 12]], float)
    g = np.array([[0, 0, 10, 10]], float)
    g_out = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    r = XP.recall_variants(*dets(([50, 50, 60, 60], .9, CAR), ([1, 1, 11, 11], .8, CAR)), g_out, np.array([0, 0]), ign)
    # GT 0 lies inside the ignore region -> removed from the denominator; detection 2 is inside -> removed
    assert r["veh"] == (1, 1, 1)
    r = XP.recall_variants(*dets(([0, 0, 10, 10], .9, CAR)), g, np.array([0]), ign)
    assert r["veh"] == (0, 0, 0)


def test_denominator_counts_every_eligible_box():
    """no detections: n_gt = all boxes outside ignore regions, whatever their attributes (none are filtered)."""
    r = XP.recall_variants(*dets(), G, GC, NO_IGN)
    assert all(v == (0, 3, 0) for v in r.values())


def test_read_uavdt_gt(tmp_path):
    (tmp_path / "M0000_gt_whole.txt").write_text("1,1,10,20,30,40,1,1,1\n1,2,0,0,5,5,2,3,3\n2,1,11,20,30,40,1,1,2\n")
    (tmp_path / "M0000_gt_ignore.txt").write_text("1,9,100,100,50,50,1,-1,-1\n")
    g, c, ig = XP.read_uavdt_gt("M0000", 1, tmp_path)
    assert g.tolist() == [[10, 20, 40, 60], [0, 0, 5, 5]] and c.tolist() == [0, 2]
    assert ig.tolist() == [[100, 100, 150, 150]]
    g, c, ig = XP.read_uavdt_gt("M0000", 2, tmp_path)
    assert c.tolist() == [1] and len(ig) == 0
