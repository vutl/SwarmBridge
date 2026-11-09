"""A6.0-L1: policy-level COCO AP50 over the frames a policy captured.

Each planned frame of each episode is one evaluation image: ground truth = that frame's labels; detections = those
of the (image, mode) the policy delivered in time (none if dropped / late / Qmin-violating). Matching inside one
image does not depend on the other images, so pycocotools' per-image matching (evaluateImg, IoU 0.5, area 'all',
maxDets 500 as the VisDrone toolkit) is computed once per (pool, mode, image) and cached; a policy's AP50 is then
COCO's accumulate step (score-sorted TP / FP cumsums, precision envelope, 101 recall points, mean over categories
with ground truth) over its frame instances. Check: delivering every pool image once in mode m reproduces COCOeval's
AP50 of mode m (test_a6l1.py).
"""
import pickle
from pathlib import Path

import numpy as np

import a5_common as C
from a5_fidelity import read_gt

ANN = {"val": C.REPO / "data/visdrone/VisDrone2019-DET-val/annotations",
       "testdev": Path("/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev/annotations")}
DETS = {p: Path(f"/home/uservutl/work/a6l1/dets_{p}.npz") for p in ("val", "testdev")}
CACHE = Path("/home/uservutl/work/a6l1")
REC_THRS = np.linspace(0.0, 1.00, 101)


def _coco_gt(images, gts):
    from pycocotools.coco import COCO
    anns, aid = [], 1
    for i, name in enumerate(images):
        g, gc, ig = gts[name]
        for bx, c in zip(g, gc):
            anns.append(dict(id=aid, image_id=i, category_id=int(c) + 1, bbox=[bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]],
                             area=float((bx[2] - bx[0]) * (bx[3] - bx[1])), iscrowd=0)); aid += 1
        for bx in ig:
            for c in range(10):
                anns.append(dict(id=aid, image_id=i, category_id=c + 1, bbox=[bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]],
                                 area=float((bx[2] - bx[0]) * (bx[3] - bx[1])), iscrowd=1)); aid += 1
    gt = COCO()
    gt.dataset = dict(images=[dict(id=i) for i in range(len(images))], annotations=anns,
                      categories=[dict(id=c + 1, name=str(c)) for c in range(10)])
    gt.createIndex()
    return gt


def build(pool, dets_file=None, ann_dir=None, cache=None):
    """{'images': names, 'npig': [I, 10], 'modes': {mode: {img: [(cat, scores, tp, ign)]}}} (cached)."""
    from pycocotools.cocoeval import COCOeval
    cache = Path(cache or CACHE) / f"ap_{pool}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    D = np.load(dets_file or DETS[pool])
    images = [str(x) for x in D["images"]]
    ann = Path(ann_dir or ANN[pool])
    gts = {n: read_gt(ann / (Path(n).stem + ".txt")) for n in images}
    gt = _coco_gt(images, gts)
    out = dict(images=images, npig=np.zeros((len(images), 10), int), modes={})
    first = True
    for mode in [k for k in D.files if k != "images"]:
        arr = D[mode]
        res = [dict(image_id=int(r[0]), bbox=[float(x) for x in r[1:5]], score=float(r[5]), category_id=int(r[6]))
               for r in arr]
        per = {}
        if res:
            dt = gt.loadRes(res)
            E = COCOeval(gt, dt, "bbox")
            E.params.imgIds = list(range(len(images)))
            E.params.maxDets = [1, 100, 500]
            E.evaluate()
            nA, nI = len(E.params.areaRng), len(E.params.imgIds)
            for ci, cat in enumerate(E.params.catIds):
                for ii in range(nI):
                    e = E.evalImgs[ci * nA * nI + ii]       # area index 0 = 'all'
                    if e is None:
                        continue
                    if first:
                        out["npig"][ii, ci] = int(np.sum(e["gtIgnore"] == 0))
                    sc = np.asarray(e["dtScores"][:500], float)
                    if len(sc):
                        per.setdefault(ii, []).append((ci, sc, np.asarray(e["dtMatches"][0, :500] > 0),
                                                       np.asarray(e["dtIgnore"][0, :500], bool)))
            first = False
        out["modes"][mode] = per
    pickle.dump(out, open(cache, "wb"))
    return out


def ap50(instances, built):
    """instances: list of (pool, image index in built[pool]['images'], mode or None). Returns COCO AP50."""
    sc = [[] for _ in range(10)]
    tp = [[] for _ in range(10)]
    ig = [[] for _ in range(10)]
    npig = np.zeros(10)
    for pool, i, mode in instances:
        b = built[pool]
        npig += b["npig"][i]
        if mode is None:
            continue
        for ci, s, t, g in b["modes"][mode].get(i, ()):
            sc[ci].append(s); tp[ci].append(t); ig[ci].append(g)
    aps = []
    for ci in range(10):
        if npig[ci] == 0:
            continue
        if not sc[ci]:
            aps.append(0.0)
            continue
        s, t, g = np.concatenate(sc[ci]), np.concatenate(tp[ci]), np.concatenate(ig[ci])
        o = np.argsort(-s, kind="mergesort")
        t, g = t[o], g[o]
        tps = np.cumsum(t & ~g).astype(float)
        fps = np.cumsum(~t & ~g).astype(float)
        rc = tps / npig[ci]
        pr = (tps / (fps + tps + np.spacing(1))).tolist()
        for k in range(len(pr) - 1, 0, -1):
            if pr[k] > pr[k - 1]:
                pr[k - 1] = pr[k]
        q = np.zeros(len(REC_THRS))
        inds = np.searchsorted(rc, REC_THRS, side="left")
        for ri, pi in enumerate(inds):
            if pi < len(pr):
                q[ri] = pr[pi]
        aps.append(float(q.mean()))
    return float(np.mean(aps)) if aps else float("nan")
