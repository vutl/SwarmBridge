"""Find the camera-axis convention: try R / R^T x 48 signed axis permutations, score how concentrated the seg colour at
projected box centres is (the right convention puts most centres on the single 'vehicle' colour)."""
import json, sys, itertools, collections
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))

root = Path(sys.argv[1]); meta = root / "v1.0-mini"
load = lambda n: json.loads((meta / f"{n}.json").read_text())
samples, sds, cals, anns = load("sample"), load("sample_data"), load("calibrated_sensor"), load("sample_annotation")
cal_of = {c["token"]: c for c in cals}
sd_by = collections.defaultdict(list)
for d in sds: sd_by[d["sample_token"]].append(d)
ann_by = collections.defaultdict(list)
for a in anns: ann_by[a["sample_token"]].append(a)

def qR(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

perms = []
for p in itertools.permutations(range(3)):
    for s in itertools.product([1, -1], repeat=3):
        M = np.zeros((3, 3)); M[range(3), p] = s
        if np.linalg.det(M) > 0 or True:
            perms.append(M)
cases = []
for s in samples[::40][:8]:
    for d in sd_by[s["token"]]:
        c = cal_of[d["calibrated_sensor_token"]]
        seg = np.asarray(Image.open(root / "sweeps" / Path(d["filename"]).parent.name / (Path(d["filename"]).stem + "_seg.png")).convert("RGB"))
        cases.append((qR(c["rotation"]), np.array(c["translation"]), np.array(c["camera_intrinsic"]), seg,
                      np.array([a["translation"] for a in ann_by[s["token"]]]) + np.array([0, 0, 0.0])))
results = []
for flip in ((1, 1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, -1)):
  F = np.diag(flip)
  for transpose in (False, True):
    for pi, M in enumerate(perms):
        votes = collections.Counter(); n_in = 0
        for R, t, K, seg, X in cases:
            Rw = R.T if transpose else R
            P = M @ (Rw.T @ ((X - t) @ F).T)  # (mirrored) world -> sensor -> cv camera
            ok = P[2] > 1
            uv = (K @ P[:, ok])[:2] / P[2, ok]
            inside = (uv[0] >= 0) & (uv[0] < 800) & (uv[1] >= 0) & (uv[1] < 450)
            for u, v in uv[:, inside].T.astype(int):
                votes[tuple(int(x) for x in seg[v, u])] += 1
            n_in += inside.sum()
        if n_in > 50:
            k = votes[(95, 90, 229)]  # vehicle colour (checked on a bottom-cam seg image)
            results.append((k / n_in, n_in, transpose, pi, k, flip))
results.sort(reverse=True)
for r in results[:6]:
    print(f"vehicle share {r[0]:.2f} n_in {r[1]} transpose {r[2]} hits {r[4]} world-flip {r[5]}\n M=\n{perms[r[3]]}")
