import sys, json, glob
from multiprocessing import Pool
sys.path.insert(0, "/mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom")
from r1b_run import load_clip, base_cfg
from r1b_sim import run_clip
R = "/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/r1b/"
L = json.load(open(R + "lock_v1.json"))
params = dict(th_T=2, th_F=8, N_max=10**9, w_lost=1, w_drift=1, w_unc=0)
K = ["tp", "fp", "fn", "n_frame", "n_tok", "n_skip", "dropped_compute", "ul_lost", "ul_late", "oos_obs"]
def job(p):
    c = load_clip(p, L["frame_q"]); out = []
    for cond in ["A1", "D"]:
        r = run_clip(c, base_cfg(L, cond, L["R_F"], 1000, params))
        out.append((c.name, cond, {k: r[k] for k in K}))
    return out
if __name__ == "__main__":
    paths = sorted(glob.glob("/home/uservutl/work/r1b/precomp/tune/*.pkl"))
    with Pool(6) as pool:
        res = [x for o in pool.map(job, paths) for x in o]
    tot = {}
    for nm, cond, r in res:
        t = tot.setdefault(cond, {k: 0 for k in K})
        for k in K: t[k] += r[k]
    print("TOTAL", json.dumps(tot))
    A = {nm: r for nm, c, r in res if c == "A1"}; D = {nm: r for nm, c, r in res if c == "D"}
    diff = sorted(((D[n]["tp"] - A[n]["tp"], D[n]["fp"] - A[n]["fp"], n) for n in A), key=lambda x: x[0] - x[1])
    for d in diff[:6] + diff[-3:]:
        n = d[2]; print(n, "dTP", d[0], "dFP", d[1], "A1", A[n], "D", D[n])
    print("n_identical", sum(A[n] == D[n] for n in A), "/", len(A))
