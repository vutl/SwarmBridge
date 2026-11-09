import sys; sys.path.insert(0, ".")
import numpy as np, a5_common as C, a6l1_task as TK
tab, pools = C.load_table(False)
tt = TK.task_table(tab)
M = C.MODES
ok = [j for j, m in enumerate(M)]
for pool in ("train", "test"):
    I = pools[pool]
    V = tt.v[I][:, ok]; B = tab.bits[I][:, ok] / 1e6
    R = (tt.tp[I][:, ok]) / np.maximum(tt.ngt[I].sum(), 1) * len(I)   # per-image TP share (sum -> recall)
    print(f"== {pool}: {len(I)} frames")
    # fixed modes: (mean bits, mean v, recall)
    fixed = [(B[:, j].mean(), V[:, j].mean(), tt.tp[I, ok[j]].sum() / tt.ngt[I].sum(), M[ok[j]]) for j in range(len(ok))]
    for lam in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        pick = np.argmax(V - lam * B, axis=1)          # oracle per-image (knows true v)
        ob, ov = B[np.arange(len(I)), pick].mean(), V[np.arange(len(I)), pick].mean()
        orc = tt.tp[I, np.array(ok)[pick]].sum() / tt.ngt[I].sum()
        # best fixed mode with mean bits <= oracle bits
        cand = [f for f in fixed if f[0] <= ob + 1e-12]
        bf = max(cand, key=lambda f: f[1]) if cand else None
        # fixed-mode frontier interpolation: best v among fixed at <= bits
        print(f"lam {lam:4.2f} oracle bits {ob:.3f} Mbit v {ov:.3f} recall {orc:.3f} | best fixed <= bits: "
              f"{bf[3] if bf else '-'} bits {bf[0] if bf else 0:.3f} v {bf[1] if bf else 0:.3f} recall {bf[2] if bf else 0:.3f}")
