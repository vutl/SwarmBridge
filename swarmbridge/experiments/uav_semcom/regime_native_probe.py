"""Order-of-magnitude probe for the operating-regime analysis: AVIF / JPEG2000 payload, PSNR and CPU encode time at
NATIVE VisDrone resolution (no resize), same subset rule as A1 (sorted, every 5th), first N images. CPU only."""
import sys, time, json, io
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from imcodecs import PillowCodec
src, out, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
files = sorted(src.glob("*.jpg"))[::5][:n]
modes = [PillowCodec("AVIF", q) for q in (5, 20, 50, 80)] + [PillowCodec("JPEG2000", r) for r in (100, 25)]
with open(out, "w") as f:
    for p in files:
        img = np.asarray(Image.open(p).convert("RGB"))
        for m in modes:
            t0 = time.perf_counter(); b = m.encode(img); te = time.perf_counter() - t0
            t0 = time.perf_counter(); rec = m.decode(b); td = time.perf_counter() - t0
            mse = np.mean((img.astype(float) - rec.astype(float)) ** 2)
            f.write(json.dumps(dict(image=p.name, h=img.shape[0], w=img.shape[1], mode=m.name, bytes=len(b), enc_s=te, dec_s=td,
                                    psnr=float(10 * np.log10(255 ** 2 / max(mse, 1e-9))))) + "\n"); f.flush()
print("NATIVE_DONE")
