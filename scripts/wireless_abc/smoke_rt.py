"""Sionna RT smoke test (CPU / LLVM by default). Usage: python smoke_rt.py [cpu|gpu]
One BS (8x8 UPA, 28 GHz) in the built-in Munich scene, receivers at 3 UAV altitudes; prints path counts, LoS flags,
channel shapes and timing. GPU mode must be launched through the shared GPU lock."""
import sys, time
import mitsuba as mi
mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
mi.set_variant("llvm_ad_mono_polarized" if mode == "cpu" else "cuda_ad_mono_polarized")
import numpy as np
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

t0 = time.time()
scene = load_scene(rt.scene.munich)
scene.frequency = 28e9
scene.tx_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.add(Transmitter(name="bs", position=[8.5, 21, 27]))
for k, h in enumerate([20, 50, 100]):
    for i, (x, y) in enumerate([(45, 90), (-60, 30), (120, -40)]):
        scene.add(Receiver(name=f"uav{k}_{i}", position=[x, y, h]))
t1 = time.time()
paths = PathSolver()(scene, max_depth=3, los=True, specular_reflection=True, diffuse_reflection=False, refraction=False)
t2 = time.time()
a, tau = paths.cir(out_type="numpy")
print("mode", mode, "scene load %.1fs" % (t1 - t0), "solve %.1fs" % (t2 - t1))
print("cir a shape", a.shape, "tau shape", tau.shape)
pw = (np.abs(a) ** 2).sum(axis=tuple(range(1, a.ndim)))     # per receiver
inter = paths.interactions.numpy() if hasattr(paths.interactions, "numpy") else np.array(paths.interactions)
print("rx power (dB):", np.round(10 * np.log10(pw + 1e-30), 1))
print("interactions shape", inter.shape)
