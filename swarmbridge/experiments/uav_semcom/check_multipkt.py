"""Check: does the one-image-per-UAV-per-slot cap drive the A3 story? MultiPkt env keeps serving the queue inside the
slot (next image is encoded right after the previous one is sent, same mode as the action's mode if it has none)."""
import sys, math, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import env as E, policies as P

class MultiPkt(E.UavSemcomEnv):
    def step(self, actions):
        c = self.c; dt = c["dt"]; reward, energy_j = 0.0, 0.0
        for k, (mv, ch, pl, md) in enumerate(actions):
            q = self.uavs[k].queue
            if q and q[0].mode < 0:
                h = q[0]; h.mode = int(md); h.ready = max(self.t, h.t_cap) + self.tab.enc_s[h.mode]; h.rem_bits = self.tab.bits[h.img, h.mode]
        tx = {}
        for k, (mv, ch, pl, md) in enumerate(actions):
            q = self.uavs[k].queue
            if ch > 0 and q and q[0].mode >= 0 and q[0].ready < self.t + dt:
                tx[k] = (int(ch), c["p_levels"][int(pl)], self._gain_draw(self.uavs[k].pos))
        for k, (ch, p, g) in tx.items():
            interf = sum(p2 * g2 for k2, (ch2, p2, g2) in tx.items() if k2 != k and ch2 == ch) if c["interference"] else 0.0
            rate = c["B_total"] / c["M"] * math.log2(1 + p * g / (self.noise_w() + interf))
            u = self.uavs[k]; tcur = self.t
            while u.queue:
                h = u.queue[0]
                if h.mode < 0:
                    h.mode = int(actions[k][3]); h.ready = max(tcur, h.t_cap) + self.tab.enc_s[h.mode]; h.rem_bits = self.tab.bits[h.img, h.mode]
                t0 = max(tcur, h.ready)
                if t0 >= self.t + dt: break
                avail = self.t + dt - t0; sent = rate * avail
                if sent >= h.rem_bits:
                    t_end = t0 + h.rem_bits / rate; energy_j += p * (t_end - t0); u.energy += p * (t_end - t0)
                    reward += self._finish(h, t_end); u.queue.pop(0); tcur = t_end
                else:
                    h.rem_bits -= sent; energy_j += p * avail; u.energy += p * avail; break
        # rest identical to the base env: run base step with no transmission by passing channel 0
        base_r_obs = super().step([(mv, 0, pl, md) for (mv, ch, pl, md) in actions])
        return base_r_obs[0], base_r_obs[1] + reward - (c["w_energy"] * energy_j / 1000.0 if c["energy"] else 0), base_r_obs[2]
# note: transmit energy weight sign: base step subtracts w_energy*E; we subtract our tx energy the same way.

R="/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom"
MODES="cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25".split(",")
tab = E.ModeTable([R+"/a11/profile.jsonl", R+"/a12/profile.jsonl"], MODES, R+"/a11/cpu_learned.jsonl"); tr, te = tab.split()
seeds = range(1000, 1030)
pols = [P.greedy(True), P.paced(0), P.paced(3), P.staggered(None, "idx"), P.staggered(0, "far"), P.fixed(MODES.index("avif_80")), P.fixed(MODES.index("mbt2018-mean_q8"))]
for cfg in ({}, {"w_time": 0.3}):
    print("cfg", cfg)
    for cls in (E.UavSemcomEnv, MultiPkt):
        for pol in pols:
            ms = [P.run_episode(cls(tab, te, cfg), pol, s) for s in seeds]
            print(f"  {cls.__name__:13s} {pol.__name__:16s} return {np.mean([m['return'] for m in ms]):+7.2f} util {np.mean([m['mean_util'] for m in ms]):+.3f} "
                  f"ok {np.mean([m['ok_rate'] for m in ms]):.3f} mission {np.mean([m['mission_s'] for m in ms]):.0f}s", flush=True)
