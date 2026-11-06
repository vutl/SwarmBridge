# Operating regime (env v2)

Config: K=3 UAVs, M=2 channels, B_total=1 MHz, p_max=0.2 W, H=100.0 m, survey radius (1000.0, 2500.0) m, 4 images per waypoint, waypoint spacing 120.0 m, v_max=20.0 m/s, deadline U(4.0, 12.0) s.

Per-channel rate at waypoints (max power, no interference): median 2.25 Mb/s, p10 1.81, p90 2.88.

Capture rate at full survey speed: 0.67 images/s per UAV, 2.00 images/s in total.

| mode | kbit (test mean) | airtime @ median (s) | enc (s) | dec (s) | PSNR (dB) | rho_link | rho_enc |
|---|---|---|---|---|---|---|---|
| cap_t2i | 0.5 | 0.000 | 0.228 | 0.303 | 9.7 | 0.00 | 0.15 |
| thumb64_up | 6.5 | 0.003 | 0.008 | 0.003 | 22.0 | 0.00 | 0.01 |
| cap_thumb64_i2i_t0.6 | 7.0 | 0.003 | 0.236 | 0.292 | 19.0 | 0.00 | 0.16 |
| cap_thumb128_i2i_t0.3 | 17.3 | 0.008 | 0.247 | 0.248 | 21.9 | 0.01 | 0.16 |
| avif_5 | 23.1 | 0.010 | 0.038 | 0.003 | 25.3 | 0.01 | 0.03 |
| avif_20 | 45.8 | 0.020 | 0.052 | 0.003 | 27.6 | 0.02 | 0.03 |
| avif_50 | 158.6 | 0.070 | 0.080 | 0.004 | 33.3 | 0.07 | 0.05 |
| avif_80 | 418.3 | 0.186 | 0.101 | 0.006 | 40.3 | 0.19 | 0.07 |
| mbt2018-mean_q1 | 35.9 | 0.016 | 0.125 | 0.024 | 27.4 | 0.02 | 0.08 |
| mbt2018-mean_q3 | 86.7 | 0.038 | 0.133 | 0.024 | 31.0 | 0.04 | 0.09 |
| mbt2018-mean_q5 | 183.5 | 0.081 | 0.232 | 0.036 | 35.2 | 0.08 | 0.15 |
| mbt2018-mean_q8 | 415.3 | 0.184 | 0.236 | 0.038 | 41.5 | 0.18 | 0.16 |
| jpeg2000_100 | 62.7 | 0.028 | 0.065 | 0.006 | 25.9 | 0.03 | 0.04 |
| jpeg2000_25 | 251.2 | 0.111 | 0.065 | 0.009 | 30.8 | 0.11 | 0.04 |

Highest-quality mode (mbt2018-mean_q8) under the bandwidths of the closest papers (same geometry, same per-channel split):

| B_total | source | median rate / channel (Mb/s) | airtime (s) | rho_link |
|---|---|---|---|---|
| 1 MHz | this env (assumed) | 2.25 | 0.185 | 0.18 |
| 5 MHz | Si et al. 2023 | 6.02 | 0.069 | 0.07 |
| 10 MHz | Huang et al. 2025 | 8.28 | 0.050 | 0.05 |
| 20 MHz | Li et al. 2026 | 10.53 | 0.039 | 0.04 |

UAVs per 2 channels at which mbt2018-mean_q8 saturates the link at full survey speed (rho_link = 1): K ~ 16.3.

Native-resolution probe (20 VisDrone-DET-val images, sizes {'1920x1080': 2, '960x540': 11, '1360x765': 7}; CPU encode on the A1 machine), one frame per waypoint instead of 4 512x512 crops:

| mode | kbit / frame | PSNR | enc (s) | airtime @ median (s) | rho_link (1 frame / waypoint) |
|---|---|---|---|---|---|
| avif_5 | 65 | 27.0 | 0.07 | 0.029 | 0.01 |
| avif_20 | 132 | 29.4 | 0.08 | 0.058 | 0.01 |
| avif_50 | 511 | 36.3 | 0.12 | 0.227 | 0.06 |
| avif_80 | 1020 | 44.4 | 0.12 | 0.453 | 0.11 |
| jpeg2000_100 | 205 | 28.4 | 0.18 | 0.091 | 0.02 |
| jpeg2000_25 | 822 | 33.6 | 0.18 | 0.365 | 0.09 |

'predicted' mode information (per-mode linear fit on log bits of the probe mode avif_5, fitted on train images), test images:

| mode | R2 log bits | R2 PSNR | MAE PSNR (dB) |
|---|---|---|---|
| cap_t2i | -0.06 | 0.07 | 0.79 |
| thumb64_up | 0.85 | 0.76 | 0.69 |
| cap_thumb64_i2i_t0.6 | 0.81 | 0.80 | 0.47 |
| cap_thumb128_i2i_t0.3 | 0.93 | 0.78 | 0.62 |
| avif_20 | 0.90 | 0.58 | 1.18 |
| avif_50 | 0.65 | 0.57 | 1.12 |
| avif_80 | 0.61 | 0.62 | 0.58 |
| mbt2018-mean_q1 | 0.79 | 0.62 | 0.89 |
| mbt2018-mean_q3 | 0.70 | 0.61 | 0.82 |
| mbt2018-mean_q5 | 0.64 | 0.63 | 0.59 |
| mbt2018-mean_q8 | 0.63 | 0.61 | 0.48 |
| jpeg2000_100 | -0.06 | 0.69 | 1.17 |
| jpeg2000_25 | 0.03 | 0.62 | 1.72 |
