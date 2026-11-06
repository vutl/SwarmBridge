# Step 0A (Huang-inspired, per-task, no queues)

- LEO->UAV, noise paper (-130.0 dBm): SNR 22.1 dB zenith / 18.8 dB at 40 deg; mean rate 65.8 / 55.4 Mbit/s
- LEO->UAV, noise thermal (-103.1 dBm): SNR -4.8 dB zenith / -8.1 dB at 40 deg; mean rate 3.8 / 2.0 Mbit/s
- M1 payload fig5 (314 kbit): M1 has higher PSNR and lower delay than every generative mode when the rate exceeds 0.524 Mbit/s
- M1 payload original (28000 kbit): M1 has higher PSNR and lower delay than every generative mode when the rate exceeds 46.756 Mbit/s
- outdated-CSI delta at zenith 0.0056; pass 215 s

| payload | rate Mbit/s | greedy utility | best fixed | best fixed utility | choice M1/M2/M3_1/M3_2/M3_3 |
|---|---|---|---|---|---|
| fig5 | 0.01 | -0.362 | M3_2 | -0.471 | [0.0, 0.0, 0.801, 0.126, 0.073] |
| fig5 | 0.0251 | -0.193 | M3_3 | -0.288 | [0.0, 0.0, 0.0, 0.097, 0.903] |
| fig5 | 0.0631 | -0.059 | M3_3 | -0.098 | [0.028, 0.0, 0.0, 0.015, 0.957] |
| fig5 | 0.158 | 0.307 | M1_jpeg2000 | 0.173 | [0.85, 0.0, 0.0, 0.0, 0.15] |
| fig5 | 0.398 | 0.509 | M1_jpeg2000 | 0.509 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| fig5 | 1 | 0.553 | M1_jpeg2000 | 0.553 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| fig5 | 2.51 | 0.570 | M1_jpeg2000 | 0.570 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| fig5 | 6.31 | 0.577 | M1_jpeg2000 | 0.577 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| fig5 | 15.8 | 0.579 | M1_jpeg2000 | 0.579 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| fig5 | 39.8 | 0.580 | M1_jpeg2000 | 0.580 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| fig5 | 100 | 0.581 | M1_jpeg2000 | 0.581 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| original | 0.01 | -0.362 | M3_2 | -0.471 | [0.0, 0.0, 0.801, 0.126, 0.073] |
| original | 0.0251 | -0.193 | M3_3 | -0.288 | [0.0, 0.0, 0.0, 0.097, 0.903] |
| original | 0.0631 | -0.084 | M3_3 | -0.098 | [0.0, 0.0, 0.0, 0.015, 0.985] |
| original | 0.158 | -0.045 | M3_3 | -0.045 | [0.0, 0.0, 0.0, 0.0, 1.0] |
| original | 0.398 | -0.034 | M3_3 | -0.034 | [0.0, 0.0, 0.0, 0.0, 1.0] |
| original | 1 | -0.029 | M3_3 | -0.029 | [0.0, 0.0, 0.0, 0.0, 1.0] |
| original | 2.51 | -0.027 | M3_3 | -0.027 | [0.0, 0.0, 0.0, 0.0, 1.0] |
| original | 6.31 | 0.031 | M3_3 | -0.026 | [0.066, 0.0, 0.0, 0.0, 0.934] |
| original | 15.8 | 0.343 | M1_jpeg2000 | 0.234 | [0.878, 0.0, 0.0, 0.0, 0.122] |
| original | 39.8 | 0.517 | M1_jpeg2000 | 0.517 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| original | 100 | 0.556 | M1_jpeg2000 | 0.556 | [1.0, 0.0, 0.0, 0.0, 0.0] |
