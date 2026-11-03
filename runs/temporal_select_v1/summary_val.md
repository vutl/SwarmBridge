# temporal_select_v1 — val, seeds [0, 1, 2]

mIoU / vehicle IoU, mean over seeds; delta vs random in mIoU points with the per-seed sign count.

## budget 15% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.6817 | 0.2501 | +0.00 | 0/3 | 1.7 |
| entropy | 0.7168 | 0.3175 | +3.50 | 3/3 | 1.8 |
| ens_latent | 0.7020 | 0.3176 | +2.02 | 3/3 | 1.4 |
| ens_task | 0.7301 | 0.3432 | +4.84 | 3/3 | 3.3 |
| replica_latent | 0.7702 | 0.4453 | +8.85 | 3/3 | 1.5 |
| replica_task | 0.7736 | 0.4658 | +9.19 | 3/3 | 2.1 |
| ideal_latent | 0.7734 | 0.4535 | +9.16 | 3/3 | 1.2 |
| ideal_task | 0.7776 | 0.4708 | +9.59 | 3/3 | 1.8 |

## budget 30% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.7585 | 0.3903 | +0.00 | 0/3 | 1.6 |
| entropy | 0.8072 | 0.5116 | +4.87 | 3/3 | 1.8 |
| ens_latent | 0.7868 | 0.4698 | +2.83 | 3/3 | 1.4 |
| ens_task | 0.8131 | 0.5273 | +5.46 | 3/3 | 3.3 |
| replica_latent | 0.8287 | 0.5743 | +7.02 | 3/3 | 1.5 |
| replica_task | 0.8343 | 0.5872 | +7.58 | 3/3 | 2.1 |
| ideal_latent | 0.8300 | 0.5774 | +7.15 | 3/3 | 1.2 |
| ideal_task | 0.8355 | 0.5890 | +7.69 | 3/3 | 1.8 |

## budget 50% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.8016 | 0.4933 | +0.00 | 0/3 | 1.6 |
| entropy | 0.8429 | 0.6029 | +4.13 | 3/3 | 1.8 |
| ens_latent | 0.8289 | 0.5725 | +2.73 | 3/3 | 1.4 |
| ens_task | 0.8447 | 0.6067 | +4.31 | 3/3 | 3.2 |
| replica_latent | 0.8468 | 0.6161 | +4.53 | 3/3 | 1.5 |
| replica_task | 0.8507 | 0.6248 | +4.91 | 3/3 | 2.0 |
| ideal_latent | 0.8478 | 0.6184 | +4.62 | 3/3 | 1.2 |
| ideal_task | 0.8510 | 0.6246 | +4.94 | 3/3 | 1.8 |
