# temporal_select_v1 — val, seeds [0]

mIoU / vehicle IoU, mean over seeds; delta vs random in mIoU points with the per-seed sign count.

## budget 15% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.6792 | 0.2451 | +0.00 | 0/1 | 1.9 |
| entropy | 0.7172 | 0.3208 | +3.80 | 1/1 | 1.8 |
| ens_latent | 0.7129 | 0.3333 | +3.37 | 1/1 | 1.5 |
| ens_task | 0.7320 | 0.3519 | +5.28 | 1/1 | 3.4 |
| replica_latent | 0.7694 | 0.4439 | +9.02 | 1/1 | 1.5 |
| replica_task | 0.7764 | 0.4694 | +9.72 | 1/1 | 2.4 |
| ideal_latent | 0.7741 | 0.4549 | +9.49 | 1/1 | 1.2 |
| ideal_task | 0.7755 | 0.4672 | +9.63 | 1/1 | 1.8 |

## budget 30% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.7546 | 0.3833 | +0.00 | 0/1 | 1.8 |
| entropy | 0.8044 | 0.5063 | +4.98 | 1/1 | 1.9 |
| ens_latent | 0.7852 | 0.4679 | +3.06 | 1/1 | 1.6 |
| ens_task | 0.8079 | 0.5151 | +5.33 | 1/1 | 3.6 |
| replica_latent | 0.8237 | 0.5636 | +6.91 | 1/1 | 1.7 |
| replica_task | 0.8288 | 0.5728 | +7.42 | 1/1 | 2.3 |
| ideal_latent | 0.8233 | 0.5636 | +6.87 | 1/1 | 1.3 |
| ideal_task | 0.8300 | 0.5755 | +7.55 | 1/1 | 1.9 |

## budget 50% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.7990 | 0.4873 | +0.00 | 0/1 | 1.6 |
| entropy | 0.8415 | 0.5990 | +4.25 | 1/1 | 1.8 |
| ens_latent | 0.8237 | 0.5589 | +2.47 | 1/1 | 1.4 |
| ens_task | 0.8436 | 0.6033 | +4.46 | 1/1 | 3.2 |
| replica_latent | 0.8442 | 0.6080 | +4.51 | 1/1 | 1.4 |
| replica_task | 0.8477 | 0.6171 | +4.87 | 1/1 | 2.0 |
| ideal_latent | 0.8456 | 0.6119 | +4.66 | 1/1 | 1.1 |
| ideal_task | 0.8484 | 0.6182 | +4.93 | 1/1 | 1.8 |
