# temporal_select_v1 — test, seeds [0, 1, 2], ACK loss 1.0

mIoU / vehicle IoU, mean over seeds; delta vs random in mIoU points with the per-seed sign count.

## budget 15% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.6760 | 0.2200 | +0.00 | 0/3 | 3.3 |
| entropy | 0.7000 | 0.2608 | +2.39 | 3/3 | 3.3 |
| ens_latent | 0.6889 | 0.2785 | +1.28 | 3/3 | 2.3 |
| ens_task | 0.7074 | 0.2765 | +3.14 | 3/3 | 6.1 |
| replica_latent | 0.7462 | 0.4007 | +7.02 | 3/3 | 2.3 |
| replica_task | 0.7443 | 0.3948 | +6.83 | 3/3 | 3.6 |
| ideal_latent | 0.7572 | 0.4181 | +8.12 | 3/3 | 2.1 |
| ideal_task | 0.7627 | 0.4202 | +8.67 | 3/3 | 3.3 |

## budget 30% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.7503 | 0.3643 | +0.00 | 0/3 | 3.2 |
| entropy | 0.7894 | 0.4609 | +3.91 | 3/3 | 3.3 |
| ens_latent | 0.7851 | 0.4613 | +3.48 | 3/3 | 2.3 |
| ens_task | 0.7990 | 0.4854 | +4.87 | 3/3 | 6.1 |
| replica_latent | 0.8175 | 0.5454 | +6.72 | 3/3 | 2.3 |
| replica_task | 0.8232 | 0.5568 | +7.29 | 3/3 | 3.6 |
| ideal_latent | 0.8238 | 0.5590 | +7.35 | 3/3 | 2.1 |
| ideal_task | 0.8302 | 0.5711 | +7.99 | 3/3 | 3.3 |

## budget 50% (half random, half chosen)

| policy | mIoU | vehicle IoU | Δ mIoU vs random | seeds > random | s/rollout |
|---|---:|---:|---:|---:|---:|
| random | 0.7929 | 0.4670 | +0.00 | 0/3 | 3.2 |
| entropy | 0.8296 | 0.5608 | +3.67 | 3/3 | 3.3 |
| ens_latent | 0.8263 | 0.5592 | +3.34 | 3/3 | 2.3 |
| ens_task | 0.8328 | 0.5691 | +3.99 | 3/3 | 6.1 |
| replica_latent | 0.8407 | 0.5961 | +4.78 | 3/3 | 2.3 |
| replica_task | 0.8447 | 0.6043 | +5.18 | 3/3 | 3.6 |
| ideal_latent | 0.8435 | 0.6025 | +5.06 | 3/3 | 2.0 |
| ideal_task | 0.8465 | 0.6073 | +5.37 | 3/3 | 3.3 |
