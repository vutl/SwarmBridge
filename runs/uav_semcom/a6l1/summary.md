### Tinh chỉnh dưới reward tác vụ (train, seed 0..29)

| regime | T:best_task | R_task | R_PSNR | recall | tv tốt nhất | R_task | A5 T:best (PSNR) | R_task | R_PSNR | recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ref | fixed_avif_50 | +69.17 | +44.26 | 0.558 | tv:greedy+budget1.25+reuse | +69.17 | pacedB1.5+budget1.25+reuse | +59.16 | +50.86 | 0.560 |
| K5 | tv:greedy+budget1+reuse | +96.70 | +45.07 | 0.521 | tv:greedy+budget1+reuse | +96.70 | pacedB2+budget1+reuse | +87.95 | +42.51 | 0.519 |
| phy_mcs_blind | fixed_mbt2018-mean_q3 | +57.38 | +14.97 | 0.526 | tv:greedy+budget3 | +52.52 | pacedB2+budget1 | +43.81 | +14.98 | 0.495 |

### Đường học PPO task (val, seed 3000..3019; return tác vụ)

| run | best@ | val best | val @2M | val @1M | còn tăng ở 2M? |
|---|---|---|---|---|---|
| K5_s0 | 1200000 | +148.00 | +146.98 | +147.86 | không |
| K5_s1 | 1800000 | +150.17 | +149.65 | +148.49 | không |
| K5_s2 | 1800000 | +150.17 | +150.02 | +150.01 | không |
| phy_mcs_blind_s0 | 2000000 | +83.35 | +83.35 | +83.21 | có |
| phy_mcs_blind_s1 | 600000 | +83.30 | +83.21 | +83.23 | không |
| phy_mcs_blind_s2 | 1200000 | +83.25 | +83.24 | +83.20 | không |
| ref_s0 | 400000 | +94.82 | +94.63 | +94.76 | không |
| ref_s1 | 400000 | +94.82 | +93.98 | +94.06 | không |
| ref_s2 | 800000 | +94.59 | +93.97 | +94.48 | không |

### ref, pool test (seed 9000..9099)

| policy | R_task | R_PSNR | recall | recall nhỏ | FP/khung giao | precision | Mbit gửi | đúng hạn | năng lượng kJ | thời gian s | AP50 (gộp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T:best_task (fixed_avif_50) | 76.34 | 48.88 | 0.589 | 0.470 | 15.8 | 0.715 | 79.2 | 0.999 | 39.0 | 73 | 0.4329 |
| tv (task-tuned) (tv:greedy+budget1.25+reuse) | 76.00 | 57.61 | 0.592 | 0.475 | 15.7 | 0.718 | 106.3 | 0.998 | 39.0 | 73 | 0.4382 |
| reuse (task-tuned) (greedy+budget0.75+reuse) | 74.93 | 55.31 | 0.585 | 0.469 | 15.7 | 0.718 | 98.9 | 0.986 | 39.0 | 73 | 0.4339 |
| budget (task-tuned) (pacedB1.5+budget0.75) | 70.86 | 50.81 | 0.578 | 0.462 | 15.7 | 0.718 | 98.7 | 0.977 | 39.3 | 74 | 0.4263 |
| fixed (task-tuned) (fixed_avif_50) | 76.34 | 48.88 | 0.589 | 0.470 | 15.8 | 0.715 | 79.2 | 0.999 | 39.0 | 73 | 0.4329 |
| A5 T:best (PSNR-tuned) (pacedB1.5+budget1.25+reuse) | 68.22 | 59.51 | 0.596 | 0.481 | 15.5 | 0.723 | 141.0 | 0.992 | 40.5 | 76 | 0.4418 |
| A5 fixed (PSNR-tuned) (fixed_avif_50) | 76.34 | 48.88 | 0.589 | 0.470 | 15.8 | 0.715 | 79.2 | 0.999 | 39.0 | 73 | 0.4329 |
| A4 greedy+budget1+reuse (greedy+budget1+reuse) | 73.33 | 59.84 | 0.595 | 0.479 | 15.6 | 0.721 | 123.2 | 0.994 | 39.1 | 73 | 0.4396 |
| **PPO task** | 76.25 | 48.58 | 0.589 | 0.470 | 15.8 | 0.715 | 78.6 | 0.999 | 39.0 | 73 | 0.4325 |
| PPO PSNR (A5) | 74.50 | 59.32 | 0.596 | 0.479 | 15.7 | 0.719 | 117.4 | 0.999 | 39.6 | 74 | 0.4410 |
| PPO task, ckpt 2M | 75.80 | 46.32 | 0.586 | 0.467 | 15.8 | 0.715 | 74.5 | 0.996 | 39.0 | 73 | 0.4283 |

### ref, pool testdev (seed 9000..9099)

| policy | R_task | R_PSNR | recall | recall nhỏ | FP/khung giao | precision | Mbit gửi | đúng hạn | năng lượng kJ | thời gian s | AP50 (gộp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T:best_task (fixed_avif_50) | 35.12 | 46.33 | 0.498 | 0.369 | 12.1 | 0.656 | 109.4 | 0.997 | 39.2 | 74 | 0.3505 |
| tv (task-tuned) (tv:greedy+budget1.25+reuse) | 36.48 | 49.28 | 0.497 | 0.369 | 12.0 | 0.658 | 117.0 | 0.995 | 39.1 | 73 | 0.3476 |
| reuse (task-tuned) (greedy+budget0.75+reuse) | 28.95 | 35.95 | 0.445 | 0.337 | 11.9 | 0.651 | 89.7 | 0.932 | 38.9 | 73 | 0.3089 |
| budget (task-tuned) (pacedB1.5+budget0.75) | 26.81 | 33.50 | 0.445 | 0.335 | 11.8 | 0.652 | 89.7 | 0.932 | 39.2 | 74 | 0.3074 |
| fixed (task-tuned) (fixed_avif_50) | 35.12 | 46.33 | 0.498 | 0.369 | 12.1 | 0.656 | 109.4 | 0.997 | 39.2 | 74 | 0.3505 |
| A5 T:best (PSNR-tuned) (pacedB1.5+budget1.25+reuse) | 25.67 | 46.28 | 0.493 | 0.368 | 12.0 | 0.659 | 155.7 | 0.982 | 40.9 | 77 | 0.3440 |
| A5 fixed (PSNR-tuned) (fixed_avif_50) | 35.12 | 46.33 | 0.498 | 0.369 | 12.1 | 0.656 | 109.4 | 0.997 | 39.2 | 74 | 0.3505 |
| A4 greedy+budget1+reuse (greedy+budget1+reuse) | 32.74 | 47.02 | 0.485 | 0.365 | 12.0 | 0.657 | 124.6 | 0.977 | 39.1 | 73 | 0.3387 |
| **PPO task** | 35.45 | 46.22 | 0.498 | 0.368 | 12.1 | 0.656 | 107.9 | 0.998 | 39.2 | 74 | 0.3495 |
| PPO PSNR (A5) | 31.48 | 49.28 | 0.502 | 0.373 | 12.1 | 0.658 | 140.3 | 0.996 | 40.8 | 77 | 0.3517 |

### K5, pool test (seed 9000..9099)

| policy | R_task | R_PSNR | recall | recall nhỏ | FP/khung giao | precision | Mbit gửi | đúng hạn | năng lượng kJ | thời gian s | AP50 (gộp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T:best_task (tv:greedy+budget1+reuse) | 114.90 | 65.93 | 0.570 | 0.454 | 15.6 | 0.715 | 116.2 | 0.980 | 65.1 | 73 | 0.4176 |
| tv (task-tuned) (tv:greedy+budget1+reuse) | 114.90 | 65.93 | 0.570 | 0.454 | 15.6 | 0.715 | 116.2 | 0.980 | 65.1 | 73 | 0.4176 |
| reuse (task-tuned) (greedy+budget1+reuse) | 108.50 | 63.33 | 0.567 | 0.452 | 15.6 | 0.715 | 125.7 | 0.972 | 65.2 | 73 | 0.4163 |
| budget (task-tuned) (pacedB2+budget1) | 73.68 | 27.95 | 0.524 | 0.413 | 15.5 | 0.712 | 118.4 | 0.919 | 68.4 | 77 | 0.3797 |
| fixed (task-tuned) (fixed_mbt2018-mean_q3) | 99.38 | 26.34 | 0.557 | 0.437 | 15.5 | 0.713 | 64.4 | 0.969 | 65.7 | 74 | 0.3959 |
| A5 T:best (PSNR-tuned) (pacedB2+budget1+reuse) | 103.35 | 62.54 | 0.568 | 0.452 | 15.6 | 0.715 | 138.8 | 0.970 | 66.8 | 75 | 0.4175 |
| A5 fixed (PSNR-tuned) (fixed_mbt2018-mean_q3) | 99.38 | 26.34 | 0.557 | 0.437 | 15.5 | 0.713 | 64.4 | 0.969 | 65.7 | 74 | 0.3959 |
| A4 greedy+budget1+reuse (greedy+budget1+reuse) | 108.50 | 63.33 | 0.567 | 0.452 | 15.6 | 0.715 | 125.7 | 0.972 | 65.2 | 73 | 0.4163 |
| **PPO task** | 117.93 | 49.73 | 0.565 | 0.445 | 15.6 | 0.713 | 76.9 | 0.978 | 65.0 | 73 | 0.4019 |
| PPO PSNR (A5) | 118.58 | 64.34 | 0.578 | 0.459 | 15.7 | 0.714 | 109.3 | 0.990 | 65.9 | 74 | 0.4184 |
| PPO task, ckpt 2M | 117.23 | 47.77 | 0.564 | 0.444 | 15.6 | 0.713 | 74.2 | 0.977 | 65.0 | 73 | 0.4010 |

### K5, pool testdev (seed 9000..9099)

| policy | R_task | R_PSNR | recall | recall nhỏ | FP/khung giao | precision | Mbit gửi | đúng hạn | năng lượng kJ | thời gian s | AP50 (gộp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T:best_task (tv:greedy+budget1+reuse) | 43.95 | 46.36 | 0.445 | 0.331 | 11.6 | 0.655 | 115.3 | 0.939 | 65.1 | 73 | 0.3075 |
| tv (task-tuned) (tv:greedy+budget1+reuse) | 43.95 | 46.36 | 0.445 | 0.331 | 11.6 | 0.655 | 115.3 | 0.939 | 65.1 | 73 | 0.3075 |
| reuse (task-tuned) (greedy+budget1+reuse) | 39.56 | 42.44 | 0.433 | 0.326 | 11.5 | 0.654 | 117.4 | 0.928 | 65.1 | 73 | 0.2987 |
| budget (task-tuned) (pacedB2+budget1) | 8.73 | 8.66 | 0.406 | 0.302 | 11.4 | 0.652 | 112.4 | 0.888 | 68.3 | 77 | 0.2788 |
| fixed (task-tuned) (fixed_mbt2018-mean_q3) | 41.38 | 31.55 | 0.462 | 0.335 | 10.9 | 0.666 | 82.5 | 0.989 | 65.8 | 74 | 0.3067 |
| A5 T:best (PSNR-tuned) (pacedB2+budget1+reuse) | 36.56 | 42.92 | 0.441 | 0.332 | 11.7 | 0.653 | 131.6 | 0.935 | 66.3 | 75 | 0.3036 |
| A5 fixed (PSNR-tuned) (fixed_mbt2018-mean_q3) | 41.38 | 31.55 | 0.462 | 0.335 | 10.9 | 0.666 | 82.5 | 0.989 | 65.8 | 74 | 0.3067 |
| A4 greedy+budget1+reuse (greedy+budget1+reuse) | 39.56 | 42.44 | 0.433 | 0.326 | 11.5 | 0.654 | 117.4 | 0.928 | 65.1 | 73 | 0.2987 |
| **PPO task** | 58.68 | 52.44 | 0.469 | 0.341 | 11.1 | 0.664 | 95.5 | 0.996 | 65.1 | 73 | 0.3137 |
| PPO PSNR (A5) | 50.45 | 54.38 | 0.479 | 0.351 | 11.5 | 0.661 | 130.2 | 0.993 | 67.5 | 76 | 0.3275 |

### phy_mcs_blind, pool test (seed 9000..9099)

| policy | R_task | R_PSNR | recall | recall nhỏ | FP/khung giao | precision | Mbit gửi | đúng hạn | năng lượng kJ | thời gian s | AP50 (gộp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T:best_task (fixed_mbt2018-mean_q3) | 64.36 | 20.00 | 0.559 | 0.439 | 15.5 | 0.714 | 38.7 | 0.971 | 39.3 | 74 | 0.3983 |
| tv (task-tuned) (tv:greedy+budget3) | 63.07 | 34.29 | 0.581 | 0.465 | 15.7 | 0.716 | 73.2 | 0.988 | 39.5 | 74 | 0.4225 |
| reuse (task-tuned) (pacedB1.5+budget0.5+reuse) | 28.60 | -14.71 | 0.412 | 0.313 | 15.1 | 0.695 | 31.5 | 0.804 | 41.0 | 77 | 0.2925 |
| budget (task-tuned) (pacedB2+budget1) | 54.77 | 27.43 | 0.550 | 0.437 | 15.6 | 0.715 | 75.2 | 0.947 | 40.7 | 77 | 0.4026 |
| fixed (task-tuned) (fixed_mbt2018-mean_q3) | 64.36 | 20.00 | 0.559 | 0.439 | 15.5 | 0.714 | 38.7 | 0.971 | 39.3 | 74 | 0.3983 |
| A5 T:best (PSNR-tuned) (pacedB2+budget1) | 54.77 | 27.43 | 0.550 | 0.437 | 15.6 | 0.715 | 75.2 | 0.947 | 40.7 | 77 | 0.4026 |
| A5 fixed (PSNR-tuned) (fixed_mbt2018-mean_q3) | 64.36 | 20.00 | 0.559 | 0.439 | 15.5 | 0.714 | 38.7 | 0.971 | 39.3 | 74 | 0.3983 |
| A4 greedy+budget1+reuse (greedy+budget1+reuse) | -1.19 | -28.91 | 0.430 | 0.337 | 15.4 | 0.711 | 51.2 | 0.766 | 39.6 | 74 | 0.3108 |
| **PPO task** | 64.28 | 20.20 | 0.558 | 0.439 | 15.6 | 0.713 | 39.3 | 0.971 | 39.3 | 74 | 0.3983 |
| PPO PSNR (A5) | 63.60 | 32.50 | 0.580 | 0.461 | 15.8 | 0.715 | 69.6 | 0.989 | 40.9 | 77 | 0.4227 |
| PPO task, ckpt 2M | 64.28 | 20.07 | 0.558 | 0.439 | 15.5 | 0.713 | 39.0 | 0.971 | 39.3 | 74 | 0.3980 |

### phy_mcs_blind, pool testdev (seed 9000..9099)

| policy | R_task | R_PSNR | recall | recall nhỏ | FP/khung giao | precision | Mbit gửi | đúng hạn | năng lượng kJ | thời gian s | AP50 (gộp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T:best_task (fixed_mbt2018-mean_q3) | 28.89 | 22.99 | 0.464 | 0.336 | 10.9 | 0.665 | 49.7 | 0.992 | 39.4 | 74 | 0.3069 |
| tv (task-tuned) (tv:greedy+budget3) | 23.13 | 27.16 | 0.480 | 0.354 | 11.6 | 0.660 | 78.1 | 0.984 | 39.7 | 75 | 0.3287 |
| reuse (task-tuned) (pacedB1.5+budget0.5+reuse) | -8.37 | -23.64 | 0.300 | 0.206 | 10.5 | 0.629 | 28.1 | 0.778 | 40.8 | 77 | 0.2013 |
| budget (task-tuned) (pacedB2+budget1) | 13.92 | 14.85 | 0.423 | 0.317 | 11.4 | 0.653 | 70.2 | 0.911 | 40.5 | 76 | 0.2900 |
| fixed (task-tuned) (fixed_mbt2018-mean_q3) | 28.89 | 22.99 | 0.464 | 0.336 | 10.9 | 0.665 | 49.7 | 0.992 | 39.4 | 74 | 0.3069 |
| A5 T:best (PSNR-tuned) (pacedB2+budget1) | 13.92 | 14.85 | 0.423 | 0.317 | 11.4 | 0.653 | 70.2 | 0.911 | 40.5 | 76 | 0.2900 |
| A5 fixed (PSNR-tuned) (fixed_mbt2018-mean_q3) | 28.89 | 22.99 | 0.464 | 0.336 | 10.9 | 0.665 | 49.7 | 0.992 | 39.4 | 74 | 0.3069 |
| A4 greedy+budget1+reuse (greedy+budget1+reuse) | -28.29 | -32.38 | 0.334 | 0.248 | 11.0 | 0.650 | 49.1 | 0.756 | 39.5 | 74 | 0.2273 |
| **PPO task** | 28.58 | 22.76 | 0.462 | 0.334 | 10.9 | 0.665 | 50.2 | 0.989 | 39.4 | 74 | 0.3058 |
| PPO PSNR (A5) | 17.16 | 22.35 | 0.482 | 0.354 | 11.7 | 0.659 | 87.3 | 0.986 | 43.4 | 82 | 0.3315 |

### So sánh định trước (theo cặp seed; CI mức 1 = seed môi trường; mức 2 = + video; mức 3 = + seed train)

| regime | pool | so sánh | chỉ số | Δ [CI mức 1] | CI mức 2 | CI mức 3 |
|---|---|---|---|---|---|---|
| ref | test | PPO task − PPO PSNR | R_task | +1.75 [+1.34, +2.16] | [+0.99, +3.05] | [+0.56, +2.99] |
| ref | test | PPO task − PPO PSNR | R_PSNR | -10.74 [-11.40, -10.08] | [-12.53, -7.09] | [-12.58, -7.13] |
| ref | test | PPO task − PPO PSNR | recall | -0.0075 [-0.0082, -0.0068] | [-0.0113, -0.0048] | [-0.0109, -0.0043] |
| ref | test | PPO task − PPO PSNR | recall nhỏ | -0.0097 [-0.0106, -0.0089] | [-0.0160, -0.0064] | [-0.0152, -0.0061] |
| ref | test | PPO task − PPO PSNR | Mbit | -38.8 [-41.6, -36.0] | [-42.0, -31.7] | [-41.4, -31.0] |
| ref | test | PPO task − PPO PSNR | đúng hạn | -0.0001 [-0.0007, +0.0006] | [-0.0007, +0.0024] | [-0.0010, +0.0028] |
| ref | test | PPO task − PPO PSNR | AP50 mỗi episode (gộp: -0.0086) | -0.0082 [-0.0093, -0.0071] | — | — |
| ref | test | T:best_task − A5 T:best | R_task | +8.12 [+7.46, +8.78] | [+6.45, +10.97] | [+6.45, +10.97] |
| ref | test | T:best_task − A5 T:best | R_PSNR | -10.62 [-11.49, -9.76] | [-14.47, -5.67] | [-14.47, -5.67] |
| ref | test | T:best_task − A5 T:best | recall | -0.0070 [-0.0082, -0.0057] | [-0.0119, +0.0013] | [-0.0119, +0.0013] |
| ref | test | T:best_task − A5 T:best | recall nhỏ | -0.0114 [-0.0127, -0.0101] | [-0.0212, -0.0044] | [-0.0212, -0.0044] |
| ref | test | T:best_task − A5 T:best | Mbit | -61.9 [-63.3, -60.4] | [-63.0, -56.5] | [-63.0, -56.5] |
| ref | test | T:best_task − A5 T:best | đúng hạn | +0.0079 [+0.0063, +0.0096] | [+0.0042, +0.0152] | [+0.0042, +0.0152] |
| ref | test | T:best_task − A5 T:best | AP50 mỗi episode (gộp: -0.0089) | -0.0081 [-0.0098, -0.0064] | — | — |
| ref | test | PPO task − T:best_task | R_task | -0.09 [-0.12, -0.05] | [-0.16, +0.02] | [-0.15, +0.06] |
| ref | test | PPO task − T:best_task | R_PSNR | -0.31 [-0.36, -0.25] | [-0.43, -0.15] | [-0.67, +0.01] |
| ref | test | PPO task − T:best_task | recall | -0.0004 [-0.0005, -0.0003] | [-0.0007, -0.0001] | [-0.0011, +0.0001] |
| ref | test | PPO task − T:best_task | recall nhỏ | -0.0004 [-0.0006, -0.0003] | [-0.0007, -0.0001] | [-0.0013, +0.0000] |
| ref | test | PPO task − T:best_task | Mbit | -0.5 [-0.6, -0.4] | [-0.7, -0.3] | [-1.4, -0.1] |
| ref | test | PPO task − T:best_task | đúng hạn | -0.0006 [-0.0008, -0.0004] | [-0.0009, +0.0002] | [-0.0010, +0.0002] |
| ref | test | PPO task − T:best_task | AP50 mỗi episode (gộp: -0.0004) | -0.0005 [-0.0006, -0.0003] | — | — |
| ref | testdev | PPO task − PPO PSNR | R_task | +3.97 [+3.22, +4.72] | [+1.80, +5.63] | [+1.07, +5.83] |
| ref | testdev | PPO task − PPO PSNR | R_PSNR | -3.06 [-3.50, -2.62] | [-4.71, -1.03] | [-4.92, -1.17] |
| ref | testdev | PPO task − PPO PSNR | recall | -0.0038 [-0.0046, -0.0031] | [-0.0079, -0.0017] | [-0.0075, -0.0012] |
| ref | testdev | PPO task − PPO PSNR | recall nhỏ | -0.0048 [-0.0056, -0.0040] | [-0.0083, -0.0021] | [-0.0082, -0.0020] |
| ref | testdev | PPO task − PPO PSNR | Mbit | -32.3 [-35.6, -29.1] | [-37.4, -23.2] | [-36.3, -21.5] |
| ref | testdev | PPO task − PPO PSNR | đúng hạn | +0.0017 [+0.0006, +0.0027] | [-0.0047, +0.0041] | [-0.0081, +0.0042] |
| ref | testdev | PPO task − PPO PSNR | AP50 mỗi episode (gộp: -0.0021) | -0.0030 [-0.0038, -0.0022] | — | — |
| ref | testdev | T:best_task − A5 T:best | R_task | +9.45 [+8.80, +10.10] | [-2.72, +12.06] | [-2.72, +12.06] |
| ref | testdev | T:best_task − A5 T:best | R_PSNR | +0.05 [-0.67, +0.77] | [-5.64, +2.81] | [-5.64, +2.81] |
| ref | testdev | T:best_task − A5 T:best | recall | +0.0056 [+0.0040, +0.0072] | [-0.0039, +0.0151] | [-0.0039, +0.0151] |
| ref | testdev | T:best_task − A5 T:best | recall nhỏ | +0.0010 [-0.0005, +0.0024] | [-0.0080, +0.0095] | [-0.0080, +0.0095] |
| ref | testdev | T:best_task − A5 T:best | Mbit | -46.3 [-48.8, -43.8] | [-60.0, -10.1] | [-60.0, -10.1] |
| ref | testdev | T:best_task − A5 T:best | đúng hạn | +0.0157 [+0.0136, +0.0178] | [+0.0043, +0.0249] | [+0.0043, +0.0249] |
| ref | testdev | T:best_task − A5 T:best | AP50 mỗi episode (gộp: +0.0065) | +0.0053 [+0.0037, +0.0069] | — | — |
| ref | testdev | PPO task − T:best_task | R_task | +0.34 [+0.17, +0.50] | [+0.02, +4.21] | [+0.03, +2.80] |
| ref | testdev | PPO task − T:best_task | R_PSNR | -0.11 [-0.23, +0.01] | [-0.31, +2.67] | [-0.55, +1.45] |
| ref | testdev | PPO task − T:best_task | recall | -0.0006 [-0.0008, -0.0004] | [-0.0013, +0.0020] | [-0.0016, +0.0016] |
| ref | testdev | PPO task − T:best_task | recall nhỏ | -0.0007 [-0.0009, -0.0005] | [-0.0018, +0.0014] | [-0.0021, +0.0014] |
| ref | testdev | PPO task − T:best_task | Mbit | -1.5 [-1.7, -1.3] | [-6.7, -0.7] | [-6.3, -0.3] |
| ref | testdev | PPO task − T:best_task | đúng hạn | +0.0003 [-0.0001, +0.0006] | [-0.0002, +0.0096] | [-0.0002, +0.0067] |
| ref | testdev | PPO task − T:best_task | AP50 mỗi episode (gộp: -0.0010) | -0.0009 [-0.0012, -0.0006] | — | — |
| K5 | test | PPO task − PPO PSNR | R_task | -0.64 [-1.20, -0.09] | [-5.67, +2.66] | [-6.15, +2.79] |
| K5 | test | PPO task − PPO PSNR | R_PSNR | -14.61 [-15.26, -13.95] | [-20.53, -9.55] | [-20.92, -9.18] |
| K5 | test | PPO task − PPO PSNR | recall | -0.0135 [-0.0144, -0.0127] | [-0.0200, -0.0083] | [-0.0201, -0.0072] |
| K5 | test | PPO task − PPO PSNR | recall nhỏ | -0.0139 [-0.0148, -0.0130] | [-0.0209, -0.0078] | [-0.0207, -0.0074] |
| K5 | test | PPO task − PPO PSNR | Mbit | -32.4 [-32.9, -31.9] | [-33.6, -30.6] | [-39.7, -24.8] |
| K5 | test | PPO task − PPO PSNR | đúng hạn | -0.0118 [-0.0130, -0.0105] | [-0.0272, -0.0019] | [-0.0278, -0.0019] |
| K5 | test | PPO task − PPO PSNR | AP50 mỗi episode (gộp: -0.0164) | -0.0163 [-0.0173, -0.0153] | — | — |
| K5 | test | T:best_task − A5 T:best | R_task | +11.55 [+10.62, +12.48] | [+8.14, +15.59] | [+8.14, +15.59] |
| K5 | test | T:best_task − A5 T:best | R_PSNR | +3.39 [+2.50, +4.29] | [+1.43, +5.23] | [+1.43, +5.23] |
| K5 | test | T:best_task − A5 T:best | recall | +0.0026 [+0.0010, +0.0043] | [-0.0014, +0.0075] | [-0.0014, +0.0075] |
| K5 | test | T:best_task − A5 T:best | recall nhỏ | +0.0018 [+0.0003, +0.0032] | [-0.0018, +0.0057] | [-0.0018, +0.0057] |
| K5 | test | T:best_task − A5 T:best | Mbit | -22.5 [-23.4, -21.7] | [-27.3, -19.2] | [-27.3, -19.2] |
| K5 | test | T:best_task − A5 T:best | đúng hạn | +0.0107 [+0.0084, +0.0129] | [+0.0030, +0.0184] | [+0.0030, +0.0184] |
| K5 | test | T:best_task − A5 T:best | AP50 mỗi episode (gộp: +0.0001) | +0.0006 [-0.0008, +0.0021] | — | — |
| K5 | test | PPO task − T:best_task | R_task | +3.03 [+2.36, +3.70] | [-2.95, +8.67] | [-2.80, +8.84] |
| K5 | test | PPO task − T:best_task | R_PSNR | -16.20 [-17.05, -15.34] | [-23.93, -6.86] | [-24.07, -7.95] |
| K5 | test | PPO task − T:best_task | recall | -0.0058 [-0.0074, -0.0042] | [-0.0150, +0.0035] | [-0.0147, +0.0019] |
| K5 | test | PPO task − T:best_task | recall nhỏ | -0.0090 [-0.0105, -0.0075] | [-0.0181, -0.0002] | [-0.0183, -0.0021] |
| K5 | test | PPO task − T:best_task | Mbit | -39.3 [-40.5, -38.2] | [-42.8, -34.0] | [-47.0, -33.0] |
| K5 | test | PPO task − T:best_task | đúng hạn | -0.0021 [-0.0042, +0.0001] | [-0.0208, +0.0183] | [-0.0208, +0.0183] |
| K5 | test | PPO task − T:best_task | AP50 mỗi episode (gộp: -0.0157) | -0.0154 [-0.0171, -0.0137] | — | — |
| K5 | testdev | PPO task − PPO PSNR | R_task | +8.23 [+7.64, +8.81] | [+4.23, +11.23] | [+1.80, +12.46] |
| K5 | testdev | PPO task − PPO PSNR | R_PSNR | -1.94 [-2.58, -1.29] | [-6.62, +3.17] | [-5.62, +3.40] |
| K5 | testdev | PPO task − PPO PSNR | recall | -0.0097 [-0.0104, -0.0090] | [-0.0154, -0.0058] | [-0.0155, -0.0052] |
| K5 | testdev | PPO task − PPO PSNR | recall nhỏ | -0.0100 [-0.0107, -0.0093] | [-0.0166, -0.0066] | [-0.0173, -0.0064] |
| K5 | testdev | PPO task − PPO PSNR | Mbit | -34.6 [-35.5, -33.8] | [-38.3, -30.2] | [-41.2, -26.4] |
| K5 | testdev | PPO task − PPO PSNR | đúng hạn | +0.0030 [+0.0022, +0.0039] | [-0.0047, +0.0059] | [-0.0065, +0.0067] |
| K5 | testdev | PPO task − PPO PSNR | AP50 mỗi episode (gộp: -0.0138) | -0.0142 [-0.0149, -0.0134] | — | — |
| K5 | testdev | T:best_task − A5 T:best | R_task | +7.39 [+6.62, +8.17] | [+5.36, +14.13] | [+5.36, +14.13] |
| K5 | testdev | T:best_task − A5 T:best | R_PSNR | +3.44 [+2.69, +4.18] | [+0.92, +10.15] | [+0.92, +10.15] |
| K5 | testdev | T:best_task − A5 T:best | recall | +0.0045 [+0.0030, +0.0061] | [-0.0031, +0.0231] | [-0.0031, +0.0231] |
| K5 | testdev | T:best_task − A5 T:best | recall nhỏ | -0.0004 [-0.0018, +0.0009] | [-0.0065, +0.0051] | [-0.0065, +0.0051] |
| K5 | testdev | T:best_task − A5 T:best | Mbit | -16.2 [-17.2, -15.2] | [-20.1, -14.8] | [-20.1, -14.8] |
| K5 | testdev | T:best_task − A5 T:best | đúng hạn | +0.0043 [+0.0022, +0.0063] | [-0.0020, +0.0267] | [-0.0020, +0.0267] |
| K5 | testdev | T:best_task − A5 T:best | AP50 mỗi episode (gộp: +0.0039) | +0.0035 [+0.0021, +0.0049] | — | — |
| K5 | testdev | PPO task − T:best_task | R_task | +14.73 [+13.94, +15.51] | [+6.36, +23.37] | [+6.33, +22.66] |
| K5 | testdev | PPO task − T:best_task | R_PSNR | +6.09 [+5.08, +7.09] | [-5.75, +20.26] | [-4.38, +19.35] |
| K5 | testdev | PPO task − T:best_task | recall | +0.0243 [+0.0218, +0.0268] | [-0.0005, +0.0661] | [+0.0013, +0.0648] |
| K5 | testdev | PPO task − T:best_task | recall nhỏ | +0.0098 [+0.0076, +0.0120] | [-0.0007, +0.0318] | [+0.0005, +0.0318] |
| K5 | testdev | PPO task − T:best_task | Mbit | -19.8 [-21.4, -18.2] | [-34.6, +14.6] | [-33.9, +15.9] |
| K5 | testdev | PPO task − T:best_task | đúng hạn | +0.0570 [+0.0538, +0.0603] | [+0.0153, +0.1335] | [+0.0156, +0.1308] |
| K5 | testdev | PPO task − T:best_task | AP50 mỗi episode (gộp: +0.0062) | +0.0057 [+0.0034, +0.0080] | — | — |
| phy_mcs_blind | test | PPO task − PPO PSNR | R_task | +0.69 [-0.13, +1.50] | [-2.12, +5.83] | [-2.09, +5.07] |
| phy_mcs_blind | test | PPO task − PPO PSNR | R_PSNR | -12.30 [-13.31, -11.28] | [-15.71, -6.71] | [-15.87, -6.84] |
| phy_mcs_blind | test | PPO task − PPO PSNR | recall | -0.0214 [-0.0229, -0.0198] | [-0.0291, -0.0110] | [-0.0298, -0.0114] |
| phy_mcs_blind | test | PPO task − PPO PSNR | recall nhỏ | -0.0226 [-0.0242, -0.0210] | [-0.0320, -0.0107] | [-0.0323, -0.0111] |
| phy_mcs_blind | test | PPO task − PPO PSNR | Mbit | -30.3 [-31.2, -29.5] | [-31.8, -26.8] | [-33.3, -26.0] |
| phy_mcs_blind | test | PPO task − PPO PSNR | đúng hạn | -0.0181 [-0.0206, -0.0157] | [-0.0355, -0.0010] | [-0.0359, -0.0012] |
| phy_mcs_blind | test | PPO task − PPO PSNR | AP50 mỗi episode (gộp: -0.0245) | -0.0245 [-0.0262, -0.0227] | — | — |
| phy_mcs_blind | test | T:best_task − A5 T:best | R_task | +9.59 [+8.45, +10.72] | [+7.03, +13.39] | [+7.03, +13.39] |
| phy_mcs_blind | test | T:best_task − A5 T:best | R_PSNR | -7.43 [-9.01, -5.86] | [-11.17, -1.07] | [-11.17, -1.07] |
| phy_mcs_blind | test | T:best_task − A5 T:best | recall | +0.0088 [+0.0055, +0.0121] | [-0.0014, +0.0244] | [-0.0014, +0.0244] |
| phy_mcs_blind | test | T:best_task − A5 T:best | recall nhỏ | +0.0021 [-0.0009, +0.0051] | [-0.0088, +0.0145] | [-0.0088, +0.0145] |
| phy_mcs_blind | test | T:best_task − A5 T:best | Mbit | -36.5 [-38.4, -34.6] | [-40.0, -29.2] | [-40.0, -29.2] |
| phy_mcs_blind | test | T:best_task − A5 T:best | đúng hạn | +0.0242 [+0.0193, +0.0290] | [+0.0070, +0.0466] | [+0.0070, +0.0466] |
| phy_mcs_blind | test | T:best_task − A5 T:best | AP50 mỗi episode (gộp: -0.0042) | -0.0045 [-0.0075, -0.0014] | — | — |
| phy_mcs_blind | test | PPO task − T:best_task | R_task | -0.08 [-0.20, +0.04] | [-0.45, +0.15] | [-0.47, +0.12] |
| phy_mcs_blind | test | PPO task − T:best_task | R_PSNR | +0.20 [+0.06, +0.34] | [-0.24, +0.44] | [-0.32, +0.45] |
| phy_mcs_blind | test | PPO task − T:best_task | recall | -0.0004 [-0.0009, +0.0001] | [-0.0017, +0.0003] | [-0.0020, +0.0005] |
| phy_mcs_blind | test | PPO task − T:best_task | recall nhỏ | -0.0003 [-0.0008, +0.0002] | [-0.0016, +0.0004] | [-0.0018, +0.0008] |
| phy_mcs_blind | test | PPO task − T:best_task | Mbit | +0.6 [+0.5, +0.7] | [+0.2, +0.7] | [+0.2, +0.8] |
| phy_mcs_blind | test | PPO task − T:best_task | đúng hạn | -0.0005 [-0.0011, +0.0002] | [-0.0027, +0.0006] | [-0.0029, +0.0007] |
| phy_mcs_blind | test | PPO task − T:best_task | AP50 mỗi episode (gộp: -0.0001) | -0.0003 [-0.0008, +0.0002] | — | — |
| phy_mcs_blind | testdev | PPO task − PPO PSNR | R_task | +11.42 [+10.65, +12.20] | [+7.71, +18.32] | [+6.91, +21.07] |
| phy_mcs_blind | testdev | PPO task − PPO PSNR | R_PSNR | +0.41 [-0.57, +1.39] | [-3.87, +9.17] | [-3.96, +11.00] |
| phy_mcs_blind | testdev | PPO task − PPO PSNR | recall | -0.0200 [-0.0216, -0.0185] | [-0.0307, -0.0145] | [-0.0303, -0.0137] |
| phy_mcs_blind | testdev | PPO task − PPO PSNR | recall nhỏ | -0.0201 [-0.0216, -0.0187] | [-0.0301, -0.0137] | [-0.0321, -0.0129] |
| phy_mcs_blind | testdev | PPO task − PPO PSNR | Mbit | -37.2 [-38.5, -35.9] | [-44.0, -33.4] | [-45.7, -30.9] |
| phy_mcs_blind | testdev | PPO task − PPO PSNR | đúng hạn | +0.0029 [+0.0011, +0.0047] | [-0.0173, +0.0073] | [-0.0126, +0.0087] |
| phy_mcs_blind | testdev | PPO task − PPO PSNR | AP50 mỗi episode (gộp: -0.0257) | -0.0268 [-0.0287, -0.0249] | — | — |
| phy_mcs_blind | testdev | T:best_task − A5 T:best | R_task | +14.97 [+13.89, +16.05] | [+11.50, +21.73] | [+11.50, +21.73] |
| phy_mcs_blind | testdev | T:best_task − A5 T:best | R_PSNR | +8.14 [+6.75, +9.54] | [+2.50, +18.19] | [+2.50, +18.19] |
| phy_mcs_blind | testdev | T:best_task − A5 T:best | recall | +0.0403 [+0.0354, +0.0453] | [+0.0184, +0.0922] | [+0.0184, +0.0922] |
| phy_mcs_blind | testdev | T:best_task − A5 T:best | recall nhỏ | +0.0191 [+0.0151, +0.0230] | [+0.0042, +0.0470] | [+0.0042, +0.0470] |
| phy_mcs_blind | testdev | T:best_task − A5 T:best | Mbit | -20.6 [-22.4, -18.7] | [-29.6, +3.7] | [-29.6, +3.7] |
| phy_mcs_blind | testdev | T:best_task − A5 T:best | đúng hạn | +0.0806 [+0.0746, +0.0866] | [+0.0432, +0.1604] | [+0.0432, +0.1604] |
| phy_mcs_blind | testdev | T:best_task − A5 T:best | AP50 mỗi episode (gộp: +0.0169) | +0.0154 [+0.0113, +0.0196] | — | — |
| phy_mcs_blind | testdev | PPO task − T:best_task | R_task | -0.31 [-0.51, -0.12] | [-1.78, +0.01] | [-1.86, -0.02] |
| phy_mcs_blind | testdev | PPO task − T:best_task | R_PSNR | -0.23 [-0.45, -0.01] | [-2.26, +0.15] | [-2.35, +0.13] |
| phy_mcs_blind | testdev | PPO task − T:best_task | recall | -0.0016 [-0.0024, -0.0008] | [-0.0121, +0.0003] | [-0.0114, +0.0001] |
| phy_mcs_blind | testdev | PPO task − T:best_task | recall nhỏ | -0.0015 [-0.0024, -0.0007] | [-0.0105, +0.0002] | [-0.0097, +0.0002] |
| phy_mcs_blind | testdev | PPO task − T:best_task | Mbit | +0.5 [+0.3, +0.6] | [-1.4, +0.8] | [-1.8, +0.8] |
| phy_mcs_blind | testdev | PPO task − T:best_task | đúng hạn | -0.0024 [-0.0035, -0.0014] | [-0.0162, -0.0001] | [-0.0160, -0.0004] |
| phy_mcs_blind | testdev | PPO task − T:best_task | AP50 mỗi episode (gộp: -0.0011) | -0.0009 [-0.0015, -0.0003] | — | — |

### Pareto: recall vs Mbit gửi (★ = không bị policy nào khác trội hơn ở cả recall↑ và Mbit↓)

| regime | pool | policy | recall | Mbit | đúng hạn | năng lượng kJ | thời gian s | ★ |
|---|---|---|---|---|---|---|---|---|
| ref | test | PPO PSNR (A5) | 0.5961 | 117.4 | 0.999 | 39.6 | 74 | ★ |
| ref | test | A5 T:best (PSNR-tuned) | 0.5960 | 141.0 | 0.992 | 40.5 | 76 |  |
| ref | test | A4 greedy+budget1+reuse | 0.5948 | 123.2 | 0.994 | 39.1 | 73 |  |
| ref | test | tv (task-tuned) | 0.5922 | 106.3 | 0.998 | 39.0 | 73 | ★ |
| ref | test | T:best_task | 0.5891 | 79.2 | 0.999 | 39.0 | 73 | ★ |
| ref | test | fixed (task-tuned) | 0.5891 | 79.2 | 0.999 | 39.0 | 73 | ★ |
| ref | test | A5 fixed (PSNR-tuned) | 0.5891 | 79.2 | 0.999 | 39.0 | 73 | ★ |
| ref | test | **PPO task** | 0.5887 | 78.6 | 0.999 | 39.0 | 73 | ★ |
| ref | test | PPO task, ckpt 2M | 0.5859 | 74.5 | 0.996 | 39.0 | 73 | ★ |
| ref | test | reuse (task-tuned) | 0.5852 | 98.9 | 0.986 | 39.0 | 73 |  |
| ref | test | budget (task-tuned) | 0.5779 | 98.7 | 0.977 | 39.3 | 74 |  |
| ref | testdev | PPO PSNR (A5) | 0.5017 | 140.3 | 0.996 | 40.8 | 77 | ★ |
| ref | testdev | T:best_task | 0.4985 | 109.4 | 0.997 | 39.2 | 74 | ★ |
| ref | testdev | fixed (task-tuned) | 0.4985 | 109.4 | 0.997 | 39.2 | 74 | ★ |
| ref | testdev | A5 fixed (PSNR-tuned) | 0.4985 | 109.4 | 0.997 | 39.2 | 74 | ★ |
| ref | testdev | **PPO task** | 0.4979 | 107.9 | 0.998 | 39.2 | 74 | ★ |
| ref | testdev | tv (task-tuned) | 0.4972 | 117.0 | 0.995 | 39.1 | 73 |  |
| ref | testdev | A5 T:best (PSNR-tuned) | 0.4929 | 155.7 | 0.982 | 40.9 | 77 |  |
| ref | testdev | A4 greedy+budget1+reuse | 0.4855 | 124.6 | 0.977 | 39.1 | 73 |  |
| ref | testdev | reuse (task-tuned) | 0.4454 | 89.7 | 0.932 | 38.9 | 73 | ★ |
| ref | testdev | budget (task-tuned) | 0.4446 | 89.7 | 0.932 | 39.2 | 74 |  |
| K5 | test | PPO PSNR (A5) | 0.5781 | 109.3 | 0.990 | 65.9 | 74 | ★ |
| K5 | test | T:best_task | 0.5704 | 116.2 | 0.980 | 65.1 | 73 |  |
| K5 | test | tv (task-tuned) | 0.5704 | 116.2 | 0.980 | 65.1 | 73 |  |
| K5 | test | A5 T:best (PSNR-tuned) | 0.5677 | 138.8 | 0.970 | 66.8 | 75 |  |
| K5 | test | reuse (task-tuned) | 0.5673 | 125.7 | 0.972 | 65.2 | 73 |  |
| K5 | test | A4 greedy+budget1+reuse | 0.5673 | 125.7 | 0.972 | 65.2 | 73 |  |
| K5 | test | **PPO task** | 0.5646 | 76.9 | 0.978 | 65.0 | 73 | ★ |
| K5 | test | PPO task, ckpt 2M | 0.5635 | 74.2 | 0.977 | 65.0 | 73 | ★ |
| K5 | test | fixed (task-tuned) | 0.5565 | 64.4 | 0.969 | 65.7 | 74 | ★ |
| K5 | test | A5 fixed (PSNR-tuned) | 0.5565 | 64.4 | 0.969 | 65.7 | 74 | ★ |
| K5 | test | budget (task-tuned) | 0.5241 | 118.4 | 0.919 | 68.4 | 77 |  |
| K5 | testdev | PPO PSNR (A5) | 0.4791 | 130.2 | 0.993 | 67.5 | 76 | ★ |
| K5 | testdev | **PPO task** | 0.4695 | 95.5 | 0.996 | 65.1 | 73 | ★ |
| K5 | testdev | fixed (task-tuned) | 0.4622 | 82.5 | 0.989 | 65.8 | 74 | ★ |
| K5 | testdev | A5 fixed (PSNR-tuned) | 0.4622 | 82.5 | 0.989 | 65.8 | 74 | ★ |
| K5 | testdev | T:best_task | 0.4451 | 115.3 | 0.939 | 65.1 | 73 |  |
| K5 | testdev | tv (task-tuned) | 0.4451 | 115.3 | 0.939 | 65.1 | 73 |  |
| K5 | testdev | A5 T:best (PSNR-tuned) | 0.4406 | 131.6 | 0.935 | 66.3 | 75 |  |
| K5 | testdev | reuse (task-tuned) | 0.4327 | 117.4 | 0.928 | 65.1 | 73 |  |
| K5 | testdev | A4 greedy+budget1+reuse | 0.4327 | 117.4 | 0.928 | 65.1 | 73 |  |
| K5 | testdev | budget (task-tuned) | 0.4056 | 112.4 | 0.888 | 68.3 | 77 |  |
| phy_mcs_blind | test | tv (task-tuned) | 0.5805 | 73.2 | 0.988 | 39.5 | 74 | ★ |
| phy_mcs_blind | test | PPO PSNR (A5) | 0.5797 | 69.6 | 0.989 | 40.9 | 77 | ★ |
| phy_mcs_blind | test | T:best_task | 0.5587 | 38.7 | 0.971 | 39.3 | 74 | ★ |
| phy_mcs_blind | test | fixed (task-tuned) | 0.5587 | 38.7 | 0.971 | 39.3 | 74 | ★ |
| phy_mcs_blind | test | A5 fixed (PSNR-tuned) | 0.5587 | 38.7 | 0.971 | 39.3 | 74 | ★ |
| phy_mcs_blind | test | **PPO task** | 0.5583 | 39.3 | 0.971 | 39.3 | 74 |  |
| phy_mcs_blind | test | PPO task, ckpt 2M | 0.5582 | 39.0 | 0.971 | 39.3 | 74 |  |
| phy_mcs_blind | test | budget (task-tuned) | 0.5499 | 75.2 | 0.947 | 40.7 | 77 |  |
| phy_mcs_blind | test | A5 T:best (PSNR-tuned) | 0.5499 | 75.2 | 0.947 | 40.7 | 77 |  |
| phy_mcs_blind | test | A4 greedy+budget1+reuse | 0.4296 | 51.2 | 0.766 | 39.6 | 74 |  |
| phy_mcs_blind | test | reuse (task-tuned) | 0.4118 | 31.5 | 0.804 | 41.0 | 77 | ★ |
| phy_mcs_blind | testdev | PPO PSNR (A5) | 0.4821 | 87.3 | 0.986 | 43.4 | 82 | ★ |
| phy_mcs_blind | testdev | tv (task-tuned) | 0.4804 | 78.1 | 0.984 | 39.7 | 75 | ★ |
| phy_mcs_blind | testdev | T:best_task | 0.4636 | 49.7 | 0.992 | 39.4 | 74 | ★ |
| phy_mcs_blind | testdev | fixed (task-tuned) | 0.4636 | 49.7 | 0.992 | 39.4 | 74 | ★ |
| phy_mcs_blind | testdev | A5 fixed (PSNR-tuned) | 0.4636 | 49.7 | 0.992 | 39.4 | 74 | ★ |
| phy_mcs_blind | testdev | **PPO task** | 0.4620 | 50.2 | 0.989 | 39.4 | 74 |  |
| phy_mcs_blind | testdev | budget (task-tuned) | 0.4233 | 70.2 | 0.911 | 40.5 | 76 |  |
| phy_mcs_blind | testdev | A5 T:best (PSNR-tuned) | 0.4233 | 70.2 | 0.911 | 40.5 | 76 |  |
| phy_mcs_blind | testdev | A4 greedy+budget1+reuse | 0.3340 | 49.1 | 0.756 | 39.5 | 74 | ★ |
| phy_mcs_blind | testdev | reuse (task-tuned) | 0.2997 | 28.1 | 0.778 | 40.8 | 77 | ★ |

### Chọn mode theo trạng thái — ref, pool test

| policy | toàn bộ |
|---|---|
| **PPO task** | avif_50 99% |
| PPO PSNR (A5) | avif_50 49% · avif_80 50% |
| T:best_task | avif_50 100% |
| A5 T:best (PSNR-tuned) | avif_50 13% · avif_80 83% |

| log10_probe_bits | tam phân vị thấp (≤4.72) | giữa | cao (>4.88) |
|---|---|---|---|
| **PPO task** | avif_50 99% | avif_50 99% | avif_50 99% |
| PPO PSNR (A5) | avif_50 39% · avif_80 60% | avif_50 50% · avif_80 49% | avif_50 58% · avif_80 41% |
| T:best_task | avif_50 100% | avif_50 100% | avif_50 100% |
| A5 T:best (PSNR-tuned) | avif_50 6% · avif_80 92% | avif_50 11% · avif_80 86% | avif_50 22% · avif_80 71% |

| vhat_max | tam phân vị thấp (≤0.70) | giữa | cao (>1.02) |
|---|---|---|---|
| **PPO task** | avif_50 99% | avif_50 99% | avif_50 99% |
| PPO PSNR (A5) | avif_50 39% · avif_80 60% | avif_50 50% · avif_80 49% | avif_50 58% · avif_80 41% |
| T:best_task | avif_50 100% | avif_50 100% | avif_50 100% |
| A5 T:best (PSNR-tuned) | avif_50 6% · avif_80 92% | avif_50 11% · avif_80 86% | avif_50 22% · avif_80 71% |

| backlog_s | tam phân vị thấp (≤0.69) | giữa | cao (>0.93) |
|---|---|---|---|
| **PPO task** | avif_50 99% | avif_50 99% | avif_50 98% |
| PPO PSNR (A5) | avif_50 30% · avif_80 70% | avif_50 51% · avif_80 48% | avif_50 73% · avif_80 24% |
| T:best_task | avif_50 100% | avif_50 100% | avif_50 100% |
| A5 T:best (PSNR-tuned) | avif_80 99% | avif_80 100% | avif_50 33% · avif_80 57% |

| snr_db | tam phân vị thấp (≤12.61) | giữa | cao (>15.40) |
|---|---|---|---|
| **PPO task** | avif_50 98% | avif_50 99% | avif_50 99% |
| PPO PSNR (A5) | avif_50 71% · avif_80 28% | avif_50 50% · avif_80 49% | avif_50 28% · avif_80 72% |
| T:best_task | avif_50 100% | avif_50 100% | avif_50 100% |
| A5 T:best (PSNR-tuned) | avif_50 23% · avif_80 68% | avif_50 12% · avif_80 85% | avif_80 95% |

| slack_s | tam phân vị thấp (≤6.59) | giữa | cao (>9.20) |
|---|---|---|---|
| **PPO task** | avif_50 99% | avif_50 99% | avif_50 99% |
| PPO PSNR (A5) | avif_50 53% · avif_80 46% | avif_50 50% · avif_80 49% | avif_50 45% · avif_80 54% |
| T:best_task | avif_50 100% | avif_50 100% | avif_50 100% |
| A5 T:best (PSNR-tuned) | avif_50 13% · avif_80 83% | avif_50 13% · avif_80 83% | avif_50 13% · avif_80 83% |

### Chọn mode theo trạng thái — K5, pool test

| policy | toàn bộ |
|---|---|
| **PPO task** | mbt2018-mean_q3 82% · avif_50 16% |
| PPO PSNR (A5) | mbt2018-mean_q3 33% · avif_50 67% |
| T:best_task | mbt2018-mean_q3 15% · avif_50 74% · avif_80 5% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 14% · avif_50 50% · avif_80 28% |

| log10_probe_bits | tam phân vị thấp (≤4.71) | giữa | cao (>4.88) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 81% · avif_50 16% | mbt2018-mean_q3 81% · avif_50 16% | mbt2018-mean_q3 82% · avif_50 16% |
| PPO PSNR (A5) | mbt2018-mean_q3 30% · avif_50 69% | mbt2018-mean_q3 33% · avif_50 66% | mbt2018-mean_q3 35% · avif_50 64% |
| T:best_task | mbt2018-mean_q3 5% · avif_50 79% · avif_80 14% | mbt2018-mean_q3 12% · avif_50 83% | mbt2018-mean_q3 27% · avif_50 62% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 6% · avif_50 30% · avif_80 61% | mbt2018-mean_q3 12% · avif_50 60% · avif_80 22% | mbt2018-mean_q3 25% · avif_50 59% |

| vhat_max | tam phân vị thấp (≤0.70) | giữa | cao (>1.02) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 81% · avif_50 16% | mbt2018-mean_q3 81% · avif_50 16% | mbt2018-mean_q3 82% · avif_50 16% |
| PPO PSNR (A5) | mbt2018-mean_q3 30% · avif_50 69% | mbt2018-mean_q3 33% · avif_50 66% | mbt2018-mean_q3 35% · avif_50 64% |
| T:best_task | mbt2018-mean_q3 5% · avif_50 79% · avif_80 14% | mbt2018-mean_q3 12% · avif_50 83% | mbt2018-mean_q3 27% · avif_50 62% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 6% · avif_50 30% · avif_80 61% | mbt2018-mean_q3 12% · avif_50 60% · avif_80 22% | mbt2018-mean_q3 25% · avif_50 59% |

| backlog_s | tam phân vị thấp (≤1.01) | giữa | cao (>1.39) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 79% · avif_50 18% | mbt2018-mean_q3 81% · avif_50 16% | mbt2018-mean_q3 85% · avif_50 14% |
| PPO PSNR (A5) | mbt2018-mean_q3 24% · avif_50 76% | mbt2018-mean_q3 34% · avif_50 65% | mbt2018-mean_q3 44% · avif_50 55% |
| T:best_task | avif_50 78% · avif_80 21% | avif_50 98% | mbt2018-mean_q3 36% · avif_50 53% |
| A5 T:best (PSNR-tuned) | avif_50 8% · avif_80 92% | avif_50 98% | mbt2018-mean_q3 38% · avif_50 43% |

| snr_db | tam phân vị thấp (≤12.46) | giữa | cao (>15.11) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 86% · avif_50 13% | mbt2018-mean_q3 81% · avif_50 17% | mbt2018-mean_q3 78% · avif_50 19% |
| PPO PSNR (A5) | mbt2018-mean_q3 35% · avif_50 64% | mbt2018-mean_q3 34% · avif_50 65% | mbt2018-mean_q3 29% · avif_50 70% |
| T:best_task | mbt2018-mean_q3 29% · avif_50 62% | mbt2018-mean_q3 13% · avif_50 80% | avif_50 81% · avif_80 13% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 23% · avif_50 48% · avif_80 18% | mbt2018-mean_q3 14% · avif_50 53% · avif_80 26% | mbt2018-mean_q3 6% · avif_50 49% · avif_80 42% |

| slack_s | tam phân vị thấp (≤6.61) | giữa | cao (>9.28) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 82% · avif_50 15% | mbt2018-mean_q3 81% · avif_50 16% | mbt2018-mean_q3 81% · avif_50 16% |
| PPO PSNR (A5) | mbt2018-mean_q3 33% · avif_50 66% | mbt2018-mean_q3 33% · avif_50 66% | mbt2018-mean_q3 32% · avif_50 67% |
| T:best_task | mbt2018-mean_q3 15% · avif_50 79% | mbt2018-mean_q3 15% · avif_50 78% | mbt2018-mean_q3 15% · avif_50 66% · avif_80 14% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 14% · avif_50 51% · avif_80 27% | mbt2018-mean_q3 14% · avif_50 49% · avif_80 29% | mbt2018-mean_q3 14% · avif_50 50% · avif_80 29% |

### Chọn mode theo trạng thái — phy_mcs_blind, pool test

| policy | toàn bộ |
|---|---|
| **PPO task** | mbt2018-mean_q3 98% |
| PPO PSNR (A5) | mbt2018-mean_q3 22% · avif_50 78% |
| T:best_task | mbt2018-mean_q3 100% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 17% · avif_50 49% · avif_80 23% |

| log10_probe_bits | tam phân vị thấp (≤4.72) | giữa | cao (>4.88) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 98% | mbt2018-mean_q3 98% | mbt2018-mean_q3 98% |
| PPO PSNR (A5) | mbt2018-mean_q3 20% · avif_50 80% | mbt2018-mean_q3 22% · avif_50 78% | mbt2018-mean_q3 25% · avif_50 75% |
| T:best_task | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 8% · avif_50 37% · avif_80 48% | mbt2018-mean_q3 14% · avif_50 58% · avif_80 17% | mbt2018-mean_q3 28% · avif_50 51% |

| vhat_max | tam phân vị thấp (≤0.70) | giữa | cao (>1.02) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 98% | mbt2018-mean_q3 98% | mbt2018-mean_q3 98% |
| PPO PSNR (A5) | mbt2018-mean_q3 20% · avif_50 80% | mbt2018-mean_q3 22% · avif_50 78% | mbt2018-mean_q3 25% · avif_50 75% |
| T:best_task | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 8% · avif_50 37% · avif_80 48% | mbt2018-mean_q3 14% · avif_50 58% · avif_80 17% | mbt2018-mean_q3 28% · avif_50 51% |

| backlog_s | tam phân vị thấp (≤1.09) | giữa | cao (>1.51) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 97% | mbt2018-mean_q3 98% | mbt2018-mean_q3 99% |
| PPO PSNR (A5) | mbt2018-mean_q3 10% · avif_50 90% | mbt2018-mean_q3 20% · avif_50 80% | mbt2018-mean_q3 38% · avif_50 62% |
| T:best_task | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% |
| A5 T:best (PSNR-tuned) | avif_50 28% · avif_80 71% | mbt2018-mean_q3 6% · avif_50 93% | mbt2018-mean_q3 41% · avif_50 26% |

| snr_db | tam phân vị thấp (≤12.61) | giữa | cao (>15.40) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 99% | mbt2018-mean_q3 98% | mbt2018-mean_q3 97% |
| PPO PSNR (A5) | mbt2018-mean_q3 32% · avif_50 68% | mbt2018-mean_q3 22% · avif_50 78% | mbt2018-mean_q3 13% · avif_50 87% |
| T:best_task | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 23% · avif_50 50% · avif_80 10% | mbt2018-mean_q3 16% · avif_50 52% · avif_80 20% | mbt2018-mean_q3 11% · avif_50 44% · avif_80 38% |

| slack_s | tam phân vị thấp (≤6.59) | giữa | cao (>9.20) |
|---|---|---|---|
| **PPO task** | mbt2018-mean_q3 98% | mbt2018-mean_q3 98% | mbt2018-mean_q3 98% |
| PPO PSNR (A5) | mbt2018-mean_q3 22% · avif_50 78% | mbt2018-mean_q3 22% · avif_50 78% | mbt2018-mean_q3 23% · avif_50 77% |
| T:best_task | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% | mbt2018-mean_q3 100% |
| A5 T:best (PSNR-tuned) | mbt2018-mean_q3 17% · avif_50 49% · avif_80 22% | mbt2018-mean_q3 16% · avif_50 48% · avif_80 23% | mbt2018-mean_q3 17% · avif_50 49% · avif_80 23% |

### Độ nhạy β (khám phá; điểm tác vụ trung bình mỗi episode: Σ(TP − β·FP)/V̄ của khung giao đúng hạn)

| regime | pool | so sánh | β = 0 | β = 0,5 | β = 1 |
|---|---|---|---|---|---|
| ref | test | PPO task − PPO PSNR | -0.01 [-0.01, -0.01] | -0.02 [-0.02, -0.02] | -0.03 [-0.03, -0.02] |
| ref | test | T:best_task − A5 T:best | -0.01 [-0.01, -0.01] | -0.02 [-0.02, -0.02] | -0.04 [-0.04, -0.03] |
| ref | test | PPO task − T:best_task | -0.00 [-0.00, -0.00] | -0.00 [-0.00, -0.00] | -0.00 [-0.00, -0.00] |
| ref | testdev | PPO task − PPO PSNR | -0.01 [-0.01, -0.01] | -0.01 [-0.01, -0.01] | -0.01 [-0.02, -0.01] |
| ref | testdev | T:best_task − A5 T:best | +0.01 [+0.01, +0.01] | +0.01 [+0.00, +0.01] | -0.00 [-0.01, +0.00] |
| ref | testdev | PPO task − T:best_task | -0.00 [-0.00, -0.00] | -0.00 [-0.00, -0.00] | -0.00 [-0.00, +0.00] |
| K5 | test | PPO task − PPO PSNR | -0.02 [-0.02, -0.02] | -0.02 [-0.02, -0.02] | -0.02 [-0.03, -0.02] |
| K5 | test | T:best_task − A5 T:best | +0.00 [+0.00, +0.01] | +0.00 [+0.00, +0.01] | +0.00 [-0.00, +0.00] |
| K5 | test | PPO task − T:best_task | -0.01 [-0.01, -0.01] | -0.01 [-0.01, -0.01] | -0.01 [-0.02, -0.01] |
| K5 | testdev | PPO task − PPO PSNR | -0.02 [-0.02, -0.02] | -0.01 [-0.02, -0.01] | -0.01 [-0.01, -0.00] |
| K5 | testdev | T:best_task − A5 T:best | +0.01 [+0.01, +0.01] | +0.01 [+0.01, +0.01] | +0.02 [+0.01, +0.02] |
| K5 | testdev | PPO task − T:best_task | +0.05 [+0.04, +0.05] | +0.06 [+0.05, +0.06] | +0.09 [+0.08, +0.09] |
| phy_mcs_blind | test | PPO task − PPO PSNR | -0.04 [-0.04, -0.03] | -0.04 [-0.04, -0.03] | -0.04 [-0.04, -0.03] |
| phy_mcs_blind | test | T:best_task − A5 T:best | +0.01 [+0.01, +0.02] | +0.01 [+0.01, +0.02] | +0.01 [+0.00, +0.02] |
| phy_mcs_blind | test | PPO task − T:best_task | -0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] |
| phy_mcs_blind | testdev | PPO task − PPO PSNR | -0.04 [-0.04, -0.04] | -0.03 [-0.03, -0.03] | -0.01 [-0.02, -0.01] |
| phy_mcs_blind | testdev | T:best_task − A5 T:best | +0.08 [+0.07, +0.09] | +0.09 [+0.08, +0.10] | +0.13 [+0.11, +0.14] |
| phy_mcs_blind | testdev | PPO task − T:best_task | -0.00 [-0.00, -0.00] | -0.00 [-0.00, -0.00] | -0.00 [-0.01, -0.00] |

