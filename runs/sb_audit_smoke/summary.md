**SB recipe audit — running results**

Updated 2026-10-30T23:49:16+09:00. Experiment: runs\sb_audit_smoke.

The primary BEV comparison is one-step mapping_xt vs sb_ve_xt_endpoint.
mapping_csi_residual is the strong CSI/residual baseline. No statistical conclusion from a partial queue.

| Job / checkpoint | Sent | Mode | mIoU | Vehicle IoU |
|---|---:|---|---:|---:|
| mapping_csi_residual_s0\evaluation_20.json | 15% | one_step | 0.3878 | 0.1228 |
| mapping_csi_residual_s0\evaluation_20.json | 30% | one_step | 0.5045 | 0.2096 |
| mapping_csi_residual_s0\evaluation_20.json | 50% | one_step | 0.6240 | 0.3112 |
| mapping_csi_residual_s0\evaluation_20.json | 70% | one_step | 0.7025 | 0.4083 |
| mapping_csi_residual_s0\evaluation_20.json | 100% | one_step | 0.7740 | 0.5045 |
| mapping_xt_s0\evaluation_20.json | 15% | one_step | 0.2874 | 0.0091 |
| mapping_xt_s0\evaluation_20.json | 30% | one_step | 0.3165 | 0.0104 |
| mapping_xt_s0\evaluation_20.json | 50% | one_step | 0.3544 | 0.0076 |
| mapping_xt_s0\evaluation_20.json | 70% | one_step | 0.3707 | 0.0030 |
| mapping_xt_s0\evaluation_20.json | 100% | one_step | 0.4233 | 0.0000 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 15% | one_step | 0.2809 | 0.0033 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 15% | sde_4 | 0.3012 | 0.0021 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 15% | mean_path_4 | 0.3061 | 0.0000 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 30% | one_step | 0.3120 | 0.0055 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 30% | sde_4 | 0.2979 | 0.0006 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 30% | mean_path_4 | 0.3068 | 0.0000 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 50% | one_step | 0.3509 | 0.0092 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 50% | sde_4 | 0.3161 | 0.0017 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 50% | mean_path_4 | 0.2908 | 0.0000 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 70% | one_step | 0.3663 | 0.0000 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 70% | sde_4 | 0.3129 | 0.0040 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 70% | mean_path_4 | 0.2979 | 0.0000 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 100% | one_step | 0.4238 | 0.0000 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 100% | sde_4 | 0.3198 | 0.0124 |
| sb_ve_xt_endpoint_s0\evaluation_20.json | 100% | mean_path_4 | 0.2893 | 0.0000 |

Image diagnostics (CIFAR validation; not paper-scale reproduction):

| Job / checkpoint | Mode | PSNR | Hole PSNR | SSIM | LPIPS AlexNet, resized 64 |
|---|---|---:|---:|---:|---:|
| image_i2sb_s0\evaluation_20.json | one_step | 12.977 | 6.957 | 0.2159 | 0.4502 |
| image_i2sb_s0\evaluation_20.json | sde_4 | 12.749 | 6.729 | 0.2112 | 0.4594 |
| image_i2sb_s0\evaluation_20.json | sde_16 | 12.734 | 6.714 | 0.2123 | 0.4578 |
| image_mapping_s0\evaluation_20.json | one_step | 15.595 | 9.575 | 0.3283 | 0.4503 |
