**Matched-sampler SB pilot (seed 0, 6k, EMA, one-step unless noted)**

| Model | 15% mIoU | 30% mIoU | 50% mIoU | 70% mIoU | 100% mIoU | veh 70% | veh 100% |
|---|---:|---:|---:|---:|---:|---:|---:|
| map_c_s0 | 0.2444 | 0.2453 | 0.2422 | 0.2410 | 0.2415 | 0.0000 | 0.0000 |
| map_x_s0 | 0.2874 | 0.3109 | 0.3480 | 0.3619 | 0.4091 | 0.0020 | 0.0000 |
| sb_c_p10_s0 | 0.2442 | 0.2440 | 0.2415 | 0.2412 | 0.2415 | 0.0000 | 0.0000 |
| sb_c_p50_s0 | 0.2442 | 0.2446 | 0.2416 | 0.2412 | 0.2415 | 0.0000 | 0.0000 |
| sb_x_p10_s0 | 0.2855 | 0.3137 | 0.3530 | 0.3699 | 0.4257 | 0.0000 | 0.0000 |
| sb_x_p50_s0 | 0.2878 | 0.3151 | 0.3539 | 0.3706 | 0.4238 | 0.0035 | 0.0000 |

Gate vs matched mapping, same seed (pass: d15>=-0.005, d30>=-0.005, d_veh(70/100)>=-0.005).
The pre-registered decision uses seed 0; 'mean' rows average the paired differences over finished seeds.

- sb_c_p10_s0 vs map_c_s0: d15=-0.0002 d30=-0.0013 d_veh=+0.0000 -> PASS
- sb_c_p50_s0 vs map_c_s0: d15=-0.0002 d30=-0.0007 d_veh=+0.0000 -> PASS
- sb_x_p10_s0 vs map_x_s0: d15=-0.0019 d30=+0.0028 d_veh=-0.0010 -> PASS
- sb_x_p50_s0 vs map_x_s0: d15=+0.0004 d30=+0.0042 d_veh=+0.0008 -> PASS

- mean sb_c_p10 over 1 seed(s): d15=-0.0002 d30=-0.0013 d_veh=+0.0000; seeds with SB ahead: 0/0/0
- mean sb_c_p50 over 1 seed(s): d15=-0.0002 d30=-0.0007 d_veh=+0.0000; seeds with SB ahead: 0/0/0
- mean sb_x_p10 over 1 seed(s): d15=-0.0019 d30=+0.0028 d_veh=-0.0010; seeds with SB ahead: 0/1/0
- mean sb_x_p50 over 1 seed(s): d15=+0.0004 d30=+0.0042 d_veh=+0.0008; seeds with SB ahead: 1/1/1
