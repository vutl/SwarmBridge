# G-L2 summary (g2_headroom.py --boot 300; g2_predict.py --boot 2000)

## Bước 2: dư địa oracle (train ∪ val; điểm recall)

Mức bit: mbt2018-mean_q3 = 0.303 Mbit/khung, avif_50 = 0.627 Mbit/khung, avif_80 = 1.266 Mbit/khung 

| tập | mức bit | O | BL (loại) | F | MX | RV | CB | **H** | O_x | BL_x | **H_x** | H (menu codec) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train∪val | mbt2018-mean_q3 | 56.68 | 54.39 (RV) | 53.45 | 53.72 | 54.39 | 54.09 | **+2.29** | 55.25 | 54.30 | **+0.95** | +2.28 |
| train∪val | avif_50 | 59.46 | 57.63 (RV) | 57.38 | 57.44 | 57.63 | 57.44 | **+1.84** | 57.85 | 57.64 | **+0.21** | +1.84 |
| train∪val | avif_80 | 59.83 | 58.71 (RV) | 57.92 | 58.50 | 58.71 | 58.66 | **+1.12** | 58.14 | 58.53 | **-0.40** | +1.12 |
| train | mbt2018-mean_q3 | 56.52 | 54.25 (RV) | 54.07 | 54.07 | 54.25 | 53.75 | **+2.28** | 55.10 | 54.19 | **+0.90** | +2.26 |
| train | avif_50 | 59.36 | 57.57 (RV) | 55.99 | 57.35 | 57.57 | 57.32 | **+1.79** | 57.74 | 57.56 | **+0.18** | +1.78 |
| train | avif_80 | 59.76 | 58.60 (RV) | 57.87 | 58.33 | 58.60 | 58.57 | **+1.15** | 58.05 | 58.39 | **-0.34** | +1.15 |
| val | mbt2018-mean_q3 | 57.02 | 55.45 (RV) | 51.70 | 54.11 | 55.45 | 55.23 | **+1.57** | 55.55 | 55.23 | **+0.33** | +1.57 |
| val | avif_50 | 59.77 | 58.15 (RV) | 57.01 | 57.65 | 58.15 | 58.06 | **+1.62** | 58.20 | 58.22 | **-0.02** | +1.62 |
| val | avif_80 | 60.02 | 58.98 (RV) | 58.94 | 58.94 | 58.98 | 58.98 | **+1.04** | 58.40 | 58.89 | **-0.48** | +1.02 |

Khám phá (thêm sau tiền đăng ký): oracle chọn một mode cố định cho mỗi video (biết đáp án của video đó)

| tập | mức bit | O (theo khung) | O_video (theo video) | BL | phần dư địa nằm giữa các video |
|---|---|---|---|---|---|
| train∪val | mbt2018-mean_q3 | 56.68 | 55.25 | 54.39 | 37% |
| train∪val | avif_50 | 59.46 | 58.10 | 57.63 | 26% |
| train∪val | avif_80 | 59.83 | 58.86 | 58.71 | 14% |
| train | mbt2018-mean_q3 | 56.52 | 55.00 | 54.25 | 33% |
| train | avif_50 | 59.36 | 57.90 | 57.57 | 18% |
| train | avif_80 | 59.76 | 58.73 | 58.60 | 11% |
| val | mbt2018-mean_q3 | 57.02 | 55.82 | 55.45 | 24% |
| val | avif_50 | 59.77 | 58.55 | 58.15 | 25% |
| val | avif_80 | 60.02 | 59.20 | 58.98 | 21% |

FP/khung và recall vật nhỏ ở cùng mức bit (train ∪ val):

| mức bit | FP/khung O | FP/khung BL | recall nhỏ O | recall nhỏ BL |
|---|---|---|---|---|
| mbt2018-mean_q3 | 16.9 | 16.6 | 45.74 | 43.06 |
| avif_50 | 17.0 | 17.6 | 48.60 | 46.73 |
| avif_80 | 17.1 | 17.2 | 49.00 | 47.87 |

Phân rã (train ∪ val): đóng góp vào O − BL theo nhóm (điểm recall của cả pool; 'tương đối' = trong nhóm)

- mbt2018-mean_q3: video đóng góp lớn nhất 0000289 (+0.56 điểm, 25% tổng); số video có đóng góp > 0: 29/46
  - n_obj: [1, 47] +0.04 điểm (tương đối +0.34, ΔMbit/khung -0.139); [48, 83] +0.95 điểm (tương đối +3.09, ΔMbit/khung +0.047); [84, 3.2e+02] +1.31 điểm (tương đối +2.26, ΔMbit/khung +0.089)
  - small_share: [0, 0.57] +0.32 điểm (tương đối +1.50, ΔMbit/khung -0.066); [0.57, 0.76] +0.93 điểm (tương đối +2.45, ΔMbit/khung +0.043); [0.76, 1] +1.04 điểm (tương đối +2.58, ΔMbit/khung +0.023)
- avif_50: video đóng góp lớn nhất 0000289 (+0.28 điểm, 15% tổng); số video có đóng góp > 0: 38/46
  - n_obj: [1, 47] +0.29 điểm (tương đối +2.56, ΔMbit/khung -0.240); [48, 83] +0.71 điểm (tương đối +2.30, ΔMbit/khung +0.094); [84, 3.2e+02] +0.84 điểm (tương đối +1.45, ΔMbit/khung +0.142)
  - small_share: [0, 0.57] +0.53 điểm (tương đối +2.45, ΔMbit/khung -0.052); [0.57, 0.76] +0.77 điểm (tương đối +2.03, ΔMbit/khung +0.070); [0.76, 1] +0.54 điểm (tương đối +1.33, ΔMbit/khung -0.018)
- avif_80: video đóng góp lớn nhất 0000276 (+0.09 điểm, 8% tổng); số video có đóng góp > 0: 39/46
  - n_obj: [1, 47] +0.30 điểm (tương đối +2.67, ΔMbit/khung -0.582); [48, 83] +0.46 điểm (tương đối +1.49, ΔMbit/khung -0.416); [84, 3.2e+02] +0.36 điểm (tương đối +0.62, ΔMbit/khung -0.332)
  - small_share: [0, 0.57] +0.34 điểm (tương đối +1.55, ΔMbit/khung -0.503); [0.57, 0.76] +0.38 điểm (tương đối +1.01, ΔMbit/khung -0.415); [0.76, 1] +0.40 điểm (tương đối +1.00, ΔMbit/khung -0.409)

Độ lặp lại giữa hai phép đo (train ∪ val, menu codec):

- tương quan TP (mức): 0.995; TP trung bình đo gốc − đo lật: +0.03
- hiệu TP mbt_q3−avif_20: tương quan giữa hai phép đo 0.854; trung bình +4.69 / +4.52; độ lệch chuẩn 5.96 / 6.04
- hiệu TP avif_50−mbt_q3: tương quan giữa hai phép đo 0.693; trung bình +1.83 / +2.13; độ lệch chuẩn 3.96 / 4.07
- hiệu TP avif_80−avif_50: tương quan giữa hai phép đo 0.408; trung bình +1.41 / +1.15; độ lệch chuẩn 2.61 / 2.34

CI 95% theo video (cluster bootstrap B = 300, khám phá):

- mbt2018-mean_q3: H [1.4795236383211383, 2.960601297261548]; H_x [0.36477200834524437, 1.4216842502059932]
- avif_50: H [1.378477260924992, 2.2794472550974225]; H_x [-0.0872092589945106, 0.5056851355175798]
- avif_80: H [0.8678575012041077, 1.4179061874982535]; H_x [-0.5725224633700401, -0.23116701469993245]

**Quy tắc 0.4:** mức bit đạt H ≥ 1,0 và H_x ≥ 0,5: ['mbt2018-mean_q3'] → ĐI TIẾP sang bước 3

fitted 16 candidates 10s

## Bước 3: chọn ứng viên trên val (Δ_val = recall − BL ở cùng bit, điểm)

| ứng viên | mbt2018-mean_q3 | avif_50 | avif_80 |
|---|---|---|---|
| A / abs / ridge | -1.02 | -0.35 | -0.22 |
| A / abs / gbm | -4.88 | -2.71 | -2.69 |
| A / pair / ridge | -1.47 | -0.48 | -0.15 |
| A / pair / gbm | -2.73 | -1.54 | -1.75 |
| A+B / abs / ridge | -5.30 | -5.48 | -5.53 |
| A+B / abs / gbm | -2.12 | -1.79 | -1.94 |
| A+B / pair / ridge | -5.30 | -5.48 | -5.53 |
| A+B / pair / gbm | -1.39 | -0.99 | -0.85 |
| A+C / abs / ridge | -0.17 | +0.04 | -0.10 |
| A+C / abs / gbm | -0.05 | -0.16 | -0.29 |
| A+C / pair / ridge | -0.17 | +0.04 | -0.10 |
| A+C / pair / gbm | -0.14 | -0.45 | -0.74 |
| A+B+C / abs / ridge | -0.19 | +0.04 | -0.18 |
| A+B+C / abs / gbm | -0.31 | -0.22 | -0.25 |
| A+B+C / pair / ridge | -0.19 | +0.04 | -0.18 |
| A+B+C / pair / gbm | -0.08 | -0.50 | -0.23 |

- mbt2018-mean_q3: chọn **A+C / abs / gbm** (Δ_val -0.05); trộn λ: [(1.8946951434379325e-05, 0.045), (1.7768155494889693e-05, 0.955)]
- avif_50: chọn **A+C / abs / ridge** (Δ_val +0.04); trộn λ: [(3.273568697377693e-06, 0.402), (1.8363205647176041e-06, 0.598)]
- avif_80: chọn **A+C / abs / ridge** (Δ_val -0.10); trộn λ: [(7.966866754097959e-07, 1.0)]

## Bước 3–4: đánh giá cuối (test / test-dev, một lần)

| mức bit | pool | ứng viên | Mbit/khung | recall | BL (loại) | **Δ** [CI video] | LOVO min | recall nhỏ | FP/khung | đa dạng | oracle | phần dư địa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mbt2018-mean_q3 | test | A+C|abs|gbm | 0.267 | 57.06 | 57.79 (RV) | **-0.73** [-1.66, +0.04] | -0.86 | 45.16 | 15.5 | 0.58 | 59.67 | -39% |
| mbt2018-mean_q3 | testdev | A+C|abs|gbm | 0.214 | 43.38 | 43.55 (CB) | **-0.17** [-1.09, +0.94] | -0.47 | 30.61 | 10.8 | 0.69 | 47.63 | -4% |
| avif_50 | test | A+C|abs|ridge | 0.563 | 60.70 | 61.10 (RV) | **-0.40** [-0.79, -0.17] | -0.48 | 49.47 | 16.6 | 0.50 | 62.46 | -30% |
| avif_50 | testdev | A+C|abs|ridge | 0.544 | 48.58 | 49.92 (RV) | **-1.33** [-2.09, -0.70] | -1.45 | 36.19 | 12.4 | 0.68 | 52.62 | -49% |
| avif_80 | test | A+C|abs|ridge | 0.986 | 60.56 | 61.17 (RV) | **-0.61** [-1.17, -0.31] | -0.77 | 49.24 | 16.3 | 0.31 | 62.76 | -38% |
| avif_80 | testdev | A+C|abs|ridge | 0.757 | 49.23 | 50.91 (RV) | **-1.67** [-2.50, -1.02] | -1.89 | 37.01 | 12.5 | 0.73 | 53.34 | -69% |

AP50 COCO (thành viên đơn định: predictor trội / RV gần bit nhất / mode cố định tốt nhất ≤ B; Mbit/khung):

- mbt2018-mean_q3 test: predictor 40.07 (0.267), RV 41.79 (0.268), cố định 40.54 (0.247)
- mbt2018-mean_q3 testdev: predictor 28.45 (0.214), RV 28.63 (0.213), cố định 28.22 (0.191)
- avif_50 test: predictor 43.78 (0.617), RV 43.92 (0.561), cố định 43.50 (0.520)
- avif_50 testdev: predictor 33.71 (0.582), RV 33.58 (0.541), cố định 31.15 (0.321)
- avif_80 test: predictor 43.95 (0.986), RV 44.61 (0.985), cố định 43.50 (0.520)
- avif_80 testdev: predictor 34.13 (0.757), RV 35.04 (0.752), cố định 34.44 (0.670)

- mbt2018-mean_q3: điều kiện 1 False, 2 False, 3 True (chi phí 40 ms CPU), 4 True → **FAIL**
- avif_50: điều kiện 1 False, 2 False, 3 True (chi phí 40 ms CPU), 4 True → **FAIL**
- avif_80: điều kiện 1 False, 2 False, 3 True (chi phí 40 ms CPU), 4 True → **FAIL**

**Quyết định G-L2: FAIL**

Chi phí đo: {"B_cpu_ms_median": 9.62203299968678, "C_gpu_ms_median": 12.324557999818353, "gpu": "NVIDIA GeForce RTX 3070", "cpu": 6, "C_cpu_threads1_ms_median": 35.140797499934706, "C_cpu_threads6_ms_median": 39.758591500003604, "C_params": 9431662}
PREDICT_DONE 26s
