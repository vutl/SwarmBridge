# SwarmBridge hướng A — A5: mở rộng dữ liệu, task fidelity, khảo sát chế độ vận hành, RL có mục tiêu

Bắt đầu 2026-11-07, tiếp nối A4 (commit `0e77387a`, báo cáo `reports/archive/uav-semcom-a/uav-semcom-a.md` mục 8).

**Câu hỏi chính của A5:** trong điều kiện nào RL thực sự mang lại lợi ích so với heuristic thích nghi mạnh nhất (reuse+budget, rút ra từ A4), và khi nào heuristic đã đủ tốt?

Không đổi reward, bandwidth hay định nghĩa nhiễu/bit/deadline để PPO thắng. A4 không bị ghi đè: mọi kết quả A5 nằm ở `runs/uav_semcom/a5/`.

## 0. Quy tắc định trước (ghi và commit TRƯỚC khi có kết quả quét hay PPO)

### 0.1 Dữ liệu
- Ảnh: toàn bộ 548 ảnh VisDrone2019-DET-val, cùng một tập con cố định 403 ảnh VisDrone2019-DET-test-dev (sắp xếp theo tên, lấy mỗi ảnh thứ 4). Test-dev là pool kiểm tra ngoài, đến từ các video khác.
- Ảnh val được chia **theo chuỗi video** (tiền tố tên file) với seed 0, nhắm tỉ lệ train 0,55 / val 0,15 / test 0,30 tính theo số khung hình:
  - train: 304 khung, 32 chuỗi;
  - val: 84 khung, 14 chuỗi;
  - test: 128 khung, 28 chuỗi.
- Ảnh A4 thuộc chuỗi test (32 khung) bị **loại hẳn**, không dùng ở đâu. Vì vậy mọi ảnh test A5 đều chưa từng được A4 hay các model PPO A4 nhìn thấy.
- Manifest kèm SHA-256 từng file: `runs/uav_semcom/a5/manifest.json`.
- Vai trò của từng pool:
  - train: dùng để fit bảng mean và bộ dự đoán từ probe, chạy các episode tinh chỉnh heuristic (seed 0..29) và train PPO.
  - val: quét chế độ để chọn regime (seed 3000..3049) và chọn checkpoint PPO (seed 3000..3019).
  - test + test-dev: chỉ dùng một lần cho đánh giá cuối, seed 7000..7099.
  - Không seed nào trùng với các bộ seed của A4.

### 0.2 Chọn chế độ (regime) để train RL

Quyết định chỉ dựa trên heuristic trên **val**; không có PPO nào tham gia.
- **R1 = ref**: cấu hình A4 trên dữ liệu A5, bắt buộc.
- **R2 = K5**: 5 UAV / 2 kênh, định trước vì đây là kịch bản tranh chấp tài nguyên.
- **R3**: trong các regime không đổi mục tiêu và không đổi K (tức loại `ref`, `K*`, `wt*`), **và** có điều khiển không tầm thường (return val của heuristic tinh chỉnh tốt nhất − return val của greedy ≥ 2; regime tải thấp mà mọi heuristic gần như bằng nhau thì không có gì để học), chọn regime mà cơ chế A4 giúp **ít nhất**. Cụ thể là regime có (return val của heuristic reuse tinh chỉnh) − max(return val của base tinh chỉnh, budget tinh chỉnh) nhỏ nhất. Đây là nơi hai quy tắc rút ra từ A4 mất ưu thế, nên là nơi kiểm tra rõ nhất xem RL có tìm ra cách khác hay không.
  - Nếu chênh lệch giữa ứng viên nhất và nhì nhỏ hơn 0,5, chọn regime có "giá trị thích nghi" lớn hơn, tức (reuse tinh chỉnh − fixed tinh chỉnh) lớn hơn.

### 0.3 Huấn luyện và chọn checkpoint
- Thuật toán: TimeMaskablePPO (discount theo giây), cùng mạng, siêu tham số, số env và 2M bước như các run A4 đã sửa.
- Mỗi regime train **joint** và **mode-only**, 3 seed (0, 1, 2).
- Ở mode-only, slot do phần slot của heuristic reuse đã tinh chỉnh cho regime đó quyết định.
- Checkpoint được lưu mỗi 2·10⁵ bước. Checkpoint dùng cho test là checkpoint có **return val cao nhất** (nếu bằng nhau thì lấy checkpoint sớm hơn). Checkpoint cuối cũng được báo cáo.
- **Điều chỉnh (2026-11-08 00:05, trước khi có bất kỳ kết quả PPO nào):** quy tắc chọn đưa ra R3 = `phy_mcs_blind`. Ở regime này họ reuse lại là họ heuristic tinh chỉnh **tệ nhất** (train −16,6, trong khi budget +15,0). Nếu giữ nguyên quy tắc, mode-only RL sẽ bị buộc chạy trên một luật slot kém. Vì ý định của quy tắc là "heuristic mạnh nhất", tôi đổi thành **T:best** = thành viên tinh chỉnh tốt nhất trên train trong ba họ base / budget / reuse. T:best được dùng ở ba chỗ:
  - phần slot của mode-only RL;
  - baseline tham chiếu;
  - heuristic thay thế khi làm action replacement.

  Ở ref và K5, T:best trùng với reuse tinh chỉnh, nên không có gì thay đổi. Ở `phy_mcs_blind`, T:best = pacedB2+budget1. Lúc đổi, chỉ có run ref joint s0 đang chạy (mới vài phút, joint không dùng heuristic); run đó bị dừng và chạy lại từ đầu.
- Thứ tự chạy: seed 0 của mọi (regime, phần), rồi seed 1, rồi seed 2.
- Nếu đường học (learning curve) val vẫn còn tăng rõ ở 2M bước, tôi sẽ ghi lại. Tôi không tự động tăng số bước.

### 0.4 So sánh cuối
- Mọi policy được đánh giá trên cùng seed, nên mọi so sánh đều theo cặp, với khoảng tin cậy t 95%.
- Baseline:
  - fixed mode tinh chỉnh;
  - greedy;
  - base (pacedB…) tinh chỉnh;
  - budget tinh chỉnh;
  - **reuse tinh chỉnh (heuristic mạnh)**;
  - hai heuristic A4 giữ nguyên tham số: pacedB1.5+budget1.5 và greedy+budget1+reuse.
- PPO A4 (td joint s0–2, td mode-only s0–2) được thử **zero-shot** trên các regime K = 3, tức khi kích thước quan sát và hành động khớp. Kết quả này được ghi rõ là zero-shot.

## 1. Dữ liệu mới, chia tập và codec profiling

Code: `a5_data.py`, `profile_a5.sh` (dùng lại `profile_a11/a12` với danh sách ảnh cố định), `a5_profile_analysis.py`. Kết quả ở `runs/uav_semcom/a5/{manifest.json, profile/, profile_analysis.json}`.

**Quy mô.** Profile 14 mode của A4 trên 548 ảnh val và 403 ảnh test-dev, cùng codec, cùng thiết lập, độ phân giải gốc.
- Thời gian encode/decode luôn lấy trung bình trên pool val. Đo trên GPU khi CPU rảnh. Riêng learned codec đo thêm trên CPU với 30 ảnh (A4 chỉ dùng 10 ảnh).
- Test-dev được profile trong lúc CPU đang train PPO, nên thời gian đo ở đó **không** được dùng.

**Tái lập so với A4.** Với 100 ảnh A4, cả 1000 cặp (ảnh, mode) không tạo sinh đều cho payload **trùng từng byte** và PSNR trùng hoàn toàn. Thời gian encode GPU khớp trong khoảng ±0,002 s. Learned codec chạy trên CPU chậm hơn A4 khoảng 5%, vì A5 đo trên 30 ảnh thay vì 10.

**Thống kê theo tập** (kB trung bình, hệ số biến thiên CV, PSNR ± độ lệch chuẩn):

| tập | n | avif_50 | avif_80 | mbt q5 | mbt q8 |
|---|---|---|---|---|---|
| train | 304 | 78 kB (CV 0,40), 35,4 ± 2,0 dB | 158 (0,39), 44,3 ± 0,9 | 80 (0,47), 35,5 ± 1,3 | 186 (0,43), 42,0 ± 1,0 |
| val | 84 | 58 (0,31), 36,6 ± 2,1 | 120 (0,30), 44,5 ± 1,3 | 53 (0,38), 36,2 ± 1,5 | 130 (0,34), 42,4 ± 1,1 |
| test | 128 | 63 (0,44), 35,9 ± 2,4 | 129 (0,42), 44,5 ± 1,1 | 65 (0,53), 36,0 ± 1,9 | 149 (0,49), 42,4 ± 1,4 |
| test-dev | 403 | 88 (0,49), 36,3 ± 2,2 | 184 (0,47), 44,6 ± 0,9 | 84 (0,58), 36,1 ± 1,8 | 206 (0,54), 42,4 ± 1,3 |

- **Payload thay đổi mạnh theo ảnh**: CV từ 0,3 đến 0,6.
- **Giữa các tập cũng khác nhau:** ảnh val và test nhẹ hơn train khoảng 20%, còn test-dev nặng hơn khoảng 15%.
- **Hệ quả:** return giữa các pool **không** so trực tiếp được với nhau. Mọi so sánh đều được làm theo cặp, trên cùng một pool.

**Nội dung ảnh có làm đổi mode tốt nhất không?** Câu trả lời là có. Với mỗi ngưỡng chất lượng, xét mode rẻ nhất đạt ngưỡng đó (chỉ tính mode không tạo sinh) trên 951 ảnh:

| ngưỡng | mode thắng nhiều nhất (tỉ lệ ảnh) | các mode thắng khác |
|---|---|---|
| PSNR ≥ 30 dB | mbt q3 (51%) | mbt q1, avif_50, avif_20, avif_5 |
| PSNR ≥ 35 dB | mbt q5 (47%) | avif_80 (33%), avif_50, mbt q3 |
| PSNR ≥ 40 dB | avif_80 (85%) | mbt q8 |

Ở ngưỡng chất lượng trung bình, không có mode nào thắng quá khoảng 50% số ảnh. Đây là cơ sở cho việc chọn mode theo từng ảnh.
- Tương quan: payload và số pixel r ≈ 0,63–0,79; bpp và số vật thể r ≈ 0,18–0,20; PSNR và số vật thể r ≈ −0,22.

**Bộ dự đoán từ probe avif_5** (fit trên train, quy tắc như A4):

| tập | RMSE log-payload | R² log-payload | MAE PSNR |
|---|---|---|---|
| train | 0,19 | 0,78 | 1,31 dB |
| val | 0,22 | 0,44 | 1,99 dB |
| test | 0,23 | 0,72 | 1,82 dB |
| test-dev | 0,24 | 0,70 | 1,71 dB |

Trên các chuỗi video chưa thấy, sai số tăng khoảng 20–50%. R² thấp ở val chủ yếu vì phương sai payload trong val nhỏ (CV 0,3).

## 2. Task fidelity (thước đo phụ; reward không đổi)

Code: `a5_det.py`, `a5_fidelity.py`, `a5_figures.py`, `a5_examples.py`. Kết quả ở `runs/uav_semcom/a5/{detector/, fidelity/}` và `reports/figures/a5/`.

**Detector.**
- YOLO11s khởi tạo từ COCO, fine-tune trên **VisDrone2019-DET-train**: imgsz 960, 40 epoch, batch 8, seed 0, bộ tham số cố định từ trước.
- Phát hiện khi chuẩn bị: 24/76 chuỗi của val cũng có khung hình trong VisDrone-train (cùng video; ví dụ chuỗi 0000072 có khung 1–2 ở train và khung 3 ở val). Để detector không bao giờ thấy cảnh của ảnh đánh giá, 297 khung train thuộc các chuỗi đó **bị loại**. Test-dev không chung chuỗi nào với train.
- Còn lại 5903 ảnh train và 271 ảnh holdout (10% số chuỗi train, dùng chọn `best.pt`).
- Holdout đạt mAP50 0,475, mAP50-95 0,286. SHA-256 của trọng số ghi trong `detector/best_pt.sha256`.

**Đánh giá.**
- Ảnh gốc và ảnh giải mã từ từng mode được chạy detector với cùng thiết lập: conf ≥ 0,01, NMS 0,7, tối đa 500 box.
- Chấm bằng nhãn VisDrone của ảnh gốc, qua pycocotools với maxDets 500. Vùng "ignored" và lớp "others" được coi là vùng bỏ qua.
- Ở mức từng ảnh, điểm hoạt động là conf ≥ 0,25 và IoU ≥ 0,5: tính recall, recall vật nhỏ (< 32² px), và F1 so với chính detection trên ảnh gốc.
- Payload của mọi cặp (ảnh, mode) khớp với profile.

**Kết quả trên tập test A5** (128 ảnh, chưa từng dùng ở đâu; tập 548 ảnh và tập in-A4 cho cùng xu hướng, xem `fidelity/coco_val.json`):

| mode | kB | PSNR | AP | AP50 | AP_small | recall@0,25 | recall vật nhỏ | F1 so với gốc |
|---|---|---|---|---|---|---|---|---|
| ảnh gốc | — | — | 0,280 | 0,452 | 0,195 | 0,607 | 0,492 | 1 |
| avif_80 | 129 | 44,5 | 0,280 | 0,450 | 0,190 | 0,604 | 0,488 | 0,96 |
| mbt q8 | 149 | 42,4 | 0,279 | 0,448 | 0,196 | 0,608 | 0,495 | 0,95 |
| mbt q5 | 65 | 36,0 | 0,269 | 0,435 | 0,195 | 0,609 | 0,496 | 0,89 |
| avif_50 | 63 | 35,9 | 0,268 | 0,434 | 0,181 | 0,589 | 0,469 | 0,90 |
| jpeg2000_25 | 109 | 33,0 | 0,254 | 0,415 | 0,169 | 0,575 | 0,455 | 0,85 |
| mbt q3 | 31 | 32,0 | 0,245 | 0,405 | 0,170 | 0,568 | 0,445 | 0,83 |
| avif_20 | 17 | 29,0 | 0,206 | 0,345 | 0,123 | 0,504 | 0,371 | 0,75 |
| mbt q1 | 13 | 28,6 | 0,175 | 0,298 | 0,113 | 0,469 | 0,333 | 0,70 |
| avif_5 | 8 | 26,6 | 0,133 | 0,230 | 0,058 | 0,387 | 0,241 | 0,60 |
| thumb128_up | 3,4 | 24,3 | 0,055 | 0,106 | 0,016 | 0,184 | 0,050 | 0,36 |

**Mode tạo sinh** (100 ảnh đầu của pool val, tức `gen_subset`):

| mode | AP50 |
|---|---|
| cap_t2i | 0,000 |
| cap_thumb64_i2i_t0.6 | 0,006 |
| cap_thumb128_i2i_t0.3 | 0,098 |
| thumb64_up (cùng tập con) | 0,022 |
| thumb128_up (cùng tập con) | 0,138 |

**Test-dev** (403 ảnh, video khác, AP tuyệt đối thấp hơn):

| mode | AP50 | mất so với ảnh gốc |
|---|---|---|
| ảnh gốc | 0,368 | — |
| avif_80 | 0,365 | — |
| avif_50 | 0,354 | −4% |
| mbt q5 | 0,344 | −7% |
| avif_20 | 0,282 | −23% |
| avif_5 | 0,199 | −46% |

**Đọc kết quả:**
1. **Từ khoảng 42–44 dB, ảnh giải mã tương đương ảnh gốc về tác vụ**: AP50 và recall vật nhỏ lệch không quá 0,005.
2. **Khoảng 35–36 dB (avif_50, mbt q5) giữ được khoảng 96% AP50** trên val/test và 93–96% trên test-dev. Recall vật nhỏ gần như nguyên vẹn với mbt q5, giảm khoảng 5% với avif_50. Payload chỉ bằng khoảng một nửa so với 44 dB.
   - Vì vậy, việc PPO A4 hạ PSNR trung bình từ 44 xuống khoảng 38 dB, bằng cách trộn avif_80 với avif_50/mbt q5, chỉ tốn khoảng 1–3% AP50 trên những ảnh được giao. Đổi lại là không rớt ảnh.
3. **Dưới 30 dB thì tác vụ mất nhanh hơn PSNR gợi ý**: avif_20 mất khoảng 24% AP50 và khoảng 25% recall vật nhỏ.
   - Ngưỡng Qmin 22–30 dB của mô hình (theo Huang và cộng sự) **cho phép** những mode mà detector xem là hỏng đáng kể. Với ứng dụng có tác vụ phát hiện, Qmin thực tế nên khoảng ≥ 32–35 dB. Đây là một giới hạn của utility hiện tại (mục 6), không phải lỗi mô phỏng.
4. **Mode tạo sinh không giữ được thông tin cho tác vụ**: chúng kém hơn cả việc phóng to thumbnail cùng payload bằng bicubic (0,098 so với 0,138). Nội dung do mô hình "vẽ thêm" làm hại detector (xem hình ví dụ). Điều này củng cố quyết định không dựa vào mode tạo sinh trong hướng A.
5. PSNR là thước đo thay thế khá tốt bên trong một họ codec, nhưng **không đồng đều giữa các họ**: JPEG 2000 có AP50 ở 33 dB thấp hơn mbt q3 ở 32 dB; mbt q5 giữ vật nhỏ tốt hơn avif_50 ở cùng PSNR.

Hình:
- `reports/figures/a5/rdt_val.png` và `rdt_testdev.png`: bpp so với PSNR, AP50, recall vật nhỏ;
- `reports/figures/a5/examples.png`: ảnh test 0000287_03401 được chọn tự động theo quy tắc ghi trong code, nơi chênh lệch recall vật nhỏ giữa avif_50 và avif_20 lớn nhất. Recall: ảnh gốc 0,84 = avif_80 = avif_50; mbt q5 0,77; avif_20 0,68; thumbnail 0,30; i2i 0,16.

## 3. Khảo sát chế độ vận hành (heuristic; chọn regime trên val)

Code: `a5_common.py` (21 regime), `a5_eval.py tune|eval`, `a5_select.py`. Kết quả ở `runs/uav_semcom/a5/{tune/, sweep_val.json, selected.json}`.

**Cách làm.** Với mỗi regime, ba họ heuristic được tinh chỉnh trên pool train, seed 0..29:
- base: 45 thành viên, gồm fixed mode, greedy, paced, stag, pacedB;
- budget: 28 thành viên;
- reuse: 59 thành viên.

Sau đó đánh giá trên pool val, seed 3000..3049. Mỗi episode heuristic chỉ mất khoảng 0,03 s, nên cả bước tinh chỉnh 21 regime hết khoảng 15 phút.

**Return trung bình trên val** (in đậm = tốt nhất trong các thành viên tinh chỉnh):

| regime | fixed | greedy | base | budget | reuse | A4 pacedB1.5+budget1.5 | A4 greedy+budget1+reuse | reuse + oracle |
|---|---|---|---|---|---|---|---|---|
| ref | +53,8 | +58,2 | +60,7 | +62,0 | **+62,9** | +62,0 | +63,7 | +66,1 |
| K2 | +49,3 | +49,3 | +49,3 | **+49,3** | **+49,3** | +49,3 | +49,2 | +50,2 |
| K4 | +64,3 | +18,2 | +64,3 | +58,2 | **+74,0** | +57,7 | +74,0 | +75,6 |
| K5 | +40,9 | −51,6 | +33,9 | +44,5 | **+77,1** | +42,2 | +79,8 | +78,1 |
| v10 | +7,6 | +7,6 | **+7,8** | +7,7 | +7,4 | +7,8 | +4,6 | +9,0 |
| v15 | +46,3 | +46,4 | +46,9 | +46,3 | **+47,5** | +46,6 | +46,2 | +49,2 |
| env_suburban | +52,9 | +52,8 | +52,9 | +59,0 | **+62,1** | +59,2 | +62,1 | +64,7 |
| env_dense | +48,5 | +6,5 | +48,5 | +46,0 | **+54,8** | +44,1 | +54,8 | +56,4 |
| env_highrise | −34,5 | −208,1 | −34,5 | −38,8 | **−16,6** | −91,4 | −19,4 | −17,6 |
| fade_k3 | +53,8 | +58,0 | +60,5 | +62,0 | **+62,8** | +62,0 | +63,7 | +66,0 |
| fade_k0 | +53,8 | +58,1 | +60,5 | +61,9 | **+63,6** | +61,9 | +63,6 | +65,9 |
| dl_tight | +41,6 | +27,9 | +41,6 | +42,6 | **+45,8** | +38,0 | +42,0 | +48,1 |
| dl_loose | +59,0 | +70,1 | +71,4 | +71,5 | **+72,2** | +69,9 | +71,0 | +74,1 |
| q_low | +69,4 | +68,1 | +69,4 | +70,0 | **+72,5** | +69,8 | +72,5 | +74,5 |
| q_high | +44,6 | +44,6 | **+47,0** | +46,9 | +38,2 | +44,0 | +37,9 | +38,6 |
| wt0 | +119,5 | +125,3 | **+140,4** | +133,3 | **+140,4** | +131,3 | +129,8 | +141,3 |
| wt0.1 | +97,6 | +102,9 | **+110,5** | +109,4 | **+110,5** | +108,2 | +107,8 | +111,7 |
| wt1 | −99,5 | −98,5 | −99,5 | −94,0 | **−90,5** | −99,7 | −90,5 | −88,0 |
| phy_gap3 | **+48,6** | +8,7 | **+48,6** | +47,6 | +47,7 | +45,3 | +49,2 | +50,1 |
| phy_mcs | +28,4 | −56,2 | +28,4 | +35,4 | **+36,0** | +30,9 | +37,1 | +36,1 |
| phy_mcs_blind | +28,4 | −56,2 | +28,4 | **+35,4** | +16,3 | +30,9 | −15,0 | +23,7 |

Thành viên được chọn trên train cho từng regime nằm trong `tune/<regime>.json` (bảng đầy đủ: `python a5_summarize.py sweep`).

**Nguyên nhân của các thay đổi** (chỉ heuristic):
1. **Tranh chấp kênh (K) là yếu tố lớn nhất.**
   - K2 (2 UAV, 2 kênh): mọi họ heuristic ngang nhau và reuse không thêm gì, vì không có ai phải chờ kênh.
   - K4, K5: các họ không dùng chung kênh tụt mạnh. Greedy ở K5 còn −51,6 vì 3 UAV phải chờ và ảnh quá hạn.
   - Dùng chung kênh khi rảnh (reuse) hơn họ tốt thứ hai +9,7 ở K4 và +32,6 ở K5.
   - Cơ chế reuse mà A4 phát hiện có giá trị **tăng theo mức tranh chấp**.
2. **Tải thấp (v10) thì mọi thứ bão hoà**: chênh lệch dưới 0,5, không có gì để điều khiển.
3. **Môi trường truyền sóng.**
   - Đô thị dày và nhà cao tầng làm giảm xác suất LoS và tăng suy hao NLoS. Thông lượng giảm mạnh nên heuristic điều nhịp/thích nghi trở nên quan trọng hơn: env_highrise có chênh lệch giữa các họ lên tới khoảng 190 return.
   - Fading nhỏ (Rician K 3 hoặc 0 dB) gần như không đổi gì, vì kênh trung bình mới là yếu tố quyết định.
4. **Deadline chặt** làm reuse lợi hơn (+3,2), vì có thêm UAV được phát sớm. **Qmin cao** (28–36 dB) thì ngược lại: reuse **thiệt** −8,8. Lý do là mode chất lượng cao có payload lớn, hai UAV chung kênh làm chậm nhau, và ảnh vượt Qmin trễ hạn.
5. **PHY không lý tưởng.**
   - SNR gap 3 dB: reuse hết lợi (−0,9).
   - MCS rời rạc với link adaptation biết nhiễu (phy_mcs): reuse còn lợi chút ít (+0,6).
   - MCS chọn theo SNR không biết nhiễu (phy_mcs_blind): dùng chung kênh làm khối truyền thất bại, reuse **thiệt −19,1**.
   - Kết luận: **ưu thế của reuse phụ thuộc vào giả định Shannon/link adaptation lý tưởng**. Phải nói rõ điều này khi viết bài.
6. **Trọng số thời gian** chỉ đổi mục tiêu. w_T = 0 và 0,1 thì heuristic staggered (bay lần lượt) tốt nhất; w_T = 1 thì nghiêng về nhanh, chất lượng thấp. Pareto chất lượng–thời gian–năng lượng nằm ở mục 3.2 (tập test).
7. **Giá trị của thông tin.** Biết chính xác payload/PSNR từng ảnh (oracle) chỉ thêm từ −1 đến +3,2 cho heuristic reuse (đa số khoảng +1,5 đến +3). Riêng phy_mcs_blind thêm +7,4, vì ở đó reuse là heuristic yếu.

**Chọn regime** (quy tắc 0.2, chạy tự động bằng `a5_select.py`, không có PPO):
- R1 = ref;
- R2 = K5;
- R3 = **phy_mcs_blind**: reuse − max(base, budget) = −19,1, nhỏ nhất trong các regime đủ điều kiện; điều khiển có ý nghĩa với best − greedy = +91,6.

Ở R3, heuristic mạnh T:best = pacedB2+budget1 (xem điều chỉnh ở 0.3).

### 3.2 Heuristic trên tập test (128 ảnh chưa dùng, seed 7000..7099) và test-dev

Δ là chênh lệch theo cặp so với **T:best** (heuristic tinh chỉnh tốt nhất trên train), kèm CI 95%. Bảng đầy đủ: `python a5_summarize.py final`; dữ liệu ở `final/test_heur.json`.

| regime | fixed | greedy | base | budget | reuse | T:best | reuse − T:best | base − T:best | budget − T:best |
|---|---|---|---|---|---|---|---|---|---|
| ref | +49,0 | +46,9 | +55,5 | +56,4 | +58,5 | +58,5 | (= reuse) | −3,1 [−4,0; −2,2] | −2,1 [−2,9; −1,3] |
| K2 | +47,5 | +47,5 | +47,6 | +47,9 | +47,9 | +47,9 | 0 | −0,3 | 0 |
| K4 | +55,9 | −3,7 | +55,9 | +47,3 | +63,8 | +63,8 | (= reuse) | −8,0 | −16,6 |
| K5 | +26,8 | −77,9 | +30,0 | +27,6 | +61,2 | +61,2 | (= reuse) | −31,2 | −33,6 |
| v10 | +5,8 | +5,8 | +6,5 | +6,2 | +6,7 | +6,7 | (= reuse) | −0,2 [−0,4; +0,1] | −0,5 |
| v15 | +42,2 | +42,3 | +43,1 | +42,3 | +44,5 | +44,5 | (= reuse) | −1,3 | −2,2 |
| env_suburban | +48,0 | +41,8 | +48,0 | +53,3 | +57,6 | +57,6 | (= reuse) | −9,6 | −4,2 |
| env_dense | +41,6 | −10,2 | +41,6 | +38,9 | +48,1 | +48,1 | (= reuse) | −6,5 | −9,2 |
| env_highrise | −57,9 | −209,4 | −57,9 | −59,3 | −35,9 | −35,9 | (= reuse) | −22,0 | −23,4 |
| fade_k3 | +49,0 | +46,6 | +55,5 | +56,4 | +58,5 | +58,5 | (= reuse) | −3,0 | −2,1 |
| fade_k0 | +49,0 | +46,6 | +55,6 | +56,4 | +58,8 | +58,8 | (= reuse) | −3,2 | −2,4 |
| dl_tight | +35,4 | +14,9 | +35,4 | +35,5 | +38,9 | +38,9 | (= reuse) | −3,5 | −3,4 |
| dl_loose | +54,7 | +63,2 | +67,1 | +67,4 | +68,6 | +68,6 | (= reuse) | −1,5 | −1,2 |
| q_low | +65,6 | +63,9 | +65,6 | +66,3 | +69,3 | +69,3 | (= reuse) | −3,7 | −3,1 |
| q_high | +33,3 | +33,3 | +41,6 | +41,7 | +35,2 | +41,7 | **−6,5 [−7,6; −5,3]** | −0,1 | (= budget) |
| wt0 | +115,0 | +115,0 | +139,3 | +131,3 | +139,3 | +139,3 | 0 | 0 | −8,0 |
| wt0.1 | +93,0 | +92,3 | +108,4 | +105,7 | +108,4 | +108,4 | 0 | 0 | −2,7 |
| wt1 | −105,0 | −112,0 | −105,0 | −100,1 | −95,5 | −95,5 | (= reuse) | −9,5 | −4,6 |
| phy_gap3 | +41,8 | −7,6 | +41,8 | +39,1 | +41,3 | +41,8 | −0,5 [−1,4; +0,5] | (= base) | −2,7 |
| phy_mcs | +20,3 | −72,3 | +20,3 | +26,7 | +27,4 | +27,4 | (= reuse) | −7,1 | −0,6 [−1,6; +0,4] |
| phy_mcs_blind | +20,3 | −72,3 | +20,3 | +26,7 | +3,9 | +26,7 | **−22,9 [−25,4; −20,4]** | −6,5 | (= budget) |

- Mọi kết luận của mục 3 (rút ra trên val) đều lặp lại trên test.
- Hai quy tắc A4 giữ nguyên tham số (greedy+budget1+reuse) gần như luôn ngang với reuse tinh chỉnh lại. Ngoại lệ là những regime mà reuse có hại: q_high còn +27,4, phy_mcs_blind còn −26,4.

**Test-dev** (403 ảnh, video khác, payload nặng hơn khoảng 15%):

| regime | fixed | greedy | base | budget | reuse | T:best | fixed − T:best |
|---|---|---|---|---|---|---|---|
| ref | +45,0 | −13,6 | +27,2 | +41,1 | +45,4 | +45,4 | −0,4 |
| K5 | +30,8 | −172,0 | −1,2 | +7,8 | +40,0 | +40,0 | −9,2 |
| phy_mcs_blind | +22,0 | −124,1 | +22,0 | +11,9 | −50,6 | +11,9 | **+10,0 [+8,7; +11,4]** |

- Khi ảnh nặng hơn, heuristic điều nhịp tinh chỉnh trên train **mất phần lớn ưu thế**: greedy sụp đổ, và ở ref mode cố định avif_50 bằng với T:best.
- Ở phy_mcs_blind, T:best (pacedB2+budget1) thua mode cố định mbt q3 tới 10 điểm.
- Nghĩa là các ngưỡng backlog/budget được tinh chỉnh trên một phân phối ảnh **không bền** khi phân phối đổi. Đây là một chỗ mà một policy thích nghi theo trạng thái (RL, hoặc heuristic tự hiệu chỉnh ngưỡng) *có thể* có lợi. Mục 4 kiểm tra PPO trên test-dev.

**Pareto theo trọng số thời gian** (heuristic T:best của từng w_T, tập test; ok = tỉ lệ ảnh giao đúng hạn):

| w_T | T:best | ok | PSNR (dB) | nhiệm vụ (s) | năng lượng (kJ) | trễ TB (s) |
|---|---|---|---|---|---|---|
| 0 | stag_idx_q1 (bay lần lượt) | 0,999 | 44,5 | 115 | 59,9 | 1,39 |
| 0,1 | stag_near_q2 | 0,999 | 44,5 | 101 | 52,9 | 1,45 |
| 0,3 (ref) | pacedB1.5+budget1.25+reuse | 0,991 | 42,6 | 76 | 40,6 | 1,80 |
| 1,0 | greedy+budget1+reuse | 0,992 | 41,0 | 74 | 39,2 | 1,55 |
| (tham chiếu) | fixed avif_50 | 1,000 | 36,0 | 73 | 39,0 | 1,31 |
| (tham chiếu) | greedy | 0,961 | 44,4 | 76 | 40,2 | 2,91 |

- Giữ chất lượng tối đa (44,5 dB) mà không rớt ảnh đòi hỏi bay chậm/lần lượt: thêm khoảng 40 s nhiệm vụ và khoảng 50% năng lượng.
- Trong khi đó, theo mục 2, 42–44 dB và 36 dB khác nhau chỉ khoảng 3–4% AP50.
- Theo góc nhìn tác vụ, các điểm w_T ≥ 0,3 hợp lý hơn.

## 4. PPO vs heuristic mạnh

Code: `a5_train.py`, `a5_eval.py`, `run_a5_cpu.sh`, `run_a5_final.sh`. Kết quả ở `runs/uav_semcom/a5/{train/, final/}`; bảng đầy đủ: `python a5_summarize.py ppo replace zeroshot fidelity`.

**Thiết lập (đúng như mục 0):**
- 3 regime × {joint, mode-only} × 3 seed = 18 run, mỗi run 2M bước.
- Siêu tham số giống A4.
- Checkpoint chọn theo return val cao nhất (seed 3000..3019).
- Mọi policy chạy trên cùng 100 seed test (7000..7099) và 100 seed test-dev, nên Δ là chênh lệch theo cặp với CI 95%.
- "Gộp 3 seed" nghĩa là trung bình return của 3 model trên từng seed môi trường, rồi mới so theo cặp.

### 4.1 Kết quả chính (checkpoint best-val, tập test)

| regime | T:best | PPO joint | Δ joint | PPO mode-only | Δ mode-only | heuristic tốt nhất *chọn sau trên test* | Δ mode-only so với nó |
|---|---|---|---|---|---|---|---|
| ref | +58,55 | +57,68 | −0,87 [−1,40; −0,35] | +58,42 | −0,13 [−0,64; +0,38] | reuse + oracle +60,96 | −2,54 [−3,10; −1,99] |
| K5 | +61,18 | +63,37 | **+2,19 [+1,28; +3,10]** | +64,31 | **+3,13 [+2,27; +3,98]** | A4 greedy+budget1+reuse +63,17 | +1,14 [+0,35; +1,93] |
| phy_mcs_blind | +26,74 | +21,21 | −5,53 [−6,67; −4,39] | +31,07 | **+4,33 [+3,50; +5,17]** | (= T:best) +26,74 | +4,33 [+3,50; +5,17] |

Từng seed (Δ mode-only so với T:best):
- ref: −0,37 / −0,12 / +0,09;
- K5: +4,33 / +0,51 / +4,54;
- phy_mcs_blind: +4,26 / +4,35 / +4,39.

**Checkpoint cuối (2M)** so với T:best:

| regime | joint | mode-only |
|---|---|---|
| ref | −1,08 | −1,15 |
| K5 | +1,10 | **−1,18** (seed 1: −4,95) |
| phy_mcs_blind | −6,13 | +3,25 |

Ở K5, mode-only không ổn định sau đỉnh: chọn checkpoint trên val là cần thiết.

Cột "heuristic tốt nhất chọn sau" lấy policy tốt nhất trong 9 heuristic **sau khi đã nhìn kết quả test**, kể cả biến thể biết trước payload/PSNR (oracle). Đây là phép so sánh có lợi cho heuristic, chỉ dùng để kiểm tra độ bền của kết luận.

**Đọc kết quả:**
- **ref:** RL không có ích. Mode-only ngang T:best; joint thua chút ít. Một heuristic biết trước thông tin ảnh còn hơn PPO 2,5.
- **K5 (tranh chấp kênh):** PPO hơn heuristic tinh chỉnh khoảng +2 đến +3.
  - Nhưng quy tắc A4 giữ nguyên tham số (greedy+budget1+reuse) cũng hơn T:best +2,0.
  - So với quy tắc đó, mode-only còn +1,1 [+0,35; +1,93], còn joint chỉ +0,2 (không có ý nghĩa).
  - Tức là phần lớn khoảng cách ở K5 là do tinh chỉnh heuristic trên train chưa tối ưu, không phải do RL tìm ra cơ chế mới.
- **phy_mcs_blind (PHY không lý tưởng, reuse có hại):**
  - Mode-only hơn heuristic mạnh nhất +4,3, đều ở cả 3 seed.
  - Joint thua −5,5: tự học cả di chuyển, kênh và công suất ở đây kém luật slot đã tinh chỉnh.

### 4.2 Test-dev (video khác, ảnh nặng hơn khoảng 15%)

| regime | T:best | Δ joint | Δ mode-only | heuristic tốt nhất chọn sau trên test-dev | Δ mode-only so với nó |
|---|---|---|---|---|---|
| ref | +45,42 | +1,37 [+0,73; +2,01] | **+2,41 [+1,78; +3,03]** | reuse + oracle +47,53 | +0,30 [−0,37; +0,96] |
| K5 | +40,03 | **+6,26 [+5,10; +7,42]** | **+12,27 [+11,32; +13,21]** | reuse + oracle +42,03 | **+10,27 [+9,24; +11,29]** |
| phy_mcs_blind | +11,94 | −0,77 [−1,82; +0,29] | +7,46 [+6,43; +8,49] | fixed mbt q3 +21,98 | **−2,59 [−3,75; −1,42]** |

- Khi phân phối ảnh đổi, ngưỡng backlog/budget tinh chỉnh trên train trở nên lệch (mục 3.2). PPO, vốn quan sát hàng đợi và payload dự đoán, giữ được hiệu năng tốt hơn.
  - Ở K5, mode-only hơn **mọi** heuristic, kể cả bản oracle, khoảng +10. Đây là khoảng cách lớn nhất của A5.
  - Ở ref, ưu thế còn +2,4 so với T:best nhưng ngang heuristic oracle.
- **Ngoại lệ:** ở phy_mcs_blind, một mode cố định (mbt q3) lại thắng mọi thứ, kể cả PPO (−2,6).
- **Lưu ý cho kết luận "RL bền hơn":** baseline ở đây là heuristic có ngưỡng *cố định*. Một heuristic tự hiệu chỉnh ngưỡng theo payload quan sát được chưa được thử, và là đối thủ công bằng hơn (mục 7).

### 4.3 Kinh nghiệm cần để học và đường học

| regime | phần | best-val step (s0/s1/s2) | episode / run | giây mô phỏng / run |
|---|---|---|---|---|
| ref | joint | 2,0M / 1,8M / 1,0M | 7,0–7,6k | 0,91–1,00M |
| ref | mode | 1,6M / 1,8M / 1,4M | 13,0k | 1,02–1,04M |
| K5 | joint | 2,0M / 1,0M / 1,8M | 3,9–4,4k | 1,03–1,19M |
| K5 | mode | 1,2M / 1,0M / 1,0M | 7,8k | 0,61–0,62M |
| phy_mcs_blind | joint | 1,6M / 1,6M / 2,0M | 4,4–6,0k | 1,16–1,49M |
| phy_mcs_blind | mode | 0,4M / 0,4M / 0,6M | 13,0k | 1,07–1,08M |

- **Joint:** khởi đầu rất âm (−200 đến −1100), mất 0,4–1,0M bước mới vượt 0. Ở ref và phy_mcs_blind, joint bão hoà sau khoảng 1,2–1,6M bước. Ở K5, seed 0 vẫn còn tăng ở 2M; vì vậy có thí nghiệm kéo dài ở 4.6, ghi riêng, không thuộc quy tắc định trước.
- **Mode-only:** khởi đầu đã gần heuristic, vì slot do heuristic điều khiển. Nó lên đỉnh sớm (0,4–1,2M bước) rồi đi ngang hoặc giảm. Ở K5 seed 1, val giảm từ +73,5 xuống +67,3.
- **Chi phí so sánh:** tinh chỉnh một regime heuristic dùng 132 thành viên × 30 seed, tức khoảng 4.000 episode, cùng bậc với số episode của một run PPO joint. Nhưng một episode heuristic chỉ mất khoảng 0,03 s, còn một bước PPO cần cả cập nhật mạng (khoảng 25–35 phút CPU mỗi run).

### 4.4 Action replacement: PPO joint yếu ở phần nào?

Mỗi lần, một hoặc nhiều phần hành động của PPO joint (best-val) được thay bằng phần tương ứng của T:best: di chuyển (M), kênh (C), công suất (P), mode (Mo). Bảng ghi Δ so với PPO joint nguyên bản, gộp 3 seed, trên tập test.

| thay bằng heuristic | ref | K5 | phy_mcs_blind |
|---|---|---|---|
| chỉ M | +0,47 | +0,97 | **−22,60** |
| chỉ C | **−2,31** | **−1,53** | +0,42 |
| chỉ P | +0,15 | +0,62 | +0,12 |
| chỉ Mo | **+3,35** | +0,46 (n.s.) | **+5,42** |
| M + P | +0,52 | **+2,62** | −22,68 |
| C + P + Mo | +0,77 | −0,67 | +5,91 |
| cả bốn (= T:best) | +0,87 | −2,19 | +5,53 |

(Đầy đủ 15 tổ hợp: `a5_summarize.py replace`.)

- **Phần kênh của PPO joint tốt hơn heuristic** ở ref và K5: thay kênh bằng heuristic thì mất 1,5–2,3.
- **Phần mode của PPO joint là điểm yếu** ở ref (+3,35 nếu thay) và phy_mcs_blind (+5,42). Điều này khớp với việc mode-only PPO (học riêng phần mode, với slot heuristic) là cấu hình tốt nhất.
- **Các phần phụ thuộc lẫn nhau rất mạnh.** Ở phy_mcs_blind, chỉ thay phần di chuyển bằng heuristic làm mất −22,6: PPO joint đã học cách di chuyển và chọn mode khớp với nhau, một luật di chuyển "lạ" phá vỡ sự khớp đó. Vì vậy không thể đọc từng phần độc lập.
- **Ở K5, M+P của heuristic cộng với C+Mo của PPO** cho +2,62 so với PPO joint, tức khoảng +4,8 so với T:best. Đây là gợi ý cho một kiến trúc lai: điều nhịp/công suất theo luật, kênh và mode do RL quyết định. Nhưng tổ hợp này được chọn sau khi nhìn kết quả test, nên chỉ là giả thuyết, chưa phải kết quả.

### 4.5 PPO A4 dùng zero-shot (không train lại) trên các regime K = 3

So với T:best của từng regime trên tập test (A4 joint / mode-only, mỗi loại 3 model gộp):
- **Gần như ngang (−0,7 đến −1,5):** ref, fade_k3, fade_k0, env_suburban.
- **Hơn T:best một chút:** phy_gap3 (+0,9 / +1,4) và dl_tight (joint +0,6, không có ý nghĩa).
- **Thua rõ:**
  - env_highrise (−139 / −156);
  - q_high (−18 / −5);
  - wt0 (−15 / −13);
  - phy_mcs_blind (joint −42);
  - v10 (joint −9,6);
  - phy_mcs (−8 / −5).

Policy A4 **không tổng quát** sang regime khác. Phần mode-only hỏng ít hơn joint, vì slot vẫn do heuristic điều khiển. Muốn dùng một policy duy nhất cho nhiều regime thì phải train với ngẫu nhiên hoá regime và cho regime vào quan sát. A5 không làm điều này.

### 4.6 Train K5 joint kéo dài lên 5M bước (thí nghiệm phụ, quyết định sau khi thấy đường học s0)

**Lý do.** Quyết định lúc 2026-11-08 02:00, khi đường học val của K5 joint s0 vẫn còn tăng ở 2M. Thí nghiệm này không thuộc quy tắc định trước.

**Cách làm.**
- 3 seed joint được train tiếp từ checkpoint 2M lên 5M, với env seed mới (`a5_train.py --resume`).
- Checkpoint chọn bằng cùng quy tắc best-val, trên toàn bộ log 0–5M.
- Đánh giá trên cùng 100 seed test.

**Kết quả trên val.** Đỉnh val ở 4,4M / 1,0M / 3,8M bước (seed 1 vẫn giữ checkpoint 1M cũ). Val tốt nhất +74,9 / +73,2 / +75,3, so với +71,5 / +73,2 / +73,8 ở 2M.

**Kết quả trên test (K5):**

| so sánh | Δ return |
|---|---|
| joint 5M (best-val) − T:best | **+3,49 [+2,59; +4,40]** (từng seed +3,72 / +3,15 / +3,60) |
| joint 5M − joint 2M | +1,30 [+0,98; +1,62] |
| joint 5M − heuristic A4 greedy+budget1+reuse (tốt nhất chọn sau) | +1,50 [+0,64; +2,36] |
| joint 5M − mode-only 2M | +0,36 [−0,29; +1,01] |
| joint 5M, checkpoint cuối − T:best | +1,73 [+0,77; +2,69] |

**Recall detector khi giao:** 0,974 / 0,974 / 0,969 so với gửi ảnh gốc, ngang joint 2M (0,972) và cao hơn T:best (0,931).

- **Train lâu hơn giúp joint ở K5:** +1,3. Joint 5M giờ ngang mode-only 2M và hơn mọi heuristic +1,5, kể cả heuristic chọn sau.
- **Nhưng:**
  1. Phải tốn 2,5 lần kinh nghiệm: khoảng 11,7k episode, 2,2M giây mô phỏng mỗi run.
  2. Checkpoint cuối kém hơn best-val 1,8, tức đường học vẫn dao động. Chọn checkpoint theo val tiếp tục là cần thiết.
  3. Mode-only không được train lâu hơn, nên so sánh joint 5M với mode-only 2M chưa cân bằng.
- Kết luận ở mục 5 không đổi về hướng. Riêng ở K5, joint không còn thua mode-only khi được train đủ lâu.

### 4.7 Task fidelity của những gì policy giao được

Đo bằng object recall của detector YOLO11s trên các khung giao đúng hạn (khung bị rớt, trễ hoặc vi phạm Qmin tính 0), chia cho recall khi gửi ảnh gốc mọi khung. Giá trị ở đây là trung bình của 3 seed. Đây là tỉ số gộp nên không có CI; nó là thước đo phụ, reward không đổi.

| pool | regime | fixed (tinh chỉnh) | T:best | PPO joint | PPO mode-only |
|---|---|---|---|---|---|
| test | ref | 0,971 | 0,980 | 0,980 | 0,981 |
| test | K5 | 0,916 | 0,931 | **0,972** | 0,952 |
| test | phy_mcs_blind | 0,920 | 0,905 | **0,961** | 0,953 |
| test-dev | ref | 0,961 | 0,953 | 0,968 | 0,968 |
| test-dev | K5 | 0,895 | 0,849 | **0,958** | 0,926 |
| test-dev | phy_mcs_blind | 0,895 | 0,807 | **0,935** | 0,927 |

- **Theo tác vụ, PPO không thua heuristic ở regime nào.** Ở K5 và phy_mcs_blind, nó giữ được 93–97% recall của ảnh gốc, trong khi T:best chỉ 81–93%. Lý do chính là PPO rớt/trễ ít khung hơn (ok 0,98–0,99 so với 0,90–0,97 của T:best).
- **Reward và thước đo tác vụ xếp hạng khác nhau.** PPO joint ở phy_mcs_blind thua T:best −5,5 theo reward nhưng có recall cao nhất (0,961). Nó bay chậm hơn (98 s so với 77 s), bị phạt thời gian, nhưng giao đủ khung với chất lượng đủ cho detector.
  - Tương tự, mode-only ở K5 có return cao nhất nhưng PSNR thấp hơn (34,6 dB so với 37,1), và recall thấp hơn joint.
- **Hệ quả:** "PPO thắng/thua" phụ thuộc vào thước đo. Một reward PSNR + thời gian không phản ánh đúng giá trị tác vụ khi Qmin cho phép mode chất lượng thấp (mục 2).

## 5. Khi nào chọn mode, reuse và điều khiển joint có ích

1. **Thích nghi theo trạng thái** (heuristic T:best, gồm chọn mode, điều nhịp và reuse, so với một mode cố định tinh chỉnh) có ích ở mọi regime có tải đáng kể:
   - +9,6 ở ref, +34 ở K5 (tập test).
   - Phần riêng của chọn mode thì không tách được trong so sánh này; mục 4.4 gợi ý nó lớn cỡ vài điểm.
   - Không có ích khi tải thấp (v10) hoặc khi Qmin quá cao (q_high), vì khi đó gần như chỉ còn một lựa chọn.
2. **Dùng chung kênh (reuse)** có ích tỉ lệ với mức tranh chấp: 0 ở K2; +2,1 ở ref so với budget; +16,6 ở K4; +33,6 ở K5. Nhưng chỉ dưới giả định PHY lý tưởng:
   - với gap 3 dB thì mất ưu thế;
   - với MCS chọn không biết nhiễu thì **có hại** (−22,9);
   - với Qmin cao cũng có hại (−6,5).
   - Nếu viết bài về reuse thì phải nói rõ điều kiện này.
3. **RL thêm gì so với heuristic mạnh:**
   - **Mode-only RL trên luật slot tốt** là cấu hình duy nhất có lợi nhất quán: +3,1 (K5) và +4,3 (phy_mcs_blind) trên test; lợi lớn hơn khi ảnh đổi phân phối (+12,3 ở K5 test-dev). Không có lợi ở ref.
   - **Joint RL** không thay được luật slot ở phy_mcs_blind (thua). Ở K5, sau 2M bước nó hơn T:best ít, ngang heuristic A4. Kéo lên 5M bước (mục 4.6) thì hơn mọi heuristic +1,5 và ngang mode-only, nhưng tốn 2,5 lần kinh nghiệm. Phần kênh nó học được tốt, nhưng phần mode thì kém.
   - **Phần lớn ưu thế của RL ở K5 biến mất** khi so với heuristic A4 không tinh chỉnh lại (còn +1,1 cho mode-only). Phần còn lại đáng kể là ở phy_mcs_blind và khi đổi phân phối ảnh.
4. **Theo tác vụ** (recall của detector), PPO tốt hơn hoặc ngang heuristic ở cả 6 cặp (regime, pool), chủ yếu vì rớt ít khung hơn.

## 6. Giới hạn của mô phỏng và đóng góp có thể bảo vệ

**Giới hạn:**
- **PHY là mô hình trừu tượng:** Shannon (mặc định), SNR gap, hoặc bảng CQI LTE với BLER cố định 10%. Không có HARQ, không có trace kênh đo thật. Ngưỡng CQI là giả định (ghi trong `env.py`).
- **Tập ảnh nhỏ:** test có 128 khung từ 28 chuỗi video; test-dev có 403 khung. Các khung trong cùng chuỗi tương quan với nhau, nên CI theo seed môi trường có thể hơi hẹp so với thực.
- **Reward dùng PSNR dự đoán.** Mục 4.7 cho thấy nó không đồng nhất với giá trị tác vụ.
- **Mode sinh ảnh** (caption → T2I, I2I) vô dụng cho phát hiện vật ở mức bit này. Mô hình không tính chi phí tính toán phía nhận ngoài độ trễ đã đo.
- **Một cell, quỹ đạo khảo sát cố định, không có di chuyển đối kháng hay hỏng UAV.** Các họ heuristic là do tôi thiết kế; có thể có heuristic tốt hơn, ví dụ heuristic tự hiệu chỉnh ngưỡng.
- **Mỗi regime train một policy riêng** (mục 4.5); không có policy tổng quát.

**Đóng góp có thể bảo vệ:**
1. **Một bản đồ chế độ vận hành có thể tái lập** (21 regime) cho bài toán giao ảnh đa UAV có chọn codec. Nó cho biết khi nào thích nghi codec, dùng chung kênh và điều nhịp có giá trị.
2. **Kết quả âm có điều kiện:** lợi ích của dùng chung kênh phụ thuộc vào giả định link adaptation lý tưởng.
3. **Kiến trúc lai** "luật slot + RL chọn mode" thắng heuristic mạnh nhất có ý nghĩa ở regime tranh chấp và regime PHY không lý tưởng, với lợi lớn hơn khi ảnh đổi phân phối. Joint RL thì không thắng.
4. **Đo task fidelity** cho thấy reward PSNR lệch với recall detector, và Qmin 22–30 dB cho phép các mode gây hại tác vụ.

## 7. Bước tiếp theo hướng tới bài báo Q1

1. **Baseline thích nghi mạnh hơn cho kết luận về đổi phân phối:** một heuristic tự hiệu chỉnh ngưỡng backlog/budget theo payload quan sát trực tuyến. Nếu nó bắt kịp PPO trên test-dev thì ưu thế "bền" của PPO không đứng vững. Việc này rẻ, chỉ tốn thời gian heuristic.
2. **Reward theo tác vụ:** một thí nghiệm riêng, định trước, thay PSNR bằng proxy recall (đã có bảng per-image của mục 2). Kiểm tra xem thứ hạng PPO/heuristic có đổi không. Đây là đổi mục tiêu, nên phải báo cáo tách riêng, không trộn với A5.
3. **Kiến trúc lai được định trước:** M+P theo luật, C+Mo do RL (gợi ý của 4.4), train và kiểm tra trên một bộ seed **mới**.
4. **PHY thực tế hơn:** đường BLER cấp link với HARQ, hoặc trace đo (ví dụ dataset UAV công khai). phy_mcs_blind cho thấy giả định PHY đổi kết luận, nên đây là điều reviewer sẽ hỏi.
5. **Quy mô thống kê:**
   - 5 seed train;
   - toàn bộ 1610 ảnh test-dev;
   - CI theo chuỗi video (bootstrap theo chuỗi) thay cho theo seed.
6. **Tổng quát hoá:** train một policy với ngẫu nhiên hoá K, môi trường truyền sóng và PHY (có regime trong quan sát), so với policy riêng từng regime (mục 4.5 cho thấy zero-shot hỏng).
