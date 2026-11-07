# SwarmBridge hướng A — A5: mở rộng dữ liệu, task fidelity, khảo sát chế độ vận hành, RL có mục tiêu

Bắt đầu 2026-11-07, tiếp nối A4 (commit `0e77387a`, báo cáo `reports/uav-semcom-a.md` mục 8).

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

## 4. PPO vs heuristic mạnh
(chưa chạy)
