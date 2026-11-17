# A-E2 / A-E3 và B-E1 / B-E2: MPC so với PPO; nén token, predictor vòng kín, AVIF qua DINOv2

Ngày 15–16/11/2026, theo phản biện #28. Giai đoạn khám phá. Seed 8000..8019 đã được dùng ở A5.5, và lưới (N, H) của MPC được xem trên chính các seed này, nên mọi so sánh ở đây là **khám phá**, chưa phải xác nhận.

## Phần A: MPC chỉ chọn mode so với PPO mode-only (A-E2, A-E3)

**Thiết kế** ([a9_mpc.py](../../swarmbridge/experiments/uav_semcom/a9_mpc.py), [test_a9_mpc.py](../../swarmbridge/experiments/uav_semcom/test_a9_mpc.py), 6 test):
- **Quyền điều khiển như nhau:** MPC chỉ chọn mode của ảnh đang đến lượt. Di chuyển, kênh và công suất do **cùng** slot policy với PPO mode-only quyết định (phần slot của heuristic T:best).
- **Thông tin như nhau:**
  - bit / PSNR từng mode chỉ qua `mode_info` (`predicted`);
  - trạng thái hàng đợi; vị trí;
  - CSI **thống kê**: path gain, xác suất LoS và **phân phối** fading tại vị trí hiện tại.
- **Không đọc fading đã rút sẵn.** Có test: đổi fading tương lai thì quyết định không đổi.
- **Khung tương lai:** dùng bit / PSNR trung bình của tập train, deadline / Qmin trung bình, tới mỗi `frame_spacing / v_max` giây.
- **Planner:** với mỗi mode ứng viên, N đường fading mẫu trên 40 slot. Mô phỏng encoder nối tiếp và hàng đợi FIFO theo từng mẫu: ảnh phía trước, ảnh hiện tại, rồi các ảnh đang chờ và H khung tương lai được chọn bằng rollout greedy. Giá trị = trung bình utility. N = 0 nghĩa là một đường tất định tại gain kỳ vọng.
- **Mô hình tốc độ:**
  - `env` = PHY thật của regime (MCS ở phy_mcs_blind; slot policy ở đó không dùng chung kênh, nên MCS là đúng, chỉ thiếu fading tương lai);
  - `shannon` = mô hình của các heuristic, sai dưới MCS.
- **Ngân sách thời gian (anytime):** mode của heuristic được tính trước và đánh giá làm phương án hiện hành. Hết giờ thì trả về mode tốt nhất đã đánh giá. Thời gian tính cả bước dự phòng.
  - Bản đầu trả về "tốt nhất trong các mode đã xét theo thứ tự chỉ số", và đã cho kết quả vô nghĩa (ref: −213, vì toàn chọn mode 0). Tôi đã bỏ bản đó.

**Kết quả (20 seed, pool test; Δ theo cặp so với trung bình 3 model PPO mode-only, CI 95 % theo seed):**

phy_mcs_blind (PPO = 30,27; T:best cùng slot policy = 26,94; H* = 26,14):

| MPC | Return | Δ so với PPO [CI] | ms / quyết định (10 tiến trình) |
|---|---|---|---|
| đúng, N = 0, H = 0 | −10,13 | −40,4 | 1,6 |
| đúng, N = 8, H = 0 | 10,69 | −19,6 | 1,9 |
| đúng, N = 0, H = 4 | 13,96 | −16,3 | 78 |
| **đúng, N = 8, H = 4** | **32,95** | **+2,68 [+1,42; +3,94]** | **77** |
| đúng, N = 32, H = 8 | 32,26 | +1,99 [+0,73; +3,24] | 127 |
| đúng, N = 64, H = 8 | 33,05 | +2,78 [+1,46; +4,10] | 131 |
| **sai (Shannon), N = 64, H = 8** (tốt nhất của Shannon) | **21,65** | **−8,62 [−10,5; −6,7]** | 130 |
| sai (Shannon), N = 8, H = 4 | 10,10 | −20,2 | 78 |
| đúng, N = 8, H = 4, ngân sách 1,2 / 5 / 20 ms | 26,44 / 26,46 / 27,36 | −3,8 / −3,8 / −2,9 | 6,5 / 8,0 / 22 (gần như mọi quyết định hết giờ) |

ref (Shannon là mô hình đúng; PPO = 58,33; T:best = 58,12):

| MPC | Return | Δ so với PPO [CI] | ms / quyết định |
|---|---|---|---|
| N = 0, H = 0 | 55,76 | −2,56 | 1,4 |
| N = 8, H = 0 | 58,60 | +0,28 [−0,58; +1,14] | 1,3 |
| N = 8, H = 4 | 60,92 | +2,60 [+1,37; +3,82] | 45 |
| N = 32, H = 4 | 61,15 | +2,83 [+1,65; +4,00] | 47 |
| N = 8, H = 4, ngân sách 1,2 / 5 / 20 ms | 58,10 / 58,12 / 58,63 | ≈ 0 | 4,7 / 7,4 / 22 |

**Đo thời gian riêng với ít tải** (3 tiến trình, 3 seed): MPC đúng N = 8, H = 4 mất 33 ms trung bình (p95 43 ms); N = 32, H = 8 mất 67 ms. PPO mất 1,2 ms (A-E1, cùng máy).

**Đọc kết quả:**
1. **Với mô hình tốc độ đúng và có nhìn trước (H ≥ 4), MPC tốt nhất:** hơn PPO khoảng +2,6 đến +2,8 return ở cả hai regime. Đổi lại, chi phí khoảng **30–100 lần PPO** (33–130 ms so với 1,2 ms).
2. **Ở mức tính toán ngang PPO** (khoảng 1–2 ms: N ≤ 8, H = 0), MPC **thua xa** PPO ở phy_mcs_blind (−19,6 đến −40) và chỉ ngang PPO ở ref. Khi rate là MCS, cần cả mẫu fading lẫn nhìn trước, vì ngưỡng CQI khiến tốc độ tại gain kỳ vọng sai lệch rất xa.
3. **Sai mô hình phá MPC:** MPC dùng Shannon dưới kênh MCS, kể cả ở mức tính toán cao nhất, vẫn thua PPO 8,6 điểm và thua cả heuristic.
   - PPO học từ trải nghiệm với kênh thật nên không có vấn đề này.
   - Đây là bằng chứng cơ chế đầu tiên, phù hợp với giả thuyết "ưu thế của PPO ở phy_mcs_blind đến từ việc nó không phụ thuộc mô hình tốc độ sai".
4. **Ngân sách ≤ 20 ms không đủ cho một lần đánh giá đầy đủ** trong bản Python hiện tại, nên MPC quay về mode của heuristic (ngang T:best).

**Giới hạn:**
- MPC viết bằng Python / numpy, chưa tối ưu. Một bản C++ hay vector hoá sâu hơn có thể nhanh hơn 10–100 lần, nên tỉ lệ chi phí phụ thuộc vào cách cài đặt.
- MPC không mô hình hoá nhiễu đồng kênh (ở ref slot policy có dùng chung kênh).
- MPC không phải cận trên tối ưu.
- 20 seed, chọn (N, H) sau khi xem kết quả.

**Hệ quả cho Paper A (đề xuất):** câu chuyện có hai vế đo được:
- (a) với mô hình đúng, planner trực tuyến tốt hơn PPO một chút nhưng đắt hơn vài chục lần;
- (b) với mô hình sai, planner sụp còn PPO thì không.

Cơ chế đáng phát triển là một bộ điều khiển kết hợp: PPO làm prior / phương án hiện hành nhanh; planner dùng một mô hình tốc độ **được hiệu chỉnh trực tuyến** từ ACK khi còn ngân sách. Thí nghiệm kế tiếp:
1. MPC với mô hình tốc độ học hoặc hiệu chỉnh từ dữ liệu: có thu hẹp khoảng cách khi sai mô hình không.
2. PPO làm phương án hiện hành cho MPC anytime.
3. Tối ưu tốc độ MPC để vẽ đường return – ms thật sự.

## Phần B: nén token, predictor vòng kín, AVIF qua DINOv2 (B-E1, B-E2)

**Thiết kế** ([b1_pipeline.py](../../swarmbridge/experiments/jepa_semcom/b1_pipeline.py), [test_b1.py](../../swarmbridge/experiments/jepa_semcom/test_b1.py), 12 test):
- **Dữ liệu:** VisDrone-VID theo split v1; lấy 1 khung trong 3, tối đa 100 khung mỗi clip.
  - 48 clip TUNE để fit PCA, head và predictor.
  - 7 clip `gate` (nguồn train giữ riêng) + 3 clip `aux_val` để đánh giá: 10 clip, 796 khung.
  - Trích token cho 58 clip + token của khung AVIF đã giải mã ở 5 mức: 9 phút, bộ nhớ đỉnh 220 MB.
- **Không gian truyền:** PCA d = 32 / 64 của token DINOv2-S, chỉ fit trên TUNE (286k patch), chuẩn hoá theo độ lệch chuẩn. Phần phương sai giữ lại: 67 % (32 chiều), 81 % (64 chiều).
- **Thông điệp:**
  - header;
  - chỉ số patch (bitmap hoặc exp-Golomb);
  - vector chuyển động 8 bit nếu dùng (có tính);
  - phần dư lượng tử bước 0,25σ, mã exp-Golomb có dấu (bậc chọn theo thông điệp).
  - Khung 0 gửi mọi patch, có tính bit.
- **Đồng bộ:** bên nhận và bên gửi giữ lịch sử **riêng**; bên nhận chỉ dùng thông tin trong bitstream (kể cả vector chuyển động); `assert` trùng từng bit ở mọi khung.
- **Predictor (vòng kín, từ trạng thái đã tái tạo):**
  - `hold`: giữ nguyên.
  - `mc`: giữ nguyên dịch theo vector chuyển động nguyên (±4 patch). Bên gửi chọn và **gửi** vector, 8 bit/khung.
  - `learned` (không JEPA): mạng conv 642k / 826k tham số, dự đoán phần thay đổi từ hai trạng thái trước, MSE. Huấn luyện với đầu vào giả lập trạng thái bên nhận (nhiễu lượng tử, patch cũ).
  - `complete` (kiểu I-JEPA): sau khi nhận k patch, bên nhận dự đoán lại latent của các patch **không** được gửi từ ngữ cảnh (patch mới + mask + trạng thái trước). Hàm mất mát chỉ tính trên patch bị che, cùng kiến trúc và cùng cỡ tham số.
    - Target encoder là DINOv2 + PCA **đóng băng**, không có encoder học và không có EMA. Đây là dự đoán latent có che với target đóng băng, **chưa phải JEPA đầy đủ**.
- **Nhiệm vụ và AVIF:**
  - cùng head tuyến tính 4 lớp, train trên token gốc 384 chiều của TUNE;
  - mọi phương pháp đưa trạng thái bên nhận về 384 chiều (PCA ngược) rồi qua head;
  - AVIF: khung 308 × 546 mã AVIF (speed 6, 4:2:0), **giải mã**, qua DINOv2-S, qua cùng head; bit = byte AVIF, mã độc lập từng khung.

**Kết quả (trung bình 10 clip; mAP với token gốc = 0,475; bit / khung tính cả khung khởi tạo):**

| Phương pháp | k | Bit / khung | mAP |
|---|---|---|---|
| PCA-32 không lượng tử (trần của d = 32) | – | – | 0,356 |
| PCA-64 không lượng tử (trần của d = 64) | – | – | 0,414 |
| d32 hold / mc / learned / complete | 16 | 4,0–4,2 k | 0,176 / **0,194** / 0,122 / 0,116 |
| d32 hold / mc / learned / complete | 64 | 10,6–10,9 k | 0,285 / **0,297** / 0,227 / 0,267 |
| d32 hold / mc / learned / complete | 128 | 18,3–18,7 k | 0,326 / **0,333** / 0,287 / 0,324 |
| d32 hold / mc | 256 | 31,4–31,6 k | 0,346 / 0,350 |
| d64 hold / mc / learned / complete | 64 | 21,2–21,7 k | 0,307 / **0,326** / 0,251 / 0,288 |
| d64 hold / mc / learned / complete | 128 | 37,3–37,7 k | 0,374 / **0,381** / 0,325 / 0,356 |
| d64 hold / mc / learned / complete | 256 | 64,6–66,7 k | 0,401 / **0,404** / 0,377 / 0,394 |

| AVIF → DINOv2 → head | q5 | q10 | q20 | q35 | q50 |
|---|---|---|---|---|---|
| kbit / khung | 18,7 | 24,7 | 36,4 | 64,2 | 118 |
| mAP | 0,300 | 0,336 | 0,386 | 0,429 | 0,458 |

**Chẩn đoán vòng hở** (lịch sử thật, MSE chuẩn hoá trên mỗi chiều; `runs/jepa_semcom/b1_diag_openloop.json`):

| d | hold | mc | learned | không gửi (top-64 cho trước): hold | không gửi: complete |
|---|---|---|---|---|---|
| 32 | 0,152 | 0,144 | 0,158 | 0,096 | 0,098 |
| 64 | 0,184 | 0,176 | 0,180 | 0,126 | 0,131 |

**Đọc kết quả:**
1. **Nén token hoạt động.** Ở cùng 64 patch mỗi khung, bit giảm khoảng 4,6 lần với d = 64 (21,7 kbit) và khoảng 9 lần với d = 32 (10,9 kbit), so với 100 kbit của token thô q4 ở B-E0. B-E0 dùng 3 clip khác và không tính khung khởi tạo vào trung bình, nên chỉ so được về cỡ. Tại khoảng 18–19 kbit, d32 k = 128 (0,326–0,333) hơn AVIF q5 (0,300).
2. **Nhưng PCA làm mất thông tin nhiệm vụ:** trần giảm từ 0,475 xuống 0,414 (d = 64) và 0,356 (d = 32). Từ khoảng 25 kbit trở lên, AVIF → DINOv2 tốt hơn mọi biến thể token: AVIF q20 0,386 @ 36 kbit so với d64 mc k = 128 0,381 @ 37,5 kbit; AVIF q35 0,429 @ 64 kbit so với 0,404 @ 65 kbit. Muốn thắng AVIF ở dải này cần một bottleneck **giữ thông tin nhiệm vụ**, không chỉ giữ phương sai.
3. **Bù chuyển động có ích nhưng ít:** +0,01 đến +0,02 mAP ở k nhỏ; MSE vòng hở giảm khoảng 5 %.
4. **Cả predictor học (MSE) lẫn bộ hoàn thiện kiểu JEPA đều kém giữ nguyên** khi chạy vòng kín, và gần như ngang (hoặc kém hơn) ngay cả ở vòng hở. Ở khoảng 10 khung/giây, phần thay đổi của token DINOv2 đóng băng giữa hai khung hầu như **không dự đoán được** từ lịch sử: hold đã giữ phần dễ, phần còn lại giống nhiễu đối với các mạng này. Vấn đề không nằm ở vòng kín.
5. **Hệ quả cho JEPA:** với target đóng băng (DINOv2 + PCA), không có lý do để kỳ vọng JEPA thắng, vì latent không được học để dự đoán được. Lập luận mạnh nhất của JEPA nằm ở việc **học biểu diễn sao cho tương lai dự đoán được mà vẫn giữ thông tin nhiệm vụ**, tức là encoder học được với target EMA. Đây là bước kế tiếp hợp lý cho Paper B (B-E3), và cũng là chỗ khác biệt với "latent predictive regression".

**Giới hạn:**
- Head train trên token gốc; PCA ngược có thể lệch phân phối. Train lại head trên token tái tạo có thể nâng trần.
- AVIF là mã độc lập từng khung; một codec video liên khung (AV1 / HEVC) sẽ là đối thủ mạnh hơn.
- 10 clip. Kênh không mất gói.
- Predictor huấn luyện ngắn (6.000 bước), chưa tinh chỉnh.

**Bước tiếp theo đề xuất cho Paper B:**
1. **Bottleneck giữ thông tin nhiệm vụ:** tuyến tính học, hàm mất mát = tái tạo + nhiệm vụ trên TUNE, thay cho PCA. Đo lại so với AVIF.
2. **B-E3, JEPA đầy đủ quy mô nhỏ:** encoder nhẹ học trên token DINOv2 (hoặc trên ảnh) với target EMA, mất mát dự đoán latent tương lai / bị che + điều chuẩn chống sụp. So với cùng encoder học bằng tái tạo / nhiệm vụ (không JEPA), cùng ngân sách tham số. Chấm theo bit – mAP vòng kín.
3. **Codec video liên khung** làm baseline thứ hai (nếu có ffmpeg / libaom trong môi trường).

## Code và kết quả

- **A:**
  - `swarmbridge/experiments/uav_semcom/a9_mpc.py`, `test_a9_mpc.py`;
  - `runs/uav_semcom/a9/ae2_mpc_phy_mcs_blind.json`, `ae2_mpc_ref.json`, `ae2_mpc_timing_3procs.json`.
- **B:**
  - `swarmbridge/experiments/jepa_semcom/b1_pipeline.py`, `test_b1.py`;
  - `runs/jepa_semcom/b1_clips.json`, `b1_extract_stats.json`, `b1_fit_stats.json`, `b1_train/`, `b1_eval.json`, `b1_diag_openloop.json`.
  - Token và mô hình nằm ở `~/work/jepa_semcom/` (ngoài git).
- GPU lock được giữ cho mọi bước GPU (trích token, huấn luyện).
