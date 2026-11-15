# A-E5 và B-E4: cổng tin cậy theo sai số mô hình cho PPO–MPC; lặp lại JEPA-32 với seed ghép cặp

Ngày 16/11/2026, theo phản biện #30. Vẫn là giai đoạn khám phá:
- A: 20 seed (8000..8019, như A-E4).
- B: 10 clip đánh giá, 3 seed huấn luyện ghép cặp.

## Phần A (A-E5): bản kết hợp PPO–MPC có cổng tin cậy theo sai số mô hình

### Cơ chế

**Bộ theo dõi sai số tốc độ** (`RateErrTracker`, [a9_mpc.py](../swarmbridge/experiments/uav_semcom/a9_mpc.py)):
- Đầu mỗi slot, với từng UAV, nó lưu tốc độ trung bình mà mô hình của planner dự đoán tại vị trí hiện tại:
  - `env` / `shannon`: trung bình Monte-Carlo qua phân phối fading thống kê;
  - `cal` / `cals`: trung bình của bộ hiệu chỉnh, hoặc Shannon nhân ρ khi khoảng SNR chưa có dữ liệu.
- Hết slot, throughput quan sát được của slot đó được ghép với dự đoán đã lưu, gom theo khoảng 2 dB SNR kỳ vọng.
- Cách làm này là **prequential**: dự đoán có trước khi slot được quan sát; có test kiểm tra.
- `bias(k)` = Σ quan sát / Σ dự đoán − 1 trong khoảng SNR hiện tại của UAV, gộp thêm khoảng lân cận cho đến khi đủ 20 s.

**Bản kết hợp có cổng sai số** (`ehyb`, [a10_hybrid.py](../swarmbridge/experiments/uav_semcom/a10_hybrid.py)):
- PPO đề xuất mode.
- MPC (N = 8, H = 4, rollout phi tiên tri, ngưỡng đổi Z = 1) **chỉ chạy khi bias đã biết và |bias| ≤ τ**. Ngược lại dùng ngay mode của PPO, nên chỉ tốn chi phí PPO.
- τ = 0,1 được **chọn trước khi chạy**. Các giá trị 0,05 / 0,2 / 0,3 chỉ là độ nhạy, không dùng để chọn.

**Mô hình quan sát** (theo phản biện #30, ghi rõ giới hạn):
- Gọi là *observed-throughput calibration*.
- Throughput của các đoạn truyền đã xong được đọc từ simulator ở ranh giới slot kế tiếp. Cách này nhân quả theo thời gian, nhưng **chưa mô phỏng giao thức ACK/feedback**, độ trễ hay chi phí của nó.

### Phát hiện phụ: log throughput theo thời gian phát bị lệch thấp

Khi viết test (mô hình PHY đúng phải có bias ≈ 0), bias đo được lại là **−12 % đến −14 %** (3 episode).

Nguyên nhân là hiệu ứng chọn mẫu:
- Slot fading tốt: hàng đợi trống sớm, nên chỉ ghi được ít thời gian ở tốc độ cao.
- Slot fading xấu: phát cả slot ở tốc độ thấp.
- Vì vậy trung bình có trọng số theo thời gian phát thấp hơn tốc độ thực của kênh.

Bộ hiệu chỉnh `cal` của A-E4 dùng đúng kiểu trọng số này, nên nó **bi quan có hệ thống**: bias +6 % đến +16 % khi đo trên quan sát theo slot.

Cách sửa (`cals`, tùy chọn `slot_weight` của `CalRates`):
- Mỗi cặp (UAV, slot) có phát được tính là **một** quan sát của tốc độ slot đó.
- UAV có phát trong slot hay không được quyết định trước khi fading của slot được rút, nên ước lượng không lệch.
- Sau khi sửa, mô hình PHY đúng có bias +3,4 % / +2,8 % / −2,0 %, còn Shannon −46 %.
- Test: `test_err_tracker_correct_model_unbiased_shannon_biased`, `test_airtime_weighting_is_biased_low`, `test_err_tracker_is_prequential` (11/11 test của a9 đạt).

### Kết quả: phy_mcs_blind

Δ so với **đúng model PPO của nó**, theo cặp seed, CI 95 %. "Gộp" = trung bình của Δ trên 3 model PPO. Thời gian đo khi 12 tiến trình chạy trên 6 nhân WSL (giống A-E4). PPO: 30,03 / 30,27 / 30,51.

| Bộ điều khiển | Return (3 model PPO) | Δ so với PPO | Tỉ lệ quyết định có lập kế hoạch | ms TB (p95) | > 100 ms |
|---|---|---|---|---|---|
| Kết hợp, PHY đúng, không cổng (A-E4) | 31,91 / 32,52 / 32,36 | gộp +1,99 [+1,00; +2,98] | 100 % | 53–102 | – |
| Kết hợp, Shannon, không cổng (A-E4) | 16,39 (model 0) | −13,64 [−16,53; −10,76] | 100 % | 65 | 17 % |
| **Kết hợp, Shannon, cổng sai số** | 30,03 / 30,27 / 30,51 | **0 (đúng bằng PPO)** | **0 %** | 3,6–4,0 (10) | 0 |
| Kết hợp, PHY đúng, cổng sai số | 30,46 / 30,75 / 30,84 | gộp +0,42 [−0,54; +1,37] | 45–49 % | 49–53 (135) | 28–30 % |
| Kết hợp, `cal` A-E4, cổng có/không dữ liệu (A-E4) | 30,40 / 30,39 / 31,77 | gộp +0,58 [−0,73; +1,89] | 99 % | 106 (146) | 65 % |
| Kết hợp, `cal` A-E4, cổng sai số | 30,70 / 30,20 / 30,15 | +0,66 / −0,07 / −0,35 | 5–23 % | 9–27 | 3–13 % |
| Kết hợp, `cals`, cổng có/không dữ liệu | 30,62 / 30,88 / 31,16 | gộp +0,62 [−0,75; +1,99] | 99 % | 94–106 (145) | 47–65 % |
| **Kết hợp, `cals`, cổng sai số** | **31,56 / 32,15 / 32,17** | **+1,53 / +1,88 / +1,67; gộp +1,69 [+0,63; +2,75]** | 88–91 % | 85–98 (144) | 41–59 % |
| MPC riêng, `cal` warm | 30,94 | +0,91 so với PPO 0 | 100 % | 101 | 56 % |
| MPC riêng, `cals` warm | 30,55 | +0,52 so với PPO 0 | 100 % | 89 | 39 % |

Độ nhạy theo τ (model PPO 0, `cals`):

| τ | 0,05 | **0,1 (chọn trước)** | 0,2 | 0,3 |
|---|---|---|---|---|
| Δ so với PPO | +0,63 | +1,53 | +0,27 | +0,67 |
| Tỉ lệ lập kế hoạch | 64 % | 91 % | 98 % | 99 % |

### Kết quả: ref

Ở regime này PHY là Shannon có tái dùng kênh, nên mô hình `env` trùng với `shannon`; planner không mô hình nhiễu đồng kênh. PPO: 58,16 / 58,32 / 58,49.

| Bộ điều khiển | Return | Δ so với PPO | Tỉ lệ lập kế hoạch | ms TB |
|---|---|---|---|---|
| Kết hợp, PHY (Shannon), không cổng (model 0) | 61,29 | +3,12 [+2,08; +4,16] | 100 % | 79 |
| Kết hợp, PHY (Shannon), cổng sai số | 58,99 / 58,88 / 59,01 | gộp +0,64 [+0,34; +0,94] | 15–16 % | 17–18 |
| Kết hợp, `cal`, cổng có/không dữ liệu | 58,24 / 57,45 / 58,62 | gộp −0,22 [−1,04; +0,59] | 99 % | 76–79 |
| Kết hợp, `cal`, cổng sai số | 58,11 / 58,35 / 58,37 | ≈ 0 | 0–1 % | 3,4–4,7 |
| **Kết hợp, `cals`, cổng sai số** | **59,98 / 60,14 / 60,44** | **gộp +1,86 [+1,14; +2,58]** (cả 3 CI > 0) | 91–93 % | 72–74 |
| MPC riêng, `cal` warm | 57,24 | −0,92 so với PPO 0 | 100 % | 78 |
| MPC riêng, `cals` warm | 60,05 | +1,89 [+0,79; +2,99] so với PPO 0 | 100 % | 69 |

### Đọc kết quả A

1. **Cổng sai số chặn được mô hình lệch.** Shannon trong môi trường MCS có bias trung vị −46 %, nên cổng không bao giờ mở. Bản kết hợp bằng đúng PPO, kể cả về chi phí, thay vì rơi xuống 16,39. Đây là điều cổng "có đủ mẫu" của A-E4 không làm được.
2. **Một bộ điều khiển chỉ dùng throughput quan sát được** (bộ hiệu chỉnh theo slot + cổng sai số) hơn PPO **+1,69** (phy_mcs_blind) và **+1,86** (ref), CI gộp > 0 ở cả hai regime.
   - Phần lớn khoảng cách so với bản dùng PHY đúng đã được thu hẹp: +1,99 ở phy_mcs_blind; +3,12 ở ref (chỉ model 0).
   - Nó không cần biết bảng MCS.
3. **Sửa trọng số hiệu chỉnh là thay đổi lớn ở ref:** MPC riêng `cal` 57,24 so với `cals` 60,05.
4. **Phần đóng góp riêng của cổng khi đi cùng `cals` chưa chắc chắn.** `cals` có cổng trừ `cals` không cổng: +0,94 / +1,26 / +1,02, nhưng CI đều chứa 0.
   - Độ nhạy theo τ **không đơn điệu**: chỉ τ = 0,1 cho +1,5; các giá trị khác +0,3 đến +0,7.
   - Ở τ ≥ 0,2 cổng gần như không chặn gì (98–99 % quyết định vẫn lập kế hoạch), nên kết quả về lại mức của bản không cổng.
   - Vì vậy chưa thể nói cổng chọn đúng *những* quyết định nên lập kế hoạch. Có thể một phần là may mắn ở τ = 0,1.
5. **Cổng hiện tại quá bảo thủ ở hai chỗ:**
   - Với **mô hình PHY đúng** (phy_mcs_blind), ước lượng bias theo khoảng SNR với khoảng 20 s dữ liệu dao động khoảng ±15 % (phân vị 10–90 %). Cổng do đó từ chối khoảng 50 % quyết định, và lợi ích giảm từ +1,99 xuống +0,42.
   - Ở **ref**, mô hình Shannon bỏ qua nhiễu đồng kênh, có bias trung vị −21 %, nên bị chặn khoảng 85 %. Nhưng chạy không cổng, nó lại cho +3,12. **Lệch về trung bình tốc độ không đồng nghĩa với ra quyết định sai.**
6. Bộ hiệu chỉnh vượt cổng dễ hơn mô hình cố định: dự đoán của nó là trung bình thực nghiệm của chính dòng dữ liệu đó (phân vị bias ±7 %, so với ±15 % của mô hình PHY đúng). Đây là bất đối xứng cần nhớ khi so các mô hình qua cùng một cổng.

### Chi phí

- Bản kết hợp có cổng tốn đúng bằng PPO (3,4–4 ms) khi cổng đóng, và khoảng 85–98 ms trung bình khi mở (phy_mcs_blind; ref 72–74 ms).
- 41–59 % quyết định vượt 100 ms trong điều kiện 12 tiến trình trên 6 nhân.
- Ngân sách vẫn là soft; chưa có hard deadline, và số vượt 100 ms chỉ là thống kê, không được cưỡng chế.

## Phần B (B-E4): JEPA-32 so với task-32 với 3 seed ghép cặp

### Thay đổi

Đúng như phản biện #30 chỉ ra, ở B-E3 hai model `task` / `jepa` có cùng khởi tạo nhưng bộ lấy mẫu NumPy dùng chung, nên chúng học trên hai chuỗi minibatch khác nhau.

Ở B-E4, mỗi model của seed S reset **cả** NumPy và torch theo S:
- cùng trọng số encoder/decoder ban đầu;
- cùng chuỗi clip/khung huấn luyện;
- predictor gắn sau cũng dùng seed S.

Kiểm tra: `task_s0` cho loss huấn luyện trùng hoàn toàn với `task-32` của B-E3.

Các thiết lập còn lại giữ nguyên B-E3: d = 32, 4000 vòng, cùng codec vòng kín, 10 clip, kết quả từng clip giữ trong `b4_eval.json`.

### Kết quả (mAP trung bình 10 clip, token gốc = 0,475)

| | seed 0 | seed 1 | seed 2 | TB |
|---|---|---|---|---|
| Trần task / jepa (không nén) | 0,475 / 0,486 | 0,490 / 0,495 | 0,492 / 0,490 | |
| mc k = 64 (khoảng 10,5 kbit): task / jepa | 0,338 / 0,374 | 0,362 / 0,379 | 0,362 / 0,359 | 0,354 / 0,371 |
| **mc k = 128 (khoảng 18 kbit): task / jepa** | 0,403 / 0,428 | 0,427 / 0,441 | 0,423 / 0,428 | **0,418 / 0,432** |
| hold k = 128: task / jepa | 0,395 / 0,425 | 0,421 / 0,433 | 0,417 / 0,417 | 0,411 / 0,425 |
| Predictor riêng JEPA, k = 128 | 0,408 | 0,415 | 0,404 | 0,409 |
| MSE vòng hở hold: task / jepa | 0,118 / 0,105 | 0,124 / 0,106 | 0,125 / 0,108 | |
| MSE vòng hở predictor riêng JEPA | 0,090 | 0,093 | 0,092 | |

Δ JEPA − task theo cặp:

| | seed 0 | seed 1 | seed 2 | TB [CI 95 % theo seed, t, n = 3] | Cặp clip×seed JEPA thắng | Clip có Δ TB > 0 |
|---|---|---|---|---|---|---|
| mc k = 64 | +0,037 | +0,017 | −0,003 | +0,017 [−0,032; +0,066] | 21/30 | 7/10 |
| **mc k = 128** | +0,025 | +0,013 | +0,005 | **+0,015 [−0,010; +0,040]** | 22/30 | 8/10 |
| hold k = 128 | +0,031 | +0,012 | 0,000 | +0,014 [−0,025; +0,053] | 22/30 | 8/10 |

- Ở cùng k, số bit hai bên gần như bằng nhau: chênh dưới 0,2 kbit ở k = 128, JEPA ít hơn chút.
- Bỏ clip có Δ lớn nhất, Δ trung bình mc k = 128 còn +0,011. Clip đóng góp lớn nhất chiếm 32 % tổng chênh lệch, so với khoảng 40 % ở B-E3.

### Đọc kết quả B

1. **Cơ chế lặp lại ổn định:** ở cả 3 seed, mã JEPA ổn định hơn theo thời gian (MSE hold thấp hơn 11–15 %), và predictor riêng của JEPA tốt nhất ở vòng hở.
2. **Lợi ích mAP nhỏ hơn và phụ thuộc seed:** +0,025 → +0,013 → +0,005. Trung bình +0,015, nhưng CI theo seed chứa 0.
   - Một phần của +0,027 ở B-E3 là do model task seed 0 yếu hơn các seed khác (0,403 so với 0,427 / 0,423).
   - JEPA dao động giữa các seed ít hơn (độ lệch chuẩn 0,007 so với 0,013).
3. **Mã ổn định hơn không tự động thành mAP cao hơn.** MSE hold giảm khá đều qua các seed, nhưng mAP thì không.
4. **Predictor riêng của JEPA vẫn kém hold trong vòng kín** ở cả 3 seed (0,409 so với 0,425), dù tốt nhất ở vòng hở. Predictor gắn sau trên mã JEPA kém hơn trên mã task ở seed 1–2.
   - Khoảng cách vòng hở / vòng kín vẫn là điểm nghẽn chính.

Kết luận khám phá: JEPA-32 **không kém hơn** và có xu hướng tốt hơn non-JEPA một chút (+0,015 mAP ở khoảng 18 kbit). Hiện chưa đủ để nói mục tiêu JEPA đem lại lợi ích riêng đáng kể. Phần chắc chắn của nhánh B vẫn là bottleneck giữ nhiệm vụ (so với PCA và AVIF nội khung).

## Giới hạn chung

- A: 20 seed khám phá, cùng tập seed với A5.5 / A-E4; τ chọn trước nhưng độ nhạy không đơn điệu; một episode mỗi seed; mô hình quan sát là đọc throughput từ simulator (chưa có ACK/feedback thực, độ trễ, chi phí); thời gian đo khi oversubscribed và không cưỡng chế deadline.
- B: 3 seed (CI rất rộng với n = 3); 10 clip đánh giá; chỉ d = 32 và cấu hình mặc định; AVIF nội khung vẫn là đối thủ duy nhất ngoài PCA, chưa có codec video liên khung.

## Cơ hội phát triển phương pháp tiếp theo

**A:**
1. **Cổng thống kê thay vì ngưỡng điểm:** chỉ từ chối khi bias *chắc chắn* vượt τ (khoảng tin cậy của bias theo số slot quan sát). Mục tiêu: giữ được +2 của mô hình PHY đúng mà vẫn chặn Shannon.
2. **Sai số liên quan đến quyết định:** thay sai số trung bình tốc độ bằng sai số dự đoán *giá trị* (utility hoặc thời gian hoàn thành dự đoán so với thực tế của các ảnh đã xong). Mục tiêu: xử lý được trường hợp mô hình lệch nhưng vẫn xếp hạng mode đúng (ref).
3. Mô hình feedback thực tế hơn: ACK trễ D slot, có mất; đo độ nhạy.
4. Sau đó mới tới PPO làm chính sách rollout và MPC nhanh có hard deadline.

**B:**
1. **Đối chứng "chỉ làm mượt theo thời gian"** (task + phạt ‖z_t − z_{t−1}‖² + VICReg, không EMA / predictor), cùng 3 seed. Nếu nó đạt bằng JEPA, phần có giá trị là tính ổn định chứ không phải mục tiêu dự đoán; cần biết điều này trước khi nói về JEPA.
2. **Huấn luyện predictor trên trạng thái bên nhận thật** (vòng kín, mô phỏng codec trong lúc huấn luyện), để biến lợi thế vòng hở của predictor JEPA thành lợi ích vòng kín.
3. Codec video liên khung (ví dụ AV1/HEVC có inter-prediction → DINOv2 → cùng head) trước khi nói về nén.
4. Chỉ sau đó mới quét trọng số JEPA / d nhỏ.

Code: [a9_mpc.py](../swarmbridge/experiments/uav_semcom/a9_mpc.py) (`RateErrTracker`, `CalRates(slot_weight)`), [a10_hybrid.py](../swarmbridge/experiments/uav_semcom/a10_hybrid.py) (`ehyb`, `cals`), [test_a9_mpc.py](../swarmbridge/experiments/uav_semcom/test_a9_mpc.py) (11 test), [b2_pipeline.py](../swarmbridge/experiments/jepa_semcom/b2_pipeline.py) (`--seed`, `--reps`). Kết quả: `runs/uav_semcom/a11/ae5_phy_mcs_blind.json`, `ae5_ref.json`; `runs/jepa_semcom/b4_eval.json`, `b4_diag_openloop.json`, `b4_train/`.
