# A-E7 và B-E6: lập kế hoạch chọn lọc theo chi phí tính toán; bottleneck có ý thức về số bit

Ngày 16/11/2026, theo phản biện #32. Vẫn là giai đoạn khám phá:
- A: 20 seed (8000..8019), giao thức `pre`.
- B: 10 clip đánh giá, 3 seed huấn luyện ghép cặp.

Cách gọi, theo phản biện:
- Bước kiểm tra của A-E6 là **endpoint-sensitivity check**, không phải chứng nhận thứ tự trên toàn khoảng.
- Khoảng [lo, hi] là **khoảng nhạy cảm thực nghiệm** (từ sai số chuẩn của các slot lịch sử, vốn có tương quan), không phải cận tin cậy có bảo đảm.
- Chẩn đoán xếp hạng là **reference-planner endorsement**, không phải độ đúng phản thực tế.

## Phần A (A-E7)

### 1. Chẩn đoán điểm giữa khoảng

Với mỗi lần đổi mode đã được chấp nhận (model PPO 0, toàn bộ 20 seed), lợi ích của việc đổi được tính lại bằng chính các mẫu tốc độ của quyết định đó, tại 9 hệ số đều nhau trên [lo, hi].

| Bộ điều khiển | regime | Số lần đổi | Có điểm giữa với lợi ích ≤ 0 | Có điểm giữa không vượt Z SE | Endorsement của lần đổi đã thực hiện |
|---|---|---|---|---|---|
| dhyb `cals`, 2 đầu mút | phy_mcs_blind | 755 | **12,1 %** | 18,9 % | 85 % |
| dhyb `cals`, lưới 5 điểm | phy_mcs_blind | 637 | 3,9 % | 6,9 % | 89 % |
| dhyb Shannon, 2 đầu mút | phy_mcs_blind | 770 | **21,8 %** | 27,3 % | 79 % |
| dhyb `cals`, 2 đầu mút | ref | 603 | 4,0 % | 9,6 % | – |
| dhyb `cals`, lưới 5 điểm | ref | 596 | 1,2 % | 4,0 % | – |
| dhyb Shannon, 2 đầu mút | ref | 660 | 6,5 % | 8,9 % | – |

**Có đảo thứ tự bên trong khoảng**, đúng như phản biện #32 lo ngại: 12 % với `cals` và 22 % với Shannon ở phy_mcs_blind.

Lưới 5 điểm giảm tỉ lệ này xuống khoảng 3–4 lần và nâng endorsement 85 → 89 %. Nhưng **không cải thiện return**:
- phy_mcs_blind: lưới trừ đầu mút = −0,37 [−1,03; +0,29];
- ref: +0,02.

Những lần đổi bị loại thêm có vẻ là lần đổi lợi ích nhỏ. Tóm lại, kiểm tra đầu mút là một xấp xỉ có lỗ hổng đo được, nhưng lỗ hổng đó hiện không làm mất return.

### 2. Bộ điều khiển hai tầng (`shyb`)

**Đo chi phí trước khi thiết kế:**
- Thời gian một lần lập kế hoạch đầy đủ hầu như **không phụ thuộc số mẫu** (37 ms ở N = 1, 40 ms ở N = 8, 43 ms ở N = 32), vì chi phí là vòng lặp Python qua mode × rollout.
- Nó tăng theo **tầm nhìn** (H = 0 / 1 / 2 / 4: 1 / 10 / 24 / 47 ms) và **số mode được đánh giá** (3 mode: 12 ms so với 14 mode: 47 ms).
- Bản thử đầu dùng ít mẫu để sàng lọc nên không nhanh hơn. Vì vậy sàng lọc phải rút ngắn tầm nhìn, không phải giảm mẫu.

**Cơ chế:**
1. PPO cho mode mặc định.
2. Sàng lọc rẻ: mọi mode với tầm nhìn H₁ = 1, khoảng 10 ms.
3. Nếu không mode nào hơn mode của PPO trong sàng lọc thì giữ PPO, không lập kế hoạch đầy đủ.
4. Ngược lại MPC đầy đủ (H = 4, N = 8) chỉ đánh giá mode của PPO và C = 2 mode đứng đầu sàng lọc, rồi áp dụng **cùng** bước kiểm tra theo sai số mô hình của A-E6.

Thời gian đo với 6 tiến trình trên 6 nhân (không oversubscribe như các lần trước); PPO đo trong cùng lượt: 1,3–1,5 ms (p95 2,1–2,3 ms).

| phy_mcs_blind | Δ so với PPO, gộp 3 model [CI] | ms TB | p95 | p99 | > 50 ms | Tỉ lệ lập kế hoạch đầy đủ |
|---|---|---|---|---|---|---|
| dhyb `cals` (A-E6, tham chiếu) | +2,82 [+1,89; +3,74] | 50,2 | 72 | 84 | 49 % | 100 % |
| dhyb `cals`, lưới 5 | +2,45 [+1,72; +3,18] | 55,0 | 90 | 105 | 54 % | 100 % |
| **shyb `cals`** (H₁ = 1, C = 2) | **+2,39 [+1,29; +3,48]** | **24,8** | **44** | **54** | **2 %** | 76 % |
| shyb `cals`, lưới 5 | +2,42 [+1,70; +3,14] | 28,6 | 63 | 77 | 13 % | 75 % |
| shyb `cals`, H₁ = 2 (model 0) | +1,89 [+0,04; +3,73] | 34,0 | 56 | – | – | 67 % |
| dhyb Shannon (model 0) | +1,54 [+0,30; +2,79] | 45,9 | 70 | – | – | 100 % |
| shyb Shannon | +1,03 [−0,30; +2,36] | 29,3 | 46 | 53 | 2 % | 95 % |

| ref | Δ so với PPO, gộp [CI] | ms TB | p95 | p99 | Tỉ lệ lập kế hoạch đầy đủ |
|---|---|---|---|---|---|
| dhyb `cals` (tham chiếu) | +2,23 [+1,82; +2,64] | 48,2 | 71 | 80 | 100 % |
| dhyb `cals`, lưới 5 | +2,25 [+1,86; +2,64] | 51,8 | 85 | 99 | 100 % |
| **shyb `cals`** | **+2,22 [+1,67; +2,77]** | **20,5** | **40** | **51** | 47 % |
| shyb `cals`, lưới 5 | +2,11 [+1,64; +2,58] | 23,5 | 59 | 74 | 46 % |
| shyb Shannon | +2,18 [+1,72; +2,64] | 21,9 | 42 | 50 | 52 % |
| dhyb Shannon (model 0) | +2,89 [+2,17; +3,61] | 41,4 | 65 | – | 100 % |

shyb `cals` trừ dhyb `cals`, theo cặp, gộp: phy_mcs_blind −0,43 [−1,22; +0,36]; ref −0,01 [−0,39; +0,36].

Chẩn đoán của shyb `cals` (phy_mcs_blind, model 0):
- endorsement 88 % so với 85 % của dhyb;
- điểm giữa có lợi ích ≤ 0: 8,3 %.

### Đọc kết quả A

1. **Bộ điều khiển hai tầng giữ phần lớn lợi ích với khoảng một nửa chi phí:**
   - phy_mcs_blind: 85 % lợi ích (+2,39 so với +2,82), thời gian TB 50 → 25 ms, p95 72 → 44 ms, quyết định > 50 ms 49 % → 2 %;
   - ref: giữ nguyên lợi ích (+2,22 so với +2,23), thời gian 48 → 20,5 ms.
   - Phần thua ở phy_mcs_blind có CI chứa 0.
2. **Chi phí lập kế hoạch tiết kiệm được đến từ cấu trúc của planner** (tầm nhìn, số mode), không phải số mẫu. Đây là thông tin thiết kế quan trọng cho bước hard deadline.
3. **Với mô hình lệch, bản hai tầng yếu đi ở phy_mcs_blind** (Shannon: +1,03, CI chứa 0, so với +1,54 của dhyb model 0).
   - Sàng lọc dùng cùng mô hình lệch nên gần như luôn đề xuất đổi (95 % quyết định), vì vậy tiết kiệm ít.
   - Hai mode được đưa vào đánh giá đầy đủ cũng là lựa chọn của mô hình lệch.
   - Ở ref (lệch 25 %), bản hai tầng vẫn tốt (+2,18).
4. **Đảo thứ tự bên trong khoảng có thật nhưng hiện chưa đáng giá để sửa:** lưới 5 điểm tốn thêm khoảng 4–5 ms TB và p95 cao hơn, không cải thiện return. Nên giữ chẩn đoán và báo cáo tỉ lệ, gọi đúng tên là kiểm tra đầu mút.
5. **Soft budget:** chưa cưỡng chế deadline. p99 của shyb khoảng 50–54 ms; nếu đặt deadline cứng 50 ms, khoảng 1–2 % quyết định sẽ vượt.

## Phần B (B-E6)

### Thiết lập

**Surrogate số bit** (`rate_surrogate`):
- R̂ = trung bình log₂(1 + |z_t − z_{t−1}| / (σ·0,25)).
- σ là độ lệch chuẩn theo chiều của mã trong batch (detach); codec chuẩn hoá mã theo chiều rồi lượng tử với bước 0,25.
- Một mã exp-Golomb có dấu bậc 0 của q dài 2⌊log₂(2|q|+1)⌋+1 bit, nên R̂ là xấp xỉ khả vi (sai khác hằng số) của độ dài mã phần dư.

**Các biểu diễn** (cùng codec 345k tham số, cùng tái tạo + nhiệm vụ + VICReg, cùng 3 seed ghép cặp):

| Biểu diễn | Thành phần thêm |
|---|---|
| `task` | – |
| `smooth` | MSE thời gian, w = 1 |
| `rateb01` | smooth + 0,1·R̂ |
| `rateb03` | smooth + 0,3·R̂ |
| **`rateonly03`** | **0,3·R̂, không MSE thời gian** |

Bản có R̂ không có predictor gắn sau; huấn luyện khoảng 41 s mỗi seed.

**Đánh giá:**
- Qua bitstream thật, vòng kín, mc, gồm mọi bit: khung đầu, chỉ số, motion, phần dư.
- So ở **cùng số bit thực tế**: nội suy theo log-bit trên k = 16..256, từng seed × clip (`b6_matched_bits.json`).

**Chỉ số sự kiện** (từ nhãn patch, bước 3 khung):
- `ap_change`: chỉ trên các patch có nhãn khác khung trước (vật xuất hiện, biến mất hay di chuyển); khoảng 2600 patch mỗi clip.
- `ap_event`: trên các khung thuộc 25 % có nhiều patch đổi nhãn nhất; khoảng 24 khung mỗi clip.

### Surrogate có theo được bit thật không?

Trên mã thật của 15 model (vòng hở, mọi token, dự đoán hold), so số bit exp-Golomb thật mỗi hệ số với R̂ và MSE:

| | task | smooth | rateb01 | rateb03 | rateonly03 |
|---|---|---|---|---|---|
| Bit EG thật / hệ số | 2,56–2,59 | 1,93–1,94 | 1,78–1,79 | 1,54 | 1,77–1,80 |
| R̂ | 0,80–0,82 | 0,56 | 0,50–0,51 | 0,41 | 0,50–0,51 |
| MSE hold | 0,118–0,126 | 0,040 | 0,031 | 0,019 | 0,038–0,039 |

- Tương quan R̂ với bit thật: 0,99999. Tương quan MSE với bit thật: 0,98.
- `rateonly03` có **MSE bằng smooth nhưng tốn ít bit như rateb01**. Đây là ví dụ cụ thể cho việc MSE thấp không đồng nghĩa với ít bit.

Số bit thực tế ở cùng k (mc, kbit / khung, TB):

| k | task | smooth | rateb01 | rateb03 | rateonly03 |
|---|---|---|---|---|---|
| 64 | 10,5 | 9,0 | 8,7 | 8,0 | 8,8 |
| 128 | 18,1 | 15,1 | 14,4 | 12,9 | 14,5 |
| 256 | 31,2 | 25,1 | 23,7 | 21,0 | 23,8 |

Trần (không nén): 0,486–0,489 cho mọi biểu diễn.

### Kết quả ở cùng số bit (mc, TB 3 seed)

**mAP tổng:**

| | 6 kbit | 9 kbit | 12 kbit | 18 kbit | 25 kbit |
|---|---|---|---|---|---|
| task | 0,285 | 0,334 | 0,366 | 0,413 | 0,440 |
| smooth | 0,284 | 0,343 | 0,382 | 0,429 | 0,457 |
| rateb01 | 0,279 | 0,345 | 0,386 | 0,431 | 0,458 |
| rateb03 | 0,286 | 0,353 | 0,395 | 0,438 | 0,458 |
| **rateonly03** | **0,306** | **0,366** | **0,402** | **0,442** | **0,465** |

**mAP trên patch thay đổi** (`ap_change`):

| | 6 kbit | 9 kbit | 12 kbit | 18 kbit | 25 kbit |
|---|---|---|---|---|---|
| task | 0,284 | 0,309 | 0,330 | 0,367 | 0,402 |
| smooth | 0,285 | 0,311 | 0,334 | 0,375 | 0,414 |
| rateb03 | 0,278 | 0,307 | 0,334 | 0,381 | 0,406 |
| **rateonly03** | **0,296** | **0,327** | **0,353** | **0,396** | **0,431** |

**Hiệu theo cặp** (TB [CI t, n = 3]; giá trị từng seed trong ngoặc tròn):

| | rateonly03 − task | rateonly03 − smooth | rateb03 − smooth |
|---|---|---|---|
| mAP @ 9k | **+0,032 [+0,016; +0,047]** (0,038 / 0,026 / 0,031) | **+0,023 [+0,021; +0,025]** | +0,010 [+0,003; +0,017] |
| mAP @ 12k | **+0,036 [+0,024; +0,049]**, 27/30 cặp | **+0,020 [+0,007; +0,033]** | +0,013 [−0,001; +0,027] |
| mAP @ 18k | **+0,030 [+0,011; +0,048]** | +0,013 [−0,010; +0,036] | +0,009 [−0,009; +0,028] |
| mAP @ 25k | +0,025 [+0,005; +0,045] | +0,008 [−0,016; +0,032] | +0,001 |
| ap_change @ 12k | **+0,023 [+0,012; +0,034]** | **+0,019 [+0,004; +0,033]** | 0,000 |
| ap_change @ 18k | **+0,029 [+0,020; +0,039]** | **+0,021 [+0,005; +0,037]** | +0,007 |
| ap_change @ 25k | +0,029 [+0,019; +0,040] | +0,017 [+0,001; +0,033] | −0,008 |

- smooth − task trên patch thay đổi: +0,001 đến +0,012, không âm.
- `ap_event` cho cùng mẫu hình (rateonly03 − smooth +0,016 đến +0,023 ở 6–18k).

### Đọc kết quả B

1. **Đưa trực tiếp chi phí bitstream vào hàm mục tiêu tốt hơn làm mượt bằng MSE.**
   - `rateonly03` hơn task-32 **+0,030 đến +0,036 mAP ở 9–18 kbit**, dương ở cả 3 seed, CI > 0.
   - Nó hơn smooth +0,020 đến +0,023 ở 6–12 kbit. Ở 18–25 kbit phần hơn nhỏ lại và CI chứa 0.
2. **Nó cũng tốt hơn trên đúng các patch có thay đổi** (vật mới xuất hiện / biến mất / di chuyển): +0,019 đến +0,021 so với smooth, +0,023 đến +0,029 so với task, CI > 0.
   - Cách giải thích hợp lý: hàm log phạt nhẹ thay đổi lớn và phạt mạnh hơn (tương đối) những dao động nhỏ. Vì vậy mã đứng yên ở nơi không có gì xảy ra, và vẫn đổi mạnh khi có vật mới.
   - MSE (bình phương) thì ngược lại: phạt thay đổi lớn nặng nhất.
   - Phù hợp với điều đó: thêm R̂ vào smooth (rateb03) giảm bit nhiều nhất ở cùng k (−15 % so với smooth), nhưng **không** cải thiện trên patch thay đổi.
3. **Làm mượt không làm hại thông tin mới so với task** (smooth − task trên patch thay đổi ≥ 0). Nhưng nó cũng không giúp nhiều như surrogate số bit.
4. Surrogate log theo bit thật gần như tuyến tính trên mọi model đã thử. Đây là bằng chứng nó đo đúng thứ ta muốn giảm, chứ không chỉ "L1 nhỏ thì chắc ít bit".

## Giới hạn chung

**A:**
- 20 seed khám phá.
- Kiểm tra đầu mút không phải chứng nhận; khoảng là khoảng nhạy cảm thực nghiệm (slot có tương quan).
- Endorsement dùng planner tham chiếu, không phải kết cục phản thực tế.
- Soft budget. Thời gian lần này đo với 6 tiến trình trên 6 nhân, nên không so trực tiếp với các con số oversubscribed của A-E5/A-E6.
- Quan sát throughput đọc từ simulator.

**B:**
- 3 seed, 10 clip.
- Chỉ một giá trị β cho rate-only (0,3), chưa quét.
- Chỉ số sự kiện dựa trên nhãn patch ở bước 3 khung (thay đổi nhãn gồm cả di chuyển, không chỉ vật mới).
- So ở cùng bit dựa trên nội suy 5 điểm k.
- **Chưa có codec video liên khung (AV1/HEVC)**; mọi nhận định về nén vẫn chỉ so với AVIF nội khung và các biến thể của chính codec này.

## Cơ hội phát triển phương pháp tiếp theo

**A:**
1. Hard deadline cho shyb: p99 khoảng 50–54 ms đã gần mức có thể cưỡng chế. Thử deadline 50 ms (nếu hết giờ thì trả mode PPO) và báo cáo số vi phạm.
2. Sàng lọc cho mô hình lệch: sàng lọc trên khoảng tốc độ thấp (đầu lo của khoảng), để mô hình lệch không đề xuất đổi ở 95 % quyết định.
3. ACK trễ / mất.

**B:**
1. **Codec video liên khung** (AV1 hoặc HEVC có inter-prediction → DINOv2 → cùng head) làm đối thủ bên ngoài. Đây là bước cần nhất trước khi nói về nén.
2. Quét nhỏ β cho rate-only (0,1 / 0,3 / 1) ở cùng bit, và thử surrogate trên phần dư sau mc thay vì hold.
3. Tích hợp làm một mode truyền trong môi trường UAV của Paper A.

Code:
- [a9_mpc.py](../swarmbridge/experiments/uav_semcom/a9_mpc.py) (`screen`, `cands`, `gain_at`)
- [a10_hybrid.py](../swarmbridge/experiments/uav_semcom/a10_hybrid.py) (`shyb`, lưới `Kg5`, `--interior-diag`)
- [test_a9_mpc.py](../swarmbridge/experiments/uav_semcom/test_a9_mpc.py) (14 test)
- [b2_pipeline.py](../swarmbridge/experiments/jepa_semcom/b2_pipeline.py) (`rate`, `rate_surrogate`, `change_masks`, `--name`, `--preds`, `--no-posthoc`)
- [test_b2.py](../swarmbridge/experiments/jepa_semcom/test_b2.py) (5 test)

Kết quả:
- `runs/uav_semcom/a13/ae7_{phy_mcs_blind,ref}.json`, `ae7_diag_*.json`
- `runs/jepa_semcom/b6_eval.json`, `b6_matched_bits.json`, `b6_diag_openloop.json`, `b6_train/`
