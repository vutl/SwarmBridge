# A-E6 và B-E5: cổng theo độ tin cậy của lợi ích quyết định; temporal smoothing so với JEPA

Ngày 16/11/2026, theo phản biện #31. Vẫn là giai đoạn khám phá:
- A: 20 seed (8000..8019).
- B: 10 clip đánh giá, 3 seed huấn luyện ghép cặp.

## Phần A (A-E6): cổng dựa trên độ tin cậy của lợi ích quyết định

### Cơ chế

**Thống kê sai số theo slot** (`RateErrTracker.stats`):
- Ngoài bias b = Σ quan sát / Σ dự đoán − 1, bộ theo dõi còn tính sai số chuẩn se của tỉ số quan sát / dự đoán trên từng slot.
- Dự đoán vẫn là prequential như A-E5: có trước khi slot được quan sát.

**Bản kết hợp quyết-định-nhận-thức** (`dhyb`):
1. PPO đề xuất mode. MPC (N = 8, H = 4, rollout phi tiên tri) đánh giá các mode; như cũ, mode tốt nhất phải hơn mode của PPO quá Z = 1 sai số chuẩn lấy mẫu.
2. **Thêm bước kiểm tra:** cùng các đường tốc độ mẫu được nhân với hai đầu của khoảng [min(1, 1+b) − K·se ; max(1, 1+b) + K·se]. Việc đổi mode chỉ được chấp nhận nếu lợi ích vẫn vượt Z sai số chuẩn ở **cả hai** đầu.
3. Như vậy sai số tốc độ đã thấy trong lịch sử được truyền vào hiệu giá trị giữa mode của PPO và mode của planner.
   - Chỉ đánh giá lại 2 mode, nên chi phí thêm nhỏ.
   - K = 2 chọn trước; K = 1 và 3 chỉ là độ nhạy.
4. Nếu chưa có ≥ 20 s dữ liệu ở khoảng SNR hiện tại thì dùng mode của PPO.

**Giao thức `pre`**, theo lưu ý về các episode test không độc lập:
- Bộ hiệu chỉnh và bộ theo dõi được khởi động bằng **10 episode hiệu chỉnh riêng**: heuristic điều khiển, ảnh từ pool val, seed 9000..9009.
- Sau đó trạng thái được đóng băng làm điểm xuất phát cho **mọi** episode test. Trong một episode vẫn cập nhật trực tuyến, nhưng episode test sau không học từ episode test trước.
- Có test kiểm tra.

**Chẩn đoán xếp hạng** (phy_mcs_blind, nơi mô hình `env` là PHY thật, chỉ khác ở fading):
- Với mỗi lần đổi mode được thực hiện, và mỗi lần đổi bị bước kiểm tra từ chối, một planner PHY đúng (N = 32) độc lập đánh giá lợi ích của việc đổi đó.
- Chỉ số: tỉ lệ được planner này "tán thành" (lợi ích trung bình > 0). Phần này nằm ngoài thời gian quyết định.

**Kiểm tra code:**
- Tách hàm đánh giá mode (`_ctx`, `_eval_mode`) xong, return từng seed của A-E4 / A-E5 được tái tạo chính xác.
- 13/13 test của a9 đạt.

### Kết quả: phy_mcs_blind

Δ so với đúng model PPO của nó; gộp 3 model PPO; CI 95 % theo cặp seed. PPO: 30,03 / 30,27 / 30,51 (3,4–3,7 ms). Mọi bản có `cals` / tracker dùng giao thức `pre`.

| Bộ điều khiển | Return | Δ so với PPO, gộp [CI] | Số lần đổi (TB) | ms TB (p95) |
|---|---|---|---|---|
| Kết hợp `cals`, không cổng, Z = 1 | 31,13 / 31,68 / 31,59 | +1,20 [−0,21; +2,60] | 1516 | 103 (145) |
| Kết hợp `cals` + cổng sai số τ = 0,1 (A-E5) | giống hệt hàng trên | +1,20 (với `pre`, cổng không bao giờ đóng) | 1516 | 103 |
| Kết hợp `cals`, không cổng, Z = 2 | 32,56 / 32,04 / 32,40 | +2,07 [+0,75; +3,38] | 1162 | 68 |
| Kết hợp `cals`, không cổng, Z = 3 | 32,29 / 31,57 / 33,24 | +2,10 [+1,11; +3,09] | 900 | 68 |
| **Quyết-định-nhận-thức, `cals`, K = 2** | **32,93 / 33,15 / 33,18** | **+2,82 [+1,89; +3,74]** | 759 | 102 (151) |
| Quyết-định-nhận-thức, PHY đúng | 33,40 / 32,51 / 32,76 | +2,62 [+1,57; +3,66] | 790 | 103 |
| **Quyết-định-nhận-thức, Shannon (bias −47 %)** | 31,58 / 30,84 / 32,30 | **+1,30 [+0,13; +2,47]** | 802 | 105 |
| Kết hợp Shannon, không cổng, Z = 1 / Z = 3 (model 0) | 16,39 / 21,36 | −13,64 / −8,67 | 2382 / 1967 | 65 |
| Kết hợp Shannon + cổng sai số | = PPO | 0 | 0 | 3,5 |
| Kết hợp PHY đúng, không cổng, Z = 1 / 2 / 3 (model 0) | 31,91 / 32,33 / 32,70 | +1,87 / +2,30 / +2,67 | 1586 / 1191 / 928 | 75 |

So sánh theo cặp, gộp 3 model:
- quyết-định-nhận-thức `cals` trừ không cổng Z = 1: **+1,62 [+0,62; +2,62]**;
- trừ không cổng Z = 2: +0,75 [−0,08; +1,58];
- trừ không cổng Z = 3: +0,72 [−0,09; +1,53].

Độ nhạy K (model 0):

| | K = 1 | K = 2 | K = 3 |
|---|---|---|---|
| `cals` | +2,32 | +2,90 | +2,76 |
| Shannon | +1,10 | +1,54 | +1,73 |

Chẩn đoán xếp hạng (model 0): tỉ lệ lần đổi được planner PHY đúng tán thành.

| Bộ điều khiển | Lần đổi đã thực hiện | Lần đổi bị bước kiểm tra từ chối |
|---|---|---|
| Không cổng, Shannon | 38 % (2382 lần) | – |
| Không cổng, PHY đúng | 70 % (1586) | – |
| Không cổng, `cals` | 71 % (1517) | – |
| **Quyết-định-nhận-thức, Shannon** | **79 %** (770) | **29 %** (1715) |
| Quyết-định-nhận-thức, `cals` | 85 % (755) | 56 % (772) |
| Quyết-định-nhận-thức, PHY đúng | 88 % (766) | 55 % (788) |

### Kết quả: ref

PHY là Shannon có tái dùng kênh, nên `env` = `shannon`, và planner không mô hình nhiễu đồng kênh (bias −25 %). PPO: 58,16 / 58,32 / 58,49.

| Bộ điều khiển | Δ so với PPO, gộp [CI] | Số lần đổi | ms TB |
|---|---|---|---|
| Kết hợp `cals`, không cổng, Z = 1 | +1,79 [+0,92; +2,66] | 1005 | 104 |
| Kết hợp `cals` + cổng sai số τ = 0,1 | +1,50 [+0,73; +2,26] | 984 | 103 |
| Kết hợp `cals`, không cổng, Z = 2 / Z = 3 (model 0) | +2,08 / +2,39 | 738 / 547 | 39 |
| **Quyết-định-nhận-thức, `cals`** | **+2,23 [+1,82; +2,64]** | 579 | 97 |
| **Quyết-định-nhận-thức, Shannon/PHY (bias −25 %)** | **+2,42 [+1,79; +3,05]** | 607 | 98 |
| Kết hợp Shannon/PHY + cổng sai số | 0 (cổng luôn đóng) | 0 | 3,2–3,7 |
| Kết hợp Shannon/PHY, không cổng, Z = 1 / Z = 3 (model 0) | +3,12 / +2,84 | 1352 / 1023 | 79 |

Quyết-định-nhận-thức `cals` trừ không cổng: +0,44 [−0,23; +1,11]; trừ cổng sai số: +0,73 [+0,17; +1,30].

### Đọc kết quả A

1. **Bộ điều khiển tốt nhất ở cả hai regime** là PPO + planner hiệu chỉnh theo slot + cổng quyết-định-nhận-thức:
   - +2,82 (phy_mcs_blind) và +2,23 (ref) so với PPO, CI gộp > 0;
   - ở phy_mcs_blind cả 3 model PPO đều +2,67 đến +2,90;
   - không cần biết PHY; giao thức `pre` cho các episode test độc lập.
2. **Phần chỉ riêng việc truyền sai số mô hình mới làm được là chống bias.**
   - Với mô hình không lệch (`cals`, PHY đúng), chỉ cần tăng ngưỡng lấy mẫu Z lên 2–3 đã có phần lớn lợi ích (+2,1 so với +2,8; hiệu số CI chứa 0). Phần lớn lợi ích so với bản Z = 1 là do **bảo thủ hơn**.
   - Với mô hình lệch (Shannon trong MCS), tăng Z không cứu được (−8,67 ở Z = 3), cổng sai số thì tắt hẳn planner (= PPO). Chỉ cổng quyết-định-nhận-thức biến mô hình lệch 47 % thành **+1,30** (CI > 0).
   - Ở ref, mô hình lệch 25 % cho +2,42, trong khi cổng sai số chặn hoàn toàn.
   - Đây là điều phản biện #31 dự đoán: mô hình lệch tốc độ vẫn có thể xếp hạng mode đúng, và cổng nên hỏi về thứ tự, không hỏi về sai số trung bình.
3. **Bằng chứng cơ chế:** bước kiểm tra nâng tỉ lệ lần đổi đúng (Shannon 38 % → 79 %; `cals` 71 % → 85 %). Lần đổi bị từ chối có tỉ lệ đúng thấp (Shannon 29 %).
   - Nhưng với `cals` / PHY đúng, khoảng một nửa lần đổi bị từ chối thực ra tốt (55–56 %): cổng vẫn bỏ lỡ nhiều cơ hội.
4. **Cổng sai số của A-E5 dư thừa** khi có warm-start riêng: bias nhỏ và luôn đã biết, nên cổng không bao giờ đóng. Kết quả +1,69 của A-E5 gắn với giao thức `warm`. Độ nhạy K gần đơn điệu, khác với τ.
5. **Chi phí chưa giảm:**
   - Cổng quyết-định-nhận-thức vẫn chạy MPC ở 100 % quyết định có dữ liệu; khoảng 100 ms TB, p95 khoảng 150 ms, khoảng 50–60 % quyết định > 100 ms (12 tiến trình trên 6 nhân, soft budget).
   - Bước kiểm tra thêm chỉ 2 mode × 2 kịch bản, nên chi phí ngang bản không cổng.
   - Bản Z = 3 đo được 68 ms chỉ vì chạy trong một lượt có ít tiến trình hơn; không so trực tiếp được.

## Phần B (B-E5): temporal smoothing so với JEPA

### Thiết lập

`smooth` dùng cùng encoder/decoder (345k tham số), cùng loss tái tạo + nhiệm vụ như task-32, và:
- **phạt thay đổi theo thời gian** w·MSE(f(x_t), f(x_{t−1})), w = 1 (bằng trọng số của số hạng JEPA);
- **cùng VICReg** (phương sai / hiệp phương sai) như JEPA;
- **không EMA target, không predictor**.

Cùng 3 seed ghép cặp: cùng khởi tạo, cùng chuỗi minibatch với task / jepa của B-E4. Cùng 10 clip, cùng codec vòng kín, kết quả từng clip giữ trong `b5_eval_smooth.json`.

Mã `smooth` ổn định hơn rất nhiều, nên ở cùng k nó tốn **ít bit hơn** (k = 128: 15,1 so với khoảng 18 kbit). Vì vậy so sánh chính dùng **cùng ngân sách bit**: mAP được nội suy theo log-bit trên đường k = 16..256, từng seed × từng clip (`b5_matched_bits.json`).

### Kết quả

| mc, mAP ở cùng bit | 8 kbit | 12 kbit | 15 kbit | 18 kbit | 25 kbit |
|---|---|---|---|---|---|
| task-32 | 0,321 | 0,366 | 0,392 | 0,413 | 0,440 |
| **smooth-32** | 0,326 | **0,382** | **0,411** | **0,429** | **0,457** |
| JEPA-32 | 0,329 | 0,379 | 0,405 | 0,426 | 0,451 |

Hiệu theo cặp (TB 3 seed, CI t với n = 3):

| mc | smooth − task | JEPA − task | JEPA − smooth |
|---|---|---|---|
| 12 kbit | **+0,016 [+0,004; +0,029]**, 3/3 seed > 0, 23/30 cặp | +0,014 [−0,028; +0,055] | −0,003 [−0,049; +0,043] |
| 18 kbit | +0,016 [−0,005; +0,038], 3/3 seed > 0, 22/30 | +0,013 [−0,014; +0,040] | −0,003 [−0,043; +0,037], 13/30 |
| 25 kbit | **+0,017 [+0,005; +0,029]**, 3/3 seed > 0, 25/30 | +0,011 [−0,015; +0,036] | −0,006 [−0,040; +0,027] |

Hold cho cùng mẫu hình (smooth − task +0,015 đến +0,017; JEPA − smooth −0,004 đến −0,008).

Ở cùng k (cách so của B-E4): smooth ≈ task về mAP (mc k = 128: 0,417 so với 0,418), nhưng tốn ít hơn khoảng 16 % bit.

Chẩn đoán vòng hở (MSE một bước, không gian mã chuẩn hoá):

| | task | JEPA | smooth |
|---|---|---|---|
| hold | 0,118–0,125 | 0,105–0,108 | **0,040** |
| mc | 0,112–0,120 | 0,098–0,101 | 0,038–0,039 |

Chi phí huấn luyện mỗi seed (4000 vòng, GPU):

| task | smooth | JEPA |
|---|---|---|
| 28–29 s | 40–61 s | 73–74 s (cộng predictor 642k tham số) |

### Đọc kết quả B

1. **Ưu thế nhỏ của JEPA được giải thích hoàn toàn bởi tính ổn định theo thời gian.**
   - Ở cùng ngân sách bit, `smooth` ít nhất bằng JEPA (JEPA − smooth ≈ −0,003 đến −0,006, CI chứa 0).
   - `smooth` hơn task-32 một cách **đều qua cả 3 seed** (+0,016 đến +0,019 từ 12 kbit trở lên), trong khi JEPA − task dao động mạnh theo seed (+0,025 / +0,012 / +0,003 ở 18 kbit).
2. Mục tiêu dự đoán của JEPA (EMA target + predictor) **không cho thêm lợi ích đo được** trong cấu hình này, nhưng tốn gấp đôi thời gian huấn luyện và thêm một predictor.
3. Đây là kết quả có nghĩa cho nhánh B, theo đúng cách phản biện #31 đề xuất. Trong SemCom dự đoán có bộ codec dư theo thời gian, **tính ổn định theo thời gian của biểu diễn giữ nhiệm vụ là yếu tố quyết định**, không phải dự đoán phức tạp.
   - Hướng đóng góp tự nhiên: học biểu diễn giữ nhiệm vụ **có ý thức về tốc độ bit** (rate-aware), với chi phí cập nhật theo thời gian là một phần của mục tiêu học.
4. Predictor gắn sau trên mã `smooth` không ổn định (k = 128: 0,310 / 0,408 / 0,319 theo seed). Mã đã gần như đứng yên, nên học phần dư còn lại khó; "giữ nguyên" + mc đã đủ.

## Giới hạn chung

**A:**
- 20 seed khám phá; τ / K chọn trước, độ nhạy K nhỏ.
- Chẩn đoán xếp hạng chỉ ở phy_mcs_blind, chỉ model 0, dùng planner PHY đúng (N = 32) làm tham chiếu chứ không phải kết cục thật.
- Quan sát throughput đọc từ simulator, chưa có ACK/feedback thực.
- Soft budget, thời gian đo khi oversubscribed và khác nhau giữa các lượt chạy.
- Hiệu chỉnh theo slot mới được kiểm tra trong hai regime này.

**B:**
- 3 seed (CI rộng), 10 clip.
- Chỉ d = 32 và w_smooth = 1.
- So sánh ở cùng bit dựa trên nội suy giữa 5 điểm k.
- Chưa có codec video liên khung.

## Cơ hội phát triển phương pháp tiếp theo

**A:**
1. **Giảm tính toán:** cổng quyết-định-nhận-thức hiện chạy MPC ở mọi quyết định. Bước tự nhiên là đánh giá rẻ trước (ví dụ chỉ mode PPO và 1–2 mode lân cận, hoặc N nhỏ), và chỉ mở rộng khi khoảng giá trị cho thấy có thể đổi thứ tự.
2. **Bớt bảo thủ:** khoảng một nửa lần đổi bị từ chối thực ra tốt. Có thể dùng khoảng bất định riêng theo mode (mode nặng nhạy với tốc độ hơn) thay vì nhân cùng một hệ số.
3. ACK trễ / mất, sau đó MPC có hard deadline.

**B:**
1. **Bottleneck có ý thức về tốc độ bit:** đưa số bit cập nhật ước lượng (ví dụ L1 hoặc entropy của phần dư theo thời gian) vào loss, quét w_smooth nhỏ (0,3 / 1 / 3) ở cùng ngân sách bit.
2. Predictor huấn luyện vòng kín trên trạng thái bên nhận thật, áp dụng cho cả `smooth` lẫn JEPA.
3. Codec video liên khung trước khi nói về nén.

Code:
- [a9_mpc.py](../swarmbridge/experiments/uav_semcom/a9_mpc.py) (`_ctx` / `_eval_mode` / `gain`, `scales`, `RateErrTracker.stats`, `freeze`)
- [a10_hybrid.py](../swarmbridge/experiments/uav_semcom/a10_hybrid.py) (`dhyb`, `pre`, `--rank-diag`)
- [test_a9_mpc.py](../swarmbridge/experiments/uav_semcom/test_a9_mpc.py) (13 test)
- [b2_pipeline.py](../swarmbridge/experiments/jepa_semcom/b2_pipeline.py) (`smooth`)

Kết quả:
- `runs/uav_semcom/a12/ae6_{phy_mcs_blind,ref}.json`, `ae6_zcontrol_*.json`, `ae6_rankdiag_phy_mcs_blind.json`
- `runs/jepa_semcom/b5_eval_smooth.json`, `b5_diag_openloop_smooth.json`, `b5_matched_bits.json`, `b5_train/`
