# A-E4 và B-E3: planner hiệu chỉnh trực tuyến lấy PPO làm phương án hiện hành; bottleneck giữ nhiệm vụ và JEPA so với non-JEPA

Ngày 16/11/2026, theo phản biện #29. Giai đoạn khám phá: 20 seed (8000..8019, đã dùng ở A5.5) cho A; 10 clip đánh giá cho B; mỗi biểu diễn của B chỉ huấn luyện một seed.

## Phần A (A-E4)

### Thay đổi so với A-E2 (theo phản biện #29)

- **Rollout phi tiên tri** (`rollout=causal`): mỗi khung tương lai được gán **một** mode chung cho mọi đường fading mẫu, chọn theo utility trung bình.
  - Bản A-E2 (`scenario`) chọn mode tương lai riêng cho từng đường mẫu, tức là lạc quan. Bản này được giữ lại để so.
- **Mô hình tốc độ hiệu chỉnh từ ACK** (`CalRates`, nhân quả):
  - Ở mỗi ranh giới slot, tốc độ thực của các đoạn truyền đã xong được gom theo khoảng 2 dB của SNR kỳ vọng. Đây là thông tin bên phát biết qua ACK.
  - Planner lấy mẫu tốc độ từ phân phối thực nghiệm của khoảng đó khi đã có ≥ 20 s thời gian phát; trước đó dùng Shannon nhân ρ (tỉ lệ quan sát / Shannon).
  - `cold`: xoá mỗi episode. `warm`: giữ dữ liệu của các episode trước trong cùng lần chạy (vẫn chỉ dùng quá khứ).
  - Có test kiểm chỉ đọc các đoạn đã ghi.
- **Bản kết hợp lấy PPO làm phương án hiện hành:**
  - PPO đề xuất mode (tính trong thời gian quyết định).
  - Planner chỉ đổi khi mức lợi trung bình theo cặp mẫu vượt Z sai số chuẩn (Z = 1).
  - Với mô hình `cal`: nếu chưa có dữ liệu cho khoảng SNR thì không lập kế hoạch, dùng luôn mode của PPO (cổng tin cậy mô hình).
- **Thời gian:** ngân sách vẫn là soft, chưa có hard deadline; số đo là thời gian thực tế, 12 tiến trình song song.
- Code: [a9_mpc.py](../../../swarmbridge/experiments/uav_semcom/a9_mpc.py), [a10_hybrid.py](../../../swarmbridge/experiments/uav_semcom/a10_hybrid.py), [test_a9_mpc.py](../../../swarmbridge/experiments/uav_semcom/test_a9_mpc.py) (8 test). Kết quả: `runs/uav_semcom/a10/ae4_phy_mcs_blind.json`, `ae4_ref.json`.

### Kết quả: phy_mcs_blind

Δ theo cặp seed, CI 95 %. Bản kết hợp so với đúng model PPO của nó; các bản khác so với trung bình 3 model PPO. PPO trung bình = 30,27.

| Bộ điều khiển (N = 8, H = 4 nếu không ghi khác) | Return | Δ so với PPO [CI] | ms / quyết định TB (p95) | kbit TB của mode chọn |
|---|---|---|---|---|
| heuristic T:best | 26,94 | −3,33 [−5,28; −1,37] | 0,2 | 483 |
| PPO (3 model) | 30,03 / 30,27 / 30,51 | – | 3,4 (6,6) | 429–463 |
| MPC PHY đúng, rollout **lạc quan** (A-E2) | 32,95 | +2,68 [+1,42; +3,94] | 89 | 539 |
| MPC PHY đúng, rollout **phi tiên tri** | 30,87 | +0,60 [−0,95; +2,15] | 98 | 546 |
| MPC PHY đúng, phi tiên tri, N = 32, H = 8 | 26,63 | −3,64 | 145 | 503 |
| MPC Shannon (sai), phi tiên tri | 13,98 | −16,29 | 98 | 835 |
| MPC hiệu chỉnh ACK, cold | 29,03 | −1,24 [−3,03; +0,54] | 98 | 514 |
| **MPC hiệu chỉnh ACK, warm** | **30,94** | +0,67 [−0,80; +2,15] | 99 | 494 |
| MPC PHY đúng, N = 8, H = 0 (nông) | 10,69 | −19,58 | 2,2 | 853 |
| **Kết hợp PPO + MPC PHY đúng** (3 model) | 31,91 / 32,52 / 32,36 | **+1,87 / +2,25 / +1,85** (cả 3 CI > 0); TB **+1,99 [+1,00; +2,98]** | 52–102 | 534–537 |
| Kết hợp PPO + MPC hiệu chỉnh warm (3 model) | 30,40 / 30,39 / 31,77 | +0,37 / +0,12 / +1,26; TB +0,58 [−0,73; +1,89] | 101–102 | 496–504 |
| Kết hợp PPO + MPC hiệu chỉnh warm, Z = 0 | 29,09 | −0,95 [−2,95; +1,05] | 49 | 493 |
| Kết hợp PPO + MPC Shannon (sai), Z = 1 | 16,39 | −13,64 | 48 | 806 |

### Kết quả: ref (Shannon là đúng; có dùng chung kênh; PPO = 58,33)

| Bộ điều khiển | Return | Δ so với PPO [CI] |
|---|---|---|
| MPC PHY đúng, phi tiên tri | 61,05 | +2,73 [+1,79; +3,67] |
| MPC hiệu chỉnh warm | 57,24 | −1,09 [−2,32; +0,15] |
| Kết hợp PPO + MPC PHY đúng (model 0) | 61,29 | +3,12 [+2,08; +4,16] |
| Kết hợp PPO + MPC hiệu chỉnh (3 model) | 58,24 / 57,45 / 58,62 | TB −0,22 [−1,04; +0,59] |

### Đọc kết quả

1. **Sửa diễn giải A-E2.** Phần lớn ưu thế +2,7 của "MPC mô hình đúng" ở phy_mcs_blind đến từ rollout lạc quan. Với rollout phi tiên tri, MPC chỉ ngang PPO (+0,6, CI chứa 0). Ở ref, rollout phi tiên tri không làm mất ưu thế (+2,7).
2. **Hiệu chỉnh từ ACK sửa được sai mô hình.** MPC Shannon chưa hiệu chỉnh đạt 13,98; khi hiệu chỉnh (warm) đạt 30,94, ngang PHY đúng (30,87).
   - Vì vậy kết luận "PPO bền hơn planner có mô hình" **không đứng vững**: một planner hiệu chỉnh trực tuyến cũng bền như vậy.
   - Bản cold (chỉ học trong episode) kém hơn (29,03), vì đầu mỗi episode phải dùng ρ·Shannon.
3. **Kết hợp PPO + planner bổ sung cho nhau khi mô hình đúng.** Bản kết hợp hơn PPO +2,0 (cả 3 model, CI > 0) và hơn MPC chạy riêng (30,87) khoảng +1,4 ở phy_mcs_blind; ở ref hơn PPO +3,1.
   - PPO cho phương án ban đầu tốt; planner sửa những quyết định mà nó tin chắc là tốt hơn.
4. **Với mô hình hiệu chỉnh, lợi ích của bản kết hợp nhỏ:** +0,6 ở phy_mcs_blind, CI chứa 0; khoảng 0 ở ref. Ở ref, mô hình hiệu chỉnh còn làm MPC kém đi (−1,1).
   - Có thể vì tốc độ thực ở ref lẫn cả nhiễu đồng kênh, phụ thuộc lịch dùng chung kênh chứ không chỉ SNR; khoảng 2 dB theo SNR thì gộp các trường hợp này lại.
5. **Ngưỡng Z = 1 giúp so với Z = 0** (+1,3), nhưng **không bảo vệ được khi mô hình lệch có hệ thống:** bản kết hợp với Shannon vẫn sụp (16,39). Sai số chuẩn theo mẫu chỉ đo nhiễu lấy mẫu, không đo độ lệch của mô hình.
6. **Chẩn đoán MPC nông (H = 0):** nó chọn mode nặng (trung bình 853 kbit so với khoảng 450–480 của PPO / heuristic), tỉ lệ rớt 3,5 %. Không nhìn trước thì planner không thấy chi phí tắc nghẽn mà ảnh lớn gây cho các khung sau. Đây là hạn chế của planner thiển cận, không phải của tối ưu trực tuyến nói chung.
7. **Nhìn xa hơn không giúp:** N = 32, H = 8 với rollout phi tiên tri kém hơn H = 4 (26,6 so với 30,9). Gán một mode chung cho mọi kịch bản trên tầm xa có lẽ quá thận trọng. Cần một rollout chính sách tốt hơn (ví dụ chính PPO làm chính sách rollout).

**Giới hạn:**
- Ngân sách thời gian soft; MPC Python chưa tối ưu, khoảng 50–100 ms dưới tải 12 tiến trình.
- 20 seed khám phá, một cấu hình (N, H) chính.
- Thời gian của PPO trong bản kết hợp được tính.

**Cơ chế đáng phát triển tiếp cho Paper A** (đề xuất, chưa chạy):
- **Lập kế hoạch có cổng tin cậy mô hình:** thay ngưỡng theo sai số lấy mẫu bằng **độ lệch của mô hình đo trực tuyến**. Ví dụ so tốc độ dự đoán với tốc độ thực qua ACK (lỗi dự đoán gần đây, kiểm định hiệu chỉnh), rồi quyết định có lập kế hoạch và đổi mode của PPO hay không.
- Điểm số 4–5 cho thấy đây chính là chỗ còn thiếu: bản kết hợp chỉ có lợi khi mô hình tốt, và cổng hiện tại không biết khi nào mô hình tốt.
- **Thêm:**
  - dùng PPO làm chính sách rollout cho các khung tương lai (thay greedy / một mode chung);
  - hiệu chỉnh theo trạng thái dùng chung kênh (cho ref);
  - một bản MPC nhanh để có hard deadline thật.

## Phần B (B-E3)

### Thiết kế

- [b2_pipeline.py](../../../swarmbridge/experiments/jepa_semcom/b2_pipeline.py), [test_b2.py](../../../swarmbridge/experiments/jepa_semcom/test_b2.py) (3 test). Kết quả: `runs/jepa_semcom/b2_eval.json`, `b2_diag_openloop.json`, `b2_train/`.
- **Đầu vào:** bản đồ token DINOv2-S đóng băng (384 × 22 × 39) của VisDrone-VID. Cùng split và cùng 10 clip đánh giá như B-E1; 48 clip TUNE để huấn luyện.
- **Backbone như nhau cho mọi biểu diễn:**
  - encoder: Conv1×1 384→256, GELU, Conv3×3 256→d;
  - decoder: Conv3×3 d→256, GELU, Conv1×1 256→384;
  - codec 345k tham số (d = 32) / 492k (d = 64); predictor 642k / 824k.
  - Receiver giải mã về 384 chiều rồi qua **cùng** head tuyến tính 4 lớp của B-E0 / B-E1.
- **`task` (non-JEPA):** tái tạo token + BCE nhiệm vụ của head cố định trên token giải mã.
- **`jepa`:** cùng hai mất mát trên encoder **online**, cộng thêm:
  - predictor (cùng kiến trúc với predictor của B-E1) dự đoán mã khung t từ mã online của t−1, t−2;
  - đích là mã của encoder **target EMA** (momentum 0,99, stop-gradient);
  - điều chuẩn VICReg (phương sai + hiệp phương sai) chống sụp.
  - Đây là JEPA trên token đóng băng (không học từ pixel), có encoder học được và target EMA.
- **Huấn luyện:** 4.000 bước, batch 16, khoảng 30–80 s mỗi mô hình trên RTX 3070. Có lock GPU.
- **Predictor công bằng:** mỗi biểu diễn được gắn **cùng** predictor huấn luyện sau (quy trình của B-E1: mạng conv, MSE, đầu vào giả lập trạng thái bên nhận) trên mã TUNE đóng băng của nó. Predictor riêng của JEPA (`own`) được chấm thêm.
- **Đánh giá:** codec round-trip của B-E1 (lịch sử gửi / nhận riêng, kiểm trùng từng bit), vòng kín, bit tính cả khung khởi tạo, per-clip giữ trong JSON.

### Kết quả (trung bình 10 clip; mAP với token gốc = 0,475)

| Biểu diễn | Trần (không lượng tử) | hold k16 | mc k16 | mc k64 | mc k128 | mc k256 |
|---|---|---|---|---|---|---|
| PCA-32 (B-E1) | 0,356 | 0,176 @4,1k | 0,194 | 0,297 @10,9k | 0,333 @18,6k | 0,350 @31,4k |
| task-32 | 0,475 | 0,210 @4,0k | 0,235 | 0,339 @10,4k | 0,402 @18,0k | 0,450 @31,0k |
| **jepa-32** | 0,482 | 0,228 @4,0k | **0,248** | **0,362 @10,5k** | **0,429 @17,9k** | **0,463 @30,4k** |
| task-64 | 0,487 | 0,227 @7,9k | 0,252 | 0,361 @21,0k | 0,427 @36,6k | 0,469 @64,3k |
| jepa-64 | 0,473 | 0,224 @7,9k | 0,246 | 0,364 @20,7k | 0,426 @35,7k | 0,459 @61,4k |

AVIF → DINOv2 → head (từ B-E1): q5 0,300 @18,7k; q20 0,386 @36,4k; q35 0,429 @64,2k; q50 0,458 @118k.

**Predictor học trong vòng kín (d = 32):** posthoc kém hold ở mọi k với cả hai biểu diễn (ví dụ jepa-32 k128: posthoc 0,369, own 0,410, hold 0,419, mc 0,429).

**Chẩn đoán vòng hở** (MSE một bước trong không gian mã chuẩn hoá, lịch sử thật):

| | hold | mc | posthoc | own |
|---|---|---|---|---|
| task-32 | 0,118 | 0,112 | 0,112 | – |
| jepa-32 | **0,105** | 0,099 | 0,102 | **0,090** |
| task-64 | 0,159 | 0,152 | 0,167 | – |
| jepa-64 | **0,122** | 0,115 | 0,127 | 0,108 |

**So sánh theo từng clip:**

| So sánh | Δ mAP TB | Thắng | Tệ nhất |
|---|---|---|---|
| jepa-32 so với task-32, mc, k = 64 / 128 | +0,023 / +0,027 | 8/10 | −0,04 |
| jepa-32 so với task-32, k = 256 | +0,013 | 6/10 | |
| jepa-64 so với task-64 (k = 128 / 256) | −0,001 / −0,010 | 5/10 và 2/10 | |
| jepa-32 mc k128 (17,9k) so với AVIF q5 (18,7k) | +0,129 | 10/10 | +0,068 |
| jepa-32 mc k128 (17,9k) so với AVIF q20 (36,4k, gấp đôi bit) | +0,043 | 10/10 | |
| jepa-32 mc k256 (30,4k) so với AVIF q35 (64,2k) | +0,033 | 8/10 | |
| task-32 mc k128 so với PCA-32 mc k128 | +0,069 | 7/10 | một clip +0,40, một clip −0,08 |

### Đọc kết quả

1. **Bottleneck giữ nhiệm vụ giải quyết vấn đề của PCA.**
   - Trần 0,475–0,487, bằng token gốc, so với 0,356–0,414 của PCA.
   - Ở cùng khoảng 18 kbit, mAP tăng từ 0,333 (PCA-32) lên 0,402 (task-32).
2. **Truyền mã học được giờ hơn AVIF qua DINOv2 trên toàn dải đã đo:**
   - ở cùng bit khoảng 18 kbit, +0,10 đến +0,13 mAP, 10/10 clip;
   - ở một nửa số bit của AVIF q20, vẫn hơn 10/10 clip.
   - AVIF ở đây chỉ mã từng khung độc lập; codec video liên khung là đối thủ phải thêm trước khi kết luận.
3. **JEPA so với non-JEPA, cùng backbone và cùng predictor:**
   - Ở d = 32, JEPA hơn khoảng +0,013 đến +0,027 mAP ở cùng bit, thắng 8/10 clip ở k = 64–128.
   - Ở d = 64 thì ngang hoặc kém nhẹ.
   - Đây là tín hiệu tích cực đầu tiên cho JEPA trong SwarmBridge, nhưng nhỏ, chỉ một seed huấn luyện, và chỉ ở d nhỏ.
4. **Cơ chế:** lợi ích đến từ **biểu diễn ổn định hơn theo thời gian**, không đến từ predictor.
   - Mã JEPA có MSE "giữ nguyên" thấp hơn (0,105 so với 0,118 ở d32; 0,122 so với 0,159 ở d64), nên hold / mc cần gửi ít phần dư hơn.
   - Predictor riêng của JEPA tốt nhất ở vòng hở (0,090) nhưng vẫn kém hold trong vòng kín, vì nó được huấn luyện trên lịch sử sạch.
   - Nói cách khác, mục tiêu JEPA làm không gian mã **dễ truyền bằng dự đoán đơn giản**.
5. **Ở d = 64, JEPA không có lợi:** mã đã đủ chiều để giữ nhiệm vụ, nên ràng buộc dự đoán được làm tái tạo kém đi (trần 0,473 so với 0,487) mà không bù lại được.

**Giới hạn:**
- Một seed huấn luyện cho mỗi biểu diễn; 10 clip.
- AVIF chỉ intra.
- Head cố định (train trên token gốc); decoder của các biểu diễn học được tối ưu cho head đó, còn đường AVIF thì đi qua DINOv2 gốc.
- JEPA trên token đóng băng, không phải trên pixel.
- Kênh không mất gói.

**Hướng phát triển tiếp cho Paper B** (đề xuất):
1. Lặp 3 seed huấn luyện cho task-32 / jepa-32 để xem +0,02 có ổn định không.
2. Quét trọng số JEPA và d (16 / 24 / 32) để xem lợi ích có tăng khi bottleneck chặt hơn không.
3. Huấn luyện predictor (own / posthoc) với đầu vào là trạng thái bên nhận thật, lấy từ chính codec (closed-loop training), để biến khả năng dự đoán vòng hở thành lợi ích vòng kín.
4. Baseline codec video liên khung.
5. Phân tích theo clip (chuyển động camera, mật độ vật thể) cho các clip mà JEPA thua.

## Ghi chú chung

- Mọi bước GPU (huấn luyện B-E3) đều giữ lock.
- Token, mô hình và mã nằm ở `~/work/jepa_semcom/` (ngoài git).
