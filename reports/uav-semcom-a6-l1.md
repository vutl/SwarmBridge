# SwarmBridge hướng A — A6.0-L1: chọn mode theo giá trị tác vụ (task-aware mode selection)

Bắt đầu 2026-11-09 tối. Kết quả ở `runs/uav_semcom/a6l1/`; A5 không bị sửa.

**Câu hỏi:** giữ nguyên hệ thống direct-link của A5 (topology, kênh, PHY, codec, profile, không gian hành động, quan sát, deadline, Qmin) và **chỉ thay số hạng chất lượng của reward**, từ PSNR sang giá trị phát hiện vật thể. Khi đó:
- policy (PPO và heuristic) có chọn mode khác đi không?
- thông tin tác vụ được giao (recall, AP50, vật nhỏ, false positive) có tăng ở cùng hoặc thấp hơn chi phí truyền, deadline, năng lượng không?

Phạm vi: đây là **điều khiển nén theo tác vụ**, không phải một bộ mã hoá ngữ nghĩa mới.

## 0. Quy tắc định trước (commit trước khi tinh chỉnh, train hay đánh giá bất kỳ thứ gì)

### 0.1 Đo giá trị tác vụ
- Mọi cặp (ảnh, mode) của bảng A5, gồm cả 3 mode sinh ảnh, được đo bằng detector A5 (YOLO11s, train trên VisDrone-train). Code: `a5_fidelity.py --save-dets`, script `run_a6l1_gpu.sh`.
- Điểm hoạt động: conf ≥ 0,25, IoU ≥ 0,5, ghép class-aware, vùng ignore như A5.2.
- Ở mỗi (ảnh i, mode m) có TP_i(m) và FP_i(m) = số detection − TP.

### 0.2 Reward tác vụ

**Giá trị tác vụ** của một khung:
- v_i(m) = (TP_i(m) − β·FP_i(m)) / V̄, với β = 0,5;
- V̄ = trung bình của (TP − β·FP) của **ảnh gốc** trên các ảnh **train**.
- Ảnh có nhiều vật thể thì đáng giá hơn: tổng TP chính là tử số của "object recall được giao".

**Utility mỗi ảnh:** ξ_q·v_i(m) + ξ_d·max(d_max − delay, 0)/d_max − 1·[vi phạm Qmin hoặc deadline], với ξ_q = ξ_d = 0,5.
- Chỉ số hạng ξ_q·clip((PSNR − Qmin)/15) của A5 được thay.
- Phạt rớt ảnh (−1), ảnh chưa chụp, năng lượng, thời gian (w_T = 0,3) và **ràng buộc Qmin** đều giữ nguyên. Qmin là yêu cầu của hệ thống; việc giữ nó là một giới hạn, xem mục 5.
- Không clip v.

**Nhãn dùng ở đâu:**
- reward khi train: nhãn ảnh train;
- chọn checkpoint: nhãn ảnh val;
- đánh giá: nhãn ảnh test và test-dev.

Nhãn **không** bao giờ vào quan sát của policy.

**Quan sát giữ nguyên như A5.**
- Policy biết payload/PSNR dự đoán của mọi mode. Các giá trị này là hàm tuyến tính của log(payload probe), tức của cùng một con số đo được trên UAV.
- Heuristic task-aware dùng v̂_i(m) = a_m + b_m·log(payload probe), fit trên train. Đây là cùng thông tin mà PPO nhìn thấy, không thêm gì.

### 0.3 Heuristic dưới reward tác vụ
- **Họ `tv:SLOT+budgetτ[+reuse]`:** giống các họ budget/pacedB/reuse của A5, nhưng giá trị dự đoán của mỗi mode dùng v̂ thay cho PSNR. 56 thành viên, cùng lưới.
- **Tinh chỉnh lại** cả các họ A5 (base, budget, reuse) **và** họ tv dưới reward tác vụ, trên train seed 0..29.
- **T:best_task** = thành viên có return train (reward tác vụ) cao nhất trong tất cả các họ.
- T:best của A5 (tinh chỉnh dưới PSNR) được giữ làm đối chứng.

### 0.4 PPO
- Mode-only, 3 seed (0, 1, 2), ở 3 regime ref, K5, phy_mcs_blind.
- Slot do phần slot của T:best_task điều khiển.
- Thuật toán, mạng, siêu tham số, 6 env, 2M bước giống hệt A5.
- Checkpoint chọn theo return **tác vụ** cao nhất trên val (seed 3000..3019; bằng nhau thì lấy checkpoint sớm hơn).
- Không tăng số bước. Nếu đường học còn tăng ở 2M, chỉ ghi nhận.

### 0.5 Đánh giá (seed mới 9000..9099, chưa từng dùng; pool test và test-dev)

**Policy:**
- PPO task (best-val);
- PPO PSNR của A5 (mode-only best-val);
- T:best_task;
- T:best của A5;
- fixed tinh chỉnh dưới reward tác vụ;
- thành viên tv tốt nhất.

**Mỗi episode báo cả hai return** (reward tác vụ và reward PSNR của A5, tính song song trong cùng một lần chạy), cùng các chỉ số:
- object recall được giao: Σ TP của khung giao đúng hạn / Σ số vật thể của mọi khung đã chụp;
- recall vật nhỏ;
- FP (tổng và mỗi khung giao) và precision;
- **AP50 chuẩn COCO** trên toàn bộ khung đã chụp; khung không giao được tính như không có detection;
- tổng bit gửi, tỉ lệ đúng hạn, năng lượng, thời gian nhiệm vụ.

**So sánh chính** (theo cặp; CI theo seed môi trường, cộng cluster bootstrap theo video với B = 50 như A5.5 0.4; seed của bản bootstrap là 30000 + 20b + i, ghi thêm trước khi chạy):
1. PPO task − PPO PSNR: recall, AP50, bit, đúng hạn;
2. T:best_task − T:best: cùng các chỉ số;
3. PPO task − T:best_task: return tác vụ, recall.

**Không** dùng "đảo thứ hạng" làm tiêu chí thành công duy nhất. Báo cáo:
- Pareto recall–bit, recall–đúng hạn, recall–năng lượng/thời gian;
- **phân bố mode được chọn** theo tam phân vị của độ phức tạp ảnh (log payload probe), v̂, backlog dự đoán, SNR kỳ vọng và slack, so sánh policy task với PSNR.

Nếu reward tác vụ không tạo khác biệt, hoặc heuristic ngang PPO, thì ghi rõ.

**Phân tích độ nhạy** (khám phá, không train lại): chấm lại điểm tác vụ với β ∈ {0; 1}.

## 1. Code, test và cách chạy

| Thành phần | File |
|---|---|
| Giá trị tác vụ, env reward tác vụ | `swarmbridge/experiments/uav_semcom/a6l1_task.py` |
| AP50 COCO theo policy | `a6l1_ap.py` |
| Tune / eval / bootstrap | `a6l1_eval.py` |
| PPO | `a6l1_train.py` |
| Bảng tổng hợp | `a6l1_summarize.py` |
| Oracle khám phá (mục 9) | `a6l1_oracle.py` |
| Hàng đợi | `scripts/uav_semcom/run_a6l1_gpu.sh`, `run_a6l1_cpu.sh` |

**Test** (`test_a6l1.py`, 7/7 qua trên dữ liệu thật; hàng đợi không cho phép skip):
- ghép detection–nhãn và vùng ignore;
- chuẩn hoá V̄;
- env tác vụ chỉ đổi reward: cùng quyết định, và return PSNR chạy song song trùng return của env A5;
- chỉ số tác vụ tính lại từ danh sách giao;
- AP50 theo policy tái tạo AP50 của COCOeval ở 6 mode, kể cả mode sinh ảnh.

**Kết quả thô:**
- `runs/uav_semcom/a6l1/{tune,train,final,boot,fidelity}`; bảng đầy đủ ở `runs/uav_semcom/a6l1/summary.md`.
- File trạng thái quyết định `final/*.dec.json.gz` (21 MB) không đưa vào git; sha256 ở `final/dec_sha256.txt`.

**Thời gian:**
- đo fidelity trên GPU: khoảng 2,5 giờ;
- tune: 3 phút;
- 9 run PPO: 4 giờ (CPU);
- đánh giá cuối và bootstrap: 50 phút.

## 2. Đo đạc: giá trị tác vụ theo mode (ảnh train, V̄ = 31,7)

| mode | payload trung vị (Mbit) | PSNR (dB) | v trung bình | recall | FP/ảnh |
|---|---|---|---|---|---|
| avif_20 | 0,16 | 28,4 | 0,79 | 0,525 | 15,2 |
| mbt2018-mean_q3 | 0,28 | 31,4 | 0,92 | 0,576 | 15,9 |
| mbt2018-mean_q5 | 0,59 | 35,5 | 0,98 | 0,607 | 17,0 |
| avif_50 | 0,60 | 35,4 | 0,96 | 0,598 | 16,2 |
| avif_80 | 1,23 | 44,3 | 1,00 | 0,617 | 16,2 |
| mbt2018-mean_q8 | 1,41 | 42,0 | 1,01 | 0,623 | 16,6 |
| cap_thumb128_i2i_t0.3 | 0,03 | 21,5 | 0,17 | 0,196 | 9,8 |
| cap_t2i | ≈0 | 9,6 | **−1,01** | 0,002 | **64,5** |

**Cơ chế chính:**
- Từ avif_50 lên avif_80, PSNR tăng 8,9 dB. Số hạng chất lượng của reward A5 tăng tới khoảng 0,3 mỗi ảnh (ξ_q·ΔPSNR/15, trước khi clip).
- Giá trị phát hiện chỉ tăng 4%, tức +0,02 mỗi ảnh sau ξ_q, trong khi payload gấp đôi.
- Dưới reward tác vụ, số hạng deadline-slack (ưu tiên payload nhỏ) vì thế lấn át phần lợi chất lượng.
- **Detector bão hoà ở chất lượng trung bình; PSNR thì không.**

**Mode sinh ảnh:** ảnh sinh từ caption (cap_t2i) tạo khoảng 64 false positive mỗi ảnh (vật thể "ảo"), nên giá trị tác vụ âm. Đây là phép đo đầu tiên trong dự án cho thấy mode sinh ảnh có hại cho detector, chứ không chỉ vô ích.

**v̂ (dự đoán từ payload probe):**
- Gần như luôn xếp mbt2018-mean_q8 cao nhất (418/432 ảnh train + test).
- Thứ hạng các mode hầu như **không đổi theo ảnh**, nên v̂ không mang thông tin để chọn mode khác nhau cho từng ảnh.

## 3. Tinh chỉnh heuristic dưới reward tác vụ (train, seed 0..29)

| regime | T:best_task | R_task | A5 T:best (PSNR) | R_task |
|---|---|---|---|---|
| ref | **fixed_avif_50** (ngang `tv:greedy+budget1.25+reuse`) | +69,17 | pacedB1.5+budget1.25+reuse | +59,16 |
| K5 | tv:greedy+budget1+reuse | +96,70 | pacedB2+budget1+reuse | +87,95 |
| phy_mcs_blind | **fixed_mbt2018-mean_q3** | +57,38 | pacedB2+budget1 | +43,81 |

Ở 2/3 regime, heuristic tốt nhất dưới reward tác vụ là **một mode cố định**.

## 4. Huấn luyện PPO

- Return val bão hoà sớm:
  - ref: best ở 0,4–0,8M bước;
  - K5: 1,2–1,8M;
  - phy_mcs_blind: 0,6–2M, chênh < 0,15 giữa các checkpoint.
- 3 seed cho return val gần như bằng nhau (ref +94,6..94,8; K5 +148,0..150,2; phy_mcs_blind +83,25..83,35).
- Không có run nào tăng có ý nghĩa ở 2M bước.

## 5. Kết quả trên seed mới 9000..9099

R = return; recall = Σ TP của khung giao đúng hạn / Σ số vật thể của mọi khung đã chụp; AP50 = COCO AP50 gộp trên mọi khung (khung không giao được thì không có detection). PPO là trung bình của 3 model.

| regime | pool | policy | R_task | R_PSNR | recall | recall nhỏ | AP50 | Mbit | đúng hạn |
|---|---|---|---|---|---|---|---|---|---|
| ref | test | **PPO task** | 76,3 | 48,6 | 0,589 | 0,470 | 0,433 | 79 | 0,999 |
| | | PPO PSNR (A5) | 74,5 | 59,3 | 0,596 | 0,479 | 0,441 | 117 | 0,999 |
| | | T:best_task (= fixed avif_50) | 76,3 | 48,9 | 0,589 | 0,470 | 0,433 | 79 | 0,999 |
| | | A5 T:best | 68,2 | 59,5 | 0,596 | 0,481 | 0,442 | 141 | 0,992 |
| ref | test-dev | **PPO task** | 35,5 | 46,2 | 0,498 | 0,368 | 0,350 | 108 | 0,998 |
| | | PPO PSNR | 31,5 | 49,3 | 0,502 | 0,373 | 0,352 | 140 | 0,996 |
| | | T:best_task | 35,1 | 46,3 | 0,498 | 0,369 | 0,351 | 109 | 0,997 |
| | | A5 T:best | 25,7 | 46,3 | 0,493 | 0,368 | 0,344 | 156 | 0,982 |
| K5 | test | **PPO task** | 117,9 | 49,7 | 0,565 | 0,445 | 0,402 | 77 | 0,978 |
| | | PPO PSNR | 118,6 | 64,3 | 0,578 | 0,459 | 0,418 | 109 | 0,990 |
| | | T:best_task | 114,9 | 65,9 | 0,570 | 0,454 | 0,418 | 116 | 0,980 |
| | | A5 T:best | 103,4 | 62,5 | 0,568 | 0,452 | 0,418 | 139 | 0,970 |
| K5 | test-dev | **PPO task** | 58,7 | 52,4 | 0,469 | 0,341 | 0,314 | 96 | 0,996 |
| | | PPO PSNR | 50,5 | 54,4 | 0,479 | 0,351 | 0,328 | 130 | 0,993 |
| | | T:best_task | 44,0 | 46,4 | 0,445 | 0,331 | 0,308 | 115 | 0,939 |
| | | A5 T:best | 36,6 | 42,9 | 0,441 | 0,332 | 0,304 | 132 | 0,935 |
| phy_mcs_blind | test | **PPO task** | 64,3 | 20,2 | 0,558 | 0,439 | 0,398 | 39 | 0,971 |
| | | PPO PSNR | 63,6 | 32,5 | 0,580 | 0,461 | 0,423 | 70 | 0,989 |
| | | T:best_task (= fixed mbt q3) | 64,4 | 20,0 | 0,559 | 0,439 | 0,398 | 39 | 0,971 |
| | | A5 T:best | 54,8 | 27,4 | 0,550 | 0,437 | 0,403 | 75 | 0,947 |
| phy_mcs_blind | test-dev | **PPO task** | 28,6 | 22,8 | 0,462 | 0,334 | 0,306 | 50 | 0,989 |
| | | PPO PSNR | 17,2 | 22,4 | 0,482 | 0,354 | 0,332 | 87 | 0,986 |
| | | T:best_task | 28,9 | 23,0 | 0,464 | 0,336 | 0,307 | 50 | 0,992 |
| | | A5 T:best | 13,9 | 14,9 | 0,423 | 0,317 | 0,290 | 70 | 0,911 |

Năng lượng và thời gian nhiệm vụ gần như không đổi giữa các policy: chênh ≤ 4 kJ và ≤ 9 s; xem `summary.md`.

### 5.1 So sánh định trước (CI mức 1 theo seed môi trường; mức 3 thêm video và seed huấn luyện)

**1. PPO task − PPO PSNR:**
- **Bit giảm 30–39 Mbit mỗi episode (−23% đến −44%)** ở cả 6 (regime, pool). CI mức 3 đều âm.
- **Recall giảm 0,4–2,1 điểm**: CI mức 3 âm ở cả 6 trường hợp.
- **AP50 giảm 0,3–2,7 điểm**: chỉ có CI theo seed môi trường, cả 6 đều âm.
- Vật nhỏ giảm tương tự: 0,5–2,3 điểm.
- R_task:
  - tăng ở 4/6 trường hợp: ref test +1,8; ref test-dev +4,0; K5 test-dev +8,2; phy_mcs_blind test-dev +11,4;
  - K5 test (−0,6) và phy_mcs_blind test (+0,7): CI mức 3 chứa 0.
- Tỉ lệ đúng hạn: không đổi, trừ hai trường hợp giảm 1–2 điểm (K5 test, phy_mcs_blind test).

**2. T:best_task − A5 T:best:**
- Bit giảm 16–62 Mbit; tỉ lệ đúng hạn tăng 0,4–8 điểm.
- Recall: −0,7 điểm ở ref test; +0,3 đến +4,0 điểm ở các trường hợp khác.
- Recall tăng rõ nhất ở phy_mcs_blind test-dev (+4,0 điểm, CI mức 3 [+1,8; +9,2]), vì mode cố định mbt q3 ít rớt ảnh hơn hẳn.

**3. PPO task − T:best_task:**
- **Không khác biệt thực tế** ở ref và phy_mcs_blind: |Δ R_task| ≤ 0,34, |Δ recall| ≤ 0,16 điểm. Ở đây PPO học lại đúng mode cố định.
- **K5 test-dev: PPO hơn rõ:**
  - R_task +14,7, CI mức 3 [+6,3; +22,7];
  - recall +2,4 điểm, CI mức 3 [+0,1; +6,5];
  - đúng hạn +5,7 điểm.
  Đây là cùng chỗ PPO thắng ở A5/A5.5: kênh chung chật, heuristic rớt ảnh.
- **K5 test:** R_task +3,0 (CI mức 3 chứa 0), nhưng recall −0,6 và AP50 −1,5 điểm, vì PPO task chọn mode rẻ hơn (mbt q3 82%).

## 6. Chọn mode theo trạng thái (pool test; chi tiết theo tam phân vị trong `summary.md`)

| regime | PPO task | PPO PSNR | T:best_task | A5 T:best |
|---|---|---|---|---|
| ref | avif_50 99% | avif_50 49% · avif_80 50% | avif_50 100% | avif_80 83% · avif_50 13% |
| K5 | mbt q3 82% · avif_50 16% | avif_50 67% · mbt q3 33% | avif_50 74% · mbt q3 15% · avif_80 5% | avif_50 50% · avif_80 28% · mbt q3 14% |
| phy_mcs_blind | mbt q3 98% | avif_50 78% · mbt q3 22% | mbt q3 100% | avif_50 49% · avif_80 23% · mbt q3 17% |

**Reward tác vụ đổi rõ mode được chọn: bỏ hẳn avif_80, chuyển sang avif_50 hoặc mbt q3.**

Nhưng PPO task gần như **không phụ thuộc trạng thái**. Ở cả 5 biến (độ phức tạp probe, v̂, backlog, SNR, slack), tỉ lệ mode chính giữa ba tam phân vị chênh ≤ 8 điểm.

Policy PSNR thì phụ thuộc trạng thái:
- **PPO PSNR ở ref:**
  - backlog thấp → avif_80 70%, backlog cao → avif_50 73%;
  - SNR cao → avif_80 72%.
- **A5 T:best:** cũng chọn avif_80 khi backlog thấp.

Lưu ý: v̂ là hàm đơn điệu của payload probe, nên tam phân vị của v̂ trùng tam phân vị độ phức tạp; đây không phải hai bằng chứng độc lập.

## 7. Pareto (recall ↑, Mbit ↓)

Ở cả 6 (regime, pool):
- policy tác vụ (PPO task, T:best_task hoặc mode cố định tương ứng) nằm trên biên Pareto ở **phía ít bit**;
- PPO PSNR nằm trên biên ở **phía recall cao**.

Reward tác vụ không mở rộng biên; nó dời điểm hoạt động dọc theo biên. Mỗi điểm recall đổi lấy khoảng 15–80 Mbit mỗi episode, tuỳ regime.

K5 test-dev:
- PPO task trội hơn mọi heuristic **thích nghi** ở cả recall lẫn bit.
- So với mode cố định mbt q3, PPO task hơn 0,7 điểm recall nhưng tốn thêm 13 Mbit.
- Đây là hiệu ứng tắc nghẽn, không phải hiệu ứng tác vụ.

## 8. Độ nhạy β (khám phá, chấm lại không train lại)

Với β ∈ {0; 0,5; 1}, dấu của mọi so sánh định trước không đổi.
- PPO task − PPO PSNR luôn ≤ 0 về điểm tác vụ mỗi khung: −0,01 đến −0,04, vì PPO task bỏ chất lượng để lấy slack.
- K5 test-dev PPO task − T:best_task luôn > 0: +0,05..+0,09.

Kết luận không do lựa chọn β = 0,5.

## 9. Khám phá sau khi xem kết quả: dư địa của chọn mode theo từng ảnh

`a6l1_oracle.py` là offline, không có kênh và deadline. Oracle biết giá trị tác vụ **thật** của từng ảnh và chọn argmax_m v_i(m) − λ·bits_i(m), so với mode cố định tốt nhất có payload trung bình ≤ payload của oracle.

| pool | λ | oracle Mbit/khung | oracle v / recall | mode cố định tốt nhất | Mbit | v / recall |
|---|---|---|---|---|---|---|
| test | 0,2 | 0,37 | 1,06 / 0,603 | mbt q3 | 0,25 | 0,96 / 0,568 |
| test | 0,4 | 0,25 | 1,02 / 0,589 | mbt q3 | 0,25 | 0,96 / 0,568 |
| test | 0,8 | 0,19 | 0,99 / 0,568 | avif_20 | 0,13 | 0,83 / 0,504 |
| train | 0,4 | 0,29 | 0,98 / 0,555 | avif_20 | 0,16 | 0,79 / 0,475 |

- Ở cùng khoảng 0,25 Mbit/khung, oracle hơn mode cố định khoảng 2 điểm recall.
- Ở cùng recall 0,568, oracle tốn ít hơn khoảng 25% bit.
- Đây là **cận trên lạc quan**: chọn max trên các phép đo có nhiễu (detector trên một ảnh) có thiên lệch dương.
- Policy hiện có không chạm tới dư địa này vì quan sát không chứa tín hiệu nào dự đoán được **mode nào tốt hơn cho ảnh nào** (mục 2: v̂ xếp hạng gần như cố định).

## 10. Kết luận A6.0-L1

1. **Reward tác vụ thay đổi lựa chọn mode, có ý nghĩa và bền.** Policy bỏ chất lượng cao (avif_80), giảm 23–44% bit, tăng tỉ lệ đúng hạn so với heuristic PSNR. Đổi lại, recall giảm 0,4–2,1 điểm và AP50 giảm 0,3–2,7 điểm so với PPO PSNR.
   - Tức là nó chọn **điểm hoạt động rẻ hơn trên cùng biên Pareto**, chứ không giao được **nhiều thông tin tác vụ hơn**.
   - Recall và AP50 **không** tăng so với PPO PSNR ở trường hợp nào.
2. **Phần lớn hiệu ứng là tĩnh.** Dưới reward tác vụ, policy tốt nhất ở ref và phy_mcs_blind là một mode cố định (avif_50, mbt q3). PPO học lại đúng mode đó, không có lựa chọn theo trạng thái.
3. **PPO chỉ thêm giá trị ở K5 test-dev**, do tắc nghẽn: chính hiệu ứng đã thấy ở A5/A5.5, nay lặp lại dưới reward khác.
4. **Mode sinh ảnh có giá trị tác vụ âm** với detector này (ảo giác FP).
5. Như quy tắc 0.5 yêu cầu, ghi nhận nguyên trạng: **không có hiệu ứng "task-aware" động** trong hệ này. Thứ hạng giữa PPO và heuristic dưới reward tác vụ giống dưới PSNR: hơn ở tắc nghẽn, ngang ở nơi khác.

**Giới hạn:**
- Một detector (YOLO11s) ở một điểm hoạt động (conf 0,25).
- β và ξ_q = ξ_d cố định; reward tác vụ có tỉ lệ khác reward PSNR, nên đánh đổi chất lượng–slack bị dời.
- Qmin (theo PSNR) vẫn được giữ, đây là ràng buộc của hệ thống.
- Chỉ mode-only, 3 seed.
- AP50 chỉ có CI theo seed môi trường, vì bootstrap theo video không lưu danh sách giao.

## 11. Đề xuất cổng quyết định tiếp theo

**Câu hỏi còn mở có giá trị:** có tín hiệu **nhân quả, rẻ, trên UAV** nào dự đoán được *mode nào giữ được vật thể cho ảnh nào*, đủ để thu một phần dư địa oracle ở mục 9 không?

Nếu không có, chọn mode theo tác vụ trong hệ này chỉ là chọn một mode cố định tốt, và RL không cần thiết ngoài tình huống tắc nghẽn.

**Cổng G-L2 (offline; CPU + GPU ngắn; không RL, không simulator mới):**
- Bộ dự đoán ứng viên:
  - (a) detector nhỏ chạy trên ảnh probe hoặc thumbnail trên UAV (số box, kích thước, độ tin cậy);
  - (b) đặc trưng ảnh rẻ (biên, năng lượng tần số cao);
  - (c) v̂ hiện tại làm đối chứng.
- Train trên train, chọn trên val.
- Đo trên test và test-dev: tỉ lệ dư địa oracle thu được, ở cùng bit với mode cố định tốt nhất.
- Oracle phải được khử thiên lệch trước (chia đôi phép đo hoặc bootstrap) rồi mới tính tỉ lệ.
- **Ngưỡng đề xuất (cần thống nhất trước khi chạy):**
  - thu ≥ 30% dư địa đã khử thiên lệch **và** ≥ +1 điểm recall ở cùng bit, CI theo video không chứa 0 → đi tiếp L2 (đưa bộ dự đoán vào quan sát, PPO / heuristic);
  - ngược lại → dừng nhánh "task-aware mode selection".

**Phương án song song, không cần thí nghiệm mới:** viết A5 + A5.5 + L1 thành một kết quả về **phương pháp đánh giá**: "RL so với heuristic mạnh dưới bất định theo video, và reward PSNR so với reward tác vụ".
- Novelty audit (`reports/novelty-audit-a5-l1-2026-11-10.md`) đánh giá phần này ở mức trung bình.
- Khoảng trống rõ nhất là chưa ai cho thấy reward tác vụ đổi lựa chọn mode trên cùng hệ thống. L1 cho thấy điều đó, cùng với giới hạn của nó.

**Không đề xuất lúc này:** relay, MARL, CARLA hay UAV3D. Chúng không giải quyết câu hỏi tín hiệu dự đoán ở trên.
