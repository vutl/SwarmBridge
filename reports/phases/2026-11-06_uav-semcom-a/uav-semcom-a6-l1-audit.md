# SwarmBridge hướng A — Kiểm toán phương pháp A6.0-L1 (bước 1 của đợt G-L2)

Ngày 10/11/2026. Báo cáo L1 gốc (`reports/archive/uav-semcom-a/uav-semcom-a6-l1.md`) **giữ nguyên**, không sửa số nào. Báo cáo này kiểm tra ba điểm phương pháp mà người review nêu, rồi nói kết luận nào của L1 còn đứng.

**Không train PPO trong đợt này.** Mọi phép đo dùng các model PPO đã có (A5 mode-only và L1), heuristic, và simulator A5 sẵn có.

## 0. Quy tắc định trước (commit trước khi chạy bất kỳ phép đo nào của báo cáo này)

### 0.1 Confound: PPO L1 và PPO A5 dùng phần slot khác nhau

Mode-only PPO không tự quyết slot (bay, kênh, công suất): phần slot do một heuristic điều khiển.
- PPO PSNR (A5): phần slot của **T:best A5** (tinh chỉnh dưới reward PSNR).
- PPO task (L1): phần slot của **T:best_task** (tinh chỉnh dưới reward tác vụ).

| regime | slot của PPO PSNR (S_psnr) | slot của PPO task (S_task) |
|---|---|---|
| ref | pacedB1.5 + reuse (từ `pacedB1.5+budget1.25+reuse`) | slot của `fixed_avif_50` = `policies.fixed`: bay theo nhiệm vụ, chia kênh, công suất max, không pacing, không reuse |
| K5 | pacedB2 + reuse | greedy + reuse (từ `tv:greedy+budget1+reuse`) |
| phy_mcs_blind | pacedB2 | slot của `fixed_mbt2018-mean_q3` (như ref) |

Vậy so sánh "PPO task − PPO PSNR" của L1 thay **hai** thứ cùng lúc: reward của mode policy và phần slot.

### 0.2 Đánh giá chéo với cùng phần slot

**Ô mới** (pool test và test-dev, seed 9000..9099, 3 regime):
- PPO task (3 model best-val của L1) chạy với S_psnr;
- PPO PSNR (3 model best-val của A5) chạy với S_task;
- heuristic ghép: phần mode của T:best_task + slot S_psnr, và phần mode của T:best A5 + slot S_task.

Ô chéo đường (mỗi policy với slot của chính nó) chạy lại để kiểm tính tất định: phải trùng số L1.

**Phân rã:**
- Hiệu ứng của reward lên mode policy, khi slot cố định: Δ_mode(S) = PPO task@S − PPO PSNR@S, với S = S_psnr và S = S_task.
- Hiệu ứng của slot, khi mode policy cố định: Δ_slot = policy@S_task − policy@S_psnr.

**Chỉ số:** bit gửi, recall, recall vật nhỏ, AP50 (COCO theo policy như L1), tỉ lệ đúng hạn, R_task, R_PSNR, phân bố mode. CI mức 1 (t ghép cặp theo seed môi trường, trung bình 3 model); không chạy bootstrap video ở bước này vì tốn kém. Mức 1 đã biết là hẹp hơn thực tế (A5.5), nên chỉ dùng để xem dấu.

**Giới hạn đã biết:** PPO ở ô chéo chạy với phần slot nó chưa từng gặp khi train (lệch phân phối). Một khác biệt nhỏ ở ô chéo có thể do lệch phân phối chứ không do reward.

**Quy tắc kết luận (C1 của L1: "reward tác vụ làm giảm bit và giảm recall so với PPO PSNR"):**
- **"Không phụ thuộc slot"** nếu ở mỗi (regime, pool), Δ_mode có bit < 0 và recall < 0, CI mức 1 không chứa 0, **ở cả hai** S.
- **"Phụ thuộc slot"** nếu dấu đổi hoặc CI chứa 0 ở một trong hai S.
  Khi đó đề xuất (không chạy) một thí nghiệm train đối chứng nhỏ: PPO task với S_psnr, 3 seed. Thí nghiệm này cần người dùng đồng ý.

### 0.3 Qmin theo PSNR trong reward tác vụ

L1 giữ ràng buộc Qmin ~ U(22, 30) dB. Một khung có PSNR < Qmin bị tính là vi phạm (−1), **không được tính vào recall**, kể cả khi detector vẫn dùng được nó.

**(a) Tĩnh, trên ảnh train:** với mỗi mode, báo
- xác suất vi phạm Qmin: trung bình theo ảnh của P(PSNR_i(m) < Qmin);
- giá trị tác vụ trung bình v;
- recall và payload.

**(b) Biến thể "noqmin":** cfg `qmin = (0, 0)`, nên không khung nào vi phạm Qmin. Số lần rút ngẫu nhiên không đổi, nên mọi biến ngẫu nhiên khác giữ nguyên.

Thêm chỉ số **recall_ontime**: Σ TP của khung giao trước deadline, bất kể Qmin, chia cho Σ số vật thể. Chỉ số này được tính cho **mọi** policy ở mục 0.3–0.4.

### 0.4 Độ nhạy của trọng số chất lượng–deadline (chỉ train / val)

**Lưới 8 cấu hình:** ξ_q ∈ {0,5 (L1); 1; 2; 4}, giữ ξ_d = 0,5, nhân với Qmin {bật, tắt}.

Với mỗi cấu hình và mỗi regime (ref, K5, phy_mcs_blind):
- tinh chỉnh lại mọi họ heuristic (cùng lưới như L1: base, budget, reuse, tv) dưới reward tác vụ của cấu hình đó, trên train seed 0..29;
- best_task(cấu hình) được đánh giá trên **val** seed 3000..3049.

**Đối chiếu trên val, cùng seed:**
- A5 T:best (PSNR);
- PPO PSNR (3 model);
- PPO task (3 model);
- T:best_task của L1.

**Chỉ số:** recall_ontime, recall, Mbit gửi, đúng hạn, AP50.

**Quy tắc chọn (cố định trước):** với mỗi regime, trong 8 cấu hình, chọn cấu hình có recall_ontime val cao nhất trong số những cấu hình gửi **không nhiều bit hơn PPO PSNR** trên val. Bằng nhau thì lấy ít bit hơn. Nếu không cấu hình nào thoả, chọn recall_ontime cao nhất và ghi rõ.

Heuristic của cấu hình được chọn chạy **một lần** trên test và test-dev, seed 9000..9099. Nó được so với PPO PSNR và T:best_task chạy lại trên cùng seed, CI mức 1.

Câu hỏi cần trả lời: có trọng số tác vụ nào đạt recall như PPO PSNR với ít bit hơn không? Test không được dùng để chọn gì.

### 0.5 Kết luận nào của L1 còn đứng

Mỗi kết luận được ghi một trong ba nhãn: **đứng vững**, **cần sửa lại cách nói**, **không đứng**.

| | Kết luận L1 |
|---|---|
| C1 | Reward tác vụ giảm bit 23–44% và giảm recall 0,4–2,1 điểm so với PPO PSNR |
| C2 | Policy tác vụ tốt nhất là một mode cố định ở ref và phy_mcs_blind |
| C3 | PPO chỉ thêm giá trị ở K5 test-dev (nghẽn kênh) |
| C4 | Detector bão hoà ở chất lượng trung bình (cơ chế) |
| C5 | v̂ từ payload probe không phân biệt mode theo ảnh |

## 1. Code, cách chạy, tính tái lập

| Thành phần | File |
|---|---|
| Đánh giá chéo, tune theo cấu hình, val, final | `swarmbridge/experiments/uav_semcom/a6l1_audit.py` |
| Bảng tổng hợp | `a6l1_audit_summarize.py` |
| Hàng đợi (CPU, khoảng 1,5 giờ) | `scripts/uav_semcom/run_a6l1_audit.sh` |
| Kết quả thô; bảng đầy đủ ở `summary.md` | `runs/uav_semcom/a6l1/audit/` |

**Tái lập:**
- Ô chéo đường (policy với slot của chính nó) trùng từng episode với L1 (đã so 16 episode).
- Cấu hình ξ_q = 0,5 có Qmin chọn lại đúng T:best_task của L1, cùng return train (+69,17 / +96,70 / +57,38).

## 2. Kết quả 0.2: tách reward khỏi phần slot

**Δ_mode** = PPO task − PPO PSNR khi **cùng** phần slot. CI mức 1.

| regime | pool | ΔMbit @ S_psnr | Δrecall @ S_psnr | ΔMbit @ S_task | Δrecall @ S_task |
|---|---|---|---|---|---|
| ref | test | −40,2 | −0,82 [−0,89; −0,76] | −35,0 | −0,60 [−0,68; −0,53] |
| K5 | test | −31,6 | −1,33 [−1,41; −1,24] | −32,7 | −1,26 [−1,34; −1,17] |
| phy_mcs_blind | test | −30,4 | −2,13 [−2,28; −1,98] | −29,7 | −1,79 [−1,98; −1,61] |
| ref | test-dev | −37,6 | −0,66 [−0,72; −0,59] | −24,5 | **−0,05 [−0,13; +0,03]** |
| K5 | test-dev | −33,2 | −0,90 [−0,97; −0,84] | −32,3 | −0,38 [−0,48; −0,27] |
| phy_mcs_blind | test-dev | −37,2 | −1,83 [−1,99; −1,67] | −31,4 | −0,57 [−0,80; −0,34] |

AP50 đi cùng chiều với recall (−0,6 đến −2,6 điểm), trừ ref test-dev @ S_task (+0,15).

**Hiệu ứng của slot** (cùng mode policy, đổi slot):
- Với PPO task: nhỏ, ≤ 0,27 điểm recall.
- Với PPO PSNR trên test-dev: lớn hơn, −0,33 / −0,59 / −1,44 điểm. Đặt PPO PSNR lên slot của mode cố định (không pacing) làm nó rớt khung nhiều hơn.

Đây là ô lệch phân phối đã báo trước ở 0.2. Một phần hiệu recall trên test-dev của L1 (đặc biệt phy_mcs_blind test-dev) đến từ slot, không chỉ từ reward.

**Theo quy tắc 0.2:**
- C1 "không phụ thuộc slot" ở 5/6 (regime, pool).
- **ref test-dev phụ thuộc slot:** với S_task, mất mát recall biến mất; bit vẫn giảm 24 Mbit.

**Heuristic, ghép phần mode với cùng slot.** Ở 11/12 ô, phần mode tinh chỉnh theo tác vụ gửi ít bit hơn (−2 đến −62 Mbit) **và** recall cao hơn (+0,3 đến +5,5 điểm); ngoại lệ là ref test @ S_psnr (−0,68). Lý do: phần mode của heuristic tinh chỉnh theo PSNR làm rớt khung (đúng hạn +0,8 đến +9,7 điểm khi đổi sang mode tác vụ).

Vậy:
- **với heuristic,** tinh chỉnh theo tác vụ là cải thiện Pareto;
- **với PPO,** vốn đã ít rớt khung, nó chỉ là một điểm rẻ hơn với recall thấp hơn.

## 3. Kết quả 0.3: Qmin

**(a) Qmin ~ U(22, 30) dB loại những mode nào** (ảnh train):

| mode | P(vi phạm) | v | recall | Mbit |
|---|---|---|---|---|
| avif_5 | 0,51 | 0,57 | 0,355 | 0,08 |
| avif_20 | 0,25 | 0,79 | 0,475 | 0,16 |
| mbt q1 | 0,27 | 0,75 | 0,442 | 0,12 |
| jpeg2000_100 | 0,44 | 0,61 | 0,385 | 0,25 |
| mbt q3 | 0,02 | 0,92 | 0,541 | 0,28 |
| avif_50 trở lên, mbt q5 trở lên | 0 | ≥ 0,96 | ≥ 0,56 | ≥ 0,59 |

Mode sinh ảnh hầu như luôn vi phạm. Bảng đủ ở `summary.md`.

**(b) Bỏ Qmin, ξ_q = 0,5:** heuristic tốt nhất chuyển sang mode **rẻ hơn** và recall **thấp hơn** trên val:
- phy_mcs_blind: avif_20, recall 0,456 so với 0,518;
- K5: recall 0,530 so với 0,554;
- ref: gần như không đổi.

Với ξ_q ≥ 1, bỏ Qmin cho lại đúng lựa chọn như khi có Qmin. Vậy **Qmin không loại các mode tốt cho tác vụ.** Ở ξ_q = 0,5, nó còn đang giữ recall.

**Phát hiện phụ: cách đếm recall dưới Qmin bỏ sót khung dùng được.**
- Khung giao **đúng hạn** nhưng PSNR < Qmin vẫn có detection hữu ích, nhưng recall của L1 không đếm chúng.
- Ở K5 test-dev, T:best_task có recall 0,445 nhưng recall_ontime 0,473 (+2,8 điểm). Ở các ô khác chênh ≤ 0,9 điểm.
- Điều này ảnh hưởng C3 (mục 5).

## 4. Kết quả 0.4: trọng số chất lượng–deadline

**Trên train:**
- Tăng ξ_q từ 0,5 lên 2 **không đổi** heuristic tốt nhất ở cả 3 regime: ref avif_50, K5 tv:greedy+budget1+reuse, phy_mcs_blind mbt q3.
- Chỉ ξ_q = 4 ở ref chuyển sang tv:greedy+budget1+reuse: val recall_ontime 0,574 ở 119 Mbit, ngang A5 T:best (0,575 ở 135 Mbit).

**Chọn trên val theo quy tắc:**

| regime | cấu hình chọn | lý do |
|---|---|---|
| ref | xq1_noqmin | |
| K5 | xq0.5_noqmin | các cấu hình recall cao hơn đều vượt trần bit của PPO PSNR |
| phy_mcs_blind | xq0.5 | = L1 |

**Một lần trên seed 9000..9099:** cả ba **thua PPO PSNR** về recall_ontime.

| | ref | K5 | phy_mcs_blind |
|---|---|---|---|
| test | −0,79 | −2,85 | −1,48 |
| test-dev | −1,75 | −3,67 | −1,74 |

Cả ba gửi ít hơn 19–57 Mbit; AP50 cùng chiều.

**Trả lời câu hỏi 0.4:** trong lưới đã thử, **không có trọng số tác vụ nào đạt recall của PPO PSNR với ít bit hơn.** Reward tác vụ chỉ dời điểm hoạt động dọc đường đánh đổi bit–recall, không đẩy đường đó ra ngoài.

## 5. Kết luận nào của L1 còn đứng

| | Kết luận L1 | Nhãn | Ghi chú |
|---|---|---|---|
| C1 | Reward tác vụ giảm bit 23–44%, giảm recall 0,4–2,1 điểm | **cần sửa lại cách nói** | Khi giữ cùng slot: bit giảm 24–40 Mbit ở cả 12 so sánh; recall giảm 0,4–2,1 điểm ở 11/12, riêng ref test-dev @ S_task ≈ 0. Một phần mất mát recall trên test-dev của L1 đến từ phần slot. |
| C2 | Policy tác vụ tốt nhất = mode cố định ở ref, phy_mcs_blind | **đứng vững** | Giữ ở mọi ξ_q ≤ 2 (ref) và mọi ξ_q (phy_mcs_blind), có hoặc không Qmin |
| C3 | PPO chỉ thêm giá trị ở K5 test-dev | **không đứng (về recall)** | PPO task − T:best_task ở K5 test-dev: recall +2,43 nhưng **recall_ontime −0,23 [−0,35; −0,11]**. Ưu thế "+2,4 điểm recall" chỉ đến từ việc T:best_task gửi khung PSNR thấp bị Qmin loại. PPO vẫn hơn về R_task và tỉ lệ đúng hạn **dưới luật Qmin**, không hơn về số vật thể giao đúng hạn. |
| C4 | Detector bão hoà ở chất lượng trung bình | **đứng vững** | Đo lại bằng ảnh lật (G-L2): avif_80 − avif_50 chỉ +1,2..1,4 TP/ảnh, và hiệu này lặp lại kém giữa hai phép đo (tương quan 0,41) |
| C5 | v̂ từ payload probe không phân biệt mode theo ảnh | **đứng vững** | G-L2: predictor dùng payload mọi mode (bộ A) còn thua luật RV |

**Hệ quả cho A5 / A5.5 (kiểm tra thêm, khám phá).** PPO PSNR − A5 T:best, seed 9000..9099:

| | ref | K5 | phy_mcs_blind |
|---|---|---|---|
| recall, test | +0,01 | +1,04 | +2,98 |
| recall_ontime, test | −0,12 | +0,36 | +1,50 |
| recall, test-dev | +0,89 | +3,85 | +5,88 |
| recall_ontime, test-dev | +0,48 | +1,12 | +2,61 |

Ưu thế về detection của PPO so với heuristic A5 **co lại khoảng một nửa đến hai phần ba** khi đếm mọi khung giao đúng hạn, nhưng còn dương ở 5/6 trường hợp. Câu "PPO ≥ heuristic về recall" của A5 cần ghi kèm giới hạn này. Kết quả return của A5.5 không đổi, vì nó đo đúng reward đã định nghĩa.
