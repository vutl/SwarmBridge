# SwarmBridge hướng A — Kiểm toán phương pháp A6.0-L1 (bước 1 của đợt G-L2)

Ngày 10/11/2026. Báo cáo L1 gốc (`reports/uav-semcom-a6-l1.md`) **giữ nguyên**, không sửa số nào. Báo cáo này kiểm tra ba điểm phương pháp mà người review nêu, rồi nói kết luận nào của L1 còn đứng.

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
