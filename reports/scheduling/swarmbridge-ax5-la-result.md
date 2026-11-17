# A-X5 (phản biện #39): link adaptation thực tế, scheduler kênh train ngẫu nhiên hoá, baseline lùi theo NACK

Ngày chạy: 2026-11-17. Code: `env_la.py`, `la_heur.py`, `ax5_phy_validate.py`, `ax5_eval.py`, `ax5_summary.py`, test
`test_env_la.py` (15 test). Kết quả trong `runs/uav_semcom/ax5/`:
- `phy_validation.json`
- `gate_K5_la{,_rand}.json`
- `eval.json.gz` và `eval_summary.json`
- `train/K5_la{,_rand}_c_s*.json`

Checkpoint không đưa lên git. Báo cáo này tiếp nối [swarmbridge-ax2b-mcs-result.md](swarmbridge-ax2b-mcs-result.md).

Câu hỏi trung tâm của phản biện #39: *scheduler học được có khai thác được việc dùng chung kênh khi có lợi, và lùi lại
khi link không tin cậy, mà không làm hỏng mode controller hiện có không?*

## Trả lời ngắn

**Trong môi trường này: chưa.**
1. Với link adaptation thực tế, hầu như không còn trường hợp nào dùng chung kênh có lợi. Ở cả 7 điều kiện thử, luật tốt
   nhất đều không dùng chung kênh.
2. Scheduler train ngẫu nhiên hoá không học được cách lùi lại. Nó dùng chung kênh khoảng 0,86 thời gian phát ở mọi điều
   kiện và thua heuristic tune 43 điểm trên chính phân phối nó được train.
3. Scheduler train ở điều kiện danh nghĩa học được cách gần như không dùng chung kênh (0,05). Ở điều kiện danh nghĩa nó
   chỉ ngang heuristic của chính điều kiện đó (−1,4, CI chứa 0). Khi feedback chậm thì thua (−15 đến −19).
4. Cổng NACK cứu được phần lớn thiệt hại của cả hai scheduler học được (+18 đến +102 điểm), nhưng không vượt được
   heuristic không dùng chung kênh. Ngoại lệ duy nhất là scheduler cũ (train với MCS lý tưởng) cộng cổng NACK khi
   feedback tốt: +2,2 [−3,4, +7,9], không có ý nghĩa thống kê, 2 seed dương và 1 seed âm.

## Luồng

### 1. Mô hình PHY mới (bước A, đã kiểm chứng trước khi có RL)

**Khung thời gian.** Mỗi slot 1 s chia thành block 100 ms. Fading vẫn rút một lần mỗi slot như môi trường gốc.

**Chọn MCS cho từng block.**
- Lấy CQI báo cáo trễ D block. Mỗi báo cáo = SINR đo được + độ lệch b + nhiễu σ·z.
- Nếu chưa có báo cáo thì dùng SNR thống kê, không tính nhiễu.
- Trừ đi offset OLLA, rồi chọn MCS cao nhất có ngưỡng nằm dưới giá trị đó.

**Lỗi block.**
- BLER là hàm logistic theo dB, độ dốc 2,4/dB, đúng 10% tại ngưỡng CQI.
- SINR dùng để xét lỗi là SINR xấu nhất trong block. Nhiễu đồng kênh tính theo thời gian, chỉ xuất hiện khi UAV kia
  thực sự đang phát.

**ACK/NACK.**
- Ngẫu nhiên, dùng số ngẫu nhiên chung theo (block, UAV), nên mọi policy gặp cùng một chuỗi may rủi.
- Block hỏng giao 0 bit, phải phát lại (ARQ, không kết hợp soft).
- **HARQ có kết hợp chưa làm.** Đó là ablation riêng như phản biện yêu cầu.

**OLLA.** Mỗi UAV một offset, nhắm BLER 10%. NACK thì offset tăng một bước. ACK thì offset giảm bước·0,1/0,9. Kẹp trong
±10 dB.

**Kiểm chứng độc lập** (`ax5_phy_validate.py`, `test_env_la.py`):
- Link đứng yên: BLER đạt 0,099–0,102, hiệu suất khoảng 0,89–0,94 so với MCS lý tưởng. Đúng với mọi σ, b, D trong test.
- Tắt OLLA mà CQI lạc quan thì BLER > 0,5. Như vậy OLLA chính là thứ sửa lệch.
- Fading mỗi slot, khi D tăng từ 1 lên 20 block, hiệu suất so với lý tưởng giảm:
  - Rician: 0,85 → 0,66
  - Rayleigh: 0,67 → 0,37
- Trong môi trường, heuristic không dùng chung kênh (`fixed_mbt2018-mean_q3`):
  - Goodput 473 kbit/s ở K5_la, so với 643 ở MCS lý tưởng (K5_mcsg).
  - BLER 0,12.
- Heuristic dùng chung kênh (`pacedB2+budget1+reuse`): BLER khoảng 0,30, goodput chỉ khoảng 94 kbit/s.
- **Lưu ý:** tắt hẳn OLLA lại cho goodput cao hơn (540 so với 473 kbit/s, BLER 0,21). Như vậy mục tiêu 10% không tối ưu
  throughput ở đây; nguyên nhân chưa phân tích. Chúng tôi giữ 10% vì đó là chuẩn và phản biện yêu cầu.

### 2. Điều kiện (regime)

Tất cả dùng `phy = la`, K = 5 UAV, M = 2 kênh. Tham số ẩn là thứ policy không bao giờ quan sát được.

| Regime | σ CQI (dB) | Trễ D (block) | Bước OLLA | Lệch b (dB) | Ghi chú |
|---|---|---|---|---|---|
| K5_la_good | 0,5 | 1 | 0,5 | 0 | |
| K5_la (danh nghĩa) | 1 | 2 | 0,5 | 0 | |
| K5_la_l50 | 1 | 2 | 0,5 | 0 | phép đo chỉ thấy 50% nhiễu đồng kênh |
| K5_la_noisy | 3 | 2 | 0,5 | 0 | |
| K5_la_slow | 1 | 10 | 0,5 | 0 | |
| K5_la_bad | 3 | 20 | 0,5 | 3 | |
| **K5_la_rand** (train) | U(0, 3) | U{1..20} | U(0,1, 1,0) | U(−2, 4) | rút lại mỗi episode, ẩn |

### 3. Baseline

**T:best@REG.** Heuristic tune lại trên 30 seed train cho từng regime, dùng họ A5 đầy đủ.
- K5_la: `fixed_avif_20` (−83,3 trên seed train).
- K5_la_rand: `fixed_mbt2018-mean_q1` (−124,2).
- Họ luật chọn kênh SCH cũng chọn luật không dùng chung kênh.
- Mode thích ứng (họ budget) thua mode cố định: −94,7 so với −83,3 ở K5_la.

**T:reuse@K5_la_rand.** Heuristic dùng chung kênh tốt nhất: `pacedB2+budget0.5+reuse`.

**Cổng NACK (`+gate`)** là baseline lùi theo feedback mà phản biện yêu cầu.
- Mỗi UAV theo dõi tỉ lệ NACK trên block dùng chung của chính nó (10 block gần nhất, cần ít nhất 3).
- Khi tỉ lệ vượt θ, UAV bị đánh dấu trong `cool` slot. Kênh có người bị đánh dấu trở thành kênh riêng: UAV gấp nhất giữ
  kênh, những UAV khác chuyển sang kênh trống hoặc chờ.
- Hết `cool` thì thử lại.
- θ và cool được tune trên K5_la_rand (9 cặp, 30 seed train): chọn θ = 0,4, cool = 30. Cổng chỉ dùng thông tin
  ACK/NACK nhân quả.

**C:K5_mcsg (+gate).** Scheduler C-only của A-X2b, train trên MCS lý tưởng, dùng với cổng NACK. Đây chính là "scheduler
hiện tại + quy tắc ngừng dùng chung kênh khi NACK tăng".

### 4. Train (bước B)

C-only giống A-X2: PPO chọn kênh, còn di chuyển, công suất và mode khoá theo T:best của regime train. 2M bước, 3 seed,
chọn checkpoint theo val. Hai nhóm:
- `K5_la_rand`: ngẫu nhiên hoá theo episode. Đây là nhóm chính.
- `K5_la`: điều kiện danh nghĩa, làm đối chứng.

Quan sát thêm cho mỗi UAV 7 đặc trưng nhân quả:
- SINR báo cáo gần nhất và tuổi báo cáo
- INR ước lượng
- offset OLLA
- tỉ lệ NACK trên 10 và 50 block
- tỉ lệ NACK trên block dùng chung

Không đưa σ, D, b hay bước OLLA vào quan sát.

### 5. Eval

- 100 seed môi trường mới (8800–8899), test pool, so sánh theo cặp.
- 9 policy × 7 điều kiện.
- Phân tích thích ứng: tương quan Spearman giữa tỉ lệ thời gian phát có dùng chung kênh của từng episode và tham số ẩn
  của episode đó trên K5_la_rand; và tỉ lệ dùng chung kênh theo điều kiện cố định.

## Kết quả

### Return (cao hơn là tốt; số học được là trung bình 3 seed train)

| Policy | good | nom | l50 | noisy | slow | bad | **rand** |
|---|---|---|---|---|---|---|---|
| T:best@K5_la (avif_20, không dùng chung kênh) | −55,9 | **−64,9** | **−64,9** | **−73,0** | −138,4 | −119,7 | −113,6 |
| T:best@K5_la_rand (mbt q1, không dùng chung kênh) | −68,2 | −74,3 | −74,3 | −78,7 | **−121,8** | **−108,0** | **−103,9** |
| T:reuse@K5_la_rand | −102,3 | −107,5 | −98,8 | −103,0 | −151,2 | −141,6 | −134,4 |
| T:reuse@K5_la_rand +gate | −89,2 | −99,0 | −97,4 | −100,5 | −146,2 | −138,3 | −131,4 |
| C:K5_mcsg (scheduler cũ) | −155,8 | −176,3 | −171,7 | −182,6 | −273,5 | −256,6 | −243,5 |
| C:K5_mcsg +gate | **−53,7** | −89,7 | −85,2 | −114,5 | −220,1 | −197,4 | −182,9 |
| C:K5_la (train danh nghĩa) | −57,0 | −66,3 | −66,2 | −75,1 | −153,0 | −138,5 | −126,3 |
| C:K5_la_rand (train ngẫu nhiên hoá) | −100,7 | −108,0 | −95,2 | −102,3 | −173,3 | −156,6 | −146,8 |
| C:K5_la_rand +gate | −71,3 | −78,4 | −77,2 | −80,9 | −133,9 | −120,0 | −114,4 |

### So sánh theo cặp (Δ return, CI 95% trên 100 seed môi trường; trong ngoặc là theo từng seed train)

**Scheduler học được so với heuristic dùng cùng mode.** Phép so này tách riêng phần chọn kênh.

| So sánh | good | nom | slow | bad | rand |
|---|---|---|---|---|---|
| C:K5_la − T:best@K5_la | −1,09 [−2,47, +0,30] | −1,37 [−3,16, +0,43] | −14,63 [−18,98, −10,28] | −18,77 [−22,24, −15,29] | −12,63 [−16,03, −9,23] |
| C:K5_la_rand − T:best@K5_la_rand | −32,46 | −33,67 | −51,48 | −48,57 | **−42,91 [−46,88, −38,94]** (−43,1 / −43,1 / −42,5) |
| C:K5_la_rand+gate − T:best@K5_la_rand | −3,01 [−4,29, −1,73] | −4,08 | −12,14 | −11,96 | −10,45 [−13,25, −7,65] |
| C:K5_mcsg+gate − T:best@K5_la | **+2,22 [−3,40, +7,85]** (+6,2 / −5,5 / +5,9) | −24,77 | −81,72 | −77,70 | −69,31 |

**Tác dụng của cổng NACK** (cùng policy, có cổng trừ không có cổng):
- C:K5_la_rand: +18 đến +39, có ý nghĩa ở mọi điều kiện.
- C:K5_mcsg: +53 đến +102.
- T:reuse: +1 đến +13.

### Chỉ số link và hệ thống (trên K5_la_rand)

| Policy | ok | BLER | BLER block dùng chung | Thời gian phát có dùng chung kênh | Goodput (kbit/s) | Năng lượng (kJ) | Jain | UAV kém nhất ok |
|---|---|---|---|---|---|---|---|---|
| T:best@K5_la_rand | 0,703 | 0,208 | – | 0,000 | 222,6 | 67,4 | 0,994 | 0,632 |
| T:reuse@K5_la_rand | 0,632 | 0,333 | 0,355 | 0,915 | 72,4 | 74,3 | 0,976 | 0,497 |
| C:K5_mcsg +gate | 0,542 | 0,194 | 0,584 | 0,107 | 249,9 | 70,1 | 0,906 | 0,327 |
| C:K5_la | 0,637 | 0,216 | 0,399 | 0,152 | 214,3 | 68,0 | 0,951 | 0,449 |
| C:K5_la_rand | 0,606 | 0,324 | 0,357 | 0,863 | 80,3 | 67,9 | 0,952 | 0,424 |
| C:K5_la_rand +gate | 0,677 | 0,215 | 0,584 | 0,113 | 191,7 | 67,7 | 0,987 | 0,575 |

Các chỉ số khác:
- **Phần thời gian phát bị hỏng (overhead phát lại ARQ)** gần bằng BLER: 0,20–0,33.
- **Độ trễ quyết định** của scheduler học được: p99 3,1–3,5 ms mỗi slot, kể cả khi có cổng.

### Thích ứng

**Tỉ lệ thời gian phát có dùng chung kênh theo điều kiện cố định (good → bad):**

| Policy | good | nom | noisy | slow | bad |
|---|---|---|---|---|---|
| C:K5_la_rand | 0,853 | 0,862 | 0,861 | 0,867 | 0,848 |
| C:K5_la_rand +gate | 0,178 | 0,167 | 0,165 | 0,111 | 0,105 |
| C:K5_la | 0,037 | 0,050 | 0,072 | 0,193 | 0,223 |
| C:K5_mcsg +gate | 0,171 | 0,165 | 0,159 | 0,105 | 0,097 |
| T:reuse@K5_la_rand | 0,856 | 0,870 | 0,862 | 0,941 | 0,922 |
| T:reuse +gate | 0,201 | 0,187 | 0,170 | 0,118 | 0,111 |

**Spearman** giữa tỉ lệ dùng chung kênh của từng episode và tham số ẩn, trên K5_la_rand, n = 100 × 3:
- **C:K5_la_rand:** σ −0,15 (p = 0,008), D +0,03, b −0,08. Gần như không phản ứng với điều kiện.
- **C:K5_la_rand +gate:** D −0,28 (p < 0,001). Phần thích ứng có được là do cổng.
- **C:K5_la:** D +0,37, b +0,27. Thích ứng **ngược chiều**: feedback càng chậm, CQI càng lạc quan thì càng dùng chung
  kênh nhiều. Có lẽ vì đây là trạng thái nó chưa từng gặp khi train.

### Train

- **Nhóm K5_la:** val return cải thiện đều, từ −123 lên −33/−34/−35, hội tụ khoảng 1,4–2M bước.
- **Nhóm K5_la_rand:** từ −112 lên khoảng −95 ở 400k bước rồi đi ngang tới 2M. Checkpoint tốt nhất −90,3 / −95,7 /
  −90,5. Cả 3 seed kẹt ở cùng một mức.

Val không so được trực tiếp với test vì khác pool ảnh. Ở test, heuristic không dùng chung kênh tốt hơn policy này 43
điểm.

## Diễn giải

1. **Dùng chung kênh có lợi ở A-X2b chỉ vì link adaptation biết trước nhiễu.**
   - Khi CQI phải đo từ block trước, nhiễu đồng kênh lại bật/tắt theo hàng đợi của UAV kia. Phép đo không dự đoán được
     block tiếp theo, và OLLA một offset mỗi UAV không sửa được SINR có hai trạng thái.
   - Hệ quả: block dùng chung hỏng khoảng 30% ngay cả với feedback 1 block. Có cổng thì còn 34–71%, vì những block dùng chung sót lại rơi vào lúc cổng đang thử lại. Goodput của luật
     dùng chung kênh chỉ còn khoảng 1/3.
   - Đây là kết quả về mô hình, không phải về RL. Nó khớp với thang λ của A-X2b.
2. **Train ngẫu nhiên hoá thất bại, không chỉ là chưa thích ứng.**
   - Policy kẹt ở nghiệm dùng chung kênh khoảng 0,86, trong khi nghiệm tốt nhất ở mọi điều kiện là không dùng chung
     kênh. Cùng kiến trúc ở điều kiện danh nghĩa lại tìm ra nghiệm gần như không dùng chung kênh.
   - Giả thuyết chưa kiểm chứng: action phân tách theo UAV khiến việc "mỗi kênh chỉ một UAV" trở thành bài toán phối
     hợp. Ngẫu nhiên hoá làm return nhiễu thêm nên gradient không đủ rõ để thoát. Đây chính là điều bước C (PPO chọn
     kênh tuần tự có mask) kiểm tra được.
3. **Cổng NACK là một cơ chế lùi lại tốt nhưng không biến dùng chung kênh thành có lợi.**
   - Có cổng, mọi policy hội tụ về khoảng 0,1–0,2 thời gian dùng chung kênh và giảm khi feedback chậm. Đó đúng là kiểu
     thích ứng phản biện muốn thấy, nhưng đến từ luật, không đến từ học.
   - Phần dùng chung kênh còn sót lại vẫn kéo return xuống so với không dùng chung chút nào. Riêng ở điều kiện good,
     C:K5_mcsg+gate ngang hoặc hơn nhẹ (+2,2, n.s.).
4. **Mode controller.**
   - Ở cả hai regime LA, heuristic tune đều chọn **mode cố định**. Các luật mode thích ứng (budget/greedy) thua vì dự đoán
     tốc độ của chúng dùng "CQI lý tưởng × 0,9", lạc quan so với hiệu suất thực 0,37–0,94 (bước A).
   - Chênh lệch giữa hai T:best (avif_20 so với mbt q1) từ −17 đến +12 điểm tuỳ điều kiện. Lớn hơn mọi lợi ích chọn kênh
     ở đây.
   - Như vậy vế "không làm hỏng mode controller thích ứng" **chưa trả lời được**. Pilot này khoá mode theo T:best, chưa
     chạy PPO mode hay MPC trên LA. Muốn trả lời thì bộ dự đoán tốc độ của mode controller phải được hiệu chỉnh theo
     goodput thực của LA, giống bài học hiệu chỉnh ở A-E5.

## Audit sau phản biện #40 (18/11)

ChatGPT chỉ ra một chỗ không nhất quán trong `env_la._loop_once`:
- Kết quả ACK/NACK của block xét theo SINR xấu nhất trong block.
- Nhưng bit gửi ở phần đầu block, trước khi nhiễu xuất hiện, đã được cộng vào ảnh và **không bị hoàn lại** khi block
  bị lật sang NACK.

Đo bằng bộ đếm mới (`bits_in_nack_frac`, `tasks_done_in_nack_blocks`; 10 seed, heuristic của K5_la_rand):

| Điều kiện | Luật | Bit được tính trong block NACK / mọi bit được tính | Ảnh xong trong block NACK |
|---|---|---|---|
| K5_la | không dùng chung kênh | 0,000 | 0 / 260 |
| K5_la | dùng chung kênh | 0,004 | 1,2 / 260 |
| K5_la_good | dùng chung kênh | 0,005 | 0,9 / 260 |
| K5_la_bad | dùng chung kênh | 0,003 | 1,1 / 260 |

Lỗi có thật nhưng nhỏ. Nó chỉ xảy ra ở block dùng chung kênh và làm các policy dùng chung kênh được lợi. Vì vậy các
kết luận ở trên (không dùng chung kênh tốt nhất; scheduler học được thua) không đổi, và nếu có thay đổi thì chỉ mạnh
thêm. Bản sửa (hoàn bit của block NACK) sẽ đi cùng phần audit PHY của phản biện #40. Kết quả trong bảng trên là của
code chưa sửa.

## Giới hạn

- Mới có ARQ, chưa có HARQ có kết hợp. Theo hướng dẫn: chưa có shadowing tương quan, chưa có LoS động.
- LA chỉ dựa trên phép đo của chính UAV, không biết scheduler xếp ai chung kênh.
  - Một BS tập trung thật biết mình ghép cặp ai, nên có thể dự đoán nhiễu từ CQI trễ của cả link mong muốn lẫn link
    nhiễu. Mô hình hiện tại chưa có trường hợp trung gian "LA biết lịch" này, chỉ có LA lý tưởng (mcsg) và LA mù lịch
    (la).
  - Đây có thể là chỗ duy nhất dùng chung kênh còn có lợi.
- Lỗi block xét theo SINR xấu nhất trong block. Cách này bi quan khi nhiễu chỉ xuất hiện một phần block.
- 2M bước, 3 seed, một kiến trúc (factorised MaskablePPO). Chưa chạy bước C.
- Chênh lệch giữa val và test: val tốt hơn test khoảng 30–55 điểm ở cả hai nhóm. Pool ảnh khác nhau; xếp hạng không
  đổi.

## Đề xuất cho phản biện #40 (chưa chạy)

1. **Bước C như phản biện #39:** PPO chọn kênh tuần tự có mask, so với factorised, cùng ngân sách, trên K5_la_rand. Mục
   tiêu cụ thể là xem cách chọn kênh tuần tự có thoát được nghiệm dùng chung kênh khoảng 0,86 không, và có học được
   "lùi lại" mà không cần cổng không.
2. **LA biết lịch (scheduler-aware LA)** làm ablation PHY trước khi tốn thêm công cho RL. BS dùng CQI trễ của link nhiễu
   và quyết định ghép cặp của chính nó để tính SINR dự kiến. Nếu dùng chung kênh vẫn không có lợi khi LA biết lịch thì
   nhánh "học dùng chung kênh" trong env này nên dừng: không có gì để học.
3. **Hiệu chỉnh bộ dự đoán tốc độ cho mode controller theo goodput LA**, rồi mới chạy lại câu hỏi về mode controller.
4. **HARQ có kết hợp** làm ablation riêng, như phản biện đã nêu.

Không viết lại Method.
