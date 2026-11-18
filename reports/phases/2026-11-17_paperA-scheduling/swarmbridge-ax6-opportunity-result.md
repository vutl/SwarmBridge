# A-X6 (phản biện #41): sửa PHY, link adaptation biết lịch, và có cơ hội lập lịch không

Ngày chạy: 2026-11-18. Không train RL; các checkpoint cũ chỉ được chạy eval. Code:
- `env_la.py` (thêm ghi bit theo cả block, `la_mode`, `la_interf`, `la_fade`)
- `ja_sched.py`
- `ax6_eval.py`, `ax6_summary.py`
- `test_env_la_atomic.py` (13 test mới; tổng 28 test đều pass)

Kết quả ở `runs/uav_semcom/ax6/`: `invariance.json.gz`, `tune_*.json`, `eval.json.gz`, `eval_mcsg.json.gz`,
`mcsg_vs_ax2b.json`, `summary.json`, `summary.txt`. Tiếp nối [A-X5](swarmbridge-ax5-la-result.md).

Câu hỏi của phản biện: *với PHY mô hình hoá nhất quán và thông tin mà BS thật sự có, dùng chung kênh còn tạo ra lợi ích
đáng kể không?*

## Trả lời ngắn

1. **Sửa lỗi tính bit không đổi kết luận A-X5.**
   - Code cũ chạy lại cho ra đúng từng giá trị return đã lưu (chênh lệch lớn nhất 0).
   - Code đã sửa làm return của mỗi policy thay đổi tối đa khoảng 2 điểm.
   - Thứ hạng và mọi so sánh chính của A-X5 giữ nguyên.
2. **Cơ hội lớn nhất không phải dùng chung kênh mà là chọn UAV nào được phát.**
   - Một heuristic chọn tổ hợp kênh, chỉ dùng mô hình kênh thống kê (nhân quả) và **không dùng chung kênh**, hơn luật
     least-slack đã tune +8,18 [+6,13, +10,22] ở điều kiện danh nghĩa.
   - Trên K5_mcsg, chính heuristic đó hơn T:best +12,36. Như vậy nó ngang scheduler RL của A-X2b (+11,28; hiệu +1,09,
     CI chứa 0).
   - **Phần lớn lợi ích +11,3 của RL ở A-X2b đến từ việc chọn tổ hợp UAV, không phải từ dùng chung kênh.** ChatGPT đã
     đúng.
3. **Dùng chung kênh chỉ có lợi nhỏ, và phụ thuộc giả định.**
   - Với link adaptation mù lịch, dùng chung kênh luôn hại.
   - Với link adaptation biết lịch (nhân quả), phần lợi riêng của dùng chung kênh nằm trong khoảng −5 đến +6 và đổi dấu
     theo giả định nhiễu, độ trễ feedback và thang thời gian fading. Chỉ có ý nghĩa thống kê ở 2/8 cấu hình: +3,0
     (feedback tốt, nhiễu liên tục) và +5,9 (fading mỗi block, nhiễu chỉ khi phát).
   - Với link adaptation oracle (cận trên), lợi ích là +5,5 / +6,6.
   - Với MCS lý tưởng (K5_mcsg) là +3,2.
4. **Phần lớn cơ hội nằm ở thông tin mà BS không thể có trong simulator hiện tại.**
   - Biết trước fading và trạng thái LoS của slot tới cho thêm +12,5 (danh nghĩa) và +61 (feedback chậm).
   - BS không thể có thông tin này một cách nhân quả, vì simulator rút **cả fading lẫn trạng thái LoS độc lập ở mỗi slot
     1 s**. Đây là giả định mô hình quyết định cơ hội học ở đây.

## Luồng làm

### 1. Sửa cách tính bit theo block (`la_atomic`, mặc định bật)

**Quy tắc mới:** một block là một codeword.
- Bit gửi trong block chỉ là tạm tính.
- Một ảnh có bit cuối nằm trong block chỉ hoàn thành khi block được ACK. Thời điểm hoàn thành là lúc gửi bit cuối,
  như trước.
- Khi block thành NACK, kể cả do nhiễu xuất hiện giữa block:
  - mọi bit gửi trong block được trả lại cho ảnh;
  - các ảnh đã "xong" trong block quay lại đầu hàng đợi để phát lại;
  - nếu hạn chót đã qua thì ảnh bị bỏ, đúng một lần.
- Cuối block là một sự kiện, nên kết quả được chốt ngay.

**Test.** Có test cho:
- hoàn lại bit và thứ tự hàng đợi;
- nhiễu xuất hiện giữa block (không ảnh nào xong hai lần, không bit nào bị đếm đôi);
- ảnh chỉ hoàn thành sau khi block của nó đóng;
- ảnh mở lại sau hạn chót bị bỏ đúng một lần;
- policy không dùng chung kênh không bị ảnh hưởng;
- `la_atomic=False` tái tạo đúng dữ liệu A-X5.

**Kiểm tra bất biến (9 policy của A-X5 × 3 điều kiện × 100 seed, ghép cặp):**
- `la_atomic=False` tái tạo **đúng từng giá trị** return đã lưu (chênh lệch lớn nhất 0).
- Bật sửa lỗi:
  - policy không dùng chung kênh không đổi (chênh 0,00);
  - policy dùng chung kênh đổi từ −1,98 đến +0,37;
  - mỗi episode có 0–1,6 ảnh phải phát lại; bit bị hoàn lại chiếm ≤ 0,7 %.
- So sánh chính của A-X5 với T:best@K5_la_rand trên K5_la_rand, cũ → mới:

| Policy | Code cũ | Code đã sửa |
|---|---|---|
| C:K5_la_rand | −42,91 | −43,49 |
| C:K5_la | −22,35 | −22,16 |
| C:K5_la_rand+gate | −10,45 | −10,51 |
| T:reuse | −30,50 | −30,52 |

Ở K5_la_good, scheduler cũ + cổng NACK đổi từ +14,57 thành +13,29. Return, ok, goodput và tỉ lệ trễ hạn đều đổi rất ít.

### 2. Link adaptation: ba chế độ (`la_mode`)

- **blind** (như A-X5): dùng SINR báo cáo trễ của chính UAV, trong đó đã lẫn nhiễu của block cũ; một offset OLLA.
- **sched**, biết lịch, **nhân quả**:
  - BS biết lịch cấp kênh của chính nó và biết UAV đồng kênh nào đang phát (qua grant và buffer report).
  - BS ước lượng SINR = p_k g_k / (N + Σ p_j g_j) từ báo cáo SNR/W **trễ** của từng UAV, với cùng độ trễ, độ lệch và
    nhiễu như blind.
  - Mỗi UAV có hai offset OLLA: một cho block dự đoán là dùng chung, một cho block dự đoán là phát một mình.
  - Không dự đoán được UAV nhiễu bắt đầu phát sau khi block đã mở.
- **oracle**: **chỉ để làm cận trên, không triển khai được.** Dùng SINR thật ở đầu block và chọn MCS tối đa goodput kỳ
  vọng.

### 3. Hai giả định nhiễu (`la_interf`)

Đây là **giả định thí nghiệm**, không phải mô tả hành vi LTE/NR. UE có grant nhưng buffer rỗng sẽ phát padding hay bỏ
qua là tuỳ thủ tục MAC và cấu hình.

- **bursty:** UAV đồng kênh chỉ gây nhiễu khi đang thực sự gửi dữ liệu ảnh.
- **persistent:** mọi UAV đồng kênh được cấp lịch, còn hoạt động và còn ảnh chưa giao (kể cả đang mã hoá) giữ grant cả
  slot và phát padding khi chưa có dữ liệu sẵn. Padding gây nhiễu và tốn năng lượng (`pad_air_s`).

### 4. Thang thời gian (`la_fade`) và cách chúng tương tác

- **Mặc định, "slot":** fading nhỏ (small-scale) **và trạng thái LoS/NLoS** được rút độc lập mỗi slot 1 s. Trong một slot,
  10 block 100 ms có cùng kênh.
  - CQI trễ D block: D + 1 block đầu mỗi slot dùng báo cáo của slot trước, mà slot trước không nói gì về kênh hiện tại.
    Với D ≥ 10, mọi báo cáo đều từ slot trước.
  - ACK/NACK cập nhật OLLA ngay khi block đóng, chưa có độ trễ vòng HARQ.
  - Hệ quả cho bộ lập lịch, vốn quyết định ở đầu slot: **thông tin nhân quả duy nhất về slot tới là thống kê** (vị trí,
    xác suất LoS, path loss).
  - Một biến thể dùng báo cáo trễ để chọn UAV (`info=report`) làm UAV bị bỏ đói: báo cáo xấu không bao giờ được đo lại.
    Thử trên 3 seed, 25–37 % ảnh bị bỏ. Biến thể này được giữ làm kết quả âm, không dùng.
- **"block"** (độ nhạy): fading nhỏ rút lại mỗi block, trạng thái LoS vẫn theo slot.

### 5. Các policy (cùng tải và reward)

Mọi policy dùng di chuyển, công suất và mode của T:best@K5_la (`fixed_avif_20`); chỉ việc gán kênh khác nhau.

| Policy | Gán kênh |
|---|---|
| NR | luật T:best: M UAV có ít slack nhất, không dùng chung kênh |
| RU | NR + mọi UAV còn lại dùng chung kênh tải ít nhất |
| RUG | RU + cổng NACK |
| JA-NR-c / JA-R-c | **gán tổ hợp (joint assignment)**, nhân quả: duyệt cả 243 cách gán (5 UAV × {nghỉ, kênh 1, kênh 2}) |
| JA-NR-o / JA-R-o | như trên, nhưng biết fading và LoS thật của slot tới. **Cận trên, không triển khai được** |

Chi tiết của JA:
- Mục tiêu tối đa Σ_k w_k · min(R̂_k · dt, B_k). w_k = (1/slack)^α. R̂_k là tốc độ dự đoán (CQI của SINR dự đoán, trừ
  back-off m dB nếu dùng chung kênh). Nhiễu ước lượng kiểu mean-field theo mức hoạt động của UAV đồng kênh.
- Bản nhân quả (c) lấy trung bình qua 32 lần rút từ mô hình kênh thống kê ở vị trí hiện tại.
- Hậu tố NR = cấm dùng chung kênh; R = cho phép.
- Họ tham số α ∈ {0, 1, 2} × m ∈ {0, 3, 6} được tune trên 30 seed train cho **từng** cấu hình. Kết quả tune ở
  `tune_*.json`.
- JA oracle không phải tối ưu toàn cục: nó tham lam theo từng slot, mục tiêu cố định, nhiễu ước lượng mean-field.

**Eval:** 100 seed mới (8800–8899), ghép cặp. Cấu hình: điều kiện {nom, good, slow, blockfade} × link adaptation {blind,
sched, oracle} × nhiễu {bursty, persistent} (12 cấu hình), cộng K5_mcsg.

## Kết quả

### Phân rã cơ hội (Δ return, CI 95 % trên 100 seed)

| Cấu hình | Lợi ích chọn UAV, nhân quả (JA-NR-c − NR) | Lợi ích dùng chung kênh, nhân quả (JA-R-c − JA-NR-c) | Lợi ích thông tin (JA-NR-o − JA-NR-c) | Dùng chung kênh khi biết trước kênh (JA-R-o − JA-NR-o) | RUG − NR |
|---|---|---|---|---|---|
| nom, blind, bursty | +8,18 [6,13, 10,22] | **−25,69** | +12,49 | −14,97 | −8,56 |
| nom, blind, persistent | +8,18 | −0,16 (gần như không dùng chung) | +12,49 | +0,00 | −8,20 |
| nom, sched, bursty | +8,18 | +0,17 [−1,72, 2,05] | +12,49 | −5,57 | +2,25 [0,62, 3,89] |
| nom, sched, persistent | +8,18 | +1,59 [−0,04, 3,21] | +12,49 | +0,01 | +2,32 [0,78, 3,87] |
| good, sched, bursty | +6,13 [4,75, 7,51] | +0,11 | +9,03 | −3,10 | +1,50 [0,17, 2,83] |
| good, sched, persistent | +6,13 | **+3,00 [1,99, 4,01]** | +9,03 | −0,01 | +0,49 |
| slow, sched, bursty | +2,96 [−1,84, 7,75] | +3,05 [−2,08, 8,19] | **+61,49** | −2,92 | +2,86 |
| slow, sched, persistent | +2,96 | +4,06 [−1,09, 9,21] | +61,49 | +2,71 [0,79, 4,64] | +4,74 [0,27, 9,20] |
| blockfade, sched, bursty | +6,88 [4,16, 9,60] | **+5,94 [3,50, 8,38]** | +2,44 | −1,73 | +4,05 |
| blockfade, sched, persistent | +6,88 | **−5,13 [−7,72, −2,54]** | +2,44 | +3,16 | +3,42 |
| nom, **oracle LA**, bursty | +4,20 | **+6,57 [6,22, 6,92]** | +2,07 | +1,90 | +1,32 |
| nom, **oracle LA**, persistent | +4,20 | **+5,47 [5,14, 5,81]** | +2,07 | +1,10 | +1,22 |
| **K5_mcsg** (MCS lý tưởng, A-X2b) | **+12,36 [11,15, 13,58]** | +3,16 [1,96, 4,35] | +12,04 | +0,60 | – |

Thêm:
- **RU** (dùng chung kênh không kiểm soát) thua NR 44–81 điểm ở mọi cấu hình có LA không oracle; với LA oracle thì hơn
  +10,4.
- **So với RL của A-X2b trên K5_mcsg** (cùng seed, trung bình 3 checkpoint đã lưu):
  - JA-NR-c +1,09 [−0,36, +2,54];
  - JA-R-c **+4,24 [+3,15, +5,33]**;
  - JA oracle +13,1 / +13,7.

### UAV có trạng thái nào mà dùng chung kênh có lợi không

Đo bằng tỉ lệ slot có tranh chấp (nhiều UAV có dữ liệu hơn số kênh) mà JA cho phép dùng chung kênh **chọn** dùng chung,
theo bộ dự đoán của chính nó:

| Cấu hình | Nhân quả | Oracle |
|---|---|---|
| nom, sched, persistent | 0,32 | 0,01 |
| good, sched, persistent | 0,34 | 0,01 |
| nom, sched, bursty | 0,46 | 0,72 |
| LA oracle | 0,94–1,00 | 0,43–0,84 |

- Khi nhiễu liên tục (dễ dự đoán) và biết trước kênh, dùng chung kênh gần như không bao giờ đáng: 1 % số slot.
- Khi nhiễu chỉ xuất hiện lúc phát, bộ dự đoán mean-field chọn dùng chung kênh nhiều nhưng return lại giảm (−5,6). Bộ dự
  đoán lạc quan ở đây.
- Kết luận: tỉ lệ này phụ thuộc mạnh vào mô hình dự đoán, nên chỉ là chỉ báo phụ. Chỉ báo chính là các cột Δ return ở
  bảng trên.

### Chỉ số khác

- **Ở nom, sched, persistent:**
  - ok 0,752 (NR) → 0,761 (JA-NR-c) → 0,767 (oracle);
  - drop 0,021 → 0,008 → 0,000;
  - năng lượng 66,2 → 66,0 → 65,2 kJ;
  - Jain ≥ 0,995;
  - UAV kém nhất ok 0,685 → 0,695 → 0,704.
- **Padding (persistent):** 1–20 s airtime mỗi episode, năng lượng không đổi đáng kể.
- **Block dùng chung không dự đoán được** (sched, bursty): ≤ 2 % số block dùng chung.
- **BLER của block dùng chung** (policy dùng chung có kiểm soát): sched 0,24–0,43 tuỳ policy và độ trễ; blind
  0,39–0,43; oracle LA 0,08–0,31.

## Diễn giải

1. **Sửa lỗi block giữ nguyên kết luận A-X5.** Đã kiểm bằng chạy lại ghép cặp, không chỉ đếm bit.
2. **"No-reuse tốt nhất" ở A-X5 chỉ đúng trong tập luật đã thử.**
   - Luật least-slack đã tune không phải scheduler không-dùng-chung-kênh tốt nhất: chỉ cần chọn UAV theo kênh thống kê và
     backlog là đã hơn +6…+8 (nhân quả).
   - Đây là điều phản biện #41 lưu ý, và nó đúng.
3. **Kết quả A-X2b phải đọc lại.**
   - Scheduler RL +11,3 trên K5_mcsg ngang một heuristic không dùng chung kênh. Phần dùng chung kênh chỉ thêm +3,2 (và
     JA-R-c hơn RL +4,2).
   - Diễn giải trước của tôi ("RL thắng khi heuristic dùng mô hình tốc độ sai") không đúng cho A-X2b: ở đó RL thắng vì
     họ heuristic đã tune quá yếu.
4. **Dùng chung kênh dưới PHY thực tế và thông tin nhân quả: lợi ích nhỏ, không ổn định.**
   - Cần link adaptation biết lịch; mù lịch thì luôn hại.
   - Có lịch thì dao động −5…+6 tuỳ giả định, có ý nghĩa ở 2/8 cấu hình.
   - Cổng NACK đơn giản (RUG) cho +1,5…+4,7 dưới link adaptation biết lịch, ngang hoặc hơn JA-R.
   - **Không đủ cơ sở để train RL riêng cho việc dùng chung kênh.**
5. **Cơ hội lớn còn lại (+12,5, tới +61 khi feedback chậm) là thông tin kênh.**
   - Với fading và LoS rút độc lập mỗi slot, bộ lập lịch nhân quả không thể có thông tin này.
   - Vật lý thật thì khác: UAV bay khoảng 20 m mỗi slot, nên trạng thái LoS/che khuất có tương quan theo không gian (vài
     chục mét), còn fading nhỏ thì không (Doppler ~160 Hz).
   - Như vậy **trạng thái LoS tương quan theo thời gian, vốn bị hoãn ở #39, chính là giả định quyết định có gì để học
     không.** Nếu LoS tương quan, một phần của +12,5 trở nên nhân quả. Khi đó một scheduler học (hoặc lọc) được trạng
     thái kênh từ ACK/CQI mới có lý do tồn tại.

## Giới hạn

- JA là một họ heuristic tham lam theo slot, nhiễu ước lượng mean-field, tune trên 30 seed train.
  - Riêng nom/blind/bursty: thành viên R tốt nhất trên train vẫn thua NR, nhưng bảng báo cáo nó như một họ "cho phép
    dùng chung kênh".
  - Cặp blockfade cho kết quả trái dấu (+5,9 / −5,1), cho thấy việc chọn thành viên khi tune còn nhiễu.
- Một tải duy nhất (mode và di chuyển của T:best@K5_la; K5_mcsg dùng T:best@K5_mcsg).
- Chưa có HARQ kết hợp. Chưa có độ trễ vòng ACK. LoS chưa tương quan.
- A-X2 (K5 Shannon / ref, +2,5 / +3,1) chưa được kiểm lại theo cách này.
- Bộ điều khiển mode chưa được hiệu chỉnh goodput dưới LA (mục 6 của phản biện #40), còn mở.

## Đề xuất cho phản biện #42 (chưa chạy)

1. **Không train RL cho dùng chung kênh** với PHY hiện tại.
2. **Thêm trạng thái LoS / shadowing tương quan theo quãng đường bay** (Gudmundson, khoảng cách tương quan cỡ vài chục
   mét), giữ fading nhỏ độc lập.
   - Đo lại "lợi ích thông tin": bao nhiêu trong +12,5 trở thành nhân quả khi BS lọc trạng thái kênh từ CQI và ACK.
   - Chỉ khi phần đó lớn mới có lý do cho một scheduler học (RL hoặc lọc Bayes + JA).
3. **Đặt JA-NR-c / JA-R-c làm baseline lập lịch mạnh** cho mọi so sánh RL về sau, thay cho T:best / SCH.
4. **Kiểm lại A-X2 (K5, ref) bằng JA** để biết +2,5 / +3,1 của RL ở đó là chọn UAV hay dùng chung kênh.
5. **Hiệu chỉnh goodput cho bộ điều khiển mode dưới LA** (câu hỏi Paper A còn mở).

Không viết lại Method.
