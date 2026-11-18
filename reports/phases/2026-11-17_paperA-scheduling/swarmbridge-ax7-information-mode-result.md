# A-X7 (phản biện #42): tách cơ hội thông tin, kênh tương quan, và quay lại bộ điều khiển mode

Ngày chạy: 2026-11-18. Không train RL; các checkpoint cũ (PPO mode-only của K5) chỉ chạy eval.

**Code:**
- `env_la.py`: LoS tương quan theo không gian, shadowing, probe SRS, `slot_los`.
- `ch_filter.py`: bộ lọc LoS Bayes nhân quả.
- `ja_sched.py`: thêm các mức thông tin `los` / `filt`, số mẫu Monte Carlo, latency, tốc độ theo Shannon.
- `ax7_eval.py`, `ax7_summary.py`.
- `ax2_eval.py`: thêm scheduler `ja`, `jae`, `ls` và `--env-extra`.
- `test_ax7.py`: 7 test mới; tổng 35 test đều pass.

**Kết quả** ở `runs/uav_semcom/ax7/`:
- `e1_decomp`, `e2_mc`, `e3_corr`, `e4_ax2` (`.json.gz`)
- `d_mode_nom*.json`, `d_mode_c30.json`, `best_stack.json`
- `tune_*.json`, `summary.json`, `summary.txt`

Tiếp nối [A-X6](swarmbridge-ax6-opportunity-result.md).

## Trả lời ngắn

1. **Khoảng +12,5 của oracle ở A-X6 gần như toàn bộ là fading nhỏ, không phải LoS.**
   - Oracle chỉ biết LoS (không biết fading nhỏ) hơn JA nhân quả −0,8…+1,0, không có ý nghĩa thống kê ở cả 4 cấu hình.
   - Phần fading nhỏ (oracle đầy đủ − oracle LoS) là +8,7…+63,5.
   - LoS ở đây hiếm: P(LoS) khoảng 3 % ở 1–2,5 km và độ cao 100 m, khoảng 10 slot-ứng-viên mỗi episode.
   - Fading nhỏ bị "đóng băng" cả slot 1 s là giả định của simulator. Thực tế ở Doppler khoảng 160 Hz, fading nhỏ không
     dự đoán được ở thang 1 s. Khi fading được rút lại mỗi block (A-X6), khoảng cách oracle chỉ còn +2,4.
   - **Vậy +12,5 không phải cơ hội nhân quả.** ChatGPT đã nghi đúng.
2. **Kênh tương quan theo quãng bay (10 / 30 / 100 m, giữ nguyên xác suất LoS biên) không tạo cơ hội mới.**
   - Bộ lọc nhân quả cải thiện dự đoán (Brier 0,0288 → 0,0238 ở 100 m có SRS) nhưng lợi ích return chỉ −0,5…+0,2,
     không có ý nghĩa thống kê.
   - Chính oracle LoS cũng chỉ +0…+1. Không có gì để lọc.
   - Probe SRS mỗi slot giúp +2,7…+4,2 cho **mọi** policy. Lợi ích đến từ link adaptation có báo cáo mới, không từ việc
     chọn UAV.
3. **A-X2 trên Shannon: JA có dùng chung kênh ngang hoặc hơn RL.**
   - K5: +1,34 [0,60, 2,07]; ref: +0,84 [0,47, 1,22].
   - Ở K5 (Shannon lý tưởng), JA không dùng chung kênh thua 15. Dùng chung kênh thật sự có lợi khi PHY lý tưởng.
   - Cả ba kết quả RL lập lịch (A-X2 K5, ref, A-X2b) đều bị một heuristic mạnh bắt kịp.
4. **JA rẻ và ổn định.**
   - Kết quả không nhạy với số mẫu Monte Carlo (4–128: trong ±1,6).
   - Latency mỗi slot: p50 0,5–1,7 ms, p99 0,7–3,1 ms (128 mẫu, có dùng chung kênh: p99 12 ms). Ngang PPO (p99 3,1–3,5 ms).
5. **Câu hỏi trung tâm của Paper A có bằng chứng dương lần đầu dưới PHY thực tế:**
   - **PPO–MPC hiệu chỉnh online** hơn PPO zero-shot +11,7…+17,1 ở mọi scheduler.
   - **Có tương tác giữa lập lịch và mode, nhưng nhỏ:** +4,3…+4,9 (CI > 0) khi so cùng cách bay.
   - **Tổ hợp JA + PPO–MPC hơn tổ hợp heuristic tốt nhất +13,27 [10,30, 16,24]** (3/3 checkpoint dương).
   - **Lưu ý:** heuristic mode chưa được tune lại cùng JA dưới LA. Đây đúng là lỗi đã làm hỏng A-X2b, nên chưa được
     dùng con số này làm luận điểm trước khi làm đối chứng đó.

## Luồng làm

### A. Tách cơ hội thông tin (PHY A-X6, LoS độc lập mỗi slot)

Các mức thông tin của JA (cùng tải và reward, chỉ khác việc gán kênh; mỗi mức tune riêng trên 30 seed train):

| Mức | Thông tin |
|---|---|
| c, `stat` | **Nhân quả:** mô hình kênh thống kê theo vị trí |
| l, `los` | **Oracle chỉ LoS** (không nhân quả): trạng thái LoS/NLoS thật của slot tới, fading nhỏ không biết |
| o, `oracle` | **Oracle đầy đủ** (không nhân quả): độ lợi kênh thật của slot tới |

Cả hai oracle chỉ là chẩn đoán cho họ policy này, không phải cận trên của tối ưu toàn cục.

| Cấu hình | JA-NR-c − NR | Oracle LoS (l − c) | Oracle đầy đủ (o − c) | Phần fading nhỏ (o − l) |
|---|---|---|---|---|
| nom, sched, persistent | +8,18 [6,13, 10,22] | −0,82 [−2,06, 0,42] | +12,49 | +13,31 [11,39, 15,23] |
| nom, sched, bursty | +8,18 | −0,82 | +12,49 | +13,31 |
| good, sched, persistent | +6,13 [4,75, 7,51] | +0,33 [−0,39, 1,04] | +9,03 | +8,70 |
| slow, sched, persistent | +2,96 [−1,84, 7,75] | −2,03 [−7,82, 3,76] | +61,49 | +63,52 |

- Với bản cho phép dùng chung kênh (R), oracle LoS cho −2,7…+1,0, cũng không có gì.
- **LoS hiếm có chi phối không?** Không. Mỗi episode có khoảng 10 slot-ứng-viên ở trạng thái LoS, mọi episode đều có.
  Lợi ích của oracle LoS không tương quan với số slot LoS (Spearman −0,15…+0,07, p > 0,1). LoS đã quá hiếm (khoảng
  3 %) để mang giá trị lập lịch: một UAV gần như luôn NLoS, và mô hình thống kê đã biết điều đó.

### A'. Độ nhạy Monte Carlo và latency (100 seed)

| JA (nom, sched) | 4 mẫu | 8 | 32 (mặc định) | 128 |
|---|---|---|---|---|
| NR: return so với 32 | +0,13 | −1,57 [−3,18, 0,03] | – | +0,46 |
| NR: latency p50 / p99 (ms) | 0,46 / 0,71 | 0,51 / 0,90 | 0,65 / 1,29 | 1,31 / 2,15 |
| R (persistent): return so với 32 | +0,35 | +0,94 | – | +0,96 |
| R: latency p50 / p99 (ms) | 0,61 / 0,95 | 0,78 / 1,26 | 1,49 / 3,00 | 5,11 / 12,75 |

Return không nhạy với số mẫu. Latency nhỏ, cùng cỡ PPO; thêm vào ngân sách 50 ms của bộ điều khiển mode không đáng kể.

### B. Kênh tương quan theo không gian (`ch_corr_m`)

- **Mô hình:** biến ẩn Gauss AR(1) theo quãng bay (Gudmundson), ρ = exp(−d/d_c). LoS khi Φ(z) < p_los(vị trí), nên
  **xác suất LoS biên ở mọi vị trí giữ nguyên**. d_c = 0 tái tạo đúng env cũ (có test).
- **Kiểm chứng:** P(LoS | LoS ở slot trước) > 0,5 khi d_c = 100 m, so với khoảng 3 % khi độc lập.
- **Fading nhỏ** vẫn độc lập mỗi slot.
- **Sensitivity:** thêm shadowing 4 dB tương quan; cấu hình này đổi phân phối biên.

**Bộ lọc nhân quả (`ch_filter`, `filt`):**
- Hậu nghiệm LoS tại thời điểm báo cáo trễ gần nhất: likelihood Rician so với Rayleigh của độ lợi đo được.
- Đẩy tới slot hiện tại qua copula AR(1) và quãng đã bay; biết d_c.
- Không có báo cáo, hoặc báo cáo đã cũ, thì dự đoán tự trở về prior. **Không có bỏ đói theo thiết kế.**

**Probe:** SRS mỗi slot (`la_srs_period=1`). UAV chưa có đo trong slot vừa qua được đo ở đầu slot. Giả định chi phí
airtime và năng lượng bằng 0.

| d_c | SRS | Bộ lọc (f − c) | Oracle LoS (l − c) | Oracle đầy đủ (o − c) | Brier prior → lọc | Dùng chung trên lọc (R-f − NR-f) |
|---|---|---|---|---|---|---|
| 0 | có | 0 (giống hệt) | +0,18 | +10,82 | 0,0327 → 0,0327 | +1,36 |
| 10 m | không | −0,01 | −0,73 | +12,52 | 0,0312 → 0,0312 | **+2,26 [0,85, 3,67]** |
| 10 m | có | −0,02 | +0,08 | +10,84 | 0,0317 → 0,0313 | +2,16 [0,67, 3,65] |
| 30 m | không | −0,02 | −0,04 | +12,42 | 0,0299 → 0,0296 | −0,30 |
| 30 m | có | +0,06 | +0,18 | +10,25 | 0,0303 → 0,0292 | +1,37 |
| 100 m | không | +0,24 [−0,30, 0,77] | −0,63 | +13,67 | 0,0284 → 0,0264 | +1,86 [0,08, 3,65] |
| 100 m | có | −0,53 | −0,02 | +10,69 | 0,0288 → 0,0238 | +0,04 |
| 30 m + shadow 4 dB | không | −0,42 | +1,06 | **+23,22** | 0,0297 → 0,0296 | **+4,23 [1,82, 6,63]** |

- SRS − không SRS, cùng policy: +2,7…+4,2 (mọi d_c; NR, JA-NR-c, JA-NR-f đều vậy).
- **Shadowing làm khoảng cách oracle tăng lên +23,2**, nhưng bộ lọc hiện tại chỉ theo dõi LoS, không theo dõi shadowing,
  nên không lấy lại được. Đây là chỗ duy nhất còn khả năng có dư địa nhân quả: shadowing có tương quan chừng vài chục
  mét là thực tế, và đo được qua CQI. Kết quả này chưa kiểm.

### E4. Kiểm lại A-X2 (Shannon) bằng JA, ghép cặp với RL đã lưu

| Regime | RL C-only − NR | JA-NR-c − RL | JA-R-c − RL |
|---|---|---|---|
| K5 | +2,52 | −17,82 | **+1,34 [0,60, 2,07]** |
| ref | +3,07 | −1,47 | **+0,84 [0,47, 1,22]** |

NR khớp T:best đã lưu (chênh 0,00).

### D. Bộ điều khiển mode dưới link adaptation thực tế (scheduler × mode)

**Thiết lập:**
- Env: K5_la, LA biết lịch, nhiễu persistent, quan sát giống K5.
- Checkpoint PPO mode-only của K5 (train trên Shannon), chạy zero-shot.
- **PPO–MPC** = bộ điều khiển chọn lọc của A-E8 (cals np-any, ngân sách 50 ms). Hiệu chỉnh goodput **online, cold** từ
  throughput thực mỗi episode. Đây chính là phần "hiệu chỉnh goodput cho bộ điều khiển mode" mà phản biện yêu cầu.
- Scheduler:
  - `tbest_env` = luật least-slack không dùng chung kênh của T:best@K5_la, di chuyển kiểu mission;
  - `jae` = JA, di chuyển mission;
  - `ls` = least-slack, di chuyển paced (hover khi hàng đợi dài) của T:best@K5;
  - `ja` = JA, di chuyển paced.
- Mode:
  - `heur_env` = mode cố định avif_20, tune cho LA;
  - `heur` = luật mode thích ứng của T:best@K5;
  - `ppo`;
  - `ppo+mpc`.

**Lần chạy đầu bị lẫn biến.** Lần đầu `ja` dùng di chuyển paced còn `tbest_env` dùng di chuyển mission, và cho "tương
tác" +19. Lần chạy lại so cùng cách bay; chỉ kết quả cùng cách bay được dùng dưới đây.

| Return (TB 3 ckpt) | heur_env | heur | ppo | ppo+mpc |
|---|---|---|---|---|
| tbest_env (mission) | −64,93 | −77,39 | −77,73 | −60,60 |
| jae (mission) | −56,75 | **−54,41** | −63,91 | −48,14 |
| ls (paced) | −72,11 | −72,40 | −66,62 | −53,29 |
| ja (paced) | −64,84 | −55,24 | −52,88 | **−41,14** |

**Hiệu ứng (CI 95 %, 100 seed):**
- **Mode, ở cùng scheduler:**
  - ppo+mpc − ppo: +11,7…+17,1 ở cả bốn scheduler. Hiệu chỉnh online là bắt buộc; bộ kiểm toán dịch vụ cho thấy tốc độ
    dự đoán trước hiệu chỉnh cao hơn thực tế khoảng 25–30 % (svc 0,69–0,79).
  - ppo zero-shot − mode cố định: −12,8 (tbest_env) đến +12,0 (ja).
- **Scheduler, ở cùng cách bay:**
  - JA − least-slack: +7,3 / +8,2 (mode cố định), +13,7 / +13,8 (ppo), +12,2 / +12,5 (ppo+mpc), +17,2 / +23,0 (heur).
- **Tương tác** (lợi ích của JA dưới ppo+mpc − dưới mode cố định): **+4,88 [2,08, 7,67]** (paced), **+4,28 [0,71,
  7,85]** (mission).
- **Cách bay cũng tương tác:** paced hại khi mode cố định (−7…−8), giúp khi mode là PPO hoặc PPO–MPC (+7…+11), và
  gần như không đổi khi dùng luật mode thích ứng heuristic (+5,0 / −0,8).
- **Tổ hợp tốt nhất so với tổ hợp heuristic tốt nhất** (ja|ppo+mpc − jae|heur): **+13,27 [10,30, 16,24]**, theo
  checkpoint +13,8 / +11,4 / +14,6.
- Với LoS tương quan 30 m, các con số giống hệt về chất (`d_mode_c30`).

## Diễn giải

1. **Lập lịch kênh đơn thuần không còn là chỗ cho RL ở env này.**
   - Mọi lợi ích lập lịch nhân quả mà RL từng có đều bị một heuristic JA bắt kịp hoặc vượt.
   - Phần còn lại trong oracle là fading nhỏ bị đóng băng theo slot, một giả định của simulator.
   - LoS tương quan không mở ra cơ hội, vì LoS quá hiếm ở hình học này.
2. **Chỗ đáng đầu tư là phối hợp lập lịch, cách bay và mode trong điều kiện kênh không chắc chắn.**
   - Mode thích ứng cần hiệu chỉnh tốc độ online: +12…+17 so với PPO zero-shot.
   - Lợi ích của JA lớn hơn khi mode thích ứng (+4–5).
   - Cách bay paced chỉ có lợi khi mode do PPO / PPO–MPC chọn.
   - Đó là luận điểm Paper A (PPO + MPC chọn lọc, hiệu chỉnh online), giờ có thêm trục scheduler.
3. **Chưa được khẳng định "tổ hợp học được hơn heuristic"** cho tới khi họ heuristic mode (budget/greedy/fixed) được tune
   lại **cùng** JA dưới LA. Bài học của A-X2b áp dụng trực tiếp ở đây.

## Giới hạn

- PPO mode-only train trên Shannon K5 và chạy zero-shot; chưa train lại trên LA (theo phản biện).
- Bộ điều khiển mode dùng hiệu chỉnh cold mỗi episode.
- Heuristic mode chưa tune lại cùng JA dưới LA.
- Bộ lọc chỉ theo dõi LoS (không theo dõi shadowing), và biết d_c. SRS được giả định miễn phí.
- LoS hiếm do hình học (1–2,5 km, 100 m). Ở hình học UAV gần BS hơn hoặc bay cao hơn, LoS không còn hiếm, và kết luận ở
  mục B có thể khác.
- Một tải (bộ ảnh / deadline của A5); 100 seed; JA tham lam một slot.

## Đề xuất cho phản biện #43 (chưa chạy)

1. **Đối chứng công bằng cho tổ hợp:** tune họ heuristic mode (budget τ, greedy, fixed) cùng JA và cùng cách bay trên
   seed train dưới LA. Sau đó so JA + PPO–MPC với heuristic tốt nhất đã tune cùng. Nếu +13 còn, mới là bằng chứng cho
   Paper A.
2. **Hiệu chỉnh cho heuristic mode** (cùng bộ hiệu chỉnh online của MPC) làm baseline. Mục đích: tách phần "hiệu chỉnh"
   khỏi phần "PPO + lập kế hoạch".
3. **Tuỳ chọn:** bộ lọc shadowing + JA. Đây là chỗ duy nhất còn khoảng cách oracle lớn (+23) có khả năng nhân quả.
   Chỉ làm nếu phản biện thấy shadowing 4 dB / 30 m hợp lý cho UAV.
4. **Không train RL lập lịch.** Nếu train lại thì chỉ train PPO mode-only trên LA, và chỉ sau mục 1.

Không viết lại Method.
