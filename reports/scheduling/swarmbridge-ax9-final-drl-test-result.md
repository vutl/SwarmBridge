# A-X9 (phản biện #44): phép thử cuối cùng có giới hạn cho giả thuyết DRL

Ngày chạy: 2026-11-18. Protocol được đăng ký trước và commit trước khi chạy bất cứ thứ gì
([protocol](swarmbridge-ax9-protocol.md), commit `7acd8b19`). Cổng quyết định, seed test và quy tắc chọn đều đúng
như protocol; không có gì được chỉnh sau khi xem kết quả test.

**Code:**
- `ax9_run.py` (các giai đoạn sched / heur / train / test), `ax9_summary.py` (cổng quyết định).
- `a5_train.py` thêm `--cfg-extra`, `--slot-move`, `--slot-ja`.
- `ax2_eval.py` thêm `SCHED~MEMBER@MOVE`, `ppola`, `+mpcnc`.
- `a9_mpc.py` có mô hình tốc độ cho PHY "la".
- `test_ax9.py`: 5 test; tổng 45 test đều pass.

**Kết quả** ở `runs/uav_semcom/ax9/`:
- `sched.json`, `heur.json`, `train/ppo_la_s{0,1,2}.json` (đường cong validation), `ppo_la_ckpts.json`;
- `test.json`, `test.deliv.json.gz`, `summary.json`, `summary.txt`.
- Checkpoint PPO không nằm trong git (`/home/uservutl/work/ax9/`).

Tiếp nối [A-X8](swarmbridge-ax8-fair-baseline-result.md).

## Trả lời ngắn

**Cổng không qua: 4/5 điều kiện FAIL → DỪNG nhánh DRL** (theo đúng quy tắc đã đăng ký).

Δ = (PPO-LA + MPC) − (I\* + MPC), 100 seed test mới 8300–8399:

| Điều kiện | Ngưỡng | Đo được | |
|---|---|---|---|
| 1. Δ trung bình | ≥ +2,0 | **−0,14** | FAIL |
| 2. Cận dưới CI 95 % (seed môi trường) | > 0 | [−2,83, +2,56] | FAIL |
| 3. Δ > 0 cho cả 3 seed train | cả 3 | +0,75 / −0,71 / −0,45 | FAIL |
| 4. Recall không giảm có ý nghĩa (cận trên CI ≥ 0) | ≥ 0 | **−0,031 [−0,034, −0,027]** | FAIL |
| 5. Latency tích hợp p99 | ≤ 50 ms | 47,1 ms | PASS |

CI theo 3 seed train (t-interval): −0,14 [−2,07, +1,80].

Ba điều rút ra:
1. **Train PPO trên đúng PHY giúp nhiều so với PPO train trên Shannon** (+16,8 một mình, +8,0 khi kèm MPC, cả 3 seed).
   Đây là mức sai lệch phân phối của A-X7/A-X8.
2. **Nhưng PPO-LA chỉ ngang heuristic tốt nhất** (PPO-LA − H\* = +0,35 [−3,19, +3,88]), và kèm MPC thì ngang MPC lấy
   heuristic làm incumbent (−0,14).
3. **Return ngang nhưng recall của detector thấp hơn**:
   - PPO-LA chuyển 30–50 % ảnh sang mode rẻ (mbt2018-mean_q1, avif_20). Ít drop hơn (0,028 so với 0,078), nhưng ít
     ảnh đến hạn hơn và chất lượng thấp hơn.
   - Recall: 0,465 so với 0,514 của H\*; kèm MPC 0,477 so với 0,507.
   - Hàm thưởng A5 coi hai cách đánh đổi này gần như bằng nhau; detector thì không.

## Luồng làm

**Env (cố định cho mọi arm):**
- K5_la, link adaptation biết lịch, nhiễu persistent, không SRS.
- PPO-LA quan sát thêm đặc trưng LA nhân quả (`obs_la`: CQI trễ, tuổi báo cáo, INR đo, offset OLLA, tỉ lệ NACK 10 / 50
  block, NACK trên block dùng chung). Không thấy kênh ẩn hay tham số simulator.
- `obs_la` chỉ đổi quan sát, không đổi động học.

**Giai đoạn 1: tune lập lịch + cách bay (seed train 0–29).**
- Lưới: 4 cách bay {mission, pacedB1.5, pacedB2, pacedB3} × (12 thành viên JA + least-slack), mỗi ô chạy 3 luật mode
  đại diện.
- Đóng băng: **JA:stat:R:a1:m3 @ pacedB3**, điểm −67,44 (trung bình 3 luật mode).

| Cách bay | JA tốt nhất | JA không dùng chung kênh tốt nhất | least-slack |
|---|---|---|---|
| mission | −77,28 | −78,18 | −92,55 |
| pacedB1.5 | −73,53 | −73,98 | −88,18 |
| pacedB2 | −72,47 | −73,41 (thiết lập A-X8) | −87,39 |
| pacedB3 | **−67,44** | −69,17 | −83,33 |

- JA hơn least-slack +15…+16 ở mọi cách bay (A-X8: +8…+16). Hover lâu hơn (B3) tốt hơn B2 khoảng +5.
- **JA tốt nhất có dùng chung kênh với lề 3 dB** (R:m3) ở cả 4 cách bay, nhưng chỉ hơn bản không dùng chung kênh
  +0,5…+1,7. Khớp A-X6: dùng chung kênh có kiểm soát dưới LA biết lịch cho lợi ích nhỏ.

**Giai đoạn 2: tune heuristic mode và incumbent MPC (seed train, lập lịch đã đóng băng).**
- 22 luật thường + 8 luật hiệu chỉnh online.
- Heuristic tốt nhất **H\* = fixed_mbt2018-mean_q3** (−63,47), giống A-X8.
- Top 5 chạy làm incumbent cho MPC hiệu chỉnh:
  - mbt q3: +MPC −54,37 (**I\***);
  - greedy+budget1: −54,89;
  - avif_20: −57,95;
  - greedy+budget0.75: −58,19;
  - mbt q1: −60,75.
- I\* = H\*. Luật hiệu chỉnh online lại không vào top 5 (tốt nhất hc=greedy+budget2: −89,1), giống A-X8.

**Giai đoạn 3: train PPO-LA.**
- Mode-only, 3 seed, 2M bước mỗi seed, siêu tham số A5, cùng JA và cách bay đã đóng băng.
- Validation mỗi 200k bước trên pool VAL (seed 3000–3019); chọn checkpoint có validation return tốt nhất.

| Seed | Chọn | Validation return mỗi 200k bước |
|---|---|---|
| 0 | 800k | −32,2 −8,2 −9,2 **−7,9** −9,5 −13,2 −14,6 −13,3 −17,5 −13,2 |
| 1 | 400k | −11,6 **−3,5** −7,8 −7,4 −7,3 −10,2 −6,7 −10,3 −12,4 −14,3 |
| 2 | 400k | −15,3 **−5,2** −9,0 −14,7 −20,6 −10,3 −13,2 −13,5 −18,6 −12,2 |

- Validation đạt đỉnh sớm (400k–800k) rồi xấu dần ở cả 3 seed.
- Vậy ngân sách 2M bước không phải là giới hạn; train lâu hơn không có dấu hiệu giúp.
- Pool VAL có thang return khác pool test (ảnh và kịch bản khác), nên không so trực tiếp với số test.

**Giai đoạn 4: test một lần trên seed 8300–8399** (chưa từng dùng).

## Kết quả test

| Arm | Return | Theo seed train | Đúng hạn | Drop | Goodput kbit/s | Năng lượng kJ | Jain | UAV tệ nhất | Recall | Latency tích hợp p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| H\* = I\* (mbt q3) | −39,31 | | 0,898 | 0,078 | 409,8 | 72,9 | 0,995 | 0,812 | **0,514** | 3,6 |
| I\* + MPC hiệu chỉnh | **−32,27** | | 0,908 | 0,056 | 407,2 | 71,6 | 0,996 | 0,834 | 0,507 | 43,0 |
| I\* + MPC không hiệu chỉnh | −49,22 | | 0,877 | 0,100 | 472,3 | 74,9 | 0,993 | 0,778 | 0,494 | 44,2 |
| PPO-LA | −38,96 | −38,8 / −38,1 / −40,0 | 0,856 | 0,028 | 372,6 | 69,3 | 0,995 | 0,776 | 0,465 | 7,2 |
| PPO-LA + MPC | −32,41 | −31,5 / −33,0 / −32,7 | 0,884 | 0,028 | 382,4 | 69,8 | 0,996 | 0,812 | 0,477 | 47,1 |
| PPO-Shannon | −55,78 | −47,2 / −75,4 / −44,8 | 0,858 | 0,092 | 425,3 | 74,4 | 0,991 | 0,749 | 0,487 | 7,2 |
| PPO-Shannon + MPC | −40,41 | −39,1 / −43,6 / −38,6 | 0,890 | 0,065 | 422,1 | 72,5 | 0,995 | 0,801 | 0,493 | 46,8 |

- Tỉ lệ quyết định vượt 50 ms: ≤ 0,0005 ở mọi arm.
- Phần còn lại sau đúng hạn + drop (ví dụ PPO-LA: 1 − 0,856 − 0,028) là ảnh trễ hạn hoặc chưa xong khi hết episode.

**Mode được chọn** (tỉ lệ ảnh):
- H\*: 100 % mbt q3.
- I\* + MPC: 82 % mbt q3; MPC đổi sang avif_5 / mbt q1 / avif_20 cho khoảng 18 %.
- PPO-LA, theo seed:
  - mbt q3: 48 / 70 / 47 %;
  - mbt q1: 30 / 16 / 5 %;
  - avif_20: 22 / 14 / 47 %.
- PPO-Shannon: chủ yếu mbt q3, kèm avif_50 / avif_5. Seed 1 lệch hẳn (−75,4).

**Tương phản ghép cặp** (CI 95 % theo seed môi trường; recall là chênh lệch tuyệt đối):

| Tương phản | Return | Recall | Đúng hạn |
|---|---|---|---|
| **Cổng: (PPO-LA + MPC) − (I\* + MPC)** | −0,14 [−2,83, +2,56] | −0,031 [−0,034, −0,027] | −0,024 [−0,029, −0,019] |
| Hiệu chỉnh: (I\* + MPC) − (I\* + MPC không hiệu chỉnh) | **+16,95 [+12,79, +21,11]** | +0,014 [+0,008, +0,019] | +0,032 |
| MPC trên incumbent: (I\* + MPC) − I\* | +7,03 [+3,96, +10,11] | −0,006 [−0,010, −0,003] | +0,011 |
| PPO-LA − H\* | +0,35 [−3,19, +3,88] | −0,049 [−0,053, −0,044] | −0,041 |
| PPO-LA − PPO-Shannon | +16,82 [+13,78, +19,86] | −0,022 [−0,025, −0,018] | −0,002 n.s. |
| Sai lệch phân phối: (PPO-LA + MPC) − (PPO-Shannon + MPC) | +8,00 [+5,56, +10,43] | −0,016 [−0,019, −0,013] | −0,005 |
| (PPO-LA + MPC) − H\* | +6,90 [+3,36, +10,43] | −0,037 [−0,041, −0,033] | −0,013 |
| (PPO-Shannon + MPC) − H\* | −1,10 [−4,52, +2,32] | −0,021 [−0,025, −0,017] | −0,008 |

- Theo seed train, PPO-LA − PPO-Shannon: +17,0 / +17,7 / +15,8.
- Theo seed train, (PPO-LA + MPC) − (PPO-Shannon + MPC): +8,9 / +7,4 / +7,7.

## Ablation hiệu chỉnh (mục 4 của phản biện #44)

- Cùng incumbent (mbt q3), cùng tìm kiếm (N 8, H 4, rollout nhân quả), cùng ngân sách 50 ms.
- Chỉ khác mô hình tốc độ:
  - **có hiệu chỉnh:** goodput hiệu chỉnh online từ throughput quan sát được;
  - **không hiệu chỉnh:** tốc độ CQI danh nghĩa × (1 − BLER mục tiêu), không thang lệch.
- Bỏ hiệu chỉnh: −16,95 return. **Tệ hơn cả không dùng MPC** (−9,92 so với H\*).
- Bản không hiệu chỉnh tin vào tốc độ danh nghĩa quá lạc quan, nên chọn mode nặng (avif_50 / avif_80 cho 22 % ảnh) và
  drop tăng lên 0,100.
- Vậy giá trị của MPC ở đây hoàn toàn đến từ mô hình tốc độ được hiệu chỉnh online. Lập kế hoạch với mô hình
  danh nghĩa là có hại.

## Đọc kết quả

1. **Giả thuyết DRL bị bác trong phạm vi đã đăng ký.**
   - PPO train trên đúng PHY, với quan sát LA nhân quả, cùng lập lịch, cách bay và hàm thưởng như lúc test, vẫn không
     là incumbent tốt hơn cho MPC so với một mode cố định đã tune.
   - Ba seed train cho kết quả rất sát nhau (−31,5…−33,0 kèm MPC), nên đây không phải do một seed xui.
2. **Lợi ích của PPO-LA so với PPO-Shannon (+16,8) là sửa sai lệch phân phối, không phải lợi thế của DRL.**
   - Nó đưa PPO về đúng mức heuristic tốt nhất, không vượt qua.
3. **Return và recall không đồng hướng.**
   - PPO-LA đạt cùng return với H\* bằng cách giảm drop (0,078 → 0,028) và dùng mode rẻ hơn. Detector mất 4,9 điểm
     recall.
   - MPC cũng thêm +7 return nhưng mất 0,6 điểm recall.
   - Hàm thưởng A5 (đúng hạn, phạt trễ, phạt thời gian) không đo đúng giá trị cho tác vụ phía sau. Đây là quan sát mới,
     liên quan trực tiếp đến yêu cầu "cơ chế có giá trị cho tác vụ" của phản biện #44.
4. **Phần bền của Paper A, theo thứ tự độ lớn:**
   - hiệu chỉnh online mô hình tốc độ cho planner (+17 so với planner danh nghĩa);
   - JA so với least-slack (+15…+16);
   - cách bay paced so với mission (+5…+10);
   - MPC chọn lọc trên incumbent (+7);
   - dùng chung kênh có kiểm soát (+0,5…+1,7).
   - Tất cả đều là cơ chế dựa trên mô hình hoặc heuristic. Theo phản biện #44, gộp chúng lại (JA + MPC) **chưa phải là
     tính mới**.

## Quyết định (theo quy tắc đã đăng ký)

- **DỪNG nhánh DRL.** Không SAC / DSAC / PPO tự hồi quy / MAPPO.
- **Không viết lại Method.** Chưa có cơ chế phối hợp mới có giá trị cho tác vụ.
- Shadowing vẫn hoãn.
- Kết quả âm của A-X5…A-X9 và bảng sửa kết luận được giữ nguyên.

**Ứng viên cho bước sau** (chỉ đề xuất, chưa chạy; cần phản biện chọn):
- (i) **Mục tiêu theo tác vụ thay cho return.**
  - A-X9 cho thấy hai policy cùng return có thể chênh 5 điểm recall.
  - Câu hỏi: một planner nhân quả có tối ưu được tiện ích detector dự đoán (không dùng GT lúc chạy) dưới bất định LA
    không, và lợi ích có vượt mode cố định đã tune không?
  - Rủi ro: tiện ích theo mode trong bảng fidelity hiện tính từ GT; cần một ước lượng không nhãn trước khi coi là
    nhân quả.
- (ii) **Phối hợp ba biến điều khiển** (gán kênh, nhịp bay, mức nén) dưới bất định LA, chấm bằng tiện ích tác vụ thay
  cho return.
  - Cần kiểm tra trước: có tương tác thật giữa ba biến không? A-X8 thấy tương tác lập lịch × mode không ổn định.
- (iii) **Đọc lại thư viện `pdf/`** (cảm nhận cộng tác, HARQ ngữ nghĩa, lập lịch cho cảm nhận cộng tác) trước khi chọn
  cơ chế, để kiểm tra tính mới.
  - Chưa đọc trong vòng này.
