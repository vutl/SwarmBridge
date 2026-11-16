# A-E11 và A-X1 (phản biện #36)

## Tóm tắt

- **A-E11.** Một cửa sổ trượt đơn giản (mỗi dải SNR chỉ giữ 30 quan sát gần nhất) xử lý được lỗi của A-E10 trên seed mới:
  - mismatch Shannon → MCS: từ −7,5 lên +0,45 so với PPO;
  - mismatch MCS → Shannon: từ −13,0 lên −0,14;
  - kênh xấu đi giữa episode: từ −2,0 lên +1,6;
  - khi warm-start khớp: không mất gì ở PHY MCS, mất −0,5 ở ref.
- **Phát hiện phụ quan trọng hơn.** Trên seed mới, warm-start khớp chỉ hơn cold start +0,4 (PHY MCS, không có ý nghĩa thống kê) và +0,9 (ref). Lợi ích của Paper A chủ yếu đến từ việc hiệu chỉnh online trong chính episode, không phải từ warm-start.
- **A-X1.** Pilot trên 100 seed mới, không train mới.
  - Họ heuristic chọn kênh 16 biến thể không hơn T:best.
  - Phần chọn kênh của PPO joint A5 hơn heuristic +2,07 ở ref. Ở K5, nó chỉ có ích khi đi cùng phần chọn mode được train chung (+1,78 so với heuristic-kênh + PPO-mode; riêng phần chọn kênh thì −0,92).
  - Tương tác giữa chọn kênh và chọn mode có thật ở K5. Đề xuất A-X2 train một bộ chọn kênh, hoặc kênh + mode, đúng cấu hình, chờ duyệt.

## 1. A-E11: thích nghi phần hiệu chỉnh khi kênh lệch

### 1.1 Bài toán

- Ở A-E10, mô hình tốc độ được hiệu chỉnh trước bằng 10 episode. Khi điều kiện lúc triển khai khác lúc hiệu chỉnh, bộ điều khiển thua PPO 6–12 điểm.
- Nguyên nhân: hàng nghìn quan sát cũ lấn át vài trăm quan sát mới. Cả mô hình lẫn "chứng chỉ độ tin cậy" của nó đều được tính từ chính kho dữ liệu cũ đó.

### 1.2 Ba cách sửa

Cả ba đều nhỏ và dễ giải thích; cơ chế PPO và MPC không đổi.

| Tên | Cách làm |
|---|---|
| `win=W` (baseline theo độ mới) | Bộ hiệu chỉnh và bộ theo dõi sai số chỉ giữ W quan sát gần nhất trong mỗi dải SNR 2 dB. Dữ liệu cũ, kể cả warm-start, bị đẩy ra dần. |
| `tcold=W` (tách thống kê) | Mô hình tốc độ vẫn dùng warm-start, nhưng **khoảng nhạy cảm** chỉ tính từ sai số online của episode hiện tại (W quan sát gần nhất). Khi chưa có dữ liệu online thì dùng mode của PPO. |
| `sd=trust` (giám sát rồi chuyển) | Một bộ giám sát (a9_mpc.OnlineMonitor) so tốc độ quan sát được trong episode này với dự đoán của warm-start đã đóng băng. Báo động khi \|log tỷ lệ trung bình\| > log 1,5 và lệch hơn 3 SE. Sau báo động, planner chuyển sang mô hình chỉ dùng dữ liệu online (cold); khi mô hình đó thiếu dữ liệu thì dùng PPO. |

**Về ngưỡng báo động.** Ngưỡng 10 % ban đầu báo động sai cả khi warm-start khớp, vì:
- 10 episode hiệu chỉnh chỉ phủ dải SNR khoảng 10–16 dB;
- episode test lên tới 19 dB, nên dự đoán phải ngoại suy và lệch khoảng +15 %;
- một cửa sổ đơn lẻ có \|log tỷ lệ\| tới khoảng 0,38.

Tôi cũng thử đặt ngưỡng bằng sai số prequential giữa các episode hiệu chỉnh, nhưng không dùng được: khi hiệu chỉnh MCS-blind với chính sách dùng chung kênh của ref, sai số tự nhiên đã rất lớn. Vì vậy chọn ngưỡng cố định 50 %.

### 1.3 Kịch bản

Mỗi regime test có kịch bản khớp, các kịch bản warm-start lệch, và một kịch bản thay đổi giữa episode. Trong mọi kịch bản, PPO và cold chạy trên cùng môi trường test.

- **PHY MCS (P = phy_mcs_blind):**
  - khớp;
  - hiệu chỉnh ở ref (Shannon), là lệch mạnh của A-E10;
  - hiệu chỉnh ở phy_gap3 (Shannon trừ 3 dB), lệch vừa;
  - hiệu chỉnh với BLER 0 (tốc độ cao hơn 11 %), lệch nhẹ;
  - thay đổi giữa episode: từ t = 40 s, BLER tăng 0,1 → 0,5 (tốc độ còn khoảng ×0,56).
- **ref (R):**
  - khớp;
  - hiệu chỉnh ở phy_mcs_blind (lệch mạnh);
  - hiệu chỉnh ở phy_gap3 (đánh giá thấp tốc độ 3 dB);
  - thay đổi giữa episode: từ t = 40 s, PHY chuyển Shannon → Shannon trừ 3 dB.

### 1.4 Quy trình

- **Dev.** Seed 8000–8019, là seed phát triển của A-E5…A-E8. Thử 4 cấu hình: win=30, win=100, tcold=30, sd=trust.
- **Quy tắc chọn, chốt trước khi chạy.** Trong mỗi họ, chọn cấu hình có kết quả trong kịch bản tệ nhất cao nhất (maximin). Kết quả: **win=30** và **tcold=30**. sd=trust có trung bình dev tốt hơn nhưng xếp thứ hai theo quy tắc, nên chỉ báo cáo như biến thể phụ.
- **Mốc thời gian.** Lựa chọn được commit trước khi chạy đánh giá cuối (6edf0e18, `runs/uav_semcom/a17/SELECTION.md`).
- **Đánh giá cuối.** Seed 8600–8619, chưa từng dùng.
- **Thống kê.** Ghép cặp theo seed môi trường, lấy trung bình qua 3 checkpoint PPO (checkpoint i so với checkpoint i), khoảng tin cậy t qua 20 seed, giống A-E9 và A-E10.
- **Một lần chạy không hợp lệ đã được bỏ.** Lần dev đầu chạy 14 tiến trình trên WSL chỉ có 6 CPU. 87–94 % quyết định bị cắt bởi ngân sách 50 ms, nên mọi bộ điều khiển gần như chỉ còn là PPO. Kết quả đó được giữ ở `a17/invalid_oversubscribed/` để làm hồ sơ, không dùng. Mọi lần chạy khác dùng 6 tiến trình: abort ≤ 0,6 %, p99 42–45 ms.

### 1.5 Kết quả cuối

Return so với PPO, trên seed 8600–8619:

| Kịch bản | PPO | cold | pre (A-E9) | **win=30** | **tcold=30** | sd=trust (phụ) |
|---|---|---|---|---|---|---|
| P khớp | 32,53 | +1,35 [+0,53; +2,18] | +1,73 [+0,51; +2,95] | **+1,75** [+1,09; +2,41] | +1,21 | +1,76 |
| P ← ref (lệch mạnh) | | +1,35 | **−7,54** [−9,76; −5,33] | +0,45 [−0,79; +1,68] | +0,60 | −0,08 |
| P ← gap3 | | +1,35 | −0,67 | +1,37 [+0,12; +2,61] | +1,10 | +0,72 |
| P ← BLER 0 | | +1,35 | +1,20 | +1,72 | +1,29 | +1,38 |
| P, thay đổi giữa episode | 14,75 | +1,04 | **−2,01** [−3,30; −0,73] | +1,60 [+0,22; +2,98] | +0,32 | +1,58 |
| R khớp | 59,28 | +1,21 [+0,85; +1,58] | +2,10 [+1,48; +2,72] | +1,59 [+1,10; +2,07] | +0,97 | +1,88 |
| R ← phy_mcs_blind (lệch mạnh) | | +1,21 | **−13,03** [−15,26; −10,80] | −0,14 [−0,74; +0,46] | −0,26 [−0,45; −0,07] | −1,34 |
| R ← gap3 | | +1,21 | +0,29 | +1,49 | +0,21 | +0,81 |
| R, thay đổi giữa episode | 51,30 | +1,07 | +1,61 | +0,82 | +0,82 | +1,20 |
| **Tệ nhất / trung bình (9 kịch bản)** | | | −13,03 / −1,59 | **−0,14 / +1,18** | −0,26 / +0,70 | −1,34 / +0,88 |

So với cold:

| Kịch bản | pre | win=30 |
|---|---|---|
| P khớp | +0,38 [−1,03; +1,79] | +0,40 [−0,42; +1,22] |
| R khớp | +0,88 [+0,49; +1,28] | +0,37 [+0,06; +0,69] |
| P ← ref | −8,89 | −0,90 [−1,94; +0,14] |
| R ← phy_mcs_blind | −14,24 | −1,35 [−1,87; −0,83] |
| P, thay đổi giữa episode | −3,05 | +0,56 |
| Các kịch bản lệch nhẹ / vừa | | từ +0,02 đến +0,37 |

### 1.6 Phát hiện, báo động sai và chi phí

**Phát hiện và báo động sai (sd=trust):**
- Lệch mạnh: báo động 100 % sau 11–13 s.
- Thay đổi giữa episode: báo động 98 % (P) và 65 % (R), trung vị 14,5–16 s sau khi thay đổi.
- Lệch vừa (gap3): báo động 95–97 %, trung vị 27–39 s.
- Báo động sai khi warm-start khớp: 7 % (P) và 17 % (R) số episode.
  - Ở R, khoảng cách này ngoài vùng hiệu chỉnh; lệch thật vì ngoại suy, nhưng không có hại.

**Hạn chót và chi phí tính toán:**
- late = 0 trong mọi biến thể, kể cả khi lệch.
- Ảnh đúng hạn (ok):
  - pre khi lệch mạnh: 0,973–0,974;
  - win=30 trong cùng các kịch bản lệch mạnh: 0,986–0,997;
  - PPO: 0,986–1,000 khi không đổi PHY, 0,977–0,996 khi đổi PHY giữa episode.
- p99 thời gian quyết định 42–44 ms, giống A-E9. Cửa sổ trượt không thêm chi phí đáng kể.

### 1.7 Kết luận A-E11

1. **Lỗi A-E10 sửa được bằng một cách rất đơn giản.** Với cửa sổ 30 quan sát mỗi dải SNR, không còn kịch bản nào thua PPO có ý nghĩa thống kê; tệ nhất là −0,14 với CI chứa 0.
   - Mất mát khi warm-start khớp: 0 ở P và −0,51 [−1,00; −0,02] ở R.
   - Cách tách thống kê và cách báo động không tốt hơn baseline cửa sổ trượt.
2. **Warm-start mang lại ít lợi ích.** Trên seed mới, cold đã đạt +1,35 / +1,21 so với PPO; warm-start khớp chỉ thêm +0,4 / +0,9. Khi lệch mạnh, win=30 vẫn kém cold 0,9–1,35.
   - Nếu không biết chắc điều kiện triển khai khớp lúc hiệu chỉnh, cold là lựa chọn an toàn và đơn giản nhất. win=30 là thỏa hiệp: ngang hoặc hơn cold ở mọi kịch bản trừ lệch mạnh.
3. **Hệ quả cho Paper A.**
   - Nên viết lại: lợi ích chủ yếu đến từ **hiệu chỉnh online có bộ nhớ giới hạn** (cold hoặc win=30), còn warm-start chỉ là tùy chọn với lợi ích nhỏ.
   - Câu "bền trước sai lệch mô hình" chỉ đúng với bộ nhớ giới hạn hoặc cold, không đúng với warm-start đóng băng (A-E10).
   - Không có bảo đảm hình thức nào. Ngưỡng log 1,5 và cửa sổ 30 là lựa chọn heuristic, được chọn trên seed dev.
4. **Giới hạn.**
   - Chỉ có 2 regime và 7 kiểu lệch, các kiểu lệch đều là thay đổi PHY/BLER.
   - Chưa thử lệch về hình học hay độ trễ feedback cộng với lệch.
   - Thay đổi giữa episode chỉ xảy ra một lần, tại t = 40 s.

## 2. A-X1: lập lịch kênh kết hợp chọn mode (pilot, không train mới)

### 2.1 Câu hỏi và cách làm

**Câu hỏi.** Chọn UAV nào được phát trên kênh nào (scheduling) có tương tác với việc chọn mode không, và một bộ chọn kênh học được có hơn heuristic tốt không?

**Hệ thống giữ nguyên** (bài toán giao ảnh multi-UAV của A5):
- cùng simulator, phần thưởng, pool ảnh test và seed môi trường (8700–8799, 100 seed mới);
- danh sách ảnh, hạn chót, ngưỡng chất lượng và fading được sinh từ seed, độc lập với policy;
- di chuyển và công suất cho mọi tổ hợp đều lấy từ phần slot của T:best;
- chỉ hai thứ thay đổi: **ai chọn kênh** và **ai chọn mode**.

**Bộ chọn kênh:**
- `tbest`: quy tắc của T:best, là EDF theo slack, cộng dùng chung kênh ở K5 và ref.
- `sch`: SCH*, tốt nhất trong họ 16 quy tắc mới (`ax1_sched.py`), tune trên train pool seed 0–29 trước pilot.
  - Họ quy tắc có 3 trục: ưu tiên EDF hoặc slack/tốc độ; bỏ hoặc không bỏ ảnh không còn kịp; không dùng chung kênh, dùng chung theo tải nhỏ nhất (giống A5), hoặc dùng chung có kiểm tra SINR với ngưỡng 0,5 hay 0,25.
  - Cả hai regime chọn `SCH:a0:s1:least`, tức EDF, bỏ ảnh không còn kịp, dùng chung kênh. Trên train nó chỉ hơn T:best +0,1–0,2.
- `jc`: **phần chọn kênh của PPO joint A5** (checkpoint best-val). PPO này được train chung với di chuyển, công suất và mode; ở đây hai phần di chuyển và công suất được thay bằng heuristic.

**Bộ chọn mode:**
- `heur`: quy tắc budget của T:best.
- `ppo`: PPO mode-only A5, vốn được train với bộ chọn kênh của T:best.
- `jm`: phần chọn mode của chính PPO joint đó.

**Kiểm tra tương đương trước khi đọc kết quả** (`test_ax1.py`, 6 test, đều pass):
- SCH a0/s0 khớp đúng return của quy tắc A5 (có và không có reuse);
- vòng lặp pilot `tbest|heur` khớp đúng `run_episode_ev`;
- `tbest|ppo` khớp `a5_eval.rl_episode` (mode-only);
- `jc|jm` khớp tổ hợp thay thế hành động "HPHP" của A5.

**Quy ước số liệu.** Return là trung bình qua 3 checkpoint PPO; khoảng tin cậy ghép cặp theo seed.

### 2.2 Kết quả

| Tổ hợp (kênh \| mode) | K5 return | ok | recall | ms/slot | ref return | ok | recall |
|---|---|---|---|---|---|---|---|
| tbest \| heur | 61,74 | 0,967 | 0,566 | 0,2 | 58,93 | 0,991 | 0,595 |
| sch \| heur | 61,73 | 0,967 | 0,566 | 0,3 | 58,93 | 0,991 | 0,595 |
| jc \| heur | 60,81 | 0,968 | 0,565 | 2,4 | **61,00** | 0,995 | 0,599 |
| tbest \| ppo (chuẩn A5) | 64,56 | 0,990 | 0,578 | 0,2 | 58,38 | 0,999 | 0,596 |
| sch \| ppo | 64,56 | 0,990 | 0,578 | 0,3 | 58,37 | 0,999 | 0,596 |
| jc \| ppo | 64,24 | 0,990 | 0,578 | 2,2 | 59,32 | 1,000 | 0,596 |
| jc \| jm | **66,34** | 0,991 | 0,582 | 2,2 | 58,44 | 0,999 | 0,596 |
| joint \| jm (PPO cũng chọn di chuyển và công suất) | 63,46 | 0,998 | 0,590 | 1,9 | 57,93 | 0,999 | 0,596 |

Các so sánh ghép cặp (return):

| So sánh | K5 | ref |
|---|---|---|
| C1: sch − tbest (mode heur) | −0,01 [−0,19; +0,16] | 0,00 |
| C2: jc − sch (mode heur) | **−0,92** [−1,80; −0,05] | **+2,07** [+1,58; +2,56] |
| C3: jc − sch (mode ppo) | −0,32 [−0,72; +0,09] | +0,95 [+0,70; +1,20] |
| C4: tương tác (jc\|ppo − jc\|heur) − (sch\|ppo − sch\|heur) | +0,61 [−0,24; +1,46] | **−1,12** [−1,61; −0,63] |
| C5: jc\|jm − tbest\|ppo | **+1,78** [+1,11; +2,44] | +0,06 [−0,29; +0,42] |
| C6: joint\|jm − jc\|jm | −2,88 [−3,53; −2,22] | −0,51 [−0,73; −0,29] |
| tbest\|ppo − tbest\|heur (lợi ích của học mode) | +2,82 [+1,97; +3,66] | −0,55 [−1,11; +0,02] |

Các metric khác:
- **Hạn chót.** late = 0 ở mọi tổ hợp.
- **Công bằng giữa UAV.** Chỉ số Jain ≥ 0,999 ở mọi tổ hợp. Tỷ lệ đúng hạn của UAV kém nhất: 0,92 với mode heuristic ở K5, 0,97–0,99 với mode học được.
- **Năng lượng.** Như nhau, trừ joint ở K5 (78,6 kJ so với 66–67 kJ), vì joint tự học cách bay.
- **Dùng chung kênh** (tỷ lệ airtime có nhiễu đồng kênh):
  - K5: jc|jm 0,77, tbest|ppo 0,75;
  - ref: jc|heur 0,43, tbest|heur 0,48 (jc dùng chung ít hơn).
- **Độ trễ quyết định.**
  - Scheduler PPO: 1,5–2,4 ms/slot. Heuristic: 0,1–0,3 ms.
  - p99 của cả bộ scheduler + mode: ≤ 3,8 ms, chưa tính MPC.

### 2.3 Đọc kết quả

1. **Heuristic chọn kênh đã chạm trần trong họ quy tắc đơn giản.** 16 biến thể (EDF, trọng số theo tốc độ, bỏ ảnh không còn kịp, dùng chung có kiểm tra nhiễu) không hơn T:best.
   - Việc **dùng chung kênh** có ảnh hưởng rất lớn: ở K5 trên train, có dùng chung là +42,5, không dùng chung là +8,3.
   - Dùng chung có kiểm tra SINR "khi cả hai cùng phát" thì quá bảo thủ, vì nhiễu chỉ xảy ra khi cả hai thực sự cùng phát.
2. **Bộ chọn kênh học được có giá trị, nhưng phụ thuộc regime và bộ chọn mode đi kèm.**
   - ref: riêng phần chọn kênh đã hơn heuristic +2,07 (mode heur) và +0,95 (mode PPO). jc|heur là tổ hợp tốt nhất ở ref.
   - K5: riêng phần chọn kênh lại kém (−0,92). Nhưng cặp kênh + mode **train cùng nhau** hơn heuristic-kênh + PPO-mode +1,78, và hơn T:best +4,59.
   - Như vậy, **ở K5 cách chọn kênh và cách chọn mode phụ thuộc lẫn nhau**: một bộ chọn kênh tốt cho một bộ chọn mode lại không tốt cho bộ khác.
   - Đây là bằng chứng về tương tác "ai được phát" × "phát bằng mode nào" mà review #36 muốn kiểm tra. Kết quả khớp với giả thuyết post-hoc ở mục 4.4 của A5 (+4,8 so với T:best), nay được xác nhận trên 100 seed mới.
3. **Học di chuyển và công suất có hại** (C6: −2,9 ở K5, −0,5 ở ref). Điều này ủng hộ việc giữ di chuyển và công suất theo heuristic, như review đã đề nghị.
4. **Hai lưu ý quan trọng.**
   - PPO mode-only được train với bộ chọn kênh của T:best. Ghép nó với `jc` là đặt nó ra ngoài phân phối train, nên C4 âm ở ref có thể do lệch phân phối chứ không phải do tương tác thật.
   - `jc` không được train cho cấu hình này: lúc train, di chuyển và công suất do chính PPO quyết định. Pilot vì vậy chỉ đo **cận dưới** của một bộ chọn kênh được train đúng cách.
5. **Recall của detector gần như không đổi:** 0,57–0,60 và trên vật nhỏ 0,45–0,48. Lý do là mọi tổ hợp đều giao gần như đủ ảnh đúng hạn. Lợi ích về return đến từ PSNR, năng lượng và thời gian, không phải từ phát hiện vật.

### 2.4 Đề xuất A-X2 (chưa làm, chờ duyệt)

Một bộ chọn kênh học được, train đúng cấu hình. Không dùng MPC trên tổ hợp kênh.

- **Hành động.** Kênh của từng UAV ∈ {0 (im lặng), 1..M}, dạng MultiDiscrete phân rã theo UAV.
  - Ở K5 có 3^5 = 243 tổ hợp.
  - Không có ràng buộc tổ hợp cứng, vì môi trường cho phép dùng chung kênh. Mask: UAV không hoạt động hoặc hàng đợi rỗng thì bị buộc chọn 0.
  - Di chuyển và công suất bị mask để buộc theo heuristic T:best. Nhờ đó dùng nguyên wrapper, quan sát và không gian hành động của A5.
- **Thuật toán.** MaskablePPO với chiết khấu theo thời gian (`timeppo`), giống hệt A5, để mọi khác biệt chỉ đến từ cách giới hạn hành động.
  - Không chọn discrete SAC: với MultiDiscrete phân rã, nó cần entropy riêng cho từng nhánh và kém ổn định.
  - Không chọn DQN: phải liệt kê 243 tổ hợp và mất cấu trúc phân rã.
  - Chi phí suy luận đã đo ở trên: khoảng 2 ms/slot.
- **Hai biến thể**, vì pilot cho thấy ở K5 kênh và mode phụ thuộc lẫn nhau:
  - (i) chỉ chọn kênh, mode theo heuristic budget;
  - (ii) chọn kênh + mode, di chuyển và công suất theo heuristic.
- **Quy mô.** K5 và ref, 3 seed mỗi cấu hình, 2M bước, CPU, khoảng 6 run × 30 phút.
  - Chọn checkpoint trên val như A5.
  - Đánh giá trên seed mới, so với T:best, SCH*, PPO mode-only và jc|jm.
- **Kiểm tra quan trọng nhất.** Gắn PPO–MPC hybrid (win=30) lên trên scheduler học được. Lợi ích của MPC có còn không khi scheduler đã thay đổi, và tổng chi phí quyết định có nằm trong 50 ms không?
