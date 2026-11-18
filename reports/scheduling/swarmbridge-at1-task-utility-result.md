# A-T1 (phản biện #45): dư địa của tiện ích tác vụ trong simulator mạng, cùng rà soát tính mới

Ngày chạy: 2026-11-18.
- **Bước A** (rà soát tính mới): [swarmbridge-task-utility-novelty-review-2026-11-18.md](swarmbridge-task-utility-novelty-review-2026-11-18.md).
- **Bước B** (oracle tác vụ): [protocol](swarmbridge-at1-protocol.md), đăng ký trước, commit `de11c3dd`, trước khi chạy
  test.

**Code:**
- `at1_task.py`, `at1_run.py`, `at1_summary.py`.
- `at1_explore.py` (chẩn đoán thêm sau khi xem kết quả).
- `ax2_eval.py` thêm `t=` / `tc=`, `+tmpc=`, `DELIV_EXT`.
- `test_at1.py`: 5 test; cùng `test_ax9.py`, 10 test đều pass.

**Kết quả** ở `runs/uav_semcom/at1/`:
- `tune.json`, `test.json`, `test.deliv.json.gz`, `summary.txt`, `summary.json`, `explore_count.{txt,json}`;
- `flip/per_image_flip_test.jsonl`: đo lại detector trên ảnh lật cho 128 ảnh test × 13 mode, GPU qua khoá.

## Trả lời ngắn

**Cổng đăng ký trước: QUA cả 4 điều kiện → theo quy tắc, đi tiếp bước C** (bộ dự đoán tiện ích nhân quả).

Nhưng có ba điều thay đổi cách hiểu kết quả, và bước C phải được thiết kế theo chúng:
1. **Phần dư địa lặp lại được gần như toàn bộ là "ảnh nào có nhiều vật thể"**, không phải "mode nào hợp ảnh nào".
   - Một oracle chỉ biết số vật thể của ảnh (chẩn đoán thêm sau khi xem kết quả) đạt +1,53 [1,06, 2,00]. Con số này
     gần bằng dư địa đã kiểm chéo (+1,71).
   - Nó xếp ngang oracle chọn bằng phép đo lật (53,29 so với 53,30 khi chấm bằng phép đo gốc).
2. **Cơ chế là giảm tắc nghẽn.**
   - Oracle gửi ảnh ít vật thể bằng mode rất rẻ (avif_5, mbt q1), nên bớt 25 % bit và giảm drop từ 0,059 xuống 0,025.
   - Lợi ích lớn nhất ở episode tắc nghẽn: +5,1 điểm, so với +1,4 khi ít tắc nghẽn.
   - Đây là điều G-L2 (tĩnh, khớp bit, không có mạng) không thấy được.
3. **Lợi ích có được nhờ vượt sàn PSNR (Qmin).**
   - Theo recall đúng hạn của detector (chỉ số đăng ký): oracle +3,00, kiểm chéo +1,71.
   - Theo recall tuân thủ Qmin (đúng hạn **và** PSNR ≥ Qmin): oracle **−2,15**.
   - Qmin là ràng buộc hệ thống thừa hưởng từ A5. Nếu nó là yêu cầu thật (ví dụ con người cần xem ảnh), dư địa này
     không dùng được.

Kết quả về mặt bằng chứng (bước A): không có khoảng trống thuật toán nào được bảo vệ chắc chắn. Ước lượng giá trị ảnh
không cần nhãn để quyết định offload đã có (ORIC, SEC 2024; DAO, AdaCompress). MPC hiệu chỉnh online với giới hạn tính
toán đã có ở dòng ABR (RobustMPC, FastMPC, Oboe, Fugu).

## Luồng làm

**Đã có trước (không làm lại):**
- **A6-L1:** reward tác vụ chỉ dời điểm hoạt động sang mode cố định rẻ hơn.
- **G-L2:** offline, mỗi ảnh một mode, khớp bit trung bình, không có mạng.
  - Oracle hơn luật biết payload +2,29 / +1,84 / +1,12; kiểm chéo bằng ảnh lật còn +0,95 / +0,21 / −0,40.
  - Bộ dự đoán nhân quả không hơn luật biết payload.
- Câu hỏi mới của A-T1: khi có hàng đợi, deadline, LA và 5 UAV chia 2 kênh, thông tin tiện ích từng ảnh có đáng hơn
  không?

**Env:**
- K5_la, LA biết lịch, nhiễu persistent, không SRS.
- Lập lịch và cách bay đóng băng từ A-X9: `JA:stat:R:a1:m3 @ pacedB3`.
- Thông tin kênh của mọi bộ điều khiển là nhân quả (mô hình tốc độ hiệu chỉnh online).

**Chỉ số:** recall đúng hạn trên toàn khối việc = Σ TP của ảnh giao đúng hạn / Σ vật thể của mọi ảnh đã chụp. Ảnh trễ
hoặc rớt tính 0.

**Nguồn tiện ích:**
- `mean`: TP trung bình theo mode trên ảnh train; nhân quả.
- `gt`: TP thật của ảnh ở từng mode; oracle, không triển khai được.
- `flip`: TP đo lại trên ảnh lật; dùng để kiểm chéo.

**Bộ điều khiển:**
- mode cố định;
- **RV:** argmax U(m) − λ·b̂(m);
- **TB:** luật ngân sách backlog với giá trị tác vụ;
- **TMPC:** MPC chọn lọc hiệu chỉnh của A-E8, mục tiêu Σ TP đúng hạn;
- tham chiếu: q3 và q3 + MPC mục tiêu A5.

**Tune (seed train 0–29), chọn theo recall train:**

| Nhóm | Thành viên chọn | Recall train |
|---|---|---|
| F_task (mode cố định) | mbt q3 | 48,02 |
| q3 + MPC (mục tiêu A5) | | 48,21 |
| RV-mean | λ = 20 | 47,85 |
| TB-mean | τ = 3 | 38,21 |
| TMPC-mean | | 47,34 |
| RV-gt | λ = 40 | **51,29** |
| TB-gt | τ = 2 | 41,70 |
| TMPC-gt | | 47,86 |

Theo quy tắc đăng ký:
- **C\*** (nhân quả mạnh nhất) = q3 + MPC;
- **O\*** (oracle chính) = RV-gt.

## Kết quả test (seed mới 9200–9299)

| Arm | Recall đúng hạn (gt) | (flip) | Recall tuân thủ Qmin | Tỉ lệ khung đúng hạn | Drop | Mbit | Năng lượng kJ | Return A5 | Mode chính |
|---|---|---|---|---|---|---|---|---|---|
| q3 (H\*) = F_task | 51,84 | 51,87 | 51,12 | 0,914 | 0,086 | 59,7 | 73,5 | −43,7 | q3 100 % |
| **q3 + MPC (C\*)** | 51,76 | 51,76 | 50,45 | 0,941 | 0,059 | 57,5 | 72,2 | −35,9 | q3 81 % |
| RV-mean | 52,16 | 52,15 | 51,19 | 0,919 | 0,081 | 58,8 | 73,0 | −42,6 | q3 97 % |
| TB-mean | 41,96 | 41,91 | 37,74 | 0,946 | 0,054 | 59,2 | 71,6 | −113,5 | trộn, có cap_t2i 14 % |
| TMPC-mean | 52,05 | 52,06 | 51,19 | 0,917 | 0,083 | 62,2 | 73,8 | −43,7 | q3 96 % |
| **RV-gt (O\*)** | **54,77** | 53,66 | 48,31 | 0,975 | 0,025 | 43,2 | 69,4 | −51,9 | q3 36 %, avif_20 24 %, mbt q1 23 %, avif_5 9 % |
| TB-gt | 45,15 | 44,69 | 38,28 | 0,985 | 0,015 | 44,3 | 68,3 | −93,0 | trộn |
| TMPC-gt | 52,60 | 52,33 | 51,40 | 0,920 | 0,080 | 62,2 | 73,3 | −45,8 | q3 87 % |
| RV-flip | 53,30 | 54,42 | 46,31 | 0,975 | 0,025 | 42,5 | 69,2 | −55,8 | như RV-gt |
| TB-flip | 45,03 | 45,56 | 38,14 | 0,985 | 0,015 | 44,0 | 68,4 | −90,4 | |
| TMPC-flip | 52,19 | 52,40 | 50,88 | 0,917 | 0,083 | 61,9 | 73,4 | −47,0 | |

Latency tích hợp p99:
- luật: ≤ 3,8 ms;
- q3 + MPC: 42,9 ms;
- TMPC: 33–36 ms.

## Cổng quyết định (đăng ký trước)

O\* = RV-gt so với C\* = q3 + MPC; điểm recall; CI ghép cặp theo 100 seed môi trường.

| Đại lượng | Giá trị | |
|---|---|---|
| H (chọn và chấm bằng cùng một phép đo) | +3,00 [+2,54, +3,46] | |
| **H_x (chọn bằng phép đo này, chấm bằng phép đo kia)** | **+1,71 [+1,29, +2,14]** | ≥ +1 và CI > 0: **PASS** |
| Tỉ lệ khung đúng hạn | +3,39 [+2,72, +4,06] điểm | không thấp hơn: **PASS** |
| Năng lượng | −3,8 % | không cao hơn 2 %: **PASS** |

**Quyết định: đi tiếp bước C.**

## Phụ (không thuộc cổng)

**Giá trị thông tin trong cùng một bộ điều khiển** (gt / kiểm chéo):
- RV-gt − RV-mean: +2,60 / **+1,32 [0,80, 1,84]**.
- TB-gt − TB-mean: +3,19 / +2,92. Nhưng cả hai TB đều kém hơn C\* rất xa (−6,6 và −9,8): họ luật này hỏng dưới mục
  tiêu chỉ tính TP.
- **TMPC-gt − TMPC-mean: +0,55 / +0,20 [−0,18, 0,59].**
  - MPC với mục tiêu "Σ TP đúng hạn" gần như không dùng được thông tin tiện ích. Trên train nó còn kém incumbent
    (47,34 so với 48,02).
  - Lý do: tầm nhìn 4 ảnh và giá trị 0 cho ảnh rớt khiến nó tham chất lượng cho ảnh hiện tại, rồi đẩy tắc nghẽn ra
    ngoài tầm nhìn. Mục tiêu A5 có số hạng "đến sớm" nên tự bảo vệ hàng đợi.
  - Đây là kết quả âm của chính bộ điều khiển, ghi lại như vậy.

**Recall tuân thủ Qmin so với C\*:**
- Các arm nhân quả: +0,7…+0,95. Các luật không vượt Qmin vì vẫn chủ yếu dùng q3.
- RV-gt: −2,15. RV-flip: thấp hơn nữa (46,31).

**Độ tập trung** (RV-gt so với C\*, theo từng ảnh trong từng episode, TP gt):
- Tổng lãi 51 977 TP = 84 033 dương − 32 056 âm.
- 10 % ảnh có lãi lớn nhất mang 66 % phần dương.
- Theo tam phân vị số vật thể: ít 12 %, vừa 28 %, **nhiều 60 %**.
- Theo mức tắc nghẽn của episode (drop của C\*): thấp +1,39, vừa +2,67, **cao +5,14**.
- Theo video: không video nào áp đảo. Lớn nhất 0000330: +8 688 TP, tức 17 % tổng.

## Chẩn đoán thêm sau khi xem kết quả: oracle chỉ biết số vật thể

`at1_explore.py`: U_i(m) = n_gt_i / trung bình train của n_gt × μ_m.
- Nó biết ảnh có bao nhiêu vật thể, nhưng không biết mode nào hợp ảnh đó, và không dính nhiễu detector.
- λ tune trên train (best 40, recall train 50,02), chạy trên cùng seed test.

| | Recall đúng hạn | Tuân thủ Qmin | Đúng hạn | Mbit |
|---|---|---|---|---|
| RV-count | 53,29 | 48,82 | 0,971 | 45,3 |
| − C\* | **+1,53 [+1,06, +2,00]** | | | |
| − RV-mean | +1,13 [+0,56, +1,70] | | | |
| − RV-gt | −1,47 [−1,74, −1,20] | | | |

Diễn giải:
- Phần oracle gt hơn oracle "số vật thể" (+1,47) gần đúng bằng phần mất đi khi chọn bằng phép đo lật (54,77 → 53,30).
- Tức là phần "mode nào hợp ảnh nào" chủ yếu không lặp lại được, khớp với G-L2.
- **Phần lặp lại được là biết ảnh nào đáng giá (nhiều vật thể) để gửi đúng hạn, và gửi rẻ những ảnh ít vật thể để
  giải phóng kênh.**
- Lưu ý: chỉ số là recall gộp theo vật thể, nên ảnh ít vật thể đóng góp ít theo định nghĩa. Nếu tính recall trung bình
  theo ảnh hoặc mAP theo ảnh, mức lợi có thể khác. Chưa đo.

## Đọc kết quả

1. **Mạng làm dư địa tiện ích lớn hơn bản tĩnh.**
   - G-L2 tĩnh (khớp bit): kiểm chéo +0,95 ở mức bit thấp.
   - Trong simulator: +1,71, lớn nhất khi tắc nghẽn.
   - Lợi ích không đến từ chọn mode khéo hơn cho từng ảnh, mà từ **phân bổ thời gian phát giữa các ảnh theo giá trị**.
     Ảnh ít vật thể bị hạ xuống mode rẻ, nên ảnh nhiều vật thể không bị rớt.
2. **Thứ bước C cần ước lượng là số vật thể (giá trị) của ảnh, nhân quả, trên UAV.**
   - Không cần ước lượng hiệu TP giữa các mode; G-L2 cho thấy việc đó không khả thi.
3. **Rủi ro về tính mới của bước C.**
   - Ước lượng lợi ích từng ảnh từ detection cục bộ để quyết định offload đã có (ORIC, SEC 2024).
   - Phần khác của mình: dùng ước lượng đó để phân bổ thời gian phát giữa nhiều UAV có deadline trên kênh LA, nơi
     tắc nghẽn biến thông tin giá trị thành recall.
   - Đây vẫn là đóng góp hẹp; cần kiểm kỹ trước khi claim.
4. **Qmin phải được quyết định trước bước C.** Dưới recall tuân thủ Qmin, mọi oracle đều thua C\*.

## Đề xuất bước C (chưa chạy; cần phản biện duyệt)

Cổng offline + simulator, cùng env, cùng seed train/val. Seed test mới.

**Bộ dự đoán số vật thể nhân quả, thứ tự theo chi phí:**
- (a) payload probe AVIF-5 (đã có trên UAV; ảnh nhiều chi tiết thường nhiều vật thể), fit trên train;
- (b) thống kê ảnh rẻ (G-L2 bộ B, khoảng 10 ms);
- (c) detector nhẹ trên UAV (G-L2 bộ C, 35–40 ms CPU, cùng trọng số với trạm: trường hợp lạc quan).

Nhãn chỉ dùng trên train/val để fit; lúc quyết định không truy cập nhãn.

**Bộ điều khiển:** RV với U = n̂_i × μ_m, λ tune trên train.

**Cổng đề xuất:**
- recall đúng hạn hơn C\* ≥ +1,0, CI > 0;
- thu ≥ 50 % dư địa của oracle số vật thể;
- tính cả chi phí bộ dự đoán vào latency và năng lượng;
- báo cả recall tuân thủ Qmin.

**Cần phản biện quyết định trước:**
- (1) Qmin có phải yêu cầu thật không;
- (2) chỉ số chính giữ recall gộp theo vật thể hay thêm recall theo ảnh / AP50;
- (3) simulator rút ảnh độc lập từ pool, nên bộ dự đoán dựa trên tương quan giữa các khung liên tiếp (khung trước của
  cùng UAV) không kiểm được ở đây. Muốn kiểm phải dùng chuỗi thật (VisDrone-VID).

**Không làm:** chưa phê duyệt điều khiển chung (phản biện #45 mục D). Không RL.

## Giới hạn

- Một detector, một điểm hoạt động, menu mode A5, 128 ảnh test lấy lại nhiều lần qua các episode.
- Phép lật chỉ kiểm một loại bất ổn của detector.
- TMPC chỉ thử một dạng mục tiêu (TP đúng hạn, tầm nhìn 4). Một mục tiêu có số hạng hàng đợi có thể khác; chưa thử.
- Oracle "số vật thể" là chẩn đoán sau khi xem kết quả, không thuộc cổng.
- Recall A-X9 dùng file fidelity A5, thiếu dòng của 3 mode sinh ảnh với 448/548 ảnh. Các arm A-X9 hầu như không dùng
  mode sinh ảnh, nên ảnh hưởng không đáng kể. A-T1 dùng file L1 đầy đủ, cùng giá trị trên mọi dòng chung.
- Đây là offload độc lập nhiều UAV, không phải cảm nhận cộng tác.
