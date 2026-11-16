# A-E9 và B-E8: độ bền của bộ điều khiển Paper A với feedback kém; phân bổ bit theo nhiệm vụ trong HEVC (review #34)

Ngày 2026-11-16. Đây là vòng khám phá, không có cổng duyệt.

- Code:
  - A: `swarmbridge/experiments/uav_semcom/a9_mpc.py` (`Feedback`), `a10_hybrid.py` (các option `fbd`/`fbl`),
    `ae9_summary.py`.
  - B: `swarmbridge/experiments/jepa_semcom/x265roi.c`, `b4_roi.py`, `test_b4.py`.
- Kết quả:
  - A: `runs/uav_semcom/a15/` (ae9_phy_mcs_blind.json, ae9_ref.json, ae9_summary.json).
  - B: `runs/jepa_semcom/b8/`.

## Tóm tắt

**A-E9 (feedback trễ/mất, 20 seed môi trường mới 8500–8519, chưa từng dùng khi phát triển):**
- Lợi ích so với PPO vẫn giữ trên seed mới. Ở phy_mcs_blind, np-any đạt +2,18 [+1,30, +3,06]; trên seed phát triển là
  +2,86. Ở ref đạt +1,91 [+1,42, +2,39], gần như bằng seed phát triển (+1,96).
- Feedback trễ 1–3 slot và mất 30–60 % báo cáo **không làm giảm đáng kể** ở bất kỳ bộ điều khiển nào: mọi mức suy giảm
  ghép cặp theo seed đều có CI chứa 0, với độ lớn tối đa khoảng 0,6.
- Lý do chính là **warm-start** (calibrator và tracker được nạp trước từ 10 episode hiệu chỉnh). Khi bỏ warm-start
  (cold):
  - lợi ích chỉ còn +0,65 (phy) / +0,96 (ref);
  - feedback trễ 3 slot + mất 30 % làm tăng số quyết định thiếu dữ liệu (phải dùng PPO) từ 678 lên 1005 (phy) và từ
    533 lên 876 (ref).
- Ngân sách 50 ms vẫn giữ: p99 41–44 ms, 0–0,04 % quyết định vượt 50 ms. Có các ngoại lệ đơn lẻ 105 / 110 / 139 ms.
  Planner không giới hạn vượt 50 ms ở khoảng 55 % quyết định.
- Với Shannon dưới MCS trên seed mới: np-pess +1,32 [−0,13, +2,76], n2 +0,93, planner đầy đủ +0,42.
  - np-pess − planner đầy đủ: +0,90 [+0,06, +1,74].
  - np-pess − n2: +0,39 [−0,52, +1,30]. Trên seed phát triển là +1,19 với CI > 0, nên **lợi thế của ứng viên bi quan
    so với top-2 yếu hơn trên seed mới**.

**B-E8:** xem mục 2.

## 1. A-E9

### 1.1 Cách làm

- **Mô hình feedback** (`a9_mpc.Feedback`):
  - Mỗi báo cáo throughput (một UAV, một slot) sinh ra ở cuối slot.
  - Báo cáo mất với xác suất P, độc lập theo UAV và slot. Hạt giống mất chỉ phụ thuộc (seed, episode, slot, UAV), nên
    calibrator và tracker mất **cùng** các báo cáo.
  - Báo cáo không mất chỉ dùng được sau D slot (slot dài 1 s). Báo cáo còn đang trên đường khi episode kết thúc bị bỏ.
  - Tracker dự đoán cho mỗi slot bằng calibrator đúng ở trạng thái lúc đầu slot, tức chỉ với các báo cáo đã đến.
  - Feedback kém cũng áp dụng trong 10 episode hiệu chỉnh của warm-start.
  - D = 0, P = 0 cho kết quả giống hệt bản cũ (có test).
- **Giới hạn:** trạng thái hàng đợi mà PPO và slot policy nhìn thấy vẫn là quan sát của môi trường A5, phản ánh tức thì
  số bit đã gửi. Feedback kém chỉ tác động lên mô hình tốc độ của planner.
- **Điều kiện:** (D, P) ∈ {(0, 0), (1, 0), (3, 0), (0, 0,3), (0, 0,6), (3, 0,3), (3, 0,6)}.
- **Bộ điều khiển** (cùng checkpoint PPO, chỉ chọn mode, protocol pre):

  | Mô hình tốc độ | Bộ điều khiển |
  |---|---|
  | cals | np-any 50 ms (chính), n2 50 ms (selective cũ), all không ngân sách (planner đầy đủ) |
  | Shannon (phy) | np-pess 50 ms, n2 50 ms, all (chỉ ở (0, 0) và (3, 0,3)) |

  Thêm một chẩn đoán: cals np-any ở protocol cold, tại (0, 0) và (3, 0,3).
- **Seed:** 8500–8519. Không trùng seed phát triển 8000–8019, seed hiệu chỉnh 9000–9009, seed val/test của A5 hay seed
  A4.
- **Thống kê (theo review):**
  1. Với mỗi seed môi trường: lấy hiệu (bộ điều khiển với checkpoint i) − (PPO checkpoint i), rồi trung bình qua 3
     checkpoint.
  2. Tính khoảng t trên 20 giá trị theo seed (19 bậc tự do).
  
  Trung bình theo từng checkpoint được báo cáo riêng. Ba checkpoint không được tính là quan sát độc lập.
- **Máy chạy:** 6 tiến trình trên 6 nhân, giống A-E8. Thời gian tính cả PPO.

### 1.2 Kết quả: feedback lý tưởng, seed mới

Return so với PPO (PPO: 31,33 ở phy, 58,65 ở ref). Cột "Seed phát triển" là số A-E8; dấu "–" là không có số tương ứng.

| | Seed mới | Seed phát triển | Từng checkpoint | ms TB | p99 | Vượt 50 ms |
|---|---|---|---|---|---|---|
| phy cals np-any 50 ms | **+2,18 [+1,30, +3,06]** | +2,86 | 1,72 / 2,70 / 2,13 | 26,1 | 44,0 | 0,04 % |
| phy cals n2 50 ms | +2,26 [+1,16, +3,36] | +2,39 | 1,42 / 2,79 / 2,57 | 23,3 | 42,1 | 0,02 % |
| phy cals all, không ngân sách | +2,16 [+1,30, +3,03] | +2,82 | | 52,7 | 93,6 | 55 % |
| phy Shannon np-pess 50 ms | +1,32 [−0,13, +2,76] | +1,90 | | 24,8 | 43,7 | 0,02 % |
| phy Shannon n2 50 ms | +0,93 [−0,01, +1,87] | +0,71 | | 25,5 | 42,8 | 0,01 % |
| phy Shannon all | +0,42 [−1,01, +1,84] | +1,30 | | 52,4 | 93,7 | 53 % |
| ref cals np-any 50 ms | **+1,91 [+1,42, +2,39]** | +1,96 | 2,21 / 2,17 / 1,34 | 22,0 | 42,8 | 0 |
| ref cals n2 50 ms | +2,08 [+1,53, +2,62] | +2,22 | | 20,0 | 40,6 | 0 |
| ref cals all | +1,68 [+1,14, +2,22] | +2,23 | | 50,4 | 84,3 | 48 % |
| phy cals np-any, **cold** | +0,65 [−0,12, +1,43] | – | | 21,4 | 43,9 | 0,01 % |
| ref cals np-any, **cold** | +0,96 [+0,54, +1,38] | – | | 19,0 | 42,6 | 0 |

So sánh ghép cặp ở (0, 0):

| Cặp | Hiệu |
|---|---|
| phy cals np-any − n2 | −0,08 [−0,78, +0,62] |
| phy cals np-any − all | +0,02 [−0,83, +0,87] |
| ref cals np-any − n2 | −0,17 [−0,53, +0,19] |
| ref cals np-any − all | +0,22 [−0,25, +0,69] |
| phy Shannon np-pess − n2 | +0,39 [−0,52, +1,30] |
| phy Shannon np-pess − all | +0,90 [+0,06, +1,74] |

### 1.3 Kết quả: suy giảm theo feedback

Hiệu (điều kiện) − (0, 0) của cùng bộ điều khiển, ghép cặp theo seed.

| Bộ điều khiển | (1, 0) | (3, 0) | (0, .3) | (0, .6) | (3, .3) | (3, .6) |
|---|---|---|---|---|---|---|
| phy cals np-any | −0,05 | −0,15 | −0,09 | +0,11 | +0,05 | +0,36 |
| phy cals n2 | −0,94 [−1,95, +0,07] | −0,38 | −0,12 | −0,55 | +0,36 | −0,32 |
| phy cals all | +0,50 | +0,22 | −0,06 | −0,40 | +0,54 | −0,64 [−1,42, +0,15] |
| phy Shannon np-pess | −0,03 | −0,06 | +0,20 | −0,30 | −0,45 | +0,47 |
| phy Shannon n2 | +0,01 | −0,49 | −0,01 | −0,17 | −0,09 | +0,11 |
| ref cals np-any | +0,03 | +0,03 | −0,02 | +0,19 | −0,08 | −0,09 |
| ref cals n2 | −0,02 | −0,18 | −0,16 | −0,13 | −0,10 | +0,09 |
| ref cals all | −0,38 | +0,13 | +0,08 | −0,15 | +0,13 | +0,10 |

- Mọi CI đều chứa 0. Độ rộng CI ở phy khoảng ±0,8–1,1, ở ref khoảng ±0,4–0,6.
- Không có xu hướng đơn điệu theo D hay P.
- Bộ điều khiển chính (np-any) ổn định nhất: |hiệu| ≤ 0,36 ở cả hai regime.
- Cold (3, .3) − cold (0, 0): −0,08 (phy), −0,29 [−0,74, +0,17] (ref).

### 1.4 Thời gian và fallback

| | p99 | Vượt 50 ms | Ngoại lệ lớn nhất | Fallback/cắt ngắn do ngân sách |
|---|---|---|---|---|
| 50 ms, mọi điều kiện, phy | 41–44 ms | 0–0,04 % | 105 ms (np-any, (3, .3)), 110 ms (Shannon n2, (1, 0)), 139 ms (n2, (0, .6)) | ≤ 2,2 % (chủ yếu trunc_full / skip_full) |
| 50 ms, mọi điều kiện, ref | 40–43 ms | 0 | 47,8 ms | 0–0,05 % |

- Với warm-start, không có quyết định nào phải dùng PPO vì tracker thiếu dữ liệu.
- Ở cold, khoảng 22–32 % quyết định ở phy phải dùng PPO vì lý do đó.

### 1.5 Đọc kết quả

1. **Lợi ích không phải sản phẩm của việc chọn seed.**
   - Trên seed mới, ở mức seed, bộ điều khiển chính vẫn hơn PPO khoảng +2,2 (phy) và +1,9 (ref), CI > 0.
   - Mức ở phy thấp hơn seed phát triển (+2,86), nhưng vẫn nằm trong CI cũ.
   - Cả ba checkpoint đều dương.
2. **Selective không thua planner đầy đủ, nhưng np-any cũng không hơn n2 trên seed mới.** Lựa chọn np-any làm biến thể
   chính được giữ vì chi phí tương đương (26 so với 23 ms) và recall tốt hơn (A-E8), không phải vì return cao hơn.
3. **Feedback trễ/mất ở mức 1–3 slot, 30–60 % không thành vấn đề vì warm-start đã cung cấp phần lớn dữ liệu.** Đây là
   điều cần nói thẳng trong bài: thuật toán phụ thuộc vào một mô hình tốc độ được hiệu chỉnh trước. Feedback trong
   episode chỉ là phần bổ sung. Ở cold, feedback kém làm tăng số quyết định thiếu dữ liệu khoảng 50 %, nhưng return
   giảm không rõ (−0,08 / −0,29), vì khi thiếu dữ liệu bộ điều khiển trả về PPO.
4. **Shannon dưới MCS:**
   - Ứng viên bi quan vẫn tốt hơn planner đầy đủ (+0,90, CI > 0).
   - Lợi thế so với top-2 trên seed mới (+0,39) yếu hơn seed phát triển (+1,19) và CI chứa 0.
   - Kết luận "ứng viên bi quan hữu ích khi mô hình lệch" chỉ còn là xu hướng, chưa được xác nhận ngoài tập phát triển.
5. **Ngân sách vẫn là cooperative.** Có các ngoại lệ đơn lẻ 105–139 ms trong khoảng 9 000 quyết định mỗi biến thể,
   nhiều khả năng do lịch hệ điều hành hoặc gom rác khi 6 tiến trình chạy trên 6 nhân. Không có bảo đảm cứng.

## 2. B-E8: phân bổ bit theo nhiệm vụ bên trong HEVC liên khung

### 2.1 Cách làm

**Encoder**
- `x265roi.c` là một adapter C nhỏ, gọi trực tiếp API libx265 4.1 và đặt `x265_picture.quantOffsets`: mỗi frame có một
  offset QP cho từng block 16×16 (35×20 block).
- x265 cộng offset này lên quyết định của rate control CRF và adaptive quantization.
- Cấu hình giống hệt HEVC của B-E7 (preset slow, tune zerolatency, keyint vô hạn, không B-frame, không lookahead, không
  scenecut, không SEI phiên bản).
- Encoder chỉ nhận frame t và bản đồ offset của frame t trước khi mã frame t.
- Test xác nhận:
  - đổi frame và offset tương lai thì các frame quá khứ đã giải mã và packet của chúng giữ nguyên;
  - không có offset thì kết quả trùng đường ffmpeg của B-E7: bit lệch ≤ 3 %, ảnh giải mã gần như trùng;
  - offset âm làm giảm sai số trong vùng. Trên frame VisDrone thật, offset −12 ở CRF 38 cho MSE vùng 80, gần mức CRF 26
    đồng đều (71).
- Offset QP nằm ngay trong bitstream HEVC (cu_qp_delta), nên bộ thu không cần thông tin phụ. Số bit là toàn bộ stream,
  như B-E7.

**Bản đồ quan trọng của frame t** (tính ở phía gửi, chỉ từ các frame ≤ t)

| Phương pháp | Cách tính |
|---|---|
| uniform | không có offset |
| motion | phần dư độ sáng sau khi bù chuyển động camera toàn cục (dịch nguyên pixel, phase correlation) |
| task | UAV chạy DINOv2-S + head cố định trên frame thô t; lấy xác suất lớp lớn nhất theo patch |
| taskev | max(p_t, \|p_t − p_t−1 đã bù chuyển động\|): thêm thay đổi điểm nhiệm vụ (vật xuất hiện, biến mất, di chuyển) |
| oracle | **chỉ để chẩn đoán**, không triển khai được: nhãn GT của frame t |

**Phân bổ theo hạng**
- Các block có độ quan trọng > 0,1, tối đa f·700 block cao nhất, nhận offset −A.
- Các block còn lại nhận offset +A·n/(700 − n), để offset trung bình bằng 0.
- Tốc độ bit không cố định. Các phương pháp được so ở cùng số bit thật: quét CRF 30…45 rồi nội suy theo log(bit) cho
  từng clip, như B-E7.

**Giao thức**
- (A, f) ∈ {4, 8} × {0,15, 0,3} được chọn riêng cho từng phương pháp trên 4 clip **dev**: các clip TUNE dài nhất, mỗi
  clip từ một video nguồn khác nhau và không trùng nguồn nào của tập đánh giá.
- Tiêu chí chọn: trung bình hiệu AP và ap_change so với uniform ở 6 / 9 / 12 kbit.
- Sau đó chạy **một lần** trên 10 clip đánh giá của B-E7, với cùng DINOv2 và head.
- Kết quả chọn: motion A4 f0,3; task A4 f0,15; taskev A4 f0,15; oracle A8 f0,3.
- Lưu ý: head được fit trên các clip TUNE, nên điểm trên clip dev là lạc quan.

**Chi phí phía gửi**
- DINOv2-S + head: 8–9 ms/frame trên RTX 3070 (batch 1, fp16). Trên UAV sẽ cần bộ tăng tốc nhúng.
- Bản đồ chuyển động: khoảng 10 ms/frame bằng numpy (chưa tối ưu).
- Tạo bản đồ và offset từ điểm số: khoảng 0,5 ms.

### 2.2 Kết quả (10 clip đánh giá)

AP trung bình trên các clip mà mọi phương pháp đều nằm trong dải bit:

| kbit/frame (số clip) | uniform | motion | task | taskev | oracle (GT) |
|---|---|---|---|---|---|
| 4,5 (8) | 0,346 | 0,336 | 0,359 | 0,362 | 0,359 |
| 6 (9) | 0,376 | 0,369 | 0,382 | 0,385 | 0,385 |
| 9 (10) | 0,404 | 0,397 | 0,411 | 0,413 | 0,413 |
| 12 (10) | 0,421 | 0,412 | 0,426 | 0,427 | 0,428 |

Hiệu ghép cặp so với uniform (t-interval qua clip, kèm số clip dương):

| | 4,5k | 6k | 9k | 12k | 18k |
|---|---|---|---|---|---|
| task, AP | **+0,013 [+0,003, +0,023]** 7/9 | +0,006 [−0,002, +0,015] | +0,007 [−0,002, +0,016] | +0,005 [−0,001, +0,012] | +0,002 |
| taskev, AP | **+0,016 [+0,006, +0,026]** 8/9 | **+0,009 [+0,003, +0,016]** 9/10 | +0,009 [−0,001, +0,019] | +0,007 [−0,001, +0,014] | +0,004 |
| taskev, ap_change | +0,010 [−0,007, +0,026] | +0,005 [−0,004, +0,015] | +0,002 [−0,004, +0,007] | +0,002 [−0,005, +0,009] | +0,003 |
| taskev, ap_event | **+0,015 [+0,001, +0,030]** | **+0,011 [+0,001, +0,021]** | +0,009 [−0,002, +0,019] | +0,004 | +0,004 |
| motion, AP | −0,009 | −0,008 | −0,007 | −0,008 [−0,015, −0,002] | −0,008 |
| oracle, AP | +0,013 [+0,001, +0,025] | +0,009 | +0,010 | +0,008 | +0,009 |
| oracle, ap_change | +0,001 | −0,000 | −0,000 | −0,002 | −0,002 |

- **Tiết kiệm bit ở cùng AP** (chỉ tính các clip có đường AP–bit tăng đơn điệu, 5–6/10 clip): task/taskev khoảng 11–13 %
  ở 4,5–6 kbit (CI > 0) và khoảng 11 % ở 9–12 kbit (CI chứa 0). Với ap_change: khoảng 0 %.
- **PSNR ở cùng số bit:** task/taskev thấp hơn uniform khoảng 0,25–0,3 dB, đúng như kỳ vọng khi chuyển bit từ nền sang
  vật thể.
- **taskev so với task:** giống hệt nhau trên 7/10 clip (bitstream trùng từng bit). Với f = 0,15, top các block luôn
  là các block có xác suất vật thể cao, nên số hạng sự kiện chỉ thay đổi phân bổ ở 3 clip. Hiệu taskev − task là
  +0,001…+0,006 với CI chứa 0.
- **So với dev:** trên dev, task đạt khoảng +0,019 AP và +0,020 ap_change ở 9 kbit. Trên tập đánh giá, mức tăng AP chỉ
  còn khoảng một nửa và **mức tăng ap_change biến mất**.

### 2.3 Đọc kết quả

1. **Phân bổ bit theo nhiệm vụ trong HEVC có tác dụng nhưng nhỏ.**
   - Khoảng +0,01 AP, tức khoảng 10–13 % bit ở cùng AP, ở vùng 4,5–6 kbit/frame. Có ý nghĩa thống kê ở mức bit thấp
     nhất, mờ dần khi tăng bit.
   - Để so sánh: khoảng cách giữa codec ngữ nghĩa và HEVC ở B-E7 là 0,04–0,09.
2. **Thứ cần nhất cho hướng "event-sensitive" là ap_change, và nó chưa cải thiện.**
   - ap_event có tăng ở 4,5–6 kbit, nhưng chủ yếu do frame sự kiện có nhiều vật thể: task ROI đã tăng AP chung.
   - Bản taskev hiện tại gần như không kích hoạt số hạng sự kiện. Đây là lỗi thiết kế của cách phân bổ theo hạng
     (thay đổi nhiệm vụ không bao giờ vượt được xác suất vật thể), không phải bằng chứng rằng thông tin sự kiện vô dụng.
3. **ROI theo nhãn GT (oracle) không tốt hơn ROI theo DINOv2.** Nhãn patch không phải bản đồ quan trọng tối ưu cho
   bộ thu DINOv2. Mức trần của kiểu "ưu tiên vùng có vật" vì vậy cũng chỉ khoảng +0,01.
4. **Motion-only gây hại** (khoảng −0,008 AP): phần dư chuyển động phần lớn là nền có kết cấu, không phải vật thể.
5. **Chi phí:** phía gửi phải chạy DINOv2 cho mỗi frame (8–9 ms trên GPU desktop) để được khoảng 10 % bit. Trên UAV,
   đây là một trade-off cần cân nhắc.
6. **Theo tiêu chí của review #34** ("nếu cải thiện ap_change hoặc ap_event ở cùng bitrate thì đáng đầu tư"):
   - ap_event có cải thiện nhỏ, có ý nghĩa ở mức bit thấp;
   - ap_change thì không.
   
   Tôi đánh giá kết quả là **yếu / chưa kết luận được về phần event-sensitive**, và có hai lựa chọn cho phản biện:
   - (b′) Một vòng nhỏ nữa cho số hạng sự kiện: dành riêng một hạn mức block cho \|Δp\| lớn nhất, độc lập với p_t, và
     đo trực tiếp trên ap_change. Tối đa 2 cấu hình, chọn trên dev.
   - (a) Chuyển sang lớp tăng cường ngữ nghĩa trên nền video như review đã đề cập.

### 2.4 Giới hạn

- 10 clip đánh giá, mỗi clip ≤ 100 frame, 4 clip dev.
- Lưới (A, f) nhỏ. Chỉ phân bổ theo hạng với hai mức offset.
- Bản đồ quan trọng ở độ phân giải block 16 px, trong khi x265 gom offset theo nhóm lượng tử 32×32 (qg-size mặc định).
- Bù chuyển động toàn cục chỉ là dịch nguyên pixel.
- Chưa thử AV1 hay feedback từ bộ thu.
- Head không được fit lại trên token của video đã giải mã.
- Chỉ số tiết kiệm bit dựa trên các clip có đường đơn điệu (5–6/10).

## 3. Đề xuất bước tiếp theo

**A**
- Cố định bản hiện tại (np-any với cals; np-pess khi mô hình lệch mạnh) làm phương pháp chính và viết Method. Bản nháp:
  `reports/swarmbridge-paperA-method-draft.md`.
- Phải nêu rõ sự phụ thuộc vào warm-start.
- Nếu cần thêm thí nghiệm, nên làm một phép thử warm-start cũ hoặc lệch, ví dụ hiệu chỉnh ở regime ref rồi chạy ở
  phy_mcs_blind, thay vì tiếp tục tăng độ trễ.

**B**
- Xin phản biện chọn (b′) hoặc (a), như ở mục 2.3.
