# Songdo bước 0–1: audit, codec thật và câu hỏi "truyền thông có phải nút thắt không"

Ngày 15/11/2026. Theo phản biện #25: CONDITIONAL GO chỉ cho bước 0–1 (CPU, không train, không tải ảnh/video). Báo cáo này gồm năm deliverable mà phản biện yêu cầu trước khi xin S2/S3. **Kết luận chính: với T1–T3 trên Songdo, truyền thông không phải nút thắt.** Vì vậy S2/S3 không được xin duyệt như một bài toán truyền thông (mục 6).

## 0. Luồng thí nghiệm (đọc trước)

1. **Dữ liệu:** quỹ đạo xe của Songdo Traffic v2, coi như đầu ra perception chạy trên drone (geo-trax, offline).
   - Ngày **6/10/2022** là ngày validation. Phiên **PM1** (15:00–15:30, cả 20 giao lộ) dùng để mô phỏng. Phiên **AM3** chỉ dùng để dựng bảng hướng làn tĩnh.
   - Ngày 7/10 (phiên PM5) đã bị mở khi kiểm khả thi → **AUDITED-TEST**: chỉ dùng cho audit, không dùng để chọn bất cứ thứ gì.
2. **Mỗi giây, với mỗi giao lộ đang có drone:**
   - drone so trạng thái xe thật với **bản sao (replica)** của trạng thái BS;
   - nếu sai số dự đoán vượt ε, hoặc xe đổi làn, nó gửi bản ghi cập nhật; xe mới gửi bản ghi "sinh"; xe mất 2 giây gửi bản ghi "chết".
3. **Mã hoá:** bản ghi được lượng tử, ghi thành bitstream thật, giải mã lại và kiểm khớp từng trường. Bit được đếm từ bitstream, không từ công thức entropy.
4. **Kênh:** mỗi gói có thể mất (Bernoulli). ACK về sau 1 bước; replica chỉ cập nhật theo ACK, nên cái gì mất sẽ được gửi lại ở bước sau.
5. **Chấm điểm:** chỉ ở những thời điểm có nhãn (mặt nạ ở mục 2). Sau đó cộng bit của mọi giao lộ theo từng giây để ra **tải của cả đội 10 drone**, rồi so với dung lượng đường lên có cơ sở vật lý.

Code: [s1.py](../../../swarmbridge/experiments/songdo/s1.py), [test_s1.py](../../../swarmbridge/experiments/songdo/test_s1.py) (9 test tổng hợp, đều qua), [run_s1.py](../../../scripts/songdo/run_s1.py), [audit_overlap.py](../../../scripts/songdo/audit_overlap.py).
Kết quả: `runs/songdo/s1_val_2022-10-06.json`, `runs/songdo/audit_v2_2022-10-06_PM1_VALIDATION.json`, `runs/songdo/audit_v2_2022-10-07_PM5_AUDITED-TEST.json`. Dữ liệu thô nằm ngoài git.

## 1. Deliverable 1a: audit thời gian, độ phủ và đồng thời (R5, R6)

Script v2 đã sửa theo R5:
- `time_overlap_s` đổi thành `span_intersection_s`: giao của hai khoảng [min t, max t], **không** phải thời gian cùng quan sát;
- thêm `concurrent_observed_seconds`, xuất ra JSON;
- ghép xe xuyên drone **một-một** (láng giềng gần nhất hai chiều, cùng lớp, ≤ 2 m), chỉ trên các giây cùng quan sát, và có báo **tỉ lệ va chạm**;
- khoảng cách chỉ tính trên các cặp được chấp nhận (≥ 3 giây).

| | 6/10 PM1 (VALIDATION) | 7/10 PM5 (AUDITED-TEST) |
|---|---|---|
| Số dòng (30 Hz) | 18,36 triệu | 26,8 triệu |
| Số giây có dữ liệu mỗi giao lộ | 177–963 | 171–1.042 |
| Khoảng trống dài nhất mỗi giao lộ | 1–703 s | 1–716 s |
| Giao lộ có ≥ 2 drone (luân phiên) | A, F, J, L, M, N, R, S | A, F, J, L, M, N, R, S |
| Giây có ≥ 2 drone **cùng lúc** | L 13, M 10, R 24, S 45 | R 20, S 51 |
| Cặp xe ghép xuyên drone (được chấp nhận) | L 25, M 19, R 35, S 97 | R 39, S 150 |
| Tỉ lệ va chạm | 0 ở cả bốn | 0 (R), 0,0002 (S) |
| Khoảng cách trung vị / p90 (cặp chấp nhận) | 0,07–0,20 m / 0,55–1,59 m | 0,14 m / 0,76–0,77 m |
| Cặp stream khác giao lộ có giây chung | 265, **không có ứng viên ghép nào** (vùng nhìn không chồng nhau) | 270, không có ứng viên |

- **Kết luận audit:** cấu trúc lặp lại giữa ngày validation và ngày test.
  - Drone tuần tra luân phiên, nên mỗi giao lộ chỉ được nhìn từng đoạn.
  - Vùng nhìn đồng thời chỉ xuất hiện lúc bàn giao (10–51 s mỗi giao lộ) và không bao giờ có giữa hai giao lộ khác nhau.
  - Khoảng cách 0,1–0,2 m giữa hai drone nhỏ hơn sai số phát hiện thông thường. Có thể là dấu vết của bước căn chỉnh geo-trax; chưa đối chiếu video.
- **Tính nhân quả** (geo-trax, arXiv 2411.02136):
  - vị trí: thô, tracker trực tuyến → nhân quả;
  - vận tốc / gia tốc: lọc Gauss σ = 14 khung → **không** nhân quả, bên gửi không dùng (có test kiểm: đổi cột `Vehicle_Speed` thành 999 thì kết quả không đổi);
  - kích thước: ước lượng trên cả vết → không dùng;
  - master frame: tri thức tĩnh.
  - Quỹ đạo được coi là **đầu ra cảm biến giả lập offline**. Chi phí tính toán perception trên drone không nằm trong bài toán này.
- **Phiên dự trữ cho xác nhận cuối:** các phiên chưa mở của ngày 7/10 (AM1–AM5, PM1–PM4). Chưa phiên nào được tải.

## 2. Deliverable 1b: mặt nạ chấm điểm (R2)

| Loại thời điểm | Định nghĩa | Chấm gì |
|---|---|---|
| **Đoạn có quan sát** | giây mà CSV giao lộ có dòng (có drone phía trên) | T2 (sai số vị trí BS, sót, thừa), T1 (đếm theo nhánh mỗi 10 s); mô phỏng chặn gửi / mất gói tại đây |
| **T3 trong đoạn** | gốc s và đích s + h đều nằm trong **cùng** đoạn có quan sát | ADE theo xe ở h = 1, 2, 3 s; chỉ xe có mặt ở cả hai thời điểm |
| **Lúc drone quay lại** | giây đầu tiên sau một khoảng trống | dự đoán của BS **trước** thông điệp mới, ghép Hungarian ≤ 5 m với quan sát; sau đó BS xoá vết của giao lộ (ID không xuyên drone) |
| **Khoảng không có drone** | mọi giây khác | **không chấm**; không có GT |

Có test tổng hợp kiểm mặt nạ: hai đoạn, khoảng trống 11 s, đúng 30 giây được chấm, đúng một lần chấm lúc quay lại.

## 3. Deliverable 2: codec M0 / M1 chạy round-trip (R4)

- **Lượng tử:** vị trí 0,25 m (so với gốc giao lộ), vận tốc 0,25 m/s.
- **Mã:** exp-Golomb có dấu. Bậc k của phần dư vị trí và vận tốc được chọn theo từng gói và ghi trong header.
- **Header ứng dụng (49 bit):** drone 4 bit, seq 12, giây trong ngày 17, số bản ghi 10, k_pos 3, k_vel 3.
- **Bản ghi:**
  - mọi bản ghi: kiểu 2 bit; ID cục bộ mã chênh lệch sau khi sắp xếp (exp-Golomb bậc 3);
  - **sinh:** lớp 2 bit, chỉ số làn, vị trí tuyệt đối; thêm vận tốc 2D (CV) hoặc tốc độ dọc làn (lane);
  - **cập nhật:** phần dư vị trí so với dự đoán, 1 bit đổi làn (+ làn mới), phần dư vận tốc;
  - **chết:** chỉ ID.
- **Gói:** chia đều cho tới khi mỗi gói ≤ MTU 1.400 byte; cộng 28 byte IP/UDP mỗi gói. Mọi gói được giải mã lại, và `assert` kiểm khớp từng trường (chạy trên mọi gói của toàn bộ thí nghiệm).
- **Vòng đời và mất gói:**
  - ACK trên đường xuống 12 bit / gói, báo riêng;
  - replica chỉ áp dụng bản ghi đã được ACK, nên bản ghi mất được gửi lại tự động ở bước sau (xe mới vẫn là "sinh" cho tới khi được xác nhận);
  - BS dùng chung bộ dự đoán với replica.
  - Giả định đã ghi: mất cả gói hoặc không mất; ACK không mất; không có timer truyền lại riêng.
- **M0:** số xe trung bình theo `Road_Section` trong mỗi cửa sổ 10 s, mã phần dư so với cửa sổ trước, cộng một header mỗi cửa sổ.
- **Độ dài mã lý tưởng −log₂ p** không được dùng làm số bit chính.

## 4. Deliverable 3: đường cong bit – chất lượng (validation 6/10 PM1, 20 giao lộ, 12.427 giây có quan sát)

Trung bình có trọng số theo số giây quan sát.
- **Bit UL** = bitstream ứng dụng (header + bản ghi).
- **+IP** = cộng 28 byte mỗi gói.
- Thực tế mỗi giao lộ gửi khoảng 1 gói / giây (0,999).

| Bộ dự đoán | ε (m) | Bit UL / s / giao lộ (min–max) | +IP | T2 sai số TB / p95 (m) | T3 ADE 1 / 2 / 3 s (m) |
|---|---|---|---|---|---|
| giữ nguyên | 0,5 | 529 (161–1.089) | 752 | 0,13 / 0,35 | 3,13 / 5,98 / 8,53 |
| giữ nguyên | 1 | 516 (158–1.071) | 740 | 0,22 / 0,77 | 3,21 / 6,07 / 8,62 |
| giữ nguyên | 2 | 498 | 722 | 0,38 / 1,61 | 3,38 / 6,24 / 8,79 |
| giữ nguyên | 4 | 471 | 694 | 0,73 / 3,28 | 3,73 / 6,60 / 9,16 |
| vận tốc không đổi | 0,5 | 550 (159–1.017) | 773 | 0,12 / 0,32 | **0,62 / 1,53 / 2,66** |
| vận tốc không đổi | 1 | 491 (144–892) | 715 | 0,18 / 0,73 | 0,73 / 1,66 / 2,83 |
| vận tốc không đổi | 2 | 421 | 644 | 0,33 / 1,48 | 0,95 / 1,97 / 3,20 |
| vận tốc không đổi | 4 | **372** (110–651) | 593 | 0,58 / 2,84 | 1,31 / 2,41 / 3,71 |
| theo làn | 0,5 | 547 | 771 | 0,12 / 0,29 | 1,63 / 3,44 / 5,34 |
| theo làn | 1 | 507 | 731 | 0,17 / 0,67 | 1,71 / 3,54 / 5,45 |
| theo làn | 2 | 460 | 684 | 0,29 / 1,40 | 1,88 / 3,75 / 5,70 |
| theo làn | 4 | 423 | 646 | 0,52 / 2,77 | 2,16 / 4,07 / 6,07 |

Tham chiếu:
- **M0** (chỉ số đếm theo nhánh): 9,1 bit/s / giao lộ (7,3–12,3).
- **Toàn bộ trạng thái mỗi giây** (mọi xe, vị trí + vận tốc tuyệt đối): 2.020 bit/s / giao lộ (972–3.459).

Mất gói (ε = 1):

| Bộ dự đoán | p mất | Gói mất | Bit UL / s | T2 sai số TB / p95 | Sót | T1 MAE | T3 ADE 1 s |
|---|---|---|---|---|---|---|---|
| CV | 0 | 0 | 491 | 0,18 / 0,73 | 0 | 0,134 | 0,73 |
| CV | 0,1 | 1.134 / 12.408 | 510 | 0,24 / 0,83 | 0,25 % | 0,156 | 0,81 |
| CV | 0,3 | 3.402 / 12.411 | 556 | 0,40 / 1,41 | 0,84 % | 0,202 | 1,05 |
| lane | 0,3 | 3.402 / 12.414 | 558 | 0,71 / 3,13 | 0,81 % | 0,202 | 2,33 |

Lúc drone quay lại (ε = 1; 53 lần; khoảng trống trung vị 247 s, tối đa 703 s):

| Bộ dự đoán | Tỉ lệ xe thật ghép được ≤ 5 m | Tỉ lệ xe BS dự đoán đúng chỗ | Sai số trên các cặp ghép được |
|---|---|---|---|
| giữ nguyên | 37,3 % | 34,0 % | 1,86 m |
| vận tốc không đổi | 30,3 % | 27,6 % | 1,44 m |
| theo làn | 31,4 % | 28,6 % | 1,67 m |

Đọc kết quả:
- **Bit gần như không phụ thuộc bộ dự đoán.** Từ "giữ nguyên" sang "vận tốc không đổi" ở ε = 1 chỉ giảm 5 % bit ứng dụng (516 → 491).
  - Lý do: giờ cao điểm chiều, phần lớn xe đứng chờ; và mỗi gói có chi phí cố định 49 bit header + 224 bit IP/UDP.
  - Ở ε = 1 với CV, chi phí cố định chiếm khoảng 38 % số bit tính cả IP (273 / 715).
- **Bộ dự đoán quyết định T3, không quyết định bit.** CV giảm ADE 1 s từ 3,2 m xuống 0,7 m với cùng số bit. Đây là chất lượng dự báo, không phải tiết kiệm truyền thông.
- **T1 không phụ thuộc truyền thông:**
  - T1 MAE 0,134 như nhau ở mọi cấu hình không mất gói. Sai số này đến từ những làn không có trong bảng hướng làn (làn không xác định), không từ bit.
  - Tỉ lệ "thừa" 1,9 % đến từ quy tắc báo xe chết sau 2 giây.
- **Mất gói 30 % chỉ thêm khoảng 13 % bit** (gửi lại), sót < 1 %, và sai số vẫn bị chặn quanh ε.
- **Lúc quay lại sau khoảng 4 phút,** chỉ khoảng 30–37 % xe còn ở gần chỗ BS dự đoán; ngoại suy CV còn tệ hơn giữ nguyên. Đây là bài toán **dự báo** khi không có quan sát. Không lượng bit nào sửa được, vì lúc đó không có drone nào ở giao lộ.

## 5. Deliverable 4: dung lượng đường lên và kiểm tra "có bị giới hạn không" (R3)

**Tải của cả đội** = tổng bit (tính cả IP/UDP) của mọi giao lộ đang được quan sát trong từng giây (1.787 giây). Trung bình khoảng 7 giao lộ được quan sát cùng lúc.

| Cấu hình | p50 | p95 | max (kbit/s) |
|---|---|---|---|
| giữ nguyên, ε = 0,5 (đắt nhất không mất gói) | 5,75 | 7,76 | 10,56 |
| CV, ε = 1 | 5,39 | 7,34 | 9,90 |
| CV, ε = 4 (rẻ nhất) | 4,47 | 6,15 | 9,16 |
| CV, ε = 1, mất 30 % | 5,78 | 8,27 | **11,19** |
| lane, ε = 1, mất 30 % | 5,86 | 8,26 | 11,08 |

Kịch bản dung lượng có cơ sở:
- Yêu cầu UAV trong 3GPP và các ứng dụng: Zeng, Wu, Zhang, "Accessing from the sky", Proc. IEEE 2019, bảng I–II.
- Băng 1 MHz dùng chung: Shannon ở SNR 0 / 10 / 20 dB. Với UAV ở độ cao 140–150 m, kênh chủ yếu là LoS.

| Kịch bản | Dung lượng | Tải max / dung lượng | Dư |
|---|---|---|---|
| Đường lên payload UAV (3GPP) | 50 Mbit/s | 0,02 % | > 4.000× |
| Ứng dụng giám sát (UL, trễ 3 s) | 10 Mbit/s | 0,11 % | ~ 900× |
| Băng 1 MHz chung, SNR 0 dB | 1,0 Mbit/s | 1,1 % | ~ 90× |
| Băng 1 MHz chung, SNR 10 / 20 dB | 3,46 / 6,66 Mbit/s | 0,3 / 0,17 % | 300–600× |
| **Chỉ riêng kênh điều khiển C&C** (UL, trễ 50 ms) | 60–100 kbit/s | 11–19 % | **5–9×** |

- **Hạn chót:** bước quyết định là 1 s. Toàn bộ tải max 11,2 kbit của một giây mất khoảng 11 ms ở 1 Mbit/s. Trễ không phải ràng buộc.
- **Kiểm tra bảo thủ hơn:**
  - Gửi **toàn bộ trạng thái mỗi giây** (2,0 kbit/s / giao lộ, tối đa 3,5) với khoảng 7–10 giao lộ cùng lúc: ước tính 14–35 kbit/s cho cả đội. Vẫn vừa kênh C&C.
  - Ngoại suy tuyến tính lên 10 Hz (chưa đo): khoảng 0,14–0,35 Mbit/s, vẫn dưới băng 1 MHz ở SNR 0 dB.
- **Kết luận R3: truyền thông KHÔNG bị giới hạn** với T1–T3 bằng trạng thái có cấu trúc trên Songdo, ở mọi kịch bản có cơ sở vật lý. Chỉ khi tự đặt ngân sách nhỏ hơn khoảng 10 kbit/s cho cả 10 drone thì mới có cạnh tranh băng thông, và phản biện đã không cho phép giới hạn nhân tạo như vậy.

## 6. Deliverable 5: đánh giá novelty / gap và quyết định

**Prior art liên quan trực tiếp (đã xác minh tóm tắt):**
- **Dead-reckoning / nén quỹ đạo có cận sai số:** Cao, Wolfson, Trajcevski, "Spatio-temporal data reduction with deterministic error bounds", VLDB J. 2006; các biến thể dead-reckoning cho cơ sở dữ liệu đối tượng di động (ví dụ Lange, Dürr, Rothermel 2008). M1 của S1 về bản chất chính là kỹ thuật này.
- **Ước lượng từ xa theo ngưỡng:** Sun, Polyanskiy, Uysal, "Sampling of the Wiener process for remote estimation over a channel with random delay", IEEE TIT 2020. Chính sách tối ưu là ngưỡng, và MSE quy về AoI khi lấy mẫu không phụ thuộc quá trình. Có nhiều mở rộng nhiều nguồn chung kênh, và AoII (Maatouk và cộng sự).
- **V2X:** ETSI TR 103 562 / TS 103 324 (Collective Perception): quy tắc chọn đối tượng đưa vào bản tin, theo chuyển động và theo giá trị thông tin. CAM có quy tắc phát theo thay đổi vị trí / hướng / tốc độ. Ngưỡng cụ thể chưa đối chiếu nguyên văn.
- **JEPA cho chuỗi thời gian / truyền thông:** TS-JEPA (arXiv 2406.04853), CHARM (ICML 2026).

| Quyết định | Kết luận | Lý do |
|---|---|---|
| **Lập lịch Songdo (S3) cho T1–T3** | **NO-GO** | Tải cả đội max 11,2 kbit/s (kể cả IP/UDP và mất gói 30 %) so với 60 kbit/s của riêng kênh C&C và 1–50 Mbit/s cho payload. Không có cạnh tranh tài nguyên, nên chẳng có gì để lập lịch. Muốn có bài toán thì phải giới hạn nhân tạo (bị cấm). |
| **Bộ dự đoán học để giảm bit (S2)** như đóng góp truyền thông | **NO-GO** | Cận trên của khoản tiết kiệm nhỏ: kể cả khi bản ghi tốn 0 bit, vẫn còn khoảng 273 bit/s chi phí cố định trên 715 bit/s. Toàn bộ đã thấp hơn dung lượng 1–4 bậc. Bộ dự đoán ảnh hưởng T3 / dự báo lúc quay lại, nhưng đó là bài **dự báo quỹ đạo**, không phải SemCom. |
| **JEPA trên Songdo** | **NO-GO** | Không có nút thắt truyền thông để latent giải quyết. Trạng thái có cấu trúc + dead-reckoning đã đạt T2 ≤ ε với vài trăm bit/s / giao lộ. Đúng như R1, JEPA không bị loại về lý thuyết, nhưng chưa có giá trị gia tăng nào được chứng minh và không có chỗ để chứng minh. |
| **JEPA trên Town3 (S4, hướng J)** | **Không xin duyệt lúc này** | Cùng câu hỏi R3 áp dụng cho Town3: bản đồ BEV 1 kbit/frame cũng rẻ so với mọi đường lên có cơ sở. S4 chỉ có nghĩa nếu nhiệm vụ giữ lại cần nội dung dày (ảnh / đặc trưng) mà bit thật sự bị giới hạn. Phải chứng minh điều đó trước khi đề xuất S4. |
| **Gap còn lại đáng xét (chưa đề xuất)** | để mở | (a) Dự báo trạng thái giao lộ lúc drone quay lại sau 4–12 phút (bài dự báo, không cần truyền thông); (b) lập kế hoạch tuần tra / bàn giao cho 10 drone theo giá trị thông tin (bài điều khiển, không phải băng thông); (c) truyền thông chỉ thành nút thắt khi nội dung là ảnh / video (M3), nhưng Songdo chỉ có 29 video mẫu 60 s và không được tải ảnh trong đợt này. |

## 7. Giới hạn

- Một phiên validation (6/10 PM1, giờ cao điểm chiều). Chưa chạy phiên AM; lưu lượng sáng có thể khác, nhưng tải phải tăng khoảng 6× mới chạm kênh C&C và khoảng 90× mới chạm băng 1 MHz ở 0 dB.
- Quỹ đạo là đầu ra geo-trax, không phải GT gán tay. Kết quả là tương đối so với pipeline đó; chi phí perception trên drone không tính.
- Bước 1 Hz. 10 Hz chỉ là ngoại suy, chưa đo.
- ACK không mất, mất cả gói, kênh Bernoulli độc lập. Chưa mô hình hoá trễ hàng đợi hay ACK trễ nhiều bước.
- Bảng hướng làn dựng từ phiên AM3 của **cùng ngày** validation (không phải ngày test).
- Không có số liệu nào của ngày 7/10 PM5 được dùng để chọn ε, bộ dự đoán hay kịch bản.

## 8. Đề nghị

1. Đóng S2/S3 trên Songdo **như bài toán truyền thông**; ghi kết luận "structured-state dead-reckoning đủ, đường lên không bị giới hạn" như một kết quả âm có số đo.
2. Không xin GPU cho JEPA lúc này (cả Songdo lẫn Town3) cho đến khi có một kịch bản mà bit thật sự bị giới hạn theo nghĩa của R3.
3. Nếu muốn tiếp tục với dữ liệu Songdo: chọn rõ một trong hai hướng không phải SemCom ở mục 6 (dự báo lúc quay lại, hoặc lập kế hoạch tuần tra), mỗi hướng cần một đề cương và lần duyệt riêng.
