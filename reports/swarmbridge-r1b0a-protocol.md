# SwarmBridge R1b-0a: protocol (bản nháp để khoá; chỉ kiểm H1)

Ngày 10/11/2026. Dựa trên thiết kế v3 ([swarmbridge-r1b-design.md](swarmbridge-r1b-design.md)) và quyết định phản biện ngày 10/11.

**Trạng thái:**
- Đã có: split v1, code mô phỏng, 25 unit test (đều qua), đo thời gian tính toán trên CPU.
- **Chưa chạy thí nghiệm chính.** Blocker: chưa có xác nhận phạm vi Active Semantic Recovery (mục 9).
- Các số đánh dấu **[khoá trước khi chạy]** sẽ điền theo đúng quy tắc ở đây từ dữ liệu TUNE, rồi commit trước khi chấm G7.

## 1. Dữ liệu (data-integrity amendment, đã duyệt)

File `runs/uav_semcom/r1b/split_v1.json`, sinh bằng `r1b_split.py` từ `vid_audit.json`.

| Tập | Clip | Khung gốc | Vai trò |
|---|---|---|---|
| **G7 (cổng)** | 020_00406, 138_00000, 244_01440, 278_00001, 289_00001, 295_02300, 316_01288 | 3.370 | Chấm H1 một lần |
| Cách ly | 289_06922 (cùng nguồn 289 với G7) | 210 | Không dùng ở đâu |
| TUNE | 48 clip VID-train còn lại | 20.621 | Tinh chỉnh, đo, khoá |
| AUX | VID-val 086, 117, 268 | 1.791 | Kết quả phụ, **không** quyết định cổng |

**Quy tắc chọn:**
- **Nguồn đủ điều kiện:** không có số nguồn trong DET-train và không có khung trùng DET-train đã xác nhận. Có 8 nguồn đạt: 20, 138, 244, 278, 289, 295, 316, 360.
- **G7** lấy 7 nguồn có số nhỏ nhất, mỗi nguồn một clip (clip có tên nhỏ nhất). Nguồn 360 không dùng và nằm trong TUNE.
- **Lý do đổi từ bản trước:** bản trước chọn theo tên clip nên có hai clip cùng nguồn 289. Giờ mỗi clip G7 là **một nguồn độc lập**, nên "5/7 clip" đọc là "5/7 nguồn".

**Kiểm tra lại trùng lặp trước khi khoá:**
- So pHash mọi khung (25.992) giữa G7, AUX, cách ly và TUNE, và giữa các nguồn G7 với nhau. Kết quả: **0 cặp xác nhận** (NCC ≥ 0,9).
- 5 cặp nghi ở đúng ngưỡng Hamming 10 đều có NCC < 0,9. Xem tay 3 cặp tốt nhất (244/239, 278/309, 268/243): cảnh khác nhau, chỉ giống bố cục phố.
- Script kiểm cấu trúc split: 0 vi phạm. Unit test `test_split_v1` khoá danh sách này.

**Hạn chế còn lại:**
- G7 có cảnh trùng **DET-val** (pool đánh giá R1a): 244, 289, 295, 316. Đây là cổng khả thi có kiểm soát, không phải tập xác nhận mới hoàn toàn.
- pHash không bắt được kiểu "cùng thành phố, khác địa điểm".

## 2. Phát lại, độ phân giải, chỉ số

- **Nhịp phát lại:** lấy mỗi 3 khung gốc, giả định 10 khung/s. Đây là **giả định phát lại**, không phải fps thật đã xác nhận của VisDrone-VID, và được báo kèm mọi kết quả.
- **Độ phân giải xử lý:** khung thu về cạnh dài **1344 px**, bằng độ phân giải VID nhỏ nhất. GT và detection theo cùng tỉ lệ.
- **GT được chấm:** lớp 1–10 với score = 1. Vùng bỏ qua là lớp 0 và lớp 11. Output không khớp GT mà nằm > 50% trong vùng bỏ qua thì bị loại, không tính FP.
- **Chỉ số** (`r1b_metrics.py`), trên output của tracker BS dự đoán tới tick f, chỉ dùng gói đã xử lý trước t_f:
  - object-time P / R / F1, class-agnostic, IoU ≥ 0,5, Hungarian theo từng tick (**chỉ số chính**);
  - IDF1;
  - FP-giây / phút; số track không bao giờ khớp và thời lượng trung bình của chúng;
  - độ trễ báo lần đầu (trung vị, p90) của track GT có ≥ 3 tick, cùng tỉ lệ không bao giờ được báo;
  - UL / DL kbit/s, ms UAV / BS mỗi tick, số khung bị bỏ vì bận tính toán, số gói trễ hạn hoặc mất, belief F1 (chẩn đoán: \hat T_f so với bảng BS thật).

## 3. Điều kiện thông tin (`r1b_sim.py`)

Mọi điều kiện dùng chung: detector, tracker (`r1b_track.py`, cùng tham số), menu SKIP / TOK / FRAME, lớp policy, kênh và ngân sách. Chỉ cách tạo \hat T_f khác nhau.

| Mã | \hat T_f | Downlink |
|---|---|---|
| A0 (phụ) | Bản sao ảo của tracker BS, nạp những gì UAV đã gửi **ngay lúc gửi**. FRAME → detection YOLO11n của chính UAV làm proxy | 0 |
| **A1** | Như A0, nhưng chỉ nạp khi ACK tới, theo thứ tự chụp. Thứ chưa ACK coi như BS **chưa có** (thận trọng) | 32 bit / gói được xử lý, gộp theo lượt gửi |
| **B** | Box BS tại khung đã xử lý gần nhất, không id. Box khớp track cục bộ của UAV tại tick đó thì đi theo track ấy; còn lại đứng yên | 24 + 52 bit/box |
| C0 (phụ) | Delta trạng thái, không phiên bản / băm; snapshot định kỳ | 60 + 87 bit/op (xoá 12) |
| **C1** | Delta có phiên bản + CRC32 tại t_state + snapshot định kỳ và khi bị yêu cầu; header uplink 48 bit | 124 + 87 bit/op (xoá 12) |
| **D** | Bảng thật của tracker BS tại thời điểm quyết định (không triển khai được) | 0 |

**Downlink chung:**
- BS gửi tối đa một thông điệp mỗi khi link downlink rảnh và gộp nội dung, nên không có hàng đợi downlink.
- Thêm 96 bit header ứng dụng mỗi gói, ở cả hai chiều.

**Sửa định dạng so với v3** (trước khi có kết quả):
- Toạ độ dùng 12 bit (1/4096 khung) thay 8 bit. 1/256 khung ở 1344 px là khoảng 5 px, quá thô cho vật nhỏ.
- Vận tốc 10 bit có dấu.
- Header C1: 124 bit.

## 4. Kênh, deadline, ngân sách

- **Uplink:**
  - FIFO tốc độ R_ul, trễ lan truyền 10 ms;
  - gói ≤ 1200 byte; mất Bernoulli 1% mỗi gói, thông điệp mất nếu mất bất kỳ gói nào;
  - **D_max = 300 ms** tính từ lúc chụp tới lúc tới BS. Trễ thì bỏ nhưng **vẫn tính bit**.
- **Kiểm soát nhận gói phía UAV** (thông tin cục bộ, như nhau ở mọi điều kiện): hành động bị hạ xuống hành động rẻ hơn nếu, theo hàng đợi uplink của chính UAV và kích thước ước lượng, nó sẽ trễ deadline.
  - Kích thước FRAME ước lượng bằng EMA, khởi tạo bằng trung bình TUNE; không bao giờ dùng kích thước của chính clip.
- **Token bucket** ở R_ul, độ sâu 1 s.
- **Downlink:** link tốc độ DL_cap = 64 kbit/s (16 chỉ báo); d_dl và p_dl theo kịch bản (N: 100 ms, 1%).

## 5. Tính toán và thời gian nhân quả (mới; trả lời phản biện về timing)

**Đo trên VID TUNE** (30 khung từ 30 clip, cạnh dài 1344, trung vị; `runs/uav_semcom/r1b/timing_tune.json`). Máy đo là desktop 6 luồng, **không phải phần cứng UAV**.

| Việc | ms (trung vị / p90) |
|---|---|
| YOLO11n @640, 6 luồng | 29 / 33 |
| YOLO11n @960, 6 luồng | 47 / 52 |
| YOLO11n @640, 1 luồng | 59 / 62 |
| AVIF q20 mã hoá (145 kbit trung vị) | 92 / 118 |
| AVIF q35 mã hoá (261 kbit) | 104 / 139 |
| AVIF q50 mã hoá (516 kbit) | 130 / 174 |
| AVIF giải mã ở BS | 5 |
| YOLO11s ở BS (GPU, R1a) | 13 |

**Mô hình chính: "serial".**
- UAV có **một** bộ xử lý làm tuần tự: detect → track / policy (2 ms) → mã hoá nếu FRAME.
- Khung chụp lúc bộ xử lý còn bận thì **bị bỏ**: không detect, không quyết định, tracker cục bộ không cập nhật.
- Thời điểm gói sẵn sàng bằng thời điểm chụp cộng toàn bộ thời gian tính toán, và deadline tính từ lúc chụp.

**Hệ quả đã thấy trên dữ liệu tổng hợp và trong unit test:**
- 29 + 2 + 92 = 123 ms > 100 ms, nên **mỗi FRAME làm mất tick kế tiếp**.
- Không hành động nào được coi là hoàn thành tức thời.

**Phân tích độ nhạy: "pipelined".** Detector chạy trên bộ tăng tốc riêng, mã hoá trên CPU. Hai tài nguyên, mỗi cái có hàng đợi riêng. Chỉ báo, không là tiêu chí.

**BS:** một hàng đợi GPU.
- FRAME tốn 20 ms (giải mã + YOLO11s + tracker); TOK tốn 1 ms.
- Phản hồi chỉ được tạo sau khi xử lý xong.

## 6. Các số khoá trước khi chạy

| Mục | Quy tắc | Giá trị |
|---|---|---|
| FRAME | AVIF speed 6, 4:2:0, cạnh dài 1344, q20. Đổi sang q35 nếu trên TUNE recall YOLO11s trên khung giải mã thấp hơn khung gốc > 5 điểm | [khoá trước khi chạy] |
| c_det, c_enc | Trung vị TUNE ở cấu hình đã khoá | 29 ms; 92 ms (q20) |
| frame_bits0 | Trung bình bit FRAME trên toàn TUNE | [khoá trước khi chạy] |
| R_F | Tốc độ tối thiểu để một FRAME trung vị kịp deadline khi hàng đợi rỗng: (bit + header) / (D_max − c_det − c_trk − c_enc − d_ul) | ≈ 146 kbit / 167 ms ≈ **0,88 Mbit/s** (q20, số đo sơ bộ) |
| R_ul | {1, 2, 4} × R_F | ≈ {0,88; 1,75; 3,5} Mbit/s [khoá theo R_F cuối] |
| DL_cap | 64 kbit/s (16 phụ) | |
| D_max, p_ul, d_ul | 300 ms, 1%, 10 ms | |
| N_ref, W_hold, N_hist | 20 tick, 10 tick, 64 phiên bản | |
| Tracker | iou_gate 0,3; conf_high 0,5; conf_new 0,4; conf_confirm 0,6; alpha 0,7; beta 0,3; max_miss 2 FRAME; max_age 50 tick | |
| Seed kênh | 0 (cùng seed cho mọi điều kiện) | |

**Tại sao R_ul tính theo R_F:** dưới R_F, FRAME gần như không bao giờ kịp deadline, và thí nghiệm chỉ còn là chọn token. Đặt mức thấp nhất bằng R_F giữ cả ba hành động khả thi ở mọi mức.

## 7. Policy và tinh chỉnh

**Lớp policy chung:**
- Điểm lệch g = w_new·#mới + w_lost·#mất + w_drift·#trôi + w_unc·diện tích không chắc (%).
- Luật: FRAME nếu g ≥ θ_F hoặc đã ≥ N_max tick không có FRAME; TOK nếu g ≥ θ_T (chỉ gửi đối tượng mới / trôi, tối đa 64); còn lại SKIP.
- Cả hai đều qua kiểm tra bucket và deadline.
- k_lost = 3.

**Lưới** (256 cấu hình):
- θ_T ∈ {0,5; 1; 2; 4};
- θ_F ∈ {2; 4; 8; ∞};
- N_max ∈ {30; ∞};
- w_new = 1; w_lost ∈ {0; 1}; w_drift ∈ {0; 1}; w_unc ∈ {0; 0,5}.

**Tinh chỉnh:**
- Mỗi điều kiện × mỗi mức R_ul tinh chỉnh **riêng** trên **48 clip TUNE**, kịch bản N.
- Chọn F1 gộp lớn nhất; hoà thì lấy UL thấp hơn. UL ≤ R_ul và DL ≤ cap tự đảm bảo nhờ bucket và link.
- **Độ nhạy:** tham số chọn cho A1 áp cho mọi điều kiện.

## 8. Cổng H1 (G7, kịch bản N, DL_cap 64)

ΔF1 là hiệu object-time F1 gộp 7 nguồn, đơn vị điểm phần trăm. "Thắng" một mức R_ul nghĩa là: ΔF1(C1 − A1) gộp ≥ +2,0 **và** ΔF1 > 0 ở **≥ 5/7 nguồn**.

| Kết luận | Điều kiện |
|---|---|
| **PASS** | Thắng ở ≥ 2/3 mức R_ul, **và** C1 − B ≥ 0 gộp ở các mức đó, **và** không mức nào C1 − A1 < −1 |
| **FAIL (không có dư địa)** | D − A1 < +2 ở mọi mức |
| **FAIL (phản hồi có giá không dùng được)** | D − A1 ≥ +2 ở ít nhất một mức nhưng C1 − A1 < +1 ở mọi mức. Báo rõ và **dừng** semantic ACK |
| **INCONCLUSIVE** | Các trường hợp còn lại |

**Báo cáo bắt buộc:**
- theo từng nguồn: mọi chỉ số ở mục 2;
- gộp 7 nguồn kèm CI bootstrap theo nguồn (B = 2000), **chỉ để mô tả**;
- A0, C0, mô hình pipelined, DL_cap 16 và 3 clip AUX: phân tích phụ.

**Sau cổng:**
- H1 FAIL: báo nguyên nhân, dừng mở rộng phản hồi ngữ nghĩa.
- H1 PASS hoặc INCONCLUSIVE: báo theo từng nguồn và đề xuất phép thử tiếp (H2 với S1 / S2 đã khoá ở thiết kế v3 mục 7).

Không train PPO / SAC, không chạy R2, không tải VID test-dev, chưa làm H3 / x264.

## 9. Active Semantic Recovery: blocker

Claude **không đọc** nhánh ASR. Đề nghị người phụ trách (Codex) làm một **kiểm toán phạm vi chỉ đọc**: không sửa, không chạy lại. Câu hỏi và bảng cần điền ở mục 10 của thiết kế v3 (bản cập nhật).

**Nếu ASR đã có đồng bộ niềm tin bên nhận tương đương C1:**
- không xây lại và gọi đó là đóng góp mới;
- R1b-0a khi ấy chỉ còn giá trị như phép đo độc lập giá trị gia tăng của phản hồi (H1), và phải trích dẫn ASR.

## 10. Kiểm thử (`test_r1b.py`, 25/25 qua)

| Nhóm | Phép thử |
|---|---|
| Giao thức (thiết kế v3 mục 4) | Quay vòng 16 bit (phiên bản và t_state); 10⁵ thông điệp với mất 10%, lặp 5%, đảo thứ tự → bản sao bằng đúng trạng thái công bố sau mỗi snapshot (> 3.000 lần kiểm); CRC bắt 100% sai lệch ±1 và cả cùng nội dung khác thời điểm; op hỏng → desync; dự đoán số nguyên trùng bit sau 1.000 tick |
| Chỉ số | GT làm output → P = R = F1 = IDF1 = 100, trễ 0; trễ cấy 7 tick → trễ báo đúng 0,7 s; vùng bỏ qua không tính FP |
| Tracker | GT làm detection → F1, IDF1 > 99 |
| Nhân quả | Cắt clip sau tick m không đổi hành động trước m (6 điều kiện); khi mất hết downlink, hành động của A0 / A1 / B / C0 / C1 **không phụ thuộc** detection của BS, còn D thì có (rò rỉ trạng thái ẩn bị chặn) |
| Thời gian | Serial: mọi FRAME (≥ 29 + 2 + 92 = 123 ms) làm mất tick kế; pipelined thì không; kiểm soát nhận gói từ chối FRAME không kịp deadline; FRAME trễ bị BS bỏ nhưng vẫn tính bit |
| Split | Danh sách G7 / cách ly / TUNE / AUX đúng, 0 vi phạm, 0 trùng nội bộ |

**Lỗi tìm ra nhờ test** (đã sửa trước khi có bất kỳ kết quả thật nào):
- bộ chấm FP lấy nhầm danh sách nên FP luôn bằng 0;
- trạng thái chuẩn bị lượng tử hoá lại, sinh op cập nhật thừa;
- ước lượng kích thước FRAME lấy trung vị của chính clip, tức là nhìn vào tương lai.

## 11. Việc còn lại trước khi chạy cổng

1. **Xác nhận phạm vi ASR**, hoặc người dùng quyết định khi biết blocker.
2. Tính trước trên TUNE, G7 và AUX (GPU qua lock): YOLO11n @640 trên mọi khung phát lại; AVIF ở cấu hình đã khoá; YOLO11s trên khung giải mã; bit từng khung.
3. Khoá FRAME, frame_bits0, R_F, R_ul theo mục 6. Commit.
4. Tinh chỉnh trên TUNE và commit tham số đã chọn.
5. Chấm G7 **một lần**; viết kết quả H1.
