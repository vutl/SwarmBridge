# SwarmBridge R1b-0a: protocol (bản nháp để khoá; chỉ kiểm H1)

Ngày 10/11/2026. Dựa trên thiết kế v3 ([swarmbridge-r1b-design.md](swarmbridge-r1b-design.md)) và quyết định phản biện ngày 10/11.

**Trạng thái (cập nhật sau phản biện thứ hai, 10/11):**
- Đã có: split v1, code mô phỏng, đo thời gian tính toán trên CPU.
- Đã sửa theo phản biện:
  - chỉ số theo lớp là chính;
  - kênh ngẫu nhiên ghép cặp, có khoá;
  - downlink chính xác đến từng bit;
  - B+;
  - xử lý quan sát sai thứ tự;
  - chẩn đoán track ma.
- **52 unit test đều qua.** Báo cáo kiểm chứng kỹ thuật: [swarmbridge-r1b0a-validation.md](swarmbridge-r1b0a-validation.md).
- Chưa xem bất kỳ kết quả nào trên G7.
- **Chưa chạy thí nghiệm chính.** Blocker ASR **đã giải** ngày 10/11 (mục 9): ASR không có cơ chế tương đương C1.
- **Protocol khoá ở commit này.** Các số [khoá trước khi chạy] được điền theo đúng quy tắc mục 6 từ dữ liệu TUNE, commit, rồi mới tính trước và chấm G7.
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
  - **chỉ số chính:** object-time P / R / F1 **theo lớp**. TP phải cùng lớp VisDrone (1–10) **và** IoU ≥ 0,5; Hungarian theo từng tick, trong từng lớp. Đúng box sai lớp tính là 1 FP + 1 FN;
  - **phụ:** cùng chỉ số nhưng class-agnostic (chỉ đo định vị), hậu tố `_ag`;
  - IDF1 theo lớp (ghép từng lớp rồi gộp) và IDF1 class-agnostic. Cả hai khớp py-motmetrics 1.4.0 tới 1e-9 trên dữ liệu có đổi id, đổi lớp và FP;
  - lớp YOLO 0–9 ánh xạ sang lớp VisDrone 1–10 khi tính trước detection;
  - FP-giây / phút; số track không bao giờ khớp và thời lượng trung bình của chúng;
  - độ trễ báo lần đầu (trung vị, p90) của track GT có ≥ 3 tick, cùng tỉ lệ không bao giờ được báo;
  - UL / DL kbit/s, ms UAV / BS mỗi tick, số khung bị bỏ vì bận tính toán, số gói trễ hạn hoặc mất, belief F1 (chẩn đoán: \hat T_f so với bảng BS thật).

## 3. Điều kiện thông tin (`r1b_sim.py`)

Mọi điều kiện dùng chung: detector, tracker (`r1b_track.py`, cùng tham số), menu SKIP / TOK / FRAME, lớp policy, kênh và ngân sách. Chỉ cách tạo \hat T_f khác nhau.

| Mã | \hat T_f | Downlink |
|---|---|---|
| A0 (phụ) | Bản sao ảo của tracker BS, nạp những gì UAV đã gửi **ngay lúc gửi**. FRAME → detection YOLO11n của chính UAV làm proxy | 0 |
| **A1** | Như A0, nhưng chỉ nạp khi ACK tới, theo thứ tự chụp. Thứ chưa ACK coi như BS **chưa có** (thận trọng) | 11 + 16 bit / gói được xử lý, gộp theo lượt gửi |
| **B** | Box BS tại khung đã xử lý gần nhất: **chỉ box**, không id, không lớp, không conf. Box khớp track cục bộ của UAV tại tick đó thì đi theo track ấy; còn lại đứng yên | 27 + 48 bit/box |
| **B+** | Như B, thêm **lớp và conf** của từng box. UAV phát hiện được lệch lớp | 27 + 55 bit/box |
| C0 (phụ) | Delta trạng thái, không phiên bản / băm, có cờ snapshot; snapshot định kỳ | 64 + 87 bit/op (xoá 12) |
| **C1** | Delta có phiên bản + CRC32 tại t_state + snapshot định kỳ và khi bị yêu cầu; header uplink 48 bit | 127 + 87 bit/op (xoá 12) |
| **D** | Bảng thật của tracker BS tại thời điểm quyết định (không triển khai được) | 0 |

**Downlink chung:**
- BS gửi tối đa một thông điệp mỗi khi link downlink rảnh và gộp nội dung, nên không có hàng đợi downlink.
- Thêm 96 bit header ứng dụng mỗi gói, ở cả hai chiều.

**Định dạng wire chính xác từng bit** (`r1b_proto.py`):
- Mọi thông điệp downlink và header đồng bộ uplink được **pack thành bytes** rồi **giải mã ở UAV**. Mô phỏng chỉ chuyển bytes, không chuyển đối tượng Python.
- Bit downlink = 8 × số byte (đệm cho tròn byte) + 96 bit header mỗi gói.
- Trường ngoài miền thì báo lỗi, không cắt ngầm.
- Một track mà dự đoán số nguyên rời miền biểu diễn được thì bị **xoá** khỏi trạng thái công bố, nên snapshot nào cũng serialise được.

**Sửa định dạng so với v3** (trước khi có kết quả):
- Toạ độ dùng 12 bit (1/4096 khung) thay 8 bit. 1/256 khung ở 1344 px là khoảng 5 px, quá thô cho vật nhỏ.
- Vận tốc 10 bit có dấu.
- `n_ops` 11 bit thay 8 bit: id 10 bit nên một delta chạm tối đa 1024 track; 8 bit chỉ chứa 255.
- Header C1 127 bit; header C0 64 bit (thêm cờ snapshot, vì C0 không có trường base).

**Chính sách khi có lớp:** nếu \hat T_f có lớp (A0 / A1 / B+ / C0 / C1 / D) và lớp khác lớp mà UAV thấy, đối tượng được tính là "trôi" và có thể được gửi trong TOK. B không có lớp nên không thấy loại lệch này.

**Giới hạn của menu hành động:** TOK chỉ gửi đối tượng mới hoặc trôi. Nó **không** xoá trực tiếp được một track sai ở BS; chỉ FRAME (hai lần không thấy) mới xoá được.
- Chẩn đoán bắt buộc: track ma ở BS (giây / phút); tỉ lệ thời gian track ma mà UAV **không** thấy vật ở đó; tỉ lệ track ma UAV vừa không thấy vừa biết BS đang giữ.
- Nếu H1 FAIL, **không** được diễn giải là "mọi phản hồi ngữ nghĩa đều vô dụng" khi chẩn đoán cho thấy phần lớn lỗi là track ma mà menu không sửa được.

## 4. Kênh, deadline, ngân sách

- **Uplink:**
  - FIFO tốc độ R_ul, trễ lan truyền 10 ms;
  - gói ≤ 1200 byte; mất Bernoulli 1% mỗi gói, thông điệp mất nếu mất bất kỳ gói nào;
  - **D_max = 300 ms** tính từ lúc chụp tới lúc tới BS. Trễ thì bỏ nhưng **vẫn tính bit**.
- **Kiểm soát nhận gói phía UAV** (thông tin cục bộ, như nhau ở mọi điều kiện): hành động bị hạ xuống hành động rẻ hơn nếu, theo hàng đợi uplink của chính UAV và kích thước ước lượng, nó sẽ trễ deadline.
  - Kích thước FRAME ước lượng bằng EMA, khởi tạo bằng trung bình TUNE; không bao giờ dùng kích thước của chính clip.
- **Token bucket** ở R_ul, độ sâu 1 s.
- Gói vào hàng đợi uplink **lúc mã hoá xong**. Ở mô hình pipelined, một TOK sẵn sàng sớm có thể vượt một FRAME đang mã hoá.
  - BS xử lý quan sát cũ đến muộn theo kiểu "out-of-sequence": detection khớp với track đã có quan sát mới hơn thì bị tiêu thụ, nhưng track không bị kéo lùi và không sinh track trùng.
  - Số quan sát sai thứ tự được báo.
- **Ngẫu nhiên kênh ngoại sinh, ghép cặp giữa các điều kiện.** Mỗi lần rút là một hash có khoá của (seed, luồng, tick chụp hoặc tick gửi, chỉ số thông điệp, chỉ số gói).
  - Luồng UL và DL tách riêng.
  - Gói uplink chụp ở tick f gặp cùng số phận ở mọi điều kiện nếu cùng số gói, không phụ thuộc lịch sử hành động.
  - Trạng thái Gilbert–Elliott là chuỗi theo khe 10 ms, độc lập với lưu lượng.
- **Seed kênh khoá trước:**
  - chấm G7: {0, 1, 2, 3, 4};
  - tinh chỉnh trên TUNE: {1000, 1001}. Không chọn seed sau khi xem kết quả.
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
| R_ul | {1, 2, 4} × R_F (tiêu chí); thêm 0,5 × R_F chỉ báo, nơi FRAME gần như không kịp và token có thể có lợi nhất | ≈ {0,88; 1,75; 3,5} (+0,44) Mbit/s [khoá theo R_F cuối] |
| DL_cap | 64 kbit/s (16 phụ) | |
| D_max, p_ul, d_ul | 300 ms, 1%, 10 ms | |
| N_ref, W_hold, N_hist | 20 tick, 10 tick, 64 phiên bản | |
| Tracker | iou_gate 0,3; conf_high 0,5; conf_new 0,4; conf_confirm 0,6; alpha 0,7; beta 0,3; max_miss 2 FRAME; max_age 50 tick | |
| Seed kênh | G7: {0, 1, 2, 3, 4}; TUNE: {1000, 1001}; cùng seed cho mọi điều kiện (ghép cặp) | |

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

**Sửa đổi v1.1** (10/11; lý do là thời gian tính). **Đính chính thời điểm** (theo phản biện): sửa đổi này làm **sau một lần chạy thử kỹ thuật trên TUNE**, khi đã thấy F1 của 16 lần chạy, nhưng **trước khi chấm G7**. Câu "trước mọi kết quả tinh chỉnh" trong commit 8950b1ff là không chính xác. Không có rò rỉ từ G7.
- Đo trên dữ liệu thật: một điểm lưới × một điều kiện × một mức × một seed trên 48 clip TUNE mất khoảng 50 s trên một nhân. Lưới 256 × 2 seed × 7 điều kiện × 4 mức sẽ mất khoảng 33 giờ.
- **Lưới mới 64 điểm:** θ_T ∈ {0,5; 1; 2; 4}, θ_F ∈ {2; 4; 8; ∞}, N_max ∈ {30; ∞}, w_lost ∈ {0; 1}; cố định **w_drift = 1** (trôi, gồm cả lệch lớp, là tín hiệu cốt lõi) và **w_unc = 0**.
- **Một seed tinh chỉnh (1000).** Seed chấm G7 giữ nguyên {0..4}.
- Lưới như nhau cho mọi điều kiện.
- Ghi minh bạch: lúc đo tốc độ đã thấy F1 của 16 lần chạy thử trên TUNE. Đó là dữ liệu tinh chỉnh, không dùng để chọn lưới.

**Tinh chỉnh:**
- Mỗi điều kiện × mỗi mức R_ul tinh chỉnh **riêng** trên **48 clip TUNE**, kịch bản N.
- Chọn F1 gộp lớn nhất; hoà thì lấy UL thấp hơn. UL ≤ R_ul và DL ≤ cap tự đảm bảo nhờ bucket và link.
- **Độ nhạy:** tham số chọn cho A1 áp cho mọi điều kiện.

## 8. Cổng H1 (G7, kịch bản N, DL_cap 64)

**Định nghĩa ΔF1:**
- ΔF1 là hiệu object-time F1 **theo lớp**, đơn vị điểm phần trăm.
- Với mỗi nguồn, F1 lấy **trung bình trên 5 seed kênh** đã khoá; "gộp" là trên 7 nguồn × 5 seed.
- "Thắng" một mức R_ul: ΔF1(C1 − A1) gộp ≥ +2,0 **và** ΔF1 trung bình theo seed > 0 ở **≥ 5/7 nguồn**.
- Baseline gợi ý mạnh nhất là **max(B, B+)** tại từng mức, chọn sau tinh chỉnh trên TUNE (không chọn trên G7).

| Kết luận | Điều kiện |
|---|---|
| **PASS** | Thắng ở ≥ 2/3 mức R_ul, **và** C1 − max(B, B+) ≥ 0 gộp ở các mức đó, **và** không mức nào C1 − A1 < −1 |
| **FAIL (không có dư địa)** | D − A1 < +2 ở mọi mức |
| **FAIL (phản hồi có giá không dùng được)** | D − A1 ≥ +2 ở ít nhất một mức nhưng C1 − A1 < +1 ở mọi mức. Báo rõ và **dừng** semantic ACK |
| **INCONCLUSIVE** | Các trường hợp còn lại |

**Báo cáo bắt buộc:**
- theo từng nguồn: mọi chỉ số ở mục 2, cả theo lớp và class-agnostic;
- gộp 7 nguồn kèm CI bootstrap theo nguồn (B = 2000), **chỉ để mô tả**;
- **độ ổn định qua seed:** ΔF1 gộp của từng seed, và số seed (trên 5) cho cùng kết luận ở từng mức;
- chẩn đoán track ma (mục 3);
- **tỉ lệ FRAME / TOK tới BS kịp và nguyên vẹn** theo điều kiện × mức.
- **Mất gói kiểu tất cả hoặc không** (khai báo 10/11, trước khi chấm G7):
  - FRAME 289 kbit cần khoảng 31 gói. Với p_ul = 1% mỗi gói, khoảng 27% FRAME mất trọn, trong khi TOK (1 gói) mất khoảng 1%.
  - Đây là mô hình **không có FEC hay truyền lại**; kết quả không được khái quát cho mọi mạng không dây.
  - Độ nhạy chỉ báo: p_ul ∈ {0; 5%} với tham số đã chọn;
- phân tích phụ: A0, C0, mô hình pipelined, DL_cap 16, mức 0,5 × R_F và 3 clip AUX.

**Sau cổng:**
- H1 FAIL: báo nguyên nhân, dừng mở rộng phản hồi ngữ nghĩa.
- H1 PASS hoặc INCONCLUSIVE: báo theo từng nguồn và đề xuất phép thử tiếp (H2 với S1 / S2 đã khoá ở thiết kế v3 mục 7).

Không train PPO / SAC, không chạy R2, không tải VID test-dev, chưa làm H3 / x264.

## 9. Active Semantic Recovery

**Đã giải (10/11).** Codex kiểm toán chỉ đọc: ASR không có bản sao trạng thái bên nhận, vòng ACK, tracking video hay policy online. Bảng đối chiếu và tiền lệ nội bộ (Town3 `temporal_select`; UAV3D v1.1, nơi phản hồi có tính phí **không** còn lợi) nằm ở thiết kế v3 mục 10.1. Báo cáo H1 phải trích các tiền lệ này.

Nội dung cũ của mục (giữ để truy vết):

Claude **không đọc** nhánh ASR. Đề nghị người phụ trách (Codex) làm một **kiểm toán phạm vi chỉ đọc**: không sửa, không chạy lại. Câu hỏi và bảng cần điền ở mục 10 của thiết kế v3 (bản cập nhật).

**Nếu ASR đã có đồng bộ niềm tin bên nhận tương đương C1:**
- không xây lại và gọi đó là đóng góp mới;
- R1b-0a khi ấy chỉ còn giá trị như phép đo độc lập giá trị gia tăng của phản hồi (H1), và phải trích dẫn ASR.

## 10. Kiểm thử (`test_r1b.py`, 52/52 qua; chi tiết ở báo cáo kiểm chứng)

Bảng dưới là bộ 25 test ban đầu. Phần bổ sung sau phản biện thứ hai (wire format, chỉ số theo lớp, motmetrics, ghép cặp kênh, timing chính xác, ngân sách, sai thứ tự, B+, track ma) nằm ở [swarmbridge-r1b0a-validation.md](swarmbridge-r1b0a-validation.md).

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
