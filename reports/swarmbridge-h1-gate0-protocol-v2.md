# SwarmBridge H1: protocol Gate 0 v2 (bản nháp để duyệt)

Ngày 11/11/2026. Viết theo phản biện thứ tư về [swarmbridge-direction-reassessment.md](swarmbridge-direction-reassessment.md) (commit `498eb171`).

- **Trạng thái:** H1 "Cooperative Unique-Object Semantic Reporting" được duyệt ở mức **ứng viên nghiên cứu**.
- **Tài liệu này:** chỉ là protocol và đặc tả.
- **Chưa làm:** chưa tải dataset lớn, chưa train, chưa chạy cổng nào, chưa viết code.
- **Ngoài phạm vi:** không RL, không U2U, không đọc nhánh ASR / UAV3D.
- **R1b FAIL:** giữ nguyên.

**Thay đổi so với bản reassessment** (các mục cũ bị thay thế):

| Mục cũ | Thay bằng |
|---|---|
| Một oracle "ORA" duy nhất | Phân loại oracle, mục 2 |
| Chỉ số chính "recall đối tượng duy nhất" | F1 đối tượng duy nhất theo lớp, cùng precision, recall, khai báo sai, bit trùng; mục 3 |
| Điều kiện "mỗi biểu diễn ≥ 5%" | Bỏ; thay bằng kiểm tra cạnh tranh ngân sách và độ nhạy quyết định, mục 6 |
| Dùng 14 cặp MDMT-test làm GATE | Không dùng làm tập xác nhận nữa, mục 5 |
| "Người dùng chấp nhận rủi ro license" | Bỏ; thay bằng quyết định truy cập dữ liệu, mục 8 |

## 0. Câu hỏi khoa học (phát biểu lại)

**Câu hỏi:** khi token đối tượng đã rẻ và BS có thể tự khử trùng lặp sau khi nhận, việc tiêu thông tin điều phối để các UAV **tránh gửi trùng từ trước** có còn đáng không? Nếu có, ở chế độ nào: biểu diễn nào, ngân sách nào, độ chồng lấn nào?

Hai câu cần trả lời trước khi code hệ lớn:
1. **Có dư địa phối hợp thật không?** Đo bằng oracle chỉ biết danh tính (mục 2), so với các baseline độc lập và khử trùng ở BS đã tối ưu.
2. **Dư địa đó có khai thác được không,** bằng cơ chế không biết ground truth và sau khi tính chi phí điều phối (Gate 1)?

## 1. Ước lượng nháp trước khi chạy (số học, chưa phải kết quả)

Chỉ dùng thống kê đã đo trên nhãn MDMT-test công khai và wire format R1b; không phân tích dữ liệu mới.

| Đại lượng | Giá trị |
|---|---|
| Đối tượng GT mỗi khung mỗi drone | 51 (600.923 box / 5.883 khung / 2) |
| Bit mỗi token (box 12-bit × 4, lớp, conf) | 48–55 (`r1b_proto`: HINT_BOX 48, HINT_BOX_PLUS 55), chưa tính header |
| Gửi hết token, cả cặp, mỗi khung | ≈ 102 token ≈ 5,6 kbit |
| Phần trùng (cả hai drone thấy) | both / (a + b) = 0,456 / 1,456 ≈ **31%** ≈ 1,8 kbit mỗi khung |
| Ở 10 khung/s | trùng ≈ 18 kbit/s trên tổng ≈ 56 kbit/s |

**Hệ quả cần kiểm (dự đoán, ghi trước):**
- **Chỉ với SKIP / TOK**, phối hợp chỉ có giá trị khi uplink mỗi cặp nhỏ hơn cỡ 5,6 kbit mỗi epoch. Đó là một chế độ rất hẹp so với uplink LTE/5G vài Mbit/s (trace AERPAW).
- Muốn báo cho UAV biết đối tượng nào BS đã có, downlink phải mang **cỡ bằng** chính phần trùng: ID và box của đối tượng đã biết, mỗi đối tượng khoảng 48 bit.
- Vì vậy **khả năng cao TOK-only không có dư địa ròng**, trừ khi downlink rẻ hơn uplink nhiều lần.

Hai lưu ý:
- Các con số dùng **mọi đối tượng GT**. Detector thật thấy ít hơn (recall khoảng 30–70%), nên phần trùng thực tế còn nhỏ hơn.
- Nếu đúng như dự đoán, chỗ đáng xét là **biểu diễn nặng**: mảnh ảnh (crop) cho vật nhỏ mà detector UAV bỏ sót hoặc không chắc, và khung ảnh. Ở đó một bản trùng tốn hàng kbit. Theo R1a, ảnh thắng recall ở ngân sách cao. Luật quyết định khai báo trước ở mục 7.

## 2. Phân loại oracle

**Nguyên tắc:** GT cross-view ID chỉ được dùng ở **bộ chấm** và ở **oracle đã khai báo**. Mọi chính sách thực tế và mọi bước khử trùng thực tế ở BS **không** được dùng GT ID.

| Mã | Tên | Biết gì | Không biết gì | Dùng để |
|---|---|---|---|---|
| **R0** | Không oracle (thực tế) | Detection của mình, độ tin cậy đã hiệu chỉnh trên TUNE, thông điệp điều phối thực sự nhận được | GT ID, kết quả | Baseline và cơ chế thực tế |
| **O-ID-BS** | Danh tính lý tưởng **ở BS** | BS biết chính xác cặp detection nào cùng một vật thể (từ GT), **sau khi đã nhận** | Detection nào sẽ đúng | Baseline khử trùng ở BS mạnh nhất có thể |
| **O-ID-TX** | Danh tính lý tưởng **ở bên gửi** | Như O-ID-BS, nhưng biết **trước khi gửi**, nên tránh gửi trùng; điều phối miễn phí | Detection nào sẽ đúng | Cận lạc quan cho lợi ích phối hợp |
| **O-OUT** | Biết kết quả (hindsight) | Biết trước báo cáo nào thành TP sau xử lý ở BS | – | Chỉ báo, cận tuyệt đối; **không** dùng để quy lợi ích cho phối hợp |

**Định nghĩa lợi ích phối hợp:** **Δ_coord = U(O-ID-TX) − max[U(B0), U(O-ID-BS)]**.
- Hai vế cùng có danh tính lý tưởng. Khác biệt duy nhất là khử trùng **trước** hay **sau** khi truyền.
- Vì vậy rò rỉ thông tin từ GT ID (mục 2.1) tác động như nhau lên cả hai, và không bị tính thành lợi ích phối hợp.

**U(O-OUT) − U(O-ID-TX)** là phần lợi ích chỉ có được nhờ biết tương lai. Phần này báo riêng.

### 2.1. Rò rỉ của oracle danh tính và cách chặn

- Danh tính lý tưởng ngầm tiết lộ một phần TP: một detection ghép được với detection ở góc kia qua cùng GT ID thì gần như chắc chắn là TP.
- **Cách chặn:** oracle danh tính chỉ được dùng cho **quan hệ trùng** ("nếu gửi d thì d′ thừa"). Giá trị của từng báo cáo phải lấy từ độ tin cậy đã hiệu chỉnh trên TUNE: lớp, kích thước, conf, không dùng trạng thái "có cặp".
- **Chẩn đoán bắt buộc:** báo precision của detection có cặp so với không có cặp, để đo mức rò rỉ.
- Biến thể **O-ID-noisy:** danh tính có lỗi liên kết ở tỉ lệ đo được trên TUNE. Đây là cầu nối sang Gate 1.

### 2.2. Oracle phải tối ưu thật

- **SKIP / TOK, cùng giá token.** Bài toán mỗi epoch: chọn tập token trong ngân sách để tối đa F1 kỳ vọng sau fusion ở BS (mục 3.3). Nhóm trùng có kích thước ≤ 2, nên giải **chính xác** bằng ILP nhỏ (khoảng 100 biến mỗi epoch). Không dùng greedy rồi gọi là oracle.
- **Gói nặng (crop / FRAME).** Một FRAME phủ nhiều đối tượng cùng lúc, nên đây là bài toán chọn gói có giá trị phủ, cũng giải bằng ILP. Chỉ mở khi điều kiện ở mục 7 thỏa.
- **Kiểm tra:** in khoảng cách đối ngẫu (duality gap) của solver; bằng 0 với mọi epoch của tập TUNE.

## 3. Đặc tả chỉ số và bộ chấm

### 3.1. Đơn vị

- **Epoch:** một cặp khung đồng bộ, lấy mẫu đều. Tần số đặt theo fps thật của dataset, khoá trước khi chạy 0B.
- **Chế độ snapshot:** BS dựng lại bản đồ đối tượng của epoch từ báo cáo nhận được trong epoch đó. Đây là bài toán làm tươi bản đồ, vì vật di chuyển nên box cũ không còn đúng.
- **Độ nhạy:** chế độ có dự đoán thời gian, trong đó BS ngoại suy box epoch trước bằng vận tốc hằng; chỉ báo.
- **Không có mô hình thời gian truyền hay hàng đợi,** nên kết quả gọi là **"nhận thức offline theo ngân sách bit"**, không gọi là "đúng hạn".

### 3.2. Chỉ số chính: F1 đối tượng duy nhất theo lớp

**Khai báo ở BS.** Sau khi BS nhận báo cáo, khử trùng và fusion, BS xuất danh sách đối tượng. Mỗi đối tượng gồm lớp, độ tin cậy, và box ở ít nhất một góc nhìn.

**Ghép với GT:**
1. Mỗi khai báo được ghép với GT **theo từng góc nhìn**: Hungarian, cùng lớp, IoU ≥ 0,5.
2. GT ở hai góc mang cùng ID được gộp thành **một đối tượng duy nhất**.
3. Một khai báo có box ở cả hai góc được tính TP nếu khớp ở ít nhất một góc. Nếu hai góc khớp **hai** đối tượng GT khác nhau, đó là lỗi liên kết: tính 1 TP + 1 FP_assoc.

**Đếm:**

| Loại | Định nghĩa |
|---|---|
| **TP** | Đối tượng duy nhất có ≥ 1 khai báo khớp |
| **FP_wrong** | Khai báo không khớp GT nào |
| **FP_dup** | Khai báo **thứ hai trở đi** khớp cùng một đối tượng duy nhất (BS không gộp được) |
| **FN** | Đối tượng duy nhất (có mặt ở ≥ 1 góc, không nằm trong vùng ignore) không được khai báo đúng |

- **Chỉ số chính:** F1 = 2·TP / (2·TP + FP_wrong + FP_dup + FP_assoc + FN), micro trên các epoch của mỗi chuỗi, rồi gộp.
- **Báo kèm:** precision, recall, từng loại FP, **bit thực tế đạt được** (gồm header), **bit trùng đã giao** (bit của báo cáo mà đối tượng của nó đã được báo cáo khác trong cùng epoch giao rồi), bit downlink điều phối, và F1 class-agnostic.
- **Thống kê:** CI bootstrap theo chuỗi, chỉ để mô tả; độ ổn định qua seed nếu có ngẫu nhiên.

### 3.3. Fusion ở BS (cho tính bổ sung)

- **Toán tử fusion khai báo trước:** với hai báo cáo cùng lớp được BS liên kết, độ tin cậy gộp p = 1 − (1 − p₁)(1 − p₂). Đây là noisy-OR, như R1a; các p là xác suất đã hiệu chỉnh trên TUNE. BS khai báo nếu p ≥ τ_decl.
- τ_decl được tinh chỉnh trên TUNE cho từng điều kiện. Thêm độ nhạy "τ dùng chung".
- **Tính bổ sung được đo sau fusion:** tỉ lệ TP chỉ được khai báo vì có đủ hai báo cáo, tức mỗi báo cáo riêng đều dưới τ_decl.
- Theo phản biện: với chỉ số "ít nhất một báo cáo đúng", hàm mục tiêu là hàm phủ, có tính submodular. Hàm chỉ có thể **không** submodular qua toán tử fusion. Không khẳng định điều đó từ lỗi detection riêng lẻ.

## 4. Ma trận baseline

| Mã | Mô tả | Điều phối | Thông tin | Tinh chỉnh trên TUNE | Vai trò |
|---|---|---|---|---|---|
| **B0** | Mỗi UAV độc lập: gửi top-k theo độ tin cậy đã hiệu chỉnh, trong phần ngân sách của mình | Không | R0 | Cách chia ngân sách giữa UAV (bằng nhau / theo số detection / tỉ lệ tối ưu), ngưỡng conf, τ_decl | Baseline độc lập mạnh |
| **B1-real** | Gửi trong ngân sách như B0; BS **tự liên kết** (đăng ký hai tập box bằng RANSAC homography trên tâm box theo lớp, không dùng ảnh, không GT), rồi khử trùng và fusion | Không | R0 | Tham số liên kết, τ_decl | Đối thủ nguy hiểm nhất ở mức thực tế |
| **B1-ID** = O-ID-BS | Như B1 nhưng BS biết danh tính lý tưởng | Không | O-ID-BS | τ_decl | Khử trùng ở BS mạnh nhất |
| **B2** | Điều phối đơn giản do BS chủ trì: UAV1 gửi trước; BS ánh xạ đối tượng đã nhận sang khung UAV2 (homography ở B1-real) và phát danh sách "đã có"; UAV2 bỏ phần đã có. Downlink tính đủ bit. | Có (DL) | R0 | Thứ tự, kích thước danh sách | Phối hợp kiểu GOSC / RACooper, không cần mô hình ngữ nghĩa |
| **B3** | Phân vùng tĩnh kiểu EMP: chia vùng chồng lấn bằng một đường trong khung chung; mỗi UAV chỉ báo phía mình | Có (một lần) | R0 | Vị trí đường chia | Phối hợp hình học rẻ |
| **O-ID-TX** | Mục 2 | Miễn phí | Danh tính lý tưởng | ILP chính xác | Cận lạc quan |
| **O-OUT** | Mục 2 | Miễn phí | Kết quả | ILP chính xác | Chỉ báo |

- **Liên kết của B1-real và B2 là thành phần thực tế duy nhất có thể thành cơ chế mới:** liên kết xuyên góc chỉ từ token ngữ nghĩa, không pose, không ảnh. Trước tiên nó là baseline; nếu nó đủ tốt thì phối hợp phía bên gửi càng khó có lợi.
- **Không có B nào dùng RL.**

## 5. Dữ liệu, split và chống thiên lệch chọn

| Tập | Nguồn | Trạng thái | Dùng cho |
|---|---|---|---|
| **EXPL** | Nhãn MDMT-test công khai (14 cặp chuỗi), đã dùng để đo chồng lấn khi chọn hướng | **Đã thấy** (chọn dataset dựa trên metadata) | Chỉ Gate 0A: khám phá, chọn mức ngân sách. Không bao giờ dùng làm tập xác nhận. |
| **TUNE** | Chuỗi MDMT train (nhãn và ảnh có trong bản đầy đủ) | Chưa thấy | 0B: hiệu chỉnh độ tin cậy, tinh chỉnh baseline, kiểm tra hợp lệ |
| **HOLD** | Một phần chuỗi MDMT train/val, chọn bằng băm tên chuỗi **trước khi tải** (cố định seed, ghi vào protocol) | Chưa thấy | 0B: tập xác nhận, chấm một lần |
| **REPL** | VisionTrack (2 drone, người đi bộ, ID xuyên góc có sẵn) hoặc M3OT (2 drone, xe, CC BY 4.0, phải dựng ID và kiểm ≥ 300 cặp bằng tay) | Chưa thấy | Lặp lại độc lập; chỉ báo |

### 5.1. VisionTrack: quy tắc dùng, ghi trước khi mở bất kỳ nhãn nào (11/11)

- Dữ liệu đã tải về `data/visiontrack/` (không vào git). Kích thước khớp từng byte với OneDrive: `train.zip` 36.255.903.058 byte, `test.zip` 36.191.427.443 byte.
- **`train`** (VT-TUNE) được phép khám phá:
  - đọc định dạng và license kèm theo nếu có;
  - xác minh ID xuyên góc nhìn;
  - đo độ chồng lấn;
  - kiểm rò rỉ với VisDrone-DET;
  - tinh chỉnh ở 0B.
- **`test`** (VT-HOLD) là tập xác nhận chưa ai nhìn:
  - trước khi chấm, chỉ được kiểm CRC của zip và đếm số chuỗi, số khung, số file;
  - **không** tính bất kỳ thống kê nhãn nào (chồng lấn, số đối tượng, lớp);
  - chỉ chấm một lần, sau khi protocol v2.1 được khoá.
- Vai trò của VisionTrack (thay REPL bằng tập 0B chính, hay giữ làm REPL) được quyết định từ kết quả trên `train` và quyền sử dụng. Quyết định đó không dựa vào `test`.

### 5.2. VisionTrack `train`: kết quả khảo sát (11/11; chỉ `train`, `test` chưa mở)

| Đại lượng | Giá trị |
|---|---|
| Cấu trúc | 24 cảnh × 2 góc (View1/View2), 57.508 ảnh 1920×1080, 28.754 cặp khung đồng bộ (số khung hai góc bằng nhau ở mọi cảnh) |
| Nhãn | `gt/gt.txt` dạng MOT (frame, id, x, y, w, h, 1, 1, 1), cùng `gt/ViewN.txt` (6 cột, cùng nội dung). Chỉ một lớp: **người đi bộ**. Không có file license hay README trong zip |
| ID xuyên góc | **Có, đã kiểm bằng mắt:** vẽ box và ID ở cảnh 00145road khung 100; người 1, 2, 3 trùng khớp giữa hai góc nhìn rất khác nhau |
| Đối tượng mỗi khung mỗi góc | **10,3** trung bình. 12/24 cảnh chỉ có 2–4 người (dàn dựng); vài cảnh đông 26–50 người (football, basketball, canteen) |
| Tỉ lệ trùng (cả hai thấy / hợp) | **0,511** gộp; theo cảnh 0,19–1,00 |
| Cỡ box | Chiều cao p10 / p50 / p90 = 34 / 73 / 185 px. Drone bay thấp, chụp nghiêng; vật to hơn MDMT nhiều (diện tích trung vị 2.368 so với 1.078 px²) |

**Hệ quả cho H1:**
- **Dùng được** để kiểm tra liên kết xuyên góc (B1-real) với góc nhìn rất khác nhau, và để lặp lại kết quả. Đây là dữ liệu thật với ID đã xác minh.
- **Yếu nếu dùng làm bằng chứng chính:**
  - ít đối tượng, nên dòng token chỉ khoảng 0,55 kbit mỗi khung mỗi góc, và phần trùng tuyệt đối rất nhỏ;
  - vật to, nên lý do cần biểu diễn nặng (vật nhỏ bị detector UAV bỏ sót) gần như không có;
  - chỉ có một lớp.
- Một kết quả "không có dư địa" trên VisionTrack có thể đến từ bản chất dataset, không từ cơ chế.
- **Toàn vẹn:** CRC của mọi mục trong hai zip đều đúng (`testzip`). `test`: 22 cảnh × 2 góc, 58.038 ảnh, 87 file nhãn. Riêng `00012night_View2` thiếu `gt/gt.txt` nhưng có `View2.txt`, mà ở `train` hai file này có cùng nội dung. Bộ đọc nhãn sẽ dùng `ViewN.txt` làm nguồn chính. Ngoài các con số đếm này, chưa đọc gì từ `test`.
- **Quyết định:** VisionTrack giữ vai trò **REPL / kiểm tra liên kết**, không thay MDMT làm tập 0B chính. MDMT (51 đối tượng mỗi khung, vật nhỏ, có xe) vẫn là ứng viên chính, chờ trả lời về quyền và đường tải.

- **Rò rỉ với dữ liệu train detector:** kiểm MDMT và VisDrone-DET bằng pHash như audit VID (cùng nhóm phát hành).
- Chuỗi nào trùng nguồn thì đưa ra khỏi HOLD.
- **Không dùng kết quả HOLD để chọn lại** policy, chỉ số hay mức ngân sách.

## 6. Kiểm tra hợp lệ trên TUNE (thay cho điều kiện "≥ 5%")

Không qua thì kết luận **INVALID** (thiết kế lại), không phải FAIL:
1. **Ngân sách tạo cạnh tranh:** ở mỗi mức B, tổng bit gửi hết của B0 ≥ 1,5 × B (ngân sách thật sự chặn).
2. **Quyết định thay đổi được tiện ích:** ở mỗi mức B, tập tối ưu của O-ID-TX hơn một tập khả thi chọn ngẫu nhiên ít nhất 5 điểm F1. Nếu không, mọi lựa chọn đều như nhau.
3. **Baseline và oracle được tối ưu hợp lệ:** ILP có duality gap bằng 0; B0, B1, B2, B3 được tinh chỉnh với cùng ngân sách tìm kiếm; báo lưới đã dùng.
4. **Trần hợp lệ:** U(O-OUT) ≥ U(O-ID-TX) ≥ U(O-ID-BS) ≥ U(B1-real) trên TUNE. Nếu O-ID-TX < B0, oracle đã được định nghĩa sai.
5. **Chỉ số đủ nhạy:** U(O-OUT) − U(B0) ≥ 5 điểm F1 ở ít nhất một mức. Nếu không, chỉ số không thể phân biệt được gì.

Biểu diễn không bắt buộc phải xuất hiện. Nếu TOK luôn thắng FRAME ở một chế độ, không chọn FRAME là hợp lý.

## 7. Ba tầng cổng

### Gate 0A: cơ hội (EXPL, chỉ nhãn, không ảnh, không detector)

- **Câu hỏi:** với **token hoàn hảo** (box GT làm detection, p = 1), ở các ngân sách khác nhau, khử trùng lý tưởng giải phóng được bao nhiêu bit, và nâng độ phủ đối tượng duy nhất bao nhiêu? Đây là cận trên của cơ hội cho TOK-only.
- **Đầu ra:**
  - đường cong phủ theo ngân sách cho gửi độc lập và khử trùng lý tưởng (trước / sau truyền);
  - bit downlink tối thiểu cần để báo "đã có";
  - **các mức B** cho 0B (khoá vào protocol v2.1 trước khi tải dữ liệu 0B).
- **Luật quyết định khai báo trước:**
  - Nếu ở mọi mức B mà uplink thực tế có thể có (≥ 64 kbit/s mỗi UAV), cơ hội ròng của TOK-only (bit uplink tiết kiệm trừ bit downlink điều phối) < 10% ngân sách: **TOK-only kết luận "không có dư địa" ngay**, không tốn 0B cho TOK-only.
  - Khi đó 0B chuyển sang menu **SKIP / TOK / CROP / FRAME** dạng gói. Đây là chỗ một bản trùng tốn hàng kbit; R1a đã có số bit thật cho các biểu diễn này.
  - **Câu hỏi để duyệt:** quy tắc này lệch với gợi ý "mở HYB/FRAME chỉ sau khi chứng minh có dư địa". Tôi đề xuất nó vì kết quả TOK-only âm do token quá rẻ không nói gì về biểu diễn nặng. Nếu không duyệt, TOK-only âm thì H1 dừng.

### Gate 0B: nhận thức thực tế (TUNE → HOLD, detector thật)

- **Detector:** YOLO11n trên UAV; YOLO11s ở BS cho crop/FRAME. Cả hai có sẵn, train trên VisDrone. Không train lại. Báo recall từng góc để biết độ lệch miền.
- **Điều kiện:** B0, B1-real, B1-ID, B2, B3, O-ID-TX, O-OUT; cùng ngân sách; F1 đối tượng duy nhất theo lớp.
- **Luật** (ngưỡng **mong muốn**, khoá ở v2.1 sau 0A, trước khi chạy TUNE của 0B):

  | Kết luận | Điều kiện |
  |---|---|
  | **Có dư địa** | Δ_coord = U(O-ID-TX) − max[U(B0), U(O-ID-BS)] ≥ +5 F1 ở ≥ 2/3 mức B trên HOLD, CI bootstrap theo chuỗi có cận dưới > 0 |
  | **Không có dư địa** | Δ_coord < +2 ở mọi mức → dừng H1 |
  | **INCONCLUSIVE** | Còn lại |

- **Phân tích phụ:**
  - Δ_coord theo độ chồng lấn từng chuỗi;
  - tỉ lệ TP cần fusion hai báo cáo;
  - tương quan lỗi detector giữa hai góc vs giữa hai mức chất lượng cùng góc (H2);
  - rò rỉ của oracle danh tính (mục 2.1);
  - REPL.

### Gate 1: phối hợp triển khai được (chỉ khi 0B có dư địa)

- **Cơ chế thực tế:** liên kết dự đoán, không GT. Ứng viên gồm B2 cải tiến, giá trị biên theo giá kiểu GOSC, và phối hợp chỉ giữa các cặp chồng nhau.
- Downlink điều phối, lỗi liên kết và giá trị biên được tính đủ.
- **Đạt nếu:** U(cơ chế) − max[U(B0), U(B1-real)] ≥ 50% × Δ_coord, sau khi tính bit downlink vào cùng ngân sách (hoặc theo tỉ giá DL/UL khai báo trước).
- Chạy trên dữ liệu chưa dùng: phần còn lại của HOLD hoặc REPL.

## 8. Truy cập dữ liệu và quyền sử dụng (quyết định)

| Dataset | Đã kiểm (11/11) | Quyết định |
|---|---|---|
| **VisionTrack** (github.com/FoxCanned/GMT) | Repo mới cập nhật 10/11/2026. Code license **Apache-2.0** (chỉ áp dụng cho code). README có link OneDrive và Baidu cho dataset, nhưng còn câu "will be released soon" (dòng 50) và câu "code hiện không dùng được" (dòng 1). **Không thấy license dữ liệu.** 3 issue, không issue nào về điều khoản dữ liệu.<br>**Cập nhật 11/11** (liệt kê bằng token ẩn danh của chính trang OneDrive; chỉ metadata, không tải):<br>• thư mục `VisionTrack` 72,45 GB, chỉ gồm `train.zip` 36,3 GB và `test.zip` 36,2 GB, **không có file LICENSE/terms**;<br>• link nhãn là một file `gmt_json.rar`, 53,1 MB, sửa lần cuối 18/10/2026 (nhãn JSON của các dataset trong bài). | **Kiểm trước tiên.**<br>• Hỏi tác giả về điều khoản dữ liệu (mẫu ở phụ lục).<br>• Nếu được duyệt: tải riêng `gmt_json.rar` (53 MB, chỉ nhãn) để xác minh ID xuyên góc và đo độ chồng lấn như MDMT, trước khi quyết định tải 72 GB ảnh. |
| **MDMT** (github.com/VisDrone/Multi-Drone-...) | README chỉ có link Baidu (25,8 GB) và "Please cite"; **không có license**. 31 issue, nhiều issue về khó tải (#8, #20, #22); không issue nào về điều khoản. Nhãn test có trên GitHub. | **Không coi "không thấy license" là được phép.** Gửi thư hỏi tác giả liên hệ (theo bài TMM 2023; địa chỉ lấy từ PDF) về ba quyền: (1) dùng cho nghiên cứu, (2) công bố kết quả và hình, (3) phát hành dữ liệu dẫn xuất (split, nhãn liên kết, token). Trong lúc chờ: chỉ dùng nhãn test công khai cho 0A (phân tích, không phân phối lại). Chưa tải 25,8 GB. |
| **M3OT** (Figshare, Sci. Data 2025) | **CC BY 4.0** theo agent kiểm dataset và theo phản biện; tôi chưa tự mở trang Figshare để kiểm lại điều khoản của file dữ liệu. Không có ID xuyên drone đã xác minh. | Dự phòng có điều khoản rõ nhất. Phải dựng ID (homography mặt đất + Hungarian + bỏ phiếu theo track) và kiểm ≥ 300 cặp bằng tay. Vật rất nhỏ (< 1024 px² ở 640×512), nên rủi ro lỗi cao. |
| Air-Co-Pred (CARLA, 4 drone) | HuggingFace, không gated. | Chỉ cho nghiên cứu mở rộng theo N sau Gate 1; sát họ UAV3D. |

**Thứ tự đề xuất:**
1. VisionTrack: kiểm nội dung bằng trình duyệt và hỏi điều khoản.
2. Gửi thư MDMT.
3. Chỉ tải dữ liệu nào trả lời được câu hỏi quyền sử dụng.

Gate 0A chạy được ngay trên nhãn MDMT-test công khai, không cần tải ảnh.

## 9. Checklist trước khi viết code

- [ ] Phản biện duyệt protocol v2, gồm quy tắc ở mục 7 về mở biểu diễn nặng khi TOK-only âm.
- [ ] Ghi danh sách chuỗi HOLD bằng băm tên chuỗi, seed cố định, **trước** khi tải dữ liệu 0B.
- [ ] Quyết định dữ liệu 0B theo mục 8: VisionTrack hoặc MDMT có trả lời về quyền, hoặc M3OT.
- [ ] Kiểm pHash giữa dữ liệu 0B và VisDrone-DET train.
- [ ] Đặc tả bộ chấm (mục 3) thành hàm có test trên dữ liệu tổng hợp: TP, các loại FP, FN, bit trùng, trường hợp liên kết sai, fusion. Có test chứng minh **bộ chấm là nơi duy nhất đọc GT ID**, ngoài các oracle được đánh dấu.
- [ ] ILP oracle có test: duality gap bằng 0; kết quả so với vét cạn trên ví dụ nhỏ.
- [ ] Kiểm tra rò rỉ: chính sách R0 chạy được khi GT ID bị xoá khỏi đầu vào.
- [ ] Gate 0A chạy trên EXPL → khoá mức B và ngưỡng vào v2.1 → commit → mới tải hoặc chạy TUNE của 0B.
- [ ] Bộ chấm 0B viết và test xong trước khi chấm HOLD; từ chối chấm lại như `r1b_run.py gate`.

## Phụ lục: mẫu thư hỏi quyền sử dụng dữ liệu (tiếng Anh)

Người gửi điền tên và đơn vị. Gửi tới tác giả liên hệ ghi trong bài báo tương ứng.

> Subject: Request for terms of use — [MDMT / VisionTrack] dataset
>
> Dear Prof. [Name],
>
> We are researchers at [lab, university] working on communication-efficient multi-UAV perception. We would like to use the [MDMT / VisionTrack] dataset, which we found at [repository URL]. We could not find a license or terms of use for the data itself, so we would like to ask whether you permit the following:
>
> 1. use of the images and annotations for non-commercial academic research;
> 2. publication of results, tables and a small number of example figures derived from the dataset, with citation of your paper;
> 3. release of derived artefacts that contain no original images (for example train/test split lists, cross-view association labels, and detection outputs) to support reproducibility.
>
> If there is a data-use agreement we should sign, we would be glad to do so. [For MDMT: we would also appreciate an alternative download mirror if one exists, as the Baidu link is difficult to access from outside China.]
>
> Thank you very much for releasing this benchmark.
>
> Best regards,
> [Name, affiliation]
