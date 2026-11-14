# RQ1-E: liên kết xuyên góc bằng ngoại hình (đăng ký trước)

Ngày 14/11/2026, theo phản biện #24. Bản này **tách riêng** khỏi protocol RQ1 v2 (chỉ hình học, Phase 0 NOT IDENTIFIABLE); kết quả đó giữ nguyên. Thiết kế gốc: [swarmbridge-rq1-reid-feasibility-memo.md](swarmbridge-rq1-reid-feasibility-memo.md).

**Commit trước khi tính bất kỳ embedding nào.** E0 được khoá đầy đủ ở đây. E1 sẽ được đăng ký trong một commit riêng **sau** khi có kết quả E0 và **trước** khi chạy E1.

**Cơ sở dùng dữ liệu:** theo lời người dùng, giáo sư hướng dẫn cho phép lab dùng MDMT cho nghiên cứu nội bộ (xác nhận miệng, do người dùng thuật lại; chưa có văn bản; tác giả chưa trả lời; dữ liệu từ mirror không chính thức). Ảnh, crop, embedding gắn với vật thể và mọi dữ liệu theo khung **không** đưa lên GitHub; chỉ commit code, protocol, kết quả tổng hợp.

## 1. Dữ liệu và chia thời gian

- 8 chuỗi MDMT test đã qua sàng lọc ID: 26, 31, 34, 52, 56, 57, 61, 62. Không mở val, không HOLD.
- Epoch: đúng các khung đã dùng trong pilot 0B (cùng bản ghi). **Nửa đầu** mỗi chuỗi cho E0; **nửa sau** cho E1.
- Detection: detection YOLO11n đã ghi trong bản ghi pilot. Không chạy lại detector.
- Khai báo ở BS: TOKEN như Phase 0 (token mạnh, lượng tử hoá, NMS theo lớp, ngưỡng τ_decl = 0,3).

## 2. Trích đặc trưng ngoại hình (cố định)

- **Mô hình:** `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, `model.safetensors` sha256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`, giấy phép Apache-2.0. Không train, không fine-tune.
- **Crop (trên UAV, từ ảnh gốc 1920 × 1080):** với mỗi detection đã khai báo, lấy box gốc chưa lượng tử của detection YOLO11n tương ứng (khớp IoU lớn nhất). Cửa sổ vuông cùng tâm, cạnh = max(1,5 × cạnh dài của box, 32 px), dời vào trong ảnh nếu tràn; nếu ảnh nhỏ hơn cạnh thì cắt bớt.
- **Tiền xử lý:** RGB, đổi cỡ thẳng về 224 × 224 (bicubic), chia 255, chuẩn hoá theo trung bình / độ lệch ImageNet (0,485, 0,456, 0,406) / (0,229, 0,224, 0,225).
- **Embedding:** `pooler_output` của mô hình (token CLS sau layernorm), 384 chiều, chuẩn hoá L2, float32, batch 256.
- **Bản nén** (để đo, không dùng cho quyết định E0): dấu của phép chiếu ngẫu nhiên Gauss cố định R ∈ ℝ^{m × 384}, seed 0, m ∈ {64, 128, 256}; khoảng cách Hamming chuẩn hoá. Thêm bản lượng tử 8 bit mỗi chiều (384 × 8 = 3.072 bit) làm mốc.
- **Đo tốc độ trên máy thật** (RTX 3070, CPU 6 lõi): thời gian giải mã ảnh, cắt và tiền xử lý, suy luận mô hình; tổng thời gian. Không giả định trước.

## 3. Bộ liên kết E0 (cố định; GT không được dùng ở đây)

Mỗi epoch, từng lớp riêng:
1. **Ứng viên từ ngoại hình:** khoảng cách d = 1 − cosine giữa embedding hai góc. Một cặp (i, j) là ứng viên nếu i và j là láng giềng gần nhất của nhau, **và** qua phép thử tỉ số d₁ / d₂ ≤ 0,8 theo cả hai chiều (d₂ là khoảng cách tới ứng viên cùng lớp gần thứ hai; nếu không có ứng viên thứ hai thì phép thử coi như qua).
2. **Đăng ký:** homography góc 1 → góc 2 bằng `cv2.findHomography` RANSAC (40 px) trên tâm đáy box của các ứng viên (gộp mọi lớp); cần ≥ 8 ứng viên.
3. **Hợp lệ** (không nhãn) nếu: homography hữu hạn; |det| phần 2 × 2 trong [0,1; 10]; ≥ 8 inlier RANSAC; **và** tỉ lệ inlier trong các ứng viên ≥ 0,5 (chặn đăng ký sai). Không hợp lệ thì epoch đó không có homography.
4. **Ghép cuối** (để báo, không quyết định E0): cùng lớp, Hungarian trên sai số chiếu, ngưỡng 40 px.

E0 dùng embedding của **mọi** detection, không tính chi phí truyền: đây là chẩn đoán **FREE-APPEARANCE**, một cận trên về tính khả thi, **không phải** chính sách truyền triển khai được.

## 4. Chỉ số E0 (GT chỉ để chẩn đoán)

Theo từng chuỗi, trên các epoch nửa đầu:
- **Độ chính xác ứng viên:** trong các cặp ứng viên mà cả hai đầu khớp được một ID GT (IoU ≥ 0,5), tỉ lệ cặp có cùng ID. Báo thêm tỉ lệ cặp không kiểm được.
- **Độ phủ ứng viên:** số cặp ứng viên đúng / số cặp lý tưởng (cặp detection đã khai báo có cùng ID GT ở hai góc).
- **Tỉ lệ epoch đăng ký hợp lệ.**
- **Sai số chẩn đoán:** chiếu tâm đáy box GT góc 1 (ID có ở hai góc) bằng homography, trung vị khoảng cách tới box GT góc 2; lấy trung vị trên các epoch hợp lệ.
- **Tỉ lệ đăng ký sai:** trong các epoch hợp lệ, tỉ lệ có sai số chẩn đoán > 100 px.
- Độ chính xác và độ phủ của ghép cuối so với cặp lý tưởng.

## 5. Quy tắc quyết định E0 (khoá)

Một chuỗi **đạt** nếu đồng thời:
- (i) đăng ký hợp lệ ở ≥ 50% epoch nửa đầu;
- (ii) sai số chẩn đoán trung vị ≤ 40 px;
- (iii) tỉ lệ đăng ký sai ≤ 10%;
- (iv) độ chính xác ứng viên ≥ 0,8;
- (v) độ phủ ứng viên ≥ 0,25 (để vài cặp đúng không tạo ra một PASS giả).

**E0 PASS** nếu ≥ 6/8 chuỗi đạt với embedding đầy đủ 384 chiều. Nếu không: **E0 FAIL → NO-GO cho pipeline ngoại hình zero-shot này trên MDMT**. Khi đó không thử backbone khác, không tìm ngưỡng khác, không học mô hình.

**Độ rộng embedding chính cho E1** (quy tắc khoá trước, áp lên nửa đầu): độ rộng nhỏ nhất trong {64, 128, 256, 3.072} bit mà bản nén cũng đạt (i)–(v) ở ≥ 6/8 chuỗi. Nếu không độ rộng nào đạt, E1 không chạy và báo "ngoại hình đã nén không đủ". Các độ rộng khác chỉ là phân tích độ nhạy.

## 6. E1: các cam kết (protocol chi tiết sẽ đăng ký riêng trước khi chạy)

- **Không bootstrap miễn phí:** mỗi chính sách tự thu thông tin xuyên góc ban đầu bằng các gói được tính phí thật. Không chính sách nào thừa hưởng homography của E0 (ước lượng từ đặc trưng chưa truyền) hay từ GT.
- `AMB-EMB` phải xác định detection mơ hồ **trước khi** nhận embedding được yêu cầu; nếu chưa có đăng ký thì phải có cách dò nhân quả, định nghĩa trước.
- **Tính đủ chi phí:** bootstrap, TOKEN, embedding, header gói, yêu cầu đường xuống. Báo độ trễ và tính khả thi của vòng hỏi–đáp; không gọi là khả thi đầu-cuối nếu vi phạm hạn chót.
- Cùng giới hạn đường lên cho mọi chính sách; bit đạt được báo riêng. Baseline không được biết trước chi phí thực của `AMB-EMB`.
- So sánh `TT`, `ALL-EMB`, `CONF-EMB`, `RAND-EMB`, `AMB-EMB` và oracle `ideal` (đánh dấu); `AMB-CROP` là phụ.
- Giữ điều kiện NO-GO: nếu `ALL-EMB` đã đủ tốt và rẻ thì đóng hướng truyền chọn lọc.
- Chạy đúng một lần; không săn heuristic sau khi xem kết quả.

## 7. Lưu trữ

- Embedding và crop: `~/work/h1_pilot0b/rq1e/` (ngoài git).
- Commit: code `swarmbridge/experiments/uav_semcom/h1_rq1e.py`, test tổng hợp, protocol này, kết quả tổng hợp theo chuỗi.
