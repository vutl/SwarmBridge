# RQ1 bằng ngoại hình: memo khả thi kỹ thuật

Ngày 14/11/2026, theo yêu cầu sửa hướng sau RQ1 Phase 0. Memo này **không** chạy suy luận mới trên ảnh MDMT, không train, không tải trọng số. Các kết quả âm trước đó giữ nguyên.

**Đính chính cách hiểu Phase 0:** Phase 0 cho kết quả NOT IDENTIFIABLE trên **bản ghi cache** của pilot 0B (chỉ có box, lớp, độ tin cậy) với bộ liên kết chỉ dùng hình học. Kết quả đó **không** có nghĩa bản MDMT trên máy thiếu thông tin để liên kết: ảnh RGB thật của hai UAV vẫn còn, và ngoại hình của vật thể chưa được khai thác.

## 1. Dữ liệu đã có trên máy

Kiểm lại ngày 14/11 trên bản mirror local, chỉ đọc danh sách tệp và nhãn, không đọc pixel; dùng lại kiểm toán ngày 12/11.

| Phần | Trạng thái |
|---|---|
| Tệp | 1 zip 16,7 GB (`data/mdmt/`, ngoài git); 21.894 mục |
| Test | 12/14 chuỗi có đủ hai góc (71, 73 thiếu UAV 2); 11 cặp qua quy tắc căn chỉnh; **8 cặp qua sàng lọc ID** (26, 31, 34, 52, 56, 57, 61, 62) |
| Val | 5 chuỗi đủ cặp (22, 36, 46, 49, 72; 1.930 cặp khung); **chưa mở nhãn, giữ nguyên** |
| Train | chỉ có UAV 1, không dùng được cho cặp |
| Ảnh | RGB 1920 × 1080, đồng bộ theo thứ tự tệp; 270–700 khung mỗi chuỗi test |
| Nhãn | CVAT XML, box + lớp + ID theo từng khung, ID chung giữa hai góc (đã sàng lọc) |
| Lớp (11 chuỗi test) | xe 67%, người 27%, xe đạp 5%, xe buýt 0,6% |
| Kích thước box (cạnh dài) | trung vị 20–107 px tuỳ chuỗi; phân vị 10% là 14–35 px |
| ID chung mỗi khung | 13–88 |
| Calibration / pose camera | **không có** |
| Tệp khác trong mirror | 6 checkpoint detector `.pth` (mmdetection) và 2 config `.py` do người đăng mirror để lại: **không nạp, không chạy** (pickle, nguồn không tin cậy) |

## 2. Mô hình pretrained có thể dùng lại

Nguồn đã xác minh trên trang chính thức; không tải gì.

| Mô hình | Dùng zero-shot được? | Giấy phép | Kích thước embedding | Nhận định |
|---|---|---|---|---|
| **DINOv2 ViT-S/14** | có, mọi lớp | code và trọng số Apache-2.0 (trừ biến thể XRay / Cell) | 384 | **Mặc định.** Giấy phép sạch nhất |
| OpenCLIP ViT-B/32 (LAION) | có, mọi lớp | MIT | 512 | so sánh phụ |
| FastReID `veri_sbs_R50-ibn` (xe) | có cho xe; lệch miền lớn (camera mặt đất ↔ drone 50–150 m) | code Apache-2.0; trọng số không ghi giấy phép; dữ liệu train (VeRi-776, VERI-Wild) chỉ cho phi thương mại | 2048 | so sánh riêng cho xe |
| OSNet x1.0 (MSMT17) | chỉ người; chưa rõ có hợp với người nhìn từ trên xuống, 15–60 px | code và trọng số MIT; MSMT17 cần thỏa thuận (chưa đọc được) | 512 | tuỳ chọn cho người |
| MIA-Net (bài MDMT) | **không** | repo không có LICENSE | 2048 (FastReID train trên crop MDMT) | xem dưới |

**MIA-Net không dùng được làm baseline không GT:**
- Đọc `demo/supplement_MIA.py`: ở khung đầu, nó đọc box và ID **của GT**, ghép ID giữa hai góc, rồi khớp homography bằng RANSAC.
- Trọng số Re-ID train trên MDMT chưa phát hành; issue hỏi về chúng chưa được trả lời.
- Không có giấy phép cho code hay dataset.
- Nhận xét: chính baseline của MDMT cũng khởi tạo liên kết bằng GT. Điều này khớp với Phase 0: phần khó là có được cặp tương ứng ban đầu mà không cần GT.

**Rủi ro chung:** mọi mô hình trên nhận ảnh 224 px (hoặc 256 × 128). Box 15–60 px phải phóng to 4–15 lần; với DINOv2, một crop 32 px thật chỉ phủ khoảng 2 × 2 patch. Đây là rủi ro cần đo, không phải điều đã được ghi nhận.

**Tải trọng số an toàn:** dùng định dạng safetensors (ví dụ `facebook/dinov2-small` trên Hugging Face). Không dùng `torch.hub` (chạy `hubconf.py`) và không nạp `.pth` không rõ nguồn.

## 3. Thiết kế đánh giá liên kết tối thiểu

**Mỗi epoch** (cùng cách lấy mẫu như pilot 0B), mỗi UAV có detection YOLO11n đã ghi (box, lớp, độ tin cậy).

**Embedding** tính trên UAV, từ crop quanh mỗi detection mạnh, bằng DINOv2-S, không train. Nó chỉ đến BS nếu được truyền (mục 4).

**Bộ liên kết ở BS** (không GT):
1. **Ứng viên từ ngoại hình:** với các detection có embedding đã nhận ở cả hai góc, lấy láng giềng gần nhất hai chiều theo cosine, cùng lớp, có lọc tỉ số kiểu Lowe 0,8 (ngưỡng chuẩn, không tinh chỉnh).
2. **Kiểm chứng hình học:** khớp homography bằng RANSAC trên các ứng viên đó. Đây là chỗ Phase 0 thiếu: giờ RANSAC có cặp tương ứng thật thay vì mọi cặp cùng lớp.
3. **Liên tục theo thời gian:** homography và các cặp đã ghép của epoch trước làm khởi tạo; một tracker IoU đơn giản trên mỗi góc giữ nhãn vết cục bộ, nên một cặp đã ghép được giữ cho tới khi bị mâu thuẫn.
4. **Ghép cuối:** Hungarian trên chi phí kết hợp sai số chiếu và khoảng cách ngoại hình (nếu có); không ghép nếu vượt ngưỡng.

**GT chỉ dùng để chấm điểm:** không dùng để tạo embedding, chọn cặp tương ứng, chọn ngưỡng, hay khởi tạo homography.

**Chỉ số:** P / R / F1 vật thể duy nhất sau hợp nhất; FP trùng, FP liên kết (ghép nhầm), vật bị sót; độ chính xác và độ phủ của liên kết trên các detection khớp GT; tỉ lệ epoch đăng ký thất bại; kết quả từng chuỗi.

## 4. Mô hình truyền thông

| Gói | Bit | Ghi chú |
|---|---|---|
| TOKEN | 55 bit / detection (+ header theo MTU, như pilot) | đã có |
| Embedding lượng tử | chiếu ngẫu nhiên cố định (không học) xuống m chiều rồi lấy dấu: **m ∈ {64, 128, 256} bit / detection** | không cần fit, nên không rò rỉ dữ liệu; tuỳ chọn PCA chỉ fit trên nửa đầu, không nhãn |
| CROP | bit AVIF **đo thật** (q50) của crop quanh detection | cần mã hoá crop mới (thuộc thí nghiệm, không thuộc memo) |
| Yêu cầu từ BS | 24 bit / yêu cầu trên đường xuống, báo riêng; một vòng hỏi–đáp trong epoch | như pilot 0B |

BS **không** được dùng embedding hay crop chưa được truyền.

## 5. Các chiến lược so sánh (cùng ngân sách bit mỗi epoch)

| Chiến lược | Gửi gì |
|---|---|
| `TT` | chỉ TOKEN; liên kết chỉ bằng hình học (= Phase 0) |
| `ALL-EMB` | TOKEN + embedding cho mọi detection |
| `CONF-EMB` | embedding cho k detection có độ tin cậy cao nhất (hoặc thấp nhất) |
| `RAND-EMB` | embedding cho k detection ngẫu nhiên (seed cố định) |
| **`AMB-EMB`** | BS xác định detection mơ hồ (không GT, từ hình học + trạng thái theo thời gian) rồi yêu cầu embedding chỉ cho chúng |
| `AMB-CROP` | như trên nhưng yêu cầu crop thật; BS tự tính embedding và detection trên crop |
| `ideal` (đánh dấu) | liên kết theo ID GT: chỉ là mốc chẩn đoán |

k được chọn để số bit của `CONF-EMB` / `RAND-EMB` bằng số bit mà `AMB-EMB` thực tế dùng, theo từng epoch.

## 6. Tính toán

- 8 chuỗi, lấy mẫu như pilot: khoảng 308 epoch × 2 góc = khoảng 620 khung; khoảng 10–85 detection mạnh mỗi góc mỗi epoch → khoảng 25.000 crop.
- DINOv2-S: dưới 1 phút trên GPU (chạy qua `gpulock_run.py`), khoảng 10 phút trên CPU.
- AVIF cho crop: khoảng 3 ms mỗi crop trên CPU → 1–2 phút mỗi mức chất lượng.
- Phần liên kết và chấm điểm: vài phút trên CPU.
- **Tổng: dưới 1 giờ GPU**, không train.

## 7. Còn thiếu gì cho đánh giá đủ chuẩn công bố

1. **Giấy phép dữ liệu:** quyền dùng MDMT vẫn chỉ là xác nhận miệng của giáo sư; mirror không chính thức; tác giả chưa trả lời; repo MDMT không có LICENSE. Đây là điểm chặn cho công bố. Ảnh, crop, embedding và mọi dữ liệu theo khung không được đưa lên GitHub công khai.
2. **Kiểm chứng độc lập:** 8 chuỗi EXPL đã dùng nhiều lần nên chỉ cho kết quả khám phá. 5 chuỗi val phải giữ cho đúng một lần chạy xác nhận theo protocol đã khoá.
3. **Một bộ dữ liệu thứ hai** (CrossTrack-UAV) để kiểm tính tổng quát.
4. **Không có metadata camera.**
5. **Chi phí tính embedding trên phần cứng UAV** chưa đo; hiện chỉ có proxy CPU desktop.
6. **Lệch miền của Re-ID** với vật nhỏ nhìn từ trên xuống.
7. **Prior art:** MIA-Net đã làm liên kết xuyên drone bằng Re-ID trên MDMT. Đóng góp tiềm năng chỉ có thể nằm ở **quyết định truyền**: khi nào cần thông tin liên kết, loại nào, có đáng số bit không.

## 8. Đề xuất một thí nghiệm đăng ký trước (RQ1-E) và tiêu chí dừng

**E0: tính xác định được (chỉ nửa đầu của 8 chuỗi).** Embedding miễn phí ở đây, vì chỉ hỏi liệu ngoại hình có cho cặp tương ứng đúng không.
- Đạt nếu: đăng ký hợp lệ với sai số chẩn đoán trung vị ≤ 40 px ở ≥ 6/8 chuỗi, **và** độ chính xác của cặp ứng viên từ ngoại hình ≥ 0,8 (đo bằng GT, chỉ để chẩn đoán).
- **Dừng** nếu không đạt: "ngoại hình zero-shot không đủ trên MDMT", không chạy E1.

**E1: quyết định truyền (nửa sau của 8 chuỗi).**
- Kết quả chính: F1 vật thể duy nhất của `AMB-EMB` so với chiến lược cố định tốt nhất trong {`ALL-EMB` cắt theo ngân sách, `CONF-EMB`, `RAND-EMB`} ở cùng số bit, từng chuỗi.
- **GO (khám phá)** nếu đồng thời:
  - `AMB-EMB` hơn chiến lược cố định tốt nhất ≥ 2 điểm F1 (trung bình) và dương ở ≥ 6/8 chuỗi;
  - `AMB-EMB` hồi phục ≥ 50% khoảng cách F1(`ideal`) − F1(`TT`) với ≤ 10% số bit của một FRAME.
- **NO-GO** nếu:
  - `AMB-EMB` − cố định tốt nhất < 1 điểm (truyền chọn lọc không đáng); **hoặc**
  - F1(`ideal`) − F1(`ALL-EMB`) < 1 điểm và `ALL-EMB` dùng ≤ 10% bit của một FRAME (gửi embedding cho mọi thứ đã đủ rẻ, không cần cơ chế chọn lọc).
- **INCONCLUSIVE** trong các trường hợp còn lại.
- `AMB-CROP` báo kèm; mọi lợi ích của nó phải tách được giữa "detection tốt hơn" và "ngoại hình".

**Sau E1:**
- NO-GO → dừng RQ1 trên MDMT.
- GO → đúng một lần chạy xác nhận trên 5 chuỗi val sau khi khoá lại protocol, và chỉ tính tới công bố khi có cơ sở dữ liệu bằng văn bản.

**Không làm:** train hoặc fine-tune; mở HOLD; PPO / MAPPO; Gate 1; tìm proxy mới; đổi ngưỡng sau khi xem kết quả.

**Cần duyệt trước khi chạy:** xử lý ảnh MDMT mới (cắt crop, chạy DINOv2, mã hoá AVIF crop) trong phạm vi nội bộ của lab, và tải trọng số DINOv2-S dạng safetensors.
