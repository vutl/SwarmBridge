# SwarmBridge R1a: các họ biểu diễn có đánh đổi thật không? (ảnh tĩnh VisDrone-DET, offline)

Ngày 10/11/2026. Bước đầu tiên của ứng viên C trong `reports/swarmbridge-next-scenario-r0.md`, protocol đã sửa theo phản biện ngày 10/11.
- Được phép: train YOLO11n (qua khoá GPU) và đo các họ IMG / TOK / HYB / ROI.
- Không train PPO/SAC, không dựng simulator mới, không đụng UAV3D / ASR, không tải VisDrone-VID.

## 0. Quy tắc định trước (commit trước khi train hay đo bất cứ gì)

### 0.1 Câu hỏi

Trên ảnh UAV thật với detector thật, các **họ biểu diễn khác bản chất** (ảnh, token đối tượng, ảnh nền + token, ảnh nền + vùng chi tiết) có tạo **đánh đổi thật** giữa độ chính xác nhiệm vụ, bit, tính toán trên UAV và độ trễ không? Tức là có ít nhất hai họ không bị họ nào khác thống trị trên các ngân sách vận hành hợp lý?

R1a **không** kiểm hành động "không gửi" và **không** kiểm trạng thái hiểu biết của BS. Đó là phép kiểm chính của C, ở R1b. Vì vậy R1a không phải cổng sinh tử của C (xem 0.7).

### 0.2 Dữ liệu và detector

**Ảnh:** các pool A5 (train 304 / val 84 / test 128 ảnh VisDrone2019-DET-val, tách theo video; test-dev 403 ảnh). Nhãn chỉ dùng để chấm; riêng ROI oracle dùng nhãn để làm cận trên.

**Detector ở BS:** YOLO11s của A5 (train trên VisDrone-DET-train, imgsz 960). Bảng kết quả theo (ảnh, mode) đo ở L1 được dùng lại cho họ IMG.

**Detector trên UAV:** YOLO11n, khởi tạo từ trọng số COCO (`yolo11n.pt`).
- Train cùng dữ liệu và cùng cách chia như detector A5: VisDrone-DET-train, bỏ các chuỗi trùng với val, 10% chuỗi làm holdout.
- imgsz 640, 40 epoch, batch 16, seed 0, deterministic, augmentation mặc định.
- `best.pt` chọn theo fitness trên holdout.
- Hai mức năng lực, cùng trọng số: **UAV-640** (suy luận ở 640) và **UAV-960** (suy luận ở 960).
- Nó chưa từng thấy ảnh nào trong các pool A5.

**Điểm hoạt động:**
- **recall** = TP ở conf ≥ 0,25, IoU ≥ 0,5, ghép theo lớp, vùng ignore như A5.2;
- **AP50** COCO trên toàn bộ detection được giao;
- recall vật nhỏ (< 32² px);
- FP mỗi khung.

### 0.3 Các họ biểu diễn (lưới cố định)

| Họ | UAV làm gì | BS làm gì | Bit |
|---|---|---|---|
| **IMG(m)** | mã hoá ảnh, 14 mode A5 | YOLO11s trên ảnh giải mã (bảng L1) | payload thật |
| **TOK(τ, u)** | YOLO11n mức u ∈ {640, 960}; gửi các box conf ≥ τ, τ ∈ {0,1; 0,25; 0,4} | dùng box làm detection | 48 bit header + 52 bit/box (lớp 4, toạ độ chuẩn hoá 4 × 11, conf 4) |
| **HYB(τ, bg, u)** | token (conf ≥ τ, τ ∈ {0,25; 0,4}) + ảnh nền bg ∈ {avif_5, mbt q1, avif_20, mbt q3} | YOLO11s trên ảnh nền, gộp với token | token + payload ảnh nền |
| **ROI(τ_hi, q, bg, u)** | token cho box conf ≥ τ_hi (τ_hi ∈ {0,25; 0,4}); chia ảnh thành ô 320 px, bước 288 px; chọn ô chứa tâm của **detection không chắc của YOLO11n** (0,05 ≤ conf < τ_hi); mã hoá ô bằng q ∈ {avif_50, avif_80}; kèm ảnh nền bg ∈ {không, avif_5} | YOLO11s trên từng ô, ở **cùng tỉ lệ phóng như ảnh đầy đủ** (imgsz ô = 320 × 960 / cạnh dài ảnh, làm tròn bội 32, nên không có lợi thế phóng to), cộng ảnh nền, gộp với token | token + 24 bit/ô + payload ô + ảnh nền |
| ROI-oracle (cận trên, **không** triển khai được) | như ROI, nhưng chọn ô chứa vật ground-truth mà token bỏ sót | như ROI | như ROI |

**Gộp ở BS:** hợp các nguồn, rồi NMS theo lớp với IoU 0,6, giữ điểm cao nhất. Không dùng nhãn để chọn ROI; nhãn chỉ dùng ở ROI-oracle.

### 0.4 Chi phí được tính

- **Bit:** đủ header, metadata, toạ độ ô, token, payload ảnh nền và ô.
- **Tính toán trên UAV:**
  - suy luận YOLO11n, đo khi máy rảnh: CPU 1 luồng, CPU mọi luồng, GPU;
  - mã hoá ảnh nền và ô: thời gian AVIF đo thật;
  - mã hoá token: không đáng kể, ghi 0;
  - với IMG(m): thời gian mã hoá của bảng A5, cộng probe AVIF-5 cho luật RV_probe.
- **Tính toán ở BS:** số lần và thời gian chạy YOLO11s (IMG: 1; HYB: 1; ROI: số ô + ảnh nền), cộng thời gian gộp.
- **Độ trễ suy ra:** tính toán trên UAV + bit / tốc độ link danh nghĩa {1; 4} Mbit/s + tính toán ở BS.
- **Năng lượng:** không có phép đo năng lượng thật, nên **không báo** (chỉ báo thời gian). Ghi rõ là giới hạn.

### 0.5 Chọn cấu hình (chỉ train / val) và đánh giá (test, test-dev một lần)

**Ngân sách:** B ∈ {0,005; 0,02; 0,08; 0,3; 1,2} Mbit/khung. Ba mức sau là **mức vận hành hợp lý**: ở đó các codec ảnh chuẩn (avif_5 trở lên) chạy được.

**Chọn trên val:** với mỗi họ không phải ảnh và mỗi B, chọn **một** cấu hình có recall val cao nhất trong các cấu hình có bit val trung bình ≤ B (bằng nhau thì lấy ít bit hơn).

**Đại diện của IMG tại B** là luật triển khai được **RV_probe** (G-L2 mục 7.1, khớp đúng bit B), kèm mode cố định tốt nhất ≤ B để tham khảo.

**Trên test và test-dev:**
- Δ_B(họ) = recall(cấu hình đã chọn) − RV_probe(B). So ở **cùng ngân sách**, nên họ dùng ít bit hơn B không bị phạt.
- Δ'_B = RV_probe(B) − max recall của các họ không phải ảnh tại B.
- CI 95% bằng cluster bootstrap theo video (B = 2000), tính lại cả hai vế trong mỗi bản bootstrap.
- Báo kèm: AP50, recall vật nhỏ, FP/khung, bit thực tế, ms trên UAV, ms ở BS.

### 0.6 Kết luận (cố định trước)

**PASS: có đánh đổi thật.** Cần cả ba:
1. Ở ít nhất một mức vận hành hợp lý (B ≥ 0,08), một họ không phải ảnh có **Δ_B ≥ +2,0 điểm** recall, CI theo video > 0 trên test **và** test-dev.
2. Ở ít nhất một mức B khác, ảnh **không bị thống trị**: Δ'_B ≥ +2,0 điểm, CI > 0 trên cả hai pool.
3. Cấu hình thắng ở điều kiện 1 khả thi: YOLO11n mức tương ứng chạy ≥ 5 lần/s trên CPU proxy (mọi luồng, máy rảnh). Recall vật nhỏ và FP/khung của nó được báo, kể cả khi xấu hơn.

**INCONCLUSIVE:**
- có họ không phải ảnh với Δ_B > 0 (CI > 0 trên cả hai pool) nhưng < 2 điểm;
- hoặc điều kiện 1 chỉ đạt trên một pool;
- hoặc điều kiện 3 không đạt;
- hoặc chỉ hơn ở chiều tính toán/độ trễ mà không hơn ở recall.

**FAIL (không có biểu diễn khả thi nào ngoài ảnh):** ở mọi B ≥ 0,08 không họ nào khác có Δ_B > 0 với CI > 0 trên cả hai pool.

Mức +2 điểm là ngưỡng **ý nghĩa thực tiễn**, không phải tiêu chí sinh tử của C.

### 0.7 Hệ quả

| Kết quả | Hệ quả |
|---|---|
| PASS | Đề xuất R1b (thiết kế đã sửa, mục 2) |
| INCONCLUSIVE | Vẫn có thể sang R1b, vì R1b kiểm "không gửi" và trạng thái hiểu biết; ghi rõ biểu diễn nào đáng giữ |
| FAIL | Xem lại thiết kế biểu diễn trước khi xây R1b (ví dụ detector UAV mạnh hơn hoặc token kèm ROI khác). Không tự động bỏ C |

### 0.8 Ca thất bại và cận trên

**Ca thất bại** (báo bắt buộc):
- các ảnh mà họ được chọn mất nhiều vật nhỏ nhất so với IMG ở cùng B;
- đặc điểm của chúng: số vật thể, tỉ lệ vật nhỏ, độ phân giải, video.

**Cận trên:** ROI-oracle cho biết bao nhiêu phần thua của token là do **chọn ô sai**, bao nhiêu là do chính detector ở BS.
