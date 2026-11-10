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

## 1. Code, cách chạy, chi phí

| Thành phần | File |
|---|---|
| Train YOLO11n | `swarmbridge/experiments/uav_semcom/r1a_det.py`, `scripts/uav_semcom/run_r1a_det.sh` (22 phút GPU) |
| Đo | `r1a_measure.py`, `scripts/uav_semcom/run_r1a_measure.sh` (khoảng 25 phút GPU + đo thời gian khi máy rảnh) |
| Đánh giá | `r1a_eval.py` |
| Kết quả | `runs/uav_semcom/r1a/eval.json`, `runs/uav_semcom/r1a/summary.md`. Detection thô (`~/work/r1a/measure/measure.pkl`) không vào git; sha256 của trọng số ở `~/work/r1a/measure/sha256.txt` |

**YOLO11n trên UAV:** mAP50 holdout 0,31 (YOLO11s của BS: 0,475); 2,6 triệu tham số.

**Thời gian** (trung vị 30 ảnh, máy rảnh; CPU proxy là desktop 6 luồng, không phải phần cứng UAV):

| | ms/khung |
|---|---|
| YOLO11n @640 | 26 (CPU), 15 (GPU) |
| YOLO11n @960 | 50 (CPU), 14 (GPU) |
| YOLO11s ở BS, khung đầy đủ | 13 (GPU) |
| YOLO11s ở BS, mỗi ô | 9 (GPU) |
| Mã hoá AVIF (bảng A5) | 72–134 |
| Mã hoá mbt2018 trên CPU (bảng A5) | 460–910 |

Cả hai mức YOLO11n đều chạy ≥ 5 lần/s (≈ 20 và 38 lần/s).

## 2. Kết quả chính (cấu hình chọn trên val, chấm test / test-dev một lần)

Recall (điểm) ở ngân sách B. Δ_B = họ − RV_probe(B); CI cluster bootstrap theo video, B = 2000. Ngoặc ở các cột recall nhỏ / FP / AP50 là giá trị của RV_probe.

| pool | B | RV_probe (ảnh) | họ: cấu hình | recall | **Δ_B** [CI] | Mbit thực | recall nhỏ | FP/khung | AP50 | ms UAV / BS |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 0,08 | 42,9 | TOK: u960, τ 0,25 | 52,2 | **+9,2** [+6,4; +12,1] | 0,003 | 39,9 (29,2) | 17,7 (12,2) | 28,2 (26,7) | 50 / 0 |
| | | | HYB: u960, τ 0,25, nền avif_5 | 55,1 | **+12,1** [+9,6; +14,9] | 0,069 | 42,7 (29,2) | 24,0 (12,2) | 31,6 (26,7) | 121 / 13 |
| test | 0,3 | 58,4 | HYB: nền mbt q3 | 61,1 | **+2,7** [+1,3; +4,1] | 0,250 | 49,6 (46,4) | 22,6 (15,4) | 41,0 (**42,6**) | 526 / 13 |
| test | 1,2 | 61,1 | ROI: u960, τ 0,25, ô avif_50, nền avif_5 | 63,8 | **+2,7** [+1,7; +3,5] | 0,636 | 53,1 (49,9) | 42,7 (16,2) | 39,5 (**44,3**) | 441 / 93 |
| test-dev | 0,08 | 27,4 | TOK: u960, τ 0,25 | 45,3 | **+17,8** [+12,9; +21,6] | 0,002 | 32,2 (16,7) | 12,8 (8,3) | 24,2 (17,7) | 50 / 0 |
| | | | HYB: nền avif_5 | 47,5 | **+20,1** [+15,3; +23,8] | 0,099 | 33,8 (16,7) | 16,9 (8,3) | 27,4 (17,7) | 121 / 13 |
| test-dev | 0,3 | 46,7 | HYB: nền mbt q3 | 51,8 | **+5,1** [+3,1; +7,1] | 0,323 | 38,8 (33,9) | 17,2 (11,1) | 32,8 (31,6) | 526 / 13 |
| test-dev | 1,2 | 51,5 | ROI: ô avif_50, nền avif_5 | 54,2 | **+2,7** [+1,5; +3,5] | 0,727 | 41,3 (38,6) | 30,3 (12,5) | 32,2 (**36,3**) | 488 / 102 |

- Ở ngân sách rất thấp (0,005 và 0,02 Mbit), ảnh gần như vô dụng (recall 3–13), còn token ≈ 2–3 kbit cho recall 45–52 (Δ +37 đến +49).
- Token chỉ hơn ảnh ở ngân sách thấp: ở B = 0,3 và 1,2, token thua ảnh 1,5–9 điểm. Ở đó HYB và ROI mới hơn.
- Bảng mọi cấu hình (Mbit, recall, recall nhỏ, FP, ms) ở `summary.md`.

## 3. Kết luận theo quy tắc 0.6: INCONCLUSIVE

| Điều kiện | Kết quả |
|---|---|
| 1 | **Đạt ở cả ba mức vận hành, cả hai pool**, CI > 0: TOK và HYB ở 0,08; HYB ở 0,3; ROI ở 1,2 |
| 3 (khả thi) | **Đạt** |
| 2 (ảnh không bị thống trị: ảnh hơn mọi họ khác ≥ 2 điểm recall ở một mức B) | **Không đạt ở mức nào.** Theo recall, ảnh bị thống trị ở mọi ngân sách |

**PASS cần cả 1 và 2, và FAIL không xảy ra.** Protocol không liệt kê sẵn trường hợp "các họ không phải ảnh thống trị ảnh". Tôi xếp nó vào INCONCLUSIVE và nói rõ lý do, không đổi quy tắc sau khi xem kết quả.

## 4. Diễn giải

**1. Có đánh đổi thật, nhưng giữa các họ không phải ảnh, và theo nhiều chiều.**
- Theo (bit, recall), TOK → HYB → ROI lần lượt tốt nhất khi ngân sách tăng:

  | Họ | Bit | Recall (test-dev) | ms UAV |
  |---|---|---|---|
  | TOK | ~2 kbit | 45 | 50 |
  | HYB nền avif_5 | ~0,1 Mbit | 48 | 121 |
  | HYB nền mbt q3 | ~0,3 Mbit | 52 | 526 |
  | ROI | ~0,7 Mbit | 54 | 488 |

- Đây đúng là yêu cầu "ít nhất hai lựa chọn không bị thống trị". Nhưng nó không đúng với câu chữ của điều kiện 2, vốn đòi chính **ảnh** phải không bị thống trị.

**2. Ảnh vẫn không bị thống trị ở các chiều khác.**
- **AP50 ở ngân sách cao:** ảnh 44,3 / 36,3 so với ROI 39,5 / 32,2 và HYB 41,0 (test).
- **FP:** ảnh 12–16 FP/khung, ROI 30–43.
- **Tính toán ở BS:** 1 lần chạy so với khoảng 10 ô.
- **Bằng chứng hình ảnh đầy đủ** cho người vận hành.
- **Lý do:** gộp nhiều nguồn (token của YOLO11n, ảnh nền và ô qua YOLO11s) làm tăng TP ở ngưỡng 0,25, nhưng điểm tin cậy từ hai detector không cùng thang. Thứ hạng detection vì thế kém hơn, nên AP thấp hơn và FP cao hơn. Hiệu chỉnh điểm theo nguồn có thể giảm phần này; chưa thử.

**3. Cơ chế chính: phát hiện trước khi nén.**
- YOLO11n chạy trên **ảnh gốc chưa nén** ngay trên UAV: recall 45 trên test-dev ở u960. Con số này cao hơn YOLO11s ở BS chạy trên ảnh avif_5 (≈ 27) và ngang ảnh nén ≈ 0,3 Mbit.
- Nén mất vật nhỏ nhiều hơn mức mà một detector nhỏ bỏ sót.
- Đây là điều đã biết trong edge video analytics (Vigil, EdgeDuet). Bản thân nó **không mới**, nhưng nó giải thích vì sao bài toán chọn mode toàn ảnh của A5–G-L2 bỏ qua lựa chọn quan trọng nhất.

**4. Chọn ô theo độ không chắc trên UAV gần bằng oracle.**
- ROI-oracle (ô chọn bằng nhãn) chỉ hơn ROI 0,2–0,5 điểm recall, nhưng dùng ít ô hơn (7,2 so với 9,8 ô; ít bit hơn khoảng 25%).
- Giới hạn của ROI nằm ở detector BS và cách gộp, không ở cách chọn ô.

**5. Ca thất bại** (test-dev; số vật nhỏ mất so với ảnh RV_probe ở cùng B):
- Tập trung ở các khung **rất dày vật nhỏ**: 100–330 vật/khung, phần lớn nhỏ (chuỗi 9999938, 9999979, 9999952, 0000066).
- TOK @1,2 mất tới 21 vật nhỏ/khung. ROI cũng còn mất tới 19 ở 0000066 (275 vật, 960×540).
- Đó là trạng thái mà ở R1b nên chuyển sang khung đầy đủ. Chi tiết trong `eval.json` (`failures_testdev`).

## 5. Giới hạn

- **Tĩnh:** không có hành động "không gửi", không có thời gian, không có trạng thái hiểu biết của BS, không mô phỏng mạng.
- **Thời gian đo trên CPU desktop,** không phải phần cứng UAV; không đo năng lượng.
- **Họ ảnh là các ảnh nén độc lập.** Ở R1b phải thay bằng codec video liên khung.
- **Cấu hình ROI chọn trên val (≤ 1,2 Mbit) dùng 0,64–0,73 Mbit thực tế.** Bit thực tế có thể vượt ngưỡng val một chút ở pool khác.
- **Hai detector cùng train trên VisDrone-DET-train;** các pool A5 không trùng chuỗi với tập train này.
- **mbt q3 làm ảnh nền tốn khoảng 0,5 s mã hoá trên CPU.** Nền avif_20 (0,19 Mbit, recall 49,9 trên test-dev, 141 ms) là lựa chọn rẻ hơn về tính toán.

## 6. Hệ quả (theo 0.7)

INCONCLUSIVE cho phép sang **thiết kế** R1b, theo bản sửa ở `reports/swarmbridge-r1b-design.md`. Tôi không tự bắt đầu R1b. Những gì nên giữ cho menu biểu diễn:
- **TOK** (u960, τ 0,25): rẻ nhất, mạnh nhất khi bit chật.
- **HYB** với nền avif_5 / avif_20: tầm 0,1–0,2 Mbit.
- **ROI**: ở R1b chuyển thành ROI-QP **trong codec video**, không dùng ô ảnh độc lập.
- **Khung đầy đủ / codec video**: cho AP, FP thấp và các khung dày vật nhỏ.

**Câu hỏi của C trở nên sắc hơn:** token mạnh đến mức "khi nào token **không đủ**" (cảnh dày vật nhỏ, đối tượng mới, BS cần bằng chứng) là quyết định trung tâm. Đó đúng là chỗ trạng thái hiểu biết của BS và phản hồi ngữ nghĩa có thể có giá trị. Điều này phải kiểm ở R1b, so với x264 và Glimpse.

## 7. Phụ lục kiểm tra độ vững (KHÁM PHÁ, sau khi xem kết quả; kết luận INCONCLUSIVE ở mục 3 giữ nguyên)

Làm theo phản biện ngày 11/11. Không train lại.
- Code: `r1a_addendum.py`.
- Kết quả: `runs/uav_semcom/r1a/addendum.json`, bảng đủ ở `runs/uav_semcom/r1a/addendum.md`.
- Mọi lựa chọn (cách hiệu chỉnh, biến thể gộp, ngưỡng) làm trên train/val rồi khoá. Test và test-dev chấm một lần.
- Phụ lục này **không có CI** theo video; chỉ dùng để xem dấu và độ lớn.

**Cách làm:**
1. **So ở bitrate thực tế.** Mỗi cấu hình đã chọn được so với RV_probe chạy ở đúng số bit cấu hình đó dùng trên pool. Báo tỉ lệ vi phạm ngân sách.
2. **RV_probe thực thi thật.** Hỗn hợp hai thành viên λ được hiện thực hoá theo từng ảnh với seed cố định; AP50 là COCO trên chính các detection đó (không nội suy AP, không dùng "thành viên trội" như bảng mục 2).
3. **Hiệu chỉnh điểm theo nguồn.**
   - Logistic trên [logit(score), log diện tích] → P(TP), fit trên pool **train**, cho từng nguồn: token 640 / 960, BS trên từng ảnh nền, BS trên ô q50 / q80.
   - Sáu biến thể gộp: điểm gốc + NMS 0,6; hiệu chỉnh + NMS {0,5; 0,6; 0,7}; hiệu chỉnh + gom cụm theo lớp với điểm max hoặc noisy-OR giữa các nguồn.
   - Mọi biến thể bỏ box có điểm gốc < 0,05 và giữ tối đa 500 box/ảnh.
   - **Chọn trên val** theo AP50: TOK → hiệu chỉnh + NMS; HYB và ROI → hiệu chỉnh + noisy-OR.
4. **Recall ở cùng FP/khung.** Ngưỡng điểm của mỗi cấu hình đặt trên **val** sao cho FP/khung bằng FP/khung của RV_probe thực thi ở cùng B trên val, rồi dùng nguyên trên test / test-dev.

### 7.1 Kết quả (các mức vận hành; bảng đủ ở `addendum.md`)

Ngoặc ở cột AP50: RV_probe thực thi ở **bit thực tế của họ**. Ngoặc ở cột recall @ cùng FP: FP/khung đạt được. Cột cuối: RV_probe thực thi tại B (recall / FP / AP50).

| pool | B | họ | Mbit thực (so với B) | Δ recall tại bit thực | AP50 hiệu chỉnh (RV @ bit thực) | recall @ cùng FP (FP đạt) | RV_probe thực thi tại B |
|---|---|---|---|---|---|---|---|
| test | 0,08 | HYB nền avif_5 | 0,069 (−14%) | **+16,4** | **31,9** (23,8) | **48,6** (12,1) | 43,0 / 12,1 / 26,7 |
| test-dev | 0,08 | HYB nền avif_5 | 0,099 (**+24%**) | **+15,9** | **27,2** (20,2) | **42,3** (9,2) | 27,4 / 9,5 / 16,5 |
| test | 0,08 | TOK | 0,003 | +50 | 28,4 (≈ 0,6) | 47,9 (9,8) | 43,0 / 12,1 / 26,7 |
| test-dev | 0,08 | TOK | 0,002 | +44 | 24,2 (≈ 0,5) | 42,1 (8,3) | 27,4 / 9,5 / 16,5 |
| test | 0,3 | HYB nền mbt q3 | 0,250 (−17%) | +3,7 | 41,2 (41,0) | 57,8 (15,5) | 58,5 / 15,4 / 42,6 |
| test-dev | 0,3 | HYB nền mbt q3 | 0,323 (+8%) | +4,8 | 32,6 (31,7) | 48,9 (12,2) | 46,8 / 11,1 / 31,6 |
| test | 1,2 | ROI | 0,636 (−47%) | +2,7 | 41,0 (**43,8**) | 54,6 (17,3) | **61,2** / 16,2 / **44,3** |
| test-dev | 1,2 | ROI | 0,727 (−39%) | +3,4 | 33,6 (**35,2**) | 46,2 (12,3) | **51,4** / 12,5 / **36,2** |

### 7.2 Đọc kết quả

1. **Vi phạm ngân sách không đổi chiều kết luận.** Hai ca vượt ngân sách (HYB test-dev +24% ở B = 0,08; +8% ở B = 0,3) vẫn hơn ảnh khi so ở **đúng** số bit đã dùng: +15,9 và +4,8 điểm.

   Dù vậy, không được gọi các ca đó là "cải thiện ở cùng ngân sách B". Chỉ được nói "ở cùng bitrate thực tế".

2. **Khi kiểm soát FP, có một điểm giao rõ:**

   | Ngân sách | Kết quả |
   |---|---|
   | **Thấp (≈ 0,08 Mbit/khung)** | Token và hybrid thắng ảnh ở **mọi** thước đo: recall, AP50 (+2 đến +8 điểm), recall ở cùng FP (+5 đến +15 điểm), recall vật nhỏ |
   | **Vừa (≈ 0,3)** | Hybrid ≈ ảnh: AP50 chênh ≤ 1 điểm; recall ở cùng FP −0,6 (test) / +2,2 (test-dev, FP cao hơn 1,1/khung) |
   | **Cao (≥ 0,7)** | **Ảnh thắng.** ROI kém ảnh ở cùng bit về AP50 (−1,6 đến −2,8) và kém ở cùng FP về recall (−5 đến −7 điểm). Ưu thế recall +2,7 ở mục 2 đến từ việc ROI chấp nhận nhiều FP hơn, không phải từ thông tin tốt hơn |

3. **Hiệu chỉnh và gộp biết nguồn có giúp nhưng không đổi kết luận.**
   - AP50 của ROI tăng khoảng 1,4 điểm (39,5 → 41,0 test; 32,2 → 33,6 test-dev).
   - HYB gần như không đổi (±0,3 điểm).
   - Giới hạn của ROI là **thật**: ghép kết quả từ ô và ảnh nền chất lượng thấp không bằng một ảnh đủ chất lượng khi bit đủ.

4. **Hệ quả cho C.** Hai kiểu biểu diễn không bị thống trị theo nghĩa chặt (cùng FP):
   - token/hybrid khi bit chật;
   - ảnh khi bit đủ.

   **Bên nào thắng phụ thuộc ngân sách bit khả dụng,** tức trạng thái kênh và hàng đợi, đúng loại trạng thái mà A5.5 cho thấy đoán được. Ngưỡng giao khoảng 0,3 Mbit/khung trên dữ liệu này.

   ROI độc lập không đáng giữ ở dạng hiện tại. Ở R1b, ROI chỉ còn dưới dạng ROI-QP trong codec video.

**Giới hạn của phụ lục:**
- Khám phá, làm sau khi xem test.
- Không có CI theo video.
- Ngưỡng FP khoá trên val với 84 ảnh.
- Hiệu chỉnh là logistic đơn giản theo nguồn.
- Không thử detector UAV khác.
