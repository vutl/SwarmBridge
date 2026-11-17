# RQ1: liên kết xuyên góc có phải một lý do để truyền? Protocol v2 (đăng ký trước)

Ngày 14/11/2026, theo phản biện #23. Câu hỏi gốc: mục 5 và 7 của [swarmbridge-collab-perception-novelty-review-2026-11-13.md](swarmbridge-collab-perception-novelty-review-2026-11-13.md).

**Bản v1 (commit 93fde5de) bị rút trước khi chạy lần nào.** Lý do: điều kiện `stale` của v1 khớp homography bằng box GT của nửa đầu, tức có dùng GT để dựng bộ liên kết; và oracle của v1 dùng GT để ghép đúng sau khi "gửi crop", nên không chứng minh được crop thật giúp phân định. v2 thay thế toàn bộ v1.

**Trạng thái:** Phase 0 được duyệt chạy ngay. Phase 1 và Phase 2 chỉ chạy nếu Phase 0 cho phép, theo đúng các quy tắc khoá dưới đây. Không ảnh MDMT mới, không chạy detector, không mã hoá AVIF, không trích đặc trưng, không train, không val / HOLD.

**Cơ sở dùng dữ liệu:** theo lời người dùng, giáo sư hướng dẫn cho phép lab dùng MDMT cho nghiên cứu này (xác nhận miệng, do người dùng thuật lại; chưa có văn bản; tác giả chưa trả lời; dữ liệu từ mirror không chính thức). Đây không phải giấy phép đã xác minh.

## 1. Hai câu hỏi tách riêng

- **RQ1-A:** liên kết xuyên góc có tạo ra một nút thắt đáng kể không? Tức F1 vật thể duy nhất mất bao nhiêu khi BS phải tự ghép hai góc, không có GT.
- **RQ1-B:** thông tin gửi thêm **thật** (kết quả YOLO11s trên crop đã tính sẵn) có làm giảm phần mất đó không, khi không cho nó nhìn danh tính GT?

## 2. Kết quả kiểm tra bản ghi (đã làm, chỉ đọc cấu trúc)

| Trường trong bản ghi pilot 0B | Có? |
|---|---|
| Detection YOLO11n trên UAV: box, lớp, độ tin cậy | có |
| Crop P1 (quanh detection bất định, độ tin cậy 0,05–0,25) và ô P2 (320 px): cửa sổ, số bit, thời gian mã hoá, detection YOLO11s ở BS (box, lớp, độ tin cậy) | có |
| FRAME: số bit, thời gian mã hoá, detection YOLO11s | có |
| Pixel của crop / FRAME, đặc trưng ngoại hình, embedding nhận dạng | **không** |

Hệ quả:
- Mọi phép liên kết không dùng GT chỉ có thể dựa trên **hình học, lớp và độ tin cậy**.
- Nếu crop thật có giúp, đó chỉ có thể là nhờ **detection tốt hơn** (box chính xác hơn, thêm vật bị sót), **không phải** nhờ thông tin ngoại hình. Không được tuyên bố "crop phân định danh tính".
- Crop chỉ có ở chỗ có detection bất định, không phải ở chỗ ghép mơ hồ; Phase 0 phải đo độ phủ này.

## 3. Bộ liên kết không dùng GT (cố định)

Đầu vào: detection mà BS đã nhận của hai góc trong epoch t (box, lớp). Điểm = tâm đáy box.

1. **Khởi tạo:** nếu epoch trước có homography hợp lệ (không GT) thì dùng nó; nếu không, ước lượng một phép đồng dạng (xoay + tỉ lệ + tịnh tiến) bằng RANSAC 2 điểm tự viết trên **mọi cặp cùng lớp** giữa hai góc:
   - 20.000 lần lặp, seed cố định; hai điểm mẫu cách nhau ≥ 50 px; tỉ lệ trong [1/3; 3];
   - điểm số = số detection **khác nhau** của góc 2 là láng giềng gần nhất cùng lớp (trong 40 px) của một detection góc 1 sau khi chiếu.
   - Lý do: `cv2.estimateAffinePartial2D` chọn nghiệm suy biến (co mọi điểm về một chỗ, tỉ lệ 0,035) trên cảnh tổng hợp khi không có cặp tương ứng cho trước; điểm số "mỗi điểm một lần" và giới hạn tỉ lệ loại nghiệm đó. Phát hiện bằng test tổng hợp, trước khi chạy trên dữ liệu.
2. **Tinh chỉnh hai lần:** lấy các cặp láng giềng gần nhất hai chiều, cùng lớp, sai số chiếu ≤ 80 px; nếu có ≥ 8 cặp, khớp homography bằng RANSAC (40 px) trên các cặp đó.
3. **Hợp lệ** nếu: homography hữu hạn; định thức phần 2 × 2 trong [0,1; 10]; và có ≥ 8 cặp láng giềng hai chiều trong vòng 40 px. Nếu không hợp lệ: epoch đó **không ghép gì** (mỗi detection là một vật), và được đếm là lỗi đăng ký. **Không bao giờ thay bằng homography từ GT.**
4. **Ghép:** cùng lớp, Hungarian trên sai số chiếu, ngưỡng 40 px.
5. **Detection mơ hồ:** có ứng viên cùng lớp ở góc kia trong vòng 80 px, và (≥ 2 ứng viên như vậy, hoặc ứng viên gần nhất xa hơn 20 px).

## 4. Phase 0: kiểm tra trước (chạy ngay; **chỉ trên các epoch nửa đầu**)

Nửa sau của mọi chuỗi không được dùng trong Phase 0. GT chỉ dùng làm thước đo chẩn đoán, không bao giờ để khớp.

Với 8 chuỗi đã qua sàng lọc ID (26, 31, 34, 52, 56, 57, 61, 62), chế độ TOKEN / TOKEN, mỗi epoch nửa đầu:
- bộ liên kết có hợp lệ không; số cặp; lý do thất bại;
- **sai số chẩn đoán:** chiếu tâm đáy box GT góc 1 (các ID có ở cả hai góc) bằng homography không GT, lấy trung vị khoảng cách tới box GT góc 2;
- số detection mơ hồ; tỉ lệ detection mơ hồ nằm trong một cửa sổ crop P1 hoặc ô P2 đã tính sẵn của cùng UAV, cùng epoch (tâm box nằm trong cửa sổ).

**Tiêu chí (khoá trước khi chạy):**
- Một chuỗi **đăng ký được** nếu bộ liên kết hợp lệ ở ≥ 50% epoch nửa đầu **và** sai số chẩn đoán trung vị (trên các epoch hợp lệ) ≤ 40 px.
- **RQ1-A xác định được** nếu ≥ 6/8 chuỗi đăng ký được. Nếu không: **RQ1-A = NOT IDENTIFIABLE FROM EXISTING RECORDS**, không chạy Phase 1, không đưa ra GO / NO-GO bằng số.
- **RQ1-B xác định được** nếu RQ1-A xác định được **và** crop / ô sẵn có (P1 ∪ P2) phủ ≥ 25% số detection mơ hồ (gộp nửa đầu của 8 chuỗi). Nếu không: **RQ1-B = NOT IDENTIFIABLE FROM EXISTING RECORDS**.

## 5. Phase 1: RQ1-A, ngân sách lỗi do liên kết (chỉ chạy nếu Phase 0 cho phép)

- Nửa sau mỗi chuỗi được chấm. Bộ liên kết khởi động từ homography hợp lệ cuối cùng của nửa đầu (không GT) rồi tự theo dõi.
- **Điều kiện liên kết:** `ideal` (theo ID GT; chẩn đoán được đánh dấu), **`free`** (bộ liên kết mục 3), `none` (không ghép).
- **Chế độ** dưới cùng giới hạn UL mỗi epoch của B1 (B_ref từ nửa đầu, κ = 1, quy tắc gửi và fallback như B1):
  - **TT** (quyết định): cả hai TOKEN;
  - **FV** (báo kèm): FRAME cho góc `NDET` của B1 (chọn ở nửa đầu, không nhãn), góc kia TOKEN.
- **Báo:** P / R / F1 vật thể duy nhất; FP sai, FP trùng, FP liên kết (ghép nhầm hai vật), vật bị bỏ sót; số cặp hợp lệ; tỉ lệ epoch đăng ký thất bại; kết quả từng chuỗi. Không bỏ chuỗi nào.
- **ΔA = F1(`ideal`) − F1(`free`)**, trung bình theo 8 chuỗi, chế độ TT.
- **Phán quyết RQ1-A:**
  - **INCONCLUSIVE** nếu > 50% epoch nửa sau (gộp) đăng ký thất bại;
  - nếu không: **NO-GO** nếu ΔA < 1 điểm F1; **GO (cơ hội)** nếu ΔA ≥ 3 điểm và ΔA > 0 ở ≥ 6/8 chuỗi; **INCONCLUSIVE** trong các trường hợp còn lại.

## 6. Phase 2: RQ1-B, thông tin gửi thêm thật có giải quyết được không? (chỉ chạy nếu Phase 0 cho phép)

Ba điều kiện, chế độ TT, bộ liên kết `free`:
- **A. Baseline:** liên kết không GT từ TOKEN.
- **B. REAL-CROP:** với mỗi detection mơ hồ (theo định nghĩa không GT ở mục 3), BS yêu cầu UAV đó gửi crop đã tính sẵn (AVIF q50) có cửa sổ chứa tâm detection, nếu có: crop P1 trước, nếu không có thì ô P2 (cùng tập crop với tiêu chí độ phủ của Phase 0). Detection YOLO11s của crop được gộp vào khai báo của góc đó (NMS theo lớp), rồi ghép lại. Chi phí: số bit đo được của crop, cộng 24 bit yêu cầu cho mỗi crop. GT không được dùng để chọn cặp mơ hồ, chọn crop, hay quyết định crop nào có ích.
- **C. GT-ORACLE:** mọi detection mơ hồ được ghép đúng theo GT. Chi phí giả định: trung vị bit crop P1 (q50) nửa đầu + 24 bit cho mỗi detection mơ hồ (vô hạn nếu không có crop nào để lấy giá). Chỉ là chẩn đoán lý thuyết thông tin, **không** phải bằng chứng rằng crop thật phân định danh tính.

**Phán quyết RQ1-B:**
- **Cơ hội lý tưởng** nếu ΔA ≥ 3 điểm và GT-ORACLE hồi phục ≥ 50% ΔA với chi phí ≤ 10% số bit trung vị của một FRAME.
- **REAL-CROP có ích** nếu nó hồi phục ≥ 50% ΔA với chi phí ≤ 10% FRAME. Ngay cả khi đạt, kết luận chỉ là "detection tốt hơn từ crop làm giảm lỗi liên kết hình học", không phải "phân định danh tính".
- Nếu chỉ GT-ORACLE đạt: "có cơ hội từ lỗi liên kết, nhưng thông tin truyền thêm hiện có chưa giải quyết được nó". Không phải GO triển khai.

## 7. Giới hạn

- 8 chuỗi; nhãn EXPL đã dùng nhiều lần: mọi kết luận là khám phá.
- Homography chỉ đúng với mặt đất.
- Bộ liên kết không GT là một thiết kế đơn giản cố định; nếu nó thất bại, điều đó không chứng minh liên kết là bất khả, chỉ cho thấy nó không xác định được từ bản ghi này với bộ này.
- Yêu cầu crop được giả định đến kịp trong cùng epoch; trễ phản hồi không được mô hình hoá.

## 8. Code

`swarmbridge/experiments/uav_semcom/h1_rq1_assoc.py` (v2) và test tổng hợp ở `test_h1_rq1.py`.
