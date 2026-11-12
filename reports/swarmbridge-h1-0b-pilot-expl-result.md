# SwarmBridge H1-CROP: kết quả pilot 0B (EXPL, khám phá)

Ngày 13/11/2026. Code readiness v3 (đã duyệt ở phản biện #11). Chạy một lần, ngưỡng không đổi sau khi thấy kết quả.

> **Đọc kết quả (phản biện #12):** Gate 0B là **PASS khám phá với hiệu ứng yếu**, không phải bằng chứng cho paper. Đơn vị độc lập là **11 chuỗi** (cụm), không phải 459 epoch.

- **Kết quả số:** `runs/uav_semcom/h1/pilot0b_expl_v1.json`.
- **Tình trạng quyền dùng dữ liệu (đã sửa 13/11 theo phản biện #12):** người nghiên cứu tự thực hiện việc dùng nội bộ, khám phá. **Giấy phép của dataset và quyền của bản mirror bên thứ ba chưa được xác minh độc lập** (chưa có trả lời của tác giả, chưa có ý kiến của lab / trường). Repo này không phát hành lại ảnh hay nhãn gốc nào; file kết quả chỉ chứa chỉ số tổng hợp, số GT theo chuỗi, cấu hình và đường dẫn. Bản trước của mục này (và của provenance) ghi rằng việc dùng "được phép" (*permitted*); khẳng định đó không có cơ sở và đã được sửa.
- **Không chạy thêm thí nghiệm MDMT nào trên ảnh** cho đến khi quyền dùng được làm rõ. Phân tích ở mục 6 chỉ dùng các bản ghi đã có.
- **Dữ liệu:** 11 cặp chuỗi MDMT-test (EXPL), 459 epoch, 2 góc, khoảng 45.000 box GT. val, HOLD và VisionTrack-test vẫn đóng.
- **Bản chất:** khám phá. Nhãn EXPL đã được dùng ở Gate 0A, và α, τ_decl được chọn trên chính các chuỗi này. Đây là sàng lọc, không phải xác nhận.

## 1. Smoke và căn chỉnh

- **Căn chỉnh ảnh–nhãn đạt.** Đã vẽ nhãn lên ảnh chuỗi 52 và 68 (cả hai có số file lệch: 52 bắt đầu từ 191, 68 từ 391) ở khung đầu, giữa và cuối. Box ôm đúng xe ở cả hai góc, và cùng ID chỉ cùng xe.
- **Smoke** (2 chuỗi × 10 epoch) chạy thông. Số ứng viên, bit và thời gian đều hợp lý. Không đọc tiêu chí nào ở bước này.

## 2. Kết quả theo tiêu chí đã đăng ký

| | Đo | Kết quả | Ngưỡng | Đạt? |
|---|---|---|---|---|
| C1 | GT mà YOLO11n (conf ≥ 0,25) bỏ sót | **49,5%** (14–82% tuỳ chuỗi) | ≥ 15% | ✓ |
| C2 | Recall của ứng viên P1 trên vật bị sót | **59,3%**; 30 ứng viên mỗi góc mỗi epoch, trong đó **12 là ứng viên sai** (55 kbit) | ≥ 50% | ✓ |
| C3 | T1 − max(T0, T0+), F1 từng góc | **q50:** đạt ở B = 76k và 305k (ΔP = −0,047 và −0,043). **q35:** không đạt ở mức B nào (ΔP = −0,055 đến −0,094) | > 0 ở cả macro và micro, và ΔP ≥ −0,05 | ✓ (chỉ q50) |
| C4 | Trùng lặp mức đề xuất; F1 duy nhất T2 − T1 | **40,7%**; **+0,3 đến +0,8 điểm** ở mọi B, cả macro và micro, mọi ρ | ≥ 20%, > 0 | ✓ |
| C5 | CPU UAV: YOLO11n + mã hoá mọi P1 | trung vị **201 ms**, p90 533 ms (q35) | trung vị ≤ 1 s | ✓ |

**Theo quy tắc đọc của protocol (mục 6), cả C1–C5 đều đạt.**

## 3. Nhưng hiệu ứng nhỏ và tập trung: đánh giá của tôi (không đổi ngưỡng)

F1 từng góc theo các mức ngân sách UL chung B (q50, α và τ_decl được chọn riêng cho mỗi ô):

| B (kbit/epoch) | T0: F1 macro / micro, P, R, UL | T1 (q50): F1 macro / micro, P, R, UL | ΔF1 macro / micro | Số chuỗi dương |
|---|---|---|---|---|
| 76 | 0,628 / 0,615; P 0,79; R 0,50; 4,5 kbit | 0,640 / 0,622; P 0,74; R 0,53; 69 kbit | +1,2 / +0,7 | 7/11 |
| 153 | như trên | 0,644 / 0,624; P 0,71; R 0,56; 131 kbit | +1,6 / +0,9 | 6/11 |
| 305 | như trên | 0,656 / 0,633; P 0,75; R 0,55; 226 kbit | +2,8 / +1,8 | 8/11 |
| 1.844 | **F (cả khung):** 0,711 / 0,696; P 0,81; R 0,61; 1.050 kbit | | | |

**1. Lợi ích của crop gần như chỉ đến từ một chuỗi.**
- Ở chuỗi 59, YOLO11n bỏ sót 82% vật, T0 chỉ đạt F1 0,24, và crop nâng lên 0,35–0,46.
- **Bỏ chuỗi 59:** ΔF1 macro chỉ còn +0,1 đến +0,8 điểm (q35 / q50, mọi B). Trung vị chênh lệch theo chuỗi là −0,5 đến +0,9 điểm.

**2. Đổi precision lấy recall.**
- Recall tăng 3–6 điểm, precision giảm 4–9 điểm, FP tăng 2,5–5,5 mỗi góc mỗi epoch.
- 40% ứng viên crop là sai (không chứa vật nào).

**3. Chi phí bit lớn so với lợi ích.**
- Token chỉ tốn 4,5 kbit mỗi epoch; crop tốn 69–226 kbit (gấp 15–50 lần) cho +1–3 điểm.
- **T0+ trùng hệt T0:** token yếu không giúp gì, vì sau lượng tử conf 3 bit chúng đều nằm dưới τ_decl được chọn. Như vậy "gửi ảnh vùng nghi ngờ" có giá trị mà "gửi token yếu" không có, nhưng giá trị đó nhỏ.

**4. Cả khung (F) mạnh hơn nhiều nhưng không vừa CPU.**
- F đạt F1 0,711, tức +8,3 điểm so với T0, ở khoảng 1 Mbit mỗi epoch.
- Nhưng mã hoá AVIF cả khung trên 1 luồng mất **1.165 ms trung vị (p90 1.429)**, vượt chu kỳ 1 s. Thắt cổ chai lại là **CPU trên UAV**, không phải đường truyền: không có điều kiện nào có airtime vượt 1 s, kể cả ở 2 Mbit/s.
- Ở các mức B thấp hơn, khung không vừa phần ngân sách của mỗi UAV, nên F quay về T0.

**5. P2 (ô 320 px)** kém T1 và tốn CPU 0,9–1,0 s trung vị (p90 1,7–2,0 s). Loại.

**6. Phối hợp (C4):**
- Heuristic T2 có trợ giúp GT chỉ thêm +0,3 đến +0,8 điểm F1 duy nhất.
- Chi phí phối hợp nhỏ: khoảng 2,3 kbit bản kê UL và 89 bit DL; T_coord khoảng 0,5 ms ở 5 Mbit/s.
- T2v gần như bằng T2.
- Đây là heuristic dùng GT, không triển khai được và không phải cận trên. Nó chỉ cho thấy dư địa phối hợp ở mức đề xuất là nhỏ trong thiết lập này.

## 4. Kết luận

- **Theo tiêu chí đã đăng ký: GO (sàng lọc).** C1–C5 đều đạt; C3 chỉ đạt với q50.
- **Đánh giá thực chất của tôi: GO yếu, gần với INCONCLUSIVE.** Crop theo detection không chắc của YOLO11n có giá trị thật nhưng nhỏ: khoảng +1–3 điểm F1, phải trả bằng precision và gấp 15–50 lần số bit của token. Lợi ích chỉ lớn ở chuỗi mà detector trên UAV gần như "mù" (chuỗi 59). Ngoài các chuỗi đó, lợi ích gần bằng 0.
- **Protocol** nói rằng nếu C1–C5 đạt thì đề xuất xác nhận trên HOLD và Gate 1, và **cần duyệt riêng**. Tôi **không** khuyên dùng HOLD cho chính sách crop cố định này: lợi ích kỳ vọng ngoài các chuỗi khó chỉ khoảng 1 điểm.

**Điều pilot cho thấy và đáng giữ** (đây là giả thuyết, chưa phải kết quả):
- Giá trị của mỗi biểu diễn phụ thuộc mạnh vào **trạng thái**:
  - token đủ tốt khi detector trên UAV tốt;
  - crop hoặc cả khung đáng giá khi detector kém (chuỗi 59);
  - khung tốt nhất nhưng bị giới hạn bởi CPU.
- Đây chính là bài toán chọn **token / crop / khung / bỏ qua theo trạng thái**, dưới ràng buộc bit và CPU, là chỗ chính sách A5 / A5.5 (PPO) có thể đóng góp.
- Bước tiếp theo hợp lý (cần duyệt): đo xem có tín hiệu rẻ trên UAV nào, ví dụ tỉ lệ detection không chắc, dự đoán được khi nào crop hoặc khung có lợi; làm offline trên chính các bản ghi này, không cần ảnh mới.

## 5. Giới hạn

- α và τ_decl được chọn trên cùng tập (lạc quan cho mọi điều kiện như nhau).
- 11 chuỗi; một chuỗi chi phối kết quả.
- Detector huấn luyện trên VisDrone, chưa tinh chỉnh trên MDMT; ánh xạ lớp. F1 không phân lớp có trong file kết quả.
- CPU là CPU desktop 1 luồng, không phải phần cứng UAV.
- Không có hàng đợi, không mất gói. Chỉ có ứng viên P1 từ detection không chắc; chưa có đề xuất theo chuyển động.
