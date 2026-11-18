# A7-S0: họ biểu diễn ngữ nghĩa (TOKEN / CROP / FRAME) trong môi trường A5 — thiết kế và cổng dư địa

Ngày 13/11/2026. Theo phản biện #14: quay lại A5 / A5.5 (PPO phân bổ tài nguyên theo trạng thái vô tuyến) và xem có kết hợp được semantic utility một cách có ý nghĩa không.

**Trạng thái:** thiết kế và cổng được commit **trước** khi chạy. Bước này **không train PPO**, không mở ảnh MDMT mới, không HOLD, không val.

## 1. Vì sao làm bước này, và vì sao chưa train

- **Đã thử một lần và thất bại:** A6.0-L1 / G-L2 đưa phần thưởng nhiệm vụ vào A5 với các mode là *mức chất lượng ảnh*. Kết quả: chính sách tốt nhất là một mode cố định; PPO chỉ có giá trị khi nghẽn (K5) hoặc khi không quan sát được nhiễu (phy_mcs_blind).
- **Điểm khác lần này:** họ biểu diễn từ pilot 0B chênh nhau khoảng 200 lần về bit (TOKEN ~5 kbit, CROP ~100–200 kbit, FRAME ~0,5–1 Mbit) và chênh nhau về CPU (0,06 s / ~0,2 s / ~1,2 s), với giá trị nhiệm vụ tăng dần (F1 0,63 / 0,66 / 0,71). Khoảng chênh lớn như vậy *có thể* làm lựa chọn phụ thuộc trạng thái kênh, điều mà các mức chất lượng ảnh không làm được.
- **Bài học R1b và G-L2:** đo dư địa bằng heuristic và cận trên thông tin trước; chỉ train khi có dư địa.

**Giả thuyết H-S:** trong môi trường A5, lựa chọn TOKEN / CROP / FRAME phụ thuộc trạng thái (tốc độ kênh, hàng đợi, hạn chót, nội dung) cho lợi ích nhiệm vụ cao hơn rõ rệt so với mode cố định tốt nhất, ở ít nhất một chế độ mạng.

## 2. Thiết lập

**Môi trường:** `a6l1_task.TaskSurveyEnv` (môi trường khảo sát A5 với phần thưởng nhiệm vụ), không đổi mô hình kênh, hàng đợi, hạn chót hay năng lượng. Ba chế độ cuối của A5.5: `ref`, `K5`, `phy_mcs_blind`.

**Bảng mode ngữ nghĩa** (thay bảng codec ảnh), dựng offline từ 918 bản ghi (chuỗi, epoch, góc) của pilot 0B:

| Mode | Nội dung | Bit | Thời gian trên UAV |
|---|---|---|---|
| TOKEN | mọi token chắc (55 bit / token + header) | đo | YOLO11n (là "probe" của môi trường) |
| CROP | token + mọi crop P1 (AVIF q50) | đo | tổng thời gian mã hoá P1 |
| FRAME | cả khung AVIF q35 | đo | thời gian mã hoá khung |

- Thời gian mã hoá mỗi mode = trung bình trên tập train (môi trường A5 dùng trung bình theo mode). Thời gian YOLO11n được tính như bước probe, luôn chạy trước quyết định mode.
- **Giá trị nhiệm vụ** như A6-L1: v = (TP − 0,5·FP) / V̄, khai báo với τ_decl = 0,3 cho mọi mode (giá trị được chọn nhiều nhất ở pilot; chọn trên EXPL, đã khai báo), khớp theo lớp, IoU ≥ 0,5. V̄ = trung bình (TP − 0,5·FP) của FRAME trên train.
- **Giả định phải khai báo:**
  - thời gian suy luận ở BS chưa đo: đặt 0 cho TOKEN và 0,05 s cho CROP / FRAME;
  - không có sàn PSNR (đặt PSNR = 99 cho mọi mode); chỉ còn ràng buộc hạn chót;
  - nếu chọn TOKEN, thời gian YOLO11n bị tính hai lần (probe + mode), khoảng 0,06 s so với hạn chót 4–12 s.
- **Thông tin nhân quả của bộ điều khiển** (sau khi YOLO11n chạy xong): bit TOKEN biết chính xác; bit CROP dự đoán tuyến tính từ số cửa sổ P1 và bit token (fit trên train); bit FRAME = trung bình train.

**Chia dữ liệu theo chuỗi (cụm):** thứ tự hash đã đăng ký ở pilot. Fold A: train = {56, 34, 52, 68, 48, 62}, test = {26, 57, 59, 31, 61}. Fold B đảo lại. Báo cả hai fold. Quy tắc chia là hash, nhưng tôi đã biết kết quả pilot theo chuỗi (chuỗi 59 khó), nên đây vẫn là khám phá.

**Chính sách** (không học; điều chỉnh trên train với seed 0..29, đánh giá trên test với seed 11000..11099):

| Ký hiệu | Mode | Thông tin |
|---|---|---|
| FIX | một mode cố định tốt nhất (cùng họ lập lịch slot của A5) | — |
| AGN | heuristic theo giá trị nhiệm vụ (`tv`: tham lam / pacedB + ngân sách thời gian) | bit dự đoán; giá trị = trung bình theo mode (không nhìn nội dung) → chỉ phụ thuộc trạng thái kênh / hàng đợi / hạn chót |
| CAU | như AGN | giá trị dự đoán từ đặc trưng trên UAV (số detection chắc / không chắc, tỉ lệ không chắc), fit trên train |
| ORA | như AGN | **cận trên thông tin**: bit thật và giá trị thật của từng ảnh (dùng GT; không triển khai được; không phải tối ưu) |

## 3. Cổng quyết định (đăng ký trước)

Chỉ số chính: **task_score** (TP − 0,5·FP của các khung giao đúng hạn, chuẩn hoá) và recall, trên test, trung bình 100 seed; kèm return nhiệm vụ.

- **G-S1 (có phụ thuộc trạng thái không):** ở ít nhất một chế độ, **ở cả hai fold**, ORA − FIX ≥ +2 điểm task_score **và** return của ORA cao hơn FIX.
  - Trượt → họ biểu diễn này cũng sụp về một mode cố định như G-L2. **Dừng** hướng "PPO chọn biểu diễn"; ghi nhận là kết quả âm.
- **G-S2 (heuristic nhân quả đã đủ chưa):** tại các ô qua G-S1, (CAU − FIX) / (ORA − FIX).
  - ≥ 70% → heuristic nhân quả đã lấy gần hết; PPO không có lý do rõ ràng. Giữ heuristic.
  - < 70% → có khoảng trống cho chính sách học; khi đó **đề xuất** (cần duyệt riêng) một pilot PPO mode-only nhỏ trên nền A5.5, không tự chạy.
- Ghi chú: ORA là cận trên về *thông tin*, không phải về *chính sách*; PPO còn có thể hơn heuristic ở chỗ mô hình của heuristic sai (như phy_mcs_blind ở A5.5). Vì vậy báo thêm AGN − FIX theo từng chế độ, và mode cố định tốt nhất của từng chế độ.

## 4. Giới hạn

- Dữ liệu nền là các bản ghi pilot 0B trên MDMT EXPL: **quyền dùng chưa được xác minh độc lập**; repo chỉ nhận chỉ số tổng hợp, bảng theo từng khung nằm ngoài git.
- Môi trường A5 là nhiệm vụ khảo sát ảnh rời (hạn chót 4–12 s), không phải video; mỗi lần chụp lấy một khung từ pool.
- 11 chuỗi, hai fold; τ_decl và ngưỡng 0,25 được chọn trên EXPL. Đây là sàng lọc khám phá.
- CPU đo trên desktop 1 luồng.
