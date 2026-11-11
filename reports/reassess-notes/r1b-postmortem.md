# R1b-0a postmortem (11/11/2026)

Nguồn: [swarmbridge-r1b0a-h1-result.md](../swarmbridge-r1b0a-h1-result.md) (commit `1655425d`) và chẩn đoán TUNE (`scripts/uav_semcom/r1b_tune_diag_D.py`). Không chạy thêm G7, không chạy thêm thí nghiệm nào.

## 1. Giới hạn của cách triển khai (không phải kết luận khoa học)

| # | Giới hạn | Bằng chứng | Hệ quả cho H1 | Lẽ ra phải phát hiện bằng |
|---|---|---|---|---|
| L1 | **CPU là nút thắt, không phải đường lên.** R_F được khoá theo thời gian phát một FRAME (≈ 151 ms). Nhưng CPU chỉ cho 1 FRAME mỗi 200 ms (29 + 2 + 104 ms, worker tuần tự, rơi tick). | Ở 2×/4×: 98–100% quyết định là FRAME, 50% tick rơi; 7 điều kiện cách nhau ≤ 0,05 F1. | Hai trong ba mức tiêu chí không thể phân biệt điều kiện nào. | Bảng "ràng buộc nào đang chặn" (compute / UL / DL) ở từng mức, trước khi khoá mức. |
| L2 | **Luật kích hoạt bão hoà: không bao giờ SKIP.** Tracker cục bộ nhận detection conf ≥ 0,05, nên luôn có "track mới"; g luôn vượt ngưỡng TOK. | SKIP = 0% ở 0,5× và 1×; TOK ≈ 90–100%. | Phản hồi chỉ còn ảnh hưởng nội dung TOK, không ảnh hưởng quyết định gửi hay không. | Kiểm tra độ đa dạng hành động trên TUNE (mỗi hành động phải xuất hiện đáng kể với ít nhất một điểm lưới). |
| L3 | **Oracle D không phải chặn trên.** Policy heuristic không đơn điệu theo độ chính xác niềm tin. | D < A1 khoảng 2 điểm ở mọi điểm lưới (TUNE) và trên G7; DL16 làm C1 *tốt hơn* (+4,2). | Nhãn "không có dư địa" chỉ đúng cho menu này. | Kiểm tra "trần hợp lệ" trên TUNE: oracle phải ≥ mọi baseline; nếu không, cổng H1 không đo được dư địa. |
| L4 | **Bản đồ BS chất lượng thấp:** precision 14–21%, track ma chiếm phần lớn. | Bảng §3.4 báo cáo H1. | F1 kém nhạy; lợi ích nhỏ bị chìm. | Ngưỡng sàn cho chỉ số trên TUNE (ví dụ precision của D ≥ 50%). |
| L5 | **Mất gói tất cả hoặc không, không có FEC hay ARQ.** | FRAME mất 27–30%; ở p_ul = 5% chỉ còn 22–28% FRAME tới. | Kênh không giống lớp liên kết thật; phạt nặng mọi hành động FRAME. | Mô hình lớp liên kết có ARQ giới hạn hoặc mã sửa lỗi. |
| L6 | **Một UAV, không có chiều hợp tác.** | Thiết kế R1b-0a. | Không nói gì về bầy; chỉ là bước kiểm tra cơ chế phản hồi. | (cố ý, theo thiết kế v3) |

L1–L4 đã **lộ ra trên TUNE trước khi chấm G7**, nhưng protocol không có lối thoát "thí nghiệm không hợp lệ". Nó chỉ có PASS / FAIL / INCONCLUSIVE, nên kết quả bị đẩy vào FAIL.

## 2. Kết luận khoa học thực sự còn đứng (hẹp)

1. **Trong thiết lập này, phản hồi trạng thái có tính phí (C1) chỉ thêm ≤ +1,24 F1 so với ACK đơn thuần (A1)** ở mức duy nhất có ràng buộc đường lên. Kết quả đều qua nguồn và seed, nhưng dưới ngưỡng +2 đã đăng ký.
2. **Biết gói nào đã tới quan trọng hơn biết chi tiết trạng thái BS:** A1 − A0 = +3,4 F1 ở 1×, so với C1 − A1 = +1,24.
3. **Gợi ý box từ BS (B / B+) gây hại** (B+ −5,4 và B −6,1 F1 so với A1 ở 1×): gợi ý cũ đi theo track cục bộ sai lệch.
4. **Bài học tổng quát:** giá trị thông tin bằng 0 khi quyết định không thay đổi. Nếu hành động bị ràng buộc khác khoá chặt (CPU, kích hoạt bão hoà), thêm thông tin cho bên gửi vô ích. Bài học này cùng chiều với hai tiền lệ nội bộ: Town3 `91179ca1` và UAV3D v1.1 `70486c1b`.

**Không được kết luận** "phản hồi ngữ nghĩa vô ích nói chung", vì L2–L3 làm phép thử không đo được trần.

## 3. Những gì tái sử dụng được

| Thành phần | Dùng lại cho | Ghi chú |
|---|---|---|
| R1a: đánh đổi TOK / HYB / IMG theo ngân sách bit | Menu biểu diễn của mỗi UAV | TOK và HYB thắng recall ở ≤ 0,3 Mbit/khung; ảnh thắng ở ≥ 0,7 Mbit/khung; điểm giao ≈ 0,3 Mbit/khung. R1a là bằng chứng nền, **không đủ** cho paper. |
| Wire format chính xác đến bit, đóng gói, header | Mọi mô phỏng có tính chi phí điều phối | `r1b_proto.py`, đã test round-trip. |
| Kênh ngẫu nhiên ghép cặp có khoá (blake2b) | So sánh ghép cặp giữa các chính sách | `r1b_sim.py`. |
| Mô hình thời gian nhân quả (tính toán trên UAV, hàng đợi, deadline) | Mọi mô phỏng UAV | Phải thêm bảng ràng buộc (L1). |
| Chỉ số theo lớp, khớp motmetrics đến 1e-9 | Đánh giá | `r1b_metrics.py`. |
| Quy trình split theo nguồn + kiểm toán rò rỉ pHash | Mọi dataset mới | `r1b_split.py`, `vid_audit`. |
| Khung chấm cổng một lần (luật H1 thành hàm có test, từ chối chấm lại) | Mọi cổng sau | `r1b_run.py`. |

## 4. Thay đổi quy trình đề xuất cho mọi cổng sau

Thêm **cổng hợp lệ trên TUNE**, trước khi đụng tập chấm. Nếu không qua thì kết luận là **INVALID** (thiết kế lại), không phải FAIL:

1. **Ràng buộc:** ở mỗi mức tài nguyên, ràng buộc mà giả thuyết nhắm tới (ví dụ uplink chung) phải là ràng buộc đang chặn.
2. **Đa dạng hành động:** mỗi hành động trong menu phải được chọn ở tỉ lệ đáng kể với ít nhất một cấu hình.
3. **Trần hợp lệ:** điều kiện oracle (hoặc tìm kiếm có biết trước) phải ≥ mọi baseline trên TUNE.
4. **Độ nhạy chỉ số:** chỉ số chính phải phân biệt được oracle và baseline yếu nhất với biên đủ lớn so với ngưỡng hiệu ứng.

Bốn kiểm tra này dùng TUNE, nên không làm rò tập chấm, và sẽ dừng R1b-0a trước G7.
