# R1b-0a: báo cáo kiểm chứng kỹ thuật bộ mô phỏng

Ngày 10/11/2026. Trả lời phản biện về commit d3d6b4bc.

**Phạm vi:**
- Mọi kiểm tra dưới đây chạy trên **dữ liệu tổng hợp** và trên metadata của split.
- **Không** chạy hay xem bất kỳ kết quả nào trên G7, TUNE hay AUX.
- Các con số F1 trên dữ liệu tổng hợp chỉ để kiểm hành vi, **không** phải bằng chứng thực nghiệm, và không được dùng để chỉnh ngưỡng PASS / FAIL.

**Code:**
- `swarmbridge/experiments/uav_semcom/`: `r1b_proto.py`, `r1b_track.py`, `r1b_metrics.py`, `r1b_sim.py`, `test_r1b.py`.
- Lệnh chạy: `cd swarmbridge/experiments/uav_semcom && python -m pytest test_r1b.py`.
- Kết quả: **52 passed**. Môi trường: Python 3.12, numpy 1.26.4, scipy 1.17.1, py-motmetrics 1.4.0 (cài thêm, chỉ dùng cho test).

## 1. Các điểm phản biện và cách đã xử lý

| # | Vấn đề | Sửa | Test chứng minh |
|---|---|---|---|
| 1 | F1 khớp box không xét lớp | **Chỉ số chính là theo lớp:** TP cần cùng lớp và IoU ≥ 0,5, ghép trong từng lớp. Class-agnostic giữ làm phụ (`*_ag`). IDF1 có hai bản: theo lớp (ghép từng lớp, gộp) và class-agnostic | `test_right_box_wrong_class_is_fp_and_fn`: đúng box sai lớp → TP = 0, FP = 1, FN = 1; bản agnostic → TP = 1. `test_idf1_matches_py_motmetrics` (3 seed, có đổi id 3%, đổi lớp 15%, FP, bỏ sót): IDF1 agnostic và IDF1 theo lớp (motmetrics chạy từng lớp rồi gộp IDTP) **trùng tới 1e-9** |
| 2 | UL / DL dùng chung một RNG, nên số lần gửi khác nhau làm lệch chuỗi mất gói | Mọi lần rút là hash có khoá (seed, luồng, tick, chỉ số thông điệp, chỉ số gói), luồng UL / DL tách riêng. Gilbert–Elliott là chuỗi ngoại sinh theo khe 10 ms. Seed khoá trước: G7 {0..4}, TUNE {1000, 1001} | `test_uplink_losses_are_paired_across_conditions`: A1, C1, B gửi cùng tick, cùng số gói → cùng số phận mất / không mất (p_ul = 0,3, > 20 tick chung). `test_channel_seeds_differ_and_ge_chain_is_exogenous`: chuỗi GE không đổi theo cách truy vấn; seed khác thì mẫu mất khác |
| 3 | `Msg` là đối tượng Python, bit tính bằng công thức | **Pack / unpack thật.** Mô phỏng chỉ chuyển bytes, UAV giải mã. Bit downlink = 8 × số byte + header gói. `n_ops` 11 bit (tối đa 2047 ≥ 1024 id). Trường ngoài miền thì báo lỗi; track rời miền biểu diễn bị xoá khỏi trạng thái công bố. C0 thêm cờ snapshot (trước đó snapshot C0 không phân biệt được trên wire, không xoá được track ma; lỗi này lộ ra khi viết wire format) | `test_state_msg_roundtrip_and_exact_size` (C1, C0: 400 tick, có snapshot): giải mã trùng mọi trường, số bit đúng công thức, số byte = ceil(bit/8). `test_full_1024_track_snapshot_and_field_extremes`: snapshot 1024 track với giá trị biên min / max → bản sao trùng hệt. `test_out_of_range_fields_are_rejected` (8 trường hợp). `test_n_ops_limit_and_bad_ids` (> 2047 op, id 1024, bytes bị cắt). `test_hint_ack_ulsync_roundtrip_sizes`. `test_fast_objects_leave_frame_without_unrepresentable_state`. Bốn test giao thức gốc giờ chạy qua wire bytes, kể cả 10⁵ thông điệp mất / lặp / đảo thứ tự |
| 4 | B chỉ có box, có thể làm baseline yếu không cần thiết | Mô tả đúng: B = **chỉ box** (27 + 48 bit/box). Thêm **B+** = box + lớp + conf (27 + 55 bit/box); UAV phát hiện được lệch lớp. Cổng H1 so C1 với **max(B, B+)** | `test_hint_ack_ulsync_roundtrip_sizes` (kích thước theo box), `test_b_plus_carries_class_and_costs_more`; B+ có mặt trong mọi test nhân quả / rò rỉ |
| 5 | Timing / mạng | Gói vào hàng đợi uplink **lúc mã hoá xong**, trước đây là lúc quyết định, nên trước đây một TOK không bao giờ vượt được FRAME ở mô hình pipelined. Kiểm tra deadline cục bộ tính cả gói đang mã hoá. BS xử lý quan sát cũ đến muộn theo kiểu out-of-sequence | `test_exact_header_overhead_and_arrival_time`: bit = payload + 48 + 96·⌈payload/9600⌉; sẵn sàng = 29 + 2 + 92 ms; tới BS = sẵn sàng + bit/R + 10 ms (chính xác). `test_budget_holds_when_real_frame_sizes_differ_from_estimate` (ước lượng 50 kbit, thật 300 kbit): bit UL ≤ R·thời lượng + độ sâu bucket + một khung. `test_deadline_admission_and_late_drop`. `test_serial_compute_drops_next_frame_after_encode`. `test_tracker_out_of_sequence_observation_does_not_roll_back`. `test_pipelined_out_of_sequence_at_bs_is_handled`: có quan sát sai thứ tự, không desync |
| 6 | TOK không xoá trực tiếp được track sai | Ghi rõ là giới hạn của menu. Thêm chẩn đoán: track ma ở BS (giây / phút); tỉ lệ track ma mà UAV không thấy vật; tỉ lệ track ma UAV vừa không thấy vừa biết BS đang giữ ("sửa được nếu menu cho phép"). Chẩn đoán đọc GT nhưng **không** dùng cho quyết định | `test_c1_belief_beats_a1_with_good_downlink_and_diagnostics` kiểm miền giá trị. Các test "không phụ thuộc trạng thái BS ẩn" chỉ so hành động, nên chẩn đoán không lọt vào quyết định |

## 2. Các test có từ trước, vẫn qua

| Nhóm | Phép thử |
|---|---|
| Giao thức | Quay vòng 16 bit (phiên bản, t_state; thêm `unwrap`); 10⁵ thông điệp mất 10%, lặp 5%, đảo thứ tự → bản sao bằng đúng trạng thái công bố sau mỗi snapshot (> 3.000 lần kiểm); CRC bắt 100% sai lệch ±1 và cùng nội dung khác thời điểm; op hỏng → desync; dự đoán số nguyên trùng bit sau 1.000 tick |
| Chỉ số | GT làm output → 100 / 100, trễ 0; trễ cấy 7 tick → 0,7 s; vùng bỏ qua không tính FP |
| Tracker | GT làm detection → F1, IDF1 > 99 |
| Nhân quả | Cắt clip không đổi hành động trước điểm cắt (**7 điều kiện**). Mất hết downlink thì hành động của A0 / A1 / B / B+ / C0 / C1 không phụ thuộc detection của BS; D thì có |
| Split | G7 / cách ly / TUNE / AUX đúng danh sách, 0 vi phạm, 0 trùng nội bộ |

## 3. Lỗi tìm ra trong hai vòng kiểm chứng (đều trước mọi dữ liệu đánh giá)

1. Bộ chấm lấy nhầm danh sách nên FP luôn bằng 0.
2. Trạng thái chuẩn bị lượng tử hoá lại, sinh op cập nhật thừa trên downlink.
3. Ước lượng kích thước FRAME lấy trung vị của chính clip.
4. F1 không xét lớp.
5. Hai điều kiện dùng chung RNG theo thứ tự gọi.
6. Snapshot C0 không phân biệt được trên wire.
7. `n_ops` 8 bit không đủ cho snapshot lớn.
8. Uplink xếp hàng theo thời điểm quyết định, nên mô hình pipelined không bao giờ tạo gói sai thứ tự.

## 4. Giới hạn còn lại (biết trước khi chạy)

- **Kiểm chứng trên dữ liệu tổng hợp** chỉ kiểm logic. Tracker chưa tinh chỉnh trên VID; tham số tracker khoá theo protocol mục 6 và **như nhau** ở mọi điều kiện.
- **Mô hình hoá còn đơn giản:**
  - B / B+ theo vết box gợi ý bằng track cục bộ của UAV (kiểu Glimpse). Đây là một cách hiện thực, không phải bản sao Glimpse;
  - token mang conf của track cục bộ;
  - BS gộp token và detection YOLO11s như cùng một loại quan sát. Gộp hiệu chỉnh theo nguồn ở R1a **chưa** đưa vào.
- **M0 (không có metadata chuyển động);** camera chuyển động làm dự đoán trong toạ độ ảnh kém đi như nhau ở mọi điều kiện.
- **Thời gian đo trên desktop 6 luồng**, không phải phần cứng UAV.
- **G7 có 7 nguồn,** 5 seed kênh. Đây là cổng khả thi, không phải xác nhận thống kê.

## 5. Trạng thái

- Phản biện cho phép chạy thí nghiệm chính sau khi có kiểm toán ASR. Thứ tự: khoá protocol bằng commit → tính trước trên TUNE → khoá FRAME / R_ul → tinh chỉnh → chấm G7 đúng một lần.
- **Blocker hiện tại:** kiểm toán phạm vi ASR (chỉ đọc) của Codex.
