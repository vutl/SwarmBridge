# R0 → R1b trên VisDrone (10–11/11)

Cổng G-L2, kịch bản R0, R1a, thiết kế và pilot R1b-0a; dừng vì H1 FAIL (không có dư địa).

| # cũ | Tài liệu | Nội dung | Trạng thái |
|---|---|---|---|
| 1 | [uav-semcom-g-l2.md](uav-semcom-g-l2.md) | Cổng G-L2: dự đoán lợi ích theo từng ảnh | **FAIL** (mục 8: vẫn FAIL sau khi sửa baseline) |
| 2 | [novelty-audit-g-l2-2026-11-10.md](novelty-audit-g-l2-2026-11-10.md) | Kiểm tra novelty sau G-L2 | xong |
| 3 | [swarmbridge-next-scenario-r0.md](swarmbridge-next-scenario-r0.md) | R0: chọn kịch bản tiếp theo (chọn C) | xong |
| 4 | [swarmbridge-r1a.md](swarmbridge-r1a.md) | R1a: token / hybrid / ROI so với ảnh, theo ngân sách bit | **INCONCLUSIVE** (đã đăng ký trước); mục 7 là khám phá |
| 5 | [swarmbridge-r1b-design.md](swarmbridge-r1b-design.md) | Thiết kế R1b v3: ablation thông tin phản hồi, các cổng H1 / H2 / H3, pilot R1b-0a | đã duyệt; H1 đã chấm (xem #9) |
| 6 | [visdrone-vid-data-audit.md](visdrone-vid-data-audit.md) | Kiểm toán VisDrone-VID: rò rỉ DET, cổng G7 theo nguồn (split v1) | xong |
| 7 | [swarmbridge-r1b0a-protocol.md](swarmbridge-r1b0a-protocol.md) | Protocol R1b-0a: split, timing nhân quả, số khoá, lưới tinh chỉnh, cổng H1 | đã khoá; G7 chấm một lần 11/11 |
| 8 | [swarmbridge-r1b0a-validation.md](swarmbridge-r1b0a-validation.md) | Kiểm chứng kỹ thuật bộ mô phỏng: chỉ số theo lớp, wire format, kênh ghép cặp, timing; 52 test | xong (dữ liệu tổng hợp) |
| 9 | [swarmbridge-r1b0a-h1-result.md](swarmbridge-r1b0a-h1-result.md) | Kết quả cổng H1 trên G7: bảng A1 / B / B+ / C1 / D, độ nhạy, chẩn đoán | **FAIL (không có dư địa)**; dừng nhánh phản hồi ngữ nghĩa |

Ghi chú nền R0: [r1-visdrone/r0-notes/](r0-notes/).

Tổng quan mọi giai đoạn: [reports/README.md](../../../README.md).
