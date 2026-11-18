# A7 / A8 / B1 (13–14/11)

Họ TOKEN / CROP / FRAME trong môi trường A5, Research Decision Review, thang sai lệch A8 (không hợp lệ), chọn góc nhìn B1 (NO-GO, nhánh đóng).

| # cũ | Tài liệu | Nội dung | Trạng thái |
|---|---|---|---|
| 18 | [swarmbridge-a7s-semantic-family-design.md](swarmbridge-a7s-semantic-family-design.md) | A7-S0: họ TOKEN / CROP / FRAME trong môi trường A5, thiết kế + cổng dư địa đăng ký trước (không train) | đã chạy, xem #19 |
| 19 | [swarmbridge-a7s-semantic-family-result.md](swarmbridge-a7s-semantic-family-result.md) | A7-S0 kết quả: FIX / AGN / CAU / ORA ở ref, K5, phy_mcs_blind, hai fold theo chuỗi | G-S1 đạt: cận trên thông tin +6–11% return, +8–10 recall ở ref / K5; dư địa là theo nội dung, không theo kênh; heuristic nhân quả không ổn định; PPO chưa chạy, chờ duyệt |
| 20 | [swarmbridge-a7s1-temporal-diagnostic.md](swarmbridge-a7s1-temporal-diagnostic.md) | A7-S1: giá trị khung t−1 / khung tráo cùng cảnh thay cho giá trị thật; tách bit thật và giá trị thật | Dư địa 91–99% từ giá trị thật; t−1 giữ 48–83%, khung tráo 30–80% → hiểu biết mức cảnh; NO-GO PPO mode-only trong A5 hiện tại |
| 21 | [swarmbridge-a7s2-bs-feedback-proxy.md](swarmbridge-a7s2-bs-feedback-proxy.md) | A7-S2: proxy không nhãn ở BS (detection mới của FRAME so với token) + heuristic thăm dò–khai thác nhân quả | Proxy xếp hạng cảnh tốt (0,80 / 0,96) nhưng dương giả ở cảnh đông; EE giữ 8% / âm / 71% / 18% dư địa → NO-GO cho cơ chế này; giá trị FRAME là đặc tính của góc nhìn |
| 22 | [swarmbridge-research-decision-review-2026-11-13.md](swarmbridge-research-decision-review-2026-11-13.md) | Research Decision Review: bằng chứng mạnh / yếu từ A5 đến A7-S2, A5 có hợp tác không, hai hướng tiếp theo kèm phép kiểm tra GO / NO-GO | Chờ quyết định; không thí nghiệm mới |
| 23 | [swarmbridge-a8-b1-preregistration.md](swarmbridge-a8-b1-preregistration.md) | Đăng ký trước A8 (thang sai lệch mô hình, PPO đóng băng so với H\*) và B1 (cơ hội hợp tác theo góc nhìn; kiểm ID xuyên góc trước) | đã khoá trước khi chạy |
| 24 | [swarmbridge-a8-b1-result-decision.md](swarmbridge-a8-b1-result-decision.md) | Kết quả A8 và B1 + báo cáo quyết định | A8 **không hợp lệ** (thang không tác động: không có nhiễu đồng kênh với các chính sách được so; Δ = +5,73 ở cả 5 mức); B1: ID xuyên góc đạt 8/11 chuỗi, tầng 1 chưa chạy (chờ cơ sở dùng dữ liệu) |
| 25 | [swarmbridge-b-feasibility-datasets.md](swarmbridge-b-feasibility-datasets.md) | Sau phản biện #19: sửa so sánh cùng ngân sách bit của B1 (Sửa đổi 1), khảo sát M3OT và CrossTrack-UAV, tính khả thi của hướng B theo từng bộ dữ liệu | B1 sẵn sàng về code; MDMT chờ quyền; M3OT giấy phép rõ nhưng chỉ 3/9 chuỗi qua sàng lọc ID, không thay thế trực tiếp được; CrossTrack-UAV cần ký thỏa thuận |
| 26 | [swarmbridge-b1-tier1-result.md](swarmbridge-b1-tier1-result.md) | B1 tầng 1 trên 8 chuỗi MDMT (sàng lọc có trợ giúp GT, cùng giới hạn bit): `VIEW` so với `TT` / `ALT` / `NDET` / `BIG` | **NO-GO cho `VIEW`**: quy tắc không nhãn `NDET` chọn cùng góc ở 8/8 chuỗi (hiệu 0,0); phân bổ theo góc vẫn hơn luân phiên +2,5 điểm F1 (7/8 chuỗi); góc tốt là UAV 2 ở 7/8 chuỗi |

Tổng quan mọi giai đoạn: [reports/README.md](../../../README.md).
