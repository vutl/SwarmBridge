# SwarmBridge: báo cáo

Cập nhật 11/11/2026. Ngoài cùng chỉ giữ tài liệu của **hướng đang làm**. Các giai đoạn trước nằm trong [archive/](archive/README.md).

## Hướng hiện tại: semantic communication cho UAV (VisDrone), đi từ R0 đến R1b

Đọc theo thứ tự:

| # | Tài liệu | Nội dung | Trạng thái |
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
| 10 | [swarmbridge-direction-reassessment.md](swarmbridge-direction-reassessment.md) | Đánh giá lại hướng sau R1b: postmortem, prior art, 3 hướng, novelty / dataset matrix, cổng 0 trên MDMT. Ghi chú nguồn: [reassess-notes/](reassess-notes/) | H1 duyệt ở mức ứng viên; Gate 0 thay bằng #11 |
| 11 | [swarmbridge-h1-gate0-protocol-v2.md](swarmbridge-h1-gate0-protocol-v2.md) | Protocol Gate 0 v2 cho H1: phân loại oracle, đặc tả bộ chấm F1 đối tượng duy nhất, ma trận baseline, split chống thiên lệch chọn, quyết định quyền dữ liệu, checklist | v2.1 khoá cho 0A; 0B chưa duyệt |
| 12 | [swarmbridge-h1-gate0a-result.md](swarmbridge-h1-gate0a-result.md) | Gate 0A (khám phá, nhãn MDMT-test, báo cáo hoàn hảo): dư địa phối hợp theo ngân sách, TOK và CROP riêng | TOK: dư địa hẹp (≈ dòng token); CROP: dư địa ở 100–500 kbit/epoch |
| 13 | [swarmbridge-h1-gate0a-audit.md](swarmbridge-h1-gate0a-audit.md) | Kiểm toán 0A: S2 là oracle có trễ; oracle chính xác (MILP); chi phí bản kê; baseline nhớ lịch sử; điều kiện C1–C6 cho CROP | TOK NO-GO; CROP GO có điều kiện; v2 sửa chi phí bản kê (mục 8) |
| 14 | [swarmbridge-h1-0b-tune-pilot-protocol.md](swarmbridge-h1-0b-tune-pilot-protocol.md) | Protocol pilot 0B-TUNE cho CROP: ứng viên không GT, AVIF thật, YOLO11n/s thật, F1 có FP, CPU, airtime, tiêu chí sàng lọc C1–C5 | phương án B (11 cặp EXPL) theo #15; chạy ảnh thật chờ cơ sở quyền dữ liệu |
| 15 | [swarmbridge-h1-0b-pilot-readiness.md](swarmbridge-h1-0b-pilot-readiness.md) | Sẵn sàng pilot 0B: adapter MDMT (chốt split / quyền / GT), lõi pilot, precompute model thật, đánh giá C1–C5, 57 test, smoke test model thật; sửa lỗi căn chỉnh số file | v3 duyệt (#11); đã chạy pilot, xem #16 |
| 16 | [swarmbridge-h1-0b-pilot-expl-result.md](swarmbridge-h1-0b-pilot-expl-result.md) | Pilot 0B trên 11 chuỗi MDMT EXPL với YOLO11n / AVIF / YOLO11s thật: C1–C5, T1 so với T0 / T0+, F, T2 | PASS khám phá, hiệu ứng yếu (+1–3 điểm, chủ yếu chuỗi 59); khung +8 điểm nhưng vượt CPU. Mục 6: cổng 'crop khi detector thiếu chắc chắn' dự đoán được (LOSO) nhưng thưởng nhỏ → chưa train PPO cho TOKEN/CROP. Quyền dữ liệu chưa xác minh |
| 17 | [swarmbridge-h1-0b-pilot-cluster-ci.md](swarmbridge-h1-0b-pilot-cluster-ci.md) | Độ tin cậy pilot 0B: bootstrap theo 11 chuỗi + bỏ từng chuỗi (từ JSON đã lưu) | Crop: CI chỉ loại 0 ở B 305k, 70–99% lợi ích từ chuỗi 59, bỏ 59 thì mọi CI chứa 0; T2 (GT) +0,3–0,7 ổn định; khung +8,3 [5,3; 11,5] |
| 18 | [swarmbridge-a7s-semantic-family-design.md](swarmbridge-a7s-semantic-family-design.md) | A7-S0: họ TOKEN / CROP / FRAME trong môi trường A5, thiết kế + cổng dư địa đăng ký trước (không train) | đã chạy, xem #19 |
| 19 | [swarmbridge-a7s-semantic-family-result.md](swarmbridge-a7s-semantic-family-result.md) | A7-S0 kết quả: FIX / AGN / CAU / ORA ở ref, K5, phy_mcs_blind, hai fold theo chuỗi | G-S1 đạt: cận trên thông tin +6–11% return, +8–10 recall ở ref / K5; dư địa là theo nội dung, không theo kênh; heuristic nhân quả không ổn định; PPO chưa chạy, chờ duyệt |
| 20 | [swarmbridge-a7s1-temporal-diagnostic.md](swarmbridge-a7s1-temporal-diagnostic.md) | A7-S1: giá trị khung t−1 / khung tráo cùng cảnh thay cho giá trị thật; tách bit thật và giá trị thật | Dư địa 91–99% từ giá trị thật; t−1 giữ 48–83%, khung tráo 30–80% → hiểu biết mức cảnh; NO-GO PPO mode-only trong A5 hiện tại |
| 21 | [swarmbridge-a7s2-bs-feedback-proxy.md](swarmbridge-a7s2-bs-feedback-proxy.md) | A7-S2: proxy không nhãn ở BS (detection mới của FRAME so với token) + heuristic thăm dò–khai thác nhân quả | Proxy xếp hạng cảnh tốt (0,80 / 0,96) nhưng dương giả ở cảnh đông; EE giữ 8% / âm / 71% / 18% dư địa → NO-GO cho cơ chế này; giá trị FRAME là đặc tính của góc nhìn |
| 22 | [swarmbridge-research-decision-review-2026-11-13.md](swarmbridge-research-decision-review-2026-11-13.md) | Research Decision Review: bằng chứng mạnh / yếu từ A5 đến A7-S2, A5 có hợp tác không, hai hướng tiếp theo kèm phép kiểm tra GO / NO-GO | Chờ quyết định; không thí nghiệm mới |
| 23 | [swarmbridge-a8-b1-preregistration.md](swarmbridge-a8-b1-preregistration.md) | Đăng ký trước A8 (thang sai lệch mô hình, PPO đóng băng so với H\*) và B1 (cơ hội hợp tác theo góc nhìn; kiểm ID xuyên góc trước) | đã khoá trước khi chạy |
| 24 | [swarmbridge-a8-b1-result-decision.md](swarmbridge-a8-b1-result-decision.md) | Kết quả A8 và B1 + báo cáo quyết định | A8 **không hợp lệ** (thang không tác động: không có nhiễu đồng kênh với các chính sách được so; Δ = +5,73 ở cả 5 mức); B1: ID xuyên góc đạt 8/11 chuỗi, tầng 1 chưa chạy (chờ cơ sở dùng dữ liệu) |
| 25 | [swarmbridge-b-feasibility-datasets.md](swarmbridge-b-feasibility-datasets.md) | Sau phản biện #19: sửa so sánh cùng ngân sách bit của B1 (Sửa đổi 1), khảo sát M3OT và CrossTrack-UAV, tính khả thi của hướng B theo từng bộ dữ liệu | B1 sẵn sàng về code; MDMT chờ quyền; M3OT giấy phép rõ nhưng chỉ 3/9 chuỗi qua sàng lọc ID, không thay thế trực tiếp được; CrossTrack-UAV cần ký thỏa thuận |

- Ghi chú nền của R0: [r0-notes/](r0-notes/).
- Hình: [figures/](figures/).
- Kết quả số: `runs/uav_semcom/`.
- Code: `swarmbridge/experiments/uav_semcom/`.

## Không thuộc nhánh này

- [wireless-abc.md](wireless-abc.md): tổng hợp hướng A/B/C ngày 06/11. Để nguyên ở đây vì `runs/active_recovery/protocol-v0.1.json` (nhánh Active Semantic Recovery do Codex làm) tham chiếu đúng đường dẫn này.

## Lưu trữ

| Thư mục | Giai đoạn |
|---|---|
| [archive/uav-semcom-a/](archive/README.md#uav-semcom-a) | UAV-SemCom A, A5, A5.5, A6-L1 cùng các kiểm toán, trước G-L2 |
| [archive/huang-abc/](archive/README.md#huang-abc) | Huang JSAC bước 0, kế hoạch proposal A/B/C |
| [archive/uav3d/](archive/README.md#uav3d) | UAV3D (đóng băng từ 04/11) |
| [archive/sb-town3/](archive/README.md#sb-town3) | SB-decoder / Town3 |
| [archive/tai-lieu/](archive/README.md#tai-lieu) | Paper, repo, prior art |
| [metadata/](metadata/README.md) | JSON kiểm toán, manifest; bảng chuyển đường dẫn [04/11](metadata/report-moves-2026-11-04.json) và [10/11](metadata/report-moves-2026-11-10.json) |

Báo cáo trong archive có thể chứa kết luận đã bị sửa ở báo cáo sau. Khi viết báo cáo mới, ưu tiên cập nhật tài liệu hiện hành thay vì tạo thêm một file review cho mỗi lượt trao đổi.
