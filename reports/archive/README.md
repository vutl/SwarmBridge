# Báo cáo gốc lưu trữ

Các file dưới đây giữ lịch sử thí nghiệm, đề xuất và review. Một số kết luận đã được sửa ở báo cáo sau; bắt đầu từ [trang tổng quan](../README.md) và các bản tổng hợp hiện hành. Lần sắp xếp chỉ cập nhật đường dẫn tham chiếu trong Markdown.

<a id="phases-2026-11"></a>

<a id="r1-visdrone"></a>

## R0 → R1b trên VisDrone (10–11/11)

Cổng G-L2, kịch bản R0, R1a, thiết kế và pilot R1b-0a; dừng vì H1 FAIL (không có dư địa).

| # cũ | Tài liệu | Nội dung | Trạng thái |
|---|---|---|---|
| 1 | [uav-semcom-g-l2.md](r1-visdrone/uav-semcom-g-l2.md) | Cổng G-L2: dự đoán lợi ích theo từng ảnh | **FAIL** (mục 8: vẫn FAIL sau khi sửa baseline) |
| 2 | [novelty-audit-g-l2-2026-11-10.md](r1-visdrone/novelty-audit-g-l2-2026-11-10.md) | Kiểm tra novelty sau G-L2 | xong |
| 3 | [swarmbridge-next-scenario-r0.md](r1-visdrone/swarmbridge-next-scenario-r0.md) | R0: chọn kịch bản tiếp theo (chọn C) | xong |
| 4 | [swarmbridge-r1a.md](r1-visdrone/swarmbridge-r1a.md) | R1a: token / hybrid / ROI so với ảnh, theo ngân sách bit | **INCONCLUSIVE** (đã đăng ký trước); mục 7 là khám phá |
| 5 | [swarmbridge-r1b-design.md](r1-visdrone/swarmbridge-r1b-design.md) | Thiết kế R1b v3: ablation thông tin phản hồi, các cổng H1 / H2 / H3, pilot R1b-0a | đã duyệt; H1 đã chấm (xem #9) |
| 6 | [visdrone-vid-data-audit.md](r1-visdrone/visdrone-vid-data-audit.md) | Kiểm toán VisDrone-VID: rò rỉ DET, cổng G7 theo nguồn (split v1) | xong |
| 7 | [swarmbridge-r1b0a-protocol.md](r1-visdrone/swarmbridge-r1b0a-protocol.md) | Protocol R1b-0a: split, timing nhân quả, số khoá, lưới tinh chỉnh, cổng H1 | đã khoá; G7 chấm một lần 11/11 |
| 8 | [swarmbridge-r1b0a-validation.md](r1-visdrone/swarmbridge-r1b0a-validation.md) | Kiểm chứng kỹ thuật bộ mô phỏng: chỉ số theo lớp, wire format, kênh ghép cặp, timing; 52 test | xong (dữ liệu tổng hợp) |
| 9 | [swarmbridge-r1b0a-h1-result.md](r1-visdrone/swarmbridge-r1b0a-h1-result.md) | Kết quả cổng H1 trên G7: bảng A1 / B / B+ / C1 / D, độ nhạy, chẩn đoán | **FAIL (không có dư địa)**; dừng nhánh phản hồi ngữ nghĩa |

Ghi chú nền R0: [r1-visdrone/r0-notes/](r1-visdrone/r0-notes/).

<a id="h1-mdmt"></a>

## H1 trên MDMT: Gate 0 và pilot 0B (11–13/11)

Đánh giá lại hướng, báo cáo hợp tác theo vật thể duy nhất, Gate 0A, pilot crop 0B; lợi ích crop nhỏ, không ổn định.

| # cũ | Tài liệu | Nội dung | Trạng thái |
|---|---|---|---|
| 10 | [swarmbridge-direction-reassessment.md](h1-mdmt/swarmbridge-direction-reassessment.md) | Đánh giá lại hướng sau R1b: postmortem, prior art, 3 hướng, novelty / dataset matrix, cổng 0 trên MDMT. Ghi chú nguồn: [reassess-notes/](h1-mdmt/reassess-notes/) | H1 duyệt ở mức ứng viên; Gate 0 thay bằng #11 |
| 11 | [swarmbridge-h1-gate0-protocol-v2.md](h1-mdmt/swarmbridge-h1-gate0-protocol-v2.md) | Protocol Gate 0 v2 cho H1: phân loại oracle, đặc tả bộ chấm F1 đối tượng duy nhất, ma trận baseline, split chống thiên lệch chọn, quyết định quyền dữ liệu, checklist | v2.1 khoá cho 0A; 0B chưa duyệt |
| 12 | [swarmbridge-h1-gate0a-result.md](h1-mdmt/swarmbridge-h1-gate0a-result.md) | Gate 0A (khám phá, nhãn MDMT-test, báo cáo hoàn hảo): dư địa phối hợp theo ngân sách, TOK và CROP riêng | TOK: dư địa hẹp (≈ dòng token); CROP: dư địa ở 100–500 kbit/epoch |
| 13 | [swarmbridge-h1-gate0a-audit.md](h1-mdmt/swarmbridge-h1-gate0a-audit.md) | Kiểm toán 0A: S2 là oracle có trễ; oracle chính xác (MILP); chi phí bản kê; baseline nhớ lịch sử; điều kiện C1–C6 cho CROP | TOK NO-GO; CROP GO có điều kiện; v2 sửa chi phí bản kê (mục 8) |
| 14 | [swarmbridge-h1-0b-tune-pilot-protocol.md](h1-mdmt/swarmbridge-h1-0b-tune-pilot-protocol.md) | Protocol pilot 0B-TUNE cho CROP: ứng viên không GT, AVIF thật, YOLO11n/s thật, F1 có FP, CPU, airtime, tiêu chí sàng lọc C1–C5 | phương án B (11 cặp EXPL) theo #15; chạy ảnh thật chờ cơ sở quyền dữ liệu |
| 15 | [swarmbridge-h1-0b-pilot-readiness.md](h1-mdmt/swarmbridge-h1-0b-pilot-readiness.md) | Sẵn sàng pilot 0B: adapter MDMT (chốt split / quyền / GT), lõi pilot, precompute model thật, đánh giá C1–C5, 57 test, smoke test model thật; sửa lỗi căn chỉnh số file | v3 duyệt (#11); đã chạy pilot, xem #16 |
| 16 | [swarmbridge-h1-0b-pilot-expl-result.md](h1-mdmt/swarmbridge-h1-0b-pilot-expl-result.md) | Pilot 0B trên 11 chuỗi MDMT EXPL với YOLO11n / AVIF / YOLO11s thật: C1–C5, T1 so với T0 / T0+, F, T2 | PASS khám phá, hiệu ứng yếu (+1–3 điểm, chủ yếu chuỗi 59); khung +8 điểm nhưng vượt CPU. Mục 6: cổng 'crop khi detector thiếu chắc chắn' dự đoán được (LOSO) nhưng thưởng nhỏ → chưa train PPO cho TOKEN/CROP. Quyền dữ liệu chưa xác minh |
| 17 | [swarmbridge-h1-0b-pilot-cluster-ci.md](h1-mdmt/swarmbridge-h1-0b-pilot-cluster-ci.md) | Độ tin cậy pilot 0B: bootstrap theo 11 chuỗi + bỏ từng chuỗi (từ JSON đã lưu) | Crop: CI chỉ loại 0 ở B 305k, 70–99% lợi ích từ chuỗi 59, bỏ 59 thì mọi CI chứa 0; T2 (GT) +0,3–0,7 ổn định; khung +8,3 [5,3; 11,5] |

Ghi chú đánh giá lại hướng: [h1-mdmt/reassess-notes/](h1-mdmt/reassess-notes/).

<a id="a7-a8-b1"></a>

## A7 / A8 / B1 (13–14/11)

Họ TOKEN / CROP / FRAME trong môi trường A5, Research Decision Review, thang sai lệch A8 (không hợp lệ), chọn góc nhìn B1 (NO-GO, nhánh đóng).

| # cũ | Tài liệu | Nội dung | Trạng thái |
|---|---|---|---|
| 18 | [swarmbridge-a7s-semantic-family-design.md](a7-a8-b1/swarmbridge-a7s-semantic-family-design.md) | A7-S0: họ TOKEN / CROP / FRAME trong môi trường A5, thiết kế + cổng dư địa đăng ký trước (không train) | đã chạy, xem #19 |
| 19 | [swarmbridge-a7s-semantic-family-result.md](a7-a8-b1/swarmbridge-a7s-semantic-family-result.md) | A7-S0 kết quả: FIX / AGN / CAU / ORA ở ref, K5, phy_mcs_blind, hai fold theo chuỗi | G-S1 đạt: cận trên thông tin +6–11% return, +8–10 recall ở ref / K5; dư địa là theo nội dung, không theo kênh; heuristic nhân quả không ổn định; PPO chưa chạy, chờ duyệt |
| 20 | [swarmbridge-a7s1-temporal-diagnostic.md](a7-a8-b1/swarmbridge-a7s1-temporal-diagnostic.md) | A7-S1: giá trị khung t−1 / khung tráo cùng cảnh thay cho giá trị thật; tách bit thật và giá trị thật | Dư địa 91–99% từ giá trị thật; t−1 giữ 48–83%, khung tráo 30–80% → hiểu biết mức cảnh; NO-GO PPO mode-only trong A5 hiện tại |
| 21 | [swarmbridge-a7s2-bs-feedback-proxy.md](a7-a8-b1/swarmbridge-a7s2-bs-feedback-proxy.md) | A7-S2: proxy không nhãn ở BS (detection mới của FRAME so với token) + heuristic thăm dò–khai thác nhân quả | Proxy xếp hạng cảnh tốt (0,80 / 0,96) nhưng dương giả ở cảnh đông; EE giữ 8% / âm / 71% / 18% dư địa → NO-GO cho cơ chế này; giá trị FRAME là đặc tính của góc nhìn |
| 22 | [swarmbridge-research-decision-review-2026-11-13.md](a7-a8-b1/swarmbridge-research-decision-review-2026-11-13.md) | Research Decision Review: bằng chứng mạnh / yếu từ A5 đến A7-S2, A5 có hợp tác không, hai hướng tiếp theo kèm phép kiểm tra GO / NO-GO | Chờ quyết định; không thí nghiệm mới |
| 23 | [swarmbridge-a8-b1-preregistration.md](a7-a8-b1/swarmbridge-a8-b1-preregistration.md) | Đăng ký trước A8 (thang sai lệch mô hình, PPO đóng băng so với H\*) và B1 (cơ hội hợp tác theo góc nhìn; kiểm ID xuyên góc trước) | đã khoá trước khi chạy |
| 24 | [swarmbridge-a8-b1-result-decision.md](a7-a8-b1/swarmbridge-a8-b1-result-decision.md) | Kết quả A8 và B1 + báo cáo quyết định | A8 **không hợp lệ** (thang không tác động: không có nhiễu đồng kênh với các chính sách được so; Δ = +5,73 ở cả 5 mức); B1: ID xuyên góc đạt 8/11 chuỗi, tầng 1 chưa chạy (chờ cơ sở dùng dữ liệu) |
| 25 | [swarmbridge-b-feasibility-datasets.md](a7-a8-b1/swarmbridge-b-feasibility-datasets.md) | Sau phản biện #19: sửa so sánh cùng ngân sách bit của B1 (Sửa đổi 1), khảo sát M3OT và CrossTrack-UAV, tính khả thi của hướng B theo từng bộ dữ liệu | B1 sẵn sàng về code; MDMT chờ quyền; M3OT giấy phép rõ nhưng chỉ 3/9 chuỗi qua sàng lọc ID, không thay thế trực tiếp được; CrossTrack-UAV cần ký thỏa thuận |
| 26 | [swarmbridge-b1-tier1-result.md](a7-a8-b1/swarmbridge-b1-tier1-result.md) | B1 tầng 1 trên 8 chuỗi MDMT (sàng lọc có trợ giúp GT, cùng giới hạn bit): `VIEW` so với `TT` / `ALT` / `NDET` / `BIG` | **NO-GO cho `VIEW`**: quy tắc không nhãn `NDET` chọn cùng góc ở 8/8 chuỗi (hiệu 0,0); phân bổ theo góc vẫn hơn luân phiên +2,5 điểm F1 (7/8 chuỗi); góc tốt là UAV 2 ở 7/8 chuỗi |

<a id="uav-semcom-a"></a>

## UAV-SemCom giai đoạn A (07–10/11, trước G-L2)

- [Hướng A: UAV image delivery, hybrid bit / semantic modes, DRL](uav-semcom-a/uav-semcom-a.md)
- [A5: mở rộng dữ liệu, task fidelity, khảo sát chế độ vận hành, RL có mục tiêu](uav-semcom-a/uav-semcom-a5.md)
- [A5.5: kiểm toán độ bền của kết luận A5](uav-semcom-a/uav-semcom-a5.5.md)
- [A6.0-L1: chọn mode theo giá trị tác vụ](uav-semcom-a/uav-semcom-a6-l1.md) và [kiểm toán phương pháp A6.0-L1](uav-semcom-a/uav-semcom-a6-l1-audit.md)
- [Kiểm toán prior-art / độ mới cho A5, A5.5, A6.0-L1](uav-semcom-a/novelty-audit-a5-l1-2026-11-10.md)

<a id="huang-abc"></a>

## Huang JSAC và proposal A/B/C (04–06/11)

- [Bước 0 theo Huang (JSAC 2025): chọn mode có cần RL không?](huang-abc/huang-step0.md)
- [Kế hoạch proposal A/B/C](huang-abc/ke-hoach-proposal-abc-2026-11-04.md)

<a id="sb-town3"></a>

## SB / Town3

- [Tổng hợp SB và Town3](sb-town3/sb-va-town3.md)

- [SwarmBridge — Báo cáo tiến độ (tính đến 30/10/2026)](sb-town3/bao-cao-tien-do-2026-10-30.md)
- [Review đề xuất chọn block sau temporal baseline](sb-town3/review-block-selection-proposal-2026-11-03.md)
- [Đọc và đối chiếu Chat GPT Chat.pdf](sb-town3/review-chat-gpt-chat-2026-11-03.md)
- [Review công việc Claude — 02/11/2026](sb-town3/review-claude-2026-11-02.md)
- [sb-audit-protocol-2026-10-30](sb-town3/sb-audit-protocol-2026-10-30.md)
- [sb-deep-research-2026-10-30](sb-town3/sb-deep-research-2026-10-30.md)
- [Hướng mới cho SB trong SwarmBridge — tổng hợp từ thư viện paper (02/11/2026)](sb-town3/sb-huong-moi-2026-11-02.md)
- [SB có ưu thế toán học gì, và điều đó dẫn tới ứng dụng nào?](sb-town3/sb-mathematical-foundations-2026-11-02.md)
- [sb-next-decision-2026-10-31](sb-town3/sb-next-decision-2026-10-31.md)
- [sb-paper-reproductions-2026-10-31](sb-town3/sb-paper-reproductions-2026-10-31.md)
- [Pilot SB với dữ liệu train khớp lúc đánh giá — kết quả (02/11/2026)](sb-town3/sb-pilot-matched-2026-11-02.md)
- [sb-training-root-audit-2026-10-31](sb-town3/sb-training-root-audit-2026-10-31.md)
- [Bước 1 — củng cố kết quả chọn block (03/11/2026)](sb-town3/step1-hardening-2026-11-03.md)
- [swarmbridge-research-direction-2026-10-30](sb-town3/swarmbridge-research-direction-2026-10-30.md)
- [Chọn block trên receiver temporal — pilot 03/11/2026](sb-town3/temporal-block-selection-2026-11-03.md)

<a id="uav3d"></a>

## UAV3D (đóng băng 04/11)

- [Tổng hợp UAV3D và kiểm toán](uav3d/uav3d-ket-qua-va-kiem-toan.md)
- [Ablation, ngân sách và hàng đợi v1.1](uav3d/uav3d-proto-v1.1-budget-queue-2026-11-04.md)
- [Protocol UAV3D v1](uav3d/uav3d-protocol-v1-2026-11-04.md)

- [Review kết quả protocol box và nhận xét ChatGPT — 04/11/2026](uav3d/codex-review-proto-box-results-2026-11-04.md)
- [Review độc lập protocol UAV3D v1 — 04/11/2026](uav3d/codex-review-uav3d-protocol-v1-2026-11-04.md)
- [Kiểm toán độc lập UAV3D mini — 04/11/2026](uav3d/codex-uav3d-independent-audit-2026-11-04.md)
- [Rà soát kế hoạch UAV3D mini — 04/11/2026](uav3d/review-uav3d-plan-2026-11-04.md)
- [UAV3D: gửi box hay gửi feature? Kết quả chạy thật (2026-11-04)](uav3d/uav3d-comm-results-2026-11-04.md)
- [Kiểm tra UAV3D v1.0-mini (2026-11-04)](uav3d/uav3d-mini-audit-2026-11-04.md)
- [UAV3D protocol v1 — kết quả gộp box (dev, CPU) — 2026-11-04](uav3d/uav3d-proto-box-results-2026-11-04.md)
- [Plan chạy UAV3D (2026-11-04)](uav3d/uav3d-run-plan-2026-11-04.md)
- [UAV3D mini: pilot tách feature của ego và đồng đội](uav3d/uav3d-separated-source-pilot-2026-11-04.md)

<a id="tai-lieu"></a>

## Tài liệu và nguồn tải

- [Tài liệu và mã nguồn tham khảo (tổng hợp)](tai-lieu/tai-lieu-va-ma-nguon.md)

- [Đọc CERF (paper + code) và CoSDH, 2026-11-04](tai-lieu/cerf-review-2026-11-04.md)
- [chatgpt-gop-y-cerf-2026-11-04](tai-lieu/chatgpt-gop-y-cerf-2026-11-04.md)
- [Repo bổ sung ngày 04/11/2026](tai-lieu/github-swarm-downloads-2026-11-04.md)
- [literature-evidence-map-2026-10-30](tai-lieu/literature-evidence-map-2026-10-30.md)
- [Nguồn của “Tóm tắt điều hành.pdf”](tai-lieu/nguon-tom-tat-dieu-hanh-2026-11-03.md)
- [Kiểm tra prior art: "Arrival-Aware Cooperative Semantic Communication for Multi-UAV Tracking" (v0), 2026-11-04](tai-lieu/prior-art-arrival-aware-2026-11-04.md)
- [Tài liệu bổ sung từ ba phần trao đổi ChatGPT](tai-lieu/tai-them-tai-lieu-swarm-2026-11-04.md)
- [Tình trạng tải tài liệu ngày 30/10/2026](tai-lieu/tinh-trang-tai-tai-lieu-2026-10-30.md)
