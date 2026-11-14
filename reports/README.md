# SwarmBridge: báo cáo

Cập nhật 14/11/2026. Ngoài cùng chỉ giữ tài liệu của **hướng đang mở**. Các nhánh đã đóng nằm trong [archive/](archive/README.md), chia theo giai đoạn.

## Trạng thái nghiên cứu

| Thành phần | Kết luận | Tài liệu |
|---|---|---|
| A5 / A5.5 PPO mode-only | Có lợi ích thực nghiệm ở một số cấu hình; cơ chế chưa giải thích được; không phải bằng chứng hợp tác | [archive/uav-semcom-a/](archive/README.md#uav-semcom-a) |
| Pilot 0B (CROP trên MDMT) | Lợi ích nhỏ, không ổn định | [archive/h1-mdmt/](archive/README.md#h1-mdmt) |
| A7 lập lịch theo ngữ nghĩa | Có dư địa lý tưởng; proxy thực tế chưa khai thác ổn định | [archive/a7-a8-b1/](archive/README.md#a7-a8-b1) |
| A8 thang sai lệch mô hình | Không hợp lệ: thang không tác động lên các chính sách được so | [archive/a7-a8-b1/](archive/README.md#a7-a8-b1) |
| B1 chọn góc nhìn | NO-GO: quy tắc không nhãn bắt kịp lựa chọn dùng GT; nhánh đóng | [archive/a7-a8-b1/](archive/README.md#a7-a8-b1) |
| Rà soát novelty | PPO / MAPPO chọn TOKEN / CROP / FRAME: NO-GO (đã có người làm); RQ2: NO-GO về thuật toán; MARL chưa có lý do | [novelty review](swarmbridge-collab-perception-novelty-review-2026-11-13.md) |
| **RQ1** (liên kết xuyên góc làm lý do truyền) | Phase 0 (chỉ hình học, bản ghi cache): NOT IDENTIFIABLE. RQ1-E E0 (DINOv2-S zero-shot, ảnh thật): **NO-GO**. Dừng trên MDMT; còn mở nếu có metadata camera hoặc Re-ID được học (cần duyệt mới) | [RQ1 protocol](swarmbridge-rq1-association-preregistration.md), [Phase 0](swarmbridge-rq1-phase0-result.md), [RQ1-E E0](swarmbridge-rq1e-e0-result.md) |
| **Hướng đề xuất mới** | Truyền thông ngữ nghĩa có dự đoán trên Town3 (JEPA chỉ là một ứng viên ở tầng trạng thái chung) | [proposal](swarmbridge-jepa-semcom-proposal.md) |
| RL hợp tác bầy đàn | **Chưa có bằng chứng thực nghiệm** | |

## Hướng đang mở

| Tài liệu | Nội dung |
|---|---|
| [swarmbridge-collab-perception-novelty-review-2026-11-13.md](swarmbridge-collab-perception-novelty-review-2026-11-13.md) | Rà soát novelty: CCSRA, HyComm, SchedCP, Where2comm, CodeFilling, CoSDH, PIB, MVSparse, MASS...; ma trận so sánh; RQ1 / RQ2 |
| [swarmbridge-rq1-association-preregistration.md](swarmbridge-rq1-association-preregistration.md) | Protocol v2 của RQ1 (RQ1-A / RQ1-B, bộ liên kết không GT, Phase 0); v1 rút trước khi chạy |
| [swarmbridge-rq1-phase0-result.md](swarmbridge-rq1-phase0-result.md) | Phase 0: NOT IDENTIFIABLE FROM EXISTING RECORDS; vì sao, và điều gì làm RQ1 đo được |
| [swarmbridge-rq1-reid-feasibility-memo.md](swarmbridge-rq1-reid-feasibility-memo.md) | Memo khả thi: RQ1 bằng ngoại hình (DINOv2 zero-shot) trên ảnh MDMT thật, mô hình truyền TOKEN / embedding / CROP, đề xuất thí nghiệm RQ1-E (chưa duyệt chạy) |
| [swarmbridge-rq1e-preregistration.md](swarmbridge-rq1e-preregistration.md) | Protocol RQ1-E (DINOv2-S zero-shot, E0 khoá trước, cam kết cho E1) |
| [swarmbridge-rq1e-e0-result.md](swarmbridge-rq1e-e0-result.md) | RQ1-E E0: **FAIL 0/8 → NO-GO** cho ngoại hình zero-shot trên MDMT; E1 không chạy; đo tốc độ thật; chẩn đoán vì sao |
| [swarmbridge-jepa-semcom-proposal.md](swarmbridge-jepa-semcom-proposal.md) | **Proposal mới:** truyền thông ngữ nghĩa có dự đoán, đồng bộ trạng thái bên nhận, trên Town3; tầng 1 mã hoá dự đoán output nhiệm vụ, tầng 2 trạng thái latent đa nhiệm vụ (JEPA là ứng viên, thiết kế 2 × 2 backbone × mục tiêu); chờ duyệt |

## Lưu trữ

| Thư mục | Giai đoạn |
|---|---|
| [archive/a7-a8-b1/](archive/README.md#a7-a8-b1) | A7 (TOKEN / CROP / FRAME trong A5), Research Decision Review, A8, B1 (13–14/11) |
| [archive/h1-mdmt/](archive/README.md#h1-mdmt) | Đánh giá lại hướng, Gate 0A, pilot 0B trên MDMT (11–13/11) |
| [archive/r1-visdrone/](archive/README.md#r1-visdrone) | G-L2, R0, R1a, R1b-0a trên VisDrone (10–11/11) |
| [archive/uav-semcom-a/](archive/README.md#uav-semcom-a) | UAV-SemCom A, A5, A5.5, A6-L1 cùng các kiểm toán (07–10/11) |
| [archive/huang-abc/](archive/README.md#huang-abc) | Huang JSAC bước 0, kế hoạch proposal A/B/C (04–06/11) |
| [archive/uav3d/](archive/README.md#uav3d) | UAV3D (đóng băng từ 04/11) |
| [archive/sb-town3/](archive/README.md#sb-town3) | SB-decoder / Town3 |
| [archive/tai-lieu/](archive/README.md#tai-lieu) | Paper, repo, prior art |
| [metadata/](metadata/README.md) | JSON kiểm toán, manifest; bảng chuyển đường dẫn [04/11](metadata/report-moves-2026-11-04.json), [10/11](metadata/report-moves-2026-11-10.json), [14/11](metadata/report-moves-2026-11-14.json) |

Báo cáo trong archive có thể chứa kết luận đã bị sửa ở báo cáo sau. Kết quả số: `runs/uav_semcom/`. Code: `swarmbridge/experiments/uav_semcom/`. Hình: [figures/](figures/).

## Không thuộc nhánh này

- [wireless-abc.md](wireless-abc.md): tổng hợp hướng A/B/C ngày 06/11. Để nguyên ở đây vì `runs/active_recovery/protocol-v0.1.json` (nhánh Active Semantic Recovery do Codex làm) tham chiếu đúng đường dẫn này.
