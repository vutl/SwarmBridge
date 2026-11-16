# SwarmBridge: báo cáo

Cập nhật 16/11/2026. Ngoài cùng chỉ giữ tài liệu của **hướng đang mở**. Các nhánh đã đóng nằm trong [archive/](archive/README.md), chia theo giai đoạn.

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
| **Songdo bước 0–1** (phản biện #25) | Codec trạng thái thật: khoảng 0,7 kbit/s / giao lộ, tải cả đội max 11,2 kbit/s so với ≥ 60 kbit/s của riêng kênh C&C → **truyền thông không bị giới hạn**; lập lịch Songdo S3, S2 (giảm bit) và JEPA trên Songdo: **NO-GO**; S4 Town3 không xin duyệt | [proposal v2.1](swarmbridge-jepa-semcom-proposal.md), [S1 feasibility](swarmbridge-songdo-s1-feasibility.md) |
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
| [swarmbridge-ae8-be7-result.md](swarmbridge-ae8-be7-result.md) | **A-E8 + B-E7 (phản biện #33):** bộ điều khiển chọn lọc với ngân sách 50 ms chủ động (tính cả PPO): giữ hạn ở 99,97–100 % quyết định, p99 40–44 ms, return như bản không ngân sách; đánh giá gộp các mức tốc độ trong một lượt; ứng viên bi quan giúp khi mô hình lệch (Shannon/MCS +1,90 so với +0,71, hiệu +1,19 CI > 0), không giúp ở ref; planner đầy đủ dạng anytime bị cắt 2/3 số quyết định trong 50 ms. B: HEVC/AV1 liên khung nhân quả → DINOv2 **thắng** rateonly03 ở 6–12 kbit/frame (0,02–0,09 mAP) và trên patch thay đổi ở mọi mức bit; chỉ ngang ở ~18 kbit → Paper B ở dạng hiện tại không đứng được; surrogate xếp hạng model đúng (0,98) nhưng theo frame chỉ 0,69–0,80 |
| [swarmbridge-ae9-be8-result.md](swarmbridge-ae9-be8-result.md) | **A-E9 + B-E8 (phản biện #34):** seed mới 8500–8519, CI theo seed môi trường: np-any +2,18 (phy) / +1,91 (ref) so với PPO; feedback trễ 1–3 slot, mất 30–60 % không làm giảm đáng kể (mọi CI chứa 0), sự bền này đến từ warm-start (cold chỉ +0,65 / +0,96); Shannon np-pess hơn planner đầy đủ (+0,90) nhưng hơn top-2 chưa rõ (+0,39). B: offset QP theo block trong HEVC (adapter libx265): ROI theo DINOv2 phía gửi +0,01 AP (~11–13 % bit) ở 4,5–6 kbit, ap_change không tăng; số hạng sự kiện gần như không kích hoạt (7/10 clip trùng bitstream); ROI theo GT không hơn; motion-only gây hại |
| [swarmbridge-ae11-ax1-result.md](swarmbridge-ae11-ax1-result.md) | **A-E11 + A-X1 (phản biện #36):** A: cửa sổ trượt 30 quan sát/dải SNR sửa lỗi A-E10 (tệ nhất −0,14 so với PPO trên 9 kịch bản, seed mới 8600–8619; warm-start đóng băng −7,5/−13,0); warm-start khớp chỉ hơn cold +0,4/+0,9 → lợi ích chủ yếu từ hiệu chỉnh online. X1: họ 16 heuristic chọn kênh không hơn T:best; phần chọn kênh của PPO joint +2,07 ở ref, ở K5 chỉ có ích khi đi cùng phần mode học chung (+1,78) → có tương tác kênh × mode; đề xuất A-X2 |
| [swarmbridge-be9-ae10-result.md](swarmbridge-be9-ae10-result.md) | **B-E9 + A-E10 (phản biện #35):** B: bản đồ sự kiện theo từng lớp + hạn mức event riêng trong HEVC: cơ chế giờ thực sự hoạt động (38 % block ưu tiên); ap_change +0,009 so với task ở 6 kbit (CI > 0, ~15 % bit), ngang bản dùng nhãn GT (+0,008) nhưng mất từ 9 kbit và ngược dấu trên dev → trần của ROI-QP khoảng 0,01, đề xuất chuyển hướng (a). A: warm-start hiệu chỉnh ở regime PHY khác làm return −6,4 (phy) / −12,1 (ref) so với PPO, tệ hơn cả cold; tracker không phát hiện → warm-start phải đúng cấu hình PHY. Method Paper A v2 |
| [swarmbridge-paperA-method-draft.md](swarmbridge-paperA-method-draft.md) | Bản nháp mục Method của Paper A (tiếng Anh): bộ điều khiển PPO + MPC chọn lọc, tham số, pseudo-code, bằng chứng A-E5…A-E9, những gì không khẳng định |
| [swarmbridge-ae7-be6-result.md](swarmbridge-ae7-be6-result.md) | **A-E7 + B-E6 (phản biện #32):** kiểm tra đầu mút có đảo thứ tự bên trong (12 % lần đổi ở phy_mcs_blind); lưới 5 điểm giảm còn 4 % nhưng không tăng return. Planner hai tầng (sàng lọc tầm nhìn ngắn → MPC 3 mode) giữ +2,39 / +2,22 so với PPO, với 25 / 20,5 ms thay vì 50 / 48 ms. B: bottleneck với surrogate số bit exp-Golomb (không làm mượt MSE) hơn task +0,030–0,036 mAP ở 9–18 kbit, và hơn smooth trên patch có vật xuất hiện / biến mất; surrogate tương quan 0,99999 với bit thật |
| [swarmbridge-ae6-be5-result.md](swarmbridge-ae6-be5-result.md) | **A-E6 + B-E5 (phản biện #31):** cổng quyết-định-nhận-thức (truyền sai số tốc độ vào hiệu giá trị PPO và planner), warm-start từ episode hiệu chỉnh riêng: +2,82 (phy_mcs_blind) / +2,23 (ref) so với PPO, CI > 0; với mô hình không lệch phần lớn lợi ích giống tăng ngưỡng Z (bảo thủ hơn), phần riêng là chống bias (Shannon lệch 47 %: +1,30 thay vì −8,67 / = PPO); tỉ lệ lần đổi đúng 38 → 79 %. B: smooth-only ≥ JEPA ở cùng bit (JEPA − smooth ≈ −0,004), smooth − task +0,016–0,019 đều 3 seed → lợi ích của JEPA là do ổn định theo thời gian |
| [swarmbridge-ae5-be4-result.md](swarmbridge-ae5-be4-result.md) | **A-E5 + B-E4 (phản biện #30):** log throughput theo thời gian phát bị lệch thấp (−12 % với mô hình đúng), sửa bằng hiệu chỉnh theo slot; cổng sai số chặn mô hình lệch (Shannon: 16,39 → bằng PPO); bộ hiệu chỉnh theo slot + cổng sai số +1,69 (phy_mcs_blind) / +1,86 (ref) so với PPO, CI > 0, không cần biết PHY; cổng quá bảo thủ với mô hình đúng, độ nhạy τ không đơn điệu. B: 3 seed ghép cặp, JEPA-32 − task-32 = +0,025 / +0,013 / +0,005 (TB +0,015, CI chứa 0); mã JEPA ổn định hơn ở cả 3 seed |
| [swarmbridge-ae4-be3-result.md](swarmbridge-ae4-be3-result.md) | **A-E4 + B-E3 (phản biện #29):** rollout phi tiên tri xoá phần lớn ưu thế cũ của MPC (≈ PPO); hiệu chỉnh từ ACK sửa sai mô hình (13,98 → 30,94); kết hợp PPO + MPC PHY đúng +2,0 so với PPO (cả 3 model), với mô hình hiệu chỉnh chỉ +0,6. B: bottleneck giữ nhiệm vụ (trần 0,475–0,487), mã học được hơn AVIF 10/10 clip; JEPA-32 hơn non-JEPA +0,02 mAP (8/10 clip) nhờ mã ổn định hơn, d64 ngang |
| [swarmbridge-ae2-be1-result.md](swarmbridge-ae2-be1-result.md) | **A-E2/E3 + B-E1/E2 (phản biện #28):** MPC mô hình đúng hơn PPO khoảng +2,7 nhưng tốn 30–100 lần thời gian; MPC mô hình Shannon sai thua PPO 8,6 điểm (cơ chế phù hợp với ưu thế PPO); ngang chi phí PPO thì MPC thua. B: PCA nén token được, bù chuyển động có ích ít; predictor học và hoàn thiện kiểu JEPA (target đóng băng) không hơn giữ nguyên, kể cả vòng hở; AVIF→DINOv2 tốt hơn từ khoảng 25 kbit |
| [swarmbridge-ae1-be0-result.md](swarmbridge-ae1-be0-result.md) | **A-E1 + B-E0 (phản biện #27):** PPO mode-only 1,2–1,7 ms / quyết định so với 40–60 µs của heuristic (return trùng A5.5); pipeline token DINOv2 + giữ nguyên + top-k + bitstream chạy, đồng bộ; 30 % patch đạt 98 % mAP; token thô đắt cỡ AVIF → cần nén token; giữ nguyên yếu khi camera di chuyển |
| [swarmbridge-two-track-plan.md](swarmbridge-two-track-plan.md) | **Kế hoạch hiện tại (phản biện #26):** hai hướng song song. Paper A: DRL cho UAV SemCom khi mô hình kênh sai và quyết định có hạn thời gian (từ A5/A5.5). Paper B: JEPA-SemCom trên VisDrone-VID. Ba thí nghiệm tối thiểu mỗi hướng |
| [swarmbridge-songdo-s1-feasibility.md](swarmbridge-songdo-s1-feasibility.md) | **Songdo bước 0–1:** audit độ phủ / đồng thời (validation + AUDITED-TEST), mặt nạ chấm điểm, codec M0 / M1 round-trip, đường cong bit – chất lượng, tải cả đội so với dung lượng đường lên; quyết định GO / NO-GO riêng cho lập lịch Songdo và JEPA |
| [swarmbridge-jepa-semcom-proposal.md](swarmbridge-jepa-semcom-proposal.md) | **Proposal v2.1:** truyền thông ngữ nghĩa có dự đoán, thích ứng theo nhiệm vụ, cho nhiều UAV; dữ liệu chính Songdo Traffic (10 drone tuần tra đồng thời, đã kiểm), Town3 cho JEPA đa nhiệm vụ; chờ duyệt |

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
