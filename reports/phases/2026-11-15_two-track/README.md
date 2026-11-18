# Hai hướng: Paper A (DRL + MPC khi mô hình kênh sai) và Paper B (JEPA-SemCom) (15–17/11, phản biện #26–#36)

**Trạng thái: ĐÓNG.**
- Paper A chuyển sang giai đoạn lập lịch ([2026-11-17_paperA-scheduling](../2026-11-17_paperA-scheduling/README.md)),
  rồi đóng ở phản biện #47.
- Paper B tạm dừng từ phản biện #35.

Mỗi file `swarmbridge-aeX-beY-result.md` chứa cả phần A và phần B của một vòng. Bản nháp Method Paper A (v2):
[swarmbridge-paperA-method-draft.md](swarmbridge-paperA-method-draft.md). Bản nháp này đã lỗi thời sau A-X8/A-X9.

## Paper A

Nền trước giai đoạn này (A5, giai đoạn [2026-11-06_uav-semcom-a](../2026-11-06_uav-semcom-a/README.md)):

| Vòng (phản biện) | Câu hỏi | Kết quả chính | Báo cáo |
|---|---|---|---|
| A5 / A5.5 (07–10/11) | PPO chọn mode có hơn heuristic mạnh nhất H\* không | +5,7 (test) / +7,8 (test-dev) ở phy_mcs_blind; K5 test-dev +12,4. Cơ chế chưa rõ | [uav-semcom-a](../2026-11-06_uav-semcom-a/README.md) |
| Kế hoạch (#26) | Tách hai hướng A / B | Paper A: amortized optimization + bền với sai mô hình | [plan](swarmbridge-two-track-plan.md) |
| A-E1 (#27) | PPO tốn bao nhiêu | 1,2–1,7 ms / quyết định so với 40–60 µs của heuristic; không đáng kể so với slot 1 s | [ae1-be0](swarmbridge-ae1-be0-result.md) |
| A-E2/E3 (#28) | PPO so với MPC | MPC mô hình đúng +2,7 nhưng đắt 30–100 lần; MPC Shannon (sai) −8,6 | [ae2-be1](swarmbridge-ae2-be1-result.md) |
| A-E4 (#29) | Rollout phi tiên tri, hiệu chỉnh từ ACK, kết hợp | MPC phi tiên tri ≈ PPO (+0,6); hiệu chỉnh ACK sửa sai mô hình (13,98 → 30,94); PPO + MPC đúng +2,0 | [ae4-be3](swarmbridge-ae4-be3-result.md) |
| A-E5 (#30) | Cổng theo sai số mô hình | Log throughput lệch −12 % → hiệu chỉnh theo slot (`cals`); cổng chặn Shannon; +1,69 / +1,86 | [ae5-be4](swarmbridge-ae5-be4-result.md) |
| A-E6 (#31) | Cổng theo độ tin cậy của lợi ích quyết định | +2,82 / +2,23; giá trị riêng là chống bias (Shannon +1,30 thay vì −8,67) | [ae6-be5](swarmbridge-ae6-be5-result.md) |
| A-E7 (#32) | Giảm chi phí | Hai tầng (sàng lọc H₁ = 1 → MPC 3 mode) +2,39 / +2,22 ở 25 / 20 ms | [ae7-be6](swarmbridge-ae7-be6-result.md) |
| A-E8 (#33) | Ngân sách 50 ms | Giữ hạn (p99 40–44 ms); ứng viên bi quan giúp khi mô hình lệch | [ae8-be7](swarmbridge-ae8-be7-result.md) |
| A-E9 (#34) | Seed mới, feedback trễ/mất | +2,18 / +1,91; feedback 1–3 slot, mất 30–60 % không hại (nhờ warm-start) | [ae9-be8](swarmbridge-ae9-be8-result.md) |
| A-E10 (#35) | Warm-start lệch regime | −6,4 / −12,1 so với PPO; tracker không phát hiện. Method v2 | [be9-ae10](swarmbridge-be9-ae10-result.md) |
| A-E11 (#36) | Sửa A-E10 | Cửa sổ 30 quan sát: tệ nhất −0,14; warm-start chỉ hơn cold +0,4 / +0,9 | [ae11-ax1](swarmbridge-ae11-ax1-result.md) |
| A-X1 (#36) | Kênh × mode có tương tác không | Có ở K5; 16 heuristic chọn kênh không hơn T:best | [ae11-ax1](swarmbridge-ae11-ax1-result.md) |

## Paper B (tạm dừng)

**Paper B** (JEPA-SemCom trên VisDrone-VID) **tạm dừng** từ phản biện #35.
- Codec video liên khung (HEVC/AV1) → DINOv2 thắng codec ngữ nghĩa ở ≤ 12 kbit (B-E7).
- ROI-QP trong HEVC có trần khoảng +0,01 AP (B-E8, B-E9).
- Hướng còn lại được đề xuất: (a) lớp tăng cường ngữ nghĩa trên nền video. Chưa duyệt.

| Vòng | Kết quả chính | Báo cáo |
|---|---|---|
| B-E0 (#27) | Pipeline token DINOv2 + giữ nguyên + top-k + bitstream chạy, đồng bộ; token thô đắt cỡ AVIF | [ae1-be0](swarmbridge-ae1-be0-result.md) |
| B-E1/E2 (#28) | PCA nén token được; predictor học / kiểu JEPA không hơn "giữ nguyên"; AVIF → DINOv2 tốt hơn từ khoảng 25 kbit | [ae2-be1](swarmbridge-ae2-be1-result.md) |
| B-E3 (#29) | Bottleneck giữ nhiệm vụ; mã học được hơn AVIF 10/10 clip; JEPA-32 +0,02 mAP nhờ mã ổn định hơn | [ae4-be3](swarmbridge-ae4-be3-result.md) |
| B-E4 (#30) | 3 seed ghép cặp: JEPA +0,025 / +0,013 / +0,005, CI chứa 0 | [ae5-be4](swarmbridge-ae5-be4-result.md) |
| B-E5 (#31) | Chỉ làm mượt theo thời gian ≥ JEPA ở cùng bit → lợi ích JEPA là do ổn định theo thời gian | [ae6-be5](swarmbridge-ae6-be5-result.md) |
| B-E6 (#32) | Surrogate số bit exp-Golomb: +0,030–0,036 mAP so với task ở 9–18 kbit | [ae7-be6](swarmbridge-ae7-be6-result.md) |
| B-E7 (#33) | HEVC/AV1 liên khung → DINOv2 thắng codec ngữ nghĩa ở ≤ 12 kbit và trên patch thay đổi → Paper B dạng hiện tại không đứng | [ae8-be7](swarmbridge-ae8-be7-result.md) |
| B-E8 (#34) | ROI-QP trong HEVC theo DINOv2 +0,01 AP (~11 % bit); ap_change không tăng; số hạng sự kiện không kích hoạt | [ae9-be8](swarmbridge-ae9-be8-result.md) |
| B-E9 (#35) | Hạn mức event riêng: ap_change +0,011 ở 6 kbit = trần GT; không bền → đề xuất hướng (a) | [be9-ae10](swarmbridge-be9-ae10-result.md) |

Tổng quan mọi giai đoạn: [reports/README.md](../../README.md). Ngữ cảnh ngắn: [CONTEXT.md](../../CONTEXT.md).
