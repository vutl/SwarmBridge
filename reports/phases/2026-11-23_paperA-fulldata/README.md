# 2026-11-23_paperA-fulldata: đánh giá mở rộng Paper A (23–25/11), XONG

**Câu hỏi:** trên dữ liệu VisDrone mới (protocol `fd_v1`, namespace `runs/uav_semcom/paperA_fd/`), lớp MPC chọn lọc
có còn đứng không, khi kiểm tra bằng các phép thử đăng ký trước:
- so với thiết kế danh định cùng cổng + dải (matched design);
- bằng ablation từng thành phần;
- ở nhiều cấu hình mạng;
- với RL train lại?

**Kết luận:** lớp này tăng return so với heuristic tune và so với cả 3 họ RL. Nhưng:
- lợi ích đến từ **dải bền vững**;
- cổng và tầm nhìn H = 4 không cần thiết;
- hiệu chỉnh ACK không thêm return ở cấu hình chuẩn và **làm giảm recall**;
- planner không bảo vệ có recall cao nhất;
- chạy tuần tự vượt 50 ms.

Khuyến nghị **C**: định vị lại Paper A là đóng góp ứng dụng / thực nghiệm. Đang chờ phản biện.

## Tài liệu (theo thứ tự đọc)

| File | Nội dung |
|---|---|
| [paperA-fd-consolidated.md](paperA-fd-consolidated.md) | **Đọc trước.** Ma trận claim / bằng chứng, vấn đề return / recall, khuyến nghị A / B / C / D, nhật ký chạy |
| [paperA-full-data-protocol.md](paperA-full-data-protocol.md) | Audit dữ liệu (rò rỉ), các pool, seed, các lần sửa protocol |
| [paperA-fd-phase1.md](paperA-fd-phase1.md) | Profile 14 mode trên ảnh mới, kiểm tra tái lập |
| [paperA-fd-phase2-prereg.md](paperA-fd-phase2-prereg.md) / [result](paperA-fd-phase2-result.md) | Cấu hình chuẩn: cal − H\* +7,28 (Holm); cal − danh định cùng thiết kế −0,10 (n.s.) |
| [paperA-fd-ablation-prereg.md](paperA-fd-ablation-prereg.md) / [result](paperA-fd-ablation-result.md) | Dải cần thiết (+15); cổng không; H4 vs H1 không lặp lại; H ≥ 1 cần thiết |
| [paperA-fd-phase4-prereg.md](paperA-fd-phase4-prereg.md) / [result](paperA-fd-phase4-result.md) | Số UAV / kênh / băng thông: G1 5/6, G2 2/6; thời gian tính toán |
| [paperA-fd-phase3-prereg.md](paperA-fd-phase3-prereg.md) / [result](paperA-fd-phase3-result.md) | PPO / QR-DQN / DSAC train lại: R1 3/3, R2 3/3; không họ RL nào hơn H\* |
| [paperA-fd-posthoc-result.md](paperA-fd-posthoc-result.md) | **Hậu kiểm (25/11, không đăng ký trước, chỉ dùng dữ liệu đã lưu).** Kiểm tra fail-fast 27.600/27.600 episode; phân rã return nc / ng (ng gửi file nhỏ hơn → ít drop, nhiệm vụ ngắn hơn → return cao hơn, nhưng PSNR thấp hơn → ít vật thể detect được hơn); PPO + cal so với H\* + cal (+0,27 n.s.); bootstrap theo video cho R5 recall |
| [paperA-novelty-audit-ea-smpc.md](paperA-novelty-audit-ea-smpc.md) | Rà soát tính mới: không có thành phần thuật toán mới |

**Bản tóm tắt cho phản biện mới:** [../../CHATGPT_HANDOFF.md](../../CHATGPT_HANDOFF.md).

**Commit:**
- `3cf4fd62`: ablation, Phase 4, zero-shot.
- `000251c7`: train RL + resume sau khi mất điện.
- `bc4d2736`: xong chuỗi, các báo cáo.

**Kiểm tra sau chuỗi:** `runs/uav_semcom/paperA_fd/supplement_{coverage,ckpts,extras}.log`.
