# SwarmBridge hướng A — A6.0-L1: chọn mode theo giá trị tác vụ (task-aware mode selection)

Bắt đầu 2026-11-09 tối. Kết quả ở `runs/uav_semcom/a6l1/`; A5 không bị sửa.

**Câu hỏi:** giữ nguyên hệ thống direct-link của A5 (topology, kênh, PHY, codec, profile, không gian hành động, quan sát, deadline, Qmin) và **chỉ thay số hạng chất lượng của reward**, từ PSNR sang giá trị phát hiện vật thể. Khi đó:
- policy (PPO và heuristic) có chọn mode khác đi không?
- thông tin tác vụ được giao (recall, AP50, vật nhỏ, false positive) có tăng ở cùng hoặc thấp hơn chi phí truyền, deadline, năng lượng không?

Phạm vi: đây là **điều khiển nén theo tác vụ**, không phải một bộ mã hoá ngữ nghĩa mới.

## 0. Quy tắc định trước (commit trước khi tinh chỉnh, train hay đánh giá bất kỳ thứ gì)

### 0.1 Đo giá trị tác vụ
- Mọi cặp (ảnh, mode) của bảng A5, gồm cả 3 mode sinh ảnh, được đo bằng detector A5 (YOLO11s, train trên VisDrone-train). Code: `a5_fidelity.py --save-dets`, script `run_a6l1_gpu.sh`.
- Điểm hoạt động: conf ≥ 0,25, IoU ≥ 0,5, ghép class-aware, vùng ignore như A5.2.
- Ở mỗi (ảnh i, mode m) có TP_i(m) và FP_i(m) = số detection − TP.

### 0.2 Reward tác vụ

**Giá trị tác vụ** của một khung:
- v_i(m) = (TP_i(m) − β·FP_i(m)) / V̄, với β = 0,5;
- V̄ = trung bình của (TP − β·FP) của **ảnh gốc** trên các ảnh **train**.
- Ảnh có nhiều vật thể thì đáng giá hơn: tổng TP chính là tử số của "object recall được giao".

**Utility mỗi ảnh:** ξ_q·v_i(m) + ξ_d·max(d_max − delay, 0)/d_max − 1·[vi phạm Qmin hoặc deadline], với ξ_q = ξ_d = 0,5.
- Chỉ số hạng ξ_q·clip((PSNR − Qmin)/15) của A5 được thay.
- Phạt rớt ảnh (−1), ảnh chưa chụp, năng lượng, thời gian (w_T = 0,3) và **ràng buộc Qmin** đều giữ nguyên. Qmin là yêu cầu của hệ thống; việc giữ nó là một giới hạn, xem mục 5.
- Không clip v.

**Nhãn dùng ở đâu:**
- reward khi train: nhãn ảnh train;
- chọn checkpoint: nhãn ảnh val;
- đánh giá: nhãn ảnh test và test-dev.

Nhãn **không** bao giờ vào quan sát của policy.

**Quan sát giữ nguyên như A5.**
- Policy biết payload/PSNR dự đoán của mọi mode. Các giá trị này là hàm tuyến tính của log(payload probe), tức của cùng một con số đo được trên UAV.
- Heuristic task-aware dùng v̂_i(m) = a_m + b_m·log(payload probe), fit trên train. Đây là cùng thông tin mà PPO nhìn thấy, không thêm gì.

### 0.3 Heuristic dưới reward tác vụ
- **Họ `tv:SLOT+budgetτ[+reuse]`:** giống các họ budget/pacedB/reuse của A5, nhưng giá trị dự đoán của mỗi mode dùng v̂ thay cho PSNR. 56 thành viên, cùng lưới.
- **Tinh chỉnh lại** cả các họ A5 (base, budget, reuse) **và** họ tv dưới reward tác vụ, trên train seed 0..29.
- **T:best_task** = thành viên có return train (reward tác vụ) cao nhất trong tất cả các họ.
- T:best của A5 (tinh chỉnh dưới PSNR) được giữ làm đối chứng.

### 0.4 PPO
- Mode-only, 3 seed (0, 1, 2), ở 3 regime ref, K5, phy_mcs_blind.
- Slot do phần slot của T:best_task điều khiển.
- Thuật toán, mạng, siêu tham số, 6 env, 2M bước giống hệt A5.
- Checkpoint chọn theo return **tác vụ** cao nhất trên val (seed 3000..3019; bằng nhau thì lấy checkpoint sớm hơn).
- Không tăng số bước. Nếu đường học còn tăng ở 2M, chỉ ghi nhận.

### 0.5 Đánh giá (seed mới 9000..9099, chưa từng dùng; pool test và test-dev)

**Policy:**
- PPO task (best-val);
- PPO PSNR của A5 (mode-only best-val);
- T:best_task;
- T:best của A5;
- fixed tinh chỉnh dưới reward tác vụ;
- thành viên tv tốt nhất.

**Mỗi episode báo cả hai return** (reward tác vụ và reward PSNR của A5, tính song song trong cùng một lần chạy), cùng các chỉ số:
- object recall được giao: Σ TP của khung giao đúng hạn / Σ số vật thể của mọi khung đã chụp;
- recall vật nhỏ;
- FP (tổng và mỗi khung giao) và precision;
- **AP50 chuẩn COCO** trên toàn bộ khung đã chụp; khung không giao được tính như không có detection;
- tổng bit gửi, tỉ lệ đúng hạn, năng lượng, thời gian nhiệm vụ.

**So sánh chính** (theo cặp; CI theo seed môi trường, cộng cluster bootstrap theo video với B = 50 như A5.5 0.4):
1. PPO task − PPO PSNR: recall, AP50, bit, đúng hạn;
2. T:best_task − T:best: cùng các chỉ số;
3. PPO task − T:best_task: return tác vụ, recall.

**Không** dùng "đảo thứ hạng" làm tiêu chí thành công duy nhất. Báo cáo:
- Pareto recall–bit, recall–đúng hạn, recall–năng lượng/thời gian;
- **phân bố mode được chọn** theo tam phân vị của độ phức tạp ảnh (log payload probe), v̂, backlog dự đoán, SNR kỳ vọng và slack, so sánh policy task với PSNR.

Nếu reward tác vụ không tạo khác biệt, hoặc heuristic ngang PPO, thì ghi rõ.

**Phân tích độ nhạy** (khám phá, không train lại): chấm lại điểm tác vụ với β ∈ {0; 1}.
