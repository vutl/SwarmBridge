# Paper A: khung bài, nguồn số liệu, việc còn thiếu

**Tên tạm:** *Does Deep RL Help Multi-UAV Semantic Image Offloading under Realistic Link Adaptation? A Reproducible
Study and a Calibrated Planning Baseline.*

**Luận điểm một câu:** khi đánh giá với link adaptation thực tế và baseline được tune công bằng, ưu thế của DRL trong
truyền ảnh ngữ nghĩa nhiều UAV biến mất; bộ lập kế hoạch MPC hiệu chỉnh online từ ACK, lấy heuristic làm incumbent, là
lựa chọn tốt nhất trong ngân sách tính toán 50 ms.

**Loại bài:** nghiên cứu đánh giá + một phương pháp vừa phải.
- Không tuyên bố thuật toán đột phá.
- Tạp chí mục tiêu chọn cùng giáo sư. Ứng viên: IEEE Open Journal of the Communications Society, Computer Networks,
  Vehicular Communications, IEEE Access.

## Đóng góp (3)

1. **Một simulator nhiều UAV có thể tái lập**, gồm:
   - mode nén thật đo trên VisDrone (14 mode, payload / PSNR / TP detector);
   - PHY link adaptation mức block (CQI trễ, OLLA, ARQ, nhiễu đồng kênh, LA biết lịch);
   - hàng đợi, deadline, năng lượng.
2. **Bằng chứng rằng giả định kênh Shannon và baseline yếu tạo ra ưu thế DRL không có thật:**
   - chính sách train trên Shannon thua 16,8 dưới LA;
   - lợi ích dùng chung kênh biến mất;
   - luật kiểu bài báo không tune thua 10–170;
   - heuristic JA ≥ RL lập lịch;
   - so sánh cuối đăng ký trước: PPO-LA + MPC − heuristic + MPC = −0,14 [−2,83, +2,56].
3. **Baseline lập kế hoạch:** MPC chọn lọc, hiệu chỉnh từ ACK (theo slot), có cổng theo độ tin cậy, chạy trong 50 ms.
   - +17 so với MPC danh nghĩa;
   - bền với feedback trễ / mất;
   - bản thân ý tưởng đã có ở ABR (RobustMPC); đóng góp là áp dụng và đánh giá cho nhiều UAV dưới LA.

## Cấu trúc và nguồn

| Mục | Nội dung | Nguồn (báo cáo → kết quả) |
|---|---|---|
| I. Introduction | Nhiều bài báo DRL cho UAV SemCom đánh giá với Shannon + baseline không tune; câu hỏi của bài | Khảo sát 19 bài trong A-X2 |
| II. Related work | DRL cho UAV SemCom (Huang JSAC 2025, Li JSAC 2026, Si 2023); lập lịch UAV bằng RL; MPC trong ABR; LA / OLLA | Rà soát novelty 13/11, 18/11 |
| III. System model | UAV, quỹ đạo khảo sát, encoder, bảng mode, PHY LA, hàng đợi, deadline, năng lượng, utility | `uav-semcom-a.md` §4, A-X5, A-X6 (`env.py`, `env_la`) |
| IV. Controllers | Heuristic (fixed, budget, paced), JA, PPO (C-only, mode-only, train Shannon / LA), MPC (danh nghĩa, hiệu chỉnh, chọn lọc, incumbent) | A-X2, A-X6, A-E4…A-E11, A-X8 |
| V-A. Shannon so với LA | Cùng bộ chính sách, hai PHY: kết luận đổi chiều | **Bảng 1: cần chạy (R1)** |
| V-B. Sức mạnh baseline | Luật bài báo không tune so với tune; JA so với RL | A-X2 (`runs/uav_semcom/ax2/`), A-X6, A-X7 |
| V-C. Lập lịch dưới LA | Dùng chung kênh, LA biết lịch, chọn UAV, oracle thông tin | A-X5, A-X6, A-X7 |
| V-D. Bộ điều khiển mode | MPC hiệu chỉnh, PPO, incumbent; ngân sách 50 ms | A-X8, A-X9 (`runs/uav_semcom/ax9/test.json`) |
| V-E. Thông tin nội dung ảnh | Oracle tiện ích detector; không tổng quát sang test-dev | A-T1, bước C |
| VI. Discussion | Khi nào RL có thể đáng: phụ thuộc dài hạn, hợp tác nhận thức (dẫn sang Paper 2) | Rà soát thiết kế 18/11 |

## Hình và bảng

| # | Nội dung | Có sẵn? |
|---|---|---|
| Hình 1 | Sơ đồ hệ thống (UAV → encoder → LA → BS) | Vẽ mới |
| Bảng 1 | **Kết luận đổi chiều:** cùng chính sách trên Shannon và trên LA | **Thiếu, chạy R1** |
| Hình 2 | Luật bài báo không tune / tune / JA / RL (return) | Có (A-X2, A-X6), cần vẽ |
| Hình 3 | Hiệu chỉnh từ ACK: MPC danh nghĩa / hiệu chỉnh / không MPC dưới LA | Có (A-X9: −16,95), cần vẽ |
| Bảng 2 | So sánh cuối đăng ký trước (A-X9) + latency p99 | Có |
| Hình 4 | Return và recall không đồng hướng (PPO-LA rẻ mode) | Có (A-X9) |
| Bảng 3 | Dư địa thông tin nội dung: A-T1 (cùng nguồn) so với bước C (test-dev) | Có |

## Việc còn thiếu (theo thứ tự)

1. **R1, bảng "kết luận đổi chiều"** (CPU, khoảng nửa ngày):
   - **Chính sách:** luật bài báo không tune, heuristic tune (H\*), JA + H\*, PPO-Shannon, PPO-LA, H\* + MPC hiệu
     chỉnh, PPO + MPC hiệu chỉnh.
   - **Môi trường:** Shannon K5 và LA K5 (persistent, LA biết lịch), cùng seed mới.
   - **Mục đích:** một bảng duy nhất cho thấy xếp hạng các phương pháp đổi khi PHY đổi.
   - Dùng lại checkpoint và tham số đã đóng băng; không train mới, không tune trên test.
2. **R2 (nên có): tái lập 1–2 phương pháp đã công bố có code** (`pdf/GithubModels/UAV-DDPG`,
   `UAV-PATH-OPTIMIZATION`):
   - giữ thuật toán của họ, thay kênh Shannon bằng LA của mình;
   - nếu kết luận của họ đổi, đó là bằng chứng ngoài cho đóng góp 2;
   - trước tiên phải đọc code để xem có thay kênh được không.
3. **Hình:** vẽ từ JSON có sẵn (script vẽ mới, không chạy lại thí nghiệm).
4. **Viết:**
   - III–IV từ báo cáo có sẵn;
   - V từ bảng / hình;
   - I–II, VI sau cùng.
5. Đọc lại toàn bài với giáo sư, rồi nộp.

## Không làm trong Paper A

Train RL mới, thêm biến điều khiển, hướng hợp tác (để cho Paper 2), SB / JEPA / Mamba.
