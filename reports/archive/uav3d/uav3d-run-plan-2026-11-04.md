# Plan chạy UAV3D (2026-11-04)

Không có checkpoint CERF (Baidu). Ta dùng checkpoint collaborative của UAV3D (Google Drive), không train detector.

## Cấu trúc model UAV3D cần nhớ
- Ảnh 256×704, 25 camera/sample. Swin-T → LSS → BEV 80 kênh, lưới 0.8 m, phạm vi ±102.4 m, **mọi camera chiếu thẳng vào hệ UAV0**.
- Lấy trung bình 5 camera của mỗi UAV → feature BEV riêng của từng UAV, `[B, 5, 80, H, W]` (`fusion_models/bevfusion.py` dòng 163–180).
- `lowerbound`: chỉ UAV0. `upperbound`: trung bình 5 UAV. `DiscoNet`: hạ độ phân giải 4×, gộp theo trọng số từng pixel, lấy UAV0.
  Không có phép biến đổi giữa các agent, vì feature đã cùng hệ tọa độ.

## Các bước
| # | Việc | GPU | Trạng thái |
|---|---|---|---|
| 1 | Môi trường `~/envs/uav3d` (py3.8, torch 1.10.1+cu113, mmcv-full 1.4.0, nvcc 11.3 + gcc 9), build op BEVFusion | không | đang chạy |
| 2 | Tải checkpoint lower/DiscoNet/V2VNet/upper và det_result + log từ Google Drive | không | |
| 3 | Suy luận lower-bound và DiscoNet trên mini val (80 sample), đối chiếu với README | ~10 phút | |
| 4 | Kiểm tra split: tên scene/token của mini có trong train của checkpoint không (log, det_result) | không | |
| 5 | Lưu feature BEV từng UAV và detection từng UAV (decoder lower-bound chạy trên feature của UAV j) cho cả 400 sample | ~20 phút | |
| 6 | Thí nghiệm output: không gửi / late fusion tất cả box / chọn box (đoán vùng đồng đội bị che, theo score) / oracle GT; đo mAP + tracking (CenterPoint greedy) theo số bit được mã hóa thật | CPU | |
| 7 | Thí nghiệm feature: upper (trung bình) / DiscoNet / feature nén hoặc chỉ gửi vùng chọn (Where2comm-style) với cùng encoder | ~30 phút | |
| 8 | Tracking dưới bị che, trễ, mất gói: box có hụt so với feature không, hụt ở đâu (cổng chính) | CPU/GPU ngắn | |
| 9 | Report, Codex kiểm tra | | |

## Quy tắc
- Box thật (GT) chỉ dùng làm oracle và để chấm điểm, không dùng làm chính sách.
- Bit được tính theo từng trường (vị trí, kích thước, yaw, score, ID, timestamp, độ bất định, header).
- Mini là dữ liệu chưa xác minh split, nên số tuyệt đối chỉ để tham khảo; so sánh giữa các phương án dùng chung detector.
- Mỗi lúc một job GPU (`runs/.swarmbridge-gpu.lock`).
