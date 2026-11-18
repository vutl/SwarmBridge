# Đọc CERF (paper + code) và CoSDH, 2026-11-04

Nguồn: `pdf/CERF Communication-Efficient and Retraining-Free Collaborative Perception.pdf` (arXiv 2609.00951, 4 trang),
repo `pdf/GithubModels/uestchjw__CERF` (1 commit "Add files via upload"), và
`pdf/CoSDH ... Intermediate-Late Hybridization.pdf`.

## CERF làm gì
Mỗi UAV gửi box 3D `[x, y, z, w, h, l, yaw, score]` kèm pose. Bên nhận đưa box về hệ tọa độ của mình, dùng
Kalman filter cộng dự báo vận tốc không đổi (K = 3 frame) để bù trễ, rồi vẽ box thành một "modality ảo" (Poture: 5 kênh
size/heading/score đặt tại ô BEV). Poture được ghép theo kênh vào BEV feature của ego, sau đó đi qua decoder của BEVFusion.
Bảng UAV3D: mAP 0.693 ở 0.04 Mbps, so với DiscoNet 0.700 ở 54.71 Mbps; AMOTA 0.744, so với DiscoNet 0.809.

## Những điểm phải dè dặt khi dùng số của CERF
1. **0.693 đến từ helper có detector mạnh hơn ego.** Bảng 4: ego là BEVFusion (0.544).
   - Ego + DETR3D (helper đơn lẻ đạt 0.618) → **0.693**; đây là số được đưa vào bảng chính.
   - Ego + PETR (0.581, có train cùng) → 0.660.
   - Ego + BEVFusion 704×256 (0.487) → **0.613**, gần bằng late fusion (0.610).
   - Trường hợp **cùng một detector** (BEVFusion + BEVFusion), là phép so công bằng với DiscoNet, **không được báo cáo**.
2. **Code phát hành khác với paper.** 2708/2719 file giống hệt repo UAV3D; chỉ có 3 file BEVFusion bị sửa hoặc thêm.
   - Không có Kalman filter, motion forecasting hay phép tính bandwidth: `single_pred` được nạp sẵn từ ngoài.
   - Không có NMS giữa các agent như paper mô tả: code ghép 5 agent × 5 kênh = 25 kênh (`bevfusion.py`, `for mm in range(5)`), lọc theo score > 0.5.
3. **Có thể có hiệu ứng ensemble.** Vòng `range(5)` không loại agent ego. Nếu `single_pred` chứa cả detection của
   detector helper trên **chính ảnh của ego**, thì một phần lợi ích là ensemble BEVFusion + DETR3D, không phải cộng tác.
   Cần kiểm tra khi có file `single_pred`.
4. **Nhiễu giữa các lần chạy có thể lớn.** Bảng 5: thêm kênh vị trí làm mAP giảm từ 0.693 xuống 0.650. Paper không báo seed hay variance.
5. **Checkpoint ở Baidu**, phải có tài khoản để tải.

Kết luận: CERF đã chiếm câu "gửi box rẻ hơn gửi feature rất nhiều trên UAV3D", nên thí nghiệm đó **không còn là đóng góp**.
Nhưng nó **chưa chứng minh công bằng** rằng box gần bằng feature khi dùng cùng một detector, nhất là ở tracking.
Phép so có kiểm soát vẫn có giá trị như một bước chẩn đoán và làm baseline.

## CoSDH (CVPR 2025): đối thủ gần nhất của ý "nâng cấp biểu diễn"
- Chọn vùng cộng tác theo cung-cầu (ego cần ở đâu, helper có gì ở đâu), rồi kết hợp intermediate và late:
  late fusion bù khi bandwidth thấp.
- Phạm vi: phát hiện **từng frame**, dữ liệu xe/V2X (OPV2V, V2XSim, DAIR-V2X); không có tracking, không có UAV, không có trễ hay mất gói theo thời gian.
- Vì vậy, "gửi box bình thường, gửi feature khi track gặp rủi ro" chỉ khác CoSDH nếu điều kiện kích hoạt dựa trên **thời gian và track**
  (sắp bị che, ID mơ hồ, message trễ hoặc mất) và được đánh giá bằng **tracking**. Phải so trực tiếp với CERF và CoSDH.

## Ảnh hưởng tới plan UAV3D
- Bước 2 cũ ("box vs feature") đổi vai: từ đóng góp thành **chẩn đoán có kiểm soát**: cùng detector đóng băng, cùng bit, đo cả detection lẫn tracking.
- Câu hỏi cổng mới: **khi dùng cùng detector, tracking có còn hụt khi chỉ gửi box không, và hụt ở đâu** (bị che, trễ, mất gói)?
  Nếu không hụt thì ý nâng cấp biểu diễn không có mục tiêu trên UAV3D.
