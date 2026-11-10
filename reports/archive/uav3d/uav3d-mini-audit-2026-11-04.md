# Kiểm tra UAV3D v1.0-mini (2026-11-04)

Mục tiêu: trả lời cổng 1 của Codex, "dữ liệu có tín hiệu cộng tác thật không?", trước khi chọn thuật toán.
Script: `swarmbridge/experiments/uav3d_audit/`. Kết quả từng object: `runs/uav3d_audit/rows_{0.15,0.3,0.5}.npz`.

## Dữ liệu có gì

| Mục | Giá trị |
|---|---|
| Scene | 20 (tất cả Town10), 16 train / 4 val theo infos pkl (320 / 80 sample) |
| Frame | 20 frame mỗi scene, **dt mô phỏng 0.2 s** (5 Hz) → mỗi scene chỉ 4 s. Timestamp tăng ~2.7 s/frame là đồng hồ thật, không phải thời gian mô phỏng (kiểm bằng quãng đường / vận tốc = 0.200 s) |
| UAV | 5 chiếc, đội hình chữ thập cố định, cùng độ cao 61.9 m, cách nhau 20 m (cặp xa nhất ~40 m), bay ~5 m/frame |
| Camera | 5 camera mỗi UAV (FRONT/BACK/LEFT/RIGHT nghiêng 45°, BOTTOM nhìn thẳng xuống), 800×450, f = 400 |
| Kèm theo | Ảnh RGB và ảnh phân đoạn `_seg.png` cho mỗi camera (xe = màu (95, 90, 229)) |
| Nhãn | 60 940 box 3D, ~155 xe mỗi sample, có ID track và vận tốc; **chỉ có xe** (27 mẫu xe CARLA), không có người đi bộ; trường visibility và num_lidar_pts đều trống |
| Dung lượng | 7.1 GB cho 400 sample → bản đầy đủ (1000 scene) ước khoảng 350 GB |

## Bẫy hình học (cần ghi nhớ)
Quaternion trong `calibrated_sensor` **không theo quy ước camera của nuScenes**. Phép chiếu đúng, tìm bằng cách dò 48 hoán vị trục × 4 phép lật và so tâm box với ảnh seg:

```
d = R(q)^T · ((X − t) ⊙ [1, 1, −1]);   camera = (d_y, d_z, d_x);   uv = K · camera / depth
```

Với phép này, 36% tâm box rơi đúng pixel xe; mọi phép khác cho dưới 6%. Phần còn lại là xe bị che. Box vẽ lên ảnh khớp xe (`runs/uav3d_audit/CAMERA_*.png`). Box trong JSON và trong pkl trùng nhau (lệch 0.00 m sau khi cộng ego).

Xe đỗ tĩnh trong Town10 thuộc bản đồ, không phải actor, nên **không có nhãn** nhưng vẫn có màu xe trong ảnh seg.

## Tín hiệu cộng tác (đo trên toàn bộ 400 sample × 25 camera)
Một UAV được tính là "thấy" một xe nếu có ít nhất một camera mà khung box chiếu lên chứa ≥ ngưỡng % pixel xe trong ảnh seg.

| Chỉ số | ngưỡng 0.15 | ngưỡng 0.3 |
|---|---|---|
| Xe nằm trong tầm nhìn hình học của cả 5 UAV | 98.1% | 98.1% |
| Xe được ít nhất một UAV thấy thật | 58.2% | 48.7% |
| Mỗi UAV tự thấy (trong số xe cả đội thấy) | 59–67% | 54–61% |
| **Đồng đội bổ sung thêm** (bị che với tôi, UAV khác thấy) | **33–41%** | **40–46%** |
| Xe chỉ đúng **một** UAV thấy (trong số xe cả đội thấy) | 24.5% | 28.6% |
| Xe trong bán kính 30 m: đồng đội bổ sung | 4.3% | 7.6% |
| Xe trong bán kính 60 m: đồng đội bổ sung | 20.9% | 27.7% |
| Xe trong bán kính 100 m: đồng đội bổ sung | 32.8% | 39.8% |

Ngưỡng 0.5 không dùng được: với camera nghiêng, khung chữ nhật bao xe chứa nhiều nền, nên xe ở gần cũng bị tính "không thấy" (chỉ 25% xe trong 30 m được thấy).

Kết luận:
- **Có tín hiệu cộng tác thật, và nó đến từ việc bị che** (nhà cao tầng, cây). Nó không đến từ chuyện ngoài tầm nhìn: gần như mọi xe đều nằm trong tầm nhìn của cả 5 UAV.
- Giá trị nằm ở tầm trung và xa (30–100 m). Ở gần UAV gần như tự thấy đủ.
- Có dư thừa: khoảng 9–16% xe được cả 5 UAV thấy, và gửi những xe đó là phí.
- Điều này khớp với README của UAV3D: cộng tác đưa mAP từ 0.544 (lower-bound) lên 0.720 (upper-bound) và AMOTA từ 0.644 lên 0.812.

## Giới hạn cho hướng nghiên cứu
1. **Scene ngắn (4 s)**: không đủ cho bài toán tối ưu dài hạn (cổng 3 của Codex). RL nhiều bước khó có ý nghĩa nếu không có simulator.
2. **Đội hình cố định 20 m**: không có relay, không có topology thay đổi. Kênh truyền phải tự mô phỏng.
3. **Bản mini chỉ có Town10, 4 scene val**: đủ để kiểm tra và suy luận, không đủ để train detector.
4. **Ngưỡng thấy/không thấy dựa trên ảnh seg là xấp xỉ**: chưa phân biệt xe bị che một phần với xe nhỏ ở xa.

## Bước tiếp theo có thể làm
- **(a)** Chạy code UAV3D (BEVFusion cộng tác) ở chế độ suy luận với checkpoint họ phát hành, trên 80 sample val của bản mini, để tái hiện khoảng cách lower → V2VNet/DiscoNet → upper. Cần một môi trường WSL riêng với torch/mmdet3d bản cũ.
- **(b)** Bài toán "gửi bằng chứng gì": dùng mặt nạ thấy/không thấy ở trên. Bên gửi biết vị trí đồng đội và bản đồ nên có thể dự đoán đồng đội bị che ở đâu. Đây chính là phiên bản đa UAV của kết quả sender-replica. Baseline: gửi trạng thái track (vị trí, kích thước, hướng, vận tốc, covariance) so với gửi đặc trưng BEV nén.
