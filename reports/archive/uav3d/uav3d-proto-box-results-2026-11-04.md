# UAV3D protocol v1 — kết quả gộp box (dev, CPU) — 2026-11-04

Phạm vi: chỉ 3 scene dev (scene_11, 13, 15; 60 frame). Chưa có khoảng tin cậy, chưa đụng test. Không chạy GPU.
Code: `swarmbridge/experiments/uav3d_comm/proto/` (ledger, boxcodec, kalman, box_proto; 70 unit test pass).
Số liệu thô: `runs/uav3d_proto/box/box_proto.json` (mAP, sha256 từng file dự đoán, kbit),
`runs/uav3d_proto/box/tracking/<method>__<ledger>/{input.json,metrics_summary.json}` (AMOTA; `input.json` ghi sha256
file đầu vào và đã được đối chiếu khớp với `box_proto.json` cho cả 61 dòng).
File dự đoán/track (~690 MB) không đưa lên git; tái tạo tất định bằng
`python -I box_proto.py DATA_ROOT runs/uav3d_comm/agents_ft_cap500 runs/uav3d_proto/box` rồi
`scripts/uav3d_proto/track_box_proto.sh`.

## Luồng thí nghiệm (nói gọn)

1. Mỗi UAV phụ j (1..4) tại frame gửi s lấy detection của chính nó (cùng decoder fine-tune, cap 500), giữ box score ≥ 0.3.
   - `box_*`: gửi nguyên box (vận tốc = 0 vì detector không ước lượng vận tốc).
   - `kf_*`: UAV phụ chạy Kalman vận tốc hằng (CV) trên detection của chính nó, chỉ gửi track đã xác nhận và vừa được
     cập nhật ở frame đó, kèm vận tốc ước lượng ("CERF-inspired", không phải tái hiện CERF).
2. Mã hóa thật bằng `boxcodec` (header 132 bit + 87 bit/box), giải mã lại; số bit tính cho MỌI packet đã phát, kể cả
   packet bị mất.
3. Kênh = ledger nhân quả: tại frame t, UAV0 dùng packet mới nhất (theo send_frame) đã tới trước t; quá 3 frame thì bỏ;
   frame đầu chưa có gì thì không dùng (cold start thật).
4. UAV0 gộp detection của mình với message nhận được: `max` (giữ bản tự tin nhất) hoặc `nor` (noisy-OR).
   `kf_*` dự báo tiến vị trí theo tuổi message × 0.2 s trước khi gộp. Cắt 500 box/frame.
5. Đo mAP (tất cả frame và frame ≥ 3) và AMOTA bằng PubTracker với dt = 0.2 s.

Kịch bản kênh: sync (trễ 0), fix1 / fix2 (trễ cố định 1 / 2 frame), rand (trễ 0..3 với xác suất 0.4/0.3/0.2/0.1,
độc lập từng link), loss0.2 / loss0.5 (mất gói, trễ 0), rand_loss0.2. Kịch bản ngẫu nhiên chạy 3 seed; bảng ghi trung bình.

Băng thông phát (trung bình mỗi frame, 4 UAV phụ): box 49.6 kbit, kf 41.1 kbit.

## Bảng chính

Ego (không truyền gì): mAP 0.512 (frame ≥ 3: 0.521), AMOTA 0.542, IDS 52.

| Kênh | box_max mAP / AMOTA | box_nor mAP / AMOTA | kf_max mAP / AMOTA | kf_nor mAP / AMOTA |
|---|---|---|---|---|
| sync | 0.612 / 0.592 | 0.665 / 0.594 | 0.607 / 0.677 | 0.648 / **0.690** |
| fix1 | 0.523 / 0.538 | 0.575 / 0.567 | 0.543 / 0.596 | 0.594 / 0.631 |
| fix2 | 0.398 / 0.341 | 0.433 / 0.408 | 0.470 / 0.480 | 0.519 / 0.529 |
| rand (3 seed) | 0.523 / 0.472 | 0.601 / 0.523 | 0.557 / 0.604 | 0.611 / 0.639 |
| loss0.2 (3 seed) | 0.578 / 0.542 | 0.649 / 0.567 | 0.592 / 0.663 | 0.639 / 0.675 |
| loss0.5 (3 seed) | 0.522 / 0.458 | 0.598 / 0.507 | 0.555 / 0.598 | 0.607 / 0.635 |
| rand_loss0.2 (3 seed) | 0.493 / 0.431 | 0.575 / 0.497 | 0.538 / 0.573 | 0.597 / 0.619 |

Độ dao động giữa 3 seed của AMOTA: ±0.002 đến ±0.021 (nửa khoảng min–max). mAP frame ≥ 3 của kf_nor:
sync 0.663, fix1 0.626, fix2 0.561, rand 0.634.

## Đọc kết quả

1. **Codec thật không làm mất kết quả cũ.** box_nor sync = 0.665 mAP, đúng bằng số noisy-OR trước đây (lúc đó box chưa
   lượng tử hóa thật). Codec box: 0 trường bị bão hòa. Codec kf: 52 trường vận tốc bị cắt (vx 19, vy 32) trên ~28 000
   track (0.2%), tốc độ ước lượng p50 2.9, p99 10.6, max 20.6 m/s. Đây là track mới khởi tạo hoặc ghép nhầm, xe thật
   chạy dưới khoảng 5 m/s, nên cắt ở ±12.8 m/s là vô hại; giữ nguyên spec và ghi lại ở đây.

2. **Trễ làm mất giá trị message, và với tracking thì mất gần hết sau 2 frame, kể cả khi đã dự báo bằng Kalman.**
   Lấy kf_nor làm baseline bù trễ tốt nhất hiện có, phần lợi so với ego:

   | | mAP (frame ≥ 3) | AMOTA |
   |---|---|---|
   | sync | +0.142 | +0.148 |
   | fix1 | +0.105 (giữ 74%) | +0.089 (giữ 60%) |
   | fix2 | +0.040 (giữ 28%) | **−0.013** (mất hết, hơi tệ hơn ego) |

   Không bù trễ (box_nor fix2) thì AMOTA 0.408, thấp hơn ego rất xa: message trễ 2 frame **gây hại** cho tracking. Kalman
   xóa phần gây hại, nhưng không lấy lại được giá trị. Đây là dạng yếu hơn nhưng bền hơn của quan sát A: "message đến muộn
   mất giá trị, giá trị phụ thuộc thời điểm đến", và nó vẫn đứng vững sau khi đã bù trễ tử tế.

3. **Mất gói ít hại hơn trễ.** loss0.5 (mất một nửa số gói, trễ 0) vẫn cho kf_nor AMOTA 0.635 (±0.021 giữa các seed), ngang fix1 (0.631) và cao hơn hẳn fix2 (0.529).
   Một message tươi thỉnh thoảng có giá trị hơn message đều đặn nhưng cũ. Đây là tín hiệu ủng hộ việc chọn gửi gì và
   khi nào theo thời điểm đến, nhưng mới là 3 scene.

4. **Cảnh báo — AMOTA của kf tăng ngay cả khi không trễ (0.690 so với 0.594), và phần lớn không phải nhờ bù trễ.**
   Trong kf_*, 76% box tự tin sau gộp mang vận tốc Kalman của UAV phụ; box_* thì 0%. PubTracker dùng vận tốc để ghép
   track, nên message dạng track-state (có vận tốc) giúp tracking hơn hẳn message dạng box, dù mAP thấp hơn
   (0.648 so với 0.665). Vì vậy:
   - so sánh "có bù trễ / không bù trễ" phải làm **bên trong cùng một loại message** (kf sync → fix1 → fix2 như bảng ở mục 2),
     không so kf với box;
   - cần thêm một đối chứng: message kf nhưng xóa vận tốc trước khi đưa vào tracker (tách "dự báo vị trí" khỏi
     "truyền vận tốc"). Việc này chạy trên CPU, chưa làm.
   - Điều này cũng là một gợi ý thật: **loại nội dung message** (box hay track-state) ảnh hưởng tracking khác với ảnh hưởng
     detection; một hệ chọn message theo mAP sẽ chọn sai.

5. **noisy-OR tốt hơn max ở mọi kịch bản**, ở cả mAP và AMOTA, cho cả box và kf.

## Kiểm tra dt của tracker

Tracker chạy với dt theo timestamp và dt = 0.2 s cho ra AMOTA giống hệt cho 4 file cũ (ego 0.542, feature 0.754,
noisy-OR 0.588, CERF-style 0.721; `runs/uav3d_comm/tracking_dtcheck/`), vì các file đó có vận tốc 0. File kf có vận tốc
thật nên protocol v1 cố định `TRACK_DT=0.2`.

## Giới hạn

- 3 scene dev, 60 frame, chưa có khoảng tin cậy (bootstrap theo scene sẽ gần như vô nghĩa với 3 scene).
  Các khẳng định ở trên là giả thuyết cho test set, chưa phải kết luận.
- Kalman phía gửi có tham số chọn tay (sigma_a 3, sigma_r 0.5, gate 3 m), chưa tune trên train. Một Kalman tốt hơn có thể
  giữ được nhiều hơn 28% ở trễ 2 frame; cần thử độ nhạy trước khi khẳng định.
- Chỉ có message dạng box. Message dạng feature (q8 có codec thật) chưa làm lại theo protocol.

## Việc tiếp theo (CPU)

1. Đối chứng kf-không-vận-tốc (mục 4).
2. Độ nhạy Kalman: sigma_a ∈ {1, 3, 6}, gate ∈ {2, 3, 5}.
3. Codec feature q8 có đếm byte; warp SE(2) tổng hợp; sửa cold start trong script cũ; baseline tracker không lọc trước
   và quét ngưỡng; manifest + hash; cập nhật protocol lên v1.1.
4. Thí nghiệm khả thi: chọn message theo ngân sách bit có hàng đợi, so top-K / novelty / Kalman + greedy với oracle dùng
   GT tương lai (chỉ để chẩn đoán).
