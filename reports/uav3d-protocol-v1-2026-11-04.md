# Protocol đánh giá UAV3D v1 (khóa trước khi chạy thí nghiệm mới)

Trạng thái: **bản nháp, chờ Codex review**. Khi chốt, gắn git tag `uav3d-protocol-v1`.
Mọi thay đổi sau khi đã xem kết quả test phải ghi thành protocol v2, không sửa đè.

Nguồn các yêu cầu:
- Đính chính trong `uav3d-comm-results-2026-11-04.md`;
- `codex-uav3d-independent-audit-2026-11-04.md`;
- góp ý của ChatGPT ngày 04/11.

## 1. Dữ liệu
| Tập | Nội dung | Được dùng để |
|---|---|---|
| Train | 17 scene mini, trừ scene_11/13/15 | Fine-tune decoder |
| Dev | scene_11, 13, 15 của mini (60 frame) | Debug, chọn siêu tham số, chọn ngưỡng. **Không gọi là test.** |
| Test | Scene val đầy đủ của UAV3D, không thuộc mini | Chỉ chạy **một lần cho mỗi phương án**, sau khi đã gắn tag |

- **Quy tắc với tập test.** Chỉ tải test sau khi đã gắn tag. Nếu đổi gì sau khi đã xem test, kết quả đó là phiên bản mới và phải báo cả hai.
- **Provenance của checkpoint.** Hiện mới có bằng chứng gián tiếp: token của 3 scene dev trùng với file detection val của tác giả. Trước khi chạy test, phải ghi lại manifest train/val chính thức của UAV3D. Nếu không có manifest, phải nói rõ giới hạn này trong mọi báo cáo.
- **Báo theo scene.** Mọi kết quả dev đều báo kèm số liệu riêng cho từng scene.

## 2. Thời gian và độ trễ
- **1 frame = 0.2 s thời gian mô phỏng.** Đã xác minh bằng quãng đường chia vận tốc từ simulator, trung vị 0.2006 s trên 31 880 cặp xe. Timestamp là đồng hồ thật (nano giây) nên **không** dùng để tính dt.
- **Tracker** dùng dt = 0.2 s, không lấy dt từ timestamp.
- **Độ trễ tính theo frame, riêng cho từng helper:** d_j ∈ {0, 1, 2, 3}. Bốn kịch bản:
  1. đồng bộ;
  2. mọi helper cùng trễ 1 frame;
  3. mọi helper cùng trễ 2 frame;
  4. trễ ngẫu nhiên **độc lập theo từng link**: d_j ~ phân phối cố định {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}, 3 seed, lưu thành file.
- **Khởi động lạnh.** Nếu helper chưa có message đủ cũ thì coi là **không có message**; không lấy frame hiện tại thay vào. Báo cả hai: tất cả frame, và chỉ các frame từ frame thứ 4 trở đi (đã đủ lịch sử).
- **Mất gói.** Bernoulli độc lập theo helper và theo frame, p ∈ {0.2, 0.5}, 3 seed, dùng lại `drops.json`. Mất cả message của helper đó. Chưa mô phỏng mất theo cụm (burst).

## 3. Message, codec và bit
- **Mọi message phải đi qua round-trip codec trước khi bên nhận dùng.**
  - Box: lượng tử theo đúng 85 bit/box như định nghĩa trong `fusion.py`, giải mã lại rồi mới gộp.
  - Feature: q8 theo từng kênh (min/max của từng kênh gửi kèm).
  - FP16 dense chỉ là **tham chiếu trên**, không phải một phương án truyền.
- **Bit = payload + 96 bit header** cho mỗi helper có gửi. Không gửi gì thì 0 bit, không tính header.
- **Báo bit/frame** theo trung bình và p95.
- **Phép so box và feature** chỉ được kết luận khi cùng ngân sách bit: có đường chất lượng theo bit, hoặc ít nhất có feature nén thưa/q4 ở mức bit gần với box. Chưa có đường đó thì không được viết "X lần rẻ hơn" như một kết luận.

## 4. Hậu xử lý
- **Box helper.** Mọi nhánh box lấy box helper từ **cùng một nguồn**: decoder ego fine-tune, cap 500 tại head.
- **Cap sau fusion là 500 cho mọi phương án**, giữ 500 box có score cao nhất. Báo cả cap ở head và cap sau fusion.
- **Detection** chấm không đặt ngưỡng score.
- **Tracking.** Quét ngưỡng score đưa vào tracker trong {0.1, 0.2, 0.3, 0.4} trên dev. Chọn **một** ngưỡng cho mỗi phương án dựa trên AMOTA ở dev, rồi khóa ngưỡng đó cho test. Báo cả đường quét.

## 5. Chỉ số và cách báo
- **Mỗi dòng bảng là đúng một file prediction**, từ đó tính mAP (AP@0.5/1/2/4 m), AMOTA, IDS và recall. Lưu sha256 của file prediction ngay cạnh kết quả.
- **Chỉ số chính:** AMOTA và mAP. **Chỉ số phụ:** recall@2 m theo nhóm (UAV0 tự thấy / chỉ đồng đội thấy / không ai thấy rõ).
- **Seed.** Mọi nhánh có train đều chạy 3 seed và báo mean ± std. Tracking cũng chạy cho cả 3 seed, không chọn một seed.
- **Báo theo scene** ở dev (xem mục 1).

## 6. Hình học
- **Warp feature theo pose đầy đủ:** tịnh tiến cộng xoay yaw.
- **Kiểm thử bắt buộc trước khi dùng.** Vẽ heatmap tâm box GT ở t−1, warp sang t, rồi so với tâm GT tại t. Với xe gần như đứng yên, sai số phải nhỏ hơn 1 ô (0.8 m). Ghi kết quả kiểm thử vào report.

## 7. Đối chứng bắt buộc cho thí nghiệm bất đồng bộ
1. Ego fine-tune: cùng dữ liệu, cùng lịch train.
2. Late fusion box, không bù trễ.
3. **Gộp dựa trên tracker:** box helper được dự báo tới thời điểm t bằng Kalman tốc độ không đổi, dựa trên track của chính helper, rồi mới gộp. Đây là cách xử lý trễ mà CERF mô tả trong paper.
4. Feature trung bình (kiến trúc cũ), có và không có train với trễ.
5. Ego cộng helper-mean, có và không có kênh tuổi (pilot của Codex).

## 8. Thí nghiệm đầu tiên dưới protocol này (chạy trên dev)
- **Câu hỏi:** giữ **riêng từng nguồn**, mỗi helper kèm tuổi message và cờ có/không có message, thì có vừa giữ được kết quả khi đồng bộ, vừa không tụt khi trễ không?
- **Kiến trúc tối thiểu (per-agent gating):**
  - Với mỗi nguồn i: w_i = conv1×1([f_i, emb(age_i), mask_i]).
  - Softmax theo i trên từng ô BEV, rồi f = Σ w_i f_i (nguồn bị mất có w_i = 0), rồi đưa qua decoder.
  - Lớp gating khởi tạo để ban đầu bằng gộp trung bình.
- **Train một model duy nhất** với trễ ngẫu nhiên độc lập theo link cộng với mất gói. Model đó được chấm trên **mọi** kịch bản ở mục 2.
- **Chỉ đọc như chẩn đoán.** Nếu không vượt đối chứng 3 (Kalman) và 4 (feature train với trễ) thì **không** thêm mạng phức tạp hơn (GRU/Mamba). Khi đó quay sang làm baseline bù chuyển động cho mạnh trước.
- **Ngoài phạm vi v1:** SB, Mamba, MARL và codec feature học được.

## 9. Việc phải xong trước khi chạy GPU
- [ ] Sửa khởi động lạnh (mục 2) trong `train_fusion.py`, `ft_tools.py`, `box_stress.py`, `feat_exp.py`, `train_separate.py`.
- [ ] Tracker dùng dt = 0.2 s.
- [ ] Codec round-trip cho box (85 bit) và feature (q8) nằm trong đường chấm.
- [ ] Warp có xoay, kèm kiểm thử ở mục 6.
- [ ] Dump lại box helper ở cap 500 từ cùng decoder ego cho mọi nhánh box.
- [ ] Script chấm in mAP + AMOTA + sha256 cho mỗi file prediction và báo theo scene.
- [ ] Codex review diff của các bản sửa trên.
