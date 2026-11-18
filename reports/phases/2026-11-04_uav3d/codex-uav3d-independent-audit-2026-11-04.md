# Kiểm toán độc lập UAV3D mini — 04/11/2026

Phạm vi: đọc code, config, log, metadata và output hiện có; tính lại detection bằng CPU. Không sửa code, report hoặc output của Claude; không chạy GPU. Đối chiếu thí nghiệm trong commit `3d062eb8` với artifact thực tế trên máy; HEAD khi bắt đầu kiểm toán là `91fd8009`. Report của Claude đã có mục đính chính lúc 11:50; nội dung dưới đây bổ sung kiểm tra, không thay thế đính chính đó.

## Kết luận ưu tiên

1. **Bảng ghép đôi tái tính đúng trên các artifact hiện có.** Cả 17 file prediction đều tồn tại, có đúng 60 token của ba scene và không có token thừa; mAP tính lại làm tròn 4 chữ số khớp từng dòng, AMOTA khớp `metrics_summary.json`. Script WSL `~/envs/track_final4.sh` ánh xạ đúng tên tracking sang cùng đường dẫn prediction như bảng. Đây là bằng chứng provenance từ script/log, không phải xác thực bằng hash prediction đã ghi lúc tracker chạy; tracker không lưu hash input.
2. **Cần sửa diễn giải chi phí feature trong bảng chính.** `phase5.sh` gọi `ft_tools.py cap` với cached BEV FP16 sau đó chuyển float để decode; đường đi đó không gọi `quant(..., 8)`. Vì vậy mAP 0.693 là của BEV cache FP16, không phải kết quả đã đo sau truyền/giải mã q8. Số khoảng 167.8 Mbit/frame là dự toán 4 helper × 80 × 256 × 256 × 8 bit; nếu giữ payload dense FP16 theo phép tính tương đương là khoảng 335.5 Mbit/frame, chưa tính overhead. Nhánh q8 riêng trong `feat_exp.py` là thí nghiệm khác, không được tự động gán mAP 0.693 cho nhánh đó. Tương tự, 85 bit/box chỉ là ước tính cấu trúc, code fusion dùng giá trị JSON float, chưa có round-trip codec. Chưa có so sánh chất lượng sau mã hóa theo cùng budget.
3. **Độ trễ ở đầu scene bị rút ngắn so với nhãn.** Trong `ft_tools.py cap`, `src = prev.get(src) or src`: ở frame đầu, cả delay1 và delay2 dùng helper cùng frame; ở frame thứ hai, delay2 thực tế chỉ trễ một frame. Với ba scene × 20 frame, delay1 gồm 3/60 frame độ trễ 0; delay2 gồm 3/60 độ trễ 0, 3/60 độ trễ 1 và 54/60 độ trễ 2. `train_fusion.py`, `feat_exp.py`, `box_stress.py` cũng dùng fallback tương tự. Đây là giả định warm-start có lợi, không chứng minh số liệu vô hiệu, nhưng không phải hệ thống khởi động lạnh với đúng độ trễ cho mọi frame. Báo riêng frame hợp lệ hoặc đưa no-message vào giai đoạn chưa có packet.
4. **Thời gian thực chưa xác minh.** Chênh lệch timestamp liên tiếp trung vị là khoảng `2.718e9` đơn vị thô. Đoạn Claude vừa đưa chia `1e6` và gọi kết quả 2718 giây, nhưng không chứng minh timestamp dùng microsecond. `track_run.py` chia `1e9`, tương đương khoảng 2.718 giây nếu đơn vị gốc là nanosecond. Không được kết luận timestamp hỏng chỉ từ phép chia đó. Độ dịch chuyển khoảng 0.94 m/frame phù hợp nhiều tốc độ và khoảng lấy mẫu, không xác định duy nhất 0.2 s. Paper và timestamp cần đối chiếu với nguồn sinh dữ liệu, đồng hồ mô phỏng và dấu thời gian từng sensor. Tracker dùng timestamp khi nhân vận tốc trong `pub_tracker.py`; trong một file feature kiểm tra, vận tốc dự đoán rất gần 0 (giá trị tuyệt đối tối đa khoảng 5.1e-6), nhưng chưa kiểm mọi input/tracker nên chưa khẳng định AMOTA không nhạy với đơn vị này.

## (a) Split và provenance checkpoint

Đối chiếu trực tiếp token của `external/uav3d_ckpt/det_lowerbound` với 400 sample mini: có đúng 60 token, 20 token thuộc mỗi `scene_11`, `scene_13`, `scene_15`. `train_fusion.py` tạo train từ cache trừ token held-out; đây là bằng chứng tách fine-tune theo token nếu hai cache đúng như đã dùng. Sự trùng với file detection được phát hành cho val ủng hộ cách chọn ba scene, nhưng bản thân file prediction không chứng minh checkpoint chưa từng train các scene đó. Cần manifest train/val và provenance của chính checkpoint. Ba scene đã tham gia lựa chọn recipe và ngưỡng: gọi là **dev**, không phải test độc lập.

## (b) Giới hạn box

Config gốc UAV3D `collaborative_perception/bevfusion/configs/nuscenes/det/centerhead/default.yaml` có `post_max_size: 83` và `max_per_img: 500`. `ft_tools.py` đặt `head.test_cfg['post_max_size'] = 500` để xuất lại nhánh ego/feature/box; `phase6.sh` cũng xuất detection mỗi UAV với cap 500 trước late fusion. Nhưng đây là cap ở head **trước** ghép các UAV: tập box late-fusion sau ghép có thể vượt 500, trong khi đầu ra một decoder feature hoặc Poture bị cap tại head. Vì vậy nói tất cả các dòng được giới hạn 500 sau fusion là không chính xác. Đây là lựa chọn quan trọng cho tracking, không chỉ một chi tiết trình bày. Cần công bố cap đầu vào và đầu ra, và báo các cấu hình tracker/cap/score như một đường trade-off.

## (c) Bảng paired mAP / AMOTA

Đọc `runs/uav3d_comm/paired_det_track.json`, tính lại mAP qua `evaldet.evaluate` cho từng prediction trên 60 token; so AMOTA với `runs/uav3d_comm/tracking/<name>/metrics_summary.json`; đối chiếu thêm đường dẫn trong `~/envs/track_final4.sh`. 17/17 dòng khớp các giá trị đã ghi ở độ chính xác bảng, 17/17 không thiếu hoặc thừa token. Chưa chạy lại official TrackingEval hoặc độc lập xác thực source code evaluator trong kiểm toán này.

Kiểm tra mAP tách scene bằng cùng evaluator (một seed/cấu hình cho từng nhánh):

| Scene | Ego | Feature sạch | Feature trễ 2 frame | Noisy-OR sạch | Noisy-OR trễ 2 frame |
|---|---:|---:|---:|---:|---:|
| scene_11 | 0.5396 | 0.7160 | 0.4943 | 0.6765 | 0.4969 |
| scene_13 | 0.4763 | 0.6747 | 0.4223 | 0.6596 | 0.4492 |
| scene_15 | 0.5239 | 0.6854 | 0.4431 | 0.6584 | 0.4482 |

Suy giảm mAP khi trễ xuất hiện ở cả ba scene trong hai nhánh này; chúng vẫn chỉ là ba scene dev, không phải 60 scene độc lập. AP tổng không phải trung bình số học của ba AP scene.

## (d) Trễ, warp và mất gói

`ft_tools.py` dùng BEV helper frame cũ, dịch theo hiệu tịnh tiến ego từ metadata bằng `grid_sample`; Poture lấy box cũ theo tọa độ global rồi trừ vị trí ego hiện tại. Phép warp chỉ thực hiện translation, không xử lý rotation; cần xác minh pose/frame convention và chiều dịch bằng kiểm thử một điểm và tình huống có ego rotation trước khi khẳng định warp đúng cho mọi scene. Cả bốn helper cùng một tuổi, không có queue, ACK, tốc độ truyền hoặc delay độc lập từng link. Phần delay ở đầu scene xem mục ưu tiên.

`feat_exp.py` tạo Bernoulli độc lập theo helper/frame cho p=0.2/0.5 và ba seed rồi ghi `drops.json`. `ft_tools.py` và `box_stress.py` đọc lại cùng map cho các điều kiện tương ứng. Với `0.5_s0`, số frame có 0/1/2/3/4 helper bị drop lần lượt là 7/15/18/16/4 trên 60 frame. Đây là mất toàn bộ message của một helper ở một frame; không mô phỏng bit error, mất burst hoặc phân mảnh packet.

## (e) Bit

`fusion.py` định nghĩa 85 bit/box = 12+12+6+24+8+7+16, và 96 bit header/helper/message; chưa có ID/covariance hay codec round-trip, FEC, đồng bộ, retransmission, ACK. `feat_exp.py` tính q8 167782.4 kbit/frame cho 4 helper, gồm 80×32 bit tham số mỗi helper. Chi phí dense FP16 là 335544.32 kbit/frame. Các số là mô hình ngân sách, không phải byte đo trên mạng. `analyze_comm.py` vẫn cộng 4 header ở policy `none` (0.384 kbit), trong khi bảng nói no-communication = 0: nên thống nhất xem có message rỗng hay không. Các thí nghiệm tính mAP không lượng tử hóa box theo 85 bit. Feature q8 và sparse tồn tại ở nhánh riêng nhưng chưa thay thế đánh giá q8 cùng checkpoint/training trong bảng chính.

## (f) Đối chứng ego

Ego fine-tune là đối chứng thiết yếu: `train_fusion.py` dùng cùng dữ liệu cache train, khởi tạo lowerbound và cùng công thức train cơ bản; `phase5.sh` dùng ego_s2 để tạo detection cho helper, sau đó train decoder Poture. Feature, Poture và late-fusion khác cấu trúc, số tham số, cách gộp, seed và/hoặc decoder; do đó phép so kiểm soát được encoder và dữ liệu, **không cô lập duy nhất loại message**. `phase5.sh` dump helper cho Poture bằng cap mặc định 83, trong khi `phase6.sh` tạo helper cap 500 cho box late fusion; cần công bố vì nguồn box hai nhánh không giống nhau. Không chứng minh checkpoint encoder là độc lập với dev bằng đối chứng này.

## Cần xử lý trước khi mở thử nghiệm mới

- Chốt nguồn timestamp/đơn vị và ghi thời gian theo frame cho đến khi xác minh; kiểm tra độ nhạy tracker nếu velocity không gần 0.
- Quy định warm-start và loại/đánh dấu các frame chưa đủ lịch sử; kiểm tra warp translation/rotation bằng ví dụ hình học có ground truth.
- Đo q8 và box codec thực sự cùng checkpoint, so cùng budget, kèm payload và overhead nhất quán; báo cap trước/sau fusion.
- Khóa protocol trên ba scene dev, rồi đánh giá tập độc lập sau khi xác định provenance checkpoint. Các phát hiện này chưa chứng minh novelty về xử lý độ trễ.
