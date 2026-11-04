# Rà soát kế hoạch UAV3D mini — 04/11/2026

Phạm vi: xem report và code audit hiện có, metadata mini, mã collaborative UAV3D, paper UAV3D/DHD/CodeFilling và abstract GOST. Chỉ đọc và phân tích trên CPU; chưa cài môi trường, tải checkpoint, chạy detector hoặc đo GPU. GOST chưa có PDF toàn văn trong workspace, nên chưa thể kết luận chi tiết phương pháp hay novelty dựa trên nó.

## Quyết định

Theo hướng ChatGPT: xuất phát từ nhiệm vụ nhiều UAV thực sự cung cấp bằng chứng bổ sung, rồi kiểm tra giá trị quyết định truyền ở hiện tại và tương lai. Bước đầu dùng UAV3D mini cho hợp tác detection/tracking với trajectory cố định; active sensing/movement cần môi trường tương tác về sau. SB/Mamba/MARL là ứng viên cần chứng minh, không là tiền đề.

Plan Claude là feasibility pilot hợp lý, với điều chỉnh quan trọng: phải kiểm tra quyền truy cập thông tin, chọn đúng split/checkpoint, so bitrate sau codec và dùng detector giả lập như kiểm thử cơ chế, không thay thế kết quả detector thực.

## 1. Report mini và hình học

Kết quả của `uav3d_audit` cho thấy **tín hiệu đáng khảo sát**, chưa phải ground truth về visibility hay occlusion:

- `axis_search.py` chọn hệ trục bằng tỷ lệ tâm GT box trùng màu semantic của xe trên `samples[::40][:8]`. Đối chiếu token trong `uav3d_infos_*.pkl`, 8 scene này là `00, 02, 04, 06, 08, 10, 12, 14`; `02` và `12` nằm trong 4 scene validation. Vì vậy đây không phải phép chọn hệ trục hoàn toàn độc lập với val.
- Kiểm tra ảnh overlay `CAMERA_BOTTOM_id_0.png`, `CAMERA_LEFT_id_2.png` cho thấy hệ trục được chọn có vẻ bám hình xe ở mẫu đã vẽ; đây là bằng chứng định tính hữu ích, chưa xác nhận toàn bộ camera/độ sâu/che khuất. Không suy ra 64% tâm không trùng pixel xe là bị che: tâm 3D có thể nằm trên nền/ngoài phần silhouette nhìn thấy, sai số chiếu, vật thể nhỏ và các nguyên nhân khác.
- `visibility.py` dùng tỷ lệ pixel **lớp xe** trong hình chữ nhật bao hình chiếu 8 góc box. Không có instance mask: pixel có thể thuộc xe khác, xe đậu không được gán nhãn hay bối cảnh; xe thật có thể bị thiếu điểm vì nhỏ, box nhiều nền, cắt biên, và điều kiện `MIN_PIX >= 20`. Các ngưỡng 0.15/0.3 là heuristic, không phải xác suất nhìn thấy.
- Tính lại đúng từ các `.npz`: ngưỡng 0.3, trong train có 47,57% object-frame được ít nhất một UAV đánh dấu visible, còn 52,43% không được đánh dấu; trong val là 53,28% được đánh dấu. Tỷ lệ đồng đội được heuristic đánh dấu có thông tin bổ sung trên các object-frame team-visible là 40,29–45,72% train và 36,52–45,98% val, tùy UAV. Đây là thống kê heuristic theo object-frame, không phải mAP tăng hay phần trăm lợi ích causal.
- Tách theo scene, giữ `scene_01` và `scene_11` ngoài lựa chọn phép chiếu/ngưỡng để kiểm tra định tính bước đầu; `scene_02` và `scene_12` đã tham gia chọn trục. Hai scene còn lại rất ít, không đủ chứng minh tổng quát. Kiểm tra thêm camera khác nhau, kích thước/độ xa, trạng thái che khuất và các trường hợp xe liền kề hoặc xe tĩnh không nhãn.
- Pose/segmentation/map/annotation thật là thông tin đặc quyền của mô phỏng. Nếu dùng các thông tin này để quyết định sender nên gửi object nào, tên đúng là **oracle proxy**, trừ khi một hệ thống thực tế có thể thu được đúng thông tin và đã tính cả chi phí. Không dùng GT vị trí/visibility để giả làm đầu ra detector.

## 2. Bảng mini trong paper khác bảng full trong README

Paper UAV3D, Table 15, trang 15, báo cáo collaborative **detection trên val mini**:

| Phương pháp | mAP mini trong paper |
|---|---:|
| Lower-bound | 0.147 |
| V2VNet | 0.265 |
| DiscoNet | 0.218 |
| Upper-bound | 0.349 |

Các số 0.544/0.720 trong README là từ bảng full UAV3D, không phải target mini. Không mặc định checkpoint liên kết trong README là checkpoint đã tạo ra Table 15: cần kiểm tra provenance và cấu hình tác giả. Nếu checkpoint train trên full, mini có thể trùng train hoặc val; phải đối chiếu scene/sample token trước khi đánh giá generalization. Cùng dùng một tập có thể tạo so sánh kỹ thuật nội bộ nhưng không đảm bảo mức độ thiên lệch giống nhau giữa mọi phương pháp.

Paper Table 11 ghi **0.5 Hz**, còn audit ước lượng **0.2 s/frame** từ chuyển động chia vận tốc. Hai thông tin chưa khớp; audit có thể dùng độ lớn vận tốc, trường actor và sampling simulator khác nghĩa với giả định trong paper. Ghi cả hai và kiểm tra nguồn CARLA/AirSim, timestamps, vector vận tốc, đối tượng đủ chuyển động, các trường hợp không đứng yên; chưa kết luận scene đúng 4 giây. 20 frame là điều xác nhận được.

## 3. V2VNet/DiscoNet trong repo UAV3D

- `mmdet3d/models/fusion_models/bevfusion.py`: từng ảnh đi qua camera backbone/neck/view transform; code chia 25 camera thành 5 nhóm và lấy trung bình 5 camera mỗi UAV rồi đưa tensor `[B, 5, C, H, W]` vào mô-đun cộng tác. Lower-bound lấy feature agent 0; upper-bound trung bình feature 5 agent. Đây là tên baseline của tác giả, không phải cận toán học cho mọi phương pháp.
- Config camera có `C=80`, x/y bound `[-102.4,102.4]` với bước 0.8 → lưới ban đầu 256×256, theo code/config; shape cần xác nhận bằng hook khi chạy thật. V2VNet giảm spatial 3 lần stride-2 → dự kiến 80×32×32; DiscoNet giảm 2 lần → 80×64×64.
- Nếu gửi tensor đặc 32 bit/giá trị và không overhead, một payload dự kiến lần lượt **2.621.440 bit (320 KiB)** và **10.485.760 bit (1.280 KiB)** cho mỗi UAV/lần gửi. Đây chỉ là phép quy đổi kích thước tensor, không phải số bit đo trên mạng. FP16/8-bit, topology, broadcast/unicast, số lần truyền, header/CSI/ACK/FEC đều thay đổi chi phí. V2VNet có 3 vòng cập nhật, nhưng code dùng `x[b,j]` làm neighbor input trong mỗi vòng; không mặc định rằng cần truyền 3 lần khi một bản sao có thể được cache.
- Trong `V2VNet.py`, `compress_level` được lưu nhưng không thấy áp dụng codec/quantization trong forward. DiscoNet forward cũng downsample/upsample và fusion trọng số pixel; không thấy entropy coding, bitstream hay phép đo bit. Giảm kích thước feature bằng conv không đồng nghĩa đã có digital codec và payload bit thật.
- Ở cả hai forward, có `warp_feat = x[b, j]`, không thấy warp giữa agents ngay tại mô-đun đó. Cần kiểm tra phép căn chỉnh tọa độ đã thực hiện trước lúc BEV được đưa vào fusion hay chưa; không kết luận sai hình học chỉ từ đoạn này.
- Chạy full-fusion một lần rồi lưu detector outputs **không đủ** để đánh giá feature bị nén/chọn: phải lưu feature trước fusion và chạy lại phần fusion/head tương ứng cho mỗi cấu hình hoặc giải thích xấp xỉ. Khi thay bitrate/noise, cần đo lại accuracy sau giải mã. Các detector khác checkpoint có thể làm sai lệch so sánh representation; cố gắng dùng cùng encoder/splits/metrics, và ghi rõ khác biệt không tránh được.

## 4. Paper gần nhất và giới hạn novelty

| Công trình | Điều đã làm | Hệ quả cho plan |
|---|---|---|
| DHD (NeurIPS 2024) | Collaborative motion prediction và chọn feature bằng chênh lệch cục bộ (SISW); paper dùng transmission ratio, đánh giá trajectory/prediction | Là đối chứng gần; cần kiểm tra code và xác định cost bit thật khi so, không coi chỉ thêm thời gian/tracking là mới. Dataset và task không giống hệt UAV3D. |
| CodeFilling (CVPR 2024) | Trao đổi thông điệp codebook và chọn vùng theo information supply/demand cho 3D detection | Ý tưởng “gửi vùng đồng đội thiếu” đã có prior art. Nếu adaptation lên UAV3D không khả thi, nêu rõ thay vì giả vờ đã so. |
| GOST (IEEE TMC 2026) | Abstract nêu semantic twinning, stale data, collaborative tracking và MADDPG trong satellite–UAV | Chưa có full text để kiểm tra state/action/reward hay xác nhận research gap. [DOI](https://doi.org/10.1109/TMC.2026.3700322). |

## 5. Kế hoạch theo cổng

1. **Tiền kiểm:** xác nhận convention chiếu trên scene chưa dùng để chọn trục; làm rõ temporal sampling và scene overlap của checkpoint; xử lý đường dẫn trong metadata/pkl và working directory. Liệt kê GPU memory, version CUDA/WSL và phụ thuộc trước khi thay môi trường.
2. **Smoke test checkpoint:** Claude thử 1 sample lower-bound trên GPU lock, ghi peak VRAM, thời gian và lỗi; sau đó một nhóm nhỏ V2VNet/DiscoNet. Chỉ khi chạy ổn mới đánh giá đủ 80 sample. Baseline paper mini ở Table 15 là tham khảo, không yêu cầu trùng khi provenance khác. Thời gian 10–30 phút hoặc vừa 8 GB là giả thuyết cần đo.
3. **Thí nghiệm representation:** đầu ra detector thực của mỗi UAV, đầu ra hợp nhất và feature trước fusion; so no communication, gửi track/object states (kèm confidence/covariance/ID theo định nghĩa khả dụng), embedding nhỏ và intermediate feature nén. Cùng task, split, điều kiện kênh, tổng budget, overhead và chất lượng đầu ra. Đo recall theo khoảng cách/che khuất, mAP; nếu tracking, thêm IDF1/HOTA/ID switch/lost tracks và độ trễ. Oracle GT/seg masks báo riêng.
4. **Cổng tuần tự:** giữ observation trajectory cố định, can thiệp gửi/không gửi từ cùng tracker state, cùng noise và policy tiếp diễn để đo ảnh hưởng nhiều frame. So random, confidence/age, greedy và bandit trước RL. 20 frame ngắn nhưng không chứng minh RL vô ích; RL trên trạng thái tracker/queue có thể hợp lệ. Chỉ cần simulator khi action thay đổi vị trí/cảm biến hoặc môi trường tương lai không có trong dữ liệu.
5. **Fallback:** detector giả lập GT + mô hình visibility/nhiễu chỉ kiểm tra giả thuyết điều khiển có thể hoạt động trong mô phỏng; không đưa làm evidence về detector thực hoặc kết luận representation win. Báo assumptions, oracle và OOD noise; gate chính cần prediction thực.

Không đặt lịch 1 tuần hay 10–30 phút GPU như cam kết; benchmark bằng một sample và extrapolate. Mỗi bước log thông tin nguồn checkpoint, seed, số frame, thời gian, GPU peak, thông tin chi phí và quyền truy cập của sender/receiver. Khi quota hoặc dependency cản trở, tiếp tục các bước CPU độc lập.
