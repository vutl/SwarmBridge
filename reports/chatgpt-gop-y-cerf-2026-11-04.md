Tôi đồng ý cho Claude triển khai UAV3D theo hướng này, nhưng cần sửa kế hoạch trước khi chạy. Dùng checkpoint có sẵn, kiểm tra lợi ích cộng tác, rồi so sánh truyền object state với learned features là cách tiếp cận hợp lý. Không cần train lại detector.

Tuy nhiên, tôi vừa tìm được một paper cực kỳ quan trọng mà Claude phải đọc trước Bước 2, nếu không chúng ta có nguy cơ lặp lại chính vòng nghiên cứu với Town3.

## 1. Có một paper gần như đã làm chính thí nghiệm Claude định làm

CERF: Communication-Efficient and Retraining-Free Collaborative Perception — 2026.

[Đọc CERF](https://arxiv.org/abs/2609.00951?utm_source=chatgpt.com) · [Source code CERF](https://github.com/uestchjw/CERF?utm_source=chatgpt.com)

Paper này dùng chính UAV3D, bên cạnh Air-Co-Pred.

Thay vì truyền dense BEV features giữa các UAV, tác giả truyền perception outputs — 3D bounding boxes. Bên nhận dùng Kalman filtering và motion forecasting để xử lý thông tin cũ, rồi chuyển các kết quả nhận được thành một biểu diễn bổ sung cho local BEV features.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://rt.http3.lol\&sz=32)

rt.http3.lol



Kết quả tác giả báo cáo trên UAV3D:

| Phương pháp                  | Bandwidth   | Detection mAP |
| ---------------------------- | ----------- | ------------- |
| Không cộng tác               | 0 Mbps      | 0,544         |
| Late fusion thông thường     | 0,04 Mbps   | 0,610         |
| V2VNet                       | 11,31 Mbps  | 0,647         |
| DiscoNet                     | 54,71 Mbps  | 0,700         |
| CERF — gửi perception output | 0,04 Mbps   | 0,693         |
| Early fusion                 | 164,80 Mbps | 0,720         |

[image](https://www.google.com/s2/favicons?domain=https://rt.http3.lol\&sz=32)

rt.http3.lol

Đây là kết quả trên benchmark trong paper, không phải kết quả Claude đã tái hiện. Các bitrate là theo giả định của tác giả, cần kiểm tra cách tính trước khi đưa vào protocol chung.

Nhưng nó cho thấy một điều rất rõ: đề xuất gửi trạng thái xe thay vì intermediate features đã được nghiên cứu trực tiếp trên UAV3D, và có kết quả rất mạnh.

Vì vậy, nếu Bước 2 chỉ chứng minh “gửi 3D boxes đủ tốt và rẻ hơn V2VNet”, thì kết quả đó không mới. CERF đã đi xa hơn: nó còn kết hợp thông tin từ bounding boxes trở lại feature-level perception.

Đây là paper Claude bắt buộc phải đưa vào nhóm đối thủ, ngang với Where2comm, CodeFilling và CoSDH.

## 2. UAV3D thật sự có tín hiệu cộng tác, nên đáng chuyển sang

Tôi đã kiểm tra repository chính thức của UAV3D.

Trên tập validation đầy đủ, tác giả công bố:

UAV3D collaborative detection

mAP theo README chính thức của UAV3D, không phải kết quả chạy trên mini.

0,450,5250,60,6750,75LowerV2VNetDiscoNetUpper

Như vậy, có khoảng cách thực nghiệm rõ giữa single-agent và collaborative feature fusion: 0,544 → 0,700 mAP.

Đây chính là thứ Town3 hiện tại thiếu. UAV3D cũng có benchmark collaborative tracking, trong đó DiscoNet đạt AMOTA 0,809 so với lower-bound 0,644.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

huiyegit/UAV3D · GitHub



Tôi ủng hộ việc dùng UAV3D. Nhưng chỉ cần tái hiện được một khoảng cách tương tự trên mini là đủ xác nhận pipeline hoạt động, không cần tái tạo toàn bộ bảng benchmark.

## 3. Tôi sẽ sửa kế hoạch Claude ở bốn điểm

### A. Không cần cài cả bốn mô hình ngay

Tài liệu chính thức yêu cầu môi trường khá cũ: Python 3.8, PyTorch 1.10.1, CUDA 11.3, MMCV 1.4.0. Lệnh đánh giá mẫu còn dùng `torchpack dist-run -np 4`, tức cấu hình bốn tiến trình GPU, không thể áp dụng nguyên xi cho RTX 3070 8 GB.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



Tôi muốn Claude thử lower-bound và DiscoNet trước. Sau khi xác nhận được inference và feature extraction trên một GPU mới mở rộng sang V2VNet.

Ngoài ra, repository có sẵn các file detection results phục vụ tracking. Nên kiểm tra xem chúng có thể dùng để chạy pilot ở tầng truyền thông hay không, trước khi cố cài mọi dependency.

### B. Phải tách hai thí nghiệm để tránh so sai

| Thí nghiệm            | Mục tiêu                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| Output communication  | Cùng một detector frozen: không gửi, gửi tất cả boxes, chọn boxes, rồi fusion/tracking              |
| Feature communication | Cùng checkpoint/kiến trúc tương ứng: gửi dense features, compressed features hoặc selected features |

Không được xem việc gửi box vào một late-fusion pipeline và gửi feature vào một DiscoNet đã được huấn luyện riêng là một ablation chỉ thay representation.

Hai hệ thống có cả kiến trúc fusion và training khác nhau.

Chúng ta vẫn có thể so rate–accuracy ở cấp độ hệ thống, nhưng muốn chứng minh representation nào tốt hơn, phải kiểm soát được các yếu tố đó.

### C. Không dùng ground-truth boxes để giả lập detector chính

Phương án fallback của Claude — lấy box thật, giữ những đối tượng nhìn thấy rồi thêm nhiễu — hữu ích để thử scheduler, nhưng nó không tái tạo được các lỗi thật như false positives, missed detections có cấu trúc và ID switches.

Tôi muốn dùng ground truth chỉ cho oracle diagnostic và evaluation, không cho chính sách triển khai.

Quan trọng hơn, 100–200 bit mỗi xe chỉ là giả định. Cần mã hóa thực tế hoặc ít nhất tính riêng position, dimensions, heading, confidence, ID, timestamp, uncertainty và overhead. Nếu không, ta lại mắc lỗi bitrate giống SwarmBridge cũ.

### D. Phải xác minh split của mini

README có link tải mini và checkpoint, nhưng chưa đủ để khẳng định mini không trùng dữ liệu đã dùng huấn luyện.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Tôi muốn Claude kiểm tra scene IDs, metadata train/val và checkpoint provenance. Nếu không xác minh được, mini chỉ nên dùng cho smoke test và phân tích định tính. Kết luận khoa học phải dựa trên các scene held-out phù hợp.

## 4. Sau khi chạy, chúng ta sẽ quyết định nghiên cứu gì?

Điểm thú vị nhất từ CERF không phải detection, mà là sự khác biệt giữa detection và tracking.

Trên UAV3D, CERF đạt detection mAP 0,693 gần DiscoNet 0,700, nhưng tracking AMOTA chỉ 0,744 so với 0,809 của DiscoNet.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://rt.http3.lol\&sz=32)

rt.http3.lol



Đây là một khoảng cách đáng nghiên cứu — mặc dù chưa biết nguyên nhân là representation, association, temporal fusion hay training.

Tôi sẽ thử một giả thuyết cụ thể:

Adaptive Semantic Representation Escalation for Cooperative UAV Tracking

Thay vì luôn gửi bounding boxes hoặc luôn gửi dense features:

- Bình thường UAV truyền compact track states.
- Khi một object bị che khuất, track có nguy cơ mất ID hoặc các UAV đưa ra bằng chứng mâu thuẫn, policy có thể yêu cầu feature giàu thông tin hơn.
- RL học lúc nào nên tăng độ giàu representation và chọn UAV nào cung cấp thông tin bổ sung, dưới ngân sách chung.

Đây là bài toán what representation to communicate, when, and from whom, với hiệu quả được đánh giá qua tracking dài hạn trên mỗi bit, chứ không chỉ mAP của từng frame.

Nhưng cần so trực tiếp với CERF, CoSDH và các phương pháp hybrid/conditional communication, vì bản thân switching giữa output và feature cũng không đảm bảo mới.

## 5. Quyết định đối với kế hoạch một tuần

| Giai đoạn                                   | Quyết định                                    |
| ------------------------------------------- | --------------------------------------------- |
| Kiểm tra mini, split, checkpoint            | Nên làm ngay                                  |
| Chạy lower-bound và DiscoNet frozen         | Nên làm                                       |
| Cache detections và intermediate features   | Nên làm                                       |
| So box transmission và feature transmission | Nên làm, thêm CERF baseline                   |
| Train Mamba / SB                            | Chưa                                          |
| Xây MARL / simulator                        | Chưa                                          |
| Viết proposal chính thức                    | Sau khi thấy kết quả theo bitrate và tracking |

Tôi cũng không xem việc chạy GPU vài phút hay một giờ là vấn đề chính. Điều đáng tránh là viết hàng nghìn dòng code để rồi phát hiện thí nghiệm đã được paper khác làm, hoặc benchmark không thể trả lời research question.

Vì vậy, tôi đồng ý cho Claude bắt đầu Bước 1, nhưng yêu cầu bổ sung CERF vào literature và baseline ngay từ đầu.

Nếu sau Bước 1–2, compact object messages đã gần tối ưu cho detection nhưng còn hụt tracking trong occlusion, burst loss hoặc delayed messages, chúng ta sẽ có một bài toán cụ thể để nghiên cứu tiếp. Nếu gap biến mất dưới cùng điều kiện, ta nên tìm một môi trường tương tác phong phú hơn thay vì ép RL/Mamba vào UAV3D.

Điểm cuối cùng tôi muốn giữ rất rõ: UAV3D là nơi tìm và xác nhận vấn đề. Novelty không nằm ở việc chuyển sang UAV3D, mà phải nằm ở cơ chế phối hợp học được để giải quyết vấn đề mà CERF, DiscoNet và các baseline hiện có chưa giải quyết tốt.