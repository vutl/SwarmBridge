# SwarmBridge hướng A — Cổng G-L2: UAV có đoán được mode nào hợp với ảnh nào không?

Ngày 10/11/2026. Đây là phép thử **offline** (feasibility gate), không phải nhánh RL mới.
- Không train PPO, SAC hay MARL.
- Không dựng simulator mới, không đụng relay, CARLA, UAV3D hay Active Recovery.
- A5, A5.5, L1 giữ nguyên.
- Phần kiểm toán L1 ở `reports/uav-semcom-a6-l1-audit.md`; phần đối chiếu tài liệu ở `reports/novelty-audit-g-l2-2026-11-10.md`.

## 0. Bài toán, giả thuyết và quy tắc định trước (commit trước mọi phép đo của G-L2)

### 0.1 Bài toán

Mỗi khung ảnh được nén bằng **một mode toàn ảnh**: AVIF, mbt2018-mean, JPEG2000, hoặc mode sinh ảnh. Trạm mặt đất chạy detector A5 (YOLO11s).

L1 cho thấy:
- policy theo tác vụ chủ yếu chọn một mode cố định;
- một oracle biết kết quả detector thật của từng ảnh ở từng mode thì hơn mode cố định khoảng 2 điểm recall ở cùng bit.

Oracle đó lạc quan, vì hai lẽ:
1. nó lấy max trên kết quả detector có nhiễu;
2. đối chứng chỉ là mode cố định, chưa có trộn mode hay luật biết payload.

**Giả thuyết H:** có một bộ dự đoán **nhân quả**, chạy được trên UAV, chọn mode theo nội dung ảnh, cho recall cao hơn **mọi** chính sách không nhìn nội dung ở **cùng bitrate trung bình**, sau khi tính chi phí của chính bộ dự đoán.

### 0.2 Dữ liệu và cách tính

**Ảnh:**
- Bảng mode A5, gồm build "val" (train 304 / val 84 / test 128 ảnh, tách theo video) và build "testdev" (403 ảnh, 49 video).
- 14 mode của bảng. Bit lấy từ bảng; payload này trùng phép mã hoá thật, đã kiểm ở A5.2.

**Kết quả detector theo (ảnh, mode)**, đã đo ở L1 (`runs/uav_semcom/a6l1/fidelity`, sha256 đã ghi):
- TP, FP = số detection − TP, TP vật nhỏ, số vật thể, số vật nhỏ;
- điểm hoạt động conf ≥ 0,25, IoU ≥ 0,5.

**Đánh giá tĩnh theo khung:**
- Mỗi ảnh của pool được gửi đúng một lần, bằng mode mà chính sách chọn.
- recall = Σ TP / Σ số vật thể; bit = trung bình bit mỗi khung.
- **Không mô phỏng mạng.** Có chủ ý: G-L2 chỉ hỏi thông tin nội dung có đáng tiền cho việc chọn mode không. Tương tác với mạng là bước sau, nếu qua cổng.

**Ba mức bit** B1, B2, B3 = bit trung bình trên ảnh train của mode cố định mbt2018-mean_q3, avif_50 và avif_80.

**Giả định chung cho mọi chính sách:** biết payload thật b_i(m) của ảnh ở mọi mode. Đó là giả định mã hoá xong mới chọn, hoặc dự đoán payload chính xác. Vì mọi chính sách cùng biết payload, khác biệt chỉ nằm ở **thông tin về giá trị tác vụ**.

### 0.3 Chính sách so sánh (ở mỗi mức bit, khớp bit trung bình trên pool đang đánh giá)

| Ký hiệu | Chính sách | Nhìn nội dung? |
|---|---|---|
| F | mode cố định tốt nhất có bit ≤ B | không |
| MX | trộn mode ngẫu nhiên không phụ thuộc ảnh: mỗi ảnh nhận mode m với xác suất p_m. Các cặp đỉnh của bao lồi (bit, recall) chọn trên train, p_m chỉnh để khớp bit | không |
| RV | biết payload, không biết giá trị: argmax_m μ_m − λ·b_i(m), với μ_m = TP trung bình của mode m trên train | chỉ qua payload |
| CB | trần bit mỗi ảnh: mode có μ_m cao nhất mà b_i(m) ≤ trần | chỉ qua payload |
| **BL** | **đối chứng mạnh nhất** = max của F, MX, RV, CB ở cùng bit, tính trên chính pool đánh giá. Chọn đối chứng tốt nhất bằng dữ liệu đánh giá là thận trọng: có lợi cho đối chứng | — |
| O | oracle lạc quan: argmax_m TP_i(m) − λ·b_i(m), dùng kết quả detector thật | biết đáp án |
| O_x | oracle kiểm chéo (mục 0.4) | biết đáp án của một phép đo, chấm bằng phép đo kia |

Việc khớp bit dùng λ (hoặc trần, hoặc p_m) để bit trung bình đúng bằng B. Nếu một chính sách đơn định không chạm đúng B, nội suy tuyến tính giữa hai điểm kề nhau. Điểm nội suy đạt được bằng cách ngẫu nhiên hoá giữa hai điểm.

### 0.4 Bước 2: dư địa của oracle (chỉ dùng ảnh train ∪ val; không đụng test, test-dev)

- **H(B) = O(B) − BL(B)**, tính bằng điểm recall, ở ba mức bit, trên train ∪ val (388 ảnh, 46 video).
  - Báo thêm riêng từng phần train và val.
  - Phân theo video: H của từng video, và tỉ lệ đóng góp của video lớn nhất.
  - Phân theo tam phân vị số vật thể và tam phân vị tỉ lệ vật nhỏ.
  - Báo cả FP mỗi khung và giá trị v = TP − 0,5·FP.
- **Phép đo độc lập để kiểm nhiễu detector:**
  - Chạy lại detector trên **ảnh giải mã đã lật ngang**, rồi lật box về. Áp dụng cho 10 mode codec và ảnh gốc, trên ảnh train ∪ val; mode sinh ảnh không đo lại. Codec là tất định, nên đây là phép đo thứ hai trên **cùng** nội dung nén; mọi khác biệt giữa hai phép đo là nhiễu của detector.
  - Oracle kiểm chéo O_x: chọn mode bằng phép đo này, chấm bằng phép đo kia, theo cả hai chiều, lấy trung bình.
  - H_x(B) = O_x(B) − BL(B), với BL cũng được chấm bằng trung bình hai phép đo, trên **menu chỉ gồm mode codec**.
  - Nhãn: **"oracle kiểm chéo bằng phép đo lật ảnh"**, không gọi là "đã khử thiên lệch". Phép lật chỉ bắt được phần nhiễu detector thấy được dưới phép lật.
- **Quy tắc dừng:** đi tiếp sang bước 3 chỉ khi ở **ít nhất một** mức bit có **H(B) ≥ 1,0 điểm và H_x(B) ≥ 0,5 điểm** trên train ∪ val. Nếu không: dừng G-L2 trước khi xây predictor, kết luận FAIL (không có dư địa thực chất).

### 0.5 Bước 3: bộ dự đoán nhân quả (chỉ khi qua 0.4)

**Đặc trưng có thể tính trên UAV** (từ ảnh gốc, trước khi gửi):
- **A — payload:** log b_i(m) của mọi mode và độ phân giải. Payload probe là một phần của A.
- **B — thống kê ảnh rẻ**, trên ảnh xám thu về rộng 480 px:
  - độ lớn gradient (trung bình, độ lệch);
  - mật độ biên;
  - phương sai Laplacian;
  - năng lượng phổ ở 3 dải tần (FFT);
  - entropy texture cục bộ;
  - độ lệch màu.
- **C — detector nhẹ trên UAV:** detector A5 chạy ở độ phân giải thấp (imgsz 480) trên ảnh gốc. Đặc trưng gồm:
  - số detection ở conf ≥ 0,25 và ≥ 0,1;
  - số box nhỏ;
  - phân vị confidence;
  - phân vị diện tích box.

  Đo latency trên CPU và GPU, cùng số tham số. Đây là proxy lạc quan: cùng trọng số với detector ở trạm, nên ghi rõ.

**Mục tiêu học:**
- "abs": TP_i(m) cho từng mode;
- "pair": hiệu D_i(m) = TP_i(m) − TP_i(mbt2018-mean_q3). Chỉ hiệu giữa các mode mới quyết định argmax, nên đây là dự đoán trực tiếp lợi ích giữa các mode.

**Mô hình:**
- ridge, với alpha chọn bằng CV theo video trên train;
- gradient boosting cây nông, tham số cố định: depth 3, 200 vòng, lr 0,05.

**Ứng viên:** 4 bộ đặc trưng (A, A+B, A+C, A+B+C) × 2 mục tiêu × 2 mô hình = 16.

**Chính sách predictor:** argmax_m T̂_i(m) − λ·b_i(m). Với "pair", T̂_i(m) = D̂_i(m); hằng số không đổi argmax.

**Quy trình:**
- Train trên train.
- Trên **val**, ở mỗi mức bit: chọn λ để khớp B, rồi chọn **một** ứng viên có Δ_val = recall − BL(cùng bit) lớn nhất.
- Test và test-dev chỉ dùng một lần cho đánh giá cuối, với ứng viên và λ đã khoá.

**Báo cáo cuối**, trên test và test-dev riêng:
- Δ = recall(predictor) − BL(tại bit thực tế của predictor);
- CI 95% bằng cluster bootstrap theo video (B = 2000, rng [557, mức bit, pool]). Trong mỗi bản bootstrap tính lại cả predictor lẫn BL;
- recall vật nhỏ, FP mỗi khung, AP50 (COCO theo policy, như L1) của predictor và của F / RV / CB gần cùng bit;
- độ đa dạng mode: tỉ lệ ảnh không nhận mode phổ biến nhất;
- tỉ lệ dư địa oracle thu được (O và O_x). Đây chỉ là chỉ số phụ;
- leave-one-video-out;
- chi phí: latency mỗi khung (CPU 1 luồng, CPU mọi luồng, GPU), so với khoảng giữa hai khung của một UAV ở regime ref (1,25 s) và thời gian mã hoá;
- ca thất bại: các ảnh predictor thua oracle nhiều nhất, và đặc điểm chung của chúng.

### 0.6 Bước 4: quyết định (cố định trước)

**PASS** nếu ở ít nhất một mức bit, ứng viên chọn trên val thoả **cả bốn** điều kiện:
1. Δ ≥ +1,0 điểm recall trên test **và** trên test-dev, CI 95% theo video có cận dưới > 0 ở **cả hai** pool. Lặp lại trên hai pool độc lập là chốt chặn chống chọn may mắn giữa 3 mức bit.
2. Bỏ từng video một (leave-one-video-out), Δ vẫn > 0 trên cả hai pool.
3. Chi phí: đặc trưng và predictor ≤ 100 ms mỗi khung trên CPU, tức ≤ 8% khoảng 1,25 s giữa hai khung, sau khi trừ phần đã có sẵn (payload của mode đã mã hoá).
4. Predictor thực sự đổi mode theo ảnh: ≥ 20% ảnh nhận mode khác mode phổ biến nhất của nó.

**INCONCLUSIVE:** điều kiện 1 đạt trên một pool; hoặc cả hai pool có Δ ≥ +1 nhưng có CI chứa 0; hoặc điều kiện 1 đạt nhưng 2 hoặc 3 không đạt.

**FAIL:** các trường hợp còn lại, kể cả khi dừng ở bước 2.

**Hệ quả:**
- **PASS:** đề xuất đưa giá trị dự đoán vào quan sát của PPO trong một ma trận kịch bản khoá trước (ref, K5, lệch PHY, ưu tiên nhiệm vụ khác nhau, deadline không đồng nhất). Bước đó cần người dùng đồng ý; không chạy trong đợt này.
- **FAIL:** đề xuất dừng nhánh chọn mode toàn ảnh theo reward tác vụ.
- **INCONCLUSIVE:** báo nguyên nhân và chỉ ra phép đo nào sẽ phân xử.

### 0.7 Không làm trong đợt này
Train PPO, SAC hay MARL; simulator mới; relay; CARLA; UAV3D; Active Recovery; ROI encoder; sửa báo cáo cũ.
