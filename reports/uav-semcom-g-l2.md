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

## 1. Code, cách chạy, chi phí

| Thành phần | File |
|---|---|
| Dữ liệu, đường cong khớp bit, đối chứng F/MX/RV/CB | `swarmbridge/experiments/uav_semcom/g2_common.py` |
| Bước 2 | `g2_headroom.py` |
| Đo lại bằng ảnh lật (GPU) | `g2_flip.py` |
| Đặc trưng trên UAV (GPU + đo CPU) | `g2_features.py` |
| Bước 3–4 | `g2_predict.py` |
| Hàng đợi | `scripts/uav_semcom/run_g2_flip.sh`, `run_g2_features.sh` |
| Kết quả; bảng đầy đủ ở `summary.md` | `runs/uav_semcom/g_l2/` |

- **Thời gian chạy:**
  - đo lật: 388 ảnh × 13 mode, khoảng 1,5 giờ GPU, chậm vì tranh CPU với kiểm toán L1;
  - đặc trưng: dưới 20 phút;
  - bước 3–4: 1 phút.
- **Kiểm tra trong lúc chạy:** payload của mọi (ảnh, mode) trong lần đo lật trùng từng byte với L1.
- **Môi trường:** đã cài thêm scikit-learn 1.9.1 vào env `semcom-a`; numpy giữ nguyên.

## 2. Bước 2: dư địa oracle (train ∪ val, điểm recall)

**Mức bit:** B1 = 0,303, B2 = 0,627, B3 = 1,266 Mbit/khung (mbt q3 / avif_50 / avif_80 trên train).

| mức bit | O (lạc quan) | BL (loại) | F | MX | RV | CB | **H** [CI video] | O_x | BL_x | **H_x** [CI video] |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 56,68 | 54,39 (RV) | 53,45 | 53,72 | 54,39 | 54,09 | **+2,29** [1,48; 2,96] | 55,25 | 54,30 | **+0,95** [0,36; 1,42] |
| B2 | 59,46 | 57,63 (RV) | 57,38 | 57,44 | 57,63 | 57,44 | **+1,84** [1,38; 2,28] | 57,85 | 57,64 | **+0,21** [−0,09; 0,51] |
| B3 | 59,83 | 58,71 (RV) | 57,92 | 58,50 | 58,71 | 58,66 | **+1,12** [0,87; 1,42] | 58,14 | 58,53 | **−0,40** [−0,57; −0,23] |

CI theo video là cluster bootstrap B = 300, khám phá. Train và val riêng cho cùng bức tranh (`summary.md`).

**1. Đối chứng mạnh nhất luôn là RV:** biết payload từng ảnh, chỉ dùng giá trị trung bình của mode. Nó hơn mode cố định 0,8–0,9 điểm (B1, B3) và 0,25 điểm (B2). So sánh oracle với mode cố định ở L1 vì thế đã phóng đại dư địa.

**2. Phần lớn dư địa của oracle lạc quan là nhiễu detector.**
- Hai phép đo (ảnh gốc và ảnh lật) gần như trùng ở mức TP: tương quan 0,995.
- Nhưng **hiệu** TP giữa hai mode lặp lại kém:

  | hiệu | tương quan giữa hai phép đo |
  |---|---|
  | mbt q3 − avif_20 | 0,85 |
  | avif_50 − mbt q3 | 0,69 |
  | avif_80 − avif_50 | 0,41 |

- Chọn bằng một phép đo rồi chấm bằng phép kia, dư địa chỉ còn +0,95 / +0,21 / −0,40 điểm. Ở B3 nó âm: chọn theo kết quả đo của từng ảnh còn tệ hơn dùng giá trị trung bình của mode.

**3. Dư địa nằm ở đâu.**
- Phần lớn dư địa (lạc quan) nằm **trong** từng video, tức giữa các khung. Oracle chọn một mode cố định cho mỗi video chỉ thu 14–37% (khám phá, thêm sau tiền đăng ký theo gợi ý của audit tài liệu).
- Không có video nào chiếm đa số: video lớn nhất chiếm 25% / 15% / 8%.
- Dư địa tập trung ở ảnh nhiều vật thể và nhiều vật nhỏ, ở B1.

**Quy tắc 0.4:** chỉ **B1** đạt (H ≥ 1,0 và H_x ≥ 0,5), nên đi tiếp sang bước 3.

## 3. Bước 3: bộ dự đoán

**Đặc trưng** (bảng đủ ở `summary.md`):
- **A:** log payload của 14 mode và số pixel;
- **B:** 13 thống kê ảnh;
- **C:** 15 đặc trưng từ detector A5 chạy ở imgsz 480.

**Chọn trên val (Δ_val, điểm):** không ứng viên nào hơn đối chứng mạnh nhất một cách đáng kể. Tốt nhất là A+C / abs / ridge, +0,04 ở B2.
- Bộ A (chỉ payload) và các mô hình có bộ B kém hơn: −0,15 đến −5,5.
- Bộ C (detector nhẹ) giúp nhiều nhất nhưng vẫn quanh 0.

Ứng viên chọn:

| mức bit | ứng viên | Δ_val |
|---|---|---|
| B1 | A+C / abs / gbm | −0,05 |
| B2 | A+C / abs / ridge | +0,04 |
| B3 | A+C / abs / ridge | −0,10 |

**Đánh giá cuối, một lần** (Δ = recall − BL tại bit thực tế của predictor; CI cluster bootstrap theo video, B = 2000):

| mức bit | pool | Mbit/khung | recall | BL (loại) | **Δ** [CI] | LOVO nhỏ nhất | đa dạng mode |
|---|---|---|---|---|---|---|---|
| B1 | test | 0,267 | 57,06 | 57,79 (RV) | **−0,73** [−1,66; +0,04] | −0,86 | 0,58 |
| B1 | test-dev | 0,214 | 43,38 | 43,55 (CB) | **−0,17** [−1,09; +0,94] | −0,47 | 0,69 |
| B2 | test | 0,563 | 60,70 | 61,10 (RV) | **−0,40** [−0,79; −0,17] | −0,48 | 0,50 |
| B2 | test-dev | 0,544 | 48,58 | 49,92 (RV) | **−1,33** [−2,09; −0,70] | −1,45 | 0,68 |
| B3 | test | 0,986 | 60,56 | 61,17 (RV) | **−0,61** [−1,17; −0,31] | −0,77 | 0,31 |
| B3 | test-dev | 0,757 | 49,23 | 50,91 (RV) | **−1,67** [−2,50; −1,02] | −1,89 | 0,73 |

- **Ở cả 6 ô, predictor kém hơn luật RV** (không nhìn nội dung ngoài payload). 4/6 ô có CI nằm hẳn dưới 0.
- AP50, recall vật nhỏ và FP đi cùng chiều (`summary.md`). AP50 của predictor thấp hơn thành viên RV gần bit nhất 0,1–1,7 điểm, trừ B2 test-dev (+0,1, nhưng predictor dùng nhiều bit hơn khoảng 8%).
- **Chi phí trên UAV** (đo khi máy rảnh; trung vị 20 ảnh):
  - thống kê ảnh (B): 9,6 ms;
  - detector imgsz 480 (C): 35–40 ms CPU, 12 ms GPU (RTX 3070), 9,4 triệu tham số.

  Điều kiện chi phí (≤ 100 ms) **đạt**. Lần đo đầu (3185 ms) bị phồng vì đo lúc 6 tiến trình kiểm toán chiếm CPU; số đó được giữ ở `features_timing_contended.json`. Quyết định không đổi.
- **Ca thất bại:**
  - Predictor tiết kiệm bit bằng cách gửi một số ảnh ở mode rất rẻ (mbt q1, avif_5, jpeg2000_100, mode sinh ảnh). Nó tưởng các ảnh đó "không cần chất lượng", nhưng chúng mất 8–34 TP mỗi ảnh. Bit dồn cho ảnh khác không bù được.
  - Ca tệ nhất là những ảnh nhiều vật thể (ví dụ test-dev 9999979: 218 vật thể, mất 34 TP).
  - Đây đúng là hệ quả của việc **hiệu** giữa các mode khó đoán: chỉ cần đoán sai vài ảnh là mất nhiều hơn phần lợi nhỏ từ các ảnh đoán đúng.

## 4. Quyết định (quy tắc 0.6)

| mức bit | 1. Δ ≥ +1 và CI > 0, hai pool | 2. LOVO > 0 | 3. chi phí ≤ 100 ms | 4. đa dạng ≥ 20% | kết luận |
|---|---|---|---|---|---|
| B1 | không | không | có (40 ms) | có | FAIL |
| B2 | không | không | có | có | FAIL |
| B3 | không | không | có | có | FAIL |

**G-L2: FAIL.**

Predictor có đổi mode theo ảnh (đa dạng 0,31–0,73), nhưng những thay đổi đó làm recall **giảm** so với luật không nhìn nội dung ở cùng bit.

## 5. Diễn giải

1. **Trên menu codec toàn ảnh, với detector A5, gần như không có tín hiệu "mode nào hợp ảnh nào" để học.**
   - Phần dư địa lặp lại được (+0,95 điểm ở bit thấp, khoảng 0 ở bit vừa và cao) nhỏ hơn mục tiêu +1 điểm ngay cả với oracle.
   - Phần còn lại là nhiễu detector: một vài detection đổi qua lại giữa hai mode gần nhau.
2. **Luật đơn giản biết payload (RV) đã lấy gần hết phần lợi theo nội dung** mà payload mang theo. Ảnh phức tạp tốn nhiều bit hơn ở mọi mode, nên RV tự hạ mode cho các ảnh đó.
3. **Kết quả khớp với L1 và phần kiểm toán.** Reward tác vụ chỉ dời điểm hoạt động dọc đường bit–recall, không tạo được lựa chọn mode theo nội dung, vì chính dữ liệu không chứa lựa chọn đó.
4. **Đối chiếu tài liệu** (`reports/novelty-audit-g-l2-2026-11-10.md`):
   - Bộ dự đoán rẻ chọn cấu hình nén theo nội dung đã có: DAO (ACM MM 2022), AdaCompress, SMR, JRD.
   - Nếu G-L2 PASS thì cũng không đủ làm đóng góp chính.
   - Điều các công trình đó thường không có là đối chứng mạnh khớp bit (trộn mode, luật biết payload) và kiểm nhiễu detector. Đây chính là thứ làm G-L2 FAIL. Đó là một phát hiện về phương pháp đánh giá, có giá trị cho một bài kiểm chứng.

**Giới hạn:**
- Detector, menu mode và tập ảnh là của A5. Chỉ có 304 ảnh train; nhiều dữ liệu hơn có thể giúp predictor.
- Phép lật chỉ bắt một loại nhiễu.
- Detector nhẹ trong bộ C dùng cùng trọng số với trạm. Đây là trường hợp lạc quan, và nó vẫn thua.
- Đánh giá tĩnh, không mô phỏng mạng.
- Không thử chọn theo vùng (ROI/QP theo khối); G-L2 không bao trùm hướng đó.

## 6. Khuyến nghị

1. **Dừng nhánh chọn mode codec toàn ảnh theo reward tác vụ.** Không đưa đặc trưng nội dung vào quan sát của PPO. Không có tín hiệu cho PPO học, và luật biết payload đã đủ.
2. **Giữ A5 → A5.5 → L1 → kiểm toán → G-L2 làm một nghiên cứu kiểm chứng.** Có thể dựng outline song song. Thông điệp chính:
   - RL chỉ có ích bền khi kênh nghẽn hoặc lệch PHY;
   - reward tác vụ không tạo thích nghi ngữ nghĩa, mà chỉ chọn một điểm rẻ hơn;
   - oracle theo ảnh phần lớn là nhiễu detector;
   - cách đếm recall dưới Qmin theo PSNR làm phồng ưu thế của policy.
3. **Nếu muốn tiếp tục SemCom thật sự,** phải đổi **thứ được gửi** (biểu diễn), không chọn thêm giữa các codec toàn ảnh. Cần một cổng mới, có audit tài liệu riêng trước khi code, vì AccMPEG, RL-RC-DoT và AdaMask đã chiếm phần lớn hướng ROI. Việc chọn hướng nào thuộc về người dùng.
4. **Không mở RL lớn, relay, SAC/MARL, CARLA, UAV3D** khi chưa có một cơ chế semantic qua được cổng kiểu G-L2.
