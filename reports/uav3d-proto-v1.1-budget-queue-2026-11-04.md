# UAV3D protocol v1.1 — ablation Kalman, chọn message theo ngân sách, hàng đợi — 2026-11-04 (tối)

Phạm vi: 3 scene dev (scene_11, 13, 15; 60 frame), CPU, không GPU. Chưa có khoảng tin cậy; mọi điều dưới đây là
giả thuyết cho test set. Đây là phần tiếp theo của `uav3d-proto-box-results-2026-11-04.md`, làm theo góp ý của ChatGPT
và Codex (`codex-review-proto-box-results-2026-11-04.md`).

Bảng đầy đủ (mAP / AMOTA từng ô, sha256 đã đối chiếu): chạy
`python -I swarmbridge/experiments/uav3d_comm/proto/summarize.py runs/uav3d_proto` (ghi `runs/uav3d_proto/summary.json`).

**TRẠNG THÁI: bản nháp, AMOTA của pilot ngân sách và hàng đợi đang chạy; sẽ cập nhật.**

## 0. Sửa lỗi theo review

- `kbit_rx_mean` cũ đếm lại packet mỗi frame nó được dùng từ cache. Nay tách ba loại: `tx` (8 × byte đã đóng gói của
  mọi packet đã phát, kể cả mất), `rx_arrived` (mỗi packet đếm một lần tại frame đến), `used` (= số cũ). Ví dụ kf_nor
  loss0.5 seed 0: tx 41.1, rx_arrived 17.9, used 33.7 kbit/frame.
- Nhánh `kf` **không** truyền track ID/covariance: đúng tên là "detection đã lọc Kalman kèm vận tốc" (KF-PV).
- Protocol nâng lên v1.1 (87 bit/box, header 132 bit, mục 10).
- Chạy lại box_proto: 61 file dự đoán cũ ra **sha256 y hệt**.
- Thống kê kênh mới xác nhận điều Codex nói: ở loss0.5, 47% message được dùng là bản cũ đọc lại từ cache (tuổi 1–3),
  16% link-frame không có message. "Mất gói" vì vậy không chỉ là "ít UAV hơn".

## 1. Ablation Kalman: vị trí dự báo hay vận tốc?

Cùng tracker phía gửi, cùng message, codec, ledger, cap. Chỉ khác: (P) dự báo vị trí theo tuổi, (V) giữ vận tốc khi đưa
vào tracker bên nhận. mAP không phụ thuộc V nên KF-0 = KF-V và KF-P = KF-PV về mAP.

AMOTA (ego 0.542):

| Kênh | KF-0 | KF-V | KF-P | KF-PV |
|---|---|---|---|---|
| sync | 0.622 | 0.690 | 0.622 | 0.690 |
| fix1 | 0.580 | **0.650** | 0.560 | 0.631 |
| fix2 | 0.413 | 0.431 | 0.459 | **0.529** |
| rand (3 seed) | 0.536 | 0.580 | 0.567 | **0.639** |
| loss0.5 (3 seed) | 0.519 | 0.563 | 0.574 | **0.635** |

mAP: KF-0/V sync 0.648, fix1 0.550, fix2 0.418, rand 0.575; KF-P/PV 0.648, 0.594, 0.519, 0.611.

Đọc:
1. **Vận tốc cho tracker** đáng +0.07 AMOTA ngay cả khi không trễ (0.622 → 0.690). Đây là lợi ích của *nội dung* message
   (có trạng thái chuyển động), không liên quan bù trễ.
2. **Dự báo vị trí** giúp mAP ở mọi mức trễ, nhưng với tracking thì phụ thuộc: ở trễ cố định 1 frame nó **làm hại**
   (KF-V 0.650 > KF-PV 0.631; KF-0 0.580 > KF-P 0.560); ở trễ 2 frame hoặc trễ/mất gói thay đổi thất thường thì có ích.
   Giả thuyết: khi mọi message trễ đúng 1 frame, vị trí cũ tạo một quỹ đạo trễ đều, tracker vẫn ghép được; dự báo
   bằng vận tốc nhiễu lại làm vị trí nhảy. Khi tuổi thay đổi giữa các frame, vị trí cũ nhảy lùi/tiến, dự báo làm mượt.
3. **Hai yếu tố cộng hưởng** ở fix2/rand/loss0.5: PV hơn KF-0 khoảng +0.10 đến +0.12, lớn hơn tổng hai tác dụng riêng.
4. Như Codex cảnh báo, chênh lệch kf vs box ở sync (0.690 vs 0.594) **không** quy hết cho vận tốc được: KF-0 (không
   vận tốc) vẫn 0.622 > box_nor 0.594 — phần còn lại do làm mượt vị trí / chọn track xác nhận / số box.

Kiểm tra bão hòa vận tốc (codec 9 bit ±25.6 m/s, `kfw_nor`): mAP khác nhiều nhất 0.001. AMOTA: *(đang chạy)*.

Độ nhạy Kalman (sigma_a ∈ {1,3,6} × gate ∈ {2,3,5}, `runs/uav3d_proto/ksens/`): mAP frame ≥ 3 ở fix2 nằm trong
0.519–0.568, mặc định 0.561, tốt nhất (sigma_a 6, gate 2) 0.568. Phần giữ được ở trễ 2 frame tăng từ 28% lên khoảng
33% của lợi ích sync — kết luận "Kalman không lấy lại được giá trị" không do chọn tham số dở. Thêm cổng "chỉ dự báo box có
tốc độ ≥ v_min" gần như không đổi mAP. AMOTA: *(đang chạy)*.

## 2. Giá trị của message theo tuổi lúc đến (chẩn đoán dùng GT)

`value_curve.py`, `runs/uav3d_proto/value/value_curve.json`. Với mỗi frame t ≥ 3 và helper j, lấy message mà j gửi ở
t − a, đặt vào t (dự báo Kalman hoặc giữ nguyên), so với GT tại t trong vùng chấm của UAV0. Trung bình mỗi message (KF-PV,
dự báo):

| tuổi a | tìm lại xe ego bỏ sót | trong đó không helper nào khác có | xác nhận xe ego đã thấy | box sai | sai số tâm |
|---|---|---|---|---|---|
| 0 | 7.50 | 1.50 | 65.7 | 50.1 | 0.72 m |
| 1 | 7.67 | 1.35 | 61.8 | 53.5 | 0.81 m |
| 2 | 7.30 | 1.27 | 55.4 | 58.5 | 0.90 m |
| 3 | 6.18 | 1.29 | 46.3 | 60.4 | 0.96 m |

Tỷ lệ tìm lại xe ego bỏ sót theo tốc độ xe (a = 0 → 3): đứng yên 0.47 → 0.41, chậm 0.41 → 0.29, nhanh 0.30 → 0.25.
Giữ nguyên vị trí (không dự báo): đứng yên 0.47 → 0.48, nhanh 0.30 → 0.18.

Đọc:
1. **Phần lớn "giá trị" của một message là xác nhận** (65 xe/message), không phải phát hiện mới (7.5, và chỉ 1.5 là duy nhất
   so với các helper khác). Phần xác nhận giảm 30% sau 3 frame, box sai tăng 20%.
2. **Giá trị giảm theo tuổi phụ thuộc đối tượng**: xe đứng yên gần như không mất giá trị (và dự báo bằng vận tốc nhiễu còn
   làm hại), xe chạy nhanh mất nhanh (dự báo giúp giữ lại một phần).
   → "arrival-time value" nên được ước lượng **theo từng đối tượng**, không theo từng message.

## 3. Chọn message dưới ngân sách bit (kênh trễ ngoại sinh)

`budget_proto.py`. Ngân sách B kbit/frame cho cả 4 helper (gửi đầy đủ KF-PV = 41.1 kbit/frame). Nội dung KF-PV, nhận
bằng kf_nor. Policy: random / top-score theo từng helper (local, B/4 mỗi helper) hoặc trung tâm (central, thấy ứng viên
của cả 4 helper); novh = bỏ box trùng (≤ 2 m) box đã chọn của helper khác; nove = bỏ box trùng detection tự tin của
UAV0 ở frame trước (cần feedback ~3.8 kbit/frame, không tính vào B); novhe = cả hai. Hai chẩn đoán dùng GT (không phải
policy, không phải cận trên): oracle_task (biết GT + detection UAV0 tại frame gửi), oracle_full (thêm: biết kênh thực tế).

mAP (rand = trung bình 3 seed):

| policy | sync B=5 | B=10 | B=20 | rand B=5 | B=10 | B=20 |
|---|---|---|---|---|---|---|
| local_rand | 0.519 | 0.535 | 0.577 | 0.513 | 0.522 | 0.553 |
| local_score | 0.555 | 0.604 | 0.641 | 0.542 | 0.579 | 0.603 |
| central_score | 0.557 | 0.605 | 0.641 | 0.544 | 0.579 | 0.604 |
| central_novh | **0.585** | **0.624** | 0.630 | **0.565** | **0.582** | 0.579 |
| central_nove | 0.515 | 0.500 | 0.494 | 0.509 | 0.498 | 0.495 |
| oracle_task | 0.602 | 0.637 | 0.657 | 0.574 | 0.601 | 0.612 |
| oracle_full | 0.602 | 0.637 | 0.657 | 0.585 | 0.607 | 0.617 |

Gửi đầy đủ (41 kbit): sync 0.648, rand 0.611. Ego 0.512.

Đọc:
1. **Top-score ở 20 kbit (46% số bit) gần bằng gửi đầy đủ** (sync 0.641 vs 0.648).
2. **Bỏ trùng giữa các helper** có ích khi ngân sách chặt (B=5: +0.028 sync), nhưng hại khi ngân sách rộng (B=20) vì bỏ
   mất phần xác nhận.
3. **Bỏ những gì UAV0 đã thấy (novelty so với receiver) làm mAP dưới cả ego.** Khớp mục 2: giá trị chủ yếu là xác nhận.
   Đây là phản ví dụ trực tiếp cho giả định "chỉ gửi thông tin mới" trong nhiều paper prior-art.
4. **Khoảng cách tới chẩn đoán GT nhỏ**: ở rand, oracle_full hơn heuristic tốt nhất 0.020 (B=5), 0.025 (B=10),
   0.013 (B=20) mAP; biết trước kênh chỉ thêm 0.005–0.011. Theo mAP, chọn *nội dung* ở tuổi ngoại sinh còn ít dư địa.
   AMOTA: *(đang chạy)*.

## 4. Hàng đợi: kích thước message quyết định thời điểm đến

`queue_proto.py`. Mỗi link helper → UAV0 có tốc độ R kbit/frame (R = 4 hoặc 8; gửi đầy đủ cần khoảng 10.8 kbit/packet),
hằng hoặc fading log-normal (σ = 0.5, 3 seed). FIFO; "drop" = gói mới hủy gói đang chờ. Nội dung KF-PV theo score.
Policy kích thước: full (gửi hết), cap{c} (≤ c·R), adapt{a} (sender biết backlog Q của mình và tốc độ trung bình:
gửi a·R − Q), oracle (biết tốc độ thực của frame hiện tại: R(s) − Q).

mAP, fading (trung bình 3 seed):

| policy | R=4 | tuổi TB | R=8 | tuổi TB |
|---|---|---|---|---|
| full_fifo | 0.466 | 2.4 | 0.493 | 1.9 |
| full_drop | 0.507 | 0.9 (≥ 93% link-frame không có message) | 0.551 | 1.0 |
| cap0.5 | 0.584 | 0.14 | 0.627 | 0.14 |
| cap0.75 | 0.599 | 0.52 | 0.621 | 0.55 |
| cap1.0 | 0.575 | 1.31 | 0.583 | 1.31 |
| adapt0.5 | 0.582 | 0.12 | 0.626 | 0.12 |
| adapt0.75 | 0.596 | 0.37 | 0.623 | 0.38 |
| adapt1.0 | 0.594 | 0.61 | 0.614 | 0.61 |
| oracle | **0.621** | 0 | **0.641** | 0 |

Kênh hằng: cap1.0 = adapt1.0 = oracle (0.634 ở R=4, 0.643 ở R=8); full_fifo 0.454 / 0.478.

Đọc:
1. **Gửi hết qua hàng đợi FIFO cho mAP dưới cả ego** (0.45–0.50 < 0.512): message đến khi đã quá 3 frame hoặc tắc.
   Đây là hiệu ứng đã biết trong lý thuyết AoI, nhưng ở đây nó đo bằng chất lượng perception thật: sai kích thước message
   làm mất toàn bộ lợi ích cộng tác.
2. **Kích thước đúng quan trọng hơn chọn box nào**: chênh lệch giữa kích thước tốt và tệ là 0.13–0.15 mAP, lớn hơn nhiều
   so với chênh lệch giữa các policy chọn nội dung ở mục 3 (≤ 0.03).
3. **Với kênh fading, biết tốc độ hiện tại đáng 0.015–0.022 mAP** so với heuristic causal tốt nhất (cap0.75 ở R=4,
   cap0.5 ở R=8). Heuristic tốt nhất đổi theo R → một policy cố định không tối ưu cho mọi điều kiện kênh.
   AMOTA: *(đang chạy)*.

## 5. Trả lời ba câu hỏi của ChatGPT (tạm thời, dev)

*(sẽ viết sau khi có AMOTA)*
