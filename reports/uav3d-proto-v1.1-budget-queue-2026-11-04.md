# UAV3D protocol v1.1 — ablation Kalman, chọn message theo ngân sách, hàng đợi — 2026-11-04 (tối)

Phạm vi: 3 scene dev (scene_11, 13, 15; 60 frame), CPU, không GPU. Chưa có khoảng tin cậy; mọi điều dưới đây là
giả thuyết cho test set. Đây là phần tiếp theo của `uav3d-proto-box-results-2026-11-04.md`, làm theo góp ý của ChatGPT
và Codex (`codex-review-proto-box-results-2026-11-04.md`).

Bảng đầy đủ (mAP / AMOTA từng ô, sha256 đã đối chiếu): chạy
`python -I swarmbridge/experiments/uav3d_comm/proto/summarize.py runs/uav3d_proto` (ghi `runs/uav3d_proto/summary.json`).

**TRẠNG THÁI (19:30):** AMOTA đã có cho ablation Kalman (đủ seed), pilot ngân sách và hàng đợi (seed 0 cho các policy chính). Các seed còn lại, độ nhạy Kalman và `central_strong` vẫn đang tracking; `summarize.py` in bảng mới nhất.

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

Kiểm tra bão hòa vận tốc (codec 9 bit ±25.6 m/s, `kfw_nor`, giữ mọi thứ khác): mAP khác nhiều nhất 0.001, AMOTA khác
nhiều nhất 0.002 (sync 0.690 = 0.690, fix1 0.633 vs 0.631, fix2 0.529 = 0.529, rand 0.640 vs 0.639, loss0.5 0.634 vs
0.635). Cắt vận tốc ở ±12.8 m/s không ảnh hưởng kết quả; giữ codec 87 bit.

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
4. **Khoảng cách tới chẩn đoán GT nhỏ theo mAP**: ở rand, oracle_full hơn heuristic tốt nhất 0.020 (B=5), 0.025 (B=10),
   0.013 (B=20) mAP; biết trước kênh chỉ thêm 0.005–0.011.

AMOTA (seed 0; gửi đầy đủ 41 kbit: sync 0.690, rand_s0 0.645; ego 0.542):

| policy | sync B=5 | B=10 | B=20 | rand_s0 B=5 | B=10 | B=20 |
|---|---|---|---|---|---|---|
| local_score | 0.595 | 0.659 | 0.694 | 0.583 | 0.630 | 0.644 |
| central_score | 0.599 | 0.655 | 0.699 | 0.580 | 0.633 | 0.648 |
| central_novh | **0.643** | **0.692** | 0.691 | **0.627** | **0.656** | 0.638 |
| central_nove | 0.569 | 0.548 | 0.540 | 0.541 | 0.521 | 0.511 |
| oracle_task | 0.674 | 0.717 | 0.720 | 0.635 | 0.661 | 0.655 |
| oracle_full | 0.674 | 0.717 | 0.720 | 0.656 | 0.672 | 0.663 |

5. **Theo AMOTA, bỏ trùng giữa các helper mạnh hơn rõ**: ở B=10 (24% số bit) novh đạt 0.692 sync, bằng gửi đầy đủ
   (0.690); ở B=5 hơn top-score +0.044. Với tracking, mỗi xe được *một* helper gửi đều đặn có giá trị hơn nhiều bản sao.
6. **Khoảng cách tới chẩn đoán GT lớn hơn theo AMOTA**: +0.025–0.031 (sync), +0.016–0.029 (rand) so với novh. Chẩn đoán GT
   ở B=10–20 còn **vượt gửi đầy đủ** (0.717–0.720 vs 0.690): gửi ít box sai hơn có lợi cho tracker.
7. novelty so với receiver (nove) dưới ego cả theo AMOTA (0.51–0.57).

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
4. **Kênh có tương quan theo thời gian (AR(1), ρ = 0.8) và sender ước lượng tốc độ từ frame trước (`est{a}`, causal qua
   ACK) không giúp gì thêm** so với `adapt{a}` chỉ dùng backlog:

   | R | kênh | cap0.5 | adapt0.5 | adapt0.75 | est0.5 | est0.75 | oracle |
   |---|---|---|---|---|---|---|---|
   | 4 | fade i.i.d. | 0.584 | 0.582 | 0.596 | 0.572 | 0.579 | 0.621 |
   | 4 | AR(1) | 0.582 | 0.584 | 0.599 | 0.582 | 0.596 | 0.623 |
   | 8 | fade i.i.d. | 0.627 | 0.626 | 0.623 | 0.608 | 0.610 | 0.641 |
   | 8 | AR(1) | 0.627 | 0.629 | 0.626 | 0.622 | 0.628 | 0.644 |

   Giải thích: backlog Q đã mang thông tin "kênh vừa chậm", nên adapt tự điều chỉnh; ước lượng một bước còn thêm nhiễu.
   Khoảng 0.02 mAP còn lại tới oracle là nhiễu tốc độ trong frame hiện tại, mà một bộ dự đoán đơn giản không lấy lại được.
   Muốn đòi phần này cần thông tin kênh tức thời (ví dụ CSI trước khi gửi) hoặc cơ chế khác (chia message thành lớp,
   gửi phần quan trọng trước) — đây là câu hỏi thiết kế, không phải câu hỏi học policy.

AMOTA (seed 0, ego 0.542; gửi đầy đủ không giới hạn kênh: sync 0.690):

| policy | R=4 const | R=4 fade | R=4 AR | R=8 const | R=8 fade | R=8 AR |
|---|---|---|---|---|---|---|
| full_fifo | 0.495 | 0.504 | 0.512 | 0.543 | 0.535 | 0.538 |
| full_drop | 0.542 | 0.540 | 0.533 | 0.548 | 0.562 | 0.579 |
| cap0.5 | 0.635 | 0.630 | 0.626 | 0.696 | 0.689 | 0.682 |
| cap0.75 | 0.674 | 0.651 | 0.635 | 0.699 | 0.665 | 0.656 |
| cap1.0 | 0.696 | 0.595 | 0.606 | 0.697 | 0.596 | 0.593 |
| adapt0.75 | 0.674 | 0.650 | 0.654 | 0.699 | 0.679 | 0.677 |
| est0.75 | 0.674 | 0.631 | 0.649 | 0.699 | 0.655 | 0.684 |
| oracle | 0.696 | 0.685 | 0.684 | 0.697 | 0.705 | 0.699 |

AMOTA đi cùng chiều mAP: gửi hết qua FIFO dưới ego; riêng chọn kích thước làm AMOTA chênh khoảng 0.2; oracle hơn heuristic
causal tốt nhất khoảng 0.03 ở R=4, 0.016–0.02 ở R=8.

## 4b. Message lũy tiến có hạn chót, và thứ tự nội dung

`queue_proto.py` policy `prog*`. Trong frame s, sender phát box theo một thứ tự, thành các nhóm 8 box giải mã độc lập
(header 132 bit một lần + 32 bit CRC/sequence mỗi nhóm); hết frame, receiver dùng mọi nhóm đã đến (tuổi 0), phần còn lại
bỏ. Không cần biết tốc độ kênh, không có hàng đợi kéo sang frame sau. Thứ tự:
- `prog`: theo score;
- `prog_novh1`: box mà một helper chỉ số nhỏ hơn cũng có trong message frame trước của nó bị đẩy xuống cuối (causal);
- `prog_lv`: logistic regression (leave-one-scene-out trên 3 scene dev) ước lượng P(box tìm lại xe UAV0 bỏ sót) từ đặc
  trưng causal (score, khoảng cách tới UAV0 / tới helper, tốc độ, trong vùng UAV0, khoảng cách tới detection tự tin của UAV0
  ở frame trước — cần feedback, số helper khác có xe đó ở frame trước);
- `prog_tp`: cùng đặc trưng, nhãn "box trúng bất kỳ xe GT nào trong vùng UAV0" (giá trị gồm cả xác nhận);
- `prog_gt`: chẩn đoán dùng GT — box trúng xe mà UAV0 bỏ sót lên trước (không phải policy).

| thứ tự | mAP R=4 fade (3 seed) | AMOTA R=4 const / fade / AR (seed 0) | mAP R=8 fade | AMOTA R=8 const / fade / AR |
|---|---|---|---|---|
| adapt0.75 (FIFO) | 0.596 | 0.674 / 0.650 / 0.654 | 0.623 | 0.699 / 0.679 / 0.677 |
| prog | 0.614 | 0.691 / 0.672 / 0.668 | 0.639 | 0.701 / 0.703 / 0.693 |
| prog_novh1 | 0.582 | 0.623 / 0.631 / 0.626 | 0.628 | 0.695 / 0.680 / 0.680 |
| prog_lv | 0.552 | 0.603 / 0.602 / 0.597 | 0.610 | 0.683 / 0.670 / 0.671 |
| prog_tp | 0.606 | 0.681 / 0.661 / 0.654 | 0.639 | 0.704 / 0.700 / 0.692 |
| prog_gt (GT) | **0.638** | **0.735 / 0.716 / 0.709** | **0.652** | 0.704 / 0.713 / 0.709 |
| oracle (FIFO, biết tốc độ) | 0.621 | 0.696 / 0.685 / 0.684 | 0.641 | 0.697 / 0.705 / 0.699 |

Đọc:
1. **Message lũy tiến giải quyết bài toán "gửi bao nhiêu" mà không cần học hay dự đoán kênh**: prog gần bằng oracle biết
   tốc độ (R=8: 0.703 vs 0.705; R=4: 0.672 vs 0.685 AMOTA). Kỹ thuật này không mới (embedded/scalable coding, deadline
   scheduling); ý nghĩa ở đây là **một scheduler học kích thước message không có đất trong mô hình này**.
2. **Còn lại là "gửi cái gì trước"**: thứ tự theo GT hơn thứ tự theo score +0.044 AMOTA (R=4 fade) và +0.010 (R=8 fade);
   ở R=4 kênh hằng, chỉ 4 kbit/link mà đạt 0.735, vượt gửi đầy đủ 41 kbit (0.690).
3. **Hai bộ ước lượng causal đơn giản không lấy được phần này, còn làm tệ hơn score**: lv (độ chính xác ở top chỉ 20–24%
   cho nhãn "xe UAV0 bỏ sót") ưu tiên nhầm nhiều box sai; tp phân loại tốt hơn score (độ chính xác 0.82–0.85 vs 0.79–0.82)
   nhưng không đổi thành AMOTA tốt hơn. Bỏ trùng giữa helper theo message frame trước (novh1) cũng tệ hơn score.
   Phần dư địa của GT nằm ở thông tin "UAV0 đang bỏ sót xe nào ngay bây giờ", mà sender không quan sát được trực tiếp.

## 3b. Sender biết receiver chắc chắn đến đâu (feedback)

`budget_proto.py` policy `central_strong{τ}`: bỏ box trùng (≤ 2 m) với detection của UAV0 có score ≥ τ ở frame trước —
không cần xác nhận cho cái receiver đã chắc; vẫn xác nhận các detection yếu. mAP (rand, 3 seed):

| policy | B=5 | B=10 | B=20 |
|---|---|---|---|
| central_score | 0.544 | 0.579 | 0.604 |
| central_novh | 0.565 | 0.582 | 0.579 |
| central_strong0.6 | 0.549 | 0.522 | 0.494 |
| central_strong0.8 | 0.557 | 0.577 | 0.566 |
| central_novh_strong0.8 | 0.561 | 0.555 | 0.551 |

Bỏ cả những xe receiver đã tự tin vẫn hại mAP ở ngân sách vừa và rộng. AMOTA: *(đang chạy, chưa có trong bản này)*.

## 5. Trả lời ba câu hỏi của ChatGPT (tạm thời, dev)

Ba câu hỏi: vấn đề này chưa được giải quyết ở đâu; vì sao baseline tốt nhất chưa đủ; thí nghiệm nào có thể chứng minh
giả thuyết sai. Trả lời dưới đây chỉ dựa trên 3 scene dev, chưa có khoảng tin cậy.

**(1) Chưa được giải quyết ở đâu?**
- Bù trễ ở receiver (How2comm, CoBEVFlow, CERF) không lấy lại được giá trị: ở trễ 2 frame Kalman tốt nhất chỉ giữ
  28–33% lợi ích mAP và **0%** lợi ích AMOTA. Phần mất này nằm ở phía gửi (gửi gì, khi nào), không ở phía nhận.
- Giả định "chỉ gửi cái mới / cái receiver chưa có" (Where2comm-style, novelty, AoI value) **sai** cho tracking trong
  pipeline này: giá trị chủ yếu là xác nhận (65 vs 7.5 xe/message); bỏ cái receiver đã thấy kéo AMOTA xuống dưới ego;
  ngay cả bỏ cái receiver đã *chắc chắn* cũng hại mAP. Bỏ trùng giữa các helper thì ngược lại có ích cho tracking khi
  ngân sách chặt. Chưa thấy công trình nào đo đánh đổi xác nhận/mới theo tracking loss — đây là khoảng trống hẹp nhưng có
  số liệu.

**(2) Vì sao baseline tốt nhất chưa đủ?**
- Về *kích thước/thời điểm*: baseline đơn giản **đã đủ**. Message lũy tiến có hạn chót gần bằng oracle biết kênh. Không
  nên đặt đóng góp vào "RL chọn bao nhiêu bit / khi nào gửi" trong mô hình kênh này.
- Về *nội dung/thứ tự*: còn dư địa đo được, +0.044 AMOTA ở băng thông chặt (R=4/link) và +0.025–0.031 theo ngân sách
  chung (B=5–10 kbit), so với heuristic tốt nhất. Hai bộ ước lượng causal đơn giản (logistic, LOSO) **không** lấy được, còn
  tệ hơn score. Nguồn dư địa là "UAV0 đang bỏ sót xe nào lúc này", sender không thấy trực tiếp.
- Vì vậy câu hỏi nghiên cứu còn đứng được là: **sender có thể suy ra (belief) điều receiver đang bỏ sót đủ tốt để chọn
  thứ tự gửi không**, từ thông tin có thật (feedback nén của receiver, lịch sử track chung, hình học che khuất), và giá
  trị đó có đo bằng tracking loss không. Đây gần với hướng "sender replica" ở CoPerception trước đây (replica lấy gần hết
  lợi ích của ideal). Nó **chưa** đòi hỏi MARL: một bộ ước lượng giá trị có giám sát là bước kế tiếp đúng (Codex).

**(3) Thí nghiệm nào có thể chứng minh sai?**
- Đã bác: "scheduler học kích thước có ích" (lũy tiến ≈ oracle); "novelty so với receiver là giá trị đúng" (dưới ego);
  "ước lượng tốc độ kênh một bước giúp" (không).
- Phép thử kế tiếp có thể bác hướng còn lại: cho sender **feedback giàu nhất có thể nhưng vẫn causal** (toàn bộ detection
  của UAV0 ở frame trước kèm score, tính bit thật) và một bộ ước lượng mạnh hơn (gradient boosting / MLP nhỏ, train trên
  scene train — cần dump detection của helper ở 17 scene train, việc này cần GPU nên chờ Codex duyệt). Nếu bộ đó vẫn không
  vượt thứ tự theo score quá +0.01 AMOTA ở R=4, thì dư địa của GT là không quan sát được, và hướng "arrival-aware semantic
  value" trên UAV3D không đủ cho một thuật toán mới → chuyển trọng tâm sang biểu diễn message (track-state có vận tốc đã
  cho +0.07 AMOTA) hoặc bối cảnh khó hơn (ít chồng lấn, nhiều UAV, kênh thật).
- Mọi kết luận trên phải lặp lại trên test (full val) sau khi khóa protocol, với khoảng tin cậy theo scene.

## 6. Hạn chế

- 3 scene, 60 frame, một bộ detector. Dev đã được dùng để chọn rất nhiều thứ trong ngày (ngưỡng, policy) → rủi ro
  overfit dev; test là bắt buộc trước mọi tuyên bố.
- Các kênh là mô hình tổng hợp (hằng / log-normal / AR(1)), độc lập giữa các link; chưa có nhiễu giao thoa, chưa có trace
  thật, chưa có chi phí feedback trong ngân sách (chỉ báo riêng).
- "Oracle"/chẩn đoán GT là best-found một bước, không phải cận trên.
- Bộ ước lượng học được chỉ train trên 2 scene (LOSO) — quá ít dữ liệu để kết luận "học không được"; kết luận đúng là
  "ước lượng tuyến tính đơn giản trên đặc trưng này không được".

## 7. File

- Code: `swarmbridge/experiments/uav3d_comm/proto/{box_proto,budget_proto,queue_proto,value_curve,kalman_sens,summarize}.py`;
  tracking: `scripts/uav3d_proto/track_worker.py` (hàng đợi chung, WSL + Windows).
- Kết quả: `runs/uav3d_proto/{box,budget,queue,ksens,value}/` (JSON + metrics tracking; file dự đoán/track lớn không lên
  git, tái tạo tất định — mọi lần chạy lại trong ngày đều ra sha256 y hệt).
