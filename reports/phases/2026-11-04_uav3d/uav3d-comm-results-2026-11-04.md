# UAV3D: gửi box hay gửi feature? Kết quả chạy thật (2026-11-04)

Code: `swarmbridge/experiments/uav3d_comm/`. Kết quả: `runs/uav3d_comm/`. Plan: `reports/archive/uav3d/uav3d-run-plan-2026-11-04.md`.
Môi trường: WSL `~/envs/uav3d` (py3.8, torch 1.10.1+cu113, mmcv-full 1.4.0, op BEVFusion build bằng nvcc 11.3 + gcc 9;
cần hạ sysroot conda về 2.17 và xóa cờ execstack của libtorch trên Ubuntu 26.04).

## Đính chính (thêm 11:50 sau góp ý của ChatGPT và Codex)
1. **Tin nhắn tóm tắt gửi bạn ghép sai cặp số.** Nhánh "box gộp tốt" (noisy-OR) có mAP 0.665 thì AMOTA là **0.588**, không phải 0.69.
   Con số 0.687 thuộc một cấu hình khác (gộp có trọng số, giới hạn 83). Bảng dưới đây mới là bảng chuẩn:
   mỗi dòng là **một** file prediction, cả mAP lẫn AMOTA chấm trên chính file đó (`runs/uav3d_comm/paired_det_track.json`).

   | Phương án (giới hạn 500) | mAP sạch | AMOTA sạch | mAP trễ 2 frame | AMOTA trễ 2 frame | mAP mất 50% | AMOTA mất 50% |
   |---|---|---|---|---|---|---|
   | ego fine-tune (seed 2) | 0.512 | 0.542 | 0.512 | 0.542 | 0.512 | 0.542 |
   | Box late fusion, giữ box mạnh nhất | 0.612 | 0.595 | 0.414 | 0.358 | 0.573 | 0.566 |
   | Box gộp có trọng số + noisy-OR | 0.665 | 0.588 | 0.465 | 0.435 | 0.586 | 0.579 |
   | Box kiểu CERF (seed 0) | 0.622 | 0.721 | 0.533 | 0.646 | 0.557 | 0.646 |
   | Feature đã train (seed 2) | 0.693 | 0.754 | 0.452 | 0.636 | 0.597 | 0.646 |

   mAP cao không kéo theo tracking tốt: hai nhánh box gộp có mAP cao nhưng AMOTA thấp, vì gửi quá nhiều box điểm thấp làm tracker sinh track rác.
2. **"Trễ làm mọi cách gộp tệ hơn không cộng tác" là nói quá.** Câu đó chỉ đúng với mAP của các cách gộp không học (late fusion, noisy-OR)
   và của feature đã train. Box kiểu CERF vẫn hơn ego (0.533). Về tracking, feature trễ 2 frame vẫn hơn ego (0.636 so với 0.542).
   Phát biểu đúng: *các cách gộp hiện tại mất phần lớn lợi ích cộng tác khi message không đồng bộ, và mức nhạy phụ thuộc vào biểu diễn, kiến trúc gộp và cách train.*
3. **Thời gian mỗi frame: đã xác minh là 0.2 s thời gian mô phỏng** (cập nhật 13:00).
   - Annotation có vận tốc `vx, vy` lấy thẳng từ trình mô phỏng, không suy ra từ timestamp.
   - Trên 31 880 cặp xe chạy gần đều tốc (lệch tốc độ dưới 5%, tốc độ ít nhất 2 m/s), quãng đường giữa hai frame chia vận tốc
     cho trung vị **0.2006 s** (p5 0.1998, p95 0.243).
   - Timestamp là đồng hồ thật tính bằng **nano giây** (bắt đầu 2023-12-04), cách nhau khoảng 2.718 s mỗi frame: trình mô phỏng chạy chậm
     hơn thời gian thật khoảng 13 lần. Tin nhắn trước của tôi chia cho 1e6 và gọi là "2718 giây" là sai đơn vị (Codex đã chỉ ra).
   - Paper ghi 0.5 Hz, không khớp với dữ liệu mini. Có thể đó là tốc độ theo đồng hồ thật, hoặc là lỗi đánh máy.
   - Hệ quả: tracker đang lấy dt từ timestamp (2.7 s). Hiện chưa sao vì detector cho vận tốc gần 0, nhưng protocol mới phải dùng dt = 0.2 s.
     Report vẫn ghi độ trễ theo frame; 1 frame = 0.2 s mô phỏng.
   - Kèm theo phát hiện của Codex: ở đầu scene, độ trễ thực tế ngắn hơn nhãn (warm-start). Với delay 2 frame thì 6/60 frame trễ ít hơn 2.
3b. **Từ audit của Codex** (`reports/archive/uav3d/codex-uav3d-independent-audit-2026-11-04.md`): mAP 0.693 là của BEV FP16, chưa đi qua lượng tử q8.
    Box chưa được lượng tử thật theo 85 bit. Poture lấy box helper ở cap 83, còn late fusion lấy ở cap 500.
    Phép warp chỉ tịnh tiến, chưa xoay. Tất cả đã được ghi nhận; phải sửa trước thí nghiệm mới.
4. **"2600 lần ít bit hơn" không phải phép so codec công bằng.** Feature được tính là dense 8 bit, chưa nén thật
   (CodeFilling, How2comm và các bài tương tự nén mạnh hơn nhiều). Kết luận được phép: *feature dense thô quá đắt, còn message dạng box là baseline rất mạnh.*
   Chưa có đường cong chất lượng theo bit cho feature đã nén, nên **câu hỏi box hay feature chưa có lời đáp.**
5. **"Kênh tuổi không giúp" chỉ đúng với kiến trúc đã thử** (gộp trung bình xong mới thêm tuổi). Chưa kiểm tra fusion giữ các nguồn tách riêng.
6. **3 scene không còn là tập test.** Tôi đã chỉnh ngưỡng, giới hạn box và cách train sau khi nhìn kết quả trên chính 3 scene này,
   nên chúng chỉ còn là tập phát triển (dev). Kết luận tổng quát cần val độc lập, và chỉ nên tải sau khi đã chốt code và protocol.
   Việc "checkpoint chưa thấy 3 scene này" mới dựa trên đối chiếu token với file val do tác giả phát hành, chưa xác minh được provenance.
7. **Xử lý trễ không phải điểm mới.** How2comm (bù trễ bằng flow), CoBEVFlow (BEV flow cho timestamp không đều) và CERF (Kalman cộng dự báo)
   đều đã làm. Mục 9 bên dưới cần đọc kèm điều này: chỉ "Mamba + tuổi" thì không đủ làm novelty.

## Tóm tắt (bản gốc 07:50, xem đính chính ở trên)
1. **UAV3D có tín hiệu cộng tác thật.** Trên 3 scene Town10 held-out, cộng tác đưa mAP từ khoảng 0.28–0.51 lên khoảng 0.5–0.69,
   và AMOTA từ 0.39 lên 0.64.
2. **Phép so công bằng** (cùng encoder đóng băng, cùng dữ liệu fine-tune, không giới hạn số box):
   **feature đã train 0.693 > box gộp tâm có trọng số + noisy-OR 0.665 > box kiểu CERF 0.622 ≈ late fusion 0.612 > không gửi 0.512.**
   Box lấy được khoảng 85% phần lợi của feature với **khoảng 2600 lần ít bit hơn** (64 kbit so với khoảng 168 Mbit mỗi frame).
3. **Trễ 0.4 s làm mọi cách gộp không học (kể cả feature) TỆ HƠN không cộng tác** (0.41–0.47 so với 0.512).
   Chỉ decoder đã học dùng box của đồng đội làm gợi ý (kiểu CERF) là còn giữ được lợi ích (0.533). Mất gói thì ít hại hơn nhiều.
   Train với message cũ chữa được tác hại (box CERF gần như phẳng: 0.58–0.61 ở mọi độ trễ) nhưng **mất đỉnh** khi không trễ
   (feature 0.693 → 0.616), và thêm kênh tuổi message không giúp. Gộp message cũ mà vẫn giữ được đỉnh là câu hỏi mở.
   Tracking (AMOTA) cho cùng bức tranh: feature 0.754, box CERF 0.721, ego 0.542; khi trễ 0.4 s thì box CERF 0.646 ≥ feature 0.636.
4. **Box trùng lặp có giá trị:** khi không giới hạn box, chính sách "chỉ gửi box bên nhận chưa có" (novelty, 0.543) và cả oracle (0.599)
   đều **thua** gửi tất cả (0.612). Gộp noisy-OR (nhiều UAV cùng xác nhận thì độ tin cậy tăng) đẩy lên 0.665–0.671.
5. Ba cái bẫy đã phát hiện, ai dùng UAV3D cũng cần biết: (a) quaternion camera không theo quy ước nuScenes;
   (b) split "mini_val" kèm theo dữ liệu không phải held-out (chỉ scene_11/13/15 nằm trong val); (c) giới hạn **83 box/frame**
   của head (trung bình 103 xe/frame trong vùng chấm ở Town10) bóp méo các phép so (+0.10 mAP cho mọi model khi bỏ giới hạn).

**Ý nghĩa cho hướng nghiên cứu:** câu hỏi "gửi box hay gửi feature" gần như đã có lời đáp (CERF, và kết quả ở đây).
Khe hở thật là **gộp message cũ và không tin cậy**: message trễ 0.4 s gây hại thay vì giúp, nếu bên nhận không biết mức nào thì nên tin.
Bộ nhớ thời gian phía bên nhận (track hoặc belief), cùng quyết định gửi gì và khi nào dưới độ trễ, là chỗ Mamba, SB-temporal và RL có việc thật để làm.

## 0. Kiểm tra độ tin cậy
- **Split.** Token của mini trùng với file detection val của họ ở đúng **3 scene: scene_11, 13, 15** (60 frame, 6162 xe trong vùng chấm).
  17 scene còn lại có khả năng nằm trong tập train của checkpoint, nên ta chỉ dùng chúng để fine-tune decoder. **Mọi số dưới đây chỉ tính trên 3 scene này.**
- **Pipeline tái hiện đúng.** Lower-bound của ta trùng 99.0% box với file họ phát hành, DiscoNet trùng 99.6% (lệch < 0.1 m).
- **Bộ chấm đúng.** Lệch ≤ 0.001 AP so với evaluator gốc của UAV3D. Tracking dùng tracker CenterPoint và bộ chấm AMOTA chính thức của họ.
- **Town10 khó hơn trung bình** của README: lower-bound đạt 0.281 ở đây, so với 0.544 trên toàn bộ val.
- **Fine-tune decoder trên 17 scene Town10 một mình đã cho +0.136 mAP** (0.281 → 0.417, có giới hạn 83 box). Nhánh ego fine-tune
  là đối chứng bắt buộc: không có nó, phần lợi do thích nghi với Town10 sẽ bị tính nhầm thành lợi do cộng tác.
- Mỗi frame cách nhau 0.2 s (5 Hz). Detector không dự đoán vận tốc. Kênh truyền không được mô phỏng, chỉ đếm bit;
  trễ và mất gói được mô phỏng bằng cách dùng message cũ hoặc bỏ message.
- Bit: box 85 bit (x, y 12 bit; z 6; l, w, h 24; yaw 8; score 7; vận tốc 16) + 96 bit header/message.
  Feature: 80 kênh × 256 × 256 ô × 8 bit cho mỗi UAV giúp.

## 1. Phép so chính (công bằng): cùng encoder, cùng dữ liệu fine-tune, giới hạn 500 box
Encoder lower-bound đóng băng (feature BEV đã lưu). Decoder khởi tạo từ lower-bound, fine-tune 6 epoch trên 340 frame
(17 scene mini), AdamW lr 2e-4. Nhánh feature/box-CERF/ego: 3 seed khi có giới hạn 83, độ lệch ±0.001–0.003.
Box late fusion dùng decoder ego đã fine-tune (seed 2) chạy trên feature của từng UAV.

| Phương án | Sạch | Trễ 0.2 s | Trễ 0.4 s | Mất 20% | Mất 50% (3 seed) | kbit/frame |
|---|---|---|---|---|---|---|
| ego fine-tune (không gửi) | 0.512 | 0.512 | 0.512 | 0.512 | 0.512 | 0 |
| Box late fusion (giữ box mạnh nhất) | 0.612 | 0.532 | 0.414 | 0.598 | 0.573 | 48 |
| **Box gộp tâm có trọng số + noisy-OR** | 0.665 | 0.594 | 0.465 | 0.639 | 0.589 | 48 |
| Box kiểu CERF (Poture, decoder đã train) | 0.622 | **0.598** | **0.533** | 0.600 | 0.556 | khoảng 50 |
| **Feature đầy đủ, gộp trung bình, đã train** | **0.693** | 0.576 | 0.452 | **0.663** | **0.596** | khoảng 168 000 |

AP chi tiết ở điều kiện sạch (0.5/1/2/4 m): ego .138/.432/.679/.801; box gộp có trọng số + noisy-OR .298/.650/.826/.887;
CERF .200/.582/.814/.892; feature .333/.671/.852/.916. Feature hơn ở mọi ngưỡng, nhiều nhất ở định vị chính xác.

Recall@2 m theo nhóm xe (nhóm lấy từ mặt nạ thấy/không thấy của ảnh seg, giới hạn 83, detector fine-tune):

| Nhóm (số xe) | ego fine-tune | Feature đã train | Box CERF | Box gộp có trọng số |
|---|---|---|---|---|
| UAV0 tự thấy (2170) | 0.963 | 0.976 | 0.974 | 0.993 |
| Chỉ đồng đội thấy (1548) | 0.553 | 0.922 | 0.870 | 0.972 |
| Không ai thấy rõ (2444) | 0.345 | 0.388 | 0.359 | 0.606 |

Tracking (AMOTA) cho bảng này: xem `runs/uav3d_comm/tracking/u_*` (mục 6).

## 2. Với checkpoint gốc (chưa fine-tune, giới hạn 83): số trên benchmark gốc
| Phương án | mAP | AP@0.5/1/2/4 m | kbit/frame | AMOTA | Đổi ID |
|---|---|---|---|---|---|
| UAV0 một mình | 0.281 | .025/.235/.403/.461 | 0 | 0.389 | 48 |
| Box: gửi tất cả, score ≥ 0.3 | 0.429 | .040/.376/.629/.669 | 16.9 | 0.599 | 69 |
| Box: novelty@0.3 | 0.422 | .041/.362/.619/.664 | 8.9 | 0.575 | 90 |
| Box: gộp tâm có trọng số, ≥ 0.3 | 0.447 | .058/.431/.629/.669 | 16.9 | 0.598 | 75 |
| Box: oracle (chỉ xe UAV0 bỏ sót, dùng GT) | 0.451 | .042/.380/.680/.701 | 5.0 | 0.644 | 114 |
| Feature thưa 5% ô (decoder chưa train để gộp) | 0.427 | .116/.428/.565/.600 | 8 600 | 0.556 | 66 |
| Feature đủ, decoder chưa train để gộp | 0.254 | .000/.091/.410/.515 | 167 800 | 0.373 | |
| DiscoNet (checkpoint UAV3D) | 0.494 | .144/.485/.650/.696 | khoảng 10 000 | 0.642 | 36 |
| Upper-bound (file họ phát hành) | 0.508 | .177/.506/.652/.696 | khoảng 42 000 | 0.644 | 43 |

Phần này cho thấy nếu chỉ nhìn checkpoint gốc mà không kiểm soát fine-tune và giới hạn box thì dễ kết luận sai theo cả hai chiều.

## 3. Trễ, bù vận tốc và mất gói (box, detector gốc)
| Điều kiện | mAP | AMOTA | Đổi ID |
|---|---|---|---|
| Không trễ, không mất | 0.429 | 0.599 | 69 |
| Trễ 0.2 s | 0.401 | 0.576 | 117 |
| Trễ 0.4 s | 0.305 | 0.394 | 182 |
| Trễ 0.2 / 0.4 s + bù vận tốc từ 2 frame | 0.367 / 0.289 | 0.446 / 0.228 | |
| Mất gói 20% / 50% (3 seed) | 0.408 / 0.372 | 0.582 / 0.536 | |

Bù vận tốc đơn giản (ước lượng từ 2 frame liên tiếp) **làm tệ hơn**, vì nhiễu vị trí 0.5–1 m của detector bị khuếch đại.
CERF dùng Kalman filter cộng dự báo chuyển động, nhưng phần này không có trong code phát hành và paper không báo cáo thí nghiệm trễ.

## 4. Message bản đồ bằng chứng (heatmap CenterHead)
| Biến thể (decoder gốc) | Giới hạn 83 | Giới hạn 500 | kbit/frame | Recall "không ai thấy rõ" |
|---|---|---|---|---|
| UAV0 một mình | 0.281 | 0.317 | 0 | 0.090 |
| Heatmap, lấy max | 0.409 | 0.433 | 5 800 | 0.235 |
| Heatmap, noisy-OR | 0.409 | 0.449 | 5 800 (thưa: 840) | 0.396 |
| Box late fusion (cùng detector) | 0.429–0.447 | | 17–28 | 0.234 |

Noisy-OR cộng dồn được bằng chứng yếu, nhưng ở mức heatmap thì sinh thêm box sai và tốn hơn 30–200 lần số bit so với box. Không đáng.

## 5. Gửi box giúp ở đâu (detector gốc, recall@2 m)
| Nhóm (số xe) | UAV0 một mình | Gửi box | Novelty | Feature 5% ô | DiscoNet | Upper |
|---|---|---|---|---|---|---|
| UAV0 tự thấy (2170) | 0.938 | 0.980 | 0.976 | 0.968 | 0.969 | 0.979 |
| Chỉ đồng đội thấy (1548) | 0.280 | 0.928 | 0.919 | 0.689 | 0.903 | 0.921 |
| Không ai thấy rõ (2444) | 0.090 | 0.234 | 0.230 | 0.156 | 0.291 | 0.273 |

## 6. Train với message cũ và mất gói (giới hạn 500, 2 seed, ±0.002–0.004)
Train y như mục 1 nhưng mỗi bước lấy message trễ d ∈ {0, 1, 2} frame (đều nhau) và bỏ mỗi UAV giúp với xác suất 0.2.
`+tuổi` = thêm một kênh đầu vào bằng tuổi message (d/2).

| Nhánh | Sạch | Trễ 0.2 s | Trễ 0.4 s | Mất 50% |
|---|---|---|---|---|
| ego (không gửi) | 0.512 | 0.512 | 0.512 | 0.512 |
| Feature, train thường | **0.693** | 0.576 | 0.452 | **0.596** |
| Feature, train có trễ/mất | 0.616 | **0.662** | 0.548 | 0.553 |
| Feature, train có trễ/mất + tuổi | 0.622 | 0.659 | 0.552 | 0.559 (1 seed) |
| Box CERF, train thường | 0.622 | 0.598 | 0.533 | 0.556 |
| Box CERF, train có trễ/mất | 0.589 | 0.607 | **0.584** | 0.554 |
| Box CERF, train có trễ/mất + tuổi | 0.590 | 0.607 | 0.583 | 0.551 (1 seed) |

- Train với message cũ **chữa được tác hại của trễ**: mọi nhánh lại hơn ego ở 0.4 s. Box gần như phẳng theo độ trễ (0.584–0.607).
- Cái giá là **mất đỉnh ở điều kiện sạch** (feature 0.693 → 0.616; AP@0.5 0.333 → 0.14). Decoder không biết message cũ bao nhiêu
  nên học một mức bù "trung bình".
- **Thêm kênh tuổi không giúp** (khác < 0.01). Với gộp trung bình, feature hiện tại và feature cũ đã bị trộn trước decoder.
  Muốn dùng tuổi cần kiến trúc giữ các nguồn tách biệt, hoặc dự báo chuyển động cho từng xe (detector này không có vận tốc).
  **Đây là câu hỏi mở cụ thể nhất từ cả buổi.**

## 7. Tracking (AMOTA, tracker CenterPoint + bộ chấm chính thức)
Có giới hạn 83 box (3 seed cho các nhánh đã train):

| Phương án | Sạch | Trễ 0.2 s | Trễ 0.4 s | Mất 50% |
|---|---|---|---|---|
| ego fine-tune | 0.541 | 0.541 | 0.541 | 0.541 |
| Feature đã train | **0.692** | **0.686** | 0.581 | **0.643** |
| Box CERF (box gốc) | 0.661 | 0.658 | 0.589 | 0.625 |
| Box CERF (box fine-tune) | 0.685 | 0.658 | **0.608** | 0.628 |
| Box late fusion ≥ 0.3 (detector fine-tune) | 0.684 | | | |
| Box gộp có trọng số (detector fine-tune) | 0.687 | | | |
| Box novelty@0.3 (detector fine-tune) | 0.618 | | | |
| **Box oracle** (detector fine-tune) | **0.763** | | | |

Giới hạn 500 (nhánh đã train: seed 2 cho feature, seed 0 cho box CERF):

| Phương án | Sạch | Trễ 0.2 s | Trễ 0.4 s | Mất 50% |
|---|---|---|---|---|
| ego fine-tune | 0.542 | 0.542 | 0.542 | 0.542 |
| **Feature đã train** | **0.754** | **0.743** | 0.636 | 0.646 |
| Box CERF đã train | 0.721 | 0.725 | **0.646** | 0.646 |
| Box late fusion (không giới hạn, khoảng 750 box/frame) | 0.595 | 0.545 | 0.358 | 0.566 |
| Box gộp có trọng số + noisy-OR (không giới hạn) | 0.588 | 0.562 | 0.435 | 0.579 |

- Tracking khớp với detection: feature dẫn ở điều kiện sạch, box CERF bằng hoặc nhỉnh hơn khi trễ 0.4 s.
- **Late fusion không giới hạn thì tracking tệ đi** (0.684 có giới hạn → 0.595): hàng trăm box score thấp làm tracker greedy sinh track rác.
  Tập box tốt cho detection (mAP) khác tập box tốt cho tracking. Decoder đã train (feature hoặc CERF) cho ra tập box sạch hơn.
- **Box oracle vượt feature ở tracking** (0.763 so với 0.692): chọn đúng box để gửi vẫn là đòn bẩy lớn, và novelty đơn giản thì chưa làm được.

Bảng đầy đủ: `runs/uav3d_comm/tracking/tracking_table.json` (`track_table.py`).

## 8. Giới hạn
- Chỉ 3 scene held-out Town10 (60 frame, khoảng 6000 xe). Seed variance nhỏ (±0.003), nhưng **scene variance chưa đo**.
  Muốn kết luận chắc cần val đầy đủ (150 scene, khoảng 50 GB ảnh), và việc tải đó do bạn quyết định.
- Fine-tune chỉ trên 340 frame Town10, 6 epoch. Nhánh feature có thể còn tăng nếu có nhiều dữ liệu hơn.
- Trễ được mô phỏng bằng message 1–2 frame trước (0.2/0.4 s), chưa có trễ ngẫu nhiên theo từng UAV hay theo tải kênh.
- Kênh truyền chưa mô phỏng: chỉ đếm bit, chưa có tốc độ, hàng đợi hay nhiễu.
- Box kiểu CERF là bản tự làm lại (Poture 6 kênh tại ô tâm box, gộp max theo score), không phải checkpoint của tác giả.

## 9. Đề xuất bước tiếp theo (bạn quyết định)
1. **Gộp message trễ và không tin cậy** (khe hở rõ nhất). Đã biết: train với message cũ chữa được tác hại nhưng mất đỉnh;
   một kênh tuổi không đủ. Cần so: (a) gộp giữ nguồn tách biệt (ghép kênh hoặc attention theo nguồn) + tuổi;
   (b) bộ nhớ thời gian phía bên nhận (ConvGRU / Mamba trên BEV hoặc trên track) để dự báo trạng thái hiện tại từ message cũ;
   (c) belief kiểu SB-temporal. Mục tiêu: giữ đỉnh khi không trễ (≈ 0.69) **và** không tụt khi trễ.
2. **Quyết định gửi khi có trễ:** bên gửi biết message sẽ đến muộn d frame, vậy gửi box hiện tại, box đã dự báo hay không gửi.
   Đây là chỗ RL hoặc lập lịch có ý nghĩa: có hiệu ứng tuần tự thật, vì message cũ có thể gây hại.
3. **Giá trị của xác nhận lặp lại:** noisy-OR cho thấy box trùng có ích. Thay vì "chỉ gửi cái mới", học giá trị message theo mức giảm bất định của bên nhận.
4. Mở rộng dữ liệu: val đầy đủ UAV3D (cần tải khoảng 50 GB ảnh val, hoặc toàn bộ bản v1.0).
