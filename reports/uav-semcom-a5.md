# SwarmBridge hướng A — A5: mở rộng dữ liệu, task fidelity, khảo sát chế độ vận hành, RL có mục tiêu

Bắt đầu 2026-11-07, tiếp nối A4 (commit `0e77387a`, báo cáo `reports/uav-semcom-a.md` mục 8).

**Câu hỏi chính của A5:** trong điều kiện nào RL thực sự mang lại lợi ích so với heuristic thích nghi mạnh nhất (reuse+budget, rút ra từ A4), và khi nào heuristic đã đủ tốt?

Không đổi reward, bandwidth hay định nghĩa nhiễu/bit/deadline để PPO thắng. A4 không bị ghi đè: mọi kết quả A5 nằm ở `runs/uav_semcom/a5/`.

## 0. Quy tắc định trước (ghi và commit TRƯỚC khi có kết quả quét hay PPO)

### 0.1 Dữ liệu
- Ảnh: toàn bộ 548 ảnh VisDrone2019-DET-val, cùng một tập con cố định 403 ảnh VisDrone2019-DET-test-dev (sắp xếp theo tên, lấy mỗi ảnh thứ 4). Test-dev là pool kiểm tra ngoài, đến từ các video khác.
- Ảnh val được chia **theo chuỗi video** (tiền tố tên file) với seed 0, nhắm tỉ lệ train 0,55 / val 0,15 / test 0,30 tính theo số khung hình:
  - train: 304 khung, 32 chuỗi;
  - val: 84 khung, 14 chuỗi;
  - test: 128 khung, 28 chuỗi.
- Ảnh A4 thuộc chuỗi test (32 khung) bị **loại hẳn**, không dùng ở đâu. Vì vậy mọi ảnh test A5 đều chưa từng được A4 hay các model PPO A4 nhìn thấy.
- Manifest kèm SHA-256 từng file: `runs/uav_semcom/a5/manifest.json`.
- Vai trò của từng pool:
  - train: dùng để fit bảng mean và bộ dự đoán từ probe, chạy các episode tinh chỉnh heuristic (seed 0..29) và train PPO.
  - val: quét chế độ để chọn regime (seed 3000..3049) và chọn checkpoint PPO (seed 3000..3019).
  - test + test-dev: chỉ dùng một lần cho đánh giá cuối, seed 7000..7099.
  - Không seed nào trùng với các bộ seed của A4.

### 0.2 Chọn chế độ (regime) để train RL

Quyết định chỉ dựa trên heuristic trên **val**; không có PPO nào tham gia.
- **R1 = ref**: cấu hình A4 trên dữ liệu A5, bắt buộc.
- **R2 = K5**: 5 UAV / 2 kênh, định trước vì đây là kịch bản tranh chấp tài nguyên.
- **R3**: trong các regime không đổi mục tiêu và không đổi K (tức loại `ref`, `K*`, `wt*`), **và** có điều khiển không tầm thường (return val của heuristic tinh chỉnh tốt nhất − return val của greedy ≥ 2; regime tải thấp mà mọi heuristic gần như bằng nhau thì không có gì để học), chọn regime mà cơ chế A4 giúp **ít nhất**. Cụ thể là regime có (return val của heuristic reuse tinh chỉnh) − max(return val của base tinh chỉnh, budget tinh chỉnh) nhỏ nhất. Đây là nơi hai quy tắc rút ra từ A4 mất ưu thế, nên là nơi kiểm tra rõ nhất xem RL có tìm ra cách khác hay không.
  - Nếu chênh lệch giữa ứng viên nhất và nhì nhỏ hơn 0,5, chọn regime có "giá trị thích nghi" lớn hơn, tức (reuse tinh chỉnh − fixed tinh chỉnh) lớn hơn.

### 0.3 Huấn luyện và chọn checkpoint
- Thuật toán: TimeMaskablePPO (discount theo giây), cùng mạng, siêu tham số, số env và 2M bước như các run A4 đã sửa.
- Mỗi regime train **joint** và **mode-only**, 3 seed (0, 1, 2).
- Ở mode-only, slot do phần slot của heuristic reuse đã tinh chỉnh cho regime đó quyết định.
- Checkpoint được lưu mỗi 2·10⁵ bước. Checkpoint dùng cho test là checkpoint có **return val cao nhất** (nếu bằng nhau thì lấy checkpoint sớm hơn). Checkpoint cuối cũng được báo cáo.
- Thứ tự chạy: seed 0 của mọi (regime, phần), rồi seed 1, rồi seed 2.
- Nếu đường học (learning curve) val vẫn còn tăng rõ ở 2M bước, tôi sẽ ghi lại. Tôi không tự động tăng số bước.

### 0.4 So sánh cuối
- Mọi policy được đánh giá trên cùng seed, nên mọi so sánh đều theo cặp, với khoảng tin cậy t 95%.
- Baseline:
  - fixed mode tinh chỉnh;
  - greedy;
  - base (pacedB…) tinh chỉnh;
  - budget tinh chỉnh;
  - **reuse tinh chỉnh (heuristic mạnh)**;
  - hai heuristic A4 giữ nguyên tham số: pacedB1.5+budget1.5 và greedy+budget1+reuse.
- PPO A4 (td joint s0–2, td mode-only s0–2) được thử **zero-shot** trên các regime K = 3, tức khi kích thước quan sát và hành động khớp. Kết quả này được ghi rõ là zero-shot.

## 1. Dữ liệu mới, chia tập và codec profiling
(đang chạy)

## 2. Task fidelity
(đang chạy)

## 3. Khảo sát chế độ vận hành
(đang chạy)

## 4. PPO vs heuristic mạnh
(chưa chạy)
