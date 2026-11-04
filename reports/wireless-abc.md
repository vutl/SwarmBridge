# Wireless A/B/C pilot — tài liệu tổng hợp

Plan gốc: plan pilot của Codex (05/11), dựa trên `reports/ke-hoach-proposal-abc-2026-11-04.md`. Phạm vi: feasibility/dev,
tối đa 10 GB dữ liệu mới, 8 giờ GPU, một job GPU tại một thời điểm (khóa chung Windows–WSL).

## Mốc 0 — audit và các điểm thiết kế phải chốt trước khi code

### 0.1 Môi trường (xong)

- Máy: RTX 3070 8 GB (driver 595.97), WSL 7 GB RAM, ổ WSL còn 896 GB, ổ C còn 689 GB.
- WSL env mới `~/envs/wireless-abc` (Python 3.12, uv): torch 2.12/cu130, numpy 2.5.4, sionna-rt 2.2.0, sionna 2.2.0,
  gymnasium 1.4.0, stable-baselines3 2.9.2. Cài đúng các phiên bản trên (torch 2.12.1+cu130, CUDA khả dụng), 5.2 GB. Lệnh: `scripts/wireless_abc/setup_env.sh`; manifest khóa: `scripts/wireless_abc/wireless-abc.lock.txt`.
- Env cũ (`uav3d`, `sb`, Windows `.venv`, `.venv-sb-papers`) không bị đụng.
- Smoke Sionna RT trên **CPU (LLVM)**, `scripts/wireless_abc/smoke_rt.py cpu`: cảnh Munich, BS 8×8 UPA 28 GHz, 9 điểm UAV
  ở 20/50/100 m, max_depth 3 (LoS + phản xạ gương): nạp cảnh 0.5 s, giải path 1.1 s. Ở 20 m, 2/3 điểm **không có path
  nào** (bị chắn hoàn toàn với độ sâu 3); ở 50 m và 100 m mọi điểm có path, công suất −77…−84 dB. → NLoS có thật ở độ
  cao thấp; ray tracing cho scene nhỏ chạy được trên CPU, có thể không cần GPU cho RT.

### 0.2 Điểm thiết kế cần sửa trong plan (Claude, chờ Codex xác nhận)

1. **A1: mô hình phép đo pilot phải là beam training đường xuống.** BS 16 antenna, UAV 1 antenna: nếu pilot đường lên
   thu số hoàn toàn, *một* ký hiệu pilot đã cho 16 phép đo của h, nên "pilot dài 4/8/16" không tạo ra chế độ thiếu đo.
   Đề xuất: BS phát Np ký hiệu với precoder w_k (k = 1..Np), UAV đo y_k = w_kᴴh + n_k. Np = 4/8 < 16 là thiếu xác định
   — đúng chế độ mà prior sinh có lợi (Zhou TWC 2025 ghi nhận lợi ích ở mật độ pilot 0.2–0.6). Hai loại w_k: DFT
   (quét beam một phần, sát thực tế SSB) và ngẫu nhiên đơn mô-đun (nén). Ghi rõ loại nào trong mọi bảng.
2. **A1: thêm baseline GMM.** Ở 16 chiều phức, estimator Gaussian-mixture (nhóm Fesl) cho posterior đa đỉnh *giải
   tích*, rẻ và thường rất mạnh. SB/flow/DDPM phải thắng GMM, không chỉ thắng LMMSE. GMM còn làm posterior tham chiếu
   cho coverage.
3. **B: thêm baseline "local + prior theo vị trí".** Mỗi UAV dùng prior học từ quỹ đạo train trong cùng scene (kiểu
   channel knowledge map), không có message. Nếu thiếu baseline này, lợi ích "hợp tác" có thể chỉ là lợi ích của prior
   học được. Hợp tác chỉ có lý do khi có thông tin *hiện tại* mà bản đồ tĩnh không có → scene B phải có **blocker di
   động**, và báo kết quả tách riêng: hình học tĩnh / blocker động.
4. **C: payload box quá nhỏ so với kênh mmWave — bài toán chọn message có thể biến mất.** Message box KF-PV đầy đủ khoảng
   41 kbit/frame cho 4 helper, tức khoảng 50 kbps mỗi helper. Một link mmWave đã căn beam (100 MHz, SNR 0 dB) cho cỡ
   100 Mbps, nên toàn bộ message gửi xong trong < 1 ms. Khi đó "chọn box nào gửi" không còn ràng buộc; C co lại thành quản
   lý beam/pilot với phần thưởng "link có sống lúc hạn chót không", và các phát hiện UAV3D về kích thước/thứ tự không
   dùng tới. Cần chọn một trong:
   - (a) payload nặng hơn: feature BEV (cỡ Mbit/frame, như Where2comm/DiffCP), cạnh tranh thật với dung lượng khi beam
     không chắc;
   - (b) kênh băng hẹp dùng chung cho nhiều UAV (sub-6 GHz, không beam hẹp), khi đó bỏ phần beam;
   - (c) giữ box và định nghĩa rõ C là "beam/pilot management đo bằng tracking", bỏ phần chọn nội dung semantic.
   Đề xuất của Claude: (c) cho vòng pilot (rẻ, đúng phạm vi 8 giờ GPU), ghi rõ semantic chỉ là thước đo; (a) để vòng
   sau nếu C qua gate.
5. **C: thang thời gian.** Ở 28 GHz, UAV 10 m/s: f_D ≈ 930 Hz, thời gian kết hợp fading nhỏ khoảng 0.45 ms. Bước điều
   khiển 1 ms đã dài hơn thời gian kết hợp, nên quyết định không thể theo CSI tức thời; cái chậm là **góc/beam** (ở 100 m,
   10 m/s → khoảng 0.1 rad/s; độ rộng beam 16 antenna khoảng 0.12 rad → beam đổi theo thang ~1 s). Đề xuất: bước điều
   khiển 5–10 ms, quyết định ở mức beam, fading nhỏ lấy trung bình trong bước (OFDM băng rộng).
6. **C: ngân sách PPO.** 100k bước × 1 ms = 100 s mô phỏng = 500 frame perception, trong khi dev UAV3D chỉ có 180 frame
   và phần thưởng tracking chỉ có ở mốc 0.2 s (thưa: 1/200 bước). Với bước 10 ms: 20 bước/frame, 100k bước ≈ 5 000
   frame → cần các scene train (export detection 17 scene train, có trong mốc 4). Nếu vẫn thiếu, giảm đồng đều và ghi
   "chưa hội tụ" như plan đã nói.

### 0.3 Thứ tự việc của Claude ngay sau khi Codex xác nhận 0.2

1. Khóa manifest env; smoke test Sionna RT trên CPU và GPU (một scene nhỏ, có khóa GPU).
2. Unit test chung (complex↔real, DFT unitary, SNR/noise, pseudoinverse, posterior Gaussian/GMM giải tích, coverage).
3. A0: tải dataset Fesl (Zenodo, ~2.9 GB, chỉ giải nén phần cần), chạy checkpoint trên ≤ 1 024 mẫu.
4. Scene RT nhỏ cho B/C (BS, 3 UAV, blocker di động, mặt phản xạ), cache ≤ 2 GB.
