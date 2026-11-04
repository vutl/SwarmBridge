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

### 0.2 Điểm thiết kế cần sửa trong plan (đã được Codex chấp nhận 05/11; C chốt phương án (c))

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

### 0.3 Kết quả mốc 0 (05/11)

**Khóa môi trường.** `scripts/wireless_abc/setup_env.sh` giờ dựng lại đúng bản khóa: Python 3.12.16 (uv) +
`uv pip sync wireless-abc.lock.txt` (torch từ index PyTorch cu130, còn lại PyPI). Chạy `--dry-run` trên env hiện có:
"Would make no changes" (77 package).

**Thống kê RT có phân loại** (`scripts/wireless_abc/rt_stats.py`, kết quả `runs/wireless_abc/rt_smoke/rt_stats.json`).
Cảnh Munich, BS cao 27 m tại (8.5, 21), 28 GHz, lưới 12×12 điểm trên 400×400 m quanh BS ở mỗi độ cao. Điểm nằm dưới mái
nhà (tia từ 1 km chiếu xuống chạm hình học trước khi tới điểm) bị loại: 22/144 điểm ở 20 m, 0 ở 50/100 m.
Lớp: `los` = có đường trực tiếp; `nlos` = không có đường trực tiếp nhưng còn đường khác; `nopath` = solver không tìm thấy
đường nào với cấu hình đó.

| Độ cao | Cấu hình solver | LoS | NLoS | không có đường | trung vị công suất LoS / NLoS (dB) |
|---|---|---|---|---|---|
| 20 m | LoS + phản xạ gương, độ sâu 3 | 0.393 | 0.279 | 0.328 | −105.0 / −127.5 |
| 20 m | + nhiễu xạ | 0.393 | 0.549 | 0.057 | −105.5 / −152.8 |
| 20 m | + phản xạ khuếch tán | 0.393 | 0.270 | 0.336 | −105.0 / −126.3 |
| 20 m | độ sâu 5 | 0.393 | 0.344 | 0.262 | −105.0 / −135.3 |
| 50 m | (mọi cấu hình) | 0.965 | 0.021–0.035 | 0–0.014 | −104.7 / −125…−132 |
| 100 m | (mọi cấu hình) | 0.993 | 0–0.007 | 0–0.007 | −105.1 / — |

Đọc:
1. Lần smoke trước ("2/3 điểm ở 20 m không có path") là do điểm rơi vào trong nhà và solver thiếu nhiễu xạ; không phải
   bằng chứng mất sóng.
2. **NLoS chỉ đáng kể khi UAV thấp hơn hoặc ngang BS** (20 m, BS 27 m). Ở ≥ 50 m gần như toàn LoS trong vùng 400 m.
   Với LoS và vị trí UAV/BS đã biết, beam tính thẳng từ hình học, không cần pilot. → Scene chính cho A/B/C phải là
   **UAV bay thấp (≤ độ cao BS) trong phố**, hoặc BS thấp hơn; ghi rõ phạm vi này trong mọi kết luận.
3. "Không có đường" giảm từ 33% xuống 6% khi bật nhiễu xạ, nhưng đường nhiễu xạ yếu hơn LoS khoảng 48 dB (trung vị) —
   gần như không dùng được ở mmWave. Đường phản xạ yếu hơn khoảng 22 dB. Quỹ đường truyền (ví dụ 30 dBm, độ lợi
   beam 16 antenna ~12 dB, nhiễu 100 MHz ~ −87 dBm): LoS ~ SNR 24 dB; phản xạ ~ 2 dB; nhiễu xạ ~ −24 dB. Vùng quan tâm
   là NLoS **qua phản xạ**, nơi chọn đúng beam phản xạ quyết định link sống hay chết.
4. Cấu hình solver dùng cho trace: độ sâu 5 + nhiễu xạ (để không đếm nhầm "không có đường"), ghi trong config.

**Unit test lõi tuyến tính** (`swarmbridge/experiments/wireless_abc/linear.py`, `test_linear.py`): **14/14 qua**.
Gồm complex↔real, CN(0,1) đúng công suất, DFT unitary và hướng beam, pilot đơn vị (DFT/ngẫu nhiên), phương sai nhiễu theo
SNR, năng lượng pilot cố định (không chuẩn hóa theo kênh), pseudoinverse khôi phục h khi đủ pilot không nhiễu, posterior
LMMSE khớp Monte Carlo (hiệp phương sai và coverage 95%), GMM một thành phần = LMMSE, trọng số GMM = Bayes chính xác,
GMM nhận đúng trạng thái trong prior hai đỉnh, E|wᴴh|² = wᴴ(Σ+μμᴴ)w khớp Monte Carlo, mật độ CN chuẩn hóa.
Còn chờ (viết khi có code tương ứng): endpoint/sampler bridge (A1), thời điểm hiệu lực pilot/feedback, queue, mất gói,
cold start cho môi trường C (codec/sổ bit của UAV3D đã có 71 test trong `uav3d_comm/proto/test_proto.py`).

**A0:** đang tải `Diffusion_channel_est.zip` (2.871 GB, md5 cfe59362…) vào `data/wireless_abc/fesl/` (gitignored).

### 0.4 Ghi chú môi trường (05/11, 01:50)

Để chạy code Fesl (A0) đã thêm `h5py`, `tqdm`, `torchvision 0.27.1+cu130`, `pytorch-fid 0.3.0` (không deps). Lệnh cài
torchvision đầu tiên tự nâng torch lên 2.15 — đã đưa về **torch 2.12.1+cu130** (kiểm tra `torch.cuda.is_available()`),
lock file cập nhật lại.

## A0 — chạy lại checkpoint Fesl (xong)

`swarmbridge/experiments/wireless_abc/a0_fesl.py`, `scripts/wireless_abc/run_a0.sh`; kết quả `runs/wireless_abc/a0/`.
Dataset Zenodo 20737830 (2.871 GB, md5 khớp `cfe59362…`), giải nén chọn lọc 2 file (82 MB test 3GPP + 1.9 GB QuaDRiGa;
tổng dữ liệu mới 4.6 GB). Code và checkpoint của tác giả, Tester của tác giả (SNR-matched start, quá trình ngược tất
định), 1 024 mẫu test đầu (tác giả: 10 000), CPU, torch 2.12 (tác giả: 2.1).

| SNR (dB) | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3GPP: ta / công bố (NMSE dB) | −1.34 / −1.37 | −4.34 / −4.37 | −7.97 / −8.00 | −11.71 / −11.73 | −15.72 / −15.75 | −19.87 / −19.92 | −24.16 / −24.20 | −27.97 / −27.95 | −30.93 / −30.85 | −37.19 / −37.18 | −39.97 / −40.00 |
| QuaDRiGa LoS: ta / công bố | −7.80 / −7.56 | −9.29 / −9.01 | −10.79 / −10.45 | −13.09 / −12.76 | −16.23 / −15.95 | −19.56 / −19.29 | −22.92 / −22.67 | −26.19 / −26.01 | −30.28 / −30.18 | −35.32 / −35.23 | −40.07 / −40.01 |

Pipeline, chuẩn hóa và NMSE khớp: 3GPP lệch ≤ 0.08 dB; QuaDRiGa lệch 0.06–0.34 dB, cùng chiều (tập con 1 024 mẫu dễ hơn
một chút). Chưa tái lập phần huấn luyện; chưa tái lập paper SB-ISAC (A0 thứ hai) — để sau A1 nếu còn ngân sách.

### 0.5 Việc tiếp theo

1. A0: giải nén chọn lọc, chạy checkpoint Fesl trên ≤ 1 024 mẫu, đối chiếu NMSE–SNR.
2. Mốc 1: scene trace cho B/C — UAV bay thấp (15–25 m) theo quỹ đạo trong phố, blocker động (giải lại path mỗi bước khi
   hình học đổi), đo độ bền beam từ trace.
