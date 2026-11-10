# Wireless A/B/C pilot — tài liệu tổng hợp

Plan gốc: plan pilot của Codex (05/11), dựa trên `reports/archive/huang-abc/ke-hoach-proposal-abc-2026-11-04.md`. Phạm vi: feasibility/dev,
tối đa 10 GB dữ liệu mới, 8 giờ GPU, một job GPU tại một thời điểm (khóa chung Windows–WSL).

**ĐÍNH CHÍNH (05/11 trưa, sau audit của Codex và ChatGPT):** vòng đêm có lỗi implementation và kết luận diễn đạt quá rộng.
Lỗi: (1) bước covariance của GMM bị liên hợp ngược (GMM trong A1 không hợp lệ); (2) coverage đo bằng 16 mẫu bị lệch
xuống; (3) B dùng ước lượng độ chính xác đầy đủ nhưng tính bit như gửi pilot lượng tử, và thiếu baseline local có bộ nhớ;
(4) C chỉ tính phí quét cho `ssb20`, dùng payload cố định và cho receiver dùng message ngay tại frame phát (delay 0);
(5) "oracle" của C chỉ là oracle chọn beam dưới lịch round-robin, không phải cận trên của bài toán. Kết quả sau sửa nằm ở
mục **"Vòng sửa theo audit"** cuối tài liệu; các mục đêm qua giữ nguyên để đối chiếu nhưng **không còn là kết luận**.
A0b là "thử nghiệm bridge tự dựng (Brownian, σ = 1)", **không** phải tái lập recipe VE-DSB của paper.

**Tóm tắt sau audit (05/11 chiều):** A — SB 0/18 cả trên link lọc lẫn toàn bộ link, GMM đã sửa là baseline mạnh; B — hợp tác so với bản đồ + lịch sử riêng: +1.3…+2.5% rate / +3.5…+6.6% outage, ổn định nhưng gate 0/6; C — greedy hơn round-robin 0.011 AMOTA, causal cách oracle 0.008, chưa có cận trên thật. Xem cuối tài liệu.

Tóm tắt cũ (05/11 sáng, đã bị thay): A — SB không qua gate (0/18); B — hợp tác có lợi chỉ ở vùng chưa có bản đồ; C — quét beam 20 ms ≈ oracle, không mở RL.

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
một chút). Chưa tái lập phần huấn luyện.

## A0b — thử nghiệm bridge tự dựng trên thiết lập kênh của paper SB-ISAC (không phải tái lập recipe)

Lưu ý (audit): bridge ở đây là nội suy Brownian với σ = 1 và sampler tự viết, **không** phải lịch VE (c = 2.6, k = 0.4)
và bộ tích phân mũ của paper WCL, cũng không phải recipe đầy đủ của bài TCCN. Kết quả chỉ nói: recipe tự dựng này chưa
thắng hồi quy cùng mạng.

`a0_sbisac.py`, `runs/wireless_abc/a0_sbisac/` (GPU 20 phút). Kênh sensing tổng hợp như paper: ULA N = 32 hai phía,
L = 3 path, LS = H + nhiễu (pilot trực giao), 7 500 train / 2 500 test, một model mỗi SNR. **Cùng một CNN** (4 lớp, 64
kênh, miền góc) cho: hồi quy thường; bridge huấn luyện kiểu Brownian/VE, suy luận ODE **1 bước** từ LS (như paper);
bridge ngẫu nhiên 10 bước (như bài TCCN). 6 000 bước, 3 seed. NMSE (dB, test):

| SNR | LS | LMMSE | hồi quy | bridge 1 bước | bridge 10 bước ngẫu nhiên | bridge 1 bước − hồi quy | seed bridge tốt hơn |
|---|---|---|---|---|---|---|---|
| −10 | +10.00 | −0.72 | −7.33 | −7.48 | −7.60 | −0.15 | 3/3 |
| −5 | +5.00 | −1.67 | −11.25 | −11.03 | −10.80 | +0.21 | 0/3 |
| 0 | 0.00 | −3.45 | −14.33 | −14.22 | −12.50 | +0.11 | 0/3 |
| +5 | −5.00 | −6.37 | −17.71 | −17.31 | −13.27 | +0.40 | 0/3 |
| +10 | −10.00 | −10.40 | −21.44 | −19.73 | −13.58 | +1.72 | 0/3 |

Paper báo DSB hơn UNet cùng kiến trúc 0.7–2 dB. Trong bản dựng lại, **hồi quy thường tốt hơn bridge 1 bước** ở mọi SNR ≥
−5 dB (0.1–1.7 dB); bridge chỉ hơn 0.15 dB ở −10 dB; bridge ngẫu nhiên 10 bước kém rõ ở SNR cao. LS = −SNR (kiểm tra
đúng). LMMSE yếu như trong paper (góc ngẫu nhiên đều → hiệp phương sai gần đẳng hướng). **Giới hạn:** mạng nhỏ hơn (paper:
UNet + attention); phân bố góc/hệ số path và lịch nhiễu VE không được paper nêu đủ → "chưa lặp lại được trong bản dựng
lại", không phải bác bỏ paper.

## Mốc 1 — trace kênh (xong)

`swarmbridge/experiments/wireless_abc/trace_gen.py` → `data/wireless_abc/trace_v1/` (54 MB, gitignored, tái tạo tất định
từ seed 0). Cảnh Munich (Sionna RT 2.2, CPU/LLVM), BS (8.5, 21, 27) m, ULA 16 phần tử, 28 GHz; UAV 1 antenna ở 15–25 m,
4 UAV/episode, 20 s, mẫu RT mỗi 100 ms, 24 path mạnh nhất/mẫu kèm Doppler (vận tốc UAV); solver độ sâu 5 + nhiễu xạ.
120 episode: chẵn = **bay đoàn** (UAV u đi lại đúng đường UAV 0 trễ 2u s, lệch ngang ~3 m), lẻ = **độc lập** (đối chứng).
Quỹ đạo kiểm tra va chạm với hình học (không đi xuyên/dưới mái nhà). Trung bình mỗi mẫu: có LoS 44%, không có đường
13% (bay đoàn 0.439/0.121, độc lập 0.431/0.136). 28–38 s CPU/episode.
Kiểm tra mô hình mảng: beam tính từ hình học (GPS) trùng beam tốt nhất ở **99.8%** mẫu LoS (dấu ngược lại: 12%).
**Phạm vi:** cảnh tĩnh, không có vật chắn di động. Ở 15–25 m, link BS–UAV bị nhà chắn; xe cộ (2–4 m) gần như không ảnh
hưởng. Vì vậy B được kiểm bằng "vùng chưa có bản đồ", không phải blocker động.

## A1 — ít pilot, posterior kênh và chọn beam (xong; gate SB: **không qua**)

`a1_bench.py`, `runs/wireless_abc/a1/a1_results.json` (GPU qua khóa chung, 31 phút). Split theo episode 72/24/24; loại
35% mẫu có công suất < median − 40 dB (link không dùng được); dev/test 8 000 mẫu cố định. Pilot y = Wᴴh + n (beam
training đường xuống), Np ∈ {4, 8, 16}, DFT một phần / ngẫu nhiên, SNR tham chiếu −5/5/15 dB (chuẩn hóa toàn cục theo
median train). Quyết định: codebook DFT 16 beam, rate log2(1 + SNR·|wᴴh|²), outage SNR < 0 dB. 3 seed (khởi tạo mạng,
nhiễu, beam ngẫu nhiên). Mô hình học: MLP 4×256, 3 000 bước AdamW, batch 512, cùng đầu vào (y, ước lượng LMMSE, SNR);
lấy mẫu S = 16. Chọn (quy tắc, NFE) trên dev, đọc trên test.

Rate / outage trên test (trung bình 3 seed), một số cấu hình:

| pilot | Np | SNR | LS | LMMSE | GMM | hồi quy | Gaussian có đk | DDPM | flow | bridge (SB) | bridge σ=0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 4 | −5 | 1.05 / 0.743 | 1.88 / 0.502 | 1.19 / 0.678 | **2.14 / 0.450** | 2.10 / 0.462 | 1.71 / 0.571 | 2.11 / 0.458 | 2.12 / 0.458 | 2.13 / 0.451 |
| random | 4 | +5 | 3.14 / 0.486 | 4.20 / 0.362 | 3.18 / 0.434 | **5.06 / 0.266** | 5.05 / 0.267 | 4.34 / 0.354 | 5.01 / 0.272 | 5.02 / 0.270 | 4.98 / 0.275 |
| random | 4 | +15 | 5.49 / 0.287 | 6.56 / 0.276 | 4.80 / 0.300 | 7.72 / 0.229 | **7.79 / 0.200** | 6.96 / 0.265 | 7.71 / 0.228 | 7.72 / 0.226 | 7.60 / 0.240 |
| dft | 4 | +5 | 2.97 / 0.410 | 3.75 / 0.334 | 3.60 / 0.394 | **4.25 / 0.331** | 4.19 / 0.341 | 3.45 / 0.402 | 4.20 / 0.335 | 4.19 / 0.330 | 4.09 / 0.336 |
| dft | 16 | +15 | **8.02 / 0.144** | 8.01 / 0.145 | 7.90 / 0.185 | 7.90 / 0.181 | 7.96 / 0.152 | 7.67 / 0.213 | 7.94 / 0.166 | 7.90 / 0.177 | 7.92 / 0.173 |

Bảng đủ 18 cấu hình: `python summarize.py a1 runs/wireless_abc/a1/a1_results.json`.

**Gate SB (≥ 5% rate hoặc ≥ 10% outage so với đối chứng mạnh nhất ở NFE ≤ của bridge, cùng chiều 3 seed): 0/18.**
Bridge kém đối chứng tốt nhất ở 16/18 cấu hình (−0.4…−6.1% rate), ngang ở 2.
1. **Mô hình học có ích; mô hình sinh không thêm gì.** Với 4 pilot, hồi quy hơn LMMSE rõ (random 5 dB: rate 5.06 vs 4.20,
   outage 0.266 vs 0.362). Bridge, flow, Gaussian có điều kiện và hồi quy gần bằng nhau; bridge chọn trên dev hầu hết là
   "mean, NFE 1", tức hoạt động như mạng hồi quy (khớp nhận định của *Rethinking Flow and Diffusion Bridge Models*).
   Bridge σ = 0 kém σ = 1 một chút ở Np = 4 (≤ 0.12 rate).
2. **Uncertainty chưa dùng được cho quyết định rủi ro.** Coverage khoảng 80% / 95% của beam gain (Np = 4, test): LMMSE
   0.24–0.31 / 0.35–0.42; GMM 0.20–0.47 / 0.31–0.60; Gaussian có đk 0.20–0.57 / 0.32–0.67; flow@8 0.47–0.66 / 0.61–0.74;
   bridge@8 0.42–0.68 / 0.56–0.78. Mọi phương pháp sai > 5 điểm phần trăm → theo plan, không dùng cho quyết định rủi ro.
   Quy tắc "min outage" không được chọn trên dev ở cấu hình nào.
3. **Đủ pilot, SNR cao: cổ điển thắng** (LS/OMP ở Np = 16, 15 dB).
4. **Chưa đủ bằng chứng (không phải kết quả âm):** DDPM chưa hội tụ sau 3 000 bước (NMSE +5…+10 dB); GMM 32 thành phần fit
   kém trên dữ liệu không dừng (NMSE ≈ 0 dB, kém LMMSE).
5. NMSE (random Np 4, 5 dB, seed 0): hồi quy −13.9, Gaussian −13.9, bridge@1 −13.6, flow@4 −13.0, LMMSE −3.1 dB.
   Độ trễ (ms/mẫu, GPU): hồi quy 0.002, bridge NFE 1/8 0.004/0.019, flow NFE 1/8 0.004/0.017.

**Kiểm tra hội tụ** (`runs/wireless_abc/a1_long/`, 15 000 bước thay vì 3 000, Np = 4, cả hai loại pilot, 3 seed; GPU 15
phút): gate SB **0/6**; bridge giờ bằng hồi quy (−1.5…+0.5% rate, không cấu hình nào cùng chiều thắng ở cả 3 seed);
flow và Gaussian có đk cũng ngang. DDPM vẫn chưa hội tụ (NMSE +0.6…+6.5 dB) nhưng coverage khá hơn (≈ 0.65 / 0.76) —
vẫn dưới danh định. Kết luận A không đổi khi huấn luyện lâu hơn.

→ **A: không có bằng chứng SB có lợi** cho ước lượng, uncertainty hay tốc độ ở bài toán này. B/C dùng LMMSE hoặc hồi quy.

## B — thông tin hiện tại từ UAV khác (xong; gate: **qua một phần, có điều kiện**)

`b_coop.py`, `runs/wireless_abc/b/b_results.json` (CPU, 30 phút). Ở 28 GHz kênh phức đổi sau ~1 cm nên chỉ chia sẻ được
cấu trúc góc: mọi phương pháp là LMMSE với hiệp phương sai prior khác nhau — global (train), **map** (kênh thật của train
trong bán kính 5 m — bản đồ khảo sát offline), **coop** (map trộn với ước lượng của UAV khác trong 5 m, ≤ 10 s trước,
nhân quả), **shuffle** (message từ episode khác, cùng chi phí), **oracle** (kênh thật của UAV khác — chẩn đoán). Hệ số
trộn chọn trên dev. Message: Np số phức 2×8 bit + vị trí 48 bit + thời gian 16 bit; trung bình 13.6 message được dùng
cho mỗi quyết định ở bay đoàn (≈ 1.3–1.7 kbit; không tốn thêm pilot).

Bay đoàn (test, 3 seed; rate / outage):

| pilot | Np | SNR | global | map | coop | shuffle | oracle | Δrate | Δoutage (tương đối) |
|---|---|---|---|---|---|---|---|---|---|
| dft | 4 | +5 | 3.46 / 0.365 | 4.88 / 0.247 | 5.10 / 0.222 | 4.88 / 0.247 | 5.35 / 0.199 | +4.4% | +10.3% |
| random | 4 | −5 | 1.59 / 0.586 | 2.39 / 0.365 | 2.62 / 0.295 | 2.40 / 0.363 | 2.76 / 0.263 | +9.7% | +19.3% |
| random | 4 | +5 | 4.11 / 0.375 | 5.00 / 0.257 | 5.24 / 0.226 | 5.01 / 0.257 | 5.34 / 0.206 | +4.7% | +12.0% |
| random | 8 | +15 | 7.45 / 0.235 | 8.14 / 0.154 | 8.26 / 0.147 | 8.14 / 0.154 | 8.38 / 0.125 | +1.4% | +4.3% |

- Coop hơn map ở **12/12** cấu hình bay đoàn, cùng chiều cả 3 seed (+1.4…+9.7% rate, +4.3…+19.3% outage); qua gate
  (≥ 5% rate hoặc ≥ 10% outage) ở **6/12**. Shuffle = map: lợi ích đến từ đúng thông tin. Bay độc lập: hầu như không
  có message trong 5 m → coop = map (đối chứng đúng).
- **Lợi ích chỉ ở nơi bản đồ chưa phủ** (gộp mọi cấu hình, quy tắc outage, bay đoàn, test):

  | vùng | tỉ lệ điểm | map | coop | oracle |
  |---|---|---|---|---|
  | bản đồ có phủ, có message | 41% | 5.64 / 0.173 | 5.61 / 0.169 | 5.70 / 0.161 |
  | bản đồ **không** phủ, có message | 29% | 4.22 / 0.328 | **4.65 / 0.245** | 5.06 / 0.183 |

  Trong cảnh tĩnh, hợp tác **thay cho bản đồ ở vùng chưa khảo sát** (+10% rate, −25% outage); ở vùng đã có bản đồ, nó
  không thêm gì. Oracle còn hơn coop ở vùng chưa phủ (5.06 vs 4.65) → còn dư địa cho fusion tốt hơn (attention/GNN),
  nhưng chỉ trong chế độ này.
- **Fusion học được (attention nhỏ, `b_attn.py`, `runs/wireless_abc/b_attn/`, GPU 10 phút; Np = 4, cả hai loại pilot,
  3 SNR, 3 seed; huấn luyện trực tiếp theo rate kỳ vọng trên episode train, map của điểm train loại chính episode đó):**
  gộp test bay đoàn (rate / outage; `b_join.py`):

  | vùng | map | trộn hiệp phương sai | attention | attention không message | oracle |
  |---|---|---|---|---|---|
  | bản đồ không phủ, có message | 3.92 / 0.365 | 4.62 / 0.270 | **4.65 / 0.274** | 4.45 / 0.310 | 5.27 / 0.188 |
  | bản đồ có phủ, có message | 5.69 / 0.175 | 5.72 / 0.172 | 5.19 / 0.235 | 5.07 / 0.242 | 5.75 / 0.172 |
  | không bản đồ, không message | 4.00 / 0.369 | = | 4.55 / 0.318 | 4.43 / 0.322 | = |

  Attention không thu hẹp khoảng cách tới oracle ở vùng chưa có bản đồ (4.65 ≈ trộn hiệp phương sai 4.62; oracle 5.27);
  ở vùng có bản đồ nó kém LMMSE-với-map (chỉ thấy hồ sơ beam, không thấy đủ hiệp phương sai). Message vẫn có giá trị
  trong mạng học (+0.20 rate so với cùng mạng không message). Kết luận: lợi ích hợp tác ổn định qua hai cách fusion;
  fusion phức tạp hơn chưa đem thêm gì.
- Chưa kiểm: blocker động, 2 vs 3 UAV riêng.

## C — quản lý beam/pilot đo bằng tracking (xong; gate mở PPO: **không qua**)

`c_feas.py` (`runs/wireless_abc/c/`, 24 episode test), `uav3d_comm/proto/c_track.py` (`runs/uav3d_proto/wireless_c/`).
Đường lên UAV → BS, một RF chain, 100 MHz, UAV 23 dBm, nhiễu −87 dBm; quét beam = burst SSB 16 beam (0.57 ms) với SNR đo
từ quỹ đường truyền SSB (BS 30 dBm, 240 sóng mang con, 960 RE; cao hơn SNR dữ liệu 42.2 dB — bản chạy đầu dùng nhầm
SNR dữ liệu cho phép đo, đã sửa trước khi đọc kết quả); bước điều khiển 10 ms, kênh trong bước tiến hóa theo Doppler
từng path (mỗi 1 ms); payload 10.3 kbit/helper/frame 200 ms; phục vụ luân phiên. Độ bền beam (beam tốt nhất còn trong
3 dB): trung vị 0.1 s, p10 0.01 s, p90 0.79 s.

| chính sách | giao đúng hạn (mọi frame) | frame LoS | frame NLoS | AMOTA UAV3D (3 seed) |
|---|---|---|---|---|
| gửi đủ (không kênh) | 1.000 | — | — | 0.690 |
| oracle (CSI thật mỗi 1 ms) | 0.689 | 1.000 | 0.391 | 0.632 ±0.028 |
| quét SSB 20 ms | 0.686 | 1.000 | 0.385 | 0.630 ±0.026 |
| quét 20 ms, trung bình trượt | 0.682 | 0.999 | 0.378 | 0.629 ±0.024 |
| quét 200 ms | 0.636 | 0.999 | 0.289 | 0.606 ±0.024 |
| beam từ GPS (không pilot) | 0.631 | 0.999 | 0.278 | 0.600 ±0.028 |
| một lần quét đầu episode | 0.572 | 0.984 | 0.178 | 0.579 ±0.014 |

Bước điều khiển 5 ms cho cùng kết quả (oracle 0.690, quét 20 ms 0.685).
1. **Quét SSB 20 ms gần bằng oracle** (0.630 vs 0.632 AMOTA; giao gói 0.686 vs 0.689). Khoảng cải thiện tối đa của mọi
   cơ chế quản lý beam/pilot trong cấu hình này là ~0.002 AMOTA < 0.01 → **không mở PPO/bandit** (đúng cổng của plan).
2. Mất mát thật đến từ **vùng phủ**: oracle vẫn mất 0.058 AMOTA so với gửi đủ, vì ~61% frame NLoS không có đường đủ mạnh
   với bất kỳ beam nào. Không chính sách beam nào lấy lại được phần này; cần đổi vị trí UAV, relay hoặc tần số — điều
   khiển bay/relay, ngoài phạm vi vòng này.
3. Beam từ GPS (không tốn pilot) chỉ kém quét 20 ms 0.03 AMOTA, gần bằng quét 200 ms.
4. Đây là mô phỏng ghép trace (helper UAV3D j ↔ UAV trace j−1), chưa có đồng đăng ký hình học.

## Lặp lại ở cảnh thứ hai: Étoile (Paris)

Trace `data/wireless_abc/trace_etoile/` (186 MB, seed 1): cảnh `etoile` của Sionna, nóc nhà trung vị 19 m (p90 25.6 m),
BS (150, 0, 28) m trên nóc nhà 13 m, cùng cấu hình UAV/quỹ đạo/solver như Munich. LoS 55–58%, không có đường 2% (nhiều
mặt tường phản xạ hơn Munich). Kiểm tra mô hình mảng: beam GPS trùng beam tốt nhất 95.4% mẫu LoS.

**B (Étoile):** coop hơn map ở **12/12** cấu hình bay đoàn, cùng chiều 3 seed (+2.1…+6.8% rate, +2.6…+10.5% outage);
qua gate **5/12**; shuffle = map. Theo vùng (bay đoàn, test, quy tắc outage):

| vùng | tỉ lệ | map | coop | oracle |
|---|---|---|---|---|
| bản đồ có phủ, có message | 41% | 3.94 / 0.338 | 3.89 / 0.339 | 3.93 / 0.333 |
| bản đồ **không** phủ, có message | 27% | 1.43 / 0.722 | **1.79 / 0.623** | 2.28 / 0.491 |

→ Cùng kết luận như Munich: hợp tác chỉ có lợi ở vùng chưa có bản đồ (+25% rate, −14% outage), không thêm gì ở vùng đã
có bản đồ; còn khoảng cách lớn tới oracle.

**C (Étoile):** giao đúng hạn oracle 0.906 / quét 20 ms 0.904 / trung bình trượt 0.899 / quét 200 ms 0.868 / GPS 0.835 /
một lần quét 0.764 (NLoS giao được 84% nhờ phản xạ). AMOTA UAV3D (3 seed): gửi đủ 0.690; oracle 0.664 ±0.036; quét 20
ms trung bình trượt 0.665 ±0.035; quét 20 ms 0.658 ±0.044; quét 200 ms 0.649; GPS 0.658; một lần quét 0.631. Heuristic
tốt nhất ≥ oracle → **không mở RL** (cùng kết luận). Beam GPS kém hơn ở Étoile (dùng phản xạ nhiều hơn).

## Bảng quyết định cuối vòng pilot (dev/feasibility, chưa phải bằng chứng tổng quát)

| câu hỏi | kết quả | quyết định |
|---|---|---|
| A: cần mô hình sinh/SB? | Không: SB 0/18 gate (0/6 khi huấn luyện 15k bước); hồi quy ≈ flow ≈ bridge; uncertainty sai coverage > 5 điểm; trong thiết lập của paper SB-ISAC, hồi quy cùng mạng thắng bridge 1 bước ở SNR ≥ −5 dB | Dừng SB cho CSI trong cấu hình này; B/C dùng LMMSE/hồi quy |
| B: hợp tác trực tuyến có lợi? | Có, **chỉ ở vùng chưa có bản đồ** (Munich +10% rate / −25% outage; Étoile +25% / −14%); 0 ở vùng đã có bản đồ; vượt message tráo; lặp lại ở 2 cảnh; fusion học được không thêm gì | Hướng có điều kiện: "bầy UAV tự dựng bản đồ góc khi chưa có khảo sát". Cần thêm fusion học được (còn khoảng cách tới oracle), nhiều cảnh, chi phí bit chặt hơn |
| C: cần Deep RL cho beam/pilot? | Không: quét 20 ms ≈ oracle (Munich Δ 0.002 AMOTA; Étoile heuristic tốt nhất ≥ oracle) | Không mở PPO. Nếu tiếp C, đổi biến quyết định sang vị trí/relay (vùng phủ) |

Tài nguyên đã dùng: GPU 31 phút (A1) + 15 phút (A1 hội tụ) + 10 phút (B attention) + 20 phút (A0b) ≈ 1 giờ 16 phút / trần 8 giờ; dữ liệu mới 4.6 GB (Fesl) + 240 MB trace (Munich + Étoile); env 5.4 GB (không tính vào dữ liệu).

### Việc tiếp theo (đề xuất, chờ người dùng/Codex)

1. B là hướng duy nhất qua gate (có điều kiện). Bước kiểm tra kế: nhiều cảnh Sionna (không chỉ Munich), bản đồ phủ ít/nhiều
   khác nhau, fusion học nhỏ (attention) so với trộn hiệp phương sai, và đo chi phí bit chặt (lượng tử hiệp phương sai /
   chọn message nào gửi — đây là chỗ SemCom có thể quay lại: chọn bằng chứng kênh nào đáng gửi).
2. C: nếu muốn tiếp, câu hỏi đúng là **vị trí/relay** cho vùng NLoS (oracle beam vẫn mất 0.058 AMOTA), không phải beam.
3. A: chỉ quay lại SB nếu có chế độ posterior đa đỉnh thật (pilot rất ít + lượng tử ADC thô) và đã sửa calibration; hiện
   chưa có lý do.


## Vòng sửa theo audit (05/11 trưa) — kết quả thay cho các mục đêm qua

### Sửa và kiểm thử
- GMM: bước covariance dùng E[d dᴴ] (trước: liên hợp ngược). Test mới khôi phục covariance phức đã biết, kể cả dấu phần ảo.
- Coverage: mô hình lấy mẫu dùng S = 64 mẫu (trước 16); LMMSE/GMM dùng 512 mẫu posterior chính xác. Test mới: với
  posterior Gaussian biết trước, khoảng 80% / 95% đạt danh định (±0.02); 16 mẫu cho coverage 95% thấp hơn rõ — tức một
  phần "coverage thấp" đêm qua là lỗi evaluator.
- A1 thêm đánh giá **toàn bộ link** của episode test (không lọc theo kênh thật) và tách riêng link sâu (deep NLoS).
- B v2 (`b_coop2.py`): codec thật — gửi Np phép đo pilot lượng tử 8 bit + scale float16 + vị trí + thời gian + chỉ số
  nhiễu = **152 bit/message** (Np = 4); bên nhận giải mã và tự dựng lại ước lượng LMMSE từ bản đồ chung tại vị trí người
  gửi với nhiễu đã cộng nhiễu lượng tử (test round-trip). Baseline mới **own** = bản đồ + lịch sử ước lượng của chính UAV
  (cùng bán kính / tuổi / trọng số). Có mất gói, chu kỳ gửi và mức phủ bản đồ.
- C v2 (`c_feas2.py`, `c_track2.py`): phí quét cho mọi chính sách có quét; payload là kích thước gói codec thật của từng
  helper/frame (trung bình 10.27 kbit, tối đa 15.36 kbit); message chỉ dùng được sau khi truyền xong (hạn chót fusion
  LAT = 0 hoặc 50 ms sau mốc frame); thêm lập lịch greedy và planner chẩn đoán biết tương lai.
- Test: 18/18 qua (thêm GMM covariance, evaluator coverage, codec round-trip).

### B v2 — Munich, Np = 4 (`runs/wireless_abc/b2/`, `b2_summ.py`)

Bay đoàn, test, 3 seed (rate / outage; trọng số trộn chọn trên dev):

| pilot | SNR | map | own (map + lịch sử riêng) | coop | shuffle | oracle | coop vs own: Δrate | Δoutage |
|---|---|---|---|---|---|---|---|---|
| dft | −5 | 2.57 / 0.287 | 2.59 / 0.285 | 2.62 / 0.275 | 2.59 / 0.285 | 2.81 / 0.234 | +1.3% | +3.5% |
| dft | +5 | 4.88 / 0.247 | 5.08 / 0.221 | 5.16 / 0.211 | 5.08 / 0.221 | 5.39 / 0.193 | +1.5% | +4.2% |
| dft | +15 | 7.52 / 0.171 | 7.75 / 0.159 | 7.88 / 0.150 | 7.75 / 0.159 | 8.16 / 0.137 | +1.6% | +5.7% |
| random | −5 | 2.39 / 0.366 | 2.63 / 0.290 | 2.70 / 0.271 | 2.63 / 0.290 | 2.85 / 0.235 | +2.5% | +6.6% |
| random | +5 | 5.00 / 0.257 | 5.19 / 0.233 | 5.28 / 0.221 | 5.19 / 0.233 | 5.46 / 0.194 | +1.7% | +4.9% |
| random | +15 | 7.69 / 0.178 | 7.85 / 0.168 | 7.96 / 0.162 | 7.85 / 0.168 | 8.23 / 0.138 | +1.5% | +3.5% |

1. **Phần lớn lợi ích "hợp tác" đêm qua là lợi ích của bộ nhớ:** baseline own (lịch sử của chính UAV) đã lấy phần lớn.
   Coop vs own còn +1.3…+2.5% rate, +3.5…+6.6% outage, cùng chiều cả 3 seed ở 6/6 cấu hình; gate B (≥ 5% rate hoặc
   ≥ 10% outage) **0/6**. Shuffle = own (lợi ích còn lại đến từ đúng thông tin).
2. Sau codec thật (152 bit/message, ~13.6 message dùng cho mỗi quyết định, ~1.5 kbit/s mỗi UAV ở 10 Hz) lợi ích vẫn còn;
   bền với mất gói 30% (+1.6% / +4.6%) và gửi thưa 5 lần (+1.1% / +2.8%).
3. Vùng bản đồ không phủ (bay đoàn, gộp): own 4.38 / 0.299 → coop 4.64 / 0.267 (+6% rate, −11% outage); oracle 5.30 /
   0.180. Ở vùng có bản đồ: không thêm gì.
4. **Khi bản đồ chỉ phủ 25% hoặc không có bản đồ, hợp tác ≈ own** (random 5 dB: 4.44 vs 4.42; 4.18 vs 4.18), dù oracle
   vẫn cao hơn nhiều (5.20; 5.11). Giải thích: lịch sử và message hữu ích chủ yếu khi chúng được tạo ở chỗ vừa có bản đồ
   (UAV mang ước lượng tốt từ vài mét trước vào vùng chưa phủ); ước lượng từ 4 pilot với prior toàn cục quá nhiễu để dựng
   hiệp phương sai. **Thông tin có tồn tại (oracle) nhưng message hiện tại (4 phép đo nhiễu) quá yếu để chuyển nó** —
   đây chính là câu hỏi "gửi gì" của SemCom, chưa được giải.
5. Kết luận B sau sửa: tín hiệu hợp tác **nhỏ nhưng ổn định** sau chi phí; chưa qua gate; dư địa lớn nằm ở biểu diễn
   message, không ở fusion (attention đêm qua cũng không hơn).

### C v2 — Munich (`runs/wireless_abc/c2/`, `runs/uav3d_proto/wireless_c2_lat*/`)

Giao đúng hạn trên 24 episode test (payload thật, phí quét cho mọi chính sách): oracle-rr 0.689, ssb20-rr 0.685,
ssb20avg-rr 0.681, ssb200-rr 0.637, gps-rr 0.632, stale-rr 0.573; **oracle-greedy 0.712**, oracle-plan 0.707,
**ssb20avg-greedy 0.693**, ssb20avg-plan 0.697.

AMOTA UAV3D (3 seed; 'all_age0' = không kênh, như v1):

| chính sách (beam-lịch) | giao (ánh xạ) | AMOTA, LAT = 50 ms | AMOTA, LAT = 0 |
|---|---|---|---|
| gửi đủ, tuổi 0 | 1.000 | 0.690 | 0.690 |
| gửi đủ, tuổi 1 | 1.000 | 0.631 | 0.631 |
| oracle-greedy (chẩn đoán) | 0.589 | **0.629** ±0.020 | 0.582 |
| oracle-plan (chẩn đoán) | 0.574 | 0.617 | 0.578 |
| oracle-rr | 0.544 | 0.612 | 0.573 |
| ssb20avg-greedy (causal) | 0.556 | **0.621** ±0.022 | 0.579 |
| ssb20avg-rr | 0.531 | 0.610 | 0.570 |
| ssb20-rr (v1) | 0.539 | 0.607 | 0.574 |
| ssb200-rr | 0.469 | 0.593 | 0.557 |
| gps-rr | 0.472 | 0.595 | 0.559 |
| stale-rr | 0.415 | 0.565 | 0.529 |

1. Sửa thời điểm nhận làm mọi số thấp hơn v1 (v1 cho message tuổi 0); thứ tự chính sách beam không đổi.
2. **Lập lịch có giá trị:** greedy hơn round-robin +0.011 AMOTA (causal) và +0.017 (oracle beam). Heuristic causal mạnh
   nhất (ssb20avg-greedy, 0.621) cách đối chứng biết kênh tốt nhất (oracle-greedy, 0.629) **0.008** < 0.01.
3. Planner chẩn đoán (least-laxity) **kém** greedy → nó không phải cận trên; chưa có cận trên thật cho bài toán lập lịch.
   Vì vậy: "chưa có bằng chứng cần PPO cho beam + lập lịch trong cấu hình này", **không** phải "RL không có đất".
4. Mất mát lớn nhất vẫn là vùng phủ và tuổi message: ngay cả gửi đủ với trễ 1 frame đã mất 0.059 AMOTA so với tuổi 0.

### A1 v2 — GMM đã sửa, coverage đúng, toàn bộ link (`runs/wireless_abc/a1_v2/`, `summary_{test,test_all,test_deep}.md`)

Cùng trace, cùng seed và cùng 3000 bước như v1. Chọn rule/NFE trên dev, đọc trên test. GPU ≈ 2 giờ 25 phút.

1. **Gate SB: 0/18** trên test lọc link (như v1) và **0/18 trên toàn bộ link** của episode test, không lọc theo kênh
   thật. Bridge thua hoặc hòa phương pháp mạnh nhất khác ở mọi cấu hình. Ở SNR 15 dB, Np ≥ 8, bridge thua rõ về outage
   (random Np = 16: 0.193 so với GMM 0.150, tức −28%).
2. **GMM sau khi sửa là baseline mạnh.** Ví dụ random, Np = 4, 15 dB: 4.80 / 0.300 (v1, có lỗi) thành 7.77 / 0.203 (v2),
   tốt nhất về outage. Baseline cổ điển mạnh hơn làm khoảng cách với SB lớn thêm, không thu hẹp. Số của các mô hình học
   (hồi quy, flow, bridge) gần như không đổi so với v1.
3. **Toàn bộ link:** 33.9% mẫu link trong episode test có công suất thấp hơn trung vị hơn 40 dB (link "sâu": NLoS sâu hoặc
   không có đường truyền). Trên các link này, ngay cả beam DFT tốt nhất với CSI hoàn hảo cũng chỉ đạt 4·10⁻⁵ … 4·10⁻³
   bit/s/Hz, outage 100%. **Không bộ ước lượng kênh nào cứu được chúng**; đây là bài toán vùng phủ (vị trí/relay/độ cao),
   không phải bài toán ước lượng. Bảng test_deep chỉ để ghi nhận: chênh lệch phần trăm giữa các rate ≈ 0 không có nghĩa,
   và `summarize.py` không tính gate khi rate genie < 0.01.
4. **Calibration** (Np = 4, test lọc link; khoảng 80% / 95% của gain mỗi beam; danh định 0.80 / 0.95):

   | phương pháp | v1 (16 mẫu, GMM lỗi) | v2 (64 mẫu; LMMSE/GMM 512 mẫu posterior chính xác) |
   |---|---|---|
   | LMMSE (hiệp phương sai toàn cục) | 0.24–0.31 / 0.35–0.42 | 0.28–0.35 / 0.48–0.54 |
   | GMM | 0.20–0.47 / 0.31–0.60 | 0.64–0.67 / 0.81–0.83 |
   | Gauss dị phương sai | 0.20–0.57 / 0.32–0.67 | 0.22–0.61 / 0.42–0.73 |
   | flow @8 | 0.47–0.66 / 0.61–0.74 | 0.52–0.70 / 0.69–0.79 |
   | bridge @8 | 0.42–0.68 / 0.56–0.78 | 0.46–0.73 / 0.64–0.83 |
   | DDPM @8 (chưa hội tụ) | 0.27–0.36 / 0.45–0.50 | 0.34–0.42 / 0.60–0.62 |

   Một phần coverage thấp của v1 là do evaluator (16 mẫu): ở mức 95%, các phương pháp lấy mẫu tăng khoảng 0.05–0.15. Sau
   khi sửa, mọi phương pháp vẫn **thiếu coverage** ở mức 95%: GMM và bridge@8 tốt nhất vẫn thiếu khoảng 0.12, LMMSE
   thiếu khoảng 0.45. Kết luận "chưa dùng uncertainty cho quyết định rủi ro"
   vẫn giữ. bridge@1 / bridge0 là ước lượng điểm, không có coverage. Trên toàn bộ link coverage còn thấp hơn, vì link sâu
   nằm ngoài phân phối train.

### Bảng quyết định sau vòng audit (thay bảng đêm qua; dev/feasibility, chỉ Munich trừ khi ghi khác)

| câu hỏi | kết quả sau sửa | phát biểu được phép |
|---|---|---|
| A: SB cho CSI / chọn beam | 0/18 trên link lọc và 0/18 trên toàn bộ link; GMM đã sửa và hồi quy mạnh ngang hoặc hơn; mọi phương pháp thiếu coverage; link sâu (34%) không cứu được bằng ước lượng | "Chưa thấy lợi thế của SB trong benchmark hiện tại (ULA 16, DFT 16 beam, Np 4–16, 3000 bước)." **Không** nói "SB vô ích cho CSI". |
| B: hợp tác trực tuyến | So với baseline mạnh nhất (bản đồ + lịch sử riêng): +1.3…+2.5% rate, +3.5…+6.6% outage, cùng chiều 3 seed ở 6/6 cấu hình; gate 0/6; giữ được với codec 152 bit, mất gói 30%, gửi thưa 5 lần; vùng không có bản đồ +6% / −11%; bản đồ 25% hoặc không bản đồ: ≈ 0, trong khi oracle còn xa | "Tín hiệu hợp tác nhỏ nhưng ổn định sau chi phí; chưa qua gate. Dư địa nằm ở **gửi gì** (biểu diễn message), không ở fusion." |
| C: RL cho beam + lập lịch | Lập lịch greedy hơn round-robin +0.011 AMOTA; heuristic causal tốt nhất cách đối chứng biết kênh tốt nhất 0.008; planner chẩn đoán kém greedy nên chưa có cận trên thật; trễ 1 frame mất 0.059 | "Chưa có bằng chứng cần PPO cho beam + lập lịch trong cấu hình này; greedy đã lấy phần lớn." **Không** nói "RL không có đất". |

Tài nguyên tổng: GPU ≈ 1 giờ 16 phút (đêm) + 2 giờ 25 phút (A1 v2) ≈ 3 giờ 41 phút / trần 8 giờ. Dữ liệu mới 4.6 GB + 240 MB
(không đổi). Theo Codex: chưa thêm cảnh mới, chưa huấn luyện SB thêm, chưa mở PPO.

Câu hỏi mở cho vòng sau (chờ người dùng quyết):
1. B — biểu diễn message: gửi gì để chuyển được phần thông tin mà oracle cho thấy là có (ví dụ hướng/gain beam tốt nhất
   đã lọc, hệ số hiệp phương sai rút gọn, hay bằng chứng đã chọn theo độ bất định)? Đây là chỗ SemCom nối vào.
2. C — cận trên lập lịch thật (ví dụ quy hoạch nguyên/DP trên cửa sổ 200 ms với kênh biết trước), trước khi bàn đến RL.
3. Vùng phủ — 34% mẫu link là link sâu mà không ước lượng nào cứu được; nếu có RL thì biến quyết định nên là vị trí/độ
   cao/relay, không phải beam.


## Hybrid object–feature pilot (05/11 tối) — mốc 0 và cổng 1, phần receiver (chưa có wireless)

Plan: "SwarmBridge: Hybrid Object–Feature Communication cho multi-UAV tracking". Đã đóng vòng A/B/C; các artifact cũ giữ
nguyên. Code nằm trong `swarmbridge/experiments/hybrid_comm/`, kết quả trong `runs/hybrid_comm/`. File dự đoán không đưa vào
git vì tái tạo được từ cache BEV; SHA-256 của chúng có trong `det_summary.json`, `track_summary.json` và `hybrid_merge.json`.

### Mốc 0 — tài liệu và provenance (`runs/hybrid_comm/lit_manifest.json`)
- Đã xác minh 11 bài (arXiv ID, phiên bản, SHA-256, đều là PDF hợp lệ). Bài mới tải: HyComm (2508.07092), LIF
  (2504.21774), DeepSense 6G (2211.09769), HMS-SCP (2608.14603), HARQ cho cooperative perception (2409.09042), DRL lập lịch
  user cho collaborative perception (2502.10456). Bài đã có sẵn: JSAC (2412.05647), CoSDH, GOSC (2609.39477), CoDS, SComCP.
- Code chính thức đã clone nông, không kéo dataset/checkpoint: LIF (`uestchjw/LIF`, 805cd11c) và GOSC
  (`frezazadeh/gosc-agentic-teams`, 387b4938). Không tìm thấy code của JSAC, HyComm, HMS-SCP. Trong bản clone LIF không có
  code BoBEV/VPE.
- `Executive summary.pdf` (ChatGPT) chỉ dùng làm gợi ý. Không dùng làm bằng chứng các mục không có định danh hoặc sai, ví
  dụ "BridgeCodec NeurIPS 2022 (Mamba)", "DiffCP ICCV 2023", "Song et al. (hypothetical)".

**Đối chiếu novelty (đọc full text):**

| Công trình | Đã có | Hệ quả |
|---|---|---|
| JSAC (Huang et al.) | Chọn mode bằng DRL (DSAC, một agent), có deadline D_max 2–11 s, thời gian tính toán semantic 1.33 s; state chứa CSI thật; các mode độc lập, không lũy tiến (base+enhancement chỉ ghi là future work) | Chọn mode có deadline không còn mới |
| GOSC (Rezazadeh et al.) | Gửi/hoãn theo giá trị − giá (knapsack Lagrange); có điều kiện theo tri thức của bên nhận; có option dùng AoI; uplink chung điều phối bằng giá (nhường airtime); chỉ dùng SNR trung bình | Quyết định "nhường airtime" và tính causal đã có; mọi baseline phải gồm một quy tắc kiểu GOSC |
| CoSDH | Box + feature, gửi theo cung–cầu, ngưỡng cố định | Hybrid box+feature và gửi theo nhu cầu đã có |
| HyComm | Box trước, rồi điểm thô theo độ bất định của bên gửi, xếp theo ngân sách | Thứ tự "gọn trước, bằng chứng sau" đã có |
| LIF | Box từ UAV fusion vào BEV trên **UAV3D**: 0.718 mAP, gần early fusion 0.720; yêu cầu theo độ bất định của bên nhận | Box trên UAV3D đã rất mạnh ở mức detection |
| HMS-SCP, CoDS, SComCP | Feature đa tỉ lệ / bitstream digital LDPC / JSCC, đều trên một link, kênh AWGN/Rayleigh, chấm bằng AP | Codec feature theo kênh đã có |

Phần còn có thể mới (giả thuyết, chưa tuyên bố): bitstream lũy tiến với box là lớp 0 và tile là lớp bổ sung xuyên frame;
quyết định gửi mới / bổ sung cho nội dung đã gửi / nhường airtime, có điều kiện theo trạng thái tracker dựng từ dữ liệu đã
nhận; deadline cỡ ms trên trace ray-tracing; chấm bằng AMOTA/ID switch. Cần đọc kỹ 2409.09042 (HARQ) và 2502.10456 (lập lịch
DRL) trước khi tuyên bố phần "bổ sung" và "nhường airtime". Về cơ chế, lập lịch base/enhancement theo deadline là
streaming lũy tiến tối ưu R-D cổ điển (Chou–Miao).

### Codec (`tilecodec.py`, 6/6 test: round-trip, prefix, gói bị cắt, kênh hằng/bão hòa, pooling)
Tile 8×8 ô (6.4 m), 80 kênh, q8 theo kênh với tham số fp16. Byte gồm header, qparams, rồi lần lượt (index, tile). M2 đúng
là prefix byte của M3. Kích thước: **M2 = 658 kbit, M3 = 2.63 Mbit, 256 tile = 10.5 Mbit**; M1 (KF-PV) khoảng 10 kbit.
Thời gian (p50 / p95): điểm tile của bên gửi 8.2 / 9.4 ms, encode 14 / 17 ms, decode 1.8 / 5.7 ms, decoder BEV 9.7 / 11.4 ms.

### Cổng 1, phần receiver — 3 scene dev, mọi message tới đủ, tuổi 0 (`m0_receiver.py`, `hybrid_merge.py`)
Tile được xếp hạng theo objectness của detector cục bộ bên gửi (`obj`), đối chứng bằng norm feature (`norm`); top-64 của
hai cách chỉ trùng 36%. Receiver feature là decoder `fusion/feat_s1` (trung bình có mask); nhánh box là M1 KF-PV đồng bộ.

| receiver / message | mAP | AMOTA | ID switch |
|---|---|---|---|
| ego qua decoder feature | 0.428 | 0.472 | 46 |
| M1 box noisy-OR (không vận tốc) | 0.665 | 0.594 | 57 |
| **M1 KF-PV** | 0.649 | **0.690** | 29 |
| chỉ 64 tile `obj`, không box | 0.563 | 0.688 | 50 |
| M1 + 16 tile (M2), NMS-max | 0.651 | 0.706 | 44 |
| M1 + 64 tile (M3), NMS-max | 0.657 | 0.705 | 67 |
| M1 + 256 tile, NMS-max | 0.663 | 0.701 | 46 |
| M1 + feature đủ trong vùng 64 / 256 tile (chẩn đoán) | 0.661 / 0.665 | 0.716 / 0.708 | 42 / 41 |
| **feature đầy đủ (float / q8), không box** | 0.688 / 0.689 | **0.761** | 43 |

1. Codec đúng (q8 không mất gì). Xếp tile theo `obj` tốt hơn `norm` ở mọi K.
2. **Receiver chưa hỗ trợ input thưa.** Decoder feature chỉ có ego cho 0.428 mAP, kém decoder ego chuyên dụng (0.512),
   vì decoder được train trên trung bình của 5 UAV. Chia vùng (feature trong tile, box ngoài tile) **kém hơn M1**
   (0.49–0.51 mAP), vì thang score của hai nhánh lệch nhau: box có trung vị 0.66, nhiều detection ≈ 1; feature có trung vị
   0.41. Gộp rồi NMS-max chữa được phần lớn, nhưng kéo feature đầy đủ từ 0.688 xuống 0.665 mAP.
3. Ở mức detection, M2/M3 chỉ thêm ≤ 0.012 mAP so với M1. **Ở mức tracking, trần của lớp feature lớn: feature đầy đủ hơn
   M1 +0.071 AMOTA**; M2 +0.016; M3 +0.015. Nhiễu trên 60 frame vào khoảng ±0.01 (256 tile < 64 tile), nên +0.016 chỉ ở
   ngưỡng. ID switch tăng khi ghép (29 → 44–67).
4. Kết luận cổng 1 theo quy tắc của plan: **chưa kết luận — receiver còn hạn chế** (input thưa và quy tắc ghép). Lớp
   feature không vô ích: trần tracking là +0.07 AMOTA, đổi lại tốn khoảng 4000× bit. Việc hợp lý tiếp theo là **mốc 2**
   (plan cho phép tối đa 45 phút GPU, 3 epoch, 1 seed, chọn trên validation): một receiver nhận cùng lúc feature thưa
   (train với tập tile ngẫu nhiên), bản đồ box KF và kênh mask, để bỏ quy tắc ghép thủ công. Chưa dựng wireless, chưa mở RL.

GPU đã dùng cho pilot này ≈ 5 phút (2 lượt receiver 60 frame + smoke). Ngân sách còn ≈ 4 giờ 14 phút. Dữ liệu mới: PDF
53 MB, repo 677 MB, dự đoán 195 MB (ngoài git).

## Mốc 2 — receiver thích nghi có đối chứng (05–06/11 đêm; Codex triển khai, Claude chạy nốt) — cổng wireless: **không qua**
Thư mục `runs/hybrid_comm/codex_m2/` (`summary.md`, `summary.json`, hash của prediction/checkpoint/input tracking).
Tái lập bằng một lệnh: `bash scripts/hybrid_comm/run_codex.sh` (có resume; các bước đã xong được bỏ qua).

**Sửa trước khi train (trả lời audit của Codex và ChatGPT):**
- Phép ghép NMS-max đã đưa vào code (`core.nms_max`, `reproduce.py`). Prediction M2/M3/densein tái tạo **khớp hash** với
  artifact cũ.
- Bỏ việc chọn trục `xy`/`yx` bằng GT dev. Trục được cố định theo BEV pooling, và cả 400 frame có rotation trong metadata
  bằng identity (đã assert). `core.py` có phép biến đổi SE(2) tổng quát (`warp_grid`, `cells`) kèm test. Giả định vẫn còn:
  cache helper đã được chiếu sẵn vào grid UAV0 (`prepared/geometry.json`); đây chưa phải encoder phân tán độc lập.
- Split cố định (seed 20261105): 13 scene train, 4 validation (`scene_17/08/19/01`), dev 11/13/15 giữ nguyên. Checkpoint
  được chọn bằng loss validation. Lưu ý: decoder khởi tạo đã từng được fine-tune trên cả 17 scene non-dev.
- 13/13 test pass (codec, hình học, receiver).

**Thiết kế.** Hai receiver, cùng khởi tạo (decoder ego), cùng lịch dữ liệu, seed 0, 260 bước/epoch, 3 epoch, AdamW 2e-4.
Input có 172 kênh: ego, helper, 8 kênh thuộc tính box, mask và tuổi; mọi thứ được giải mã từ byte thật. Arm **box** luôn có
kênh feature bằng 0. Arm **joint** nhận thêm tile (0/16/64, phần lớn theo objectness, 20% ngẫu nhiên). Tuổi message là
0/1/2 frame với xác suất 0,6/0,3/0,1. Loss validation thấp nhất ở **epoch 0** cho cả hai arm (box 0,542; joint 0,533) và
tăng sau đó. Đối chứng CPU: hiệu chỉnh Platt confidence của hai nhánh **chỉ trên validation**, rồi NMS 2 m cố định.

| Phương pháp (dev, 60 frame) | bit/frame (4 helper) | mAP | AMOTA | IDS | s11 / s13 / s15 |
|---|---|---|---|---|---|
| box thích nghi, không helper | 0 | 0.512 | 0.542 | 69 | 0.589 / 0.496 / 0.533 |
| box thích nghi, M1 | 41 k | 0.554 | 0.596 | 41 | 0.640 / 0.554 / 0.588 |
| joint thích nghi, M1 | 41 k | 0.554 | 0.597 | 48 | 0.647 / 0.548 / 0.597 |
| joint thích nghi, M2 | 2.67 M | 0.573 | 0.634 | 60 | 0.685 / 0.567 / 0.639 |
| joint thích nghi, M3 | 10.5 M | 0.589 | 0.661 | 52 | 0.689 / 0.626 / 0.668 |
| joint, M2 tuổi 1 frame | 2.54 M | 0.570 | 0.636 | 68 | 0.675 / 0.596 / 0.630 |
| KF baseline (cùng packet, noisy-OR) | 41 k | 0.648 | 0.692 | 31 | 0.727 / 0.704 / 0.630 |
| **M1 + cùng NMS (đối chứng mạnh nhất)** | 41 k | 0.648 | **0.701** | 44 | 0.726 / 0.710 / 0.659 |
| M2 hiệu chỉnh trên validation + NMS | 2.67 M | 0.638 | 0.692 | 53 | 0.725 / 0.674 / 0.667 |
| M3 hiệu chỉnh trên validation + NMS | 10.5 M | 0.650 | 0.674 | 48 | 0.702 / 0.654 / 0.657 |
| feature đầy đủ q8 / float (chẩn đoán, không phải cận trên) | ~168 M (q8) | 0.689 / 0.688 | 0.763 / 0.761 | 49 / 43 | |

**Đính chính số cũ.** Khi áp **cùng phép NMS** lên M1, AMOTA tăng từ 0.690 lên 0.701 dù mAP không đổi. Vì vậy phần
"M2 +0.016 so với M1" báo ở mục trước **phần lớn là do NMS, không do tile**: M2 cũ (0.706) chỉ hơn M1 cùng NMS **+0.005**.
Con số "nhiễu ±0.01" trước đây không có phép ước lượng hỗ trợ, nên tôi rút lại. Cũng không gọi 0.761 là cận trên.

**Đọc kết quả.**
1. *Lợi ích từ fine-tune* (box thích nghi so với M1 cũ) là **âm**: 0.596 so với 0.701 AMOTA. Receiver học được kém hơn
   cách ghép box thủ công khoảng 0.10 AMOTA. Raster box vào lưới BEV có thể làm mất thông tin, nhưng chưa có ablation
   chứng minh đây là nguyên nhân. Kiến trúc, lịch train và lượng dữ liệu cũng có thể ảnh hưởng. Loss validation thấp
   nhất sau epoch đầu chỉ cho thấy train lâu hơn trong lịch hiện tại không cải thiện loss đó.
2. *Phần lợi riêng của tile* (joint so với box, cùng receiver): M1 ≈ 0 (kiểm tra đối chứng: không có tile thì hai arm
   bằng nhau); **M2 +0.038; M3 +0.065 AMOTA, dương ở cả 3 scene**. Như vậy tile có mang thông tin bổ sung thật. Nhưng mức
   tuyệt đối vẫn dưới đối chứng thủ công.
3. Cổng (≥ +0.01 so với đối chứng mạnh nhất, dương ở ≥ 2/3 scene): joint M2 −0.067, joint M3 −0.040, M2 hiệu chỉnh −0.010
   (chỉ dương ở scene 15), M3 hiệu chỉnh −0.028. **Không qua, nên không dựng wireless.**
4. Với hai receiver này, tuổi 1 frame không làm AMOTA gộp thấp hơn (M2 0.634 → 0.636; M1 box 0.596 → 0.610).
   Đây chưa phải bằng chứng tổng quát rằng message trễ không gây hại; cold start còn làm lượng message sử dụng khác nhau.
5. Timing inference cũ: chuẩn bị input M2/M3 p50 63–92 ms, p95 88–117 ms; decoder p50 15–23 ms.
   **Phần input còn gồm đọc cache BEV và packet từ đĩa.** Không được diễn giải nó là latency xử lý thuần hoặc
   bằng chứng về deadline end-to-end. Phép đo riêng dưới đây tách thời gian chép CPU–GPU và bỏ đọc đĩa.
6. Phần nào chưa đủ: **receiver** (học kém hơn ghép thủ công). Còn thông tin bổ sung thì có (mục 2). Cách chọn tile
   (objectness so với ngẫu nhiên) chưa được chấm riêng trên dev. Một seed train và 3 scene dev chỉ là bằng chứng feasibility.

**Thí nghiệm khác, được định nghĩa rõ (chưa chạy, chờ quyết định):** giữ nguyên nhánh box thủ công (M1 + NMS, 0.701),
chỉ dùng receiver joint làm *nguồn bằng chứng trong vùng tile*. Ghép detection joint-M2/M3 với box M1 bằng hiệu chỉnh Platt
fit trên validation, chỉ chạy CPU, không train lại. Nếu vẫn không đạt +0.01 so với 0.701 thì giữ box-only, và hướng hybrid
chỉ còn đáng làm khi có receiver mạnh hơn hẳn (ngoài phạm vi pilot).

Tài nguyên lượt này: GPU 21.4 phút (prepare + train 14.3 + inference 2.6). Tổng pilot hybrid ≈ 26 phút; còn ≈ 3 giờ 49
phút. Dữ liệu mới 1.9 GB ngoài git (message nhị phân, checkpoint).

### Codex xác nhận artifact và đính chính timing (06/11)

Đã kiểm tra lại 13 test và 60 cặp hash prediction/input tracking (15 phương án × gộp và ba scene), cùng `dt=0.2`.
Training và kết quả Claude chạy nốt được giữ nguyên; không chạy thêm ghép late-fusion, train hoặc wireless.

`bash scripts/hybrid_comm/run_matched.sh profile` đo đúng input builder hiện tại, dữ liệu validation đã ở RAM,
3 lượt warm-up và 12 mẫu mỗi mode. Profiling có đồng bộ CUDA; assert input trùng hoàn toàn với nhánh không profiling.

| Mode | Chuẩn bị input p50 / p95 (ms) | Chép CPU→GPU và cast ego p50 / p95 (ms) |
|---|---:|---:|
| Không helper | 4.56 / 26.05 | 3.66 / 6.42 |
| M1 | 9.50 / 12.34 | 3.99 / 5.00 |
| M2 | 48.31 / 51.98 | 20.01 / 25.70 |
| M3 | 42.05 / 45.97 | 17.66 / 21.48 |

Nguồn: `runs/hybrid_comm/codex_m2/input_profile.json`. Đây là microbenchmark trên ba frame validation, chưa gồm encoder,
encode bên gửi, radio hoặc decoder receiver. Không cộng các percentile giữa công đoạn, không suy ra M3 nhanh hơn M2
một cách tổng quát và chưa tuyên bố đạt/trượt deadline end-to-end. Microbenchmark mất thêm 17,1 giây, không thêm training;
tổng các phase GPU được ghi nhận tăng từ 21,4 lên 21,7 phút.

**Quyết định giữ nguyên:** dùng M1 + NMS làm baseline hiện tại; dừng mở wireless hybrid và PPO vì không đạt gate.
Lợi ích tile trong cặp receiver yếu là tín hiệu feasibility, chưa chứng minh lợi ích so với pipeline box mạnh.
Nếu mở thí nghiệm mới sau này, cần giả thuyết và đối chứng riêng; chưa có kết luận receiver là nút thắt duy nhất.

## Protocol ứng viên mới v0.1 — UAV mang ảnh, điều khiển vị trí và truyền bằng chứng (06/11)

**Trạng thái:** đã chọn tác vụ/dataset/codec để chuẩn bị pilot; **chưa ready-to-run**. Chưa tải VisDrone/checkpoint,
chưa inference, training, ray tracing mới hoặc RL. Không mở lại hybrid tracking. Cấu hình:
`runs/active_recovery/protocol-v0.1.json`; kiểm tra tài sản: `runs/active_recovery/readiness.json`.
Đây là protocol ứng viên có thể bác bỏ, chưa phải novelty hoặc đề tài chính đã được xác lập.

### Bộ ba được chọn

| Thành phần | Lựa chọn và giới hạn |
|---|---|
| Tác vụ | Ground station phát hiện **car/bus/truck** trong ảnh UAV đã chụp. UAV không thu ảnh mới khi bay. |
| Dataset | **VisDrone2019-DET validation** (nguồn tác giả). Dùng ảnh thật và annotation; không lấy prediction làm GT. |
| Mô hình tác vụ | YOLOv8n ở UAV, YOLOv8l ở trạm, checkpoint COCO chính thức, đóng băng, cùng `imgsz=1280`, batch 1. Khác biệt chất lượng phải đo; không mặc định model lớn hơn luôn tốt hơn trên VisDrone. |
| Codec | Pillow JPEG: bản cơ sở Q10, toàn ảnh Q70, patch gốc Q85; thêm progressive JPEG Q70 làm đối chứng. Tất cả đi qua bytes encode/decode. |

VisDrone car/bus/truck có ID 4/9/6; COCO có ID 2/5/7 (zero-based). Không tự gộp van vào car.
Giữ quy tắc ignored region của evaluator VisDrone; chốt conversion/evaluator bằng ví dụ kiểm tay trước khi chấm.
Resize giữ tỷ lệ, cạnh dài tối đa 1280, không phóng lớn ảnh nhỏ; grid patch 256 pixel trong canvas này, biên được clip.
Receiver ghép pixel patch vào bản cơ sở rồi chạy lại cùng detector; chưa dùng ghép score của hai detector.
**Không có decoder generative trong pilot v0.1.** Không gọi JPEG/patch này là generative recovery hay semantic HARQ đã tái lập.

### Khóa dữ liệu và phép đo trước khi chạy

- Chia theo nhóm sequence/flight từ filename và metadata được xác minh, seed `20261106`, tỷ lệ nhóm 20/40/40 cho
  calibration/dev/holdout. Nếu filename không đủ xác định nhóm, dùng mapping chính thức; chưa rõ thì dừng, không chia
  ngẫu nhiên từng ảnh. Lưu ID, nhóm và SHA-256 trước khi xem metric. Không hứa số ảnh mỗi split bằng đúng tỷ lệ.
- Chỉ calibration dùng chọn threshold/proxy; dev dùng gate. Holdout chưa mở cho đến khi khóa phương án. Pilot không train detector.
- Báo AP50, AP50:95 và recall riêng ba lớp; utility theo ảnh cho controller là F1 ở IoU 0.5, confidence 0.25 cố định.
  GT chỉ phục vụ chấm/offline diagnostic; không đưa F1 thật vào observation, feedback hay điều kiện dừng online.
- Đo cả YOLOv8n và YOLOv8l trên ảnh gốc trước. Nếu evaluator/model không có chất lượng đủ để phân biệt các mode,
  hoặc model trạm không đem lại lợi ích, ghi nhận thất bại của bộ ba; không âm thầm fine-tune hay đổi model trên dev.

### Cổng A — có cần truyền bằng chứng không? Chạy trước mọi bài toán mobility

Đối chứng bắt buộc: (1) UAV chạy nano rồi gửi box; (2) UAV chạy **cùng model large của trạm** rồi gửi box;
(3) gửi ảnh Q70 để trạm chạy large; (4) gửi bản cơ sở rồi patch. Box phải serialize cả lớp, tọa độ, confidence,
header và padding; cấm dùng annotation thay cho detector UAV. Đo latency tính toán và payload của cả hai model UAV.
Không cấm model large ở UAV để tạo lợi thế giả cho offloading. Chỉ được viện dẫn giới hạn onboard khi có thiết bị hoặc
ngân sách compute độc lập được khai báo và đo. Benchmark CPU/GPU trên máy hiện tại chỉ là số tham chiếu, không phải số đo phần cứng UAV.

**Dừng nhánh truyền ảnh cho tác vụ này** nếu hard output với cấu hình compute khả thi đạt utility tương đương hoặc hơn
mà rẻ hơn về bit và đúng hạn. Nếu chưa xác định compute onboard, chưa được coi cổng A là qua. Khi đó chưa cần chạy RL.

### Cổng B — correction có lợi riêng ở cùng tài nguyên không?

- So gửi mới Q70, truyền lại, progressive JPEG, bản cơ sở + patch theo thứ tự cố định/random, và patch theo yêu cầu receiver.
- Request ban đầu: xếp tile theo max `p*(1-p)` của detection hiện tại có center trong tile, lấy tối đa 3 tile chưa nhận mỗi
  vòng; tie theo tile ID. Đây là proxy chưa hiệu chỉnh, không phải xác suất lỗi thật. Receiver không thấy ảnh gốc.
  Detector dùng confidence thấp 0.01 để sinh candidate cho proxy; metric vẫn dùng ngưỡng đã khóa. Report nguy cơ bỏ sót
  vùng không có detection. Không dùng GT để chọn patch hoặc trọng số proxy.
- Đếm bản cơ sở, patch/header, ACK/request, gói mất, encode/decode và mỗi lần chạy lại detector. Không được cho correction
  thêm lượt inference miễn phí. Full-image và các baseline được phép giữ output trước đó, không buộc chạy lại vô ích.
- Progressive JPEG chỉ dùng prefix kết thúc tại scan hợp lệ sau khi đã kiểm decode. Smoke encode/decode cả file không
  chứng minh mọi prefix đều dùng được; không giả lập ưu thế bằng một đường quality–bits tự đặt.
- Gate feasibility được đăng ký trước: tăng **>=0.02 mean per-image F1** ở cùng bit/compute, hoặc giảm **>=20% tổng bit**
  với F1 giảm không quá 0.01, so với đối chứng khả thi mạnh nhất. Báo AP/recall, phân bố paired delta và bootstrap theo
  sequence. Đây là ngưỡng quyết định pilot, không phải bảo đảm ý nghĩa thống kê hay tiêu chuẩn công bố.
- Chưa có baseline semantic HARQ tái lập: không tuyên bố correction mới/hơn prior art. Muốn đưa generative recovery
  thành đóng góp phải bổ sung checkpoint và đối chứng tương ứng trong một protocol khác, trước khi chạy.

### Chỉ sau A/B — cổng C/D cho mobility và quyết định chung

- Một UAV, một BS, bay và truyền **đồng thời**. Episode giữ nguyên ảnh đã chụp. Bắt đầu Munich, 28 GHz, 100 MHz,
  vị trí BS và link budget lấy từ C2 rồi lưu lại trong config, không đổi để tạo lợi ích. Kịch bản mới là môi trường channel;
  ảnh VisDrone độc lập với cảnh Munich, không tuyên bố mô phỏng sensing/scene vật lý thống nhất.
- Bước control 0.5 s, tốc độ tối đa 10 m/s, deadline 2/5/10 s; collision/segment check trước khi hành động. Không teleport.
  Lấy mẫu nhiều điểm trên đoạn bay khi tích phân service, không dùng riêng SNR tốt tại điểm cuối cho cả slot.
- Channel ground truth truy vấn/cache theo vị trí đã chọn bằng Sionna RT. Không phát lại rate trace cố định sau khi đổi
  trajectory. Agent chỉ dùng phép đo kênh đã trả phí, lịch sử, pose, thời gian, ACK và feedback nhận thật. Map hoàn hảo
  chỉ là diagnostic; nếu MPC được dùng map học sẵn thì policy còn lại được cấp cùng map và cùng chi phí khảo sát.
- Mô hình `EFF*BW*log2(1+SNR)` của C2 chỉ là **abstraction service rate**, không gọi là waveform/MCS/LDPC đã kiểm chứng.
  Payload hoàn thành và decode xong mới thay đổi output. Chưa có mô hình công suất động cơ được khóa thì báo quãng
  đường/thời gian bay, không tự gán joule thực nghiệm; mọi kết luận energy phải chờ power model hoặc measurement.
- Thiết kế 2x2: đứng yên/bay x không correction/correction; cùng codec/model và tài nguyên. Thêm controller **có/không
  trạng thái phục hồi receiver** để kiểm chứng thông tin semantic có đổi quyết định, thay vì chỉ thêm lựa chọn truyền.
- So greedy, trajectory-first rồi communication, MPC nhân quả. State/proxy tương lai của MPC không dùng F1 thật hoặc
  CSI tương lai; oracle ghi rõ thông tin vượt trội và chỉ là best-found diagnostic nếu chưa chứng minh tối ưu.
- Giữ cùng điểm xuất phát, image, nhiễu và deadline giữa policy; đăng ký episode trước khi chấm, không chỉ chọn vùng
  NLoS có lợi cho mobility. Điểm bắt đầu đã LoS là control. Đo trên episode chưa dùng thiết kế; không suy cần RL chỉ từ
  tương tác dương hoặc MPC hơn greedy. Nếu MPC đã tốt và đủ nhanh, báo nó như nghiệm hợp lý.

**Đầu ra bước chuẩn bị:** protocol và inventory. **Chưa cho phép bước này tự mở training/experiment lớn.** Khi chuẩn bị
assets tiếp, chốt SHA checkpoint, phiên bản evaluator, split và ngưỡng compute onboard trước inference; nếu thiếu thì
giữ trạng thái chưa ready-to-run, không lấp bằng utility giả. Ngân sách GPU còn lại chưa bị sử dụng trong lượt này.

Nguồn gốc và giới hạn prior art: [VisDrone tác giả](https://github.com/VisDrone/VisDrone-Dataset),
[checkpoint YOLOv8](https://docs.ultralytics.com/models/yolov8/),
[Pillow JPEG](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html),
[hybrid-action trajectory/SemCom](https://arxiv.org/abs/2309.16713),
[receiver-driven generative semantic HARQ](https://arxiv.org/abs/2605.09897).
Các nguồn này xác minh thành phần/prior art, **không xác lập novelty của protocol ứng viên**.

## Pilot ảnh thật — cổng A đã chạy (06/11/2026)


Đã inference trên **216 ảnh dev, 30 nhóm filename**. Không train detector; chưa mở holdout.

| Phương án | AP50 | AP50:95 | Mean F1 | kbit/ảnh | xử lý p50 ms |
|---|---:|---:|---:|---:|---:|
| UAV nano → box codec | 0.3306 | 0.2220 | 0.4625 | 11.96 | 11.6 |
| UAV large → box codec | 0.4841 | 0.3525 | 0.5608 | 4.96 | 39.7 |
| Ảnh gốc → large tại trạm | 0.4841 | 0.3525 | 0.5608 | 1275.88 | 39.6 |
| JPEG Q10 → large tại trạm | 0.1795 | 0.1125 | 0.2514 | 264.74 | 40.0 |
| JPEG Q70 → large tại trạm | 0.4581 | 0.3332 | 0.5357 | 1006.92 | 39.7 |

Xử lý p50 là latency workstation GPU gồm preprocess/inference/postprocess; codec JPEG được đo riêng, không gồm radio.
Không dùng các số này làm latency hoặc năng lượng của phần cứng UAV.

| Model | CPU tham chiếu p50 / p95 ms (4 threads, 8 ảnh) |
|---|---:|
| yolov8n.pt | 85.0 / 88.6 |
| yolov8l.pt | 1073.9 / 1159.0 |

Box từ cùng model large so với ảnh gốc: ΔF1 = **+0.000000**, payload nhỏ hơn **257.5×**; so JPEG Q70 nhỏ hơn **203.2×**.

**Cổng A: chưa kết luận về triển khai.** Khi bỏ chi phí tính toán ra khỏi so sánh, gửi box từ cùng model giữ chất lượng và ít bit hơn. Điều này không chứng minh offloading ảnh vô ích: CPU tham chiếu chạy large chậm hơn GPU tại trạm. Cần so đủ compute, truyền và deadline.

**Quyết định sau đính chính:** chưa mở correction, mobility hoặc RL. Giữ offloading là ứng viên có điều kiện; không coi bộ ba đã thất bại. Phân tích không inference mới nằm ở [offloading/summary.md](../runs/active_recovery/offloading/summary.md). Năng lượng và compute onboard thực tế vẫn chưa được xác nhận.

Các giới hạn:
- Ba lớp car/bus/truck, AP từ COCOeval với maxDets500 và spatial ignored-region filter theo toolkit VisDrone; không phải điểm leaderboard 10 lớp.
- Group là prefix filename, chưa xác minh độc lập hoàn toàn theo chuyến bay. Split đã ghi trước metric; 168 ảnh holdout chưa được chấm.
- Một cặp detector COCO đóng băng, domain ảnh UAV; kết quả chưa phải đánh giá detector tốt nhất hoặc generative recovery.
- Không có power measurement, channel experiment hoặc phần cứng UAV; chưa chứng minh trade-off energy hay deadline end-to-end.

Job giữ khóa chung mất 1.93 phút (gồm CPU benchmark và chấm), peak VRAM 0.38 GiB. Không có training.

Tái lập: `bash scripts/active_recovery/run_gate_a.sh` trong WSL; từng mode và benchmark CPU có resume.
Dataset/checkpoint: `assets.json`; split/config/prediction hash: thư mục này.

## Đính chính Gate A và miền khả thi offloading (06/11/2026)

**Kết luận “dừng bộ ba” trước đó quá sớm. Gate A chưa kết luận về triển khai.** Hai phương án cùng YOLOv8l có F1
bằng nhau là đối chứng vị trí tính toán, chưa phải một cơ chế SemCom mới. Ít bit hơn không đồng nghĩa luôn tốt hơn:
large CPU tham chiếu mất khoảng 1.074 s, còn large GPU tại trạm khoảng 0.040 s. Không cần đổi dataset hay train lại.

Đã phân tích bằng CPU từ các prediction/timing cũ, **0 inference mới, 0 GPU**, không mở holdout. Cấu hình, sweep và
bản đồ ở [offloading/summary.md](../runs/active_recovery/offloading/summary.md), tái lập bằng
`python swarmbridge/experiments/active_recovery/feasibility.py REPO` trong env `active-recovery`.

- Phép xấp xỉ mean payload / median timing cho JPEG Q70 hòa thời gian với local large ở **0.974 Mbps**, nếu chưa tính
  overhead. Khi dùng payload và timing từng ảnh, ngưỡng trung vị khoảng **0.980 Mbps**, p90 **1.380 Mbps**; cộng 10%
  byte overhead đưa ngưỡng trung vị lên **1.078 Mbps**. Đây là effective goodput, không phải bitrate PHY bảo đảm.
- Với CPU median được gán cho 216 ảnh, goodput 3 Mbps, deadline 2 s, overhead byte 10% và fixed latency 20 ms:
  local large p50 **1.096 s**, ảnh gốc về trạm **0.514 s**, JPEG Q70 **0.438 s**. F1 tương ứng **0.5608/0.5608/0.5357**.
  Đây là sensitivity scenario, không phải 216 phép đo CPU ghép đôi.
- Kiểm tra trên **đúng 8 ảnh có timing CPU**: local large **1.096 s**, ảnh gốc **1.251 s**, Q70 **0.614 s** (p50).
  Mean F1 trên tám ảnh tương ứng **0.7027/0.7027/0.6622**. Tám ảnh cùng một nhóm filename; không suy tổng quát nhiều
  chuyến bay. Ví dụ này cho thấy payload biến thiên có thể đảo kết luận về tốc độ của gửi ảnh gốc.
- Nếu onboard có hiệu năng GPU giống workstation tham chiếu, local large p50 tổng chỉ **0.062 s** ở 3 Mbps: lợi thế
  offloading về thời gian biến mất. Không được chọn giả định onboard yếu chỉ để tạo kết quả dương.
- Sweep budget compute và deadline được công bố đầy đủ. Deadline 2/5/10 s là protocol gốc; các mức ngắn hơn là
  **thăm dò sau khi có kết quả**, không phải blind evidence. F1 khi quá deadline/compute budget được đặt 0 vì receiver
  chưa có kết quả cũ. Không dùng F1 thật để chọn hành động online.
- Năng lượng chỉ là sensitivity theo tỷ số công suất tăng thêm `Pcompute/Pradio` và platform baseline. Ở 3 Mbps,
  bỏ platform baseline, Q70 cần ít năng lượng UAV hơn local large trong mô hình khi tỷ số này **>0.344**; ảnh gốc
  **>0.434**. Chưa đo watt/joule thực. Đã tách giả định platform hoạt động tới khi có kết quả với hoạt động hết 2 s.

**Phạm vi kết luận:** có miền điều kiện để giữ offloading làm ứng viên. Chưa xác nhận miền đó trên phần cứng UAV,
chưa chứng minh correction có ích, chưa mô phỏng mobility và chưa có lý do mở RL. Proxy timings còn thiếu resize,
serialization box, queue và các phí vô tuyến đo thật; overhead chỉ là tham số sensitivity. Ngân sách tính toán/năng
lượng UAV độc lập và điều kiện kênh vẫn là phần cần xác định trước khi chấp nhận Gate A cho hệ thống cụ thể.

## Gate A — paired CPU đầy đủ trên 216 ảnh, 30 nhóm (06/11/2026)

**Đã chạy phép đo mới, không còn gán timing của tám ảnh cho cả tập.** Hai detector đóng băng chạy CPU trên từng ảnh
dev, hai lượt theo thứ tự đảo ngược: **864 inference CPU, thêm 12 lượt warm-up**, khoảng **10,0 phút CPU**; không train,
không GPU inference mới, không mở 168 ảnh holdout. Cùng split, checkpoint, resolution và evaluator của Gate A.
Artifact chính: [paired_cpu/summary.md](../runs/active_recovery/paired_cpu/summary.md),
[timing từng ảnh](../runs/active_recovery/paired_cpu/timing_samples.json),
[frontier](../runs/active_recovery/paired_cpu/paired_frontier.png).

### Phép đo và điểm xuất phát

- Điểm xuất phát là JPEG gốc đã lưu ở UAV. I/O đĩa, chụp ảnh và JPEG camera ban đầu không nằm trong thời gian dịch vụ.
- Đã đo riêng source decode, resize, chuyển BGR, CPU inference, box serialize/deserialize, JPEG encode/decode.
  Receiver GPU của phương án offload dùng lại timing và prediction đã đo trên **cùng từng ảnh**; không chạy thêm GPU.
- CPU local predictions được chấm lại; F1 giữa hai lượt không thay đổi. Timing detector giữa hai lượt chênh tương đối
  tuyệt đối trung vị **12,9% nano / 4,6% large**, p95 **31,7% / 13,6%**. Không coi các timing là hằng số phần cứng UAV.
- Timing dưới đây là tổng các công đoạn đã đo cộng service link giả định: goodput **3 Mbps**, overhead byte **10%**,
  fixed latency **20 ms**, deadline **2 s**. Không phải đo end-to-end trên mạng hoặc phần cứng UAV thực.

| Phương án, toàn bộ 216 ảnh dev | Mean F1 | Tổng latency p50 / p95 (s) | Giao đúng hạn |
|---|---:|---:|---:|
| Nano local → box | 0.4625 | 0.121 / 0.141 | 100% |
| Large local → box | 0.5608 | 1.145 / 1.212 | 100% |
| JPEG Q70 → large tại trạm | 0.5357 | 0.457 / 0.653 | 100% |
| JPEG gốc → large tại trạm (đối chứng thêm) | 0.5608 | 0.534 / 0.851 | 100% |

Q70 so với large local: nhanh hơn **0.688 s theo chênh hai trung vị**, đổi lại **−0.0251 mean F1**. So với nano,
Q70 thêm **+0.0732 F1** nhưng chậm hơn. Đây là đánh đổi, không phải chiến thắng tuyệt đối hoặc cơ chế SemCom mới.

### Độ rộng và độ ổn định theo nhóm

Tiêu chí mô tả được ghi lại trong lúc measurement đang chạy: Q70 không bị hai box baseline đồng thời vượt về mean
F1 giao đúng hạn và latency trung vị ở ít nhất 2/3 nhóm, tại hai goodput lân cận và một deadline gốc, **không áp
compute cap nhân tạo hoặc làm CPU chậm đi**. Đây là development assessment, không phải blind preregistration.

- Với 1.5/3/10 Mbps và deadline 2/5/10 s, Q70 không bị thống trị ở **20/30 nhóm** so với hai box baseline. 30 Mbps:
  **28/30 nhóm**. Kiểm tra từng lượt CPU độc lập đều tìm được miền đáp ứng tiêu chí, không chỉ median của hai lượt.
- Đối chứng box mạnh hơn cho metric F1: bỏ toàn bộ score dưới 0.25 và giả sử serialize/deserialize miễn phí. Q70 vẫn
  giữ miền **20/30 nhóm tại 1.5/3/10 Mbps**. Đây là bound ưu ái box, không phải phép đo mới; không tuyên bố AP giữ nguyên.
- Bootstrap theo 30 nhóm tại điểm cố định 3 Mbps/2 s: Q70−nano về mean F1 **+0.0732**, CI95% **[+0.0342,+0.1117]**;
  Q70−large **−0.0251**, CI95% **[−0.0323,−0.0169]**. Chênh **mean** latency Q70−large **−0.6958 s**, CI95%
  **[−0.7519,−0.6522]**. Các CI là pointwise/descriptive trên dev, không phải kiểm định toàn bộ grid hoặc chênh median.
- Giữ JPEG gốc làm đối chứng thứ tư. Trên bảng gộp tại 100 Mbps, JPEG gốc thống trị các phương án còn lại trong
  mô hình F1/latency này. Không dùng kết quả ba mode để tuyên bố Q70 tốt hơn mọi codec đơn giản.
- Các cell deadline dưới 2 s và compute cap vẫn là thăm dò; chưa dùng làm bằng chứng độc lập cho yêu cầu tác vụ mới.
- Power ratios, energy budgets và platform horizon giữ nguyên bản sensitivity trước; **chưa có watt/joule đo thật**.

**Chốt bước paired:** đã thấy một miền trade-off đủ rộng qua nhiều nhóm và lặp lại được trên CPU workstation tham chiếu.
Giữ offloading làm ứng viên có điều kiện, không đổi dataset hoặc train thêm detector. Chưa xác nhận triển khai trên UAV,
chưa xác lập novelty và **chưa mở patch/correction, mobility hoặc RL trong lượt này**. Phép thử correction tiếp theo,
nếu được thực hiện, phải vượt truyền lại, progressive JPEG và ảnh gốc ở cùng bit, feedback và chi phí xử lý.

Nhóm filename chỉ là proxy sequence, chưa bảo đảm độc lập theo flight; receiver GPU timing là số đo cũ; queue và kênh
thay đổi chưa được mô phỏng. Những giới hạn này không biến mất khi tăng số ảnh CPU.
Tái lập measurement: `bash scripts/active_recovery/run_paired_cpu.sh`; phân tích:
`python swarmbridge/experiments/active_recovery/analyze_paired.py REPO`. Các công đoạn có resume, model cũ giữ nguyên.

## Kiểm tra objective: delivered F1 thay vì chỉ Pareto latency (06/11/2026)

Đã hoàn tất bằng artifact có sẵn: **không inference mới, không train, không GPU, không đổi dataset hoặc deadline gốc**.
Nguồn và toàn bộ bảng: [delivered_objective/summary.md](../runs/active_recovery/delivered_objective/summary.md),
[CSV từng mode](../runs/active_recovery/delivered_objective/per_mode.csv),
[bản đồ](../runs/active_recovery/delivered_objective/delivered_F1.png).

Objective ở mỗi cặp goodput/deadline là **mean F1 giao đúng hạn lớn nhất của một mode cố định** trên tập dev;
F1 = 0 nếu trễ. Không dùng GT để chọn riêng mode từng ảnh. Completion time p50/p95/mean vẫn báo riêng, không cắt ở
deadline và không ghép thành reward có trọng số. Sai khác F1 trong **1e-4** được coi là hòa, tránh gọi rounding CPU/GPU
khoảng 1e-5 là cải thiện. Goodput/overhead vẫn là kịch bản, không phải một phép đo mạng end-to-end.

Luôn có **JPEG gốc**, Q70 và Q10, cùng nano/large-box. Q10 dùng lại prediction cũ và codec timing paired đã đo;
không có decoder/mô hình mới. Giữ đối chứng box mạnh: lọc score <0.25 không đổi metric F1 và cho serialization miễn phí.
Đối chứng này chỉ là bound ưu ái box; không khẳng định AP giữ nguyên.

### Kết quả chính

**Deadline gốc 2/5/10 s: không có cải thiện delivered F1 của offloading ngoài rounding** trong 27 tổ hợp goodput/deadline.
Chênh lệch lớn nhất chỉ **+0.000010 F1**, dưới tolerance. Đây là 27 tổ hợp tham số, **không phải 27 thí nghiệm độc lập**;
các hàng 2/5/10 s thường trùng nhau vì large local đã hoàn thành toàn bộ ảnh đúng hạn.

| Goodput | Deadline | Box tốt nhất: delivered F1 | Offload tốt nhất: delivered F1 | Mode offload tốt nhất |
|---|---:|---:|---:|---|
| 3 Mbps | 2 s (gốc) | 0.5608 | 0.5608 | JPEG gốc; hòa large-box |
| 3 Mbps | 1 s (exploratory) | 0.4625 | 0.5357 | JPEG Q70 |
| 10 Mbps | 0.5 s (exploratory) | 0.4625 | 0.5582 | JPEG gốc |
| 10 Mbps | 0.25 s (exploratory) | 0.4625 | 0.4895 | JPEG Q70 |
| 30 Mbps | 0.1 s (exploratory) | 0.0141 | 0.2333 | JPEG Q10 |

Tại 3 Mbps/2 s, large-box **1.145 s**, Q70 **0.457 s**, JPEG gốc **0.534 s** (completion time p50).
Q70 nhanh hơn nhưng F1 **0.5357** vẫn dưới **0.5608**. JPEG gốc đạt gần bằng F1 của large-box và nhanh hơn,
nhưng không tạo thêm F1 cho objective chỉ yêu cầu hoàn thành trước 2 s. Không suy lợi ích điều khiển chỉ từ thời gian dư.

Trong vùng exploratory có lợi ích nhiệm vụ thật theo mô hình: 3 Mbps/1 s, Q70 thêm **+0.0732 delivered F1** so với
box tốt nhất và dương ở **22/30 nhóm**. Đây là lý do giữ một giả thuyết deadline ngắn, **chưa phải sửa yêu cầu nhiệm vụ**.
Bootstrap chọn lại mode cố định tốt nhất trong mỗi family ở từng mẫu nhóm; các CI vẫn là descriptive/pointwise trên dev.
Không coi chúng là kiểm định toàn grid hoặc blind test.

**Quyết định:** mục tiêu gốc chưa có nền tảng để mở Gate B chỉ vì offloading nhanh hơn. Không viết correction,
không mở mobility/RL trong lượt này. Giữ miền deadline ngắn ở trạng thái exploratory cho đến khi có yêu cầu nhiệm vụ
độc lập; không đổi deadline chỉ để tạo kết quả dương. Khi đánh giá correction, JPEG gốc và các chất lượng JPEG trung
gian/progressive vẫn phải là đối chứng. Chênh Q70→ảnh gốc **0.0251 F1** là khoảng cách tới một reference, không phải
cận trên toán học của mọi thuật toán phục hồi. Chưa có bằng chứng receiver phát hiện được vùng đối tượng bị bỏ sót.

Tái lập: `python swarmbridge/experiments/active_recovery/delivered_objective.py REPO`. Công suất giả định giữ nguyên;
phép kiểm tra này không đo hoặc tối ưu năng lượng và không gộp nó vào objective F1.

## Công việc tiếp theo — bản điều khiển thu gọn theo JSAC Huang, chưa phải đề xuất mới (06/11)

**Mục tiêu:** có một baseline UAV + RL + lựa chọn mode truyền chạy được trước khi chọn đóng góp nghiên cứu mới.
Đóng băng VisDrone–YOLO–JPEG với objective cũ. Nhiệm vụ mới được lấy từ bài JSAC, không sửa deadline của pilot cũ:
giao **ảnh** từ nguồn qua UAV relay tới người dùng. Ground user cần ảnh tái tạo; gửi box/nhãn không hoàn thành cùng dịch vụ.

Đã đọc bản published PDF 13 trang: system model II, framework III, MDP IV, simulation V. Manifest nguồn và các giá trị
được công bố ở `runs/jsac_reference/reference.json`. Bản full có 5 LEO, 3 UAV, 8 user, 100 task; UAV tối đa 10 m/s,
cao 45–60 m. Eq.14 SCE dùng cả chất lượng ảnh và delay; Eq.15 còn có ràng buộc chất lượng tối thiểu và deadline.
Vì vậy thời gian hoàn thành có giá trị trực tiếp trong objective nguồn, khác objective delivered-F1 đã thử.

### Phạm vi bản nhỏ được chọn

1. **Control simulator:** một nguồn ảnh/backhaul, hai UAV relay, bốn user, nhiều yêu cầu có deadline và quality target.
   UAV thực sự thay đổi vị trí qua action; vị trí đổi channel/service tương lai. Bay và truyền không bị mặc định loại trừ.
   Queue, lượng bit còn lại, thời gian tính và thời điểm hoàn thành phải được cập nhật nhân quả.
2. **Mode bank cho bước kiểm tra controller:** dùng profile được paper công bố, gắn nhãn `published illustrative
   profile`, không gọi là codec/gen model đã chạy. Kiểm hai cách hiểu đơn vị Kb nếu chưa làm rõ; không chọn cách hiểu
   để tạo chiến thắng. Một ảnh minh họa không đại diện phân bố chất lượng toàn dataset.
3. **Baselines rồi RL:** hover + mode cố định, nearest-user/EDF greedy và planner ngắn hạn cùng thông tin; PPO có sẵn
   dùng để kiểm tra pipeline điều khiển, với action rời rạc hóa được ghi rõ. PPO không được gọi là tái lập DSAC hoặc
   thuật toán mới. Chỉ train sau test movement, queue, deadline và tính reward; smoke training không chứng minh thắng.
4. **Đầu ra cần thấy:** trajectory hai UAV, log task/mode theo thời gian, tỉ lệ hoàn thành, chất lượng, delay, SCE và
   reward curve. Báo riêng chất lượng và delay để tránh gain chỉ do normalization/reward. Chưa gắn thêm SB/Mamba.

### Các giới hạn phải giải quyết công khai

- Không tuyên bố tái lập đúng từng số Fig.6–11. Nguồn/tầng vệ tinh đơn giản hóa và PPO là khác biệt implementation.
- Chưa xác minh được repo/checkpoint chính thức của bài JSAC từ PDF và lượt tìm kiếm hiện tại. Semantic mode thật cần
  VQE/codebook và UNet được huấn luyện riêng; BLIP-2/VAE/CLIP pretrained không thay thế phần này.
- Fig.5 ghi payload bằng **Kb** nhưng chưa xác nhận là kbit hay kbyte. Lưu nguyên số và đơn vị; chưa chuyển thành bits
  như một sự thật đã biết. Mode 1/2/3 PSNR lần lượt là 32.2/11.26/16.3/18.8/23.2 cho đúng ảnh minh họa.
- Phần thuật toán mô tả return distribution và dẫn Duan et al.; không suy DSAC là MARL/distributed control chỉ từ tên.
- Slot duration, task-arrival parameter, action mapping, các scale normalization và pipeline neural còn thiếu để tái lập
  đầy đủ. Những giá trị tự chọn phải gắn nhãn implementation choices và kiểm sensitivity, không nhận là paper cung cấp.
- Bản nhỏ chỉ phục vụ reproduction/control baseline. Nó chưa kiểm chứng giả thuyết receiver recovery state → mobility,
  chưa xác lập novelty và không làm thay đổi kết luận âm của VisDrone. Chưa chạy env hoặc train RL trong lượt audit này.

Nguồn chính: [published paper](https://doi.org/10.1109/JSAC.2025.3623157),
[arXiv v3](https://arxiv.org/abs/2412.05647v3).
