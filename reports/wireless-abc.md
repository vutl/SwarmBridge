# Wireless A/B/C pilot — tài liệu tổng hợp

Plan gốc: plan pilot của Codex (05/11), dựa trên `reports/ke-hoach-proposal-abc-2026-11-04.md`. Phạm vi: feasibility/dev,
tối đa 10 GB dữ liệu mới, 8 giờ GPU, một job GPU tại một thời điểm (khóa chung Windows–WSL).

**Tóm tắt (05/11 sáng):** A — SB không qua gate (0/18); B — hợp tác có lợi chỉ ở vùng chưa có bản đồ (qua 6/12 cấu hình bay đoàn); C — quét beam 20 ms ≈ oracle (Δ 0.002 AMOTA), không mở RL. Chi tiết và bảng quyết định ở cuối.

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

## A0b — dựng lại thiết lập paper SB-ISAC (WCL 2025) để kiểm tuyên bố chính

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
