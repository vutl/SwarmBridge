# SwarmBridge — Báo cáo tiến độ (tính đến 30/10/2026)

> **Cập nhật của Codex ngày 31/10/2026:** đã hoàn tất audit BEV 30k và chạy thử checkpoint speech/vocoder chính thức. Nhận định tuyệt đối ở mục 12 rằng không paper nào thắng mapping cần thu hẹp. Xem [kết quả chạy và đối chiếu mới](sb-paper-reproductions-2026-10-31.md) và [kiểm toán nguyên nhân huấn luyện](sb-training-root-audit-2026-10-31.md). Phần dưới giữ nguyên báo cáo lịch sử ngày 30/10.

> Báo cáo này ghi lại toàn bộ những gì đã làm: dữ liệu, cách dựng nhãn, mô hình, mô phỏng kênh, các thí nghiệm, kết quả, các lỗi đã tìm ra và sửa, phần đọc tài liệu, và những gì còn mở.
> Mọi con số là **mIoU trên tập val (200 lưới)** của CoPerception-UAV Town3, trừ khi ghi khác. "3 seed" nghĩa là trung bình ± độ lệch chuẩn qua 3 lần train với seed 0, 1, 2.

---

## 0. Tóm tắt một trang

| Câu hỏi | Kết quả | Mức chắc chắn |
|---|---|---|
| **Q1** — Schrödinger Bridge (SB) khôi phục latent bị kênh làm hỏng có tốt hơn mạng hồi quy (mapping) không? | **Không.** Khi đã tinh chỉnh đúng (β₀=10, EMA), SB chỉ **ngang** mapping (0.748 vs 0.750, kênh severe) và cần 4 bước ODE; 1 bước thì thua (0.736). | Cao (3 seed, đủ baseline đối chứng) |
| **Q1a** — Reference process "biết kênh" (nhiễu theo độ tin cậy) có tốt hơn reference đẳng hướng? | **Không có tác dụng** (chênh lệch 0.0–0.2 điểm, không có ý nghĩa thống kê). | Cao |
| **Q1b** — Uncertainty của SB có biết thêm điều gì ngoài chất lượng kênh? | Có (AUROC trong cùng mức kênh ≈ 0.76 ≫ 0.5), **nhưng thua nhẹ** một mapping có đầu ra phương sai (H1: 0.766), trong khi tốn gấp 32 lần tính toán. | Cao |
| **Q2** — Dùng độ không chắc để quyết định xin gửi lại block nào? | **Có lợi rõ.** Cần **ít hơn 1.8–2.8 lần số bit gửi thêm** so với chọn theo kênh xấu nhất; chọn ngẫu nhiên không đuổi kịp. | Cao (3 seed, 3 loại kênh) |
| **Q2 (gửi 2 vòng)** — Gửi một nửa trước, xin phần còn lại theo độ không chắc? | **Tiết kiệm ≈ 30% tổng số bit** ở chất lượng gần như gửi đủ (0.814 với 70% bit so với 0.821 khi gửi 100%). | Cao (3 seed) |
| SB có giúp Q2 (chọn block)? | **Không.** Chọn theo độ phân tán của 8 mẫu SB kém H1 0.3–1.0 điểm. | Cao |
| SB có mạnh hơn khi thông tin cực ít? | **Không.** Thua mapping rõ khi thông tin ít. Lợi thế lớp xe so với H1 (+1.0–1.6) **gần như biến mất khi so với mapping thường cùng nâng cấp** (mục 11). | Cao (3 seed) |

**Câu chuyện hiện tại:** giá trị lớn nhất không đến từ mô hình sinh phức tạp mà từ việc **trạm biết chỗ nào mình chưa chắc để xin thêm đúng chỗ đó**. SB hiện có vai trò so sánh có kiểm soát (kết quả âm tính chắc chắn), cộng một tín hiệu dương về **giữ vật thể nhỏ (xe)** đang được kiểm chứng.

---

## 1. Bối cảnh và các câu hỏi nghiên cứu

**Ý tưởng gốc (từ các tài liệu Deep Research):** nhiều UAV quan sát một cảnh; trạm nhận dùng một **conditional Schrödinger Bridge** biến biểu diễn bị thiếu/hỏng thành biểu diễn "đầy đủ như khi nhận đủ bằng chứng của cả bầy"; độ bất định của bridge quyết định UAV nào cần gửi gì → vòng kín *nhận thức → bất định → giao tiếp → cập nhật*.

| | Câu hỏi | Vai trò |
|---|---|---|
| Q1 | Khôi phục latent bị hỏng: SB có đáng dùng? Reference theo kênh (Q1a) có giúp? Uncertainty (Q1b) có giá trị ngoài thông tin kênh? | Kiểm tra nền móng |
| Q2 | Dùng uncertainty để quyết định xin thêm thông tin gì, từ UAV nào, tốn ít bit nhất | Đóng góp về giao tiếp |
| Q3 | Chọn **chế độ** gửi (mức nén, dạng dữ liệu…), học chính sách (bandit/DRL) | Chưa làm |

**Thay đổi quan trọng về thiết lập (do dữ liệu, xem mục 3.6):** các UAV trong Town3 bay cách nhau ~205 m, gần như không nhìn chồng vùng. Vì vậy chuyển sang **"hướng 1" — leader setup**: mỗi UAV mã hoá vùng của mình thành latent, gửi về trạm (leader / trạm mặt đất) qua kênh riêng; trạm khôi phục từng vùng. Phần "UAV khác lấp chỗ bị che" chuyển sang Q2 và cần dataset có vùng chồng.

---

## 2. Môi trường và mã nguồn

- **Máy:** RTX 3070 8 GB (sm_86), RAM 16 GB, Windows 11.
- **Python:** venv `.venv` (Python 3.12.16 qua `uv`), **PyTorch 2.12.0 + CUDA 13.0**, torchvision 0.27.0, scipy, gdown, pypdf. Cần cài Microsoft VC++ Redistributable.
- **Quy tắc vận hành:** mỗi lúc chỉ một tác vụ GPU (chạy 2 cùng lúc đã gây hết VRAM một lần).

Cấu trúc code (`swarmbridge/`):

| Thư mục / file | Nội dung |
|---|---|
| `channel/channel.py` | Mô phỏng kênh trên latent, gộp LMMSE, bản đồ độ tin cậy `r` |
| `data/synthetic.py` | Dữ liệu giả để kiểm tra pipeline |
| `data/coperception_uav/geometry.py` | Phép chiếu ảnh ↔ mặt đất đã kiểm chứng |
| `data/coperception_uav/build_bev.py` | Dựng lưới BEV + nhãn |
| `data/real.py` | Nạp latent đã cache, áp kênh, giao diện đánh giá |
| `teacher/` | Encoder + decoder BEV (teacher), train và cache latent |
| `bridges/` | Backbone U-Net dùng chung, reference process, các mô hình (Mapping, HeteroMapping, FlowMatching, SchrodingerBridge) |
| `uncertainty/metrics.py` | AUROC, AUROC trong từng nhóm `r` |
| `experiments/q1_reference/` | Lưới thí nghiệm Q1 + script gộp kết quả |
| `experiments/q2_retx/` | Q2: gửi lại có chọn lọc |
| `experiments/q2_progressive/` | Q2: gửi 2 vòng; so sánh bộ khôi phục; thí nghiệm decoder nâng cấp |
| `scripts/` | Kiểm tra hình học, chẩn đoán nhãn, xem ảnh BEV |
| `tests/test_q1_core.py` | 5 unit test cốt lõi |

---

## 3. Dữ liệu: CoPerception-UAV Town3

### 3.1 Nguồn
- Bản đầy đủ chỉ có trên **Baidu Netdisk** (cần tài khoản). Đang dùng **bản mẫu Town3** từ Google Drive: `coperception_uav_town3.zip` (6.28 GB, 6.5 GB giải nén) + `split.py`.
- Trang dataset không ghi license, không có code; phần citation trên trang trích nhầm (DAIR-V2X/Where2comm).

### 3.2 Cấu trúc thực tế
- Định dạng **nuScenes**: 17 scene, **340 mẫu**, mỗi mẫu 5 UAV × 5 camera (FRONT/BACK/LEFT/RIGHT/BOTTOM), mỗi camera có ảnh RGB và ảnh semantic `_seg.png`, **800×450**.
- **67.576 box 3D** cho xe (21 loại xe CARLA), pose UAV, calibration camera.
- File `maps/Town03.jpg`: bản đồ đường CARLA, không có toạ độ/tỉ lệ, không có nhà → không dùng.
- **Chia tập theo `split.py`** (đọc bằng `ast`, không thực thi file tải về): Town3 gồm train = scene 2, 4, 5, 7, 8, 10–15 (11 scene), val = 0, 6, test = 1, 3, 9, 16. Chia theo scene nên không rò rỉ.

### 3.3 Những điểm "bẫy" của dataset (đã xác minh)
1. **Bản mẫu không có bản đồ BEV làm sẵn** (trang dataset nói có BEV 4 lớp) → phải tự dựng.
2. **Metadata ghi intrinsics cho ảnh 1600×900, ảnh thật 800×450** → phải chia đôi: f = 400, c = (400, 225).
3. **`calibrated_sensor` chứa pose camera trong hệ toàn cục** (cả vị trí lẫn hướng xoay theo yaw của UAV), khác chuẩn nuScenes (sensor-to-ego).
4. **Trục camera theo quy ước AirSim**: x = trục quang, y = phải ảnh, z = xuống ảnh. Pose UAV theo kiểu NED (z = −42.6 nghĩa là cao 42.6 m), box xe theo z hướng lên.
5. **Màu semantic AirSim không ổn định**: mỗi toà nhà một màu (93 màu khác nhau), có màu dùng chung cho nhiều loại vật thể (ví dụ cùng một màu là mép vỉa hè ở ảnh này, là toà nhà ở ảnh khác; một màu "trông như đường" thực ra là cỏ/mặt đất).
6. **Mặt đất không phẳng**: ~11% xe nằm trên đường trên cao (~8 m), tập trung ở vùng UAV 4.
7. **Xe bị che hoàn toàn** dưới gầm cầu (vùng UAV 3).

### 3.4 Kiểm chứng hình học
Phép chiếu cuối cùng (`geometry.py`): tia từ camera qua pixel, giao với mặt đất z = 0; và ngược lại, chiếu điểm có độ cao `z_up` lên ảnh.

| Kiểm tra | Phiên bản đầu (bỏ qua yaw) | Có yaw, mặt đất phẳng | **Có yaw + độ cao thật** |
|---|---|---|---|
| Box xe chiếu vào ảnh trúng pixel xe — UAV 0/1/2 | ~30% / 5% / ~30% | 99.6 / 99.8 / 99.5% | **99.8 / 99.9 / 100%** |
| UAV 3 | ~37% | 91.4% | **92.7%** (phần thiếu = xe dưới gầm cầu) |
| UAV 4 | ~27% | 83.9% | **99.8%** |
| Đốm xe không có box | — | 0–5.2% | **0.1–1.2%** |
| Ghép đường giữa 2 UAV nhìn chồng (IoU) | 0.19 | **0.96**, lệch còn lại 0 m | — |

Ghi chú: phiên bản đầu kết luận nhầm "box lệch"; nguyên nhân thật là phép chiếu của tôi bỏ qua hướng xoay của UAV.

### 3.5 Cách dựng nhãn BEV (`build_bev.py`)

**Lưới:** mỗi (mẫu, UAV) có một lưới riêng **96 × 96 m, ô 0.25 m (384 × 384)**, tâm tại điểm mặt đất dưới trục quang camera BOTTOM, trục theo hướng phải/xuống của ảnh chiếu xuống đất. Camera BOTTOM chụp gần thẳng xuống từ ~42.6 m, góc ngang 90°, phủ ~85 × 48 m.

**Cách tô:** với mỗi ô, chiếu ngược vào camera BOTTOM của cả 5 UAV (cùng mẫu) bằng phép chiếu đã kiểm chứng.

**3 lớp:** `0 = khác`, `1 = đường`, `2 = xe`, `255 = không quan sát được (bỏ qua)`. Lớp nhà bị gộp vào "khác" vì màu nhà không dựng sạch được.

| Lớp | Nguồn | Chi tiết |
|---|---|---|
| Đường | Ảnh semantic BOTTOM | Chỉ 5 màu đã xác nhận bằng mắt: mặt đường (235,124,133), vạch kẻ (139,242,254), vạch trắng (215,252,228), mũi tên (155,154,125) và (197,177,129). Hai màu "giống đường" bị loại sau khi kiểm tra trên nhiều ảnh: (119,113,136) là mặt đất/cỏ, (129,236,90) dùng chung với cây. |
| Xe | Box 3D | Vẽ footprint box tại **vị trí biểu kiến trên mặt phẳng z = 0 nhìn từ camera** (tia nhìn qua xe ở độ cao thật, kéo xuống đất) để khớp với ảnh đầu vào khi xe trên đường cao. |
| Xe bị che | Box 3D + ảnh semantic | Box nằm trong khung hình nhưng không camera nào thấy pixel xe ở đó (1.5% số box trong khung hình) → gán **255 (bỏ qua)**, không gán "xe" cũng không gán "khác". |

**Mỗi file `<sample>_<uav>.npz` lưu:**
- `rgb` (384×384×3): ảnh BOTTOM chiếu xuống lưới — **đầu vào của encoder**.
- `obs`: lớp mà riêng UAV này thấy.
- `gt`: nhãn đầy đủ (đường/khác từ hợp các UAV, xe từ box), **chỉ trong vùng có ít nhất 1 UAV thấy** — vì z\* là "oracle theo những gì bầy quan sát được", không phải sự thật của thế giới.
- `seen` (5×384×384): UAV nào thấy ô nào.
- `origin`, `axes`: pose của lưới (để sau này xoay/dịch feature giữa các UAV).

**Thống kê sau khi dựng:** 1.700 lưới (1.100 train / 200 val / 400 test), 236 MB. Đường chiếm 15.1%, xe 0.58% diện tích lưới; trong vùng camera của chính UAV: khác 66.9%, đường 31.9%, xe 1.14%. 90.5% pixel xe trong ảnh semantic nằm trong box (phần còn lại chủ yếu do viền 0.25 m và thân xe nghiêng).

### 3.6 Phát hiện quan trọng: các UAV hầu như không nhìn chồng nhau

| Chỉ số | Giá trị |
|---|---|
| Khoảng cách giữa 2 UAV (trung vị) | **205 m** (IQR 143–288 m) |
| Cặp UAV đủ gần để chồng vùng (< 60 m) | **5%** |
| Camera của chính UAV phủ lưới 96 m | 44.0% |
| Cả bầy phủ lưới đó | 47.5% (+3.5%) |
| Lưới không được UAV khác bổ sung chút nào | **64%** |
| Lưới được bổ sung > 10% | 16% |

→ Không thể kiểm tra "UAV khác lấp vùng bị che" trên bản Town3 này → chuyển sang **leader setup** (mục 1).

---

## 4. Teacher: encoder + decoder BEV

- **Encoder:** ResNet18 pretrain ImageNet, cắt tới `layer2` (stride 8, 128 kênh) + conv 1×1 → **latent 16 × 48 × 48**. Latent này chính là "message" UAV gửi đi; bản sạch của nó là **z\***.
- **Decoder (task head):** conv + 2 lần upsample ×2 + residual block → logits 3 lớp ở **192 × 192 (ô 0.5 m)**.
- **Nhãn train:** `gt` giới hạn trong vùng camera của chính UAV (leader setup), lấy mẫu mỗi ô thứ 2 (0.5 m).
- **Train:** 30 epoch, batch 8, AdamW 3e-4 + OneCycle, bf16, augmentation xoay 90° + lật, cross-entropy với trọng số lớp xe = 5. ~6 phút.
- **Cache:** latent được chuẩn hoá theo từng kênh (mean/std của tập train) để có công suất đơn vị (giả định của mô hình kênh); lưu `latents_{train,val,test}.pt` (z fp16, nhãn, mặt nạ vùng nhìn ở 48 và 192).

| Teacher | mIoU val | mIoU test | IoU khác / đường / xe (test) |
|---|---|---|---|
| v1 — nhãn giả định mặt đất phẳng | 0.855 | 0.844 | 0.984 / 0.956 / 0.592 |
| **v2 — nhãn đã sửa (dùng hiện tại)** | **0.882** | **0.881** | 0.986 / 0.961 / **0.697** |

Cùng model, chỉ sửa nhãn mà IoU lớp xe tăng +10.5 điểm → xác nhận độc lập rằng các lỗi nhãn là thật. Teacher v1 lưu ở `runs/teacher_v1_flatground`.

---

## 5. Mô phỏng kênh (`channel/channel.py`)

Message mỗi UAV là latent 16×48×48, chia thành **block 4×4 ô** (mỗi block = một "gói", 144 block/lưới). Mỗi UAV có trạng thái kênh riêng:

| Thành phần | Mô hình |
|---|---|
| Mất gói (erasure) | Mỗi block bị mất với xác suất `p_erase` |
| AWGN | `y = m + n / sqrt(snr_eff)` (giả định công suất đơn vị) |
| Fading Rayleigh | `|h|² ~ Exp(1)` cho từng block, `snr_eff = snr · |h|²` |
| Trễ / dữ liệu cũ (chỉ dùng ở dữ liệu giả) | `m = sqrt(a²)·z + sqrt(1−a²)·z_cũ`, `a² = exp(−tuổi/4)` |
| Vùng nhìn | Ngoài vùng camera thì không có dữ liệu |

**Gộp ở trạm (LMMSE với prior đơn vị):** coi trễ + nhiễu là nhiễu, độ chính xác bằng chứng `g = visible · a² / s²`, `s² = (1−a²) + 1/snr_eff`;
`z_p = Σ (visible·a/s²)·y / (1 + G)`, `r = G / (1 + G)`, với `G = Σ g`.
→ **`1 − r` đúng bằng phương sai lỗi còn lại của `z_p`** (đã có unit test kiểm chứng trên từng nhóm `r`). Nơi không nhận được gì: `z_p = 0`, `r = 0`.

**Điều kiện kênh dùng khi đánh giá:**

| Tên | SNR | Mất gói | Fading |
|---|---|---|---|
| easy | 20 dB | 0% | không |
| mid | 5 dB | 20% | Rayleigh |
| severe | −5 dB | 50% | Rayleigh |
| hetero | mỗi UAV ngẫu nhiên: SNR ∈ [−10, 20] dB, mất gói ∈ [0, 60%], Rayleigh 50% | | |

Khi train: SNR ∈ [−10, 20] dB, mất gói ∈ [0, 60%] (Q1, Q2 gửi lại) hoặc [0, 90%] (Q2 gửi 2 vòng — vì block chưa gửi trông giống hệt block bị mất).

---

## 6. Các phương pháp so sánh

Mọi mô hình dùng **chung một U-Net** (2 tầng, width 64, FiLM theo thời gian), AdamW 3e-4, batch 32, bf16. Đầu vào luôn có `z_p`; "+ CSI" nghĩa là thêm bản đồ `r` làm đầu vào.

| Mã | Phương pháp | Mô tả |
|---|---|---|
| M0 / M1 | **Mapping** (không / có CSI) | Hồi quy thẳng `z_p → z*`, loss MSE, 1 lần chạy mạng |
| M2 / M3 | Mapping + augmentation nhiễu **theo kênh** (không / có CSI) | Khi train cộng nhiễu lớn ở vùng `r` thấp — cùng hình dạng nhiễu với SB theo kênh nhưng không có bridge. **Baseline bắt buộc cho Q1a** |
| M4 | Mapping + augmentation nhiễu đều | Đối chứng: lợi ích đến từ hình dạng nhiễu hay chỉ từ việc có nhiễu |
| H1 | **Heteroscedastic mapping** | Mapping có thêm đầu ra phương sai (cách train "faithful" của Stirn et al. 2023: mean train bằng MSE như mapping, phương sai bằng Gaussian NLL với mean bị chặn gradient). **Baseline bắt buộc cho Q1b** |
| F1 | **Flow matching** (OT-CFM) | Học vận tốc `z* − z_p` trên đường thẳng, Euler từ `z_p`; dạng Markov (không nhận `z_p` riêng — xem lỗi #3 mục 7) |
| S0 / S1 | **SB isotropic** (không / có CSI) | Bridge kiểu I²SB/SBM có cặp: `x_t = (1−t)z* + t·z_p + sqrt(t(1−t)·β₀·Λ)·ε`; mạng dự đoán `z*` từ `[x_t, z_p, r]`; lấy mẫu bằng hậu nghiệm Brownian-bridge; NFE=1 = 1 lần gọi tại t=1; chế độ ODE = bỏ nhiễu hậu nghiệm |
| S2 / S3 | SB reference **theo r** — R1 (không / có CSI) | `Λ = λ_min + (1−r)(λ_max − λ_min)` |
| S4 | SB reference **empirical** — R2 (có CSI) | `Λ[kênh, bin(r)] = Var[z* − z_p | r]` đo từ dữ liệu (gần ý PRISM: nhiễu ~ thông tin bị phá) |

R1/R2 được chuẩn hoá để có **cùng tổng lượng nhiễu** với R0, chỉ khác phân bố theo vị trí.

**Các tuỳ chọn thêm sau khi đọc tài liệu:** EMA 0.999 (cho mọi model), trộn mẫu tại t=1 (`--t1-frac`, chưa dùng trong kết quả chính), đánh giá SB bằng ODE và trung bình 8 mẫu, **loss phân đoạn qua decoder** và **đầu ra giữ phần đã nhận** `ẑ = z_p + sqrt(1−r)·net` (thí nghiệm ở mục 11).

**Chỉ số:** mIoU (3 lớp, bỏ qua ô 255); IoU lớp xe; AUROC phát hiện ô bị phân loại sai bằng U, bằng `1−r`, và **AUROC của U trong từng nhóm `r`** (≈0.5 nghĩa là U chỉ lặp lại thông tin kênh).

---

## 7. Kiểm tra pipeline bằng dữ liệu giả và các lỗi đã sửa

Trước khi có dữ liệu thật, toàn bộ pipeline được chạy trên dữ liệu giả (trường ngẫu nhiên trơn 16×32×32, 4 UAV với vùng nhìn hình chữ nhật). Lần chạy này **không mang giá trị bằng chứng**, nhưng đã bắt được 3 lỗi mà nếu không sửa sẽ làm sai kết luận trên dữ liệu thật:

1. **Bộ gộp tin khuếch đại nhiễu:** trung bình có chuẩn hoá `Σ r·y / Σ r` biến một message SNR rất thấp thành nhiễu khổng lồ → flow matching nổ loss. Sửa: gộp LMMSE (mục 5).
2. **Dữ liệu giả có phương sai ở mép ảnh gấp 1.4–2 lần** (bicubic ngoại suy) → vi phạm giả định prior đơn vị. Sửa: sinh lưới lớn hơn rồi cắt giữa.
3. **Flow matching "học mẹo":** khi nhận cả `x_t` và `z_p`, vận tốc suy ra được thẳng `(x_t − z_p)/t` → loss train ~0 nhưng đánh giá tệ. Sửa: dùng OT-CFM chuẩn (Markov).

Lỗi 1 và 3 đều làm SB trông tốt hơn thực tế. Các lỗi khác về sau: hết VRAM khi giải mã cả lô 200 lưới (sửa: giải mã theo lô 32); chạy 2 tác vụ GPU song song gây hỏng 2 điểm của lượt quét β₀ (đã chạy lại).

**Unit test (`tests/test_q1_core.py`, đều đạt):**
- Bộ lấy mẫu SB với oracle giữ đúng phân phối biên `N((1−s)z* + s·z_p, β·s(1−s))` ở mọi bước và kết thúc đúng tại `z*`.
- R0/R1/R2 có cùng tổng nhiễu.
- R2 đặt nhiễu lớn hơn ≥3 lần ở vùng mất so với vùng tin cậy.
- Kênh tốt cho `r` cao hơn và `z_p` sai ít hơn.
- `1 − r` khớp sai số thực của `z_p` trong từng nhóm `r` (sai lệch < 0.05).

---

## 8. Kết quả Q1 — khôi phục latent

### 8.1 Lần chạy đầu (β₀ = 1, không EMA, 3 seed × 6.000 bước, nhãn đã sửa)

| | mid | severe |
|---|---|---|
| oracle (z\* sạch) | 0.882 | 0.882 |
| rx (không khôi phục) | 0.778 | 0.324 |
| **M1 Mapping** | **0.849** | **0.748** |
| M3 Mapping + nhiễu theo kênh | 0.845 | 0.741 |
| H1 Mapping + phương sai | 0.846 | 0.733 |
| F1 Flow matching | 0.836 | 0.691 |
| S1 SB isotropic | 0.844 | 0.710 ± 0.011 |
| S3 SB theo r | 0.844 | 0.707 ± 0.019 |
| S4 SB empirical | 0.845 | 0.714 ± 0.013 |

Q1b (severe): AUROC trong từng nhóm `r` — H1 **0.753**, S1 0.714, S3 0.701, S4 0.697.
→ SB thua mapping ~3.8 điểm (có ý nghĩa thống kê), Q1a và Q1b đều thua. Khôi phục latent nói chung rất đáng giá: 0.324 → 0.748 ở kênh severe.

### 8.2 Đọc tài liệu + quét β₀ (S1, seed 0, kênh severe)

Agent tra cứu (I²SB + code, Jukić + code NeMo, Richter, Cross & Ragni, "Rethinking" AAAI'26, BridgeCodec) xác nhận: implement đúng thông lệ ở các điểm chính (dự đoán z\*, ghép z_p vào đầu vào, MSE), lệch ở mức nhiễu, EMA, cách đánh giá. Quét β₀:

| β₀ | 1 bước | ODE 4 bước | TB 8 mẫu | U trong nhóm r |
|---|---|---|---|---|
| 0.1 | 0.647 | 0.651 | 0.653 | 0.579 |
| 0.3 | 0.657 | 0.670 | 0.670 | 0.571 |
| 1 | 0.692 | – | – | 0.699 |
| 3 | 0.709 | 0.730 | 0.719 | 0.757 |
| **10** | **0.728** | **0.746** | 0.735 | **0.771** |

→ **Nhiễu bridge càng lớn, SB càng tốt** (ngược khuyến nghị giảm nhiễu của I²SB). Kết luận "SB thua rõ" ở 8.1 chủ yếu do tinh chỉnh.

### 8.3 Lần chạy cuối (β₀ = 10, EMA 0.999 cho mọi model, 3 seed × 6.000 bước)

| | mid | severe (chế độ tốt nhất) | severe, 1 bước | U trong nhóm r (severe) |
|---|---|---|---|---|
| **M1 Mapping** | 0.849 | **0.750 ± 0.001** | **0.750** | – |
| M3 Mapping + nhiễu theo kênh | 0.846 | 0.746 | 0.746 | – |
| H1 Mapping + phương sai | 0.846 | 0.737 | 0.737 | **0.766 ± 0.003** |
| S1 SB isotropic | 0.851 | 0.748 (ODE 4) | 0.736 | 0.757 |
| S3 SB theo r | 0.850 | 0.748 | – | 0.757 |
| S4 SB empirical | 0.851 | 0.750 (ODE 4) | 0.738 | 0.754 |

**Đối chiếu tiêu chí** (chênh lệch ghép cặp theo seed; \* = vượt 2 sai số chuẩn):

| Tiêu chí | Kết quả | Kết luận |
|---|---|---|
| SB có đáng dùng? | S1 − M1 = −0.002 (severe), +0.001 (mid) — ngang, nhưng cần ODE 4 bước | Không có lợi thế |
| Q1a: S3/S4 hơn S1? | 0.000 / +0.002 | ❌ Không tác dụng |
| Q1a: S3/S4 hơn M3? | +0.002 / +0.004\* — nhưng S1 cũng hơn M3 cùng mức | ❌ Lợi thế không đến từ reference |
| Q1b: SB hơn H1? | −0.009\* đến −0.012\* | ❌ Thua nhẹ, tốn ×32 |

**Kết luận Q1:** kết quả âm tính chắc chắn **trong thiết lập một nguồn mỗi vùng**. Phù hợp tài liệu: SB thường chỉ thắng ở chỉ số cảm nhận, còn mIoU là chỉ số kiểu sai số. Chưa kiểm tra trường hợp nhiều UAV cùng nhìn với bằng chứng mơ hồ/mâu thuẫn.

---

## 9. Kết quả Q2 — giao tiếp dựa trên độ không chắc

### 9.1 Gửi lại có chọn lọc (H1, 3 seed)

**Cách làm:** mỗi UAV gửi toàn bộ latent một lần; trạm khôi phục bằng H1 và có bản đồ U; với ngân sách gửi lại (5–40% số block trong vùng nhìn, gộp cho 5 UAV của một mẫu), trạm xin gửi lại các block được chọn; bản gửi lại qua lần dùng kênh mới (cùng trạng thái kênh, nhiễu/mất gói mới) và gộp LMMSE với bản đầu. Số bit: 8 bit/giá trị latent → 2.048 bit/block.

**Chiến lược chọn:** ngẫu nhiên · kênh xấu nhất (`1 − r` cao) · **uncertainty H1** (gộp ngân sách cho 5 UAV) · uncertainty chia đều (mỗi UAV cùng tỉ lệ) · oracle (biết trước block sai nhiều nhất).

mIoU ở mức gửi thêm 10% (14.7 kbit/UAV):

| Kênh | Sau lần đầu | Ngẫu nhiên | Kênh xấu nhất | **Uncertainty** | Chia đều | Oracle |
|---|---|---|---|---|---|---|
| mid | 0.843 | 0.846 | 0.857 | **0.866** | 0.864 | 0.868 |
| severe | 0.728 | 0.737 | 0.743 | **0.763 ± 0.008** | 0.760 | 0.782 |
| hetero | 0.820 | 0.825 | 0.833 | **0.845 ± 0.004** | 0.842 | 0.851 |

**Tiết kiệm bit** (để đạt cùng mIoU với uncertainty, chọn theo kênh cần gấp mấy lần bit):

| Gửi thêm | mid | severe | hetero |
|---|---|---|---|
| 5% | 2.8× | 2.8× | 2.8× |
| 10% | 2.1× | 2.3× | 2.3× |
| 20% | 2.0× | 1.9× | 1.8× |

Ngẫu nhiên không đuổi kịp kể cả ở 40%. **Gộp ngân sách giữa các UAV** so với chia đều chỉ thêm 1.0–1.5× (chênh mIoU ≤ 0.4 điểm, trong mức dao động seed). Khi gộp, ngân sách dồn về UAV kênh xấu: ở 20%, 1/3 UAV kênh tệ nhất nhận ~31–34% block của chúng, 1/3 tốt nhất ~8%.

### 9.2 Gửi 2 vòng (3 seed)

**Cách làm:** cùng tổng ngân sách B: *gửi 1 lần* = mỗi UAV gửi ngẫu nhiên B% block; *gửi 2 vòng* = gửi ngẫu nhiên B/2, trạm khôi phục + chấm điểm block, xin thêm B/2 theo chiến lược. Mọi biến thể khôi phục bằng **cùng một H1** (chỉ khác cách chọn). H1 và SB được train lại với mất gói tới 90%, EMA.

Kênh hetero:

| Tổng bit/UAV | Gửi 1 lần | Ngẫu nhiên | Kênh xấu nhất | **Uncertainty H1** | Uncertainty SB (8 mẫu) | Oracle |
|---|---|---|---|---|---|---|
| 30% (45 kbit) | 0.671 | 0.656 | 0.630 | **0.705** | 0.701 | 0.775 |
| 50% (74 kbit) | 0.737 | 0.718 | 0.712 | **0.780** | 0.772 | 0.815 |
| 70% (102 kbit) | 0.777 | 0.758 | 0.772 | **0.814** | 0.804 | 0.829 |
| *Gửi đủ 100% (147 kbit)* | *0.821* | | | | | |

Kênh mid cùng xu hướng (70%: 0.837 so với gửi đủ 0.844).

- Gửi 2 vòng theo uncertainty hơn gửi 1 lần **3.5–4.3 điểm** ở cùng số bit; **70% bit đạt gần bằng gửi đủ** → tiết kiệm ~30% tổng bit (gửi 1 lần cần ~1.4× bit để bằng).
- Xin thêm ngẫu nhiên hoặc theo kênh xấu **còn tệ hơn gửi 1 lần**: chúng hay xin lại UAV kênh tệ, gửi lại vẫn hỏng.
- **SB không giúp chọn block:** SB − H1 = −0.3 / −0.7 / −1.0 điểm (hetero), −0.2 / −0.7 / −0.8 (mid), tốn ×32.
- Còn cách oracle xa ở ngân sách thấp (30%: 0.705 vs 0.775).

---

## 10. SB có mạnh hơn khi thông tin cực ít? (3 seed)

Mỗi UAV chỉ gửi ngẫu nhiên 15–100% block (kênh hetero), khôi phục bằng H1 hoặc SB (cùng các model của 9.2).

**mIoU:**

| Gửi | Không khôi phục | H1 | SB 1 bước | SB ODE 4 | SB TB 8 mẫu | SB TB xác suất 8 mẫu |
|---|---|---|---|---|---|---|
| 15% | 0.308 | **0.576** | 0.536 | 0.564 | 0.552 | 0.550 |
| 30% | 0.395 | 0.673 | 0.658 | **0.674** | 0.669 | 0.667 |
| 50% | 0.492 | 0.738 | 0.733 | **0.744** | 0.740 | 0.739 |
| 70% | 0.572 | 0.779 | 0.779 | **0.786** | 0.783 | 0.783 |
| 100% | 0.673 | 0.826 | 0.828 | **0.831** | 0.829 | 0.829 |

**IoU lớp xe:**

| Gửi | H1 | SB 1 bước | **SB ODE 4** | Chênh (ODE − H1) |
|---|---|---|---|---|
| 15% | 0.145 | 0.143 | **0.155** | +1.0 điểm |
| 30% | 0.259 | 0.259 | **0.272** | +1.3 |
| 50% | 0.373 | 0.375 | **0.388** | +1.5 |
| 70% | 0.458 | 0.463 | **0.473** | +1.6 |
| 100% | 0.568 | 0.573 | **0.580** | +1.2 |

- Lập luận "SB mạnh khi thông tin khan hiếm" **không đúng về mIoU**: ở 15% SB thua 1.2–3.9 điểm (kém ở đường/nền).
- **SB giữ xe tốt hơn H1 ổn định ở mọi mức** — đúng chỗ lý thuyết dự đoán (hồi quy làm mờ vật nhỏ thành nền).
- **Cảnh báo:** H1 đoán kém mapping thường ~1.3 điểm (mục 8.3), nên phải so lại với mapping thường → mục 11.

---

## 11. Mapping vs SB khi cả hai được nâng cấp như nhau

**Câu hỏi:** lợi thế lớp xe của SB còn không khi so với **mapping thường** và cả hai được trang bị giống nhau?

| Cấu hình | Mapping (M1) | SB (β₀=10, ODE 4 bước) |
|---|---|---|
| base | MSE latent | MSE latent |
| upgraded | + 1.0 × cross-entropy phân đoạn qua decoder đông cứng + đầu ra giữ phần đã nhận `z_p + sqrt(1−r)·net` | như bên trái |

Train mất gói tới 90%, EMA, 3 seed × 6.000 bước; đánh giá kênh hetero, gửi 15–100%, mIoU + IoU lớp xe. Script: `experiments/q2_progressive/decoders.py`, kết quả sẽ ở `runs/decoders.json`.

**Kết quả (3 seed; SB = ODE 4 bước, "SB@1" = 1 bước như SBM):**

mIoU:

| Gửi | Mapping base | **Mapping nâng cấp** | SB base | SB nâng cấp | SB@1 nâng cấp |
|---|---|---|---|---|---|
| 15% | 0.590 | **0.600** | 0.564 | 0.527 | 0.550 |
| 30% | 0.685 | **0.693** | 0.674 | 0.664 | 0.669 |
| 50% | 0.747 | **0.754** | 0.744 | 0.744 | 0.743 |
| 70% | 0.785 | **0.794** | 0.786 | 0.791 | 0.790 |
| 100% | 0.829 | 0.837 | 0.831 | **0.840** | 0.838 |

IoU lớp xe:

| Gửi | Mapping base | Mapping nâng cấp | SB base | SB nâng cấp | SB@1 nâng cấp |
|---|---|---|---|---|---|
| 15% | 0.152 | **0.161** | 0.155 | 0.151 | 0.158 |
| 50% | 0.385 | **0.396** | 0.388 | 0.390 | 0.393 |
| 100% | 0.573 | 0.589 | 0.580 | **0.598** | 0.595 |

Chênh lệch ghép cặp (trung bình, độ lệch chuẩn qua seed):
- **SB − mapping, cùng nâng cấp:** mIoU −7.3 (15%) → −1.0 (50%) → +0.3 (100%) điểm; lớp xe −1.0 → −0.6 → +0.9 (không ổn định, sd 0.7).
- **SB 1 bước − mapping, cùng nâng cấp:** mIoU −5.0 (15%) → +0.1 (100%).
- **SB − mapping, bản base:** lớp xe chỉ +0.2 đến +0.7 điểm (trước đó so với H1 là +1.0 đến +1.6).
- **Nâng cấp giúp mapping đều đặn:** +0.7 đến +1.1 mIoU, +0.9 đến +1.9 lớp xe ở mọi mức.

**Kết luận:**
1. Lợi thế lớp xe ở mục 10 **phần lớn đến từ việc H1 yếu hơn mapping thường**; so với mapping thường, SB chỉ nhỉnh rất ít (bản base, 70–100%) và **mất hẳn** khi cả hai được nâng cấp như nhau.
2. Công thức SBM (SB-training + loss nhiệm vụ + 1 bước) **không thắng** mapping cùng nâng cấp; thua rõ khi thông tin ít.
3. SB chỉ ngang/nhỉnh ở mức nhận đủ 100% — đúng vùng ít cần khôi phục nhất.
4. **Mapping + loss phân đoạn + giữ phần đã nhận** là bộ khôi phục tốt nhất hiện có → nên dùng cho Q2.

---

## 12. Tổng hợp đọc tài liệu (~30 paper)

Đọc bởi 5 lượt agent (1 lượt tra nguồn ngoài + 4 nhóm paper trong thư mục `pdf/`), mỗi nhận định kèm trích dẫn.

**Kết luận chung:**
1. **Không paper nào cho thấy SB thắng một mạng hồi quy train bằng MSE cùng backbone trên chỉ số độ chính xác/sai số.** Paper báo cáo cả hai loại chỉ số đều thấy cùng mẫu: chỉ số sai số nghiêng về đầu ra trung bình (I³SB: 1 bước cho PSNR tốt nhất; SDDBM: ODE thắng PSNR, SDE thắng FID; SBGSC nhường PSNR cho SwinJSCC ở băng thông cao; I²SB từ chối báo PSNR vì "ưa đầu ra hồi quy mờ"). So sánh cùng backbone duy nhất ở speech (Jukić 2024): hoà SI-SDR khi khử nhiễu, thắng khi khử vang, và chỉ khi có loss phụ.
2. **SB đã có trong SemCom** (SBGSC, 4/2026) → tính mới chỉ đứng ở phạm vi UAV/BEV/nhiều agent.
3. **GSBM và các biến thể "SB chính danh"** (DSBM, SF-SB, SB Flow, ASBM, Feedback SBM, Soft-constrained SB) chủ yếu cho dữ liệu **không có cặp**; khi đã có cặp, chúng quy về bridge ta đang dùng. Chi phí nhiệm vụ của GSBM chỉ uốn quãng giữa, không đổi điểm đích → **không nên làm**.
4. **Mẹo chuyển được:** loss nhiệm vụ đi qua decoder (giống loss miền thời gian/mel ở speech — cải thiện lớn và đều nhất), giữ nguyên phần đã nhận (range–null, mask), sampler ODE cho chỉ số sai số; GAN một bước chủ yếu để tăng tốc.
5. **Ảo giác phổ biến** ở mô hình sinh (WER/CER tệ hơn đầu vào ở nhiều paper, BridgeCodec 1 bước).
6. **SB cho điều khiển bầy UAV** (Optimal steering I–III, Branched SBM): bài toán có cơ sở toán học (đưa phân bố vị trí về đích với năng lượng nhỏ nhất, thêm chi phí liên lạc theo kiểu SB tổng quát), nhưng là bài toán điều khiển khác hẳn; với 5–20 UAV, gán vị trí + tối ưu quỹ đạo có thể ngang SB.
7. **Unit test lý thuyết:** SB Gaussian dạng đóng (Bunne et al.) dùng để kiểm chứng code; lưu ý bridge có cặp của ta chỉ trùng SB entropic khi hiệp phương sai cặp bằng `C_β`.

**Kiểm tra nhận định của ChatGPT (một số chỗ sai/thiếu):**
- Các paper steering **không** nói gì về "phủ vùng" hay "mất kết nối".
- "3D navigation" của Branched SBM là đám điểm trên địa hình LiDAR, không có robot.
- ">8× nhanh hơn" của SBGSC không khớp bảng của chính nó (3.7×–100× tuỳ baseline).
- Degradation-aware DSB dùng mô hình suy giảm **học được**, không phải cơ chế vật lý.
- Tài liệu "Tóm tắt nghiên cứu": nhầm SBM với BridgeCodec; gán sai tác giả OPV2V/V2X-Sim/DAIR-V2X; cho rằng Town3 có sẵn nhãn BEV 4 lớp (bản mẫu không có); cho rằng SB nên dùng ít nhiễu (thực nghiệm cho thấy ngược lại).

---

## 13. Hạn chế hiện tại

- Chỉ có **bản mẫu Town3** (340 mẫu); bản đầy đủ cần Baidu.
- **Nhãn tự dựng 3 lớp** (gộp nhà vào "khác"); không so trực tiếp được với paper dùng BEV 4 lớp chính thức.
- **Leader setup một nguồn mỗi vùng**: chưa có trường hợp nhiều UAV cùng nhìn (nơi SB có lý do lý thuyết để thắng).
- **Kênh mô phỏng đơn giản:** chưa có lượng tử hoá/VQ thật (bit tính theo giả định 8 bit/giá trị), chưa tính chi phí tin nhắn xin gửi lại và độ trễ vòng 2, lần gửi lại dùng cùng trạng thái kênh, chưa có dữ liệu cũ (staleness) trên dữ liệu thật.
- Mỗi model 6.000 bước — ngắn so với tài liệu (100k–200k bước); đã kiểm soát bằng EMA và so sánh cùng ngân sách.
- Một số thí nghiệm phụ (quét β₀) chỉ 1 seed.

---

## 14. Bước tiếp theo đề xuất

1. **Đọc kết quả mục 11** → quyết định SB có điểm mạnh riêng (giữ vật thể nhỏ) hay chỉ còn vai trò so sánh.
2. **Thu hẹp khoảng cách với oracle ở Q2:** học bộ dự đoán "xin block này tăng mIoU bao nhiêu" (value predictor) thay vì chỉ dùng độ không chắc.
3. **Làm Q2 thực tế hơn:** lượng tử hoá thật, chi phí phản hồi + độ trễ.
4. **Dataset có vùng chồng** (bản đầy đủ CoPerception-UAV chế độ dynamic, CoPerception-UAVs+, UAV3D) để mở lại câu hỏi "UAV khác lấp chỗ thiếu" — dùng lại script đo độ chồng.
5. **Q3** (chọn chế độ gửi, bandit trước DRL) — sau khi Q2 ổn định.
6. Hướng dài hạn tách riêng: SB cho điều khiển phân bố bầy UAV có ràng buộc liên lạc.

---

## Phụ lục A — Lệnh chạy chính

```powershell
# kiểm tra
.venv\Scripts\python.exe -m tests.test_q1_core
.venv\Scripts\python.exe scripts\check_geometry.py data\coperception_uav\town3
.venv\Scripts\python.exe scripts\inspect_labels.py data\coperception_uav\town3 out.png

# dữ liệu + teacher
.venv\Scripts\python.exe -m swarmbridge.data.coperception_uav.build_bev data\coperception_uav\town3 data\coperception_uav\bev_town3
.venv\Scripts\python.exe -m swarmbridge.teacher.train data\coperception_uav\bev_town3 runs\teacher --epochs 30

# Q1 (lần chạy cuối) + gộp kết quả
.venv\Scripts\python.exe -m swarmbridge.experiments.q1_reference.run --data real --models M1 M3 H1 S1 S3 S4 --steps 6000 --seed 0 --beta0 10 --ema 0.999 --out runs/q1_b10_ema_s0.json
.venv\Scripts\python.exe -m swarmbridge.experiments.q1_reference.aggregate "runs/q1_b10_ema_s*.json"

# Q2
.venv\Scripts\python.exe -m swarmbridge.experiments.q2_retx.run --seed 0 --out runs/q2_retx_v2_s0.json
.venv\Scripts\python.exe -m swarmbridge.experiments.q2_retx.aggregate "runs/q2_retx_v2_s*.json"
.venv\Scripts\python.exe -m swarmbridge.experiments.q2_progressive.run --seed 0
.venv\Scripts\python.exe -m swarmbridge.experiments.q2_progressive.restorers
.venv\Scripts\python.exe -m swarmbridge.experiments.q2_progressive.decoders --seeds 0 1 2
```

## Phụ lục B — File kết quả

| File | Nội dung |
|---|---|
| `runs/teacher/summary.json` | Teacher v2 |
| `runs/teacher_v1_flatground/summary.json` | Teacher v1 (nhãn cũ) |
| `runs/q1_synthetic*.json` | Kiểm tra pipeline (dữ liệu giả) |
| `runs/q1_real_s0.json` | Q1 đầu tiên, nhãn cũ, 1 seed (lịch sử) |
| `runs/q1_real_6k_s{0,1,2}.json` | Q1, β₀=1 |
| `runs/q1_sweep_beta{0.1,0.3,3,10}.json` | Quét β₀ |
| `runs/q1_b10_ema_s{0,1,2}.json` | **Q1 cuối** |
| `runs/q2_retx_v2_s{0,1,2}.json` | **Q2 gửi lại** |
| `runs/q2p_s{0,1,2}.json` | **Q2 gửi 2 vòng** |
| `runs/q2p_restorers_v2.json` | SB vs H1 khi thông tin ít (mIoU + lớp xe) |
| `runs/decoders.json` | Mapping vs SB, base vs nâng cấp, kể cả SB 1 bước |
