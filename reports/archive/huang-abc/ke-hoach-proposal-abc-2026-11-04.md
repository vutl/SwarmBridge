# Kế hoạch cho proposal A / B / C (SB, CSI, NLoS, beam, multi-UAV SemCom) — 2026-11-04

Người đọc: nhóm nghiên cứu SwarmBridge, ChatGPT/Codex khi phản biện. Tài liệu này thay cho các mô tả proposal trong chat.
Nguồn: đọc toàn văn 17 paper trong `pdf/` (5 agent đọc song song, Claude kiểm tra lại các trích dẫn then chốt) và
khoảng 40 lượt tìm prior art trên web. Số trang là trang PDF. Mục nào chỉ dựa trên abstract/README được ghi rõ.

## 0. Tóm tắt

- **A (SB ước lượng CSI)** không đứng được thành paper riêng: hai bài bridge cho CSI đã có (cùng nhóm Southeast Univ.),
  và mảng diffusion CSI rất đông. A chỉ còn giá trị như một **thành phần** — nếu chứng minh được *mẫu posterior* giúp ra
  quyết định tốt hơn, ở cùng chi phí tính, so với flow matching / DPS / hồi quy có uncertainty.
- **B (belief kênh chia sẻ giữa nhiều UAV)** là hướng mở nhất theo prior art và có lý do thật cho MARL (UAV này dò pilot
  có ích cho UAV khác ở các bước sau). Nhưng **tiền đề vật lý chưa được kiểm**: ở độ cao UAV phần lớn link có thể là LoS,
  và phần kênh chung giữa các UAV có thể đã nằm hết trong bản đồ tĩnh (CKM). Phải đo trước.
- **C (quyết định beam + message theo posterior kênh và giá trị semantic)**: từng mảnh đều đã có người làm; tổ hợp
  "posterior CSI → beam + message có ràng buộc rủi ro → đo bằng task thật (AMOTA)" chưa thấy. Rủi ro bị xem là
  "SAA với mẫu diffusion". Bài học UAV3D làm hẹp C lại: chọn *kích thước* message đã có lời giải (message lũy tiến),
  nên posterior chỉ có thể có giá trị ở **chọn beam/đường truyền** khi posterior đa đỉnh (blockage).
- **Đề xuất:** không chọn B hay C bằng cảm tính. Chạy **Giai đoạn 0** (CPU, ~1–1.5 tuần, chưa train mô hình sinh nào):
  dựng dữ liệu kênh UAV bằng Sionna RT, rồi đo ba khoảng cách tới oracle — F1 (giá trị chia sẻ giữa UAV, cho B),
  F2 (giá trị posterior so với ước lượng điểm, cho C), F3 (giá trị lập kế hoạch nhiều bước, cho MARL). Hướng nào qua
  ngưỡng thì đi tiếp; SB chỉ vào khi F2 qua ngưỡng và thắng ở Giai đoạn 1.

## 1. Prior art: đã có gì, còn gì

### 1.1 Bridge / SB cho CSI (đối thủ trực tiếp của A)

| Paper | Đã làm | Chưa làm |
|---|---|---|
| Chen et al., DSB channel estimation for MIMO ISAC, IEEE WCL 2025 | Bridge có cặp LS → H, VE reference, ODE **1 bước**, UNet miền góc; N=32, kênh tổng hợp băng hẹp (L=3 path), SNR −10…10 dB; hơn UNet cùng kiến trúc ~0.7–2 dB NMSE (p.5) | Không dùng likelihood lúc suy luận; tự viết "sample diversity is unnecessary" (p.4) → không có uncertainty, không quyết định nào; không so DDPM/DPS/flow; LMMSE của họ gần 0 dB ở −10 dB (đáng ngờ); không UAV, không OFDM, không mobility |
| Xu et al., Brownian-bridge channel denoising, ISAC massive MIMO, IEEE TCCN 2025 (cùng nhóm) | Bridge rời rạc kiểu BBDM, T=10 (tốt nhất 4–8 bước), noise κ₀² chỉnh **theo từng SNR**; hơn UNet 2–4 dB | Bản xác định (κ₀²=0) thua UNet (Fig. 9) → lợi ích có thể từ tính ngẫu nhiên + chỉnh theo SNR, chưa tách bạch; không flow/DPS, không cùng NFE; chỉ NMSE |
| I²SB (ICML 2023) | Bridge có cặp; khi noise → 0 thành OT-ODE = nội suy tuyến tính như flow matching (Cor. 3.5) | OT-ODE tốt cho tác vụ gần xác định, kém cho tác vụ bất định cao (Table 6) |
| Rethinking Flow and Diffusion Bridge Models for SE (AAAI 2026; README + abstract) | Flow và bridge với loss dự đoán dữ liệu ≈ mô hình dự đoán; cải thiện so với TF-GridNet "marginal" | — |

**Kết luận A:** "SB đầu tiên cho ước lượng kênh" đã bị lấy. Bridge khác flow/hồi quy chỉ khi *dùng mẫu* trong chế độ
posterior bất định thật — đúng điều cả hai bài ISAC đều không làm.

### 1.2 Diffusion cho ước lượng CSI (baseline bắt buộc)

| Paper | Chế độ có lợi được ghi nhận | Ghi chú dùng lại |
|---|---|---|
| Fesl et al., DM prior low-complexity, WCL 2024 | Gần genie-LMMSE trên 3GPP; hơn Score ~5 dB SNR trên QuaDRiGa; **≈ LS từ 30 dB** | Repo BSD, có weight 64×16, LMMSE sample/genie — **dùng lại trực tiếp** |
| Zhou et al., high-dimensional DM, TWC 2025 | Pilot 20–60% số antenna, ADC 1–3 bit; BER 64-QAM hơn LMMSE >5 dB; 4 ms vs 280 ms (SGM) | Repo chọn bước tốt nhất bằng GT (`np.min` trên NMSE) — **phải sửa** trước khi làm baseline; không license |
| Chen, Shin, Nallanathan, DM variational inference, TCOM 2025 | Hơn Langevin 5–11 dB ở SNR thấp (CDL) | Không có LMMSE dùng hiệp phương sai; không code; train CDL-C hỏng trên CDL-A |

Chưa bài nào: so cùng NFE giữa các sampler, flow matching hay SB, đo calibration posterior, kênh UAV.
Chế độ bão hòa (generative ≈ cổ điển): đủ pilot + SNR cao; kênh gần Gaussian (≈ genie LMMSE); SNR ≤ −20 dB.

### 1.3 CSI theo môi trường / cảm biến

| Paper | Làm gì | Giới hạn |
|---|---|---|
| EA-DM (Hu…Caire, arXiv 2609.16588) | DiT có điều kiện vị trí 2D + likelihood pilot, Sionna RT cảnh Munich, 28 GHz, 64×16, 32 sóng mang | UE mặt đất 1.5 m, một scene, không chia theo scene, NFE 100, một mẫu |
| Cross-modal flow (Liang…Hanzo, arXiv 2512.04966) | Camera + LiDAR + GPS → CSI không pilot, flow matching, 1–2.5 ms; SE "tới +25%" | GPS chiếm 70–84% attention; chỉ camera/LiDAR ≈ +0.2 dB NMSE (vô dụng); một giao lộ, chia ngẫu nhiên 9:1 (site-specific); suy luận dùng μ, không mẫu; code chưa phát hành |

### 1.4 Multi-UAV SemCom + RL (đối thủ của C)

| Paper | Nhiều UAV | Beamforming | CSI bất định | Task thật | RL |
|---|---|---|---|---|---|
| Wen et al., Semantic-Driven Multi-UAV CoMP, WCL 2026 | 3 | Có (SOCP) | Không (CSI hoàn hảo, p.3) | Không (hàm logistic của SINR) | PPO + AO |
| Zheng et al., Confidence-Calibrated Semantic RA, JWCN 2026 | 4–8 | Không | Không | **Có** (mAP@50 VisDrone; chọn detection theo confidence đã hiệu chuẩn + độ trùng) | SAC phân cấp |
| Zhou et al., Adaptive semantic compression, JWCN 2026 | 3–5 | Không | Một phần (chance constraint, thống kê fading đã biết) | Không (MS-SSIM) | MAPPO |
| Liao et al., GOST satellite-UAV tracking, TMC 2026 | 12 | Không | Một phần (mất gói, suy dữ liệu) | Tracking điểm tổng hợp, không MOTA | MADDPG |
| Zhao et al., UA-TSC, TNSE 2026 | 1 | Không | Không | Độ chính xác fit offline (AID) | PPO |
| Yang et al., PASC, WCL 2026 | 3 | Không | Dự đoán điểm (GRU) | Độ chính xác profile (CIFAR-100) | PPO |
| Tang, Feng, Quek, TVT 2026 (arXiv 2411.06136, abstract) | Leader-follower | MIMO fading | Có kênh không hoàn hảo | Tracking **điều khiển** (bám đội hình), không perception | Drift-plus-penalty, không RL |

Zheng et al. gần nhất với "chọn semantic evidence theo giá trị task"; chưa bài nào dùng phân phối CSI, chọn beam/pilot
theo rủi ro, hay đo bằng tracking 3D thật.

### 1.5 Prior art trên web cho B và C (abstract/HTML, chưa đọc toàn văn)

**B:**
- Zhang, Ferdowsi, Saad, Bennis, distributed CGAN cho kênh mmWave UAV (TWC 2022): UAV hợp tác *học mô hình* kênh.
- Zeng et al., CKM + predictive beamforming cho UAV (arXiv 2404.12060); CKM bằng lọc Bayes nhị phân (arXiv 2409.00016);
  BeamCKMDiff, diffusion sinh CKM theo beam cho BS–UAV (arXiv 2609.34704): một UAV, bản đồ offline.
- Que…Jin, cooperative multi-modal SLAM → radio map → beam (arXiv 2507.05718, TCOM): nhiều UE mặt đất, không RL.
- Trigka et al., ước lượng kênh thưa phân tán với support góc chung (EUSIPCO 2021): ý tưởng "scatterer chung" cổ điển.
- Imran, Charan, Alkhateeb, environment semantic communication cho beam (2024): cảm biến cố định gửi semantic môi trường.
- Active sensing học được (Sohrabi/Jiang/Yu): một tác tử.

Còn mở (theo tìm kiếm, chưa quét IEEE Xplore): suy luận **trực tuyến** trạng thái vật chắn/scatterer chung từ message nhỏ
giữa các UAV, cùng MARL quyết định UAV nào tốn pilot, dưới ngân sách bit.

**C:**
- Su, Hou, He, Simeone, conformal robust beamforming via generative channel models (arXiv 2504.06934).
- Tang…Letaief, flow-based posterior sampling → chọn port theo kỳ vọng posterior (arXiv 2605.29679).
- Zhao, Wang, Liu, generative site-specific beamforming, flow matching sinh beam từ RSRP (arXiv 2602.12552).
- Othman, Bariah, Debbah, diffusion prior cho beam alignment top-k (arXiv 2604.09653).
- Li et al., VAE channel generator + stochastic WMMSE robust beamforming (arXiv 2404.06055).
- UEP / phân bổ công suất theo importance semantic (arXiv 2604.00595, 2502.16194, 2306.14138): giả định biết CSI.
- CVaR / mean-variance RL cho mmWave (Vu et al. 2018; Safehaul; Al-Senwi et al. TVT 2023).

Không tìm thấy bài dùng **SB** sinh mẫu posterior kênh rồi đưa vào quyết định beam. Nhưng "mẫu posterior từ mô hình
sinh → quyết định" đã có (diffusion/flow), nên thay bằng SB là đóng góp yếu.

## 2. Phản biện từng proposal

### A — pilot-conditioned SB cho kênh NLoS UAV
- Bridge từ LS đã có; kết quả NMSE của nó chưa tách được khỏi "UNet tốt hơn" (1 bước ODE = mạng khử nhiễu).
- Khi τp < N (thiếu pilot), không có LS sạch để làm điểm đầu bridge; điểm đầu phải là LMMSE/pseudo-inverse — tức bridge
  bắt đầu từ ước lượng đã chứa prior, lúc đó lợi thế "điểm đầu có thông tin" so với DPS không hiển nhiên.
- Bridge không dùng likelihood lúc suy luận; DPS dùng tường minh. Muốn gọi là posterior p(H | Y, P, c) phải có kiểm tra
  measurement consistency và calibration.
- **Vai trò hợp lý của A:** Giai đoạn 1, bài kiểm tra "mẫu từ mô hình nào cho quyết định tốt nhất ở cùng NFE", chỉ khi
  F2 cho thấy posterior có giá trị quyết định.

### B — belief kênh chia sẻ cho bầy UAV
- Điểm mạnh: lý do thật cho MARL (thu thập thông tin có giá trị chung, nhiều bước); prior art gần nhất học mô hình offline
  hoặc một tác tử.
- Rủi ro vật lý (phải đo ở F1):
  1. Ở mmWave, kênh của hai UAV cách nhau vài mét gần như không tương quan; thứ chung là hình học (scatterer gần BS, vật
     chắn). Hình học tĩnh thì bản đồ (CKM) học offline đã nắm → chia sẻ trực tuyến chỉ có giá trị với **vật chắn động**
     (xe lớn, cây, công trình tạm, UAV khác) hoặc môi trường **chưa có bản đồ**.
  2. Ở độ cao ≥ 50–100 m, tỉ lệ LoS tới BS có thể rất cao → NLoS chỉ quan trọng ở độ cao thấp (đô thị, < 50 m).
- Nếu F1 cho thấy (chia sẻ) − (bản đồ tĩnh + pilot riêng) nhỏ, B chết trước khi tốn công MARL.

### C — quyết định beam + message theo posterior và giá trị semantic
- Bài học UAV3D áp vào C:
  - Kích thước message: message lũy tiến có hạn chót ≈ oracle biết tốc độ kênh → chọn rate theo posterior **khó có
    giá trị** nếu payload là lũy tiến. Baseline lũy tiến là bắt buộc.
  - Thứ tự nội dung: chọn theo score + bỏ trùng giữa helper đã rất mạnh; feedback "receiver đã chắc" không hơn khi tính
    đủ bit. Một "semantic importance" mới phải thắng các baseline đó, không chỉ thắng gửi ngẫu nhiên.
- Chỗ posterior *có thể* có giá trị: **chọn beam/đường** khi posterior đa đỉnh (đường phản xạ trái hay phải còn sống),
  vì beam sai → rate ≈ 0 với cả message lũy tiến. Đây là giả thuyết đo được ở F2.
- Baseline bắt buộc: beam theo ước lượng điểm (MMSE), quét beam top-k (tính chi phí pilot), conformal robust beamforming
  (2504.06934), robust Gaussian-error, message lũy tiến.

## 3. Giai đoạn 0 — đo khoảng cách tới oracle trước khi học (CPU)

### F0. Dữ liệu kênh UAV (Sionna RT)
- `sionna-rt` 2.2.0 (Apache-2.0) chạy được trên CPU qua Dr.Jit LLVM (chưa kiểm tra tốc độ trên máy này). Cảnh Munich
  có sẵn (EA-DM dùng cảnh này). BS 8×8 UPA, 28 GHz, OFDM 32 sóng mang (giống EA-DM để so được); UAV 1 hoặc 4 antenna.
- Lưới UAV ở 20 / 50 / 100 m; thêm quỹ đạo UAV (5–15 m/s) và **vật chắn động** (xe tải/bus di chuyển, nếu Sionna RT
  cho phép object chuyển động) để có NLoS thay đổi theo thời gian.
- Chia train/test **theo vùng** (không ngẫu nhiên 9:1 như cross-modal flow).
- Thống kê đầu ra, viết thành report trước khi làm F1–F3: tỉ lệ LoS theo độ cao; số path và độ trải góc; độ chồng lấp
  support góc phía BS giữa hai UAV theo khoảng cách; thời gian kết hợp theo tốc độ UAV.
- Nếu tỉ lệ NLoS ở mọi độ cao hợp lý đều rất thấp → báo cáo và dừng, hoặc chuyển sang A2A / độ cao thấp.

### F1. Giá trị chia sẻ giữa UAV (cho B)
Với UAV i, ước lượng beam/kênh từ Np pilot của chính nó, so bốn mức thông tin:
(a) chỉ pilot riêng (LMMSE/OMP); (b) + bản đồ tĩnh (CKM học từ vùng train); (c) + phép đo hiện tại của các UAV khác,
kết hợp bằng oracle (support chung đã biết / lọc Bayes trên trạng thái vật chắn thật); (d) biết hết (genie).
Đo: số pilot cần để mất beam gain ≤ 1 dB; outage; riêng các mẫu NLoS.
**Ngưỡng giữ B:** (c) so với (b) tiết kiệm ≥ 25% pilot hoặc ≥ 1 dB beam gain trong mẫu NLoS. Có ablation: vật chắn độc
lập giữa các UAV → lợi ích phải biến mất (nếu không, lợi ích không đến từ belief chung).

### F2. Giá trị posterior so với ước lượng điểm (cho C)
- Hai tầng: (i) mô hình tổng hợp có posterior chính xác — kênh = trộn LoS/NLoS với trạng thái chắn ẩn, tính posterior
  bằng Bayes; (ii) trên dữ liệu F0, posterior "oracle thực nghiệm" từ các mẫu cùng vùng/cùng trạng thái chắn.
- Quyết định mỗi frame: beam từ codebook + payload lũy tiến của UAV3D. Lợi ích task: hàm AMOTA theo số box đến của
  từng helper, fit từ kết quả `runs/uav3d_proto/{budget,queue}` (không cần chạy lại tracking cho mỗi quyết định; kiểm tra
  lại bằng tracking thật ở vài điểm).
- So: beam theo MMSE điểm; quét top-k (tính pilot); conformal robust; Gaussian robust; **quyết định theo posterior
  oracle** (kỳ vọng và ràng buộc outage/CVaR).
- **Ngưỡng giữ C (và SB):** posterior oracle hơn baseline tốt nhất ≥ 0.01 AMOTA (tương đương) hoặc giảm ≥ 50% tỉ lệ
  frame mà lợi ích rơi dưới ego. Nếu posterior *chính xác* còn không giúp, mô hình sinh học được không thể giúp.

### F3. Giá trị lập kế hoạch nhiều bước (cho MARL)
Trên F1: chính sách dò pilot tham lam một bước (VoI myopic) so với planner nhiều bước có oracle (tìm kiếm cạn trên tầm
nhìn ngắn, ít UAV). **Ngưỡng dùng MARL:** planner hơn tham lam rõ (≥ 10% pilot hoặc ≥ 0.5 dB). Nếu không, heuristic là đủ
— giống kết luận "RL chọn kích thước không có đất" ở UAV3D.

## 4. Giai đoạn 1 — chỉ khi F2 (hoặc F1) qua ngưỡng: so mô hình posterior (vai trò của A)

Cùng dữ liệu F0, cùng NFE ∈ {1, 4, 10, 50}:
- Cổ điển: LS, LMMSE (sample + genie covariance), OMP miền góc.
- Hồi quy: UNet cùng kiến trúc; Gaussian heteroscedastic; deep ensemble; conformal.
- Sinh: Fesl DM (repo BSD); DPS (code Zhou, đã sửa chọn bước bằng GT); conditional flow matching / OT-ODE;
  Brownian bridge (TCCN); VE-DSB 1 bước (WCL); bridge ngẫu nhiên **có** measurement guidance (ứng viên của ta).
- Đo: NMSE; energy score/CRPS; coverage của khoảng beam gain; **regret quyết định** so với posterior oracle F2; độ trễ.
- **SB giữ lại** chỉ khi thắng regret quyết định và calibration ở cùng NFE so với flow và DPS. Nếu thua: dùng mô hình
  thắng, bỏ SB, không cố cứu.

## 5. Giai đoạn 2 — hệ thống đầy đủ

- Nếu B qua: belief chung (filter/GNN) + message belief có ngân sách bit + MARL dò pilot; so với (a)(b)(c) ở F1 và với
  centralised fusion; báo pilot tiết kiệm ở cùng outage và bit chia sẻ so với lợi ích.
- Nếu C qua: beam + message theo posterior với ràng buộc chance/CVaR, đo AMOTA thật trên UAV3D (payload đã có).
- Nếu cả hai: tích hợp — belief chung cung cấp posterior cho quyết định beam + message.

## 6. Thứ tự, chi phí, điều kiện

| Bước | Việc | Tài nguyên | Ước lượng |
|---|---|---|---|
| F0 | Cài Sionna RT (WSL, CPU/LLVM), cảnh Munich + UAV, thống kê | CPU | 3–4 ngày |
| F2(i) | Mô hình tổng hợp, posterior chính xác, hàm lợi ích từ UAV3D | CPU, numpy | 2 ngày (song song F0) |
| F1, F2(ii), F3 | Trên dữ liệu F0 | CPU | 3–4 ngày |
| Giai đoạn 1 | Benchmark posterior, cùng NFE | GPU (cần duyệt) | 1–2 tuần |
| Giai đoạn 2 | B và/hoặc C | GPU | vài tuần |

Điều kiện: Giai đoạn 0 không dùng GPU và không train mô hình sinh. Mỗi bước có spec ngắn gửi Codex duyệt trước khi code
(như protocol UAV3D). Kết quả F0–F3 viết thành report với ngưỡng đã ghi ở trên **trước khi** chạy (không đổi ngưỡng sau
khi thấy số).

## 7. Những câu không được viết trong paper

- "Lần đầu dùng SB cho ước lượng kênh" (đã có, WCL 2025, TCCN 2025).
- "Lần đầu dùng mẫu từ mô hình sinh cho quyết định beam" (đã có: 2605.29679, 2602.12552, 2504.06934).
- "Lần đầu kết hợp multi-UAV + SemCom + RL" (rất nhiều, mục 1.4).
- "UAV hợp tác để học kênh" (Saad/Bennis 2022).
- Bất kỳ khẳng định "novel" nào dựa trên việc chưa tìm thấy, khi chưa quét IEEE Xplore theo từ khóa.

## 8. Việc cần người dùng quyết định

1. Đồng ý chạy Giai đoạn 0 (CPU) thay vì chọn B/C ngay?
2. Cho phép cài `sionna-rt` trong WSL (môi trường riêng, không đụng env UAV3D)?
3. Gửi tài liệu này cho Codex/ChatGPT phản biện trước khi code F0?
