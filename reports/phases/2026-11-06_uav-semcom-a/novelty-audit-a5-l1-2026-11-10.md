# Kiểm toán prior-art / độ mới cho A5, A5.5, A6.0-L1 (hướng A)

Ngày: 2026-11-10. Đây là audit chỉ đọc. Nguồn duy nhất là các PDF trong `pdf/`; không tra web, không đọc tài liệu UAV3D / Active Semantic Recovery.

## 0. Cách đọc báo cáo này

- **Trích dẫn trang:** "p.N" là số trang PDF, đếm theo ký tự ngắt trang của `pdftotext`. Số này không phải số trang in trên tạp chí; ví dụ Huang JSAC in từ trang 3942, Zhao TNSE in từ trang 11622. Riêng bài SemanticRL (rlss) và CCM có trang hình bị tách, nên từ khoảng p.9 trở đi số trang PDF chạy trước số trang in.
- **Ai đọc bài nào:** Các bài gần nhất (Huang, Zhao, JOSC, CCSRA, PASC, Hybrid-action, Cross-modal HDRL, Semantic-Driven Multi-UAV) do tôi đọc trực tiếp các phần system model, MDP/reward, mô phỏng và baseline. Nhóm collaborative perception (Where2comm, CoSDH, How2comm, SComCP, CoDS, HyComm, HMS-SCP, Drones Help Drones, HARQ, SchedCP) và nhóm phụ (SemanticRL, Data Significance, Attention-RL, CCM, GOST, Multi-Modal UAV) do hai sub-agent trích theo một mẫu cố định. Tôi đã kiểm tra lại vài chi tiết then chốt: phần thưởng label-dependent / label-free và khung 200 ms của SchedCP; Where2comm chọn supervised learning thay vì RL.
- **"Không tìm thấy"** nghĩa là tôi đã grep văn bản trích ra mà không thấy. Không có nghĩa là bài chắc chắn không có.
- **Bài "Zhao":** Trong `pdf/` chỉ có một bài có tác giả đầu là Zhao: *Anning Zhao et al., "Adaptive Task-Oriented Semantic Communication for UAVs With Joint Trajectory and Resource Optimization", IEEE TNSE vol. 13, 2026*. Có ba bài khác có tác giả tên Zhao nhưng không đứng đầu, đều được audit bên dưới:
  - Jun Zhao, tác giả thứ 2 của bài Hybrid-Action RL (GLOBECOM 2023);
  - Zhifeng Zhao, tác giả thứ 4 của SemanticRL;
  - Zhaohui Yang, đồng tác giả của PASC.
- **Bài relay / cooperative:**
  - Hai bài có UAV làm relay thật: Huang (UAV chuyển tiếp LEO → người dùng mặt đất) và Multi-Modal UAV (Hu et al., TCOM 2025, có link UAV–UAV relay).
  - "Semantic-Driven Multi-UAV Cooperative…" là CoMP downlink, tức nhiều UAV cùng phát cho người dùng, không phải relay.
  - "Drones Help Drones" là collaborative perception, không có mô hình kênh.

## 1. Kết luận ngắn

1. **A5 không mới ở khung bài toán.** Phần lõi của A5 là RL chọn mode nén cho từng ảnh dưới ràng buộc trễ, với menu trộn codec cổ điển và generative caption→diffusion. Phần này đã có gần như nguyên dạng trong Huang JSAC 2025:
   - Mode 1 là JPEG2000 truyền bit; Mode 2 là text→image; Mode 3 là text + VQ embedding với 3 mức.
   - Bộ giải là DSAC. Mỗi task có ràng buộc trễ D_max; chất lượng đo bằng PSNR.
   - Có baseline "Hovering UAV".
   - Nguồn: Huang §III-A p.5, (14) p.7, (15) và (18) p.8, §V p.9–11.

   Ngoài Huang, việc chọn mode/mức nén rời rạc bằng RL kèm deadline còn có ở Zhao TNSE, xmodal, mmuav, JOSC và PASC.
2. **A6.0-L1 không mới ở ý "reward = giá trị task đo bằng detector trên VisDrone, nhiều UAV, kênh chung".** CCSRA (Zheng et al., EURASIP JWCN 2026) đã làm đúng tổ hợp này, nhưng truyền token/feature thay vì ảnh nén:
   - YOLOv8-n fine-tune trên VisDrone2019-DET, 4 UAV, OFDM 16 subchannel, RL phân cấp.
   - Reward có hạng utility U_t = Utility(Ô_t, O_t) so với ground truth; đánh giá bằng mAP@50.
   - Nguồn: (33)–(34) p.14, (49) p.18, §6.1 p.20–21.

   SchedCP (V2X) cũng đã dùng reward "giảm detection loss" (cần nhãn), cộng thêm một phiên bản label-free, trên kênh chung có khung thời gian 200 ms (p.6–8).
3. **Không bài nào so RL với heuristic mạnh đã được tune**, càng không có heuristic tự hiệu chỉnh. Baseline của họ thuộc các loại sau:
   - DRL khác;
   - cấu hình cố định hoặc ngẫu nhiên;
   - luật đơn giản không tune (Max-Rate, Round-robin, EP, tỉ lệ nén cố định);
   - một bộ tối ưu cận thị theo mô hình: PPDO của xmodal, Lyapunov-SCA của CCSRA, exhaustive search của JOSC.

   Mọi bài đều báo RL thắng 5–60 %. Phát hiện của A5 khác hẳn: RL ≈ 0 ở regime tham chiếu, chỉ +3..4 ở vài regime, joint control yếu hơn. Đây là điểm khác biệt rõ nhất. Tuy vậy, về bản chất nó là đóng góp phương pháp đánh giá (mức trung bình), không phải đóng góp thuật toán.
4. **Phân tích bất định 3 tầng** (env seed, bootstrap theo cụm video có mô phỏng lại, training seed) **không có trong bài nào**. Mức tối đa tìm thấy:
   - xmodal: 10 seed ± 1 std (p.23, p.28);
   - JOSC: min–max trên 5 lần chạy (p.13);
   - Attention-RL: ± std trên 10 lần chạy;
   - Semantic-Driven: trung bình 500 realization (p.5);
   - Huang: chỉ "2 episode" (p.10).

   Đây là khác biệt thật nhưng thuộc loại "rigor". Đứng một mình thì yếu.
5. **Câu 3: không bài nào cho thấy reward theo task làm đổi lựa chọn mode so với reward PSNR trên cùng một hệ thống.** Các thứ gần nhất:
   - đổi trọng số chất lượng/trễ thì đổi mode (Huang Fig. 6, p.10; Hybrid-action Fig. 5a, p.5);
   - đổi trọng số task trong reward thì đổi mAP/CCSE (CCSRA Table 1, p.21–22);
   - so task-aware với rate-aware (SchedCP Fig. 12 p.11–12; Semantic-Driven RateMax p.5).

   Câu hỏi "task-aware vs PSNR-aware trên cùng menu, xem mode nào bị loại, nhất là mode generative có thể sinh FP do hallucination" vẫn trống. Đây là ứng viên đóng góp mạnh nhất của A6.0-L1, với điều kiện L1 cho thấy dịch chuyển mode đo được.

## 2. Bảng tổng hợp

Viết tắt dùng trong bảng:
- **TH** = có heuristic mạnh đã tune hay không.
- **Unc.** = cách báo cáo bất định.
- **DL** = deadline / latency.

### 2.1. Nhóm UAV / SemCom có RL (gần A5/A6 nhất)

| Bài | Setting (agent, link, nội dung truyền) | Biến quyết định | Optimizer | Objective / reward | DL | Baseline (TH?) | Dữ liệu / giá trị task | Unc. |
|---|---|---|---|---|---|---|---|---|
| **Huang JSAC 2025** | 5 LEO → 3 UAV relay → 8 user, downlink; Rician, Shannon (§II p.3–4; §V p.9). Truyền ảnh 3.5 MB qua JPEG2000 (bit) hoặc mode generative (§III-A p.5) | Ghép UAV–user, UAV–LEO, ISL, quỹ đạo UAV, **mode ∈ {1, 2, 3_1, 3_2, 3_3}** cho từng task (p.8) | DSAC (distributional SAC); benchmark MAPPO (p.8–9) | SCE = ω_q(Q−Q_min) + ω_d(D_max−D), với **Q = PSNR** (MS-SSIM ở Fig. 11); trừ phạt nếu vi phạm (14) p.7, (18) p.8 | Ràng buộc trễ từng task, D_max ~ U[2, 11] s (15h, p.8–9) | MAPPO, DSAC & Hovering UAV, bỏ Mode 3', bỏ Mode 1, so từng mode cố định (Fig. 7–11, p.10–11). **Không TH** | Không tìm thấy tên dataset; không có task metric | "2 episode" (Fig. 6, p.10) |
| **Zhao TNSE 2026 (UA-TSC)** | 1 UAV → BS uplink, LoS + AWGN (§III-A–B p.7). Truyền feature JSCC từ DeiT | Độ sâu model i, số symbol K, công suất P; quỹ đạo do Diffuser (p.10–12) | PPO cho tài nguyên (hành động rời rạc i, K + liên tục P); Diffuser offline RL cho quỹ đạo | ω₁·accuracy − ω₂·năng lượng (+ trajectory reward); accuracy = **đường cong fit f(i, K, SNR)** đo offline (§III-C p.8; (32) p.12) | Ràng buộc τ ≤ τ_max, phạt khi vi phạm (19c) p.10; (32) p.12 | LI, RIO; SAC, TD3; IQL, DP cho quỹ đạo (p.14–18). **Không TH** | AID, phân loại 30 lớp (p.14); accuracy là bảng tra, không chạy model online | Không tìm thấy |
| **JOSC (Zhou et al., JWCN 2026)** | K UAV gửi **khung video** về GCS; Shannon, băng thông B_i riêng mỗi UAV, chung tổng công suất (p.4; (14) p.8; (21e) p.10) | Gửi/không gửi frame, **tỉ lệ nén ρ** (importance map), công suất (24) p.11 | PPO kiểu CTDE (MAPPO-like) (p.10–11) | Tối thiểu tổng AoIS; similarity = **MS-SSIM** trên feature (p.5; (21) p.9) | Xác suất τ ≤ τ_max ≥ Pr_min (21c), p.8–10 | Beta-HAPPO, SSC (không nén), FCR (ρ cố định 0.8/0.7/0.5), exhaustive search (p.13). FCR là sweep tham số, **không phải TH** | OpenVideo crop 128×128 (p.13); không có task | Min–max trên 5 lần chạy, chỉ ở Fig. 4 (p.13) |
| **CCSRA (Zheng et al., JWCN 2026)** | U = 4 UAV → edge server uplink, OFDM 16 subchannel, Rician (p.20). Truyền **token ngữ nghĩa theo đối tượng** từ YOLO (p.6–9) | Gán subchannel, công suất, chọn token, **bit lượng tử {2, 4, 6, 8}** (p.15–16, p.21) | Hierarchical hybrid SAC "HH-SAC": tầng trên là vô tuyến, tầng dưới là nội dung (p.16, p.18) | λ₁·CCSE + λ₂·**U_t (mAP/recall so với GT)** − λ₃·redundancy − λ₄·fairness − λ₅·công suất ((33) p.14; (49) p.18) | Ràng buộc payload ≤ budget bit mỗi slot, "single-shot, không retransmission" ((22) p.11; Alg. p.19). Không có hàng đợi | HARL-SemCom, ARA-DRL, Lyapunov-SCA, cùng detector (p.21). **Không TH** | **VisDrone2019-DET + YOLOv8-n, mAP@50** đo bằng detector (p.20–22) | Không tìm thấy; chỉ có sensitivity theo trọng số (Table 1 p.22) |
| **PASC (Yang et al., WCL 2026)** | N = 3 UAV, 10 subchannel, 1.5 MHz, Jakes fading (p.4). Truyền patch MobileViT | Ngưỡng attention (mức nén), subchannel, băng thông, công suất (p.3) | PPO + GAE, state có kênh dự báo bằng GRU (p.3) | QoE = utility sigmoid × trễ; **utility = 0 nếu timeout** (p.3–4) | **Deadline cứng** T_max = 150 ms (p.4) | Các scheme "reactive" không dự báo, ablation bỏ GRU / bỏ ngưỡng thích nghi (p.4). **Không TH** | CIFAR-100, phân loại; utility là mô hình sigmoid (p.4) | Không tìm thấy |
| **Hybrid-action RL (Si, Zhao, Lam, Yang, GLOBECOM 2023)** | 1 UAV thu dữ liệu từ N user qua M kênh NOMA, Rician (p.2–3) | Kênh (rời rạc); scale model ρ, công suất, quỹ đạo (liên tục) (p.3–4) | 2 agent PPO: một rời rạc, một liên tục (p.3–4) | Tối thiểu thời gian nhiệm vụ + utility = trọng số giữa chất lượng tái tạo **Q(ρ) fit từ [11]** và năng lượng tính toán (p.2–3) | Giới hạn thời gian nhiệm vụ T_max (p.4) | EP (công suất đều), Triple PPO (p.5). **Không TH** | Chỉ minh họa trên CIFAR-10; Q là hàm fit | Không tìm thấy |
| **Cross-modal HDRL (Wei & Lang, JWCN 2026, in press)** | 1 UAV → trạm mặt đất, Gauss–Markov block fading (p.24). Gói ngữ nghĩa đa modality | **Mode ∈ 6 tổ hợp {RGB, TIR, DEP}** × độ hạt {4, 8, 12 token}; công suất (p.7, p.24) | Recurrent PPO (mode) + SAC (công suất), primal–dual cho ràng buộc dài hạn (p.19–22) | U_t = u_t·φ(m, g, q): φ là **MLP dự đoán task success**, học từ tập validation offline ((5)–(6) p.9) | Trễ từng slot D_max = 40 ms + ngân sách trễ/năng lượng trung bình (p.8, p.24) | Flat-PPO, DQN-SAC, **PPDO**: exhaustive search từng slot có biến dual (p.26–27). PPDO là baseline mô hình cận thị, gần "heuristic mạnh" nhất, nhưng không tune | Chất lượng modality là **chuỗi Markov tổng hợp** (p.24); không tìm thấy tên dataset | **10 seed, 500 episode test, ± 1 std** (p.23, p.28) |
| **Semantic-Driven Multi-UAV (Wen et al., WCL 2026)** | 3 UAV CoMP downlink → 3 user, 20 MHz (p.4). Truyền câu văn (QoE kiểu DeepSC) | **Tỉ lệ nén ngữ nghĩa rời rạc** (PPO); vị trí UAV + beamforming (AO: SOCP + gradient projection) (p.3–4) | HALO = PPO + alternating optimization | Tổng QoE từ semantic similarity của text (p.3) | Không tìm thấy | Fixed-SCR, RateMax, Static-UAVs, DQN-GPM, … (p.4–5). **Không TH** | Text; tổng hợp | Trung bình 500 realization (p.5) |
| **Multi-Modal UAV (Hu et al., TCOM 2025)** (sub-agent) | 2 UAV, mỗi UAV 5 user; UAV–UAV relay; FDMA (p.4–8) | **K mỗi link, quyết định luôn modality** (text hay ảnh) và mức nén; công suất, băng thông, bay (p.8–10) | Hybrid DDPG + DQN (kiểu P-DQN) | q·QoE − c·cost; độ chính xác SSIM/BLEU qua đường fit (p.5–10) | Ràng buộc D_th, chỉ phạt mềm qua sigmoid (p.9) | MADDPG, ISC, MADDPG-TC (JPEG/BPG+LDPC là **scheme riêng**), EA-MSC (p.12). **Không TH** | COCO-captions để huấn luyện transceiver; môi trường dùng đường fit | Chỉ có dải dao động reward (Fig. 6) |
| **CCM (Li et al., arXiv 2026-09)** (sub-agent) | 3 UAV, 10 terminal, uplink có nhiễu xuyên kênh (p.4–5) | Quỹ đạo, offload, migration, chia băng thông/CPU | HAN-MATD3 | Latency + năng lượng − semantic similarity (đường fit ABG) | Chỉ là hạng latency trong objective | MADDPG, MASAC, MATD3, NSGA-II. **Không TH** | Tổng hợp | Không có |

### 2.2. Nhóm collaborative perception (V2X / multi-drone; truyền feature BEV)

| Bài | Setting / nội dung | Quyết định | Optimizer | Objective | DL | Baseline (TH?) | Dữ liệu | Unc. |
|---|---|---|---|---|---|---|---|---|
| **Where2comm** (NeurIPS 2022) | N xe/drone; budget byte lý tưởng, không có mô hình kênh, không nén thêm (p.7). Truyền BEV feature thưa | Ô BEV top-k theo confidence map, ai nói với ai (p.5) | Supervised end-to-end; **chọn không dùng RL** (p.3) | AP@0.5/0.7, 3D detection | Không | Các phương pháp học trước đó. **Không TH** | OPV2V, V2X-Sim, DAIR-V2X, CoPerception-UAVs | Không ("[No]" error bar, p.13) |
| **CoSDH** (2025) | V2V/V2I, 27 Mbps chia thành cap mỗi agent (p.1, p.6). Feature + box (lai intermediate–late) | Mask cung × cầu theo ngưỡng (p.4) | Train theo detection loss; ngưỡng chỉnh tay | AP 3D | Chỉ test robustness 100/200 ms (p.12–13) | Phương pháp trước. **Không TH** | OPV2V, V2X-Sim, DAIR-V2X | Không |
| **How2comm** (NeurIPS 2023) | V2X, trễ cố định τ = 100 ms (PDF p.24) | Mask không gian–kênh (p.4–5) | Train end-to-end (detection + MI + flow) | AP 3D | Bù bất đồng bộ bằng flow; không có deadline | Phương pháp trước. **Không TH** | DAIR-V2X, V2XSet, OPV2V | Không |
| **SComCP** (2025) | V2V point-to-point, Rayleigh/AWGN; BEV feature qua JSCC (p.4–7) | Mask importance (p.6) | 3 giai đoạn: detection loss, MSE, end-to-end (p.7) | AP@0.5/0.7 | Không | Sheng et al., chuỗi số "8-bit + LDPC + QAM" (p.8). **Không TH** | OPV2V | Không |
| **CoDS** (2025) | V2V số: VQ → LDPC → QAM (p.3–4) | Mask không gian; **MCS chọn theo luật SNR** (p.9) | 4 giai đoạn supervised | AP | Không | Codec analog CNN/Transformer, chuỗi số cổ điển. **Không TH** | OPV2V | Không |
| **HyComm** (2025) | V2X, budget byte lý tưởng; **box + điểm thô** (p.3) | Top-b box rồi phần dư dành cho điểm, theo **luật cố định** (p.4) | Không học phần đóng gói | AP@0.3/0.5/0.7 | Không | Phương pháp trước. **Không TH** | DAIR-V2X, OPV2V | Không |
| **HMS-SCP** (2026) | V2X, AWGN/Rayleigh, top-k đa thang qua JSCC (p.4–5) | Top-k theo ρ cố định mỗi lần chạy (p.9, p.13) | 2 giai đoạn supervised (p.8) | AP kỳ vọng | Chỉ đo thời gian suy luận (p.13) | Where2comm, SComCP tái hiện. **Không TH** | OPV2V, DAIR-V2X | Không |
| **Drones Help Drones** (NeurIPS 2024) | 4–5 drone; tỉ lệ truyền lý tưởng (p.4) | Top-K ô theo độ bổ sung thông tin; tỉ lệ cố định 25 % (p.7) | Train end-to-end (seg + flow) | IoU/VPQ cho dự đoán quỹ đạo; mAP 3D (p.10) | Không (ghi là hướng tương lai) | Phương pháp trước. **Không TH** | Air-Co-Pred (CARLA), CoPerception-UAVs | Không |
| **HARQ cooperative perception** (2024) | V2V OFDM, QuaDRiGa (p.8, p.20) | Có retransmit hay không, theo SimCRC (p.10–15) | Supervised; nhãn SimCRC lấy từ chênh lệch perception loss (p.10) | AP + throughput | Khoảng retransmit 1 ms, tối đa 3 lần | CRC bit-level, chuỗi LDPC. **Không TH** | OPV2V | Không |
| **SchedCP (DRL user scheduling)** (2025) | Ego + N collaborator, **phổ chung** (mỗi slot 5 ms chỉ một link), WINNER+ / Jakes (p.3–4, p.8) | Chọn link nào được kênh; ô feature theo luật top-B (p.5) | **Double DQN** (p.7–9) | Reward **label-dependent**: rate + **ΔL_det**; reward **label-free** từ confidence map (p.6–7) | **Khung 200 ms mỗi frame** (thử thêm 20/50/100 ms) (p.3–4, p.8, p.11) | Nearest, Round-robin, Max-Rate: **luật đơn giản, không tune** (p.9) | V2X-Sim + DiscoNet, AP đo bằng detector | Không |

### 2.3. Nhóm phụ (sub-agent)

| Bài | Ý chính liên quan | Có làm điều A5/A6 làm không? |
|---|---|---|
| **SemanticRL** (Lu et al., 2022) | RL (self-critic REINFORCE) để **huấn luyện codec** text, reward CIDEr/BLEU; có bản mở rộng ảnh MNIST với reward MSE (p.4–8, p.23) | So objective word-level với semantic khi **huấn luyện codec**, không phải khi chọn mode. Không có deadline, không có hệ nhiều tác tử |
| **Data Significance** (Uysal et al.) | Bài định hướng: AoI, VoI, AoII; ý "significance trong ràng buộc thời gian thực" (p.1–5) | Không có hệ thống riêng. Fig. 4(a) trích kết quả policy tối ưu theo AoII khác policy tối ưu theo AoI (p.5), nhưng đó là quyết định lấy mẫu |
| **Attention-RL** (Wang et al., 2022) | BS → 30 user, OFDMA; PPO (APPO) cấp RB; gửi triple tri thức theo thứ tự quan trọng đến hết budget trễ D = 0.1 ms (p.3–9) | Có deadline từng user, có RL, nhưng là text, một BS. Baseline là các DRL khác. Fig. 9: ± std trên 10 lần chạy (p.12) |
| **GOST** (Liao et al., TMC 2026) | Satellite + 12 UAV, MADDPG; reward AoI và reward theo dõi mục tiêu mô phỏng | Không có codec, không có detector thật |

## 3. Ghi chú ngắn từng bài gần nhất

- **Huang JSAC 2025 (gần A5 nhất).**
  - Menu mode của Huang trùng về loại với menu của ta: một mode codec cổ điển (JPEG2000) và các mode generative (BLIP-2 caption → diffusion; caption + VQ latent 3 mức) (§III-A p.5). Ba mode generative "caption → diffusion" của ta về bản chất thuộc họ Mode 2/3 của Huang.
  - Họ chọn mode bằng DRL, có ràng buộc trễ từng task, và có baseline UAV đứng yên (Fig. 9, p.10–11). Baseline này tương đương "hover" của ta.
  - Họ thiếu bốn thứ:
    - (i) codec học được (mbt2018) và nhiều mức chất lượng JPEG2000/AVIF;
    - (ii) dữ liệu ảnh nêu rõ tên;
    - (iii) task metric;
    - (iv) baseline heuristic và báo cáo bất định. Chỉ có MAPPO, ablation và "2 episode".
  - Kênh của họ là downlink LEO → UAV → user, nhiều task trên mỗi slot. Không phải dòng khảo sát uplink A2G có backlog như ta.
- **Zhao TNSE 2026.**
  - PPO chọn rời rạc độ sâu model và số symbol K cho mỗi slot, có phạt khi vi phạm trễ. Reward theo task, nhưng accuracy lấy từ **bảng fit f(i, K, SNR)** (p.8), không chạy detector trên từng ảnh.
  - Chỉ có 1 UAV, bài toán phân loại AID.
  - Baseline chỉ gồm LI/RIO và các DRL khác.
- **JOSC.** Đây là setting "nhiều UAV gửi khung video, quyết định từng frame: gửi/không gửi + tỉ lệ nén, có deadline". Về cấu trúc nó gần A5 hơn Huang. Mục tiêu là AoIS dựa trên MS-SSIM, tức là pixel/feature, không phải task. Baseline tỉ lệ cố định chỉ là sweep tham số. Lựa chọn "không gửi" của họ tương tự cơ chế reuse của ta.
- **CCSRA (gần A6.0-L1 nhất).**
  - Trùng với A6.0-L1 ở: VisDrone, YOLOv8, mAP@50, nhiều UAV, kênh OFDM chung, reward có hạng utility so với ground truth, RL.
  - "Reliable semantic value" của họ là ước lượng giá trị task từ confidence đã hiệu chỉnh + chất lượng định vị + độ mới (p.8). Về chức năng nó cùng vai trò với bộ dự đoán giá trị task từ probe của ta.
  - Khác ta ở bốn điểm:
    - truyền token/feature theo đối tượng chứ không phải ảnh nén;
    - không có deadline theo hàng đợi; payload luôn khớp budget nên **không có frame "không tới"**;
    - không so với PSNR;
    - không có seed/CI.
  - Mô tả phương pháp khá chung chung (ví dụ "Utility(·) có thể là accuracy, mAP, recall…", p.14), nhưng vẫn phải coi là prior art.
- **PASC.** Có deadline cứng 150 ms, utility bằng 0 khi quá hạn, nhiều UAV dùng subchannel chung, PPO. Utility là mô hình sigmoid trên CIFAR-100, không phải detector.
- **xmodal.**
  - PPO chọn "mode" rời rạc (tổ hợp modality) × độ hạt dưới D_max.
  - Utility predictor là MLP học từ hiệu năng task offline. Chức năng này gần bộ dự đoán giá trị task của ta, nhưng dự đoán theo (mode, độ hạt, chất lượng modality) chứ không theo nội dung từng ảnh.
  - Là bài duy nhất báo 10 seed ± std. PPDO là một baseline tối ưu cận thị khá mạnh, và RL chỉ hơn nó 17 %.
- **SchedCP.** Đây là tiền lệ trực tiếp cho hai ý: "reward từ detector, nhãn chỉ dùng khi train, có bản label-free" và "kênh chung + khung thời gian mỗi frame". Tuy vậy, SchedCP chọn link, không chọn codec.

## 4. Trả lời ba câu hỏi

### Câu 1. Yếu tố nào của ta đã có người làm?

| Yếu tố của ta | Đã có? | Bằng chứng |
|---|---|---|
| RL chọn mode/codec rời rạc cho từng ảnh/slot dưới deadline | **Có, nhiều bài** | Huang (mode ∈ {1, 2, 3_1, 3_2, 3_3}, D_max; p.8); Zhao (i, K, phạt τ_max; p.10–12); xmodal (mode × độ hạt, D_max; p.7–8, p.20); JOSC (gửi/không + ρ, τ_max; p.8–11); PASC (ngưỡng nén, T_max; p.3–4); mmuav (K quyết định modality; p.8–10) |
| Menu trộn codec cổ điển và generative trong cùng một tập hành động | **Có** (Huang) | JPEG2000 + text→image + text&embedding (§III-A p.5; §V p.9). Bài khác chỉ để JPEG/BPG làm scheme riêng (mmuav p.12) |
| Reward là giá trị task đo từ đầu ra detector | **Có** | CCSRA: U_t so với GT, YOLOv8 trên VisDrone ((33) p.14, (49) p.18, p.20–21); SchedCP: ΔL_det (p.6–7). Dạng yếu hơn (bảng tra hoặc predictor offline): Zhao (p.8), xmodal (p.9), PASC (p.3–4) |
| Nhãn chỉ dùng trong reward huấn luyện | **Có** | SchedCP: label-dependent vs label-free (p.6–7). CCSRA dùng GT trong U_t |
| So task-aware với PSNR-aware | **Không tìm thấy** | Xem câu 3 |
| Nhiều UAV dùng kênh chung | **Có** | CCSRA (OFDM 16 subchannel, U = 4/6/8; p.20, p.23); PASC (10 subchannel); JOSC (chung tổng công suất); CCM, GOST, mmuav (FDMA hoặc nhiễu). Phổ chung + khung thời gian: SchedCP (V2X) |
| Điều khiển hover/move + năng lượng | **Có ở dạng quỹ đạo** | Huang (quỹ đạo + baseline Hovering, Fig. 9); Zhao (Diffuser); Hybrid-action RL. Dạng "pacing theo backlog" của ta là biến thể; joint của ta lại yếu hơn, nên không nên claim |
| VisDrone + YOLO trong bài UAV SemCom có RL | **Có** | CCSRA (p.20–21) |

### Câu 2. Còn lại gì có thể coi là khác biệt?

Đánh giá có chủ ý khắt khe:

1. **RL so với heuristic mạnh đã tune và heuristic tự hiệu chỉnh (A5, A5.5).** Mức: **trung bình, khác biệt rõ nhất.**
   - Không bài nào trong 24 bài có baseline heuristic được tune trên tập riêng.
   - Baseline mạnh nhất mà họ dùng là bộ tối ưu cận thị theo mô hình: PPDO (xmodal p.26–27), Lyapunov-SCA (CCSRA p.21), exhaustive search (JOSC p.13).
   - Kết quả "RL ≈ 0 ở regime tham chiếu, chỉ thắng ở K = 5 hoặc khi PHY mù MCS" mâu thuẫn với khẩu khí chung của mảng (+5..60 %), nên đáng công bố.
   - Rủi ro: reviewer có thể xếp nó vào "đánh giá cẩn thận hơn", không phải đóng góp mới. Cần đóng khung như một nghiên cứu kiểm chứng ("RL thật sự thắng khi nào"), và nêu rõ cơ chế ở các regime RL thắng.
2. **Bất định 3 tầng với bootstrap theo cụm video có mô phỏng lại (A5.5).** Mức: **yếu đến trung bình.**
   - Không bài nào làm. Mức cao nhất là 10 seed ± std (xmodal).
   - Đây là đóng góp phương pháp, chỉ có giá trị khi đi kèm điểm 1. Không đứng một mình được.
3. **AP50 ở cấp policy, tính cả frame không tới hạn (A6.0-L1).** Mức: **yếu đến trung bình.**
   - CCSRA đo mAP@50 sau khi truyền, nhưng ràng buộc payload ≤ budget làm mọi gói đều "tới". SchedCP vẫn có detection của ego khi thiếu feature. Collaborative perception đo AP ở budget cố định.
   - Quy ước "frame trễ hạn = không có detection, tính vào AP/recall" chưa thấy ở đâu. Nhưng đây là lựa chọn thiết kế metric, dễ bị coi là chi tiết.
4. **Bộ dự đoán giá trị task từ probe payload, dùng cho heuristic task-aware.** Mức: **yếu.**
   - Chức năng tương đương đã có: CCSRA ("reliable semantic value" từ confidence đã hiệu chỉnh, p.8), xmodal (MLP φ(m, g, q), p.9), Zhao (f(i, K, SNR), p.8).
   - Khác biệt duy nhất: dự đoán theo nội dung từng frame, từng mode, từ một probe rẻ, và dùng trong heuristic thay vì trong RL. Chỉ nâng lên mức trung bình nếu chứng minh được (a) độ chính xác dự đoán và (b) heuristic có probe bắt kịp RL task-reward.
5. **Menu trộn codec cổ điển, codec học được và generative.** Mức: **yếu.**
   - Huang đã trộn JPEG2000 với các mode generative.
   - Ta chỉ thêm mbt2018 q1..q8, AVIF/JPEG2000 nhiều mức và thumbnail, và đo trên ảnh thật VisDrone. Huang không nêu tên dataset; tôi không tìm thấy mô tả đo chất lượng cho từng ảnh.
   - Mức tăng thêm có thật nhưng nhỏ. Không nên đặt làm claim chính.
6. **Mode generative bị chấm bằng detector (hallucination có thể sinh FP).** Mức: **trung bình (tiềm năng).**
   - Huang chấm mode generative bằng PSNR/MS-SSIM. Không bài nào chấm mode caption → diffusion bằng detector.
   - Nếu L1 cho thấy reward task loại bỏ mode generative (hoặc giữ chúng ở regime cụ thể) khác hẳn reward PSNR, đây là phát hiện mới có ý nghĩa với chính hướng Huang.
7. **Dòng khảo sát uplink A2G có backlog/pacing, kênh chung, deadline từng frame, MCS mù.** Mức: **yếu.**
   - Từng thành phần đều đã có (JOSC, PASC, CCSRA, SchedCP). Cách ghép của ta là một biến thể hợp lý, không phải điểm mới.

### Câu 3. Đã có bài nào cho thấy reward theo task làm đổi lựa chọn mode so với reward PSNR chưa?

**Chưa tìm thấy.** Các bằng chứng gần nhất, không bằng chứng nào là "task vs PSNR trên cùng menu mode":

- **Huang Fig. 6 (p.10):** tăng trọng số chất lượng (PSNR) so với trễ thì tỉ lệ chọn Mode 1 / Mode 3_3 thay đổi. Đây là thay đổi trọng số pixel–trễ, không phải task.
- **Hybrid-action RL Fig. 5a (p.5):** tăng trọng số chất lượng tái tạo thì scale model lớn hơn.
- **CCSRA Table 1 (p.21–22):** tăng λ₂ (utility task) thì mAP@50 tăng từ 42.8 lên 44.9 % và CCSE giảm. Bảng không cho thấy phân bố hành động, và không có đối chứng PSNR.
- **SchedCP Fig. 12 (p.11–12):** scheduler task-aware chọn link khác Max-Rate. Đây là task so với rate.
- **Semantic-Driven (p.5):** QoE-aware so với RateMax (SCR cố định thấp nhất).
- **SemanticRL Fig. 4–6 và Attention-RL Fig. 8:** đổi objective huấn luyện codec/decoder thì output thay đổi. Đó là quyết định huấn luyện, không phải chọn mode.
- **Data Significance Fig. 4(a) (p.5):** trích kết quả policy AoII khác policy AoI, nhưng cho bài toán lấy mẫu.

Hệ quả: thí nghiệm "cùng hệ thống, cùng menu, chỉ đổi hạng chất lượng PSNR → TP − 0.5·FP, rồi so phân bố mode, recall vật nhỏ, AP50 cấp policy" là khoảng trống thật. Cần lưu ý reviewer có thể cho rằng "đổi reward thì đổi hành vi là điều hiển nhiên". Giá trị nằm ở chỗ đo được đổi **ở đâu** (vật nhỏ, mode generative, regime băng thông thấp) và **bao nhiêu** so với heuristic task-aware rẻ.

## 5. Gợi ý cách viết claim (để không bị bắt lỗi prior art)

- **Không nên claim:**
  - "lần đầu chọn mode generative/cổ điển bằng RL" (đã có Huang);
  - "lần đầu dùng reward theo detector cho UAV SemCom" (đã có CCSRA; SchedCP cho V2X);
  - "lần đầu multi-UAV kênh chung có deadline" (đã có PASC, JOSC, CCSRA).
- **Có thể claim, kèm trích dẫn đối chứng:**
  - (a) so RL với heuristic mạnh đã tune và tự hiệu chỉnh, kèm bất định 3 tầng; kết quả là lợi thế RL nhỏ và phụ thuộc regime (trích Huang, Zhao, xmodal, CCSRA làm ví dụ cho baseline yếu);
  - (b) đối chứng có kiểm soát giữa reward task và reward PSNR trên cùng menu, có mode generative chấm bằng detector;
  - (c) metric AP50 cấp policy tính cả frame trễ hạn.
- **Phải trích và phân biệt trực tiếp:** Huang JSAC 2025, CCSRA (JWCN 2026), JOSC (JWCN 2026), SchedCP, Zhao TNSE 2026, xmodal (JWCN 2026), PASC (WCL 2026).

## 6. Giới hạn của audit

- Chỉ xét các PDF có trong `pdf/`. Không tìm thêm bài ngoài thư viện, nên có thể còn prior art gần hơn, nhất là các bài 2025–2026 về "codec selection RL" hoặc "task-aware video analytics streaming" (dòng AccMPEG / DDS / Reducto bên mạng máy tính, không có trong thư viện).
- Không đọc toàn văn mọi bài. Các bài nhóm 2.2 và 2.3 dựa vào bản trích của sub-agent cộng vài lần kiểm tra lại.
- Một số ký hiệu toán bị mất khi trích văn bản (ví dụ biến ngưỡng trong action của PASC). Ở những chỗ đó tôi mô tả theo lời văn của bài.
