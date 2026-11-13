# Rà soát novelty có mục tiêu: collaborative perception và lập lịch semantic không dây

Ngày 13–14/11/2026, theo phản biện #22. Chỉ đọc tài liệu: không train, không thí nghiệm, không mở ảnh dataset. Chưa đăng ký thí nghiệm mới, chưa viết thuật toán.

Bản này bổ sung cho [swarmbridge-direction-reassessment.md](swarmbridge-direction-reassessment.md) (mục 2 và 5) và các ghi chú [reassess-notes/axis1-collab-perception-u2u.md](reassess-notes/axis1-collab-perception-u2u.md), [reassess-notes/axis2-voi-taskoriented-beliefs.md](reassess-notes/axis2-voi-taskoriented-beliefs.md). Những bài đã phân tích ở đó (EMP, C-MASS, RACooper, BHU, GOSC, PRISM, Mazandarani, Harbor, LGCP, Sun & Pappas) không được đọc lại; tôi chỉ trích kết luận.

## 0. Mức đọc

- **[TOÀN VĂN]**: đọc toàn văn từ bản PDF trong `pdf/`. Gồm CCSRA, HyComm, SchedCP (tôi tự đọc), Where2comm, How2comm, CodeFilling, CoSDH, SComCP, CoDS, DiffCP, CERF (agent đọc, tôi đối chiếu các điểm quyết định).
- **[MỘT PHẦN]**: HMS-SCP, HARQ (phần hệ thống, phương pháp, thiết lập thí nghiệm).
- **[TÓM TẮT]**: chỉ đọc trang tóm tắt arXiv hoặc trang nhà xuất bản. Gồm các bài 2024–2026 ở mục 2.4.
- **[SUY LUẬN]**: nhận định của tôi, không có trong bài.

Danh sách bài 2024–2026 do một lượt tìm kiếm web duy nhất (bị gián đoạn một lần vì giới hạn tốc độ) nên **không đầy đủ**. Các trang IEEE Xplore, ScienceDirect, MDPI phần lớn không đọc được.

## 1. Tóm tắt

- **Phần lớn đề tài "dynamic cooperative semantic scheduling" đã có người làm.** Chọn nội dung theo độ tin cậy dưới ngân sách, tránh trùng lặp giữa nhiều bên gửi, trộn gói gọn (detection) với dữ liệu thô, lập lịch bằng RL theo CSI và giá trị ngữ nghĩa, và cả phiên bản nhiều UAV có xét trùng lặp xuyên UAV, đều đã xuất bản (mục 3).
- **CCSRA không giải quyết hết**, nhưng đã chiếm phần "nhiều UAV + giá trị ngữ nghĩa có hiệu chỉnh + trùng lặp xuyên UAV + OFDMA có nhiễu + RL phân cấp". Thêm PPO / MAPPO vào một hệ multi-UAV SemCom **không** đủ novelty.
- **Hai khoảng trống còn lại sau khi đối chiếu:**
  - **RQ1:** dùng **độ bất định của liên kết xuyên góc** làm tín hiệu để quyết định truyền gì (gửi thêm để phân định "có phải cùng một vật không"), chấm bằng F1 vật thể duy nhất. Không tìm thấy trong nguồn đã kiểm. **HIGH RISK**, đáng một phép thử khả thi rẻ.
  - **RQ2:** lập lịch theo giá trị góc nhìn thay đổi theo thời gian khi **phản hồi từ BS bị trễ**. Phần thời gian và phần học trực tuyến đã có (MASS, PIB, MVSparse); chỉ phần phản hồi trễ là chưa thấy. Là đóng góp thuật toán: **NO-GO**. Là câu hỏi đánh giá trên dữ liệu thật: **HIGH RISK**, phụ thuộc dữ liệu có đảo chiều góc tốt.
- **MARL:** hiện không có lý do khoa học. Mọi bằng chứng (của dự án và của tài liệu) ủng hộ một bộ điều phối tập trung ở BS với quy tắc đơn giản làm baseline bắt buộc (mục 8).

## 2. Các bài đã rà soát

### 2.1. Ba bài đe doạ trực tiếp nhất (đọc sâu)

**CCSRA**: Zheng, Li, Zuo, "Confidence-calibrated semantic resource allocation for multi-UAV low-altitude networks", *EURASIP J. Wireless Commun. Netw.* 2026:112, DOI [10.1186/s13638-026-02655-0](https://doi.org/10.1186/s13638-026-02655-0). [TOÀN VĂN]

- **Hệ thống:** U UAV (mặc định 4; thử 6 và 8) gửi lên một edge server. Edge là bộ điều khiển tập trung. Không phải MARL: một tác tử với chính sách phân cấp (HH-SAC; tầng trên chọn kênh con và công suất, tầng dưới chọn token và độ sâu lượng tử).
- **Giá trị ngữ nghĩa** (mục 3.2, phương trình 12): w(lớp) × độ tin cậy đã hiệu chỉnh × chất lượng định vị × (1 − độ trùng). Hiệu chỉnh bằng temperature scaling có đầu bất định, khớp trên tập val của VisDrone.
- **Trùng lặp xuyên UAV** (mục 3.5): cosine lớn nhất giữa token và một **memory bank** ở edge (khởi tạo K-means, cập nhật EMA). Tác giả cố ý không dùng box: "2D bounding-box IoU is not directly comparable across UAVs unless accurate cross-view geometric registration is available". Vậy **không có liên kết vật thể xuyên góc**.
- **Kênh:** OFDMA dùng lại kênh, 16 × 180 kHz, SINR có nhiễu đồng kênh, Rician 6 dB, shadowing 4 dB; "single-shot transmission regime without retransmission". Không có hạn chót ngoài độ dài slot, không có trễ phản hồi; chi phí cập nhật memory bank không được tính.
- **Dữ liệu:** ảnh đơn VisDrone2019-DET; "inter-UAV overlap ratio is controlled by adjusting the relative deployment positions" → **chồng lấn tổng hợp**, không có ảnh nhiều góc thật, không có ID xuyên góc. Cách tạo kết quả phát hiện toàn cục từ token không đủ chi tiết để tái lập.
- **Chỉ số:** mAP@50 (44,1% mặc định), CCSE (giá trị ngữ nghĩa trên chi phí), tỉ lệ trùng. Không có F1 vật thể duy nhất.
- **Baseline:** hai RL (HARL-SemCom, ARA-DRL) và Lyapunov-SCA. Không có quy tắc đơn giản không học, không có oracle, không có mốc "không hợp tác". Không báo seed hay khoảng tin cậy.
- **Giới hạn tác giả tự nêu:** chỉ mô phỏng; "assumes centralized scheduling"; hướng sau: "decentralized and temporally aware semantic communication".
- **Hệ quả cho SwarmBridge [SUY LUẬN]:** CCSRA chiếm khung bài toán. SwarmBridge chỉ còn chỗ ở những gì CCSRA bỏ qua: ID xuyên góc thật và liên kết bất định, chỉ số vật thể duy nhất, động học theo thời gian, phản hồi trễ, codec đo thật, và baseline đơn giản (mà B1 cho thấy là rất mạnh).

**HyComm**: Hu, Peng, Yang, Chen, "Communication-Efficient Multi-Agent 3D Detection via Hybrid Collaboration", arXiv [2508.07092](https://arxiv.org/abs/2508.07092) (2025). [TOÀN VĂN]

- **Hệ thống:** N agent gửi cho một ego; LiDAR, phát hiện 3D; DAIR-V2X (thật) và OPV2V (mô phỏng). Không RL.
- **Gói tin:** "box message" (detection, chọn theo độ tin cậy dưới ngân sách) cộng "point message" (điểm thô trong box mở rộng của các detection **bất định**, ưu tiên theo độ bất định). Đây chính là ý TOKEN + bổ sung dữ liệu thô có chọn lọc, ở mức vật thể.
- **Chọn theo thông tin của chính bên gửi**, không theo những gì bên nhận đã có.
- **Kênh:** không có; dung lượng tính bằng "byte in the log scale with base 2", không nén thêm.
- **Sai lệch pose:** xử lý ở bước hợp nhất (dùng box làm mốc, theo CoAlign), không dùng để quyết định truyền.
- Một khung hình, không có thời gian, không phản hồi. Mã "will be released" (chưa thấy).
- **Hệ quả [SUY LUẬN]:** lựa chọn TOKEN / CROP / FRAME của SwarmBridge là phiên bản ảnh 2D của HyComm. Không thể coi đó là đóng góp.

**SchedCP**: Liu, Liu, Liang, Ye, Guo, Jin, "Deep Reinforcement Learning-Based User Scheduling for Collaborative Perception", arXiv [2502.10456](https://arxiv.org/abs/2502.10456) (2025). Người phản biện ghi IEEE TMC 2025; tôi chưa xác minh được venue. [TOÀN VĂN]

- **Hệ thống:** ngã tư, ego làm bộ lập lịch, 4 bên cộng tác (xe hoặc RSU); đặc trưng BEV LiDAR (DiscoNet); V2X-Sim (CARLA). Mỗi slot chỉ một liên kết được phát; fading nhỏ thay đổi giữa các slot con.
- **Trạng thái bên nhận:** mỗi bên cộng tác gửi top-K ô theo bản đồ tin cậy của mình so với bản đồ **hiện tại** của ego. Ego tạo lại bản đồ sau mỗi slot từ đặc trưng đã hợp nhất và che các ô đã gửi (phương trình 11–13). Đây là vòng lặp "bên nhận đã biết gì" trong một khung hình.
- **DDQN** với trạng thái = CSI + bản đồ tin cậy; thưởng có nhãn (giảm loss phát hiện) và **không nhãn** (tiện ích từ bản đồ tin cậy). Chi phí gửi bản đồ tin cậy "negligible", không tính.
- **Baseline:** Nearest, Max Rate, Round Robin. Ví dụ +2,9 / +4,0 AP so với Max Rate ở 300 kHz.
- **Thời gian:** chỉ là fading trong khung và cập nhật trạng thái bên nhận. Không học qua nhiều khung xem bên nào có góc tốt hơn; không có bất định liên kết (pose được phát và coi là chính xác).
- **Hệ quả [SUY LUẬN]:** "RL + CSI + giá trị ngữ nghĩa + mục tiêu không nhãn + lập lịch theo trạng thái bên nhận" là prior art.

### 2.2. Nhóm hiệu quả truyền thông trong collaborative perception

| Bài | Nguồn | Điểm chính | Mức đọc |
|---|---|---|---|
| Where2comm | NeurIPS 2022, arXiv [2209.12836](https://arxiv.org/abs/2209.12836) | Bản đồ tin cậy theo không gian; bản đồ yêu cầu R = 1 − C của bên nhận; chọn top-k dưới ngân sách; có bộ **CoPerception-UAVs** (AirSim + CARLA, 5 UAV, đội hình động). Không kênh, không thời gian. Mã: github.com/MediaBrain-SJTU/where2comm | TOÀN VĂN |
| How2comm | NeurIPS 2023 ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2023)); arXiv chưa xác minh | Truy vấn yêu cầu từ ego; **bù trễ** bằng dự đoán dòng đặc trưng (trễ 100 ms, quét 0–400 ms). Không quyết định truyền theo thời gian | TOÀN VĂN |
| CodeFilling | CVPR 2024, tr. 15481–15490 ([CVF open access](https://openaccess.thecvf.com/CVPR2024)) | Gần câu hỏi của ta nhất trong nhóm này: chia một ngân sách chung cho **nhiều** bên gửi để không "đổ đầy" quá nhu cầu bên nhận (tránh trùng lặp từ dự đoán tin cậy của mọi bên). Chi phí phát bản đồ điểm không tính. Mã: github.com/PhyllisH/CodeFilling | TOÀN VĂN |
| CoSDH | CVPR 2025, arXiv [2503.03430](https://arxiv.org/abs/2503.03430) | Cung–cầu (cầu từ mật độ điểm LiDAR của ego, cung từ độ tin cậy bên gửi); **lai đặc trưng trung gian + box**; tốc độ theo DSRC 27 Mbps | TOÀN VĂN |

### 2.3. Nhóm semantic communication cho collaborative perception

| Bài | Nguồn | Điểm chính | Mức đọc |
|---|---|---|---|
| SComCP | arXiv [2507.00895](https://arxiv.org/abs/2507.00895); TRID ghi IEEE TVT 2026 (chưa xác minh DOI) | Một cặp bên gửi–ego; chọn đặc trưng theo mặt nạ quan trọng của chính bên gửi; JSCC analog; AWGN / Rayleigh. Không nhiều agent, không thời gian | TOÀN VĂN |
| CoDS | arXiv [2512.22513](https://arxiv.org/abs/2512.22513) | Một cặp; token chỉ số codebook VQ trên LDPC + QAM, AMC theo SNR; bên nhận bỏ token giải mã hỏng | TOÀN VĂN |
| HMS-SCP | arXiv [2608.14603](https://arxiv.org/abs/2608.14603) | N agent, ngân sách chung B, đa tỉ lệ; tỉ lệ chọn cố định mỗi lần chạy, không phân bổ giữa các agent | MỘT PHẦN |
| HARQ | arXiv [2409.09042](https://arxiv.org/abs/2409.09042); CoDS trích là IEEE TCCN 2025 | Một cặp; ACK ngữ nghĩa (so bản đồ tin cậy trước và sau kênh), truyền lại sau 1 ms, tối đa 3 lần. Phản hồi là về **độ tin cậy kênh**, không về nội dung bổ sung | MỘT PHẦN |
| DiffCP | arXiv [2409.19592](https://arxiv.org/abs/2409.19592) | Chỉ gửi vector ngữ nghĩa; ego tái tạo BEV của bên gửi bằng diffusion có điều kiện theo BEV của chính nó, tức "chỉ gửi phần khác biệt giữa hai góc". Không kênh | TOÀN VĂN |
| CERF | arXiv [2609.00951](https://arxiv.org/abs/2609.00951) | **UAV** (UAV3D, Air-Co-Pred); chỉ gửi box + pose; bù trễ bằng Kalman; NMS xuyên agent. Không chọn lọc, không kênh. Mã: github.com/uestchjw/CERF | TOÀN VĂN |

### 2.4. Bài 2024–2026 tìm thêm [TÓM TẮT]

| Bài | Nguồn | Liên quan |
|---|---|---|
| PIB (Prioritized Information Bottleneck, edge video analytics) | IEEE/ACM ToN 2025, DOI [10.1109/TON.2025.3526148](https://doi.org/10.1109/TON.2025.3526148), arXiv [2409.00146](https://arxiv.org/abs/2409.00146) | **Nguy hiểm nhất ngoài CCSRA.** Nhiều camera chồng lấn ưu tiên gửi theo SNR và độ phủ vùng quan tâm để giảm trùng lặp; học trực tuyến có cận regret trên kênh dao động |
| MVSparse | arXiv [2312.15858](https://arxiv.org/abs/2312.15858) | RL trên từng camera chọn khối ảnh theo lịch sử **và phản hồi detection của camera láng giềng** từ edge. Mặt phẳng đất hiệu chuẩn tĩnh (liên kết đã biết); mục tiêu là tính toán, không phải kênh |
| MASS | arXiv [2302.13029](https://arxiv.org/abs/2302.13029) | Bandit "restless" chọn xe nào được lập lịch khi giá trị chưa biết và thay đổi; phản hồi tức thời |
| TAMP | arXiv [2601.04542](https://arxiv.org/abs/2601.04542) | BS lập lịch nhiều vùng bằng Lyapunov, theo AoI và dung lượng |
| Fresh2comm | arXiv [2502.07852](https://arxiv.org/abs/2502.07852) | AoI để cấp tài nguyên dưới trễ thay đổi |
| JigsawComm | arXiv [2511.17843](https://arxiv.org/abs/2511.17843) | Ước lượng tiện ích từng ô của từng agent, trao đổi bản đồ tiện ích, "eliminates cross-agent redundancy" dưới ngân sách |
| RDcomm | arXiv [2509.21994](https://arxiv.org/abs/2509.21994) | Lý thuyết rate–distortion thực dụng: gửi thông tin liên quan nhiệm vụ và không trùng |
| WhisperNet | CVPR 2026, arXiv [2603.01708](https://arxiv.org/abs/2603.01708) | Bên nhận lập kế hoạch yêu cầu toàn cục từ metadata độ nổi bật của các bên gửi |
| CooperTrim | ICLR 2026, arXiv [2602.13287](https://arxiv.org/abs/2602.13287) | Chọn dữ liệu chia sẻ theo bất định thời gian so với bộ nhớ của ego |
| CoRA | AAAI 2026, arXiv [2512.13191](https://arxiv.org/abs/2512.13191) | Nhánh vật thể sửa lệch do sai pose khi hợp nhất |
| STAC | arXiv [2401.15288](https://arxiv.org/abs/2401.15288) | Dùng liên kết xuyên camera để lọc khung trùng dưới băng thông hạn chế (camera cố định) |
| Polly | INFOCOM 2023, DOI [10.1109/INFOCOM53939.2023.10229045](https://doi.org/10.1109/INFOCOM53939.2023.10229045) | Tự phát hiện vùng nhìn chồng lấn và chia sẻ kết quả để không xử lý trùng (tính toán, không phải kênh) |

Bộ dữ liệu tìm được: **MDMOT** (trong bài FusionTrack, arXiv [2505.18727](https://arxiv.org/abs/2505.18727)): video nhiều drone thật; trang phụ ghi 5 drone, 20 chuỗi đồng bộ, ID xuyên góc gán tay, nhưng tóm tắt arXiv không nêu và **không có giấy phép công bố**. **MATRIX** (arXiv [2511.08615](https://arxiv.org/abs/2511.08615)): tổng hợp (Unreal / AirSim), README ghi CC BY-SA 4.0, không nói ID xuyên góc.

## 3. Ma trận so sánh

Y = có; p = một phần; – = không thấy trong phần đã đọc.

| Bài | Bên nhận | Quyết định | RL | Cảm biến / góc nhìn | Dữ liệu thật nhiều góc | Gói tin | Liên kết xuyên góc | Bổ sung / trùng lặp | Kênh | Trễ / hạn chót | Thời gian | Chỉ số | Đánh giá |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Where2comm | ngang hàng | phân tán | – | camera / LiDAR BEV; UAV mô phỏng động | p (DAIR) | đặc trưng | pose biết; nhiễu Gauss | đo qua bản đồ tin cậy bên kia | – (byte) | – | – | AP từng agent | detector thật; chia khung ngẫu nhiên |
| How2comm | ego | ego yêu cầu | – | LiDAR | p | đặc trưng | pose biết; nhiễu | p (tách vùng riêng / chung khi hợp nhất) | – | **bù trễ** | p (dự đoán) | AP ego | detector thật |
| CodeFilling | ngang hàng | phân tán, 2 vòng | – | LiDAR / camera | p | chỉ số codebook | pose biết | **Y** (ngân sách chung, nhu cầu bão hoà) | – | quét | – | AP | detector thật |
| CoSDH | ego | 2 vòng | – | LiDAR | p | **đặc trưng + box** | pose biết | p (cung–cầu) | p (tốc độ DSRC) | quét | – | AP ego | detector thật |
| HyComm | ego | bên gửi | – | LiDAR | p | **box + điểm thô** | pose có sai (CoAlign) | p (bổ sung theo bất định riêng) | – | – | – | AP | detector thật |
| SchedCP | ego | **tập trung** | **DDQN** | LiDAR, CARLA | – | đặc trưng | pose biết | **Y** (trạng thái bên nhận) | **Y** (fading, CSI) | p | trong khung | AP ego | mô phỏng |
| CCSRA | **edge** | **tập trung** | **HRL (SAC)** | ảnh 2D, **multi-UAV** | – (chồng lấn tổng hợp) | token + lượng tử | **không có** (embedding) | Y (memory bank) | **Y** (OFDMA, nhiễu, Rician) | – (một lần) | – | mAP, CCSE | mô phỏng, không CI |
| SComCP / CoDS / HARQ | ego | bên gửi | – | LiDAR | – | đặc trưng / token VQ | pose biết | – | Y (AWGN / Rayleigh; LDPC; HARQ) | p (HARQ) | – | AP | mô phỏng |
| DiffCP | ego | – | – | LiDAR | – | vector ngữ nghĩa | vị trí biết | p (khác biệt so với tiên nghiệm bên nhận) | – | – | – | AP | mô phỏng |
| CERF | ego | – | – | camera BEV, **UAV** | – | box + pose | pose biết | NMS khi hợp nhất | – | bù trễ | – | mAP ego | detector thật |
| PIB | edge | **học trực tuyến** | bandit-like | camera cố định | p | đặc trưng ưu tiên | hiệu chuẩn | Y (độ phủ) | **Y** | p | **Y** | MODA | (tóm tắt) |
| MVSparse | edge | **RL từng camera** | Y | camera cố định | p | khối ảnh | mặt phẳng đất biết | Y (phản hồi láng giềng) | – | p | **Y** | tracking | (tóm tắt) |
| MASS / C-MASS | RSU / ego | tập trung | bandit / greedy | LiDAR | p (V2V4Real) | đặc trưng | pose biết | **Y** (cặp không submodular) | p | – | **Y** (bandit) | recall | mô hình thống kê |
| **SwarmBridge hiện có** | **BS không cảm nhận** | tập trung | (A5: PPO) | **ảnh 2D, 2 UAV thật (MDMT)** | **Y** | **TOKEN / CROP / FRAME** | **ID GT; sàng lọc 8/11** | đo bằng detection thật | p (A5: MCS, nhiễu; pilot: ngân sách) | p (pilot: CPU, AVIF đo) | – (tĩnh trên MDMT) | **F1 vật thể duy nhất** | **detector + codec đo thật, CI theo chuỗi** |

Dòng cuối là những gì đã đo, không phải đóng góp. Các ô in đậm ở dòng đó là khác biệt về **thiết lập đánh giá**, không phải cơ chế.

Baseline cần tái lập nếu làm tiếp (mức tối thiểu):
1. **Góc cố định / NDET** (B1 cho thấy bắt kịp oracle GT trên MDMT).
2. **Top-k theo độ tin cậy** kiểu Where2comm, áp cho detection.
3. **Lai TOKEN + bổ sung theo bất định** kiểu HyComm.
4. **Phân bổ ngân sách chung tránh trùng** kiểu CodeFilling, có **tính chi phí phát bản đồ**.
5. **Trạng thái bên nhận + Max Rate / Round Robin** kiểu SchedCP.
6. **Độ mới so với memory bank** kiểu CCSRA (embedding, không liên kết).
7. **Bandit chọn bên gửi** kiểu MASS cho câu hỏi thời gian.
8. **Chỉ gửi token + bù trễ** kiểu CERF (mã công khai).
9. **Oracle GT có đánh dấu.**

## 4. Câu A: điều gì đã bị prior art giải quyết?

Trả lời thẳng: phần lớn.

| Thành phần của "dynamic cooperative semantic scheduling" | Đã có | Nguồn |
|---|---|---|
| Chọn nội dung gửi theo độ quan trọng / độ tin cậy dưới ngân sách | **Có** | Where2comm, CoSDH, HyComm, CCSRA, SComCP |
| Trộn gói gọn (detection) với dữ liệu thô / đặc trưng tuỳ ngân sách | **Có** | HyComm (box + điểm thô), CoSDH (đặc trưng + box) |
| Tránh trùng lặp giữa nhiều bên gửi theo những gì bên nhận đã có | **Có** | CodeFilling, SchedCP / RACooper, JigsawComm, WhisperNet; ETSI CPM |
| Phụ thuộc lẫn nhau giữa các bên gửi (không submodular) | **Có** | C-MASS |
| Lập lịch bằng RL theo CSI + ngữ nghĩa, kể cả mục tiêu không nhãn | **Có** | SchedCP, RACooper, CCSRA |
| Nhiều UAV + trùng lặp xuyên UAV + kênh có nhiễu + RL | **Có** (chồng lấn tổng hợp) | CCSRA |
| Học trực tuyến khi giá trị bên gửi thay đổi và chưa biết | **Có** | MASS (bandit), PIB (online learning), MVSparse (RL), CooperTrim |
| Tính tươi của thông tin / trễ | **Có** | TAMP, Fresh2comm (AoI); How2comm, CERF (bù trễ khi hợp nhất) |
| Phản hồi ngữ nghĩa từ bên nhận có trễ | **Một phần** | HARQ (về độ tin cậy kênh, không về nội dung) |
| Chọn góc tốt hơn giữa các camera chồng lấn | **Có** (camera cố định) | PIB, Polly, STAC; và B1 của chính ta: quy tắc NDET đã đủ trên MDMT |

Hệ quả: một đề xuất "PPO / MAPPO chọn TOKEN / CROP / FRAME cho từng UAV theo giá trị ngữ nghĩa và CSI" sẽ bị trích ngay HyComm + SchedCP + CCSRA. Đề xuất đó nên bỏ.

## 5. Câu B và C: tối đa hai câu hỏi còn khoảng trống

### RQ1. Liên kết xuyên góc bất định có phải là một lý do để truyền không?

**Câu hỏi:** khi BS không chắc hai detection từ hai UAV có phải cùng một vật không (không có pose chính xác, không có ID), lỗi đếm vật thể duy nhất (đếm trùng, gộp nhầm) tăng lên. Gửi thêm thông tin có chọn lọc (crop, đặc trưng ngoại hình, hoặc FRAME của một góc) có thể phân định các cặp mơ hồ đó. Câu hỏi là: phần lỗi do liên kết có đủ lớn không, và gửi "để phân định" có đáng giá hơn gửi "để tìm vật mới" ở cùng ngân sách không?

**Khoảng trống:**
- Mọi bài đã đọc hoặc giả định pose biết (Where2comm, How2comm, CodeFilling, CoSDH, SchedCP, CERF, SComCP), hoặc coi sai pose là vấn đề **độ bền khi hợp nhất** (HyComm, CoRA, CooperTrim), hoặc né liên kết hẳn (CCSRA dùng embedding).
- STAC dùng liên kết để lọc trùng nhưng với camera cố định đã hiệu chuẩn và không học quyết định.
- Không tìm thấy bài nào dùng **độ bất định của liên kết** làm tín hiệu quyết định **truyền gì**. Kết luận này dựa trên nguồn đã kiểm, không phải tìm kiếm hệ thống.

**Loại novelty:** mô hình hoá (giá trị của một gói tin gồm cả phần "giảm bất định liên kết") và thuật toán (quy tắc phân bổ dựa trên đó). Thêm một phần đánh giá: F1 vật thể duy nhất với FP trùng và FP liên kết **khác 0**, thay vì bằng 0 theo cấu trúc như B1.

**Rủi ro:**
- Trên MDMT, homography từ nhãn khớp tốt (8/11 chuỗi có inlier trung vị ≥ 0,70). Nếu một homography ước lượng từ token đã phân định gần hết, phần lỗi do liên kết sẽ nhỏ và RQ1 không có dư địa.
- Dữ liệu: cần nhiều góc thật có ID xuyên góc.

**Phán quyết: HIGH RISK**, đáng làm phép thử khả thi ở mục 7 trước bất kỳ thiết kế nào.

### RQ2. Lập lịch theo giá trị góc nhìn thay đổi theo thời gian khi phản hồi từ BS bị trễ

**Câu hỏi:** khi góc tốt hơn đổi chiều trong một chuyến bay, và BS chỉ biết giá trị thật của một góc sau khi nhận và xử lý (có trễ), quy tắc lập lịch nào theo kịp? Phản hồi trễ làm mất bao nhiêu so với oracle?

**Đã có:** chọn bên gửi khi giá trị thay đổi và chưa biết (MASS bandit, PIB học trực tuyến, MVSparse RL với phản hồi láng giềng, CooperTrim). Bandit có phản hồi trễ là một chủ đề chuẩn trong lý thuyết học trực tuyến [SUY LUẬN, không rà soát ở đây].

**Khoảng trống còn lại:** trễ phản hồi **được mô hình hoá trong quyết định** (không chỉ quét độ bền sau đó) cho perception nhiều UAV với dữ liệu thật có đảo chiều. Không thấy trong nguồn đã kiểm (MVSparse, WhisperNet giả định phản hồi tức thời).

**Loại novelty:** chủ yếu đánh giá và mô hình hoá; thuật toán gần như chắc là một biến thể bandit có trễ đã biết.

**Bằng chứng riêng của dự án đi ngược:**
- B1: trên MDMT, giá trị góc tĩnh; quy tắc đơn giản bắt kịp oracle.
- A7-S1 / A7-S2: dư địa là kiến thức cấp cảnh; phản hồi proxy có trễ của BS (NO-GO) không giữ được dư địa.

**Phán quyết:**
- Là đóng góp thuật toán (RL / bandit mới): **NO-GO**.
- Là câu hỏi đánh giá trên dữ liệu có đảo chiều: **HIGH RISK**, chỉ mở nếu dữ liệu mới qua được phép thử ở mục 7.

## 6. Câu D: dữ liệu

| Bộ dữ liệu | Quyền dùng | ID xuyên góc | Dùng cho |
|---|---|---|---|
| MDMT | xác nhận miệng của giáo sư (do người dùng thuật lại); tác giả chưa trả lời | 8/11 chuỗi qua sàng lọc | RQ1 (phép thử khả thi ngay trên bản ghi có sẵn). Không hợp cho RQ2 (góc tĩnh) |
| CrossTrack-UAV | thỏa thuận học thuật có chữ ký | có theo mô tả, chưa kiểm | RQ1 và RQ2; là nguồn duy nhất có khả năng có đảo chiều góc tốt (UAV cánh cố định di chuyển) |
| M3OT | CC BY 4.0 (đã xác minh) | 3/9 chuỗi qua sàng lọc, đều thuộc train | RQ1 chỉ ở mức thử phụ (vật 10–20 điểm ảnh, một lớp) |
| MDMOT | không công bố giấy phép | có theo trang phụ, chưa kiểm | phải xin tác giả |
| CoPerception-UAVs (Where2comm), UAV3D, Air-Co-Pred, MATRIX | **mô phỏng**; giấy phép chưa xác minh ở đây (trừ MATRIX: CC BY-SA 4.0 theo README) | có (mô phỏng) | thử cơ chế rẻ cho RQ2, không thay được dữ liệu thật |

Không tìm thấy bộ dữ liệu UAV thật nào vừa có giấy phép rõ, vừa có ID xuyên góc đủ dùng, vừa đã kiểm.

## 7. Câu E: phép thử khả thi rẻ, GO / NO-GO (đề xuất, chưa đăng ký, chưa chạy)

**RQ1, phép thử "ngân sách lỗi do liên kết"** (offline, trên bản ghi pilot 0B của 8 chuỗi MDMT, nhãn chỉ dùng để chấm):
1. Thay liên kết lý tưởng bằng liên kết không nhãn: homography ước lượng từ token của hai góc (RANSAC) cộng cùng lớp. Chấm F1 vật thể duy nhất với FP trùng và FP liên kết thật.
2. ΔA = F1 (liên kết lý tưởng) − F1 (liên kết không nhãn), ở mức TOKEN và ở mức một FRAME đúng góc.
3. Oracle phân định có đánh dấu: chỉ các cặp mơ hồ (sai số chiếu lại trong một dải định trước) được "gửi thêm" một crop; đo F1 hồi phục và bit tốn thêm.
- **GO** nếu ΔA ≥ 3 điểm F1 **và** oracle phân định hồi phục ≥ 50% ΔA với ≤ 10% số bit của một FRAME.
- **NO-GO** nếu ΔA < 1 điểm (homography đã đủ), hoặc oracle không hồi phục được.
- Ngưỡng là đề xuất, phải được duyệt và khoá trước khi chạy.

**RQ2, phép thử "có đảo chiều không"** (chỉ nhãn, trên dữ liệu mới, trước khi chạy detector):
1. Với mỗi cặp góc, chia chuyến bay thành đoạn ngắn; dùng nhãn tính góc nào có nhiều vật thể duy nhất chỉ mình nó thấy.
2. Đếm số lần góc tốt hơn đổi chiều và độ dài các đoạn.
3. Sau đó, nếu đạt: so oracle động theo đoạn với góc cố định, NDET, NDET trượt cửa sổ, và NDET với trễ d đoạn.
- **GO** cho bước 3 nếu có đảo chiều ở ≥ 50% số chuỗi với đoạn đủ dài để quyết định (ví dụ ≥ 5 epoch).
- **GO** cho RQ2 nếu oracle động hơn quy tắc động tốt nhất ≥ 2 điểm F1 vật thể duy nhất, dương ở ≥ 73% số chuỗi.
- **NO-GO** nếu không có đảo chiều, hoặc NDET trượt cửa sổ đã bắt kịp oracle (như B1).

## 8. Câu F: có lý do khoa học phải dùng MARL không?

**Hiện tại: không.**

- **Bên nhận là BS và đã có toàn bộ thông tin đã nhận.** Mọi bài gần nhất (SchedCP, RACooper, CCSRA, C-MASS, TAMP) dùng bộ điều khiển tập trung. Điều phối qua đường xuống (bản đồ yêu cầu, "đã biết") rẻ hơn đường lên nhiều bên ([reassess-notes/axis1](reassess-notes/axis1-collab-perception-u2u.md)).
- **Bằng chứng của dự án:** quy tắc đơn giản bắt kịp oracle GT (B1); PPO không có cơ chế giải thích được (A8); dư địa ở A7 là kiến thức cấp cảnh, không phải liên kết.
- **MARL cho MAC ngữ nghĩa đã có** (PRISM, MADRL), nên MARL tự nó không phải đóng góp.
- **Khi nào MARL mới có lý do [SUY LUẬN]:** khi các UAV phải tự quyết mà không có thông tin kịp thời từ BS (phản hồi trễ ngang thời gian thay đổi giá trị, không có U2U), và khi một bộ tập trung với thông tin trễ thua rõ một oracle có thông tin tươi. Chính phép thử RQ2 phải cho thấy khoảng cách đó trước.
- **Thứ tự bắt buộc:** quy tắc đơn giản → tối ưu tập trung (greedy có cận, kiểu C-MASS) → bandit tập trung → RL tập trung → MARL. Chỉ đi bước tiếp khi bước trước để lại dư địa đo được.

## 9. Phán quyết

| Hướng | Phán quyết | Lý do ngắn |
|---|---|---|
| PPO / MAPPO chọn TOKEN / CROP / FRAME theo ngữ nghĩa + CSI | **NO-GO** | HyComm + SchedCP + CCSRA đã chiếm |
| RQ1: liên kết bất định làm tín hiệu truyền | **HIGH RISK** | Khoảng trống thật trong nguồn đã kiểm; có thể không có dư địa nếu homography đã đủ; phép thử rẻ trên bản ghi MDMT có sẵn |
| RQ2: giá trị góc thay đổi + phản hồi trễ | **NO-GO** (thuật toán) / **HIGH RISK** (đánh giá) | Phần thời gian đã có (MASS, PIB, MVSparse); chỉ còn phản hồi trễ; cần dữ liệu có đảo chiều (CrossTrack-UAV) |

Không có hướng nào được GO. Nếu RQ1 cũng NO-GO ở phép thử khả thi, tôi đề nghị đổi trọng tâm: SwarmBridge khi đó nên là một **nghiên cứu đánh giá** (khi nào hợp tác đa UAV thật sự giúp, đo bằng detector và codec thật, ID xuyên góc thật, chỉ số vật thể duy nhất, baseline đơn giản bắt buộc), không phải một thuật toán mới.

## 10. Việc chưa làm

- Chưa đọc toàn văn PIB, MVSparse, JigsawComm, WhisperNet, MASS: đây là các bài cần đọc kỹ nhất nếu RQ1 hoặc RQ2 được mở.
- Chưa rà soát tài liệu bandit có phản hồi trễ (cho RQ2).
- Chưa xác minh venue chính thức của SchedCP và SComCP, và arXiv của How2comm.
