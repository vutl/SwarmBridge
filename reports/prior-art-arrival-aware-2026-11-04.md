# Kiểm tra prior art: "Arrival-Aware Cooperative Semantic Communication for Multi-UAV Tracking" (v0), 2026-11-04

Mục tiêu: cố **bác bỏ** tính mới của đề xuất, bằng cách tìm các công trình gần nhất.
Cách làm: đọc toàn văn (dùng `pdftotext`) 14 PDF trong `pdf/`, và tìm trên web các công trình không có trong `pdf/`.
Với nguồn web, phần lớn tôi chỉ đọc được abstract hoặc trang HTML. Mỗi chỗ như vậy đều ghi rõ ở mục 5.
Chỉ đọc và phân tích, không chạy GPU.

Nhắc lại claim cần kiểm tra:
V(m | S_t) = E[L_future(không có m) − L_future(có m)] − λ·bits(m).
Trong đó L_future là loss tracking (vị trí, bỏ sót, ID switch) trong một horizon **sau thời điểm m đến**.
Giá trị này điều kiện trên các message khác đã gửi hoặc đang trên đường (S_t) và trên trạng thái tracker của bên nhận.
Giá trị được học từ rollout (MARL kiểu CTDE), rồi dùng để chọn message, biểu diễn và ngân sách bit cho từng UAV.

## 1. Ma trận so sánh

Các cột:
- (i) gửi gì / có chọn biểu diễn không
- (ii) giá trị message được định nghĩa thế nào
- (iii) giá trị có tính đến trễ / thời điểm đến không
- (iv) giá trị có là cận biên, điều kiện trên message của agent khác không
- (v) tác vụ; có mô hình ID/association không
- (vi) phương pháp ra quyết định
- (vii) mô hình bit, queue, kênh
- (viii) dataset / simulator

"Không rõ" nghĩa là tôi không tìm thấy trong phần đã đọc.

| Công trình | (i) | (ii) | (iii) | (iv) | (v) | (vi) | (vii) | (viii) |
|---|---|---|---|---|---|---|---|---|
| **GOST**, Liao et al., IEEE TMC 25(10), 10/2026, DOI 10.1109/TMC.2026.3700322 | Trạng thái UAV (vị trí, queue, SINR, công suất) và dữ liệu cảm biến z. Không chọn biểu diễn theo message | AoI của status (reward Eq.19). "Significance" = mức suy ra được của biến từ biến khác, dùng để chỉnh tần số lấy mẫu (cuối Sec.IV-B) | **Một phần.** Ngưỡng stale T_th (Eq.15): dữ liệu stale bị thay bằng suy diễn ARIMA/KF/UIF/GAT/DECI. Không định giá message theo thời điểm đến | Một phần: biến S_N suy ra từ S_M nên S_N được gửi thưa hơn. Đây là quan hệ giữa các biến, không giữa message của các UAV | Bám **một mục tiêu điểm**, đo target loss rate và collaboration rate (Fig.10). Không có ID/association | MADDPG (CTDE) cho lập lịch power/tần số và điều khiển bay (Sec.V-A,B), thêm EWC | Rician, SINR, FDMA, xác suất thành công Eq.8; queue Q có trong state | Simulator tự viết, N_c=2, N_u=6 |
| **JOSC** (Adaptive Semantic Compression), Zhou et al., EURASIP JWCN 2026:93, doi 10.1186/s13638-026-02637-2 | Frame video đã nén ngữ nghĩa. Chọn **tỉ lệ nén**, có gửi hay không, và công suất | AoIS = độ tươi kết hợp MS-SSIM (Eq.18–19) | Có, theo kiểu AoI (tuổi tin) và ràng buộc trễ. Không xét giá trị theo tác vụ khi tin đến | Không. Mỗi UAV có AoIS riêng | Giám sát (độ giống ảnh), không tracking | CTDE DRL (critic toàn cục) | Công suất, tốc độ Shannon, ràng buộc delay (21a–d) | Mô phỏng K=3–5 UAV |
| **CERF**, Hao et al., arXiv 2609.00951, 9/2026 | **Box 3D** + pose, không chọn biểu diễn | Không định giá. Gửi mọi box | Bù trễ ở **bên nhận** bằng Kalman + dự báo vận tốc không đổi, K=3 (Eq.5). Không dùng trễ để chọn message | Không. NMS khi hợp nhất (Eq.7) | Detection + tracking (AMOTA, Table 2) + dự báo quỹ đạo. Không tối ưu theo ID | Heuristic | Chỉ ghi bandwidth (Mbps) | UAV3D, Air-Co-Pred |
| **How2comm**, Yang et al., NeurIPS 2023 | Feature thưa (lọc không gian + kênh) và feature flow | Cơ chế "mutual information-aware" giữ feature giàu thông tin, theo request của ego (Fig.2) | Bù trễ bằng flow-guided prediction. Không đưa trễ vào giá trị | Một phần: chia vùng exclusive/common với ego. Không xét giữa các sender | Detection | Học end-to-end | Bandwidth, trễ cố định | DAIR-V2X, V2XSet, OPV2V |
| **CoSDH**, CVPR 2025 | Feature thưa + **late fusion box** (hybrid luôn bật, không chọn theo message) | Cung–cầu: M_ji = D_i ⊙ S_j (Sec.3.3) | Không | Chỉ theo cặp ego–helper | Detection | Ngưỡng + học | Tổng bandwidth 27 Mbps | OPV2V, V2XSim, DAIR-V2X |
| **Where2comm**, Hu et al., NeurIPS 2022 | Feature thưa theo vùng + request map | Confidence không gian của sender × nhu cầu của ego | Không thấy | Theo cặp. Có communication nhiều vòng | Detection | Heuristic + học | Bandwidth thay đổi được | OPV2V, V2X-Sim, DAIR-V2X, CoPerception-UAVs |
| **CodeFilling**, Hu et al., CVPR 2024 | Chỉ số **codebook** (codebook cố định, không chọn biểu diễn) | Tổng information score có ngưỡng: min(C_j + Σ_i M_ij⊙C_i, u) (Eq.3a, Sec.4.2.1) | Không. Chỉ có thử độ bền với latency 0–500 ms (Fig.8) | **Có**: utility bão hòa khi cộng dồn từ mọi supporter, nên tránh gửi trùng | Detection | Tối ưu cục bộ phía supporter | Ràng buộc số vùng (3b) | DAIR-V2X, OPV2VH+ |
| **SComCP**, Gan et al., arXiv 2507.00895 | Feature quan trọng + JSCC | Mạng chọn feature theo mức quan trọng | Không | Không | Detection | Học end-to-end | AWGN/fading, SNR | OPV2V-like (V2V) |
| **CoDS**, Gan et al., arXiv 2512.22513 | Feature nén + ADC ngữ nghĩa thành bitstream | Mạng uncertainty bỏ feature lỗi giải mã | Không | Không | Detection | Học | Kênh số, kênh mã hóa | V2V |
| **AoII**, Maatouk et al., arXiv 2012.13214 | Status update của nguồn Markov | AoII = thời gian ước lượng sai × f(·) | Có (bản chất là thời gian) | Không. Một cặp phát–thu | Ước lượng trạng thái | Chính sách ngưỡng ngẫu nhiên hóa, tối ưu | Kênh tin cậy p | Lý thuyết |
| **PASC**, Yang et al., IEEE WCL 15 (2026) 4280 | Ảnh nén ViT, tỉ lệ nén thích nghi | QoE, deadline 150 ms | Dự báo **CSI** bằng GRU. Không dự báo giá trị message | Không | Truyền ảnh | DRL | Subchannel, bandwidth, power | Mô phỏng |
| **CC** (Confidence-Calibrated), Zheng et al., EURASIP JWCN 2026:112, doi 10.1186/s13638-026-02655-0 | Semantic token + chọn **độ sâu lượng tử**, subchannel, power | "Reliable semantic value" = confidence đã hiệu chỉnh × chất lượng định vị × trọng số lớp × **độ mới so với bộ nhớ ngữ nghĩa toàn cục** | Không. Truyền một lần, không truyền lại. Tác giả ghi "temporally aware" là future work | **Có** (dạng heuristic): trừ phần đã có trong bộ nhớ, do UAV khác hoặc đã giao trước đó | Detection (mAP@50). Tracking ghi là future work | RL phân cấp: lớp trên chia tài nguyên, lớp dưới chọn nội dung + bit | OFDM, 16 subchannel | VisDrone + simulator, U=4 |
| **GOSC** (Heterogeneous Agentic Teams), Rezazadeh et al., arXiv 2609.39477, 9/2026 | Bản ghi CONFIRM/INTENT/EVIDENCE + **rate** (độ tin cậy) | VoI = thay đổi utility kỳ vọng của đồng đội (Eq.4), hoặc belief divergence, AoI, throughput. Quy tắc: gửi khi giá trị > giá truyền (Thm.1) | Rất ít. Trễ/mất downlink chỉ là thử độ bền. Giá trị không giảm theo thời điểm đến | **Có**: giá trị tính trên common knowledge K đã fuse và broadcast (Prop.1). Không xét các message đang trên đường trong cùng slot | Search & rescue trên lưới. Không tracking/ID | Myopic, không học: knapsack nhiều lựa chọn + Lagrange (Eq.5, Thm.2) | Rayleigh, FBL, HARQ, budget kênh/slot | Simulator lưới + agent LLM |
| **DHD**, Wang et al., NeurIPS 2024, arXiv 2405.14674 | Feature BEV thưa (cửa sổ trượt SISW, 25%) | Lượng thông tin cục bộ | Không | Không | Dự báo quỹ đạo | Heuristic + học | Tỉ lệ truyền | Air-Co-Pred |
| **PragComm**, Hu et al., arXiv 2401.12694 (venue: không rõ) | Feature thưa theo không gian–**thời gian** + codebook | Top-k theo confidence. Thời gian: gửi khi trạng thái Kalman thay đổi ‖T(t−1)−T(t−2)‖ > δ. Nhân với phần bù của vùng bên nhận đã có (Eq.10–12) | Có module dự đoán bù message trễ. Thử latency 0–500 ms chỉ với detection | Không. Theo cặp; hợp nhất bằng max | Detection + **tracking** (HOTA/MOTA, Kalman CV). Train bằng loss detection | Luật + học (kiểu ADMM). Không RL | Budget chung (Eq.4c) | V2V4Real, OPV2V, V2X-Sim2.0 |
| **VIL2C**, Zhang et al., arXiv 2511.19146 | Message MARL; cấp bandwidth + power | VoI = KL(π_j(a\|o_j,m_i) ‖ π_j(a\|o_j)) / τ (Eq.3–5) | Trễ chỉ là **mẫu số** τ = L/R. Nội dung không mất giá theo thời điểm đến | Không. Theo cặp | Predator-Prey, SMACv2 | Mạng ResoNet học CTDE | Bandwidth/power | MARL benchmark |
| **Tran et al.**, arXiv 2601.21482, 1/2026 | Chọn cảm biến gửi | **Information gain phụ thuộc trễ**, tính theo thông tin trên mỗi joule | **Có**: cập nhật Kalman cho đo đạc trễ (posterior fusion) | Lập lịch nhiều cảm biến chung | Ước lượng trạng thái một quá trình, không ID | PPO | Trễ ngẫu nhiên, năng lượng | Mô phỏng |
| **Singh, Kamath, Kumar**, arXiv 1908.01119 | Gói tin cho ước lượng từ xa | VoI giảm theo tuổi, tăng theo độ chính xác của nguồn. Gửi gói có VoI cao nhất là tối ưu | **Có** (theo tuổi) | Không | Ước lượng Gauss-Markov, Kalman gain cố định | Chứng minh tối ưu | Kênh không tin cậy | Lý thuyết |
| **SMV** (Shapley Message Value), Xue et al., IJCAI 2022 | Truy vấn đồng đội | **Shapley = đóng góp cận biên** của message vào agent | Không | **Có** (coalition) | MARL (SMAC…) | Selector học | Không | SMAC, Hallway |
| **Aoki et al.** arXiv 2004.10927 (2020); **Abdel-Aziz et al.** IEEE TCOM 70(2) 2022, doi 10.1109/TCOMM.2021.3126650 | Đối tượng/nội dung CPM. Abdel-Aziz chọn cả **độ phân giải** (quadtree) + RB + liên kết | DRL giảm trùng lặp, tăng thông tin nhận được | Không thấy trong abstract | Một phần: giảm trùng giữa xe | Detection | DRL / federated RL | Kênh V2X, congestion | CIVS / mô phỏng |
| **ETSI CPM VoI** (TS 103 324, bản nháp 2026), qua arXiv 2609.01478 | Đối tượng trong CPM | VoI theo khoảng cách, độ khó đoán chuyển động, độ tươi | Có (độ tươi) | Có luật redundancy mitigation | Nhận biết môi trường | Luật | Tải kênh | NR-V2X sidelink |
| **CoBEVFlow** NeurIPS 2023 (2309.16940); **SyncNet** ECCV 2022 (2207.08560) | Feature + flow | Không định giá | **Bù** bất đồng bộ ở bên nhận | Không | Detection | Học | Trễ ngẫu nhiên | IRV2V, DAIR-V2X |
| **When2com** CVPR 2020 / **Who2com** ICRA 2020 | Feature | Handshake, matching score | Không | Chọn cộng sự | Segmentation/detection | Học | Không | AirSim-MAP |
| **TAMP** arXiv 2601.04542; **Update the Unseen Only** arXiv 2607.20967; **Fresh2comm** arXiv 2502.07852 | Vùng / cập nhật cảm biến | Phạt theo AoI. TAMP ánh xạ AoI + khối lượng → AP bằng hàm thực nghiệm | **Có** (AoI giảm giá trị) | "Unseen Only" tính đến cảm biến cục bộ của bên nhận | Detection (AP/mAP) | Lyapunov / online max-weight | Bandwidth, trễ | RCooper / quỹ đạo xe |

## 2. Hai–ba công trình gần nhất

**(1) GOSC (Rezazadeh et al., arXiv 2609.39477): gần nhất về mặt khung quyết định.**
- Đã có:
  - Giá trị message = giá trị cho tác vụ của đồng đội trừ chi phí truyền (giống V − λ·bits).
  - Giá trị tính **so với kiến thức chung đã được giao** (Prop.1, Eq.4), tức là cận biên theo các message đã đến.
  - Chọn chung "gửi gì, khi nào, rate nào" bằng một quy tắc, có chứng minh cấu trúc (Thm.1–2).
- Kết quả đáng chú ý cho mình:
  - Giá trị ước lượng thô là đủ. Giá trị ToM kiểu "cận biên" không hơn throughput/AoI.
  - Với 2 agent hoặc khi tài nguyên dư, "gửi hết" lại tốt hơn.
  - Hai điều này **đi cùng hướng với quan sát (B)** của mình và là lời cảnh báo cho phần "học giá trị".
- Chưa có:
  - Tracking/perception, ID/association.
  - Giá trị phụ thuộc thời điểm đến (trễ chỉ là thử độ bền).
  - Điều kiện trên message **đang trên đường**.
  - Chọn biểu diễn giàu/nghèo (chỉ có rate).
  - Học CTDE: GOSC là myopic và không học.

**(2) PragComm (Hu et al., arXiv 2401.12694) cùng CodeFilling (CVPR 2024): gần nhất về collaborative perception.**
- PragComm đã có:
  - Chọn message theo **thời gian bằng trạng thái tracker Kalman** (gửi khi trạng thái đổi).
  - Che bớt những vùng bên nhận đã có.
  - Biểu diễn codebook.
  - Bù trễ ở bên nhận.
  - **Đánh giá tracking** (HOTA/MOTA).
- CodeFilling đã có utility bão hòa khi cộng dồn nhiều supporter, tức là giá trị cận biên kiểu heuristic.
- Chưa có:
  - Giá trị theo **loss tracking ở tương lai**: PragComm train bằng loss detection, luật là ngưỡng.
  - Trễ đưa vào giá trị.
  - ID switch.
  - Chọn giữa box và feature cho từng message.
  - Ngân sách có queue.
  - MARL.

**(3) CC (Zheng et al., JWCN 2026:112), với JOSC và GOST là phụ: gần nhất về semantic comm cho multi-UAV.**
- CC đã có:
  - Chọn token + độ sâu lượng tử + subchannel + power cho nhiều UAV bằng RL phân cấp, tức là có (d) dạng "biểu diễn + ngân sách".
  - Giá trị có thành phần **độ mới so với dữ liệu UAV khác**.
- CC chưa có:
  - Thời gian (tác giả tự ghi tracking và tính thời gian là future work).
  - Cũng chính CC định nghĩa heuristic "chỉ gửi cái bên nhận chưa có" mà pilot của mình thấy **kém hơn gửi hết**.
- GOST đã có tracking cộng tác qua UAV trong mạng hạn chế, dữ liệu stale và MADDPG CTDE.
- GOST chưa có:
  - Lựa chọn message theo giá trị (họ chỉ lập lịch power/tần số theo AoI).
  - Nhiều đối tượng, ID.
  - Perception thật.
  - GOST xa hơn mình tưởng ban đầu.

## 3. Kết luận theo từng thành phần

**(a) Giá trị tại thời điểm đến: "làm một phần".**
- Trong lý thuyết ước lượng, ý "giá trị gói tin giảm theo tuổi / trễ" đã có, và đã được dùng để lập lịch:
  - Singh et al.: VoI giảm theo tuổi, ưu tiên VoI cao là tối ưu.
  - Tran et al. 2026: information gain phụ thuộc trễ, Kalman với đo đạc trễ, PPO.
  - AoI/AoII/AoIS, TAMP (AoI → AP).
- Trong collaborative perception, trễ chỉ được **bù ở bên nhận**: SyncNet, CoBEVFlow, How2comm, CERF, PragComm.
- Chưa thấy công trình nào:
  - định giá **từng message perception** theo trạng thái tracker tại thời điểm đến;
  - hoặc chỉ ra rằng message trễ 1–2 frame có giá trị **âm** (quan sát A).
- Phần còn mới hẹp: giá trị âm khi đến muộn, gắn với fusion/association, cho tracking nhiều đối tượng.

**(b) Giá trị cận biên, điều kiện trên message khác: "đã có người làm" (ở dạng chung).**
- Các dạng đã có:
  - Shapley Message Value (IJCAI 2022) là định nghĩa cận biên đúng nghĩa trong MARL.
  - GOSC tính giá trị so với common knowledge.
  - CodeFilling dùng utility bão hòa giữa các supporter.
  - CC dùng novelty giữa các UAV.
  - Luật redundancy và VoI trong ETSI CPM; Aoki/Abdel-Aziz dùng DRL giảm trùng.
- Còn trống:
  - Điều kiện trên message **đã gửi nhưng chưa đến** (đang trên đường hoặc trong queue).
  - Phép đo cận biên theo **loss tracking**, chứ không theo độ phủ thông tin.
- Quan sát (B) ("trùng vẫn có ích nhờ xác nhận và association") **mâu thuẫn trực tiếp với giả định** của CC, CodeFilling và luật ETSI. Đây là điểm có giá trị thực nghiệm, nhưng phải chứng minh chặt.

**(c) Loss tracking (có ID/association) làm mục tiêu định giá: "chưa thấy" làm mục tiêu; "làm một phần" ở khâu đánh giá.**
- PragComm và CERF có đánh giá tracking, nhưng giá trị và loss huấn luyện đều theo detection.
- GOST tối ưu target loss cho một mục tiêu, không có ID.
- Không tìm thấy công trình nào lấy ΔIDSW hoặc Δassociation làm giá trị message.

**(d) Chọn chung biểu diễn + ngân sách giữa các UAV: "làm một phần", gần như đã có.**
- Đã có:
  - CC: token + độ sâu lượng tử + kênh, RL phân cấp, nhiều UAV.
  - JOSC: tỉ lệ nén + power + gửi/không, CTDE.
  - Abdel-Aziz: nội dung + độ phân giải + RB.
  - GOSC: nội dung + rate.
  - CoSDH: intermediate + late luôn bật.
- Chưa thấy chọn **box ↔ ROI feature/token cho từng đối tượng, từng slot**, có queue và trễ, phục vụ tracking. Nhưng đây là khác biệt về mức độ chi tiết, không phải khác biệt về loại bài toán.

**(e) Tổ hợp (a)+(b)+(c)+(d) với MARL CTDE: "chưa thấy" trong một công trình.**
- Rủi ro reviewer: tổ hợp này dễ bị xem là "GOSC + PragComm + CC ghép lại".
- Nó chỉ đứng được nếu chứng minh rằng mỗi phần là **cần thiết**: bỏ phần nào thì tracking tụt, và các baseline mạnh ở mục 4 không làm được tương đương.

## 4. Claim tính mới hẹp hơn nhưng bảo vệ được

Đề xuất claim:
> Với multi-UAV camera tracking có trễ và queue, giá trị của một message object-level phải đo **tại thời điểm đến**, theo **thay đổi loss tracking (gồm association/ID)** của tracker bên nhận, và điều kiện trên các message đang trên đường.
>
> Hai hệ quả kiểm chứng được mà các tiêu chí hiện có (confidence, cung–cầu, novelty, AoI, VoI theo tuổi) bỏ sót:
> (A) message trễ có thể có **giá trị âm**;
> (B) message "trùng" có giá trị dương nhờ xác nhận và association.
>
> Một bộ ước lượng giá trị học được (counterfactual, CTDE) khai thác hai hiệu ứng này và vượt các tiêu chí đó ở cùng bit/slot.

Có thể tách thành hai bài:
- Bài phân tích/benchmark: "khi nào message giúp tracker", chỉ cần (A), (B) cùng các baseline heuristic. Rủi ro thấp hơn.
- Bài phương pháp: thêm phần học V.

Để claim đứng được, cần:
- (A) và (B) phải tái lập trên **tập test**, không chỉ 3 scene dev, có CI.
- Phải tách được: (A) có phải chỉ do bù trễ kém hay không. Cần thử với bù trễ tốt (Kalman + CV như CERF, hoặc OOSM update đúng). Nếu bù trễ đúng làm (A) biến mất thì (a) mất cơ sở.

Baseline bắt buộc (cùng detector, cùng bit/slot, cùng trễ và mất gói):
1. Gửi hết box, có bù trễ Kalman kiểu CERF; late fusion thường.
2. Where2comm (confidence × request).
3. CodeFilling (utility bão hòa).
4. CoSDH (cung–cầu + hybrid).
5. PragComm (thay đổi trạng thái Kalman + mask bên nhận). Đây là đối thủ trực tiếp ở phần temporal.
6. Novelty kiểu CC hoặc luật redundancy ETSI.
7. Lập lịch AoI / max-age / VoI giảm theo tuổi (Singh), và TAMP-like.
8. **Baseline giải tích mạnh nhất**: giảm vết hiệp phương sai (hoặc giảm xác suất association sai) của Kalman, **tính tại thời điểm đến**, kiểu Tran et al. 2026. Nếu V học được không thắng baseline này, phần "học" không đáng giá.
9. Quy tắc giá trị − giá của GOSC, với giá trị lấy từ 8.
10. Oracle counterfactual (chạy lại có/không có m) để làm cận trên.

## 5. Những gì chưa kiểm tra được

- **RAI-CoS**: không tìm thấy công trình nào tên này. Cần tên đầy đủ hoặc tác giả.
- Chỉ đọc abstract hoặc trang arXiv (không đọc toàn văn):
  - Fresh2comm (2502.07852), TAMP (2601.04542), Update the Unseen Only (2607.20967);
  - CooperTrim (ICLR 2026, 2602.13287), JigsawComm (2511.17843), SchedCP (2502.10456);
  - Tran et al. (2601.21482), Singh et al. (1908.01119);
  - Aoki (2004.10927), Abdel-Aziz (TCOM 2022), I2C (NeurIPS 2020, 2006.06455), SMV (IJCAI 2022);
  - CoBEVFlow, SyncNet.
- VIL2C và PragComm: đọc bản HTML, nhưng phần thí nghiệm của PragComm đọc chưa hết. Venue của PragComm chưa rõ.
- Wu et al., "Goal-oriented semantic communications for robotic waypoint transmission: the value and age of information approach", IEEE TWC 23(12) 2024, doi 10.1109/TWC.2024.3424493: chỉ thấy qua trích dẫn trong JOSC và GOSC, chưa đọc.
- Bản nháp ETSI TS 103 324 (VoI cho CPM): chỉ biết qua arXiv 2609.01478, chưa đọc chuẩn gốc.
- THAAC, "Adaptive Collaboration in Multiplex UAV Networks with Dynamic Communication", ACM, doi 10.1145/3840392: chưa đọc.
- Chưa tìm kỹ các mảng:
  - task-oriented multi-device edge inference (dạng distributed information bottleneck, giảm trùng giữa thiết bị);
  - sensor management cho multi-target tracking với đo đạc out-of-sequence.
  - Đây là các mảng có thể chứa công trình gần (a)+(c) mà tôi bỏ sót.
- Text GOST trích bằng `pdftotext` bị lỗi ký hiệu ở một số phương trình. Các kết luận ở trên dựa vào phần chữ, không dựa vào ký hiệu.

Nguồn web: [CoBEVFlow](https://arxiv.org/abs/2309.16940), [SyncNet](https://arxiv.org/abs/2207.08560),
[VIL2C](https://arxiv.org/abs/2511.19146), [Fresh2comm](https://arxiv.org/abs/2502.07852), [TAMP](https://arxiv.org/abs/2601.04542),
[Update the Unseen Only](https://arxiv.org/abs/2607.20967), [CooperTrim](https://arxiv.org/abs/2602.13287),
[PragComm](https://arxiv.org/abs/2401.12694), [JigsawComm](https://arxiv.org/abs/2511.17843), [SchedCP](https://arxiv.org/abs/2502.10456),
[Tran et al.](https://arxiv.org/abs/2601.21482), [Singh et al.](https://arxiv.org/abs/1908.01119),
[SMV](https://www.ijcai.org/proceedings/2022/82), [I2C](https://arxiv.org/abs/2006.06455),
[Aoki et al.](https://arxiv.org/abs/2004.10927), [Abdel-Aziz et al.](https://arxiv.org/abs/2012.03414),
[ETSI CPM/NR-V2X](https://arxiv.org/abs/2609.01478), [GOSC](https://arxiv.org/abs/2609.39477).
