# Rà soát novelty #45 (bước A): ma trận tính mới cho hệ "nhiều UAV offload ảnh theo nhiệm vụ"

Ghi chú nguồn gốc: bản này do một agent phụ soạn (đọc 6 PDF trong `pdf/`, khoảng 15 truy vấn web, kế thừa 4 báo cáo cũ), sau đó được rà lại trước khi đưa vào repo. Các nhãn [PDF] / [KẾ THỪA] / [WEB-TT] cho biết mức đọc của từng dòng. Kết quả A-T1 (oracle tác vụ trong simulator) ở [swarmbridge-at1-task-utility-result.md](swarmbridge-at1-task-utility-result.md).

Ngày 18/11/2026. Chỉ đọc tài liệu, không chạy thí nghiệm. Kế thừa 4 báo cáo cũ, không làm lại những gì chúng đã phủ:
- `reports/archive/uav-semcom-a/novelty-audit-a5-l1-2026-11-10.md` (gọi tắt **[A5]**, 24 bài trong `pdf/`);
- `reports/archive/r1-visdrone/novelty-audit-g-l2-2026-11-10.md` (**[GL2]**, dòng video analytics: DAO, AdaCompress, SMR, AccMPEG, CASVA, DACC, A²-UAV...);
- `reports/archive/rq1-novelty/swarmbridge-collab-perception-novelty-review-2026-11-13.md` (**[CP]**, collaborative perception);
- `reports/scheduling/swarmbridge-ax2-result.md` mục 6 (**[AX2]**, khảo sát 19 bài RL + 6 repo).

**Mức đọc.** [PDF] = tôi đọc lại phần hệ thống / bài toán / baseline từ PDF trong `pdf/` lần này. [KẾ THỪA] = lấy kết luận của báo cáo cũ. [WEB-TT] = chỉ thấy tóm tắt hoặc đoạn trích từ tìm kiếm. [WEB-TV] = có đọc đoạn toàn văn. "Không thấy" = không thấy trong nguồn đã kiểm, không phải "không tồn tại". Tìm kiếm web chỉ gồm khoảng 15 truy vấn, **không hệ thống**; IEEE Xplore và một số trang không đọc được.

## 0. Hệ của mình, viết lại để so cho đúng

K = 5 UAV bay khảo sát. Mỗi UAV gửi **ảnh độc lập** (VisDrone) về **một BS**, dùng chung 2 kênh. PHY thực tế: CQI trễ, OLLA, HARQ/ARQ, BLER. Mỗi ảnh chọn một mode nén toàn ảnh (AVIF / mbt2018-mean / JPEG2000, nhiều mức). BS chạy YOLO11s. Hạn chót 4–12 s mỗi ảnh. Thành phần đã kiểm chứng: (1) gán kênh chung nhân quả (JA); (2) pacing (hover khi backlog dài); (3) MPC chọn lọc, giới hạn 50 ms, mô hình tốc độ **hiệu chỉnh online** từ throughput/ACK (+17 return so với mô hình danh định); (4) mode cố định mbt q3 là heuristic rất mạnh. PPO đã được thử công bằng và đã đóng (A-X8, A-X9). Return và recall không khớp nhau (hai stack cùng return lệch 3–5 điểm recall). Dư địa oracle theo từng ảnh khoảng +1 điểm recall (G-L2).

**Đây là offloading theo nhiệm vụ, không phải collaborative perception.** BS không tự cảm nhận, không hợp nhất góc nhìn, không có trùng lặp xuyên UAV được khai thác. Vì vậy cơ chế "bên nhận đã biết gì" của dòng CP (Where2comm, SchedCP, CodeFilling) không áp dụng trực tiếp. Nhưng sự khác biệt này **không tự tạo ra tính mới**: nó chỉ đổi đối thủ gần nhất từ dòng CP sang dòng video analytics, ABR và UAV-SemCom.

## 1. Ma trận tính mới

Viết tắt cho cột giả định:
- **Kênh**: S = Shannon / dung lượng lý tưởng; LA = link adaptation có MCS rời rạc, BLER, phản hồi trễ.
- **CSI**: P = biết hoàn hảo; T = chỉ thống kê hoặc trễ.
- **Tiện ích**: GT = cần nhãn; noGT = không nhãn.
- **Kiểu**: F = hợp nhất (collaborative); I = offload độc lập.
- **Baseline**: tune hay không.

### (a) Ước lượng tiện ích nhiệm vụ không cần GT lúc ra quyết định

| Bài | Venue, năm | Giải gì | Giả định |
|---|---|---|---|
| ORIC: Qiu, Wang, Hu, Guérin, Lu, "Optimizing Edge Offloading Decisions for Object Detection" | IEEE/ACM SEC 2024; arXiv 2410.18919 [WEB-TT] | Ước lượng lợi ích mAP **từng ảnh** khi offload, chỉ từ detection cục bộ; offload top-x % | I, noGT; ngân sách tỉ lệ ảnh, không kênh vô tuyến, không deadline |
| DAO (MM'22), AdaCompress (MM'19), SMR (TPAMI'24), DACC (ComNet 2026) | [KẾ THỪA GL2] | Mạng nhẹ / thống kê rẻ dự đoán độ chính xác theo cấu hình nén, theo ảnh hoặc đoạn | I; nhãn là output DNN ở chất lượng cao (giả nhãn) hoặc GT; một nguồn, không lập lịch nhiều tác tử (DACC có Lyapunov) |
| AccMPEG, CASVA, RL-RC-DoT | [KẾ THỪA GL2] | Giả nhãn = output DNN trên bản gốc; quyết định cấp khối / đoạn | I, noGT lúc chạy; không kênh vô tuyến |
| ASTRA, arXiv 2609.07020 | 2026 [WEB-TT] | Thích nghi cấu hình runtime; "proxy ground truth" = output detector ở cấu hình vàng | I; camera cố định, nhiều camera giả lập |
| SchedCP (bản label-free) | arXiv 2502.10456 [PDF] | Thưởng không nhãn từ bản đồ tin cậy, thay cho giảm loss detection | F; S, P |
| CCSRA | EURASIP JWCN 2026 [PDF] | Giá trị ngữ nghĩa từ confidence **đã hiệu chỉnh** + chất lượng định vị | F (memory bank); hiệu chỉnh khớp offline bằng GT trên val; reward vẫn có U_t so với GT |
| Xiong et al., "Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion" | ICML 2026, PMLR 306 [PDF] | Cảnh báo: confidence của chính mô hình không đáng tin khi "tự tin mà sai" | Không phải truyền thông; chỉ là bằng chứng phụ chống dùng confidence làm proxy |

**Khoảng trống còn lại cho mình:** rất hẹp. Ước lượng không nhãn theo từng ảnh đã có nhiều (ORIC gần nhất về ý). Bằng chứng riêng của mình là **âm**: G-L2 cho thấy predictor theo ảnh không hơn luật biết payload, dư địa oracle khoảng +1 recall. Phần chưa thấy ai làm: dùng proxy không nhãn làm **hàm mục tiêu của bộ lập lịch nhiều UAV có deadline** và đo xem nó có khớp recall hơn return hiện tại không. Đây mới là giả thuyết, chưa có số liệu.

### (b) Phân bổ tài nguyên ngữ nghĩa có hiệu chỉnh độ tin cậy

| Bài | Venue, năm | Giải gì | Giả định |
|---|---|---|---|
| CCSRA: Zheng, Li, Zuo, "Confidence-calibrated semantic resource allocation for multi-UAV low-altitude networks" | EURASIP J. Wireless Commun. Netw. 2026:112, DOI 10.1186/s13638-026-02655-0 (DOI in trên PDF) [PDF + KẾ THỪA CP] | 4–8 UAV → edge; temperature scaling + đầu bất định; giá trị = lớp × conf hiệu chỉnh × định vị × (1 − trùng); HH-SAC chọn kênh con, công suất, token, bit lượng tử | Kênh S: B = T·W·log2(1+SINR) (pt. 21); Rician; "single-shot, không retransmission"; F-ish (trùng qua memory bank); VisDrone-DET + YOLOv8n, mAP@50; baseline là RL khác + Lyapunov-SCA, **không tune**, không CI |
| Các bài 2026 cùng chủ đề (che giấu ngữ nghĩa có bất định, arXiv 2603.01478; lập lịch theo SNR dự đoán, arXiv 2602.10482; C&C ngữ nghĩa nhiều UAV, arXiv 2602.00142) | [WEB-TT] | Bất định ngữ nghĩa / kênh trong phân bổ | Không dùng confidence detector; chưa đọc |

Tìm kiếm web theo tên chính xác của CCSRA không ra; venue và DOI chỉ xác minh qua PDF.

**Khoảng trống cho mình:** CCSRA hiệu chỉnh **độ tin cậy của detector** offline bằng GT. Cái mình hiệu chỉnh là **mô hình tốc độ / goodput của kênh**, online, từ ACK, không cần nhãn. Hai thứ khác loại. Vì vậy "hiệu chỉnh" không phải điểm trùng trực tiếp. Nhưng nếu mình claim "giá trị ngữ nghĩa có hiệu chỉnh" thì CCSRA chiếm. Mình không nên claim điều đó: mình không có bằng chứng rằng hiệu chỉnh confidence giúp gì trong hệ của mình.

### (c) Quỹ đạo / pacing + lập lịch + nén

| Bài | Venue, năm | Giải gì | Giả định |
|---|---|---|---|
| Huang et al., hybrid bit + generative SemCom SAGIN | IEEE JSAC 2025 [KẾ THỪA A5] | DSAC chọn mode (JPEG2000 / generative) + quỹ đạo, ràng buộc trễ từng task; có baseline "Hovering UAV" | S; downlink LEO → UAV → user; PSNR; không tune |
| Zhao et al., "Adaptive Task-Oriented Semantic Communication for UAVs With Joint Trajectory and Resource Optimization" | IEEE TNSE vol. 13, 2026, DOI 10.1109/TNSE.2026.3727280 (in trên PDF) [PDF] | PPO chọn độ sâu model + số symbol + công suất; Diffuser offline RL cho quỹ đạo | 1 UAV; LoS + AWGN, R = B·log2(1+SNR) (pt. 6); accuracy là **đường cong fit f(K, SNR)**; DP thắng PPO ở quy mô nhỏ ([AX2]) |
| A²-UAV | INFOCOM 2023 [KẾ THỪA GL2] | Chọn mức nén + triển khai mạng nhiều UAV multi-hop có deadline; YOLOv4 | Bảng Q theo lớp, không theo ảnh; tối ưu tổ hợp |
| IntUNe: Puligheddu, Casetti, Chiasserini, "Intelligent UAV Surveillance Networks With Edge-Assisted Execution of Computer Vision Tasks" | venue chưa xác minh [WEB-TT] | Tối ưu chung **tốc độ bay UAV**, số ảnh/giây, offload, tốc độ truyền; tốc độ bay ảnh hưởng độ chính xác detection | mmWave, có ràng buộc trễ; chưa đọc toàn văn |
| JOSC, PASC, Hybrid-action RL, Semantic-Driven Multi-UAV | JWCN 2026, WCL 2026, GLOBECOM 2023, WCL 2026 [KẾ THỪA A5] | Nhiều UAV / kênh con chung; nén + công suất + vị trí; deadline | S hoặc fit; MS-SSIM / sigmoid / text; không tune |
| Airtalking: Xu, Ota, Dong | IEEE IoT-J vol. 13 no. 17, 2026, DOI 10.1109/JIOT.2026.3696083 [PDF] | D2D U2U: có mã hoá ngữ nghĩa hay không, theo trễ và năng lượng; so stochastic / greedy / LinUCB | Shannon–Hartley; không detector; không phải uplink tới BS |

**Khoảng trống cho mình:** gần như không có. "Hover khi backlog dài" là một dạng đơn giản của tối ưu chung quỹ đạo và truyền thông (Huang, IntUNe, A²-UAV). Mình còn chưa chứng minh pacing tốt hơn quỹ đạo tối ưu. Điểm khác duy nhất là pacing được ghép với PHY có LA. Đó là một chi tiết, không phải đóng góp.

### (d) Lập lịch collaborative perception và HARQ ngữ nghĩa

| Bài | Venue, năm | Giải gì | Giả định |
|---|---|---|---|
| SchedCP: Liu, Liu, Liang, Ye, Guo, Jin, "Deep Reinforcement Learning-Based User Scheduling for Collaborative Perception" | arXiv 2502.10456 (2025); **venue IEEE chưa xác minh**, tìm web chỉ thấy arXiv [PDF] | DDQN chọn link theo CSI + bản đồ tin cậy; có thưởng không nhãn | F; tốc độ = dung lượng kênh, "distortion-free" (S); CSI P; baseline Nearest / RR / Max Rate không tune; RR thắng ở 500–600 kHz |
| Semantic Communication for Cooperative Perception using HARQ | arXiv 2409.09042 (2024); CoDS trích là IEEE TCCN 2025 (chưa xác minh) [PDF + KẾ THỪA] | SimCRC (Siamese) quyết định có truyền lại đặc trưng không, theo mất mát perception | F; một cặp V2V; OFDM QuaDRiGa; truyền lại sau 1 ms, tối đa 3 lần; không lập lịch nhiều người |
| HMS-SCP | arXiv 2608.14603 (2026) [KẾ THỪA CP] | Đa tỉ lệ, ngân sách chung | F; AWGN / Rayleigh; tỉ lệ cố định |
| DiffCP, HyComm, CoSDH, CodeFilling, Where2comm | [KẾ THỪA CP] | Chọn nội dung theo độ tin cậy / cung–cầu; lai box + dữ liệu thô | F; ngân sách byte lý tưởng |
| LIF: Hao et al., "Is Intermediate Fusion All You Need for UAV-based Collaborative Perception?" | arXiv 2504.21774 (2025) [PDF] | UAV gửi kết quả detection gọn; chọn vùng chia sẻ theo bất định | F; UAV3D (CARLA + AirSim, mô phỏng); băng thông tính bằng byte, không kênh |

**Khoảng trống cho mình:** không áp dụng trực tiếp. Dòng này giải bài toán khác (hợp nhất góc nhìn, giá trị phụ thuộc trạng thái bên nhận). Điểm chung với mình chỉ là lập lịch theo CSI + giá trị task. SchedCP đã có điểm này, nhưng với dung lượng Shannon và CSI hoàn hảo. Điều đáng ghi: **không bài nào trong dòng này đặt lập lịch trên LA có BLER / HARQ / CQI trễ**. Bài HARQ có truyền lại, nhưng chỉ ở một liên kết và không lập lịch. [CP] đã kết luận RL + CSI + ngữ nghĩa là prior art. Kết luận đó giữ nguyên.

### (e) Hiệu chỉnh mô hình online + lập kế hoạch giới hạn tính toán

| Bài | Venue, năm | Giải gì | Giả định |
|---|---|---|---|
| Yin, Jindal, Sekar, Sinopoli, "A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP" (MPC, **RobustMPC**, **FastMPC**) | SIGCOMM 2015, DOI 10.1145/2785956.2787486 [WEB-TT] | MPC chọn bitrate theo throughput dự đoán + buffer. FastMPC giải trước offline thành bảng để chạy rẻ (giới hạn tính toán). RobustMPC dùng ước lượng throughput bảo thủ | Một client; throughput đo ở tầng ứng dụng; QoE video |
| Mao, Netravali, Alizadeh, Pensieve | SIGCOMM 2017 [WEB-TV, grep toàn văn] | RL so với robustMPC. Pensieve mô tả robustMPC là MPC chuẩn hoá dự đoán throughput theo sai số dự đoán lớn nhất trong 5 chunk gần nhất, và ghi rằng luật này "require careful tuning and can backfire" | Như trên; báo RL hơn 12–25 % QoE |
| Akhtar et al., Oboe | SIGCOMM 2018 [WEB-TT] | Tự tune tham số của MPC / BOLA theo trạng thái mạng (phát hiện điểm đổi). Báo **hơn Pensieve khoảng 24 %** | ABR một client |
| Yan et al., "Learning in situ: a randomized experiment in video streaming" (Puffer / Fugu) | NSDI 2020 [WEB-TT] | Bộ dự đoán thời gian truyền học tại chỗ + MPC cổ điển. Thử ngẫu nhiên 8 tháng: các sơ đồ học phức tạp khó thắng sơ đồ đơn giản | Một client, Internet thật |
| StreamOptix | arXiv 2406.04632 [WEB-TT] | MPC tầng ứng dụng dùng ước lượng dung lượng từ PHY + LA dùng soft-ACK | Một liên kết |
| OLLA (thực hành chuẩn của LA) | — | Hiệu chỉnh ngưỡng SINR online theo ACK / NACK | Tầng PHY, không có mục tiêu task |
| Bài toán bandit có phản hồi trễ (Joulani 2013; Vernade 2020...) | [KẾ THỪA CP] | Học khi phản hồi trễ, cả khi môi trường không dừng | Lý thuyết |

**Khoảng trống cho mình:** đây là hàng nguy hiểm nhất. "MPC + mô hình tốc độ hiệu chỉnh online từ throughput quan sát + cổng sai số + giới hạn tính toán" về **cơ chế** đã có ở dòng ABR từ 2015–2020: RobustMPC (hiệu chỉnh bảo thủ theo sai số), FastMPC (giới hạn tính toán), Oboe (tự tune), Fugu (bộ dự đoán học tại chỗ). Kết luận "RL khó thắng MPC tốt / heuristic đơn giản" cũng đã có (Oboe so với Pensieve; Puffer). Phần còn lại chỉ là **bối cảnh**:
- nhiều UAV chia chung kênh, quyết định mode bị ghép với gán kênh;
- deadline theo từng ảnh;
- tiện ích là detection;
- sai lệch mô hình đến từ LA (MCS rời rạc, BLER, HARQ, CQI trễ) chứ không phải biến động throughput Internet.

Không thấy bài UAV-SemCom nào làm hiệu chỉnh online mô hình tốc độ. Các bài ở (b), (c), (d) đều giả định Shannon.

### (f) Phương pháp đánh giá

| Bài / nguồn | Venue, năm | Giải gì | Liên quan |
|---|---|---|---|
| Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" | NeurIPS 2021 (Outstanding Paper) [WEB-TT] | CI, IQM, chống báo điểm đơn lẻ khi có ít run | Chuẩn thống kê chung cho RL |
| Puffer (NSDI 2020) | [WEB-TT] | Thử ngẫu nhiên; nhấn mạnh sai số thống kê lớn; sơ đồ đơn giản khó thắng | Tiền lệ "reality check" cho điều khiển học trong mạng |
| [AX2]: 19 bài RL + 6 repo trong `pdf/` | [KẾ THỪA] | Không bài nào tune họ heuristic; không bài nào có CI cho mức lợi | Khác biệt của mình chỉ đúng trong phạm vi `pdf/` |
| [A5]: 24 bài | [KẾ THỪA] | Không bài nào so RL với heuristic tune; tối đa là 10 seed ± std (xmodal) | Như trên |
| [GL2] | [KẾ THỪA] | Không thấy baseline content-agnostic khớp bit (trộn trên bao lồi, luật biết payload) hay kiểm nhiễu detector bằng ảnh lật; SMR (BD-rate so với QP hằng) gần nhất | Chỉ trong nguồn đã kiểm |
| CCSRA, SchedCP, HMS-SCP, LIF | [PDF] | Đo bằng mAP / AP; không báo hiện tượng proxy objective lệch khỏi chỉ số detection | Chưa thấy ai báo "cùng return, recall lệch 3–5 điểm" |

**Khoảng trống cho mình:** trong UAV-SemCom và task-oriented offloading, gói đánh giá gồm heuristic tune, baseline khớp bit, kiểm nhiễu detector, CI theo seed, và báo cáo lệch return–recall vẫn chưa thấy. Nhưng ở mức tổng quát nó chỉ là áp dụng chuẩn đã có (Agarwal 2021, Puffer). Đây là đóng góp về độ chặt chẽ, không phải thuật toán.

## 2. Điều gì KHÔNG phải tính mới

1. Dùng recall / mAP của detector làm reward hay làm chỉ số (CCSRA, SchedCP; SchedCP có cả bản không nhãn).
2. Chọn mode / mức nén cho từng ảnh dưới deadline, bằng RL hay bằng tối ưu (Huang, Zhao, xmodal, JOSC, PASC, A²-UAV).
3. Bộ dự đoán rẻ theo nội dung để chọn cấu hình nén (DAO, AdaCompress, SMR, DACC, ORIC). Bằng chứng của chính mình (G-L2) còn là âm.
4. Hiệu chỉnh độ tin cậy detector để định giá ngữ nghĩa (CCSRA).
5. MPC có mô hình throughput hiệu chỉnh online, có cổng sai số, giới hạn tính toán (RobustMPC, FastMPC, Oboe, Fugu). OLLA là hiệu chỉnh online ở PHY.
6. "RL không thắng heuristic tune / MPC tốt" như một phát hiện tổng quát (Oboe, Puffer; [AX2] có ví dụ Gale–Shapley, DP, RR thắng RL).
7. Ghép JA + pacing + MPC + mode cố định thành một stack: đây là kết hợp các bộ điều khiển đã biết.
8. Pacing / hover theo backlog (Huang có baseline hovering; IntUNe tối ưu tốc độ bay theo độ chính xác).
9. Lập lịch theo CSI + giá trị ngữ nghĩa, HARQ ngữ nghĩa, chọn nội dung theo độ tin cậy (SchedCP, HARQ, HyComm, Where2comm...). Mình cũng không làm collaborative perception, nên không mượn được các điểm này.
10. "Nhiều UAV + VisDrone + YOLO + kênh chung" (CCSRA).

## 3. Khoảng trống còn lại có thể bảo vệ được (tối đa 3)

**G1. Sai lệch mô hình tốc độ dưới LA thực tế, đo bằng mục tiêu nhiệm vụ trong offloading nhiều UAV có deadline.**
- *Giả định phân biệt:* mọi bài UAV-SemCom và CP đã kiểm (CCSRA pt. 21, SchedCP, Zhao pt. 6, Airtalking, Huang) ra quyết định trên dung lượng Shannon với CSI hoàn hảo hoặc đã biết. Hệ của mình quyết định khi goodput thực tế (MCS rời rạc, BLER, HARQ, CQI trễ, OLLA) lệch khỏi mô hình. Mình đo được: mô hình danh định mất 17 return; lợi thế của DRL biến mất khi có LA; JA vẫn bền.
- *Loại đóng góp:* mô hình hoá và đánh giá. Câu hỏi là "các kết luận của văn liệu UAV-SemCom (RL thắng, joint control có lợi) có còn đứng khi bỏ giả định Shannon không". Không phải thuật toán mới.
- *Bài đe doạ gần nhất:* RobustMPC / Fugu / Oboe (cơ chế hiệu chỉnh và MPC), StreamOptix (MPC + LA), CCSRA (bối cảnh).
- *Độ mạnh:* trung bình nếu đóng khung là kiểm định giả định. Yếu nếu claim cơ chế.

**G2. Gói đánh giá chặt cho task-oriented offloading, kèm phát hiện "return lệch recall".**
- *Giả định phân biệt:* trong phạm vi 24 + 19 bài đã khảo sát, không bài nào tune họ heuristic, khớp bit, kiểm nhiễu detector hay báo CI. Mình có thêm một phát hiện cụ thể: hai stack cùng return lệch 3–5 điểm recall. Tức là mục tiêu kiểu "giá trị đến hạn" mà văn liệu dùng không đủ để xếp hạng chính sách theo nhiệm vụ.
- *Bài đe doạ gần nhất:* Puffer (NSDI 2020) và Agarwal 2021 về phương pháp; Oboe về kết luận.
- *Độ mạnh:* trung bình, chỉ khi đi cùng G1 trong một bài kiểu "reality check". Đứng một mình thì yếu.

**G3. (Ứng viên yếu, chưa có bằng chứng) Hàm mục tiêu không nhãn, khớp recall, cho bộ lập lịch nhiều UAV có deadline.**
- *Giả định phân biệt:* ORIC, DAO, CCSRA ước lượng tiện ích cho **một quyết định đơn** (offload hay không, cấu hình nào) mà không có tranh chấp kênh hay deadline. Mình sẽ hỏi: proxy không nhãn có sửa được độ lệch return–recall ở cấp chính sách không.
- *Bài đe doạ gần nhất:* ORIC (SEC 2024), CCSRA, SchedCP (bản label-free).
- *Rủi ro lớn:* G-L2 cho thấy dư địa theo ảnh chỉ khoảng +1 recall, nên proxy có thể không có gì để sửa. Chỉ nên giữ nếu một phép thử rẻ cho thấy proxy xếp hạng chính sách giống recall hơn return.

Nói thẳng: **không có khoảng trống thuật toán nào được bảo vệ chắc chắn.** Mọi thành phần cơ chế đã có ở ABR, video analytics hoặc UAV-SemCom. Thứ còn lại là giả định (LA thay cho Shannon) và độ chặt của đánh giá.

## 4. Rủi ro

- **Reviewer quy về "RobustMPC cho UAV".** Phải trích Yin 2015, Pensieve, Oboe, Puffer ngay từ đầu, và nói rõ phần mới là bối cảnh nhiều người dùng + LA + deadline + tiện ích detection, không phải cơ chế.
- **"Shannon hay LA chỉ là chi tiết kỹ thuật."** Cần cho thấy kết luận **đổi chiều** khi đổi giả định: ví dụ RL thắng dưới Shannon nhưng không thắng dưới LA. Mình đã có bằng chứng từ A-X2 đến A-X9, nhưng phải trình bày thành một đối chứng sạch.
- **Quy mô nhỏ:** K = 5, 2 kênh, ảnh tĩnh VisDrone thay cho chuỗi bay thật, LA là mô phỏng. Mức lợi theo recall nhỏ.
- **Tìm kiếm không hệ thống:** khoảng 15 truy vấn; IEEE Xplore không đọc được. Venue của SchedCP, bài HARQ và IntUNe chưa xác minh. CCSRA và Zhao TNSE chỉ xác minh qua header PDF (tìm theo tên chính xác trên web không ra). Mọi câu "không thấy" chỉ đúng trong nguồn đã kiểm, không dùng để claim "đầu tiên".
- **Phạm vi của [AX2] / [A5]:** nhận xét "không bài nào tune heuristic" chỉ áp dụng cho các bài trong `pdf/`. Dòng ABR (Oboe, Puffer) đã làm việc đó, nên claim phải giới hạn ở UAV-SemCom.

## Nguồn web đã thấy

- SchedCP: https://arxiv.org/abs/2502.10456
- ORIC: https://arxiv.org/abs/2410.18919
- ASTRA: https://arxiv.org/pdf/2609.07020
- Yin et al. 2015, DOI 10.1145/2785956.2787486
- Pensieve: https://www.cs.princeton.edu/~ravian/publications/pensieve.pdf
- Oboe: https://engineering.purdue.edu/~isl/papers/sigcomm18-final128.pdf
- Puffer: https://www.microsoft.com/en-us/research/publication/puffer
- StreamOptix: https://arxiv.org/abs/2406.04632
- Agarwal et al.: https://arxiv.org/abs/2108.13264
- IntUNe: https://research.chalmers.se/publication/551901
- Các bài UAV-SemCom 2026: https://arxiv.org/abs/2602.00142, https://arxiv.org/abs/2602.10482, https://arxiv.org/html/2603.01478v1
