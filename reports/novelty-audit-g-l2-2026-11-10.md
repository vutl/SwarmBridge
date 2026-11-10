# Kiểm toán prior-art cho G-L2: bộ dự đoán rẻ trên UAV chọn mode codec theo từng ảnh

Ngày: 2026-11-10. Bổ sung cho `reports/archive/uav-semcom-a/novelty-audit-a5-l1-2026-11-10.md`, vốn chỉ xét PDF trong `pdf/`. Lần này tra web (arXiv, Semantic Scholar API, trang tác giả, bản PDF mở). Không đọc `runs/active_recovery/` hay bất cứ gì của UAV3D.

## 0. Kết luận ngắn

1. **Ý lõi của G-L2 đã có người làm, và khá sát.** Ý lõi ở đây là: một mô hình rẻ phía camera nhìn nội dung ảnh, dự đoán độ chính xác của mô hình phía server cho từng cấu hình nén, rồi chọn cấu hình theo từng ảnh hoặc đoạn ngắn.
   - **DAO** (ACM MM 2022) gần nhất. Một CNN nhẹ nền YOLO chạy trên Jetson, nhận khung khoá của mỗi đoạn 1 s và dự đoán mAP của server cho cả 32 cặp (bitrate, độ phân giải). Sau đó nó chọn cặp tốt nhất theo băng thông dự báo.
   - **AdaCompress** (ACM MM 2019) chọn mức JPEG q ∈ {5, 15, …, 95} **cho từng ảnh** bằng DQN trên đặc trưng MobileNetV2. Task là phân loại.
   - **SMR** (TPAMI 2024) và **JRD / DT-JRD** (IJCV 2021, TMM) dự đoán theo từng ảnh mức nén lớn nhất mà máy còn "chịu được", rồi đặt QP cho ảnh đó.
2. **Ba họ đặc trưng G-L2 định dùng đều có tiền lệ.**
   - Kích thước payload làm tín hiệu nội dung: CASVA.
   - Thống kê ảnh rẻ (entropy, số pixel biên) dự đoán lợi ích F1 của từng khung, trên VisDrone2019: DACC.
   - Mạng nhẹ hoặc detector nhẹ trên camera: DAO, AccMPEG, Vigil.
   - Mục tiêu "hiệu giữa hai mức nén" cũng đã có: SMR dùng nó làm task phụ.
3. **Ba paper reviewer nêu đều tồn tại, venue đúng.** Cả ba làm ở cấp **vùng/macroblock** (AccMPEG, RL-RC-DoT) hoặc **mask vùng + knob H.264** (AdaMask), không phải chọn codec toàn ảnh. Không bài nào mô hình kênh vô tuyến nhiều tác tử có deadline. AccMPEG có 5 luồng chia chung một link 2,5 Mbps, nhưng không lập lịch.
4. **Phần còn khác biệt, nếu G-L2 PASS**, gồm ba điểm, đều ở mức yếu đến trung bình:
   - menu toàn ảnh gồm nhiều họ codec, có codec học được và mode sinh ảnh;
   - đối chứng không nhìn nội dung mạnh hơn hẳn (best-fixed + trộn ngẫu nhiên + luật biết payload, khớp bit trung bình), cộng bất định theo cụm video và phép kiểm nhiễu detector;
   - nếu làm tiếp: gắn bộ dự đoán vào bài toán nhiều UAV tranh kênh có deadline.

   Riêng G-L2 PASS **không đủ** làm đóng góp chính cho một bài Q1.

## 1. Phạm vi và cách tìm

- **Nguồn đã dùng:** WebSearch, WebFetch, arXiv (abs, HTML, PDF), Semantic Scholar Graph API (batch theo DOI/arXiv), PDF mở trên trang tác giả. PDF tải về được trích bằng `pdftotext` rồi grep.
- **Truy vấn chính:**
  - tên ba paper reviewer nêu;
  - "per-image JPEG quality selection detection accuracy predictor";
  - "VCM QP prediction per image";
  - "UAV/drone video analytics configuration compression";
  - "multi-UAV semantic transmission detection compression mode shared channel deadline";
  - "per-frame accuracy predictor offloading";
  - "multi-camera bandwidth allocation accuracy";
  - các hệ kinh điển: DDS, AWStream, Chameleon, Reducto, Vigil, EAAR, Glimpse, Elf, JCAB, CASVA, OneAdapt.
- **Mức đọc (cột "Đọc" ở mục 3):**
  - **F** = đọc phần phương pháp và đánh giá từ toàn văn;
  - **P** = đọc một phần toàn văn (grep các mục then chốt);
  - **A** = chỉ đọc abstract hoặc metadata.

  Ở các bài A, chỉ ghi điều abstract nói.

## 2. Kiểm chứng ba paper reviewer nêu

### 2.1 AccMPEG: đúng (MLSys 2022, Du et al.). Đọc: F

- **Trích dẫn:** Kuntai Du, Qizheng Zhang, Anton Arapin, Haodong Wang, Zhengxu Xia, Junchen Jiang, "AccMPEG: Optimizing Video Encoding for Video Analytics", *MLSys 2022*. [arXiv:2204.12534](https://arxiv.org/abs/2204.12534), [PDF MLSys](https://proceedings.mlsys.org/paper_files/paper/2022/file/853f7b3615411c82a2ae439ab8c4c96e-Paper.pdf).
- **Dự đoán gì:** "accuracy gradient" (AccGrad) của **từng macroblock 16×16**, tức độ nhạy của độ chính xác DNN phía server với chất lượng mã hoá của khối đó.
  - AccModel là MobileNet-SSD feature extractor cộng 3 lớp conv, khoảng 12 GFLOPs, chạy trên camera. Nó phân loại nhị phân mỗi khối: chất lượng cao hay thấp.
  - Khối có AccGrad ≥ α = 0,2 được mã QP 30, còn lại QP 40. Vùng chọn được nới thêm 5 khối mỗi phía.
  - AccModel chạy 1 lần mỗi 10 khung.
- **Nhãn huấn luyện:** AccGrad được tính từ DNN cuối bằng 2 forward + 1 backward. "Độ chính xác" được đo **so với output của DNN trên ảnh chất lượng cao**, không dùng nhãn người.
- **Tín hiệu:** mô hình rẻ trên camera, không cần feedback server khi chạy. Phân biệt rõ với DDS/EAAR (server-driven).
- **Task và dữ liệu:**
  - detection: FasterRCNN, YOLOv5, EfficientDet;
  - segmentation: FCN;
  - keypoint;
  - 18 video YouTube (driving, dashcam, surf).

  Không có dữ liệu drone. Phần giới hạn tự ghi rằng AccModel có thể kém với **vật rất nhỏ như xe ở xa trong video drone**.
- **Mạng:** 5 luồng chia link 2,5 Mbps, trễ 100 ms. Không có kênh vô tuyến, không lập lịch, không có deadline từng khung.
- **Baseline:**
  - AWStream (một cấu hình cho cả cửa sổ thời gian);
  - DDS, EAAR (server-driven);
  - Reducto (lọc khung);
  - Vigil;
  - autoencoder.

  Kết quả: trễ thấp hơn 10–43 % ở độ chính xác tương đương.

### 2.2 RL-RC-DoT: đúng (CVPR 2025). Đọc: F, bản arXiv v2

- **Trích dẫn:** Uri Gadot, Assaf Shocher, Shie Mannor, Gal Chechik, Assaf Hallak, "RL-RC-DoT: A Block-level RL agent for Task-Aware Video Compression", *CVPR 2025*, DOI [10.1109/CVPR52734.2025.01169](https://doi.org/10.1109/CVPR52734.2025.01169). [arXiv:2501.12216](https://arxiv.org/abs/2501.12216). Venue xác nhận qua Semantic Scholar và file supplemental trên CVF open access.
- **Quyết định:** **delta QP theo macroblock** so với QP khung trong x264. Bản đồ hành động thô được nội suy lên.
  - MDP theo khung, PPO, có tính hệ quả dài hạn của QP (γ > 0). Bản γ = 0 kém hẳn: BD-rate −9,8 % so với −24,7 %.
- **Tín hiệu:**
  - chỉ thống kê x264 có sẵn: kích thước bitstream, QP, loại macroblock, năng lượng, intra cost, propagate cost, …;
  - **không** dùng đặc trưng của mô hình task lúc suy luận;
  - khoảng 4 ms mỗi quyết định.
- **Reward:** r = r_bitrate + 20·r_task.
  - r_task so output mô hình task trên khung gốc (giả nhãn) với output trên khung tái tạo;
  - r_bitrate phạt |log(bitrate / đích)|.
- **Task và dữ liệu:** phát hiện xe bằng YOLOv5-nano trên 172 luồng BDD100K (480×320). Có thêm thử SSD, DeepLab, DAVIS, ByteTrack.
- **Mạng:** không có kênh, mạng, deadline hay đa tác tử. Chỉ có bitrate đích 50–200 kbps.
- **Baseline:** chỉ rate control x264 mặc định và các ablation. Kết quả: BD-rate −24,7 % theo precision.

### 2.3 AdaMask: đúng venue (ACM MM 2022). Đọc: **A**, không lấy được toàn văn

- **Trích dẫn:** Shengzhong Liu, Tianshi Wang, Jinyang Li, Dachun Sun, Mani Srivastava, Tarek Abdelzaher, "AdaMask: Enabling Machine-Centric Video Streaming with Adaptive Frame Masking for DNN Inference Offloading", *Proc. ACM MM 2022*, tr. 3035–3044, DOI [10.1145/3503161.3548033](https://doi.org/10.1145/3503161.3548033) ([trang Illinois Experts](https://experts.illinois.edu/en/publications/adamask-enabling-machine-centric-video-streaming-with-adaptive-fr/)).
- **Theo abstract:**
  - che (mask) khung ảnh, chỉ giữ vùng **có thể** chứa vật quan tâm;
  - một thuật toán thích nghi tìm điểm Pareto giữa độ chính xác và băng thông, điều khiển **phần bị che cùng các knob H.264 như độ phân giải**, khi mạng biến động;
  - ba kịch bản: dashcam, camera giao thông, **drone**. Abstract không nêu tên dataset;
  - kết quả: giảm tới 65 % băng thông với mất < 1 % độ chính xác; +14 % độ chính xác so với baseline khi mạng biến động.
- **Không kiểm chứng được:**
  - vùng giữ lại do ai quyết (mô hình trên camera hay feedback server);
  - mức hạt thời gian;
  - có mô hình kênh vô tuyến hay deadline không.

  Lý do: ACM DL trả 403, NSF PAR purl trả 404, Semantic Scholar ghi "CLOSED". Vì vậy không được trích các chi tiết này.

**Nhận xét chung:** Reviewer trích đúng cả ba. Cả ba đều tối ưu **trong** một codec (QP vùng, mask vùng, knob H.264). Đó là trục quyết định khác với G-L2, vốn chọn **một codec/mode toàn ảnh** từ menu khác loại. Nhưng chính trục vùng/ROI có thể là đối thủ mạnh hơn mọi cách chọn mode toàn ảnh (xem mục 6).

## 3. Bảng các paper liên quan theo 4 trục

Bốn trục:
- **(a)** cái gì được gửi;
- **(b)** tín hiệu rẻ nào dự đoán tác động lên task;
- **(c)** lập lịch truyền thông: kênh, deadline, tranh chấp, nhiều nguồn gửi;
- **(d)** RL hoặc điều khiển học.

"—" nghĩa là không có hoặc không tìm thấy.

| Bài (venue) | Hạt quyết định | (a) Gửi gì | (b) Dự đoán tác động task | (c) Lập lịch truyền thông | (d) RL / học | Task, dữ liệu | Đọc |
|---|---|---|---|---|---|---|---|
| **AccMPEG** (MLSys'22) | macroblock, mỗi 10 khung | H.264, 2 mức QP | AccModel rẻ trên camera dự đoán AccGrad | 5 luồng chung link 2,5 Mbps; không lập lịch | supervised | det/seg/keypoint; YouTube | F |
| **RL-RC-DoT** (CVPR'25) | macroblock, mỗi khung | x264, delta QP | thống kê encoder, không đặc trưng task | — (chỉ bitrate đích) | PPO | xe, BDD100K | F |
| **AdaMask** (MM'22) | vùng + knob H.264 | khung đã che + H.264 | "vùng có thể chứa vật" (cơ chế chưa rõ) | băng thông biến động (theo abstract) | thuật toán Pareto (chưa rõ) | dashcam, giao thông, **drone** | A |
| **DAO** (MM'22) [doi](https://doi.org/10.1145/3503161.3548249), [pdf](https://mason.gmu.edu/~zyan4/papers/dao_mm22.pdf) | đoạn 1 s (khung khoá) | H.264, 32 cặp (bitrate, độ phân giải) | **CNN nền YOLO trên thiết bị dự đoán mAP cho cả 32 cấu hình** | 1 thiết bị, băng thông dự báo, chọn cấu hình tối ưu | supervised + tối ưu | SSD-500, YouTube-VOS | F |
| **AdaCompress** (MM'19) [doi](https://doi.org/10.1145/3343031.3350874), [arXiv](https://arxiv.org/abs/1909.08148) | **từng ảnh** | JPEG, 10 mức q | đặc trưng MobileNetV2 → Q(ảnh, q); độ chính xác so với output ở q = 75 | — | DQN (thực chất là contextual bandit) | phân loại, cloud API; ImageNet, DNIM | P |
| **SMR** (TPAMI'24) [doi](https://doi.org/10.1109/TPAMI.2024.3393633), [arXiv](https://arxiv.org/abs/2211.06797) | **từng ảnh** (intra) | HEVC/VVC, QP | EfficientNet-B4 Siamese (ảnh gốc + ảnh nén) dự đoán tỉ lệ máy hài lòng; **task phụ dự đoán hiệu SMR giữa 2 mức** | — | supervised | 72 classifier, 98 detector; COCO | P |
| **JRD** (IJCV'21) [doi](https://doi.org/10.1007/s11263-021-01505-4); **DT-JRD** (TMM) [arXiv](https://arxiv.org/abs/2411.09308) | từng ảnh / từng vật | QP | dự đoán ngưỡng méo lớn nhất máy còn nhận ra | — | supervised | det; DT-JRD báo −29,6 % bit so với VVC | A |
| **CASVA** (INFOCOM'22) [doi](https://doi.org/10.1109/INFOCOM48880.2022.9796875), [pdf](https://www2.cs.sfu.ca/~jcliu/Papers/casva22.pdf) | đoạn 2 s | H.264 (FR, QP, RS) | **tỉ lệ kích thước đoạn khi mã lại bằng cấu hình cũ = "content dynamics"** (giống payload probe) | 1 camera, trace mạng, buffer, lag, bỏ đoạn | PPO dual-clip | det (F1 so với cấu hình vàng), seg | P |
| **DACC** (ComNet 2026) [doi](https://doi.org/10.1016/j.comnet.2026.112130) | từng khung | khung gửi lên cloud hay xử lý ở edge | **Random Forest trên std optical flow, entropy, số pixel biên → lợi ích F1** | Lyapunov chia tỉ lệ khung theo băng thông/tải | RF + Lyapunov | UA-DETRAC, **VisDrone2019**, YOLOv7 / Edge-YOLO | P |
| **Reducto** (SIGCOMM'20) [doi](https://doi.org/10.1145/3387514.3405874) | đoạn | lọc khung | đặc trưng sai khác rẻ (pixel/area/edge) + hash table K-means do server học → ngưỡng | — | profiling + clustering | đếm, bbox; camera giao thông | P |
| **DDS** (SIGCOMM'20) [doi](https://doi.org/10.1145/3387514.3405887) | vùng | luồng chất lượng thấp + gửi lại vùng chất lượng cao | **feedback DNN server** | băng thông, 2 vòng | — | det, seg | A (+ mô tả trong AccMPEG) |
| **AWStream** (SIGCOMM'18) [doi](https://doi.org/10.1145/3230543.3230554) | cửa sổ thời gian | knob ứng dụng | profile offline + online (accuracy–bandwidth) | khớp data rate với băng thông WAN | — | AR, pedestrian det | A |
| **Chameleon** (SIGCOMM'18) [doi](https://doi.org/10.1145/3230543.3230574) | định kỳ, nhiều camera | độ phân giải, frame rate, model | profiling thưa, tận dụng tương quan thời gian và giữa camera | (tài nguyên tính toán) | tìm kiếm có cấu trúc | camera giao thông | A |
| **OneAdapt** (SoCC'23) [doi](https://doi.org/10.1145/3620678.3624653) | định kỳ | knob cấu hình | AccGrad theo knob, tính bằng backprop | băng thông, GPU | gradient ascent | 4 task, 5 loại dữ liệu | A |
| **Vigil** (MobiCom'15) [doi](https://doi.org/10.1145/2789168.2790123) | khung | khung ưu tiên | xử lý ở edge cạnh camera, xếp hạng khung | **nhiều camera, phổ vô tuyến khan hiếm, lập lịch lưu lượng** | heuristic | giám sát | A |
| **JCAB** (INFOCOM'20) [doi](https://doi.org/10.1109/INFOCOM41043.2020.9155524) | slot, mỗi luồng | FR, độ phân giải | mô hình độ chính xác theo cấu hình | **nhiều luồng chia uplink, năng lượng, trễ** | Lyapunov online | edge analytics | A |
| **DeepStream** (arXiv 2023) [arXiv](https://arxiv.org/abs/2306.15129) | khung / camera | ROI (ROIDet trên Raspberry Pi) | ROI rẻ, phân bổ băng thông "content-aware" | nhiều camera tranh băng thông | DP | det | A |
| **A²-UAV** (INFOCOM'23) [doi](https://doi.org/10.1109/INFOCOM53939.2023.10229096) | theo mục tiêu | JPEG nhiều mức | **bảng Q(kịch bản/lớp, mức nén)**, không theo nội dung từng ảnh | **nhiều UAV multi-hop, định tuyến, năng lượng, deadline** | tối ưu tổ hợp (NP-hard + greedy) | 4 classifier + YOLOv4; testbed 4 DJI | P |
| **DCC** (arXiv 2021) [arXiv](https://arxiv.org/abs/2111.06263) | khung / vùng | knob codec trên drone | ngữ cảnh drone + gợi ý từ detection | drone → edge | — | phát hiện xe | A |
| **Kang et al.** (IEEE TCOM) [arXiv](https://arxiv.org/abs/2112.10948) | ảnh | khối ngữ nghĩa | mô hình nhẹ trên UAV, theo ảnh và kênh | kênh đơn UAV | DRL | phân loại cảnh trên không | A |
| **Hybrid bit + semantic, UAV-WPT** (JSAC, accepted 2026) [arXiv](https://arxiv.org/abs/2606.00668) | slot | mode bit / semantic | metric "semantic efficiency" (chưa rõ) | quỹ đạo, ghép user, thời gian thu năng lượng | DSAC | — | A |
| *Từ audit trước:* CCSRA, SchedCP, Huang JSAC | — | — | — | — | — | — | xem audit A5/L1 |

Ghi chú: EAAR (MobiCom'19, [doi](https://doi.org/10.1145/3300061.3300116)), Glimpse (SenSys'15, [doi](https://doi.org/10.1145/2809695.2809711)), Elf (MobiCom'21, [doi](https://doi.org/10.1145/3447993.3448628)) và AdaScale (MLSys'19, [arXiv](https://arxiv.org/abs/1902.02910)) chỉ đọc abstract. Chúng xử lý vùng, khung kích hoạt, phân mảnh offload hoặc chọn scale đầu vào, xa G-L2 hơn nên không đưa vào bảng.

## 4. Đối chiếu với G-L2

Mỗi thành phần G-L2 (theo `reports/uav-semcom-g-l2.md` §0.5) đối chiếu với prior art gần nhất:

| Thành phần G-L2 | Prior art gần nhất | Mức trùng |
|---|---|---|
| Bộ dự đoán rẻ chạy trước khi gửi, chọn cấu hình nén **theo từng ảnh** để tối đa độ chính xác detector ở mức bit cho trước | **DAO** (mAP cho 32 cấu hình từ khung khoá, chọn theo băng thông); **AdaCompress** (q JPEG theo ảnh); **SMR/JRD** (QP theo ảnh) | **Cao.** Khác ở menu và hạt thời gian: DAO theo đoạn 1 s; AdaCompress là phân loại |
| Đặc trưng A: payload / probe | CASVA (tỉ lệ kích thước khi mã lại); RL-RC-DoT (kích thước bitstream, thống kê encoder) | Cao |
| Đặc trưng B: thống kê ảnh rẻ (biên, entropy, phổ) | DACC (entropy, số pixel biên, optical flow → lợi ích F1, trên VisDrone2019); Reducto (pixel/edge/area) | Cao. DACC dự đoán lợi ích offload, không dự đoán mode codec |
| Đặc trưng C: detector độ phân giải thấp trên UAV | DAO (nền YOLO); AccMPEG (MobileNet-SSD); Vigil; DeepStream (ROIDet) | Cao |
| Mục tiêu "pair": hiệu TP giữa 2 mode | SMR: task phụ dự đoán hiệu SMR giữa hai mức chất lượng | Trung bình. Cùng ý, khác đại lượng |
| Đối chứng không nhìn nội dung (F, MX, RV, CB) khớp bit trung bình; chọn BL thuận lợi cho đối chứng | AccMPEG (AWStream đồng nhất); DAO (cấu hình hằng, DASH); AdaCompress (q = 75 cố định); SMR (BD-rate so với QP hằng) | **Thấp.** Không thấy bài nào dùng trộn ngẫu nhiên trên bao lồi hay luật biết payload làm đối chứng ở cùng bit trung bình. SMR, với đường QP hằng, gần nhất |
| CI bootstrap cụm video; oracle kiểm chéo bằng ảnh lật để đo nhiễu detector | — | Không tìm thấy trong các nguồn đã kiểm tra |
| Nhãn: TP so với ground truth | Phần lớn prior art dùng output DNN trên bản chất lượng cao làm giả nhãn (AccMPEG, RL-RC-DoT, CASVA, AdaCompress, SMR); DAO dùng mAP | Khác, nhưng chỉ là chi tiết |
| Menu toàn ảnh trộn AVIF / mbt2018 / JPEG2000 / mode sinh ảnh | Huang JSAC 2025 (JPEG2000 + generative, chấm bằng PSNR); prior art phía analytics chỉ chỉnh knob của **một** codec | Thấp đến trung bình |
| Nhiều UAV, kênh chung, deadline | A²-UAV, JCAB, Vigil, DeepStream, CCSRA, SchedCP. **G-L2 hiện không mô phỏng mạng** | Không áp dụng cho G-L2. Chỉ thành điểm khác khi nối vào A5 |

**Trả lời (i).** Có, "bộ dự đoán rẻ trên thiết bị chọn cấu hình nén theo nội dung để tối đa độ chính xác detector ở băng thông cho trước" đã được làm:
- DAO, ACM MM 2022, là bản gần nhất: detection, mạng nhẹ trên thiết bị, dự đoán độ chính xác cho mọi cấu hình;
- AdaCompress, MM 2019, làm theo từng ảnh, dùng RL;
- SMR (TPAMI 2024) và JRD (IJCV 2021) làm theo từng ảnh, chọn QP, có dự đoán hiệu giữa hai mức.

Ở cấp vùng có AccMPEG, RL-RC-DoT, AdaMask, DDS. Thống kê ảnh rẻ dự đoán lợi ích detection theo khung trên VisDrone có ở DACC 2026. Không nên trình bày G-L2 như một ý tưởng mới. Nó là phép kiểm định lại ý đó trong setting của ta.

## 5. Điều còn khác biệt và độ mạnh (đánh giá thẳng)

1. **Đối chứng content-agnostic mạnh, khớp bit trung bình, có bất định theo cụm video và kiểm nhiễu detector.** Độ mạnh: **trung bình nếu kết quả là âm hoặc nhỏ, yếu nếu dương.**
   - Prior art so với cấu hình hằng hoặc mặc định. SMR là ngoại lệ một phần: đo BD-rate so với cả đường QP hằng.
   - Nếu G-L2 cho thấy phần lớn "lợi ích chọn theo nội dung" biến mất khi so với trộn mode và luật biết payload, đó là phát hiện phản biện có giá trị cho cả dòng này. Nó cùng tinh thần với kết luận A5/A5.5 (RL ≈ heuristic tune).
   - Nếu dương, đó chỉ là "DAO / AdaCompress trên menu khác, đánh giá cẩn thận hơn".
2. **Menu toàn ảnh nhiều họ codec, gồm codec học được và mode sinh ảnh, chấm bằng detector.** Độ mạnh: **yếu đến trung bình.**
   - Prior art analytics chỉnh knob trong một codec. Huang có menu lai nhưng chấm bằng PSNR.
   - Tuy vậy, chọn giữa các codec chỉ là mở rộng không gian hành động. Reviewer dễ hỏi vì sao không chọn ROI hay QP vùng (mục 6). L1 cũng đã cho thấy mode sinh ảnh có giá trị task âm, nên trong G-L2 chúng gần như không bao giờ được chọn.
3. **Gắn bộ dự đoán vào nhiều UAV tranh kênh, deadline từng khung, UAV di chuyển (A5).** Độ mạnh: **trung bình, nhưng chưa có.**
   - Từng mảnh đều đã có: JCAB và Vigil cho nhiều nguồn chia uplink; A²-UAV cho nhiều UAV có deadline, với bảng accuracy theo lớp chứ không theo ảnh; CASVA và DAO cho chọn cấu hình theo nội dung + mạng nhưng chỉ một nguồn.
   - Tổ hợp "dự đoán giá trị task theo nội dung từng khung + phân bổ kênh chung nhiều UAV + frame trễ hạn tính là mất" không tìm thấy trong các nguồn đã kiểm tra.
   - Giá trị chỉ có nếu đo được tương tác: ví dụ predictor giúp **lập lịch** (UAV nào gửi, gửi gì khi kênh tắc) nhiều hơn là giúp chọn mode đơn lẻ.
4. **Dữ liệu drone với vật nhỏ.** Độ mạnh: **yếu.** AdaMask có kịch bản drone, DACC dùng VisDrone2019, A²-UAV và DCC nhắm drone. AccMPEG còn tự nêu vật nhỏ trong video drone là ca thất bại; có thể trích để biện minh, nhưng không phải điểm mới.

**Cho bài Q1:**
- G-L2 PASS đứng một mình là đóng góp gia tăng, khó qua reviewer biết DAO, AdaCompress, SMR.
- Khả thi hơn là đặt G-L2 làm một chương trong bài **đánh giá có kiểm soát**, gom A5/A5.5/L1/G-L2 thành một câu: "trong hệ nhiều UAV có deadline, các lớp thích nghi (RL, reward theo task, chọn mode theo nội dung) thực sự thêm bao nhiêu so với đối chứng mạnh".
- Hoặc dùng G-L2 làm đầu vào cho một bài toán lập lịch nhiều UAV, ở đó content-awareness tạo khác biệt đo được.
- Nếu G-L2 FAIL, kết quả âm đó vẫn dùng được trong khung "đánh giá có kiểm soát", với điều kiện có các baseline của mục 6.

## 6. Baseline nên thêm hoặc ít nhất phải bàn

1. **Bản cấp khung tương tự AccMPEG.** AccMPEG theo vùng, còn ta theo khung. Bản tương đương công bằng như sau:
   - Train một mạng nhẹ (MobileNet / backbone YOLO cắt ngắn, chạy trên ảnh thu nhỏ) dự đoán **trực tiếp** D_i(m) = TP_i(m) − TP_i(m_ref) cho mọi m. Đây là đầu ra nhiều chiều kiểu DAO.
   - Hoặc một biến thể gần AccMPEG hơn: dự đoán bản đồ AccGrad theo khối giữa hai mode, rồi gộp (tổng hay tỉ lệ khối nhạy) thành điểm cấp khung.

   Đặc trưng C của G-L2 (thống kê từ output detector độ phân giải thấp) yếu hơn một CNN học end-to-end. Nếu không làm, phải ghi rõ đây là giới hạn.
2. **DAO-style:** CNN trên thumbnail → recall/TP dự đoán cho 14 mode → argmax có λ·bit. Thực chất là ứng viên "abs" với đặc trưng học được. Nên thêm như ứng viên thứ 17 hoặc làm phân tích phụ, chi phí thấp.
3. **AdaCompress-style:** hồi quy trên embedding cố định của backbone ImageNet (MobileNetV2) thành Q(ảnh, mode). Dễ thêm như bộ đặc trưng "D".
4. **CASVA / DACC / Reducto:** đã có trong A và B. Chỉ cần trích dẫn rõ và dùng mô hình cây tương tự (DACC dùng Random Forest).
5. **Thích nghi theo video hoặc theo thời gian (server feedback kiểu Chameleon / AWStream / DDS / OneAdapt).** Đây là baseline quan trọng nhất còn thiếu.
   - Rất có thể phần lớn dư địa oracle nằm ở khác biệt **giữa video** chứ không **giữa khung**.
   - Nên thêm: (a) oracle "mode cố định tốt nhất theo từng video"; (b) chính sách nhân quả chọn mode cho khung t dựa trên kết quả detector đã đo ở các khung trước của cùng video (bản label-free: độ khớp output giữa mode thấp và mode cao, như giả nhãn của AccMPEG/CASVA).
   - Nếu predictor theo khung không hơn baseline này, nên kết luận rằng nội dung cấp video, chứ không phải cấp khung, mới là thông tin hữu ích.
6. **ROI / QP vùng trong một codec** (AccMPEG, RL-RC-DoT, AdaMask, DDS). Không bắt buộc cài đặt trong G-L2, vì G-L2 chỉ hỏi về menu toàn ảnh. Nhưng phần thảo luận phải nói thẳng: đây là trục mạnh hơn, đã báo −25..−43 % bit hoặc trễ. Nếu muốn so, một baseline rẻ là AVIF hai mức chất lượng theo box của detector nhẹ (kiểu DDS/AccMPEG đơn giản).
7. **So sánh kiểu BD-rate** (như SMR) ngoài ba mức bit B1–B3, để reviewer phía VCM đọc được.

## 7. Giới hạn của audit

- **AdaMask:** chỉ đọc abstract và metadata (ACM DL 403, NSF PAR 404, Semantic Scholar ghi closed access). Không biết cơ chế mask, mức hạt, mô hình mạng.
- **Chỉ đọc abstract hoặc metadata:** DDS, AWStream, Chameleon, OneAdapt, Vigil, JCAB, EAAR, Glimpse, Elf, AdaScale, DeepStream, DCC, Kang et al., JRD, DT-JRD, và bài hybrid bit + semantic JSAC 2026. Các nhận xét về chúng chỉ dựa vào abstract. Năm và số tập của Kang et al. (TCOM) chưa kiểm.
- **Đọc một phần toàn văn:** AccMPEG, RL-RC-DoT (bản arXiv v2, không phải bản CVF cuối), DAO, CASVA, AdaCompress, Reducto, DACC, A²-UAV, SMR (HTML, khoảng 70 % đầu).
- **Truy vấn không ra bài khớp trực tiếp:**
  - "lightweight predictor per-image JPEG quality factor selection preserve object detection accuracy";
  - "UAV image transmission DRL compression quality selection detection VisDrone";
  - "multi-UAV semantic image transmission object detection compression mode selection shared channel deadline RL YOLO";
  - "VCM per-image QP prediction lightweight network".

  Không ra kết quả không có nghĩa là không tồn tại.
- **Hạn chế kỹ thuật:** IEEE Xplore, ACM DL, ScienceDirect và DBLP chặn truy cập tự động. Semantic Scholar bị giới hạn tần suất (429) ở lượt cuối, nên CloudSeg, Concierge và bản edge của AdaCompress (TOMM 2021) chưa kiểm. WebSearch chỉ có chỉ mục Mỹ. Không tra venue tiếng Trung hay patent.
- **Cách viết:** Mọi nhận định "không có" trong báo cáo này nên hiểu là "không tìm thấy trong các nguồn đã kiểm tra". Không dùng để claim "đầu tiên".
