# SwarmBridge R0: thiết kế kịch bản và biểu diễn ngữ nghĩa cho bước tiếp theo

Ngày 10/11/2026. Đây là **báo cáo thiết kế**, chưa có code, chưa train, chưa chạy GPU.
- Không đọc, sửa hay chạy UAV3D hoặc Active Semantic Recovery (ASR). Chồng lấn với hai nhánh đó chỉ ghi nhận từ thông tin đã có.
- Ghi chú đọc paper (tiếng Anh, có số trang) do ba sub-agent viết: một nhóm paper được nêu tên, một nhóm collaborative perception và bầy UAV, một nhóm biểu diễn (token, ROI, split computing).
- Tôi tổng hợp và kiểm lại các chi tiết then chốt.
- Mức đọc ghi ở mục 2: F = đọc toàn văn phần phương pháp và đánh giá; P = đọc một phần toàn văn; A = abstract; K = kiến thức nền, chưa đọc lại lần này.
- Mọi câu "không tìm thấy" nghĩa là **không tìm thấy trong các nguồn đã kiểm**, không phải "chưa ai làm".

## 0. Tóm tắt

**Đề xuất ưu tiên: ứng viên C, "báo cáo ngữ nghĩa theo hiểu biết của trạm".**
- Mỗi UAV giữ một **bản sao những gì trạm (BS) đã biết** (dựng lại từ các xác nhận ACK).
- Ở mỗi khung video, UAV chọn: không gửi, gửi **token đối tượng** từ detector yếu trên UAV, gửi **ảnh nền chất lượng thấp kèm vùng chi tiết (ROI)**, hoặc gửi **ảnh**.
- Lựa chọn dựa vào **độ lệch giữa điều UAV thấy và điều BS đang tin**, dưới ràng buộc tính toán trên UAV, kênh chung và deadline.
- Mục tiêu nhiệm vụ đo ở BS: độ trễ báo lần đầu của mỗi đối tượng, và tỉ lệ thời gian BS biết đúng đối tượng đang có.

**Lý do khoa học:**
- **G-L2 cho thấy giá trị tác vụ theo nội dung từng ảnh hầu như không đoán được.** Chênh lệch giữa các codec toàn ảnh nhỏ và không ổn định.
- **Ứng viên C đặt quyết định lên hai thứ quan sát được và đoán được:**
  - **trạng thái hiểu biết của BS**: biết qua ACK, và có bằng chứng nội bộ từ Q2/step 1 rằng sender replica rất mạnh;
  - **tính liên tục của đối tượng qua thời gian.**
- **Các lựa chọn khác nhau về bản chất:**
  - token rẻ hơn ảnh khoảng 100 lần, nhưng thừa hưởng mọi lần bỏ sót của detector trên UAV;
  - ảnh để BS phát hiện lại được vật nhỏ.
- **Bước bầy UAV đi ra tự nhiên:** cùng cơ chế "BS đã biết gì" áp dụng cho nhiều UAV dùng chung kênh. Khi có dữ liệu chồng lấn, nó bao luôn ứng viên B (tránh báo trùng giữa các UAV).

**Kiểm chứng theo bậc**, mỗi bậc có cổng PASS/FAIL định trước, chưa có RL:
- **R1a** (tĩnh, dữ liệu VisDrone-DET có sẵn): các họ biểu diễn có thật sự mỗi họ thắng ở một vùng ngân sách bit không?
- **R1b** (video VisDrone-VID, phát lại offline, một UAV): cơ chế bản sao BS có tiết kiệm bit thật so với baseline kích hoạt/định kỳ đã tinh chỉnh không?
- **R2:** mới đưa vào simulator A5 với 5 UAV, dùng heuristic mạnh trước.

**Độ mới, nói thẳng:** trung bình.
- Từng mảnh đều đã có: Glimpse/MARLIN (gửi khung khi cảnh đổi, theo vết trên thiết bị), EdgeDuet (token + ô ảnh trên VisDrone), Qiu 2024 (offload khi detector yếu có thể bỏ sót), GOSC (giá trị thông điệp theo tri thức chung, nhưng cảm biến trừu tượng), CCSRA (token nhiều UAV trên VisDrone).
- Tổ hợp đầy đủ không tìm thấy trong các nguồn đã kiểm: chọn giữa các họ biểu diễn, có mô hình bỏ sót của detector thật, có trạng thái hiểu biết của BS, kênh chung, deadline, mục tiêu nhiệm vụ đo bằng detector thật trên video UAV.
- Nếu R1a hoặc R1b FAIL, phải dừng, không gượng ép (mục 8).

## 1. Bài học từ A5 → G-L2, dùng làm tiêu chí thiết kế

1. **Các lựa chọn phải khác nhau về bản chất, không chỉ về mức nén.** 14 mode codec toàn ảnh đều cố tái tạo cả ảnh. Detector bão hoà ở chất lượng vừa, nên chênh lệch nhiệm vụ giữa chúng nhỏ.
2. **Trạng thái quyết định phải quan sát được và đoán được một cách nhân quả.** G-L2 cho thấy predictor dựa trên nội dung ảnh thua một luật biết payload, kể cả bản payload dự đoán từ probe. Trạng thái mạng (A5.5: nghẽn, lệch PHY) thì đoán được, và đó là nơi PPO có ích.
3. **Đối chứng mạnh, khớp chi phí, khoá điểm hoạt động trên seed riêng, CI theo video, kiểm độ ổn định.** GOSC dùng cùng chuẩn này (họ baseline định kỳ đã tinh chỉnh, bootstrap ghép cặp). Họ cũng thấy "ước lượng giá trị thô + lập lịch biết kênh ≈ ước lượng chính xác", trùng với G-L2.
4. **Báo riêng chỉ số tuân thủ (Qmin) và chỉ số nhiệm vụ thật (giao đúng hạn).**
5. **Không thiết kế kịch bản để RL thắng.** Câu hỏi là quyết định thông minh có giúp nhiệm vụ không; heuristic hay RL chỉ là cách giải.

## 2. Ma trận novelty

Y = có; p = một phần; – = không tìm thấy trong phần đã đọc.

| Công trình | Đọc | ≥ 2 họ biểu diễn trong **một** quyết định | Mô hình bỏ sót của detector trên thiết bị | Tính toán trên thiết bị tính vào deadline | Nhiều nguồn, kênh chung + hàng đợi | Mục tiêu nhiệm vụ ở BS | Hiểu biết của bên nhận / tri thức chung | Trạng thái theo thời gian (video) | Detector thật trên ảnh UAV |
|---|---|---|---|---|---|---|---|---|---|
| Huang JSAC'25 (bit + sinh ảnh) | F | **Y** (bit / caption / caption + latent; chấm PSNR) | – | p (hằng số mỗi mode) | p | – | – | – | – |
| Zhao TNSE'26 | F | p (nhãn / ảnh / feature là các **sơ đồ riêng**) | – | **Y** (τ_comp theo FLOPs) | – (1 UAV) | – | – | – | – (bảng tra, phân loại) |
| CCSRA JWCN'26 | F | – (chỉ token) | p (ẩn qua ưu tiên theo detection) | – | **Y** (OFDMA, nhiễu; gửi một lần, không hàng đợi) | – | p (bộ nhớ dư thừa) | – | **Y** (YOLOv8-n, VisDrone) |
| A²-UAV INFOCOM'23 | P | – (JPEG nhiều mức) | – | – | **Y** (multi-hop) | **Y** (đúng và đúng hạn) | – | – | p (bảng theo lớp) |
| DAO MM'22 | P | – | – | – (đo, không tính) | – | – | – | p | Y (không phải UAV) |
| AccMPEG MLSys'22 | P | – (ROI trong một codec) | **Y** (lập luận + baseline Vigil) | **Y** | p | – | – | p | Y |
| EdgeDuet INFOCOM'21 | F | p (token vật lớn + ô ảnh vật nhỏ, **pipeline cố định**) | **Y** (ngưỡng kích thước theo lớp) | – | – (1 thiết bị) | – | – | Y | **Y** (VisDrone video) |
| Qiu arXiv'24 (2410.18919) | F | p (cục bộ vs offload) | **Y** (6 loại lỗi, có bỏ sót) | – | – (1 thiết bị, có dây) | – | – | – | p |
| CoTeD (bản hội nghị 2024) | F | – (một điểm cắt cố định) | – | **Y** (deadline 30 ms) | – (1 UAV) | – | – | p | p |
| He/Quek arXiv'23 (2312.01049) | F | – (độ rộng mạng × số kênh feature) | – | Y | Y | – | – | – | – (công thức giả định) |
| GOSC arXiv'26 | F | – (cố định mỗi sơ đồ) | **Y** (Bernoulli p_d, p_fa) | – | **Y** (ngân sách chung, hàng đợi) | **Y** (thời gian xong, số người cứu) | **Y** (tri thức chung phát quảng bá) | Y | – (lưới trừu tượng) |
| CodeFilling CVPR'24 | P | – | – | – | p (ngân sách lý tưởng) | – | **Y** (lấp đầy nhu cầu bên nhận) | – | – (V2X mô phỏng) |
| Where2comm / CoSDH / How2comm | F/P | p (CoSDH: feature + box) | – | – | – (ngân sách lý tưởng) | – | **Y** (bản đồ yêu cầu / cung–cầu) | p | mô phỏng |
| Glimpse SenSys'15, MARLIN SenSys'19 | K | p (khung kích hoạt vs theo vết cục bộ) | p | **Y** | – | – | p (ngầm) | **Y** | – |
| Reducto SIGCOMM'20 | K | – (lọc khung) | – | Y | – | – | – | **Y** | – |
| Mazandarani arXiv'26 (2606.31715) | A | ? (V2V ước lượng chồng lấn, gửi mảnh không trùng lên BS) | ? | ? | ? | ? | ? | ? | ? |
| **Ứng viên A** (mục 3) | — | **Y** | **Y** (đo thật) | Y | Y (ở R2) | Y | – | – | **Y** |
| **Ứng viên B** (mục 4) | — | p | Y | Y | **Y** | **Y** (đối tượng duy nhất đúng hạn) | **Y** (giữa các UAV) | p | **Y** (cần dữ liệu nhiều drone) |
| **Ứng viên C** (mục 5) | — | **Y** | **Y** | **Y** (ràng buộc ở tốc độ video) | **Y** (ở R2) | **Y** | **Y** (bản sao hiểu biết của BS) | **Y** | **Y** |

**Đọc ma trận:**
- Cột "≥ 2 họ biểu diễn trong một quyết định" chỉ có Huang và Hu TCOM'25 (ảnh vs text), cả hai chấm bằng PSNR/SSIM/BLEU, không có detector trên thiết bị.
- Cột "bỏ sót của detector trên thiết bị" có EdgeDuet, Qiu, AccMPEG, GOSC, nhưng không bài nào đặt nó trong một quyết định có ràng buộc tài nguyên **và** nhiều nguồn chung kênh.
- Mazandarani mới đọc abstract; phải đọc trước khi chọn B.

## 3. Ứng viên A: chọn họ biểu diễn theo khung, với detector trên UAV có giới hạn

**1. Nhiệm vụ và một episode.**
- Như A5: 5 UAV khảo sát, 2 kênh 1 MHz dùng chung, 1 BS, deadline mỗi khung U(4, 12) s, ảnh VisDrone-DET.
- BS cần danh sách đối tượng (người, xe…) của mọi khung đã chụp, đúng hạn.
- Chỉ số chính: **recall_ontime** và AP50, báo riêng recall vật nhỏ.

**2. Mô hình hệ thống.**
- **Trạng thái:** như A5 (vị trí, hàng đợi, SNR kỳ vọng, slack), thêm tải tính toán trên UAV (hàng đợi suy luận) và thống kê của detector trên UAV (số box, phân bố confidence, số box nhỏ).
- **Hành động:** họ biểu diễn và tham số của nó; slot (kênh, công suất) như A5.
- **Thời gian:** suy luận trên UAV + mã hoá + truyền + giải mã/suy luận ở BS, tất cả tính vào deadline.
- **Năng lượng:** thêm năng lượng tính toán.

**3. Biểu diễn được gửi:**

| Họ | UAV làm gì | BS làm gì | Bit/khung (ước lượng, kiểm ở R1a) |
|---|---|---|---|
| **IMG(m)** | mã hoá ảnh (14 mode A5) | YOLO11s phát hiện lại | 0,03–1,4 Mbit |
| **TOK(τ)** | detector nhỏ (ví dụ YOLO11n), gửi (lớp, box lượng tử, conf) các box có conf ≥ τ | dùng box trực tiếp | khoảng 64 + 52 bit/box, tức vài kbit |
| **HYB(τ, m_bg)** | token + ảnh nền rất rẻ (avif_5 / thumbnail) | phát hiện lại trên ảnh nền, gộp với token (NMS) | khoảng 0,05–0,1 Mbit |
| **ROI(τ)** | token cho box chắc chắn + ô ảnh chất lượng cao quanh vùng detector nhỏ không chắc / cụm vật nhỏ | phát hiện lại trên ô ảnh, gộp | tuỳ số ô |
| FEAT | feature trung gian (split computing) | head ở BS | **không đưa vào R1**: cần train codec feature; prior art dày (CoTeD, Shao/Zhang, He/Quek) |

**4. Vì sao không họ nào luôn thắng (giả thuyết cần kiểm).**
- TOK rẻ hơn khoảng 100 lần nhưng mất các vật mà detector nhỏ bỏ sót, nhất là vật nhỏ (AccMPEG và EdgeDuet đều nêu cơ chế này).
- IMG giữ được vật nhỏ nhưng đắt.
- Khi kênh nghẽn hoặc deadline gắt, TOK/HYB giao được đúng hạn còn IMG thì không. Khi kênh rảnh, IMG thắng.
- Ranh giới phụ thuộc trạng thái mạng: đoán được, như ở A5.5. Nó còn có thể phụ thuộc thống kê detector trên UAV, nhưng điều này phải kiểm vì G-L2 cảnh báo phần theo nội dung có thể không ổn định.

**5. Baseline mạnh nhất.**
- Từng họ cố định với tham số tinh chỉnh.
- RV_probe (luật biết payload, G-L2 7.1).
- Luật ngưỡng confidence kiểu Qiu / DeepDecision (gửi ảnh khi detector nhỏ không chắc).
- Luật ngưỡng kích thước kiểu EdgeDuet.
- Theo ngân sách: chọn họ tốt nhất ở mỗi mức bit (bao tĩnh).
- Heuristic A5 đã tinh chỉnh cho phần slot.

**6. Dữ liệu và tài sản tái dùng.**
- Dùng lại: ảnh VisDrone-DET và cách chia của A5, detector A5 ở BS, bảng fidelity IMG(m) đã đo, mã đánh giá khớp bit của G-L2 (`g2_common`), AP theo policy (`a6l1_ap`), simulator A5.
- Xây mới:
  - train detector nhỏ trên VisDrone-DET-train, cùng quy tắc loại chuỗi trùng như A5 (khoảng 1–2 giờ GPU);
  - codec token;
  - ghép ROI/HYB ở BS;
  - đo latency CPU khi máy rảnh, kèm giả định quy đổi sang phần cứng UAV (nói rõ).

**7. Đóng góp khả dĩ và rủi ro novelty.**
- Đóng góp: bản đồ thực nghiệm khi nào token / hybrid / ảnh thắng trên ảnh UAV thật với detector thật, ở cùng chi phí bit và tính toán, có mô hình bỏ sót. Kèm đối chứng mạnh.
- Rủi ro: Qiu (bỏ sót + offload), EdgeDuet (hybrid trên VisDrone), CCSRA (token nhiều UAV), DeepDecision/FastVA (cục bộ vs offload).
- Riêng A ở dạng tĩnh, độ mới **thấp đến trung bình**. Nó mạnh hơn khi là nền cho C.

**8. Pilot không cần RL: R1a, mục 7.1.**

**9. PASS/FAIL: mục 7.1.**

## 4. Ứng viên B: báo cáo hợp tác biết chồng lấn giữa các UAV

**1. Nhiệm vụ và một episode.**
- 2–5 UAV có trường nhìn chồng lấn quan sát cùng khu vực, gửi về 1 BS qua kênh chung.
- BS cần **danh sách đối tượng duy nhất** (khử trùng theo danh tính thật) trước deadline.
- Chỉ số: recall đối tượng duy nhất đúng hạn, thời gian phát sóng bị lãng phí cho báo cáo trùng.

**2. Mô hình hệ thống.**
- Như A, thêm đồng bộ thời gian giữa các UAV và thông điệp phối hợp: BS phát quảng bá "đã biết", hoặc UAV trao đổi trực tiếp, có tính chi phí.
- **Hành động:** ai báo đối tượng/vùng nào, bằng họ biểu diễn nào. Đối tượng ở vùng chồng lấn có thêm một nhánh: nhờ UAV thứ hai nhìn lại thay vì gửi lại chất lượng cao (scenario iii trong ghi chú).

**3. Biểu diễn:** như A, thêm danh sách "đã biết" từ BS.

**4. Vì sao nhiều lựa chọn đều có thể tối ưu.**
- Đối tượng UAV khác đã báo: chỉ cần token xác nhận, hoặc không gửi.
- Đối tượng chỉ một UAV thấy, hoặc UAV kia bỏ sót: cần ảnh hoặc ROI.
- Góc nhìn thứ hai khử tương quan lỗi detector tốt hơn gửi lại; đây là giả thuyết.

**5. Baseline:**
- gửi hết rồi khử trùng ở BS (NMS);
- chia vùng cố định;
- tri thức chung phát quảng bá kiểu GOSC;
- dư thừa theo bộ nhớ embedding kiểu CCSRA;
- CodeFilling.

**6. Dữ liệu (nút thắt chính).**
- VisDrone chỉ có một góc nhìn. "Chồng lấn" của CCSRA trên VisDrone là tổng hợp và không mô tả cách làm.
- Dữ liệu thật có nhãn chéo góc nhìn:
  - MDMT: 2 drone, chỉ tải qua Baidu, nhãn có lỗi;
  - MDMOT: 3–5 drone, không tìm thấy link tải;
  - M3OT: 2 drone, chỉ có xe, tải qua Figshare, nhưng không có danh tính chéo nên phải dựng bằng homography.
- Dữ liệu mô phỏng (CoPerception-UAVs, Air-Co-Pred) có danh tính chéo qua box 3D, nhưng là CARLA/AirSim, rất gần UAV3D.
- Ghi chú nội bộ: mẫu town3 của CoPerception-UAV mà dự án đang có thì các UAV cách nhau khoảng 205 m, gần như không chồng lấn.

**7. Đóng góp và rủi ro.**
- Đóng góp khả dĩ: khử trùng theo danh tính thật trên ảnh nhiều drone thật, chỉ số đúng hạn dưới tranh chấp kênh, so sánh có tính chi phí phối hợp.
- Rủi ro **cao**: CCSRA gần như đúng kịch bản này (token, nhiều UAV, VisDrone, dư thừa); GOSC; CodeFilling; Where2comm; Mazandarani (chưa đọc toàn văn). Nhánh "gửi lại" chồng với ASR.

**8. Pilot:**
- Trước hết là **cổng dữ liệu**: tải M3OT hoặc MDMT, đo tỉ lệ đối tượng nằm trong vùng chồng lấn và tỉ lệ "một drone thấy, drone kia bỏ sót".
- Dừng nếu tỉ lệ đối tượng chồng lấn < 20% hoặc mức bổ sung < 5%.

**9. PASS/FAIL:** chỉ viết chi tiết khi cổng dữ liệu qua. Hiện đề nghị **chưa làm B**, giữ làm phần mở rộng của C.

## 5. Ứng viên C (đề xuất): báo cáo ngữ nghĩa theo hiểu biết của BS

**1. Nhiệm vụ và một episode (cụ thể).**
- 5 UAV, mỗi UAV quay video ở 60–100 m. Luồng ảnh là phát lại một chuỗi VisDrone-VID, khoảng 25–30 khung/s, mỗi UAV một chuỗi khác.
- 1 BS cách 0,5–1,5 km, 2 kênh 1 MHz dùng chung (mô hình kênh A5). Episode 120 s.
- **BS phải duy trì danh sách đối tượng đang có trong vùng của mỗi UAV.** Mỗi đối tượng mới phải được báo trong ≤ T_new (ví dụ 2 s) kể từ khi xuất hiện.
- **Chỉ số chính:**
  - **độ trễ báo lần đầu** của mỗi track ground-truth;
  - **tỉ lệ đối tượng-thời gian BS biết đúng**: GT có box khớp IoU ≥ 0,5 trong danh sách hiện thời của BS;
  - tỉ lệ track sai;
  - bit/s và năng lượng.
- Chỉ số tuân thủ (nếu có ràng buộc chất lượng ảnh) báo riêng.

**Vì sao buộc phải chọn:** ở 25 khung/s, gửi ảnh mỗi khung cần cỡ 15 Mbit/s mỗi UAV (avif_50), vượt xa kênh. Token mỗi khung chỉ khoảng 0,1 Mbit/s. Ràng buộc là thật, không đặt ra để có bài toán.

**2. Mô hình hệ thống.**
- **Trạng thái của UAV k:**
  - bản sao danh sách đối tượng mà BS đang giữ, dựng từ thông điệp đã được ACK. Có biến thể ACK mất;
  - kết quả detector nhỏ trên khung hiện tại;
  - độ lệch giữa hai thứ: đối tượng mới chưa có ở BS, track ở BS đã cũ hoặc trôi, vùng detector nhỏ không chắc;
  - hàng đợi truyền, hàng đợi suy luận, SNR kỳ vọng, slack của deadline, năng lượng.
- **Hành động mỗi khung:**
  - không gửi;
  - TOK (cập nhật hoặc thêm đối tượng);
  - ROI (ô ảnh quanh đối tượng mới hoặc không chắc);
  - HYB;
  - IMG (khung khoá).
- **Ở R2:** BS lập lịch kênh cho các UAV theo độ lệch dự kiến (giá trị thông tin), so với AoI và round-robin.
- **Tính toán:** detector nhỏ không chạy được ở mọi khung trên phần cứng yếu.
  - Năng lực UAV là tham số có nhiều mức: ví dụ 5 / 10 / 25 lần suy luận mỗi giây.
  - Giữa hai lần suy luận dùng tracker rẻ (kiểu Glimpse/MARLIN).
  - Suy luận và mã hoá tính vào deadline và năng lượng.
- **Kênh:** mô hình A5 (fading, hai kênh, tái dùng kênh), hàng đợi FIFO hoặc ưu tiên, ACK có trễ.

**3. Biểu diễn và bên nhận.**
- **TOK:** (id cục bộ, lớp, box, conf) theo kiểu delta so với track BS đã có.
- **ROI:** ô ảnh avif chất lượng cao, BS chạy YOLO11s trên ô.
- **IMG:** ảnh toàn khung, BS chạy YOLO11s.
- BS gộp các nguồn bằng tracker đơn giản (Kalman + ghép IoU) để giữ danh sách đối tượng.
- **Bản sao ở UAV chạy đúng cùng thuật toán gộp** trên các thông điệp đã ACK, nên UAV biết BS đang tin gì (tính tất định của bên nhận là điều kiện để bản sao đúng).

**4. Vì sao nhiều lựa chọn đều tối ưu ở những trạng thái khác nhau.**

| Trạng thái | Lựa chọn hợp lý |
|---|---|
| Cảnh ổn định, các track BS còn đúng | không gửi, hoặc token cập nhật thưa |
| Đối tượng mới mà detector nhỏ thấy chắc | TOK |
| Đối tượng mới nhỏ, hoặc detector nhỏ không chắc | ROI |
| Cảnh đổi lớn (UAV quay đầu, vào khu mới), hoặc bản sao lệch nhiều (ACK mất) | IMG |
| Kênh nghẽn | ưu tiên UAV có độ lệch lớn nhất |

Các trạng thái này **quan sát được một cách nhân quả** (bản sao, detector nhỏ, hàng đợi), khác với "giá trị tác vụ của ảnh" trong G-L2.

**5. Baseline mạnh nhất (đều tinh chỉnh trên train, khoá trên seed và chuỗi riêng):**
- ảnh định kỳ mỗi k khung;
- token mỗi khung + ảnh mỗi k khung;
- kích hoạt theo thay đổi khung kiểu Glimpse/Reducto (sai khác pixel hoặc feature rẻ);
- kích hoạt theo số đối tượng mới của detector nhỏ;
- pipeline cố định kiểu EdgeDuet;
- lập lịch theo AoI / round-robin (R2);
- giá trị thô kiểu GOSC (R2).

Cận trên để đo khoảng cách: kích hoạt oracle biết ground-truth.

**6. Dữ liệu và tài sản.**
- **VisDrone-VID** (train/val/test-dev, video có box và id track theo từng khung): **phải tải**. Dung lượng cỡ vài GB, cần kiểm. Giấy phép nghiên cứu như VisDrone-DET. Phải kiểm VID có trùng chuỗi với DET-train hay không, để tránh rò rỉ vào detector BS.
- Dùng lại: detector BS (A5), codec AVIF / mbt, mã khớp chi phí và bootstrap theo video (G-L2), simulator A5 (R2).
- Từ R1a: detector nhỏ, codec token, ghép ROI.
- Xây mới: tracker ở BS và bản sao ở UAV, bộ phát lại video, chỉ số theo track.

**7. Đóng góp khả dĩ và rủi ro novelty.**
- **Đóng góp khả dĩ:**
  1. Chọn họ biểu diễn theo **trạng thái hiểu biết của bên nhận**, với detector thật và bỏ sót thật trên video UAV.
  2. Chứng minh (hoặc bác bỏ) rằng bản sao bên nhận cho lợi ích bit/nhiệm vụ vượt baseline kích hoạt đã tinh chỉnh.
  3. Ở R2: lập lịch kênh chung cho nhiều UAV theo độ lệch, có cảm biến thật, đúng chỗ GOSC chỉ có cảm biến Bernoulli.
- **Rủi ro novelty:**
  - Glimpse/MARLIN/Reducto: một thiết bị, khung kích hoạt + theo vết.
  - GOSC: tri thức chung, giá trị thông điệp, kênh chung, nhưng cảm biến trừu tượng.
  - EdgeDuet: hybrid trên video VisDrone.
  - Lý thuyết AoII/VoI.
  - CERF/LIF: box cho tracking, trên UAV3D.
  - ASR: khôi phục do bên nhận yêu cầu. C là phía gửi chủ động; **phải hỏi chủ nhánh ASR** để tránh trùng trước R1b.

  Độ mới **trung bình**. Nó nằm ở tổ hợp và ở bằng chứng thực nghiệm với đối chứng mạnh, không ở một cơ chế hoàn toàn mới.

**8. Pilot không cần RL: R1a rồi R1b, mục 7.**

**9. PASS/FAIL: mục 7.**

## 6. So sánh ba ứng viên

| Tiêu chí | A: họ biểu diễn theo khung | B: hợp tác biết chồng lấn | C: theo hiểu biết của BS |
|---|---|---|---|
| Các lựa chọn khác nhau về bản chất | Có (token / hybrid / ảnh) | Có | Có, mạnh hơn: thêm "không gửi" và delta |
| Trạng thái quyết định đoán được | Mạng: có; nội dung: nghi ngờ (G-L2) | Chồng lấn: có, nếu biết hình học | **Có:** bản sao BS, đối tượng mới, hàng đợi |
| Ràng buộc tính toán trên UAV có thật | Yếu ở nhịp A5 (deadline 4–12 s ≫ 40 ms) | Trung bình | **Mạnh** ở nhịp video |
| Dữ liệu sẵn sàng | **Có** (VisDrone-DET) | **Kém** (MDMT Baidu, M3OT thiếu id chéo; mô phỏng thì gần UAV3D) | Phải tải VisDrone-VID (thường tải được) |
| Rủi ro novelty | Trung bình đến cao (Qiu, EdgeDuet, CCSRA) | **Cao** (CCSRA, GOSC, CodeFilling, Mazandarani) | Trung bình (Glimpse/MARLIN, GOSC, EdgeDuet) |
| Gắn với bầy UAV | Qua kênh chung (A5) | **Trực tiếp** | Qua kênh chung + tri thức chung; mở rộng ra B khi có dữ liệu |
| Chi phí đến cổng đầu tiên | Thấp (khoảng 1–2 giờ GPU + CPU) | Trung bình (tải và dựng id chéo) | Trung bình (cần R1a + tải VID) |
| Nếu thất bại, vẫn học được gì | Bản đồ khi nào token/ảnh thắng | Ít | Bản sao BS có đáng hay không |

## 7. Đề xuất ưu tiên và các cổng pilot (định trước)

**Ưu tiên C, đi qua A như một cổng thành phần.**
- C là ứng viên duy nhất mà cả ba tiêu chí ở mục 1 cùng thoả: lựa chọn khác bản chất, trạng thái đoán được, ràng buộc thật.
- R1a (phần tĩnh của A) là cần thiết vì C dùng chung menu biểu diễn. Nếu ngay ở dạng tĩnh token/hybrid không bao giờ cạnh tranh được với ảnh, C mất nửa cơ sở.
- **B tạm gác:** dữ liệu chưa sẵn và prior art dày. Nếu C qua R2, B là phần mở rộng tự nhiên ("BS đã biết đối tượng này từ UAV khác").

### 7.1 R1a: bản đồ họ biểu diễn, tĩnh (VisDrone-DET, offline, không RL)

- **Detector nhỏ trên UAV:** YOLO11n train trên VisDrone-DET-train, cùng quy tắc loại chuỗi như A5, ở imgsz 640. Có thêm một mức thứ hai (ví dụ imgsz 960) để thấy năng lực UAV ảnh hưởng thế nào. Latency đo trên CPU khi máy rảnh; quy đổi sang phần cứng UAV ghi là giả định.
- **Họ biểu diễn:**
  - IMG (14 mode, đã có);
  - TOK(τ), τ ∈ {0,1; 0,25; 0,4};
  - HYB(τ, avif_5 hoặc thumbnail);
  - ROI(τ, avif_50 / avif_80 cho ô).

  Mỗi (ảnh, cấu hình) có: bit, ms trên UAV, TP / FP / TP nhỏ, đo **hai lần** (ảnh gốc và ảnh lật) để kiểm ổn định.
- **Đánh giá:**
  - khớp bit như G-L2;
  - recall_ontime / AP50 / recall nhỏ;
  - test và test-dev riêng;
  - CI theo video.
- **PASS (cả hai):**
  1. **Không họ nào trội khắp nơi.** Ở ít nhất một mức bit trong [bit của TOK; bit của avif_80], một họ không phải ảnh (TOK / HYB / ROI) hơn đối chứng ảnh tốt nhất (RV_probe) ≥ **+2,0 điểm** recall. Ở ít nhất một mức bit khác, ảnh hoặc ROI hơn TOK ≥ +2,0 điểm. Cả hai trên test **và** test-dev, CI theo video loại 0, ổn định dưới phép lật.

     Ngưỡng 2 điểm (cao hơn 1 điểm của G-L2) vì chênh lệch giữa các họ phải là chênh lệch bản chất.
  2. **Chi phí:** detector nhỏ chạy được ≥ 5 lần/s trên CPU proxy, khi máy rảnh.
- **Kết quả phụ (không phải điều kiện):** luật chuyển họ nhân quả theo thống kê detector nhỏ có hơn "chọn họ theo ngân sách" ≥ +1 điểm không. G-L2 cho thấy có thể không.
- **FAIL:**
  - IMG (RV_probe) trội mọi họ khác ở mọi mức bit: token/hybrid vô dụng ngay cả khi bit chật. Khi đó **dừng cả A và C**, báo lại.
  - Hoặc TOK trội ở mọi mức bit tới avif_80: bài toán tầm thường, luôn gửi token. Khi đó C chỉ còn câu hỏi lập lịch.

### 7.2 R1b: bản sao BS trên video (VisDrone-VID, phát lại offline, một UAV, không RL)

- **Policy:** các baseline mục 5.5, mỗi cái tinh chỉnh trên chuỗi train, khoá trên chuỗi val; **policy bản sao** (heuristic: gửi theo loại độ lệch, ngưỡng tinh chỉnh trên train); kích hoạt oracle.
- **Chỉ số:** độ trễ báo lần đầu (trung vị và p90), tỉ lệ đối tượng-thời gian BS biết đúng, track sai, bit/s, ms trên UAV. Đánh giá trên chuỗi test / test-dev; CI theo chuỗi video.
- **PASS (cả ba):**
  1. Ở cùng tỉ lệ đối tượng-thời gian của baseline tốt nhất đã tinh chỉnh, policy bản sao dùng **≤ 70% bit**. Hoặc ở cùng bit, tỉ lệ cao hơn ≥ +3 điểm và độ trễ báo lần đầu p90 không tệ hơn. CI theo video loại 0.
  2. Hơn baseline "kích hoạt theo đối tượng mới của detector nhỏ". Đây là đối chứng gần nhất không cần bản sao; nếu không hơn thì bản sao không mang thêm gì.
  3. Giữ được ít nhất một nửa lợi ích khi ACK mất 10%.
- **FAIL:** điều kiện 1 hoặc 2 không đạt. Khi đó cơ chế hiểu biết bên nhận không có giá trị ở nhịp video VisDrone; báo lại và không sang R2.

### 7.3 R2: nối vào mạng (chỉ khi R1a và R1b PASS; heuristic trước, chưa RL)

- 5 UAV trong simulator A5, mỗi UAV một luồng VID, kênh chung.
- **So sánh:** lập lịch theo độ lệch dự kiến, AoI, round-robin, giá trị thô kiểu GOSC, kèm phần chọn biểu diễn ở 7.2.
- **PASS:** lập lịch theo độ lệch hơn AoI / round-robin ≥ +3 điểm tỉ lệ đối tượng-thời gian ở cùng kênh, ở ít nhất một regime nghẽn, CI theo video. Đồng thời còn khoảng cách ≥ 3 điểm tới oracle; khoảng cách đó là lý do duy nhất để thử RL ở R3.

## 8. Những gì chưa chắc, và khi nào phải đề xuất lại

- **Chưa đọc toàn văn:** Mazandarani 2606.31715 (quan trọng với B); bản tạp chí CoTeD (403); ENACHI; Glimpse/MARLIN/Reducto (chỉ từ kiến thức nền). Chưa tìm trực tiếp trên IEEE Xplore.
- **Phải đọc Glimpse và MARLIN toàn văn trước R1b.** Nếu MARLIN đã có cơ chế gần bản sao bên nhận cho detection, độ mới của C giảm, và C phải dựa vào phần nhiều UAV và kênh chung (R2).
- **Chồng lấn với ASR:** cần người dùng hỏi chủ nhánh ASR xem nhánh đó có gửi ROI/ảnh theo yêu cầu của BS trên video không.
- **Nếu R1a FAIL theo nhánh "ảnh trội mọi nơi":** có thể kết luận rằng trên VisDrone với detector nhỏ thực tế, mọi biểu diễn nhẹ đều thua ảnh. Khi đó hướng SemCom nên chuyển sang câu hỏi lập lịch thuần (A5.5) hoặc dữ liệu khác, và tôi sẽ đề xuất lại thay vì gượng ép.

## 9. Cần người dùng quyết

1. Đồng ý ưu tiên C, với R1a làm cổng đầu tiên? R1a cần train YOLO11n (khoảng 1–2 giờ GPU qua khoá GPU) và vài giờ CPU.
2. Đồng ý tải VisDrone-VID cho R1b, sau khi R1a PASS?
3. Ngưỡng PASS ở 7.1–7.3 có cần chỉnh không, trước khi tôi viết protocol chi tiết và commit?
