# SwarmBridge: đánh giá lại hướng nghiên cứu (sau R1b)

> **Cập nhật 11/11 (sau phản biện):** H1 được duyệt ở mức ứng viên. Gate 0 ở mục 6 và ý "chấp nhận rủi ro license" ở mục 7 đã bị **thay thế** bởi [swarmbridge-h1-gate0-protocol-v2.md](swarmbridge-h1-gate0-protocol-v2.md): phân loại oracle, F1 đối tượng duy nhất, MDMT-test chỉ còn là tập khám phá, quyết định truy cập dữ liệu.

Ngày 11/11/2026. Chưa code, chưa train, chưa chạy thí nghiệm nào cho phần này.

**Nguồn:**
- Bốn bản ghi chú có trích trang, trong [reassess-notes/](reassess-notes/):
  - [r1b-postmortem.md](reassess-notes/r1b-postmortem.md): tôi viết;
  - [axis1-collab-perception-u2u.md](reassess-notes/axis1-collab-perception-u2u.md): khoảng 30 công trình collaborative perception, BS/RSU làm bên nhận, U2U/relay;
  - [axis2-voi-taskoriented-beliefs.md](reassess-notes/axis2-voi-taskoriented-beliefs.md): VoI, nguồn tương quan, niềm tin cũ/chung, ra quyết định đa tác tử;
  - [datasets-feasibility.md](reassess-notes/datasets-feasibility.md): kiểm toán dataset và dữ liệu kênh.
- Ghi chú R0 cũ: [r0-notes/collaborative-swarm.md](r0-notes/collaborative-swarm.md).

**Cách làm:**
- Ba sub-agent đọc paper và trang dataset; tôi tổng hợp.
- Tôi tự kiểm lại các điểm quyết định:
  - các con số GOSC trong PDF (tr. 6, 9, 11);
  - phần đánh giá của Mazandarani (văn bản arXiv v3);
  - độ chồng lấn MDMT, chạy lại trên ground truth test công khai.
- Mức đọc: F = toàn văn; P = phương pháp và đánh giá; A = abstract hoặc trích đoạn.
- "Không tìm thấy" luôn có nghĩa là **không tìm thấy trong nguồn đã kiểm**.

## 0. Tóm tắt

1. **R1b đóng, kết quả FAIL giữ nguyên.** Phần lớn nguyên nhân là giới hạn triển khai, không phải kết luận khoa học (mục 1). Bài học quy trình: thêm cổng **hợp lệ** trên TUNE trước mọi cổng chấm.
2. **Ý tưởng "nhiều UAV chồng lấn, chọn gửi gì về BS" đã có nhiều công trình làm từng phần.**
   - Xe: EMP, C-MASS, RACooper, Mazandarani/PRISM.
   - UAV: CCSRA (chồng lấn tổng hợp trên VisDrone), BHU (Air-Co-Pred, không xử lý trùng lặp tường minh).
   - Tác tử trừu tượng: GOSC.

   Không thể claim "hợp tác nhiều UAV" là mới.
3. **Bằng chứng ngoài cùng chiều với R1b: mô hình hoá chính xác đồng đội hoặc bên nhận thường ít đáng giá** (GOSC, tôi đã kiểm trong PDF):
   - một thông điệp chỉ đổi mục tiêu của đồng đội trong 6–12% của 914 quyết định;
   - bộ lập lịch tập hợp chính xác không cải thiện ở bất kỳ trường hợp nào trong 1.800 nhiệm vụ ghép cặp;
   - ToM oracle chỉ nhanh hơn 3,1 [−4,3; 10,7] slot, không có ý nghĩa.

   Lợi ích của GOSC đến từ: giá trị gần đúng, ngưỡng gửi theo **giá** của một lần giao thành công, và BS phát quảng bá "tri thức chung".
4. **U2U không cần thiết cho bài toán báo cáo về BS, trừ một số điều kiện hẹp:** U2B bị chặn hoặc ngoài tầm, uplink có những đoạn mất kết nối, hoặc sidelink rẻ và rộng hơn hẳn uplink.
   - Ở Mazandarani (đọc toàn văn), chỉ biến thể oracle "biến đổi góc nhìn hoàn hảo" vượt baseline bỏ qua trùng lặp. Biến thể làm được thực tế thì không vượt (theo Fig. 4, đọc từ hình), và tệ đi khi tăng thời gian trao đổi V2V.
   - Trùng lặp loại bỏ được không cần U2U: BS phát danh sách đã biết, phân vùng do edge quyết định, hoặc giải mã chung ở bên nhận.

   **Đề xuất: bỏ U2U khỏi câu hỏi chính.**
5. **Dữ liệu là ràng buộc cứng.**
   - Dữ liệu thật đồng bộ, có ID xuyên góc nhìn và truy cập được chỉ có **2 drone** (MDMT; VisionTrack cần kiểm thêm).
   - Với N ≥ 3 drone chỉ có mô phỏng CARLA (Air-Co-Pred), sát họ UAV3D.
   - Không có trace kênh đo nào cho nhiều UAV chung một uplink.
6. **Hướng ưu tiên đề xuất: H1 "Báo cáo hợp tác theo đối tượng duy nhất".** Giữ đúng ý "giá trị thông tin hợp tác" mà phản biện muốn, nhưng có cổng đầu tiên rẻ, có thể bác bỏ, và **không cần simulator**: cổng dư địa offline trên MDMT (mục 6).
   - Nếu oracle phối hợp (biết ID thật, phối hợp miễn phí) không hơn chính sách độc lập đã tinh chỉnh ít nhất +5 điểm, hướng này dừng.
   - Rủi ro novelty vẫn **trung bình–cao**; nói rõ ở mục 4.

## 1. Postmortem R1b (chi tiết: [reassess-notes/r1b-postmortem.md](reassess-notes/r1b-postmortem.md))

| | Nội dung |
|---|---|
| **Giới hạn triển khai** | **L1** CPU là nút thắt (≥ 2×R_F uplink không chặn). **L2** luật kích hoạt bão hoà, không bao giờ SKIP. **L3** oracle D không phải trần. **L4** bản đồ BS precision 14–21%. **L5** mất gói tất cả hoặc không, không có ARQ. **L6** một UAV. |
| **Kết luận khoa học hẹp còn đứng** | C1 chỉ hơn A1 +1,24 F1 ở mức có ràng buộc. Biết gói nào đã tới (ACK, +3,4) quan trọng hơn biết chi tiết trạng thái BS. Gợi ý box từ BS gây hại (−5,4 / −6,1). Thông tin cho bên gửi có giá trị bằng 0 khi quyết định không đổi được. |
| **Không được kết luận** | "Phản hồi ngữ nghĩa vô ích nói chung". |
| **Thay đổi quy trình** | Cổng hợp lệ trên TUNE, gồm (1) ràng buộc mục tiêu đang chặn, (2) đa dạng hành động, (3) oracle ≥ mọi baseline, (4) chỉ số đủ nhạy. Không qua thì kết luận là **INVALID** (thiết kế lại), không phải FAIL. |
| **Tái sử dụng** | Đánh đổi TOK / HYB / IMG của R1a (điểm giao khoảng 0,3 Mbit/khung); wire format chính xác đến bit; kênh ghép cặp có khoá; mô hình thời gian nhân quả; chỉ số theo lớp; split theo nguồn và kiểm toán rò rỉ; khung chấm một lần. |

## 2. Prior art: đã giải quyết gì, còn thiếu gì

### 2.1. Đã có (một reviewer sẽ trích ngay)

| Ý tưởng | Công trình gần nhất (mức đọc) |
|---|---|
| Nhiều bên gửi chồng lấn → bên nhận hạ tầng, chọn ai gửi gì | EMP (MobiCom'21, P): edge chia vùng chồng lấn theo Voronoi có tính băng thông. C-MASS (arXiv 2407.00412, P): topo nhận thức, phủ có ngân sách, greedy có cận. CCSRA (JWCN'26, P): multi-UAV → edge, điểm trùng lặp trong reward SAC, VisDrone. BHU (arXiv 2603.16927, P): multi-UAV → BS, chọn UAV + độ thưa + precoder, Air-Co-Pred. |
| Giá trị một bên gửi phụ thuộc lựa chọn bên khác | CodeFilling (CVPR'24): nhu cầu bão hoà. SchedCP / RACooper (P): bản đồ tin cậy của bên nhận cập nhật đệ quy. C-MASS: đối tượng chỉ phát hiện được bởi một **cặp**, nên hàm không submodular. PRISM (P). Luật CPM của ETSI ("bỏ đối tượng hàng xóm đã báo"). |
| BS phát "đã biết" để bên gửi bỏ trùng | GOSC (P, đã kiểm): tri thức chung, ngưỡng theo giá, uplink chung với mọi bit điều phối được tính. RACooper: bản đồ yêu cầu, chi phí phát "coi như không đáng kể". |
| Dùng sidelink để phát hiện trùng lặp | Mazandarani (MECOM'26, F): chỉ biến thể oracle thắng; biến thể thực tế thì không. |
| Relay qua bên khác khi uplink kém | Harbor (SenSys'24, P): trace di động thật, +12% độ chính xác. A2-UAV (INFOCOM'23): đa chặng vì ngoài tầm. Hu (TCOM'25, P): relay chỉ thắng ở cự ly xa. |
| Gộp trong mạng trước khi lên BS | LGCP (arXiv 2601.12749, P): sidelink → leader → RSU. Lợi ích trộn giữa chọn tập con và chia tính toán. |
| Khai thác chồng lấn ở bên nhận, không phối hợp | DJSCC phân tán (arXiv 2411.12228, 2503.21249; A). |
| Lập lịch nguồn tương quan (lý thuyết) | Sun & Pappas (MobiHoc'26, P): cảm biến chồng phủ chung TDMA; tối ưu là trộn ≤ K+1 chính sách. Shisher (arXiv 2509.01926, P): index policy; cận xấu đi khi tương quan tăng. Zhang/Vasconcelos (arXiv 2005.11438, P): ngưỡng cục bộ gần tối ưu khi cảm biến độc lập. |
| Phối hợp có cận khi chỉ một phần tác tử liên lạc | RAG (Xu, Garimella, Tzoumas, T-RO'25, P): f ≥ [OPT − Σ coin_i] / (1+κ); coin_i tính được từ hình học vùng nhìn. |
| Suy luận biên nhiều thiết bị | Shao/Mao/Zhang (TWC, P): IB phân tán, server chọn thiết bị gửi lại. Chi phí phản hồi được nói là "không nhỏ" nhưng không tính. |

### 2.2. Không tìm thấy trong nguồn đã kiểm (khoảng trống khả dĩ)

1. **Trùng lặp định nghĩa bằng ID đối tượng thật trên ảnh nhiều UAV thật.** CCSRA dùng VisDrone với chồng lấn tổng hợp không mô tả, và độ trùng đo bằng embedding. Các công trình còn lại dùng CARLA hoặc dữ liệu xe.
2. **So sánh ba họ khử trùng lặp trên cùng một ngân sách, có tính hết chi phí điều phối:**
   - (a) phía bên gửi qua sidelink;
   - (b) BS phát "đã biết";
   - (c) không phối hợp, khử ở bên nhận.

   Mazandarani chỉ so với bộ lập lịch bỏ qua trùng lặp.
3. **Chỉ số nhiệm vụ ở BS** (đối tượng duy nhất được báo đúng trước deadline) trong các công trình MAC có xét chồng lấn. Mazandarani dùng SSIM/LPIPS, PRISM dùng thông lượng.
4. **Mất gói và truyền lại** trong cơ chế nhiều bên gửi có xét chồng lấn.
5. **Kiểm định xem giá trị mức detection có thật submodular không.** Hai detection yếu từ hai góc có thể cùng xác nhận một đối tượng; C-MASS nêu điều này nhưng dùng mô hình thống kê.
6. **Bất định trong liên kết xuyên góc nhìn** (không có pose camera) **được coi là sai số của giá trị** trong bộ lập lịch.

**Cảnh báo:** khoảng trống 1–4 thiên về **đánh giá cẩn thận hơn**, không phải cơ chế mới. Chỉ 5–6 có thể thành cơ chế. Một paper Q1 cần cả hai, và cần lợi ích đo được.

## 3. Ba hướng ứng viên

### Hướng H1: Báo cáo hợp tác theo đối tượng duy nhất (ưu tiên)

1. **Câu hỏi.**
   - Khi nhiều UAV thấy chồng lấn và cùng chia một uplink có ràng buộc, mỗi UAV nên gửi gì (bỏ qua, token, mảnh ảnh, khung) để BS biết đúng **nhiều đối tượng duy nhất nhất** trước deadline?
   - Phần lợi ích phối hợp nào thu được bằng cơ chế thực tế, khi tính đủ chi phí điều phối?
   - **Vì sao vấn đề tồn tại:** trên MDMT, 45,6% đối tượng trong một khung được cả hai drone thấy (23–83% theo chuỗi; tôi đã đo lại). Gửi độc lập lãng phí bit cho đối tượng trùng, còn bỏ trùng thô bạo thì mất đối tượng chỉ một bên thấy rõ.
2. **Mô hình hệ thống.**
   - N UAV (dữ liệu thật N = 2; N = 3–4 chỉ ở mô phỏng phụ), 1 BS, uplink chung có ngân sách mỗi epoch, downlink nhỏ cho điều phối.
   - Mỗi UAV có detector nhỏ (YOLO11n); BS có detector lớn (YOLO11s).
   - **Không có U2U trong mô hình chính.**
3. **Khác prior art.**
   - Trùng lặp theo ID thật trên ảnh thật (khác CCSRA, BHU, Mazandarani).
   - Chỉ số nhiệm vụ ở BS (khác Mazandarani, PRISM).
   - Ba họ khử trùng lặp so trên cùng ngân sách, có tính chi phí điều phối (chưa thấy ai làm).
   - Menu biểu diễn đa họ từ R1a (khác GOSC, C-MASS: gói kích thước cố định).
   - Liên kết không có pose.
4. **Cơ chế (không mặc định RL).**
   - **M1. Giá trị biên theo giá:** mỗi đối tượng hoặc vùng có giá trị = xác suất BS chưa biết nó (từ danh sách "đã biết" BS phát, nhân độ tin cậy liên kết xuyên góc nhìn) × lợi ích của biểu diễn. UAV gửi khi giá trị ≥ giá do BS phát (GOSC Thm 1–2). Phân bổ theo greedy mật độ giá trị: cận (1 − 1/e)/2 cho knapsack nếu hàm submodular.
   - **M2. Phối hợp chỉ giữa các UAV có vùng nhìn chồng nhau** (coin_i > 0, theo RAG). Chỉ dùng khi M1 không đủ.
   - **M3. Xử lý phần bổ sung lẫn nhau:** gửi detection yếu nếu một UAV khác cũng thấy yếu và BS có thể hợp nhất. Đây là chỗ giá trị không submodular; chỉ đáng làm nếu cổng 0 cho thấy tỉ lệ này đáng kể.
5. **Dữ liệu.**
   - MDMT: thật, 2 drone, có ID xuyên góc nhìn trong ground truth, đánh giá chính thức dùng `gtA.ID == gtB.ID`. 88 chuỗi; test có 14 cặp, 5.883 khung, 601 nghìn box.
   - Ràng buộc: ảnh chỉ trên Baidu (25,8 GB), không license, không pose, nhãn có lỗi đã biết.
   - Bản sao chép lặp lại: VisionTrack hoặc M3OT (ID xuyên drone phải tự dựng bằng homography).
   - N ≥ 3: Air-Co-Pred (CARLA, 4 drone, ID chính xác từ `instance_token`), chỉ cho nghiên cứu mở rộng theo N.
6. **Baseline mạnh và điều kiện FAIL.**
   - Baseline:
     - (B0) mỗi UAV độc lập, có triệt trùng theo lịch sử của chính nó (bài học ACK từ R1b), tham số tinh chỉnh trên TUNE, chia ngân sách tối ưu;
     - (B1) gửi hết, BS tự khử trùng (không phối hợp);
     - (B2) phân vùng tĩnh kiểu EMP;
     - (B3) GOSC chuyển sang ảnh: giá trị gần đúng + ngưỡng theo giá, không có xử lý liên kết;
     - trần: oracle phối hợp.
   - **FAIL** nếu oracle phối hợp − B0 < +2 điểm ở mọi ngân sách (không có dư địa), hoặc cơ chế thực tế thu < 50% dư địa sau khi tính chi phí điều phối.
7. **Rủi ro.**
   - **Novelty: trung bình–cao.** CCSRA, C-MASS, GOSC, EMP rất gần. Đóng góp phải là cặp "đánh giá có ID thật + cơ chế xử lý liên kết bất định và phần bổ sung".
   - **Dữ liệu:** chỉ 2 drone thật, nên chiều "bầy" yếu; phải mở rộng N bằng mô phỏng.
   - **Compute:** thấp (detection offline, phân tích tĩnh).
   - **Khả năng Q1:** có, *nếu* dư địa thật tồn tại và cơ chế thực tế giữ được phần lớn. Bằng chứng GOSC và Zhang/Vasconcelos gợi ý lợi ích có thể chỉ có trong một vùng hẹp (chồng lấn cao, ngân sách chặn, tài nguyên vừa phải). Cổng 0 trả lời câu này rẻ.

### Hướng H2: Xác minh hợp tác — góc nhìn thứ hai hay gửi lại

1. **Câu hỏi.**
   - Với detection bất định ở BS, kéo **góc nhìn thứ hai** từ UAV khác có lợi hơn trên mỗi bit so với yêu cầu **gửi lại chất lượng cao** từ cùng UAV không?
   - Giả thuyết: góc nhìn khác làm lỗi detector bớt tương quan, còn gửi lại chỉ bỏ được lỗi nén.
2. **Mô hình.** BS quyết định kéo (pull), nên không còn vấn đề niềm tin của bên gửi như ở R1b. Lệnh kéo đi trên downlink và phải tính phí.
3. **Khác prior art.** Mô hình MDP niềm tin dạng kéo (Zakeri, arXiv 2503.12962, P) và GOSC (gom bằng chứng) dùng cảm biến trừu tượng. HARQ cho cooperative perception và ASR chỉ gửi lại từ cùng nguồn. Chưa thấy kiểm định độc lập lỗi giữa hai góc nhìn trên ảnh drone thật.
4. **Cơ chế.** Kéo tuần tự theo submodular thích nghi; chính sách ngưỡng ngẫu nhiên (Luo, Sun-Pappas).
5. **Dữ liệu.** MDMT, vùng chồng lấn. Đủ cho phép thử offline.
6. **Baseline / FAIL.**
   - Baseline: gửi lại cùng UAV ở chất lượng cao hơn; nâng chất lượng cố định cho mọi đối tượng bất định.
   - **FAIL** nếu tương quan lỗi giữa hai góc nhìn ≈ tương quan lỗi giữa hai mức chất lượng cùng góc, hoặc lợi ích trên mỗi bit < gửi lại.
7. **Rủi ro.**
   - **Chồng lấn ASR** ở nhánh gửi lại (của nhóm khác), nên phạm vi phải đặt trọng tâm vào nhánh "góc nhìn thứ hai".
   - Đóng góp hẹp hơn H1; dễ thành một **mục con** của H1.

### Hướng H3: Khi nào cần U2U — relay hoặc gộp trong mạng khi U2B bị chặn hoặc không đồng đều

1. **Câu hỏi.** Khi nào relay hoặc gộp qua U2U thắng báo cáo trực tiếp có BS điều phối, với mọi bit sidelink và điều khiển tính chung ngân sách?
2. **Mô hình.** N UAV; U2B có che chắn hoặc mất kết nối (trace kiểu Harbor, AERPAW); U2U ở băng riêng.
3. **Khác prior art.** Harbor (xe, không xử lý trùng lặp), A2-UAV (đa chặng vì tầm, không trùng lặp), LGCP (xe), PhyCom'26 về relay ngữ nghĩa đa chặng (chưa đọc được, trang trả 403).
4. **Cơ chế.** Ghép cặp helper–helpee kiểu b-matching, kết hợp giá trị biên của H1.
5. **Dữ liệu.** Chỉ mô phỏng (Air-Co-Pred hoặc CARLA, sát UAV3D) cộng trace AERPAW (1 UAV). Không có trace đo cho nhiều UAV.
6. **FAIL** nếu baseline BS điều phối không U2U ≥ U2U ở mọi kịch bản có U2B khả dụng. Bằng chứng hiện có (Hu TCOM'25 ở 300 m; Mazandarani bản thực tế) gợi ý điều này.
7. **Rủi ro.** Novelty cao (Harbor, A2-UAV); dữ liệu hoàn toàn tổng hợp; compute cao (cần simulator mới, vi phạm ràng buộc hiện tại); reviewer khó tin.

## 4. Bảng so sánh và lựa chọn

| Tiêu chí | H1 báo cáo đối tượng duy nhất | H2 góc nhìn thứ hai vs gửi lại | H3 U2U khi nào cần |
|---|---|---|---|
| Đúng mục tiêu "giá trị thông tin hợp tác" | **Cao** | Trung bình (một quyết định con) | Thấp–trung bình |
| Yếu tố bầy thật | Trung bình (N = 2 thật; N ≥ 3 mô phỏng) | Thấp | Trung bình (mô phỏng) |
| Novelty còn trống | Trung bình (đánh giá có ID thật + liên kết bất định + phần bổ sung) | Trung bình (kiểm định độc lập lỗi) | Thấp–trung bình |
| Rủi ro bị prior art phủ | Cao (CCSRA, C-MASS, GOSC, EMP) | Trung bình (ASR, HARQ) | Cao (Harbor, A2-UAV) |
| Dữ liệu thật | **Có** (MDMT, 2 drone) | **Có** (MDMT) | Không |
| Cần simulator mới | Không cho cổng 0 | Không | Có |
| Chi phí cổng đầu tiên | Thấp (offline) | Rất thấp (offline) | Cao |
| Có thể FAIL rõ ràng sớm | **Có** | **Có** | Khó |

**Đề xuất: chọn H1.**
- Đưa phép đo của H2 (tương quan lỗi giữa hai góc nhìn) vào cổng 0 như phân tích phụ: cùng dữ liệu, gần như không tốn thêm.
- **Loại U2U khỏi câu hỏi chính**, vì bằng chứng hiện có cho thấy U2U không giải quyết thêm gì khi U2B khả dụng. Chỉ mở lại nếu kịch bản có che chắn hoặc ngoài tầm được chứng minh là thực tế cho ứng dụng.

**Nói thẳng về rủi ro lớn nhất:**
- Nếu cổng 0 cho dư địa < +2, thì trên dữ liệu thật với 2 UAV, phối hợp không đáng làm.
- Khi đó tôi đề nghị **không** ép tiếp hướng hợp tác. Quay về câu hỏi một UAV hoặc menu biểu diễn (nền R1a), hoặc đổi bài toán.

## 5. Ma trận novelty (rút gọn; đầy đủ trong ghi chú axis 1–2)

Y = có; p = một phần; – = không tìm thấy trong phần đã đọc.

| Công trình | Ảnh thật nhiều góc | ID xuyên góc thật | Bên nhận là BS/hạ tầng | Uplink chung / tranh chấp | Tính chi phí điều phối | Chỉ số nhiệm vụ ở bên nhận | Detector thật | Mất gói / deadline | Xử lý phần bổ sung (không submodular) | U2U |
|---|---|---|---|---|---|---|---|---|---|---|
| Where2comm / How2comm / CoSDH | – (sim, V2X thật DAIR) | p | – (ego) | – (lý tưởng) | – | Y (AP) | Y | p (trễ) | p | – |
| CodeFilling | p (DAIR) | p | – | – | – | Y | Y | p | p (nhu cầu bão hoà) | – |
| SchedCP / RACooper | – | – | p (RSU) | Y | – (coi như 0) | Y | Y | – | p | – |
| C-MASS | p (mô hình fit từ V2V4Real) | – | Y | p (băng thông tổng) | p | Y (recall) | p (thống kê) | – | **Y** | – |
| EMP | – | – | Y (edge) | p | **Y** | – (độ trễ) | – | Y (trace) | – | – |
| Harbor | Y (LiDAR, Mcity) | – | Y | Y (trace thật) | p | Y | Y | Y | – | **Y** |
| LGCP | – | – | Y | Y (ns-3) | Y | Y | Y | Y | – | Y |
| CCSRA | – (chồng lấn tổng hợp) | – (embedding) | Y | Y (OFDMA) | – | Y (mAP) | Y | – (một lần) | – | – |
| BHU | – (CARLA) | – | Y | Y (MU-MIMO) | – | Y | Y | p (độ trễ) | – | – |
| GOSC | – (lưới trừu tượng) | n/a | Y | Y | **Y** | **Y** | – | Y (HARQ) | – | – |
| Mazandarani / PRISM | – (CARLA / tổng hợp) | p (hình học) | Y | Y | p (thời gian V2V) | – (SSIM) | – | – | – | Y (phát hiện trùng) |
| Sun & Pappas / Shisher | – (tổng hợp) | n/a | Y | Y | – | p (AoI) | – | p | – | – |
| **H1 đề xuất** | **Y (MDMT)** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y (kiểm định)** | – (cố ý) |

Dòng H1 là **mục tiêu**, chưa phải kết quả. Từng ô đã có người làm; tổ hợp đầy đủ thì không tìm thấy. Tổ hợp chỉ có giá trị khi lợi ích tồn tại.

## 6. Câu hỏi nghiên cứu đề xuất và cổng thực nghiệm đầu tiên

**Câu hỏi (H1):** trên ảnh nhiều UAV thật có ID đối tượng xuyên góc nhìn, với một uplink chung có ràng buộc và mọi bit điều phối được tính, phối hợp giữa các UAV về **gửi đối tượng nào, bằng biểu diễn nào** tăng được bao nhiêu số đối tượng duy nhất BS biết đúng đúng hạn, so với chính sách độc lập mạnh nhất? Phần nào của lợi ích đó cơ chế thực tế (không biết ID thật) giữ được?

### Cổng 0: dư địa phối hợp (offline, không simulator, không RL)

**Mục đích:** trả lời rẻ câu "phối hợp có đáng không" trước khi viết bất kỳ hệ lớn nào. Số ngưỡng dưới đây là **đề xuất để duyệt**, chưa khoá.

1. **Dữ liệu.**
   - MDMT, split theo **chuỗi**: TUNE từ chuỗi train, GATE là 14 cặp test.
   - Kiểm rò rỉ với dữ liệu train của detector (VisDrone-DET) bằng pHash như đã làm với VID; MDMT do cùng nhóm VisDrone phát hành nên rủi ro rò rỉ có thật.
2. **Detection.**
   - YOLO11n (UAV) và YOLO11s (BS), có sẵn, huấn luyện trên VisDrone; ánh xạ lớp person / car / bicycle.
   - Báo recall mỗi góc trước để biết độ lệch miền.
3. **Đơn vị và chỉ số.**
   - Epoch 1 s. Một đối tượng duy nhất (ID xuyên góc) được tính "BS biết" nếu ít nhất một UAV giao một báo cáo đúng lớp và IoU ≥ 0,5 trong epoch.
   - Chỉ số chính: **recall đối tượng duy nhất** theo ngân sách bit mỗi epoch.
4. **Menu biểu diễn:** bỏ qua / TOK (box + lớp + conf) / mảnh ảnh (HYB) / khung AVIF. Bit đo thật như R1a.
5. **Chính sách** (cùng ngân sách tổng B cho 2 UAV):
   - **B0 độc lập mạnh:** mỗi UAV tự chọn theo luật cục bộ, có triệt trùng theo lịch sử của chính nó; ngưỡng và cách chia ngân sách tinh chỉnh trên TUNE.
   - **B1** gửi hết trong ngân sách, BS tự khử trùng.
   - **ORA oracle phối hợp:** biết ID thật, điều phối miễn phí; greedy tối đa hoá số đối tượng duy nhất trên toàn bộ B.
6. **Cổng hợp lệ trên TUNE** (bài học R1b; không qua → **INVALID**, thiết kế lại):
   - ở mọi mức B, gửi hết độc lập vượt B ít nhất 2 lần (ngân sách thật sự chặn);
   - mỗi biểu diễn được chọn ≥ 5% ở ít nhất một mức;
   - ORA ≥ B0 và ≥ B1 ở mọi mức;
   - chỉ số chính phân biệt được ORA với chính sách yếu nhất ít nhất 5 điểm.
7. **Cổng dư địa trên GATE** (chấm một lần, CI bootstrap theo chuỗi):

   | Kết luận | Điều kiện |
   |---|---|
   | **PASS** | ORA − B0 ≥ +5 điểm ở ≥ 2/3 mức B và CI dưới > 0 → sang cổng 1 |
   | **FAIL** | ORA − B0 < +2 ở mọi mức → dừng hướng hợp tác |
   | **INCONCLUSIVE** | Còn lại |

8. **Phân tích phụ (chỉ báo):**
   - dư địa theo độ chồng lấn từng chuỗi (23–83%);
   - tỉ lệ đối tượng chỉ được xác nhận khi gộp hai detection yếu (phần không submodular, cho M3);
   - tương quan lỗi detector giữa hai góc vs giữa hai mức chất lượng cùng góc (cho H2).

**Cổng 1 (chỉ khi cổng 0 PASS):**
- M1 thực tế: BS phát danh sách "đã biết", liên kết xuyên góc **không** dùng ID thật, chi phí downlink tính đủ.
- Đạt nếu M1 thu ≥ 50% dư địa (ORA − B0) và hơn GOSC-chuyển-sang-ảnh (B3).
- Chưa cần mạng thời gian thực; khi đó mới đưa vào mô phỏng kênh (dùng lại `r1b_sim` có ARQ).

**Ước lượng chi phí cổng 0:**
- tải và kiểm tra dữ liệu (cần người dùng, xem dưới);
- tính trước detection: vài giờ GPU;
- code phân tích khoảng 600–900 dòng, cộng test;
- không simulator, không RL.

## 7. Việc cần người dùng

1. **Tải ảnh MDMT từ Baidu** (25,8 GB, chỉ có trên Baidu, mật khẩu trong README repo VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking). Claude không có tài khoản Baidu.
2. **Chấp nhận rủi ro license:** MDMT không ghi license. Dùng cho nghiên cứu có thể chấp nhận, nhưng cần ghi rõ trong paper.
3. **(Tuỳ chọn)** Email nhóm FusionTrack xin MDMOT (3–5 drone thật). Đây là cách duy nhất có dữ liệu thật N ≥ 3.
4. **(Tuỳ chọn)** Lấy toàn văn bài Physical Communication 2026 về relay ngữ nghĩa đa chặng cho bầy UAV (trang nhà xuất bản trả 403), nếu muốn giữ khả năng mở lại U2U.

## 8. Ràng buộc đã giữ

- Không chạy lại G7.
- Không train PPO / SAC / MAPPO.
- Không viết simulator mới.
- Không triển khai U2U.
- Không đọc nhánh ASR hay code UAV3D (chỉ ghi nhận chồng lấn dataset CARLA).
- Không coi các video VisDrone độc lập là góc nhìn đồng bộ.
