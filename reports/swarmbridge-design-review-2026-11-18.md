# Rà soát thiết kế SwarmBridge: đề xuất gốc, chỗ bị lệch, và cách khôi phục bài toán gốc

Ngày 18/11/2026, sau phản biện #47–#48 và bản sửa của #48 ("giữ nguyên đích, sửa cách đi").
- Đây là tài liệu thiết kế, **không có thí nghiệm mới**.
- Nguồn:
  - đề xuất gốc `pdf/Swarm-Conditioned Schrödinger Bridge cho Task-Oriented Multi-UAV Semantic Communications.pdf`
    (29/10);
  - các bản tóm tắt 30/10–06/11 trong `pdf/`;
  - `pdf/1. Tình hình nghiên cứu (State-of-the-art).pdf` (16/11);
  - các báo cáo trong `reports/` và `reports/archive/`.

## 0. Trả lời ngắn

- **Đích gốc của SwarmBridge** (29/10): đàn UAV chỉ truyền **bằng chứng bổ sung mà bên nhận còn thiếu**, theo giá trị
  thông tin cho tác vụ, có tính chi phí vô tuyến, trong một vòng kín:
  nhận thức → ước lượng bất định → xin thêm bằng chứng → hoàn thiện.
- **Từ 06/11, đích này bị thay** bằng bài toán offloading ảnh độc lập (theo Huang JSAC 2025).
  - Từ đó trở đi, gần như mọi thí nghiệm của Paper A đo một bài toán **không có hợp tác giữa các UAV** và **không có
    trạng thái ở bên nhận**.
  - Các kết luận âm về RL, về tiện ích từng ảnh và về bộ dự đoán đều đúng cho bài toán đó, nhưng không nói gì về đích
    gốc.
- **Phần đích gốc đã được thử thì chưa bao giờ được thử đúng:**
  - Town3 (30/10) chạy với một nguồn mỗi vùng (N = 1), tức không có bằng chứng bổ sung.
  - MDMT (11–14/11) chỉ có 2 drone, và góc nhìn tốt hầu như cố định.
  - **UAV3D (04/11) là nơi duy nhất có 5 UAV đồng bộ.** Ở đó hợp tác có giá trị lớn (mAP 0,512 → 0,665; AMOTA
    0,542 → 0,690), và giá trị đó **phụ thuộc mạnh vào truyền thông**: trễ 2 khung là mất gần hết. Nhánh này bị dừng
    sau một ngày vì chuyển hướng, không phải vì kết quả âm.
- **Có thể khôi phục đích gốc mà không xây lại từ đầu.**
  - Ghép phần vô tuyến LA đã kiểm chứng của Paper A với pipeline hợp tác UAV3D đã có (box / KF / hợp nhất / sổ cái
    nhân quả).
  - Thêm thứ còn thiếu ở cả hai: **trạng thái hiểu biết của bên nhận**.
- Quyết định bay đổi giá trị quan sát, nên chỉ đánh giá được trong simulator dựng được góc nhìn mới, hoặc bằng mô hình
  cảm biến trừu tượng. Vì vậy nên để quyết định bay ở giai đoạn 2. Giai đoạn 1 vẫn giữ được phụ thuộc dài hạn thật.

## 1. Đề xuất gốc nói gì (29/10)

**Câu hỏi khoa học** (nguyên văn): *Can a distributed UAV swarm communicate only the complementary evidence required to
reduce the uncertainty of a task-conditioned generative bridge?*

| Thành phần | Đề xuất gốc | Vai trò |
|---|---|---|
| Quan sát bổ sung nhiều UAV | Nhiều UAV cùng nhìn một cảnh; bên nhận thiếu một phần | Lý do để hợp tác |
| Giá trị thông tin có điều kiện | I(Y; M_i \| C): message của UAV i giảm bao nhiêu bất định so với những gì bên nhận đã có; nhãn oracle Δ = L(C) − L(C, M_i), rồi học bộ dự đoán giá trị | Chọn ai gửi, gửi gì |
| Chi phí vô tuyến | U_i = V_i − λ_B·bit − λ_E·năng lượng − λ_L·trễ; nhiễu, mất gói, trễ / cũ | Ràng buộc |
| Vòng kín | Bất định cao → chọn UAV trợ giúp → nhận → hoàn thiện → lặp | Truyền thông không luôn bật |
| Bộ hoàn thiện | Schrödinger Bridge có điều kiện (OT-CFM và ánh xạ tất định là baseline bắt buộc) | Thuật toán, có tiêu chí loại sớm |
| Dữ liệu | CoPerception-UAV / OpenCOOD; AirSim cho bay chủ động ở giai đoạn sau | |
| Thuật toán quyết định | "Không bắt đầu bằng MARL": oracle → bộ dự đoán giá trị → top-K / knapsack → sau đó bandit / PPO | |

Hai điểm cần nhớ:
- **Tác vụ và sự hợp tác là trung tâm. SB chỉ là một ứng viên giải.** Đề xuất có tiêu chí loại sớm: SB không hơn ánh
  xạ / OT-CFM thì bỏ SB, giữ phần "truyền theo giá trị / bất định".
- **Quyết định bay không nằm trong bài gốc.** Bay chủ động được đặt ở giai đoạn sau, với AirSim. Đòi hỏi "swarm phải
  tự bay" là mở rộng sau này, không phải lõi ban đầu.

## 2. Hành trình: đích gốc bị lệch ở đâu

Cột "còn giữ" đánh dấu bốn yếu tố của đề xuất gốc:
- **H:** hợp tác / bằng chứng bổ sung;
- **R:** trạng thái bên nhận / vòng kín;
- **W:** chi phí vô tuyến thực;
- **T:** tiện ích tác vụ.

| Giai đoạn | Làm gì | Còn giữ | Vì sao đổi | Bài học còn giá trị |
|---|---|---|---|---|
| Town3 SB (29/10–03/11) | SB khôi phục latent BEV; chọn block theo thời gian | R, T (W giản lược) | SB ≤ ánh xạ | **Thiết lập N = 1**: chưa bao giờ thử bằng chứng từ UAV khác |
| Tracking hybrid (03–05/11) | box so với đặc trưng cho tracking | H, T | danh sách box thắng | Khi tác vụ bên nhận cố định, gửi box thường thắng gửi đặc trưng |
| **UAV3D (04/11)** | 5 UAV đồng bộ, box / KF, hợp nhất noisy-OR, kênh trễ / mất | **H, T, W (đơn giản), R một phần** | dừng vì chuyển hướng 06/11 | **Hợp tác +15 mAP / AMOTA; trễ 2 khung xóa lợi ích**; mất 50 % gói −5,5 AMOTA |
| **Chuyển sang Huang (06/11)** | ảnh từng UAV → BS, chọn mode nén | W, T yếu (PSNR) | thử hướng SemCom lai bit + sinh ảnh | **Mất H và R** |
| A3–A5 (06–10/11) | PPO điều khiển bay / kênh / công suất / mode | W | PPO chung thua PPO chỉ chọn mode | Lợi ích của bay chỉ là canh nhịp thời gian; bay không đổi giá trị quan sát |
| G-L2, R1 (10–11/11) | tiện ích detector từng ảnh | T | không có dư địa | Một ảnh độc lập không chứa đủ thông tin để quyết định |
| H1 MDMT, B1, RQ1 (11–14/11) | quay lại hợp tác trên MDMT | H, T | 2 drone, góc nhìn tốt cố định (7/8 chuỗi), không liên kết được vật thể xuyên góc | Dữ liệu yếu về tính bổ sung, không phải ý tưởng sai |
| Songdo / JEPA (14–15/11) | lệnh điều khiển C&C | W | kênh không bị giới hạn | |
| Two-track, A-E, A-X, A-T1, bước C (15–18/11) | lập lịch + chọn mode dưới LA | W (rất tốt), T (recall) | DRL không hơn heuristic + MPC; tiện ích không tổng quát | **Bộ vô tuyến LA đã kiểm chứng**; Shannon và LA cho kết luận khác hẳn |

Chỗ lệch lớn nhất là ngày 06/11: bỏ H và R. Mọi thứ sau đó tối ưu một bài toán mà mỗi ảnh có giá trị riêng, bên nhận
không nhớ gì, và các UAV chỉ tranh kênh với nhau. Trong bài toán như vậy, luật cố định gần tối ưu là chuyện bình
thường.

## 3. Phần nào đang đúng hướng (giữ lại)

1. **Bộ vô tuyến thực tế** (`env.py` / `env_la`):
   - CQI trễ, BLER logistic, ACK/NACK, OLLA 10 %, ARQ;
   - nhiễu đồng kênh bền, link adaptation biết lịch;
   - scheduler JA; hàng đợi và deadline; mô hình năng lượng;
   - bộ hiệu chỉnh tốc độ từ ACK (MPC `cal`).

   Kết quả có giá trị: chính sách train trên Shannon thua 16,8 khi chạy trên LA, và lợi thế dùng chung kênh biến mất
   dưới LA thực tế. Bài toán hợp tác nào chạy trên Shannon cũng chịu rủi ro này.
2. **Pipeline hợp tác UAV3D** (archive `uav3d/`):
   - message box và KF (box + vận tốc), codec thật, hợp nhất noisy-OR;
   - sổ cái nhân quả (tx / rx_arrived / used), cache 3 khung, cold-start;
   - đánh giá mAP / AMOTA.
3. **Kỷ luật đánh giá:** baseline heuristic được tune công bằng, seed mới, chia theo video. Phản biện nào cũng sẽ đòi
   những điều này. Cần dùng ở giai đoạn đánh giá, không phải ở giai đoạn khám phá.
4. **Tri thức âm có ích cho thiết kế message:**
   - khi tác vụ bên nhận là detection cố định, box thường thắng đặc trưng;
   - SB làm decoder không hơn ánh xạ;
   - giá trị nội dung từng ảnh không dự đoán được từ đặc trưng rẻ.

## 4. Phần đã bị đơn giản hóa sai

| # | Bị đơn giản hóa | Hệ quả | Có cố ý? |
|---|---|---|---|
| S1 | **Bỏ hợp tác** (từ 06/11): mỗi UAV gửi ảnh riêng | Không còn "swarm", chỉ còn tranh tài nguyên | Có, theo Huang; không ai kiểm lại là đã rời đích gốc |
| S2 | **Bên nhận không có trạng thái**: giá trị một message không phụ thuộc BS đã biết gì | Không có giá trị thông tin có điều kiện, không có "cũ / mới" | Hệ quả của S1 |
| S3 | **"Semantic" thu về chọn 1 trong 14 mode nén một ảnh** | Tác vụ chỉ vào qua PSNR, rồi recall | Dần dần |
| S4 | **Đóng băng bay** (A5 trở đi), rồi đóng băng scheduler (A-X9) | RL chỉ còn chọn mode; quyết định gần như nhìn trước mắt | Có, để tách nguyên nhân. Hợp lý cho từng câu hỏi, sai cho đích chung |
| S5 | **Thử hợp tác trên dữ liệu yếu** (Town3 N = 1, MDMT 2 drone) và bỏ UAV3D | Kết luận "hợp tác ít giá trị" đến từ dữ liệu | Không cố ý; UAV3D bị gác do chuyển hướng |
| S6 | **Câu hỏi đánh giá thành "RL có thắng heuristic không"** thay vì "bài toán cần gì" | Chuỗi cổng chặn cho từng biến thể nhỏ | Quy trình, đã ghi nhận ở #47 |

**Điều không bị đơn giản hóa sai:** đòi baseline mạnh và PHY thực tế. Hai điều này nên giữ trong bài toán mới.

## 5. Khôi phục bài toán gốc: đề xuất (chưa triển khai)

### 5.1 Bài toán

**Cooperative multi-UAV perception with receiver-state-aware semantic communication under realistic link adaptation.**

- **Hệ thống:** K = 5 UAV quan sát chung một vùng (góc nhìn chồng lấn, che khuất khác nhau) và một BS/leader giữ trạng
  thái tác vụ: danh sách track (KF) kèm bất định và tuổi thông tin.
- **Mỗi slot:** chọn UAV nào gửi, gửi gì (không gửi / box / track KF / đặc trưng vùng / crop), trên kênh nào, công suất
  nào. Kênh là LA thực tế: CQI trễ, NACK, OLLA, nhiễu đồng kênh, deadline.
- **Giá trị của một message có điều kiện** theo những gì BS đang biết:
  - BS vừa nhận box của chiếc xe đó từ UAV khác thì gửi lại gần vô ích;
  - track sắp "cũ" hoặc vật thể sắp bị che với các UAV khác thì giá trị cao.
- **Mục tiêu:** chất lượng tracking dài hạn (AMOTA / số track bị mất / bất định trung bình), trừ chi phí airtime và
  năng lượng.

**Câu hỏi khoa học** (bám sát câu gốc): *khi giá trị của bằng chứng phụ thuộc vào những gì bên nhận đã biết và vào
kênh thực tế, việc phân bổ airtime giữa các UAV có cần quyết định nhìn xa và có điều kiện theo trạng thái bên nhận
không, hay một luật cục bộ đã đủ?*

**Giả thuyết bác bỏ được:**
- **H1:** một luật có điều kiện theo trạng thái bên nhận (ví dụ ưu tiên track có bất định / tuổi cao) hơn luật không
  điều kiện (round-robin, SNR tốt nhất, độ tin cậy cục bộ kiểu Where2comm) ở cùng airtime.
- **H2:** một bộ lập kế hoạch nhìn xa hơn luật tham lam một bước có điều kiện trạng thái, vì quyết định gửi bây giờ
  thay đổi bất định và hàng đợi ở các slot sau.
  - Nếu H2 sai, bài toán không cần RL hay MPC dài hạn; đóng góp là H1 cộng đánh giá dưới LA.
- **H3:** chính sách thiết kế trên kênh Shannon mất phần lớn lợi ích khi chạy trên LA thực tế. Paper A đã cho thấy hiện
  tượng này với lập lịch độc lập.

### 5.2 Ví dụ cụ thể: khi nào nhìn trước mắt quyết định sai

- **Thiết lập:** hai UAV, A và B, cùng thấy xe X. A có kênh tốt bây giờ. B có kênh xấu bây giờ, nhưng là UAV duy nhất
  sẽ còn thấy X sau khi X đi vào bóng tòa nhà (A mất góc nhìn sau 2 s).
- **Luật tham lam** cho A gửi X ngay, vì giá trị trên mỗi bit cao nhất bây giờ. Rồi nó dùng airtime còn lại cho mục
  khác.
- **Bộ nhìn xa** cho A gửi track KF có vận tốc (đủ dự báo 2 s), rồi để dành airtime cho B gửi X đúng lúc A mất góc nhìn,
  thay vì gửi X hai lần từ A.
- **Khác biệt:** track X không bị đứt; tổng airtime bằng hoặc ít hơn.
- **Điều kiện để khác biệt tồn tại:**
  - tầm nhìn thay đổi theo thời gian giữa các UAV (UAV3D có che khuất do nhà cửa);
  - kênh và hàng đợi làm cho "gửi lúc nào" quan trọng;
  - message có loại "dự báo được" (KF).

  UAV3D cho cả ba điều kiện ở mức thô: lợi ích KF so với box khi trễ, và mức suy giảm theo trễ.
- **Phép kiểm tra rẻ cho tính chất này** (khám phá, chưa làm):
  - trên vài cảnh UAV3D mini, với kênh LA, so ba chính sách cùng airtime: luật không điều kiện, tham lam có điều kiện
    trạng thái, và oracle lập kế hoạch biết trước tầm nhìn trong H giây tới;
  - khoảng cách giữa tham lam và oracle nói bài toán có cần nhìn xa không;
  - không cần train, có thể dùng lại pipeline UAV3D và bộ LA.

### 5.3 Dữ liệu và simulator

| Lựa chọn | Kiểm chứng được | Không kiểm chứng được | Chi phí |
|---|---|---|---|
| **UAV3D (đã có mini: 20 cảnh × 20 khung, 5 UAV, 7,1 GB)** + LA SwarmBridge | Giá trị hợp tác có điều kiện, độ cũ, lập lịch dưới LA, nhãn 3D/tracking thật (tổng hợp) | Quyết định bay (quỹ đạo cố định); thế giới thật | Thấp: pipeline có sẵn. Cần bản đầy đủ nếu 20 cảnh không đủ (kiểm tra khả năng tải) |
| CoPerception-UAV Town3 (đã có, 20 GB) | Giống trên, BEV segmentation | Bay; lần trước dùng N = 1, phải dựng lại cặp nguồn chồng lấn | Trung bình |
| Mô hình cảm biến trừu tượng (FoV, che khuất, xác suất phát hiện theo khoảng cách / góc), hiệu chỉnh trên UAV3D | **Quyết định bay**, phụ thuộc bay ↔ cảm biến ↔ kênh | Độ thật của cảm biến | Trung bình; rủi ro mô hình |
| CARLA / AirSim dựng góc nhìn mới | Bay chủ động với ảnh thật (tổng hợp) | | Cao: tích hợp, GPU, tốc độ mô phỏng |
| MDMT, VisDrone-VID, M3OT, CrossTrack-UAV | Thế giới thật | Bay; MDMT chưa rõ quyền dùng, M3OT 3/9 chuỗi, CrossTrack cần ký | |

**Đề xuất:**
- **Giai đoạn 1:** UAV3D + LA, không có quyết định bay.
  - Vẫn có hợp tác thật, trạng thái bên nhận, phụ thuộc theo thời gian và vô tuyến thực tế.
  - Ghi rõ đây là active communication, chưa phải active sensing.
- **Giai đoạn 2:** quyết định bay bằng mô hình cảm biến trừu tượng hiệu chỉnh trên UAV3D. Dùng CARLA/AirSim chỉ khi
  giai đoạn 1 cho thấy giá trị hợp tác thay đổi theo vị trí đủ lớn.

### 5.4 Nghiên cứu gần nhất phải đối mặt

| Nhóm | Đã giải quyết | Có thể còn thiếu (phải kiểm lại khi viết memo đầy đủ) |
|---|---|---|
| Where2comm, How2comm, CoSDH, CodeFilling | Gửi vùng hữu ích theo bản đồ tin cậy / cung–cầu, một khung | Thường không có bộ nhớ bên nhận dài hạn, không có kênh thực tế |
| SchedCP (DRL lập lịch người dùng cho CP), semantic HARQ cho CP, CCSRA | Lập lịch / truyền lại cho cooperative perception | Kênh Shannon / CSI hoàn hảo (đã ghi ở rà soát 13/11, 18/11) |
| GOSC, AoII / VoI scheduling | Gửi khi giá trị > chi phí; giá trị theo hiểu biết chung | Không phải perception đa góc nhìn; kênh đơn giản |
| Tracking hợp tác đa UAV đồng tối ưu cảm biến / truyền thông / quỹ đạo (2018; Dec-POMDP 2026; DMPC + policy search 2026, phản biện #48 dẫn) | Đồng tối ưu bay + cảm biến + truyền thông | Phải đọc: mô hình cảm biến, kênh, có message ngữ nghĩa hay không |
| Huang JSAC 2025, Li JSAC 2026, Si 2023 | Chọn mode SemCom + tài nguyên (+ quỹ đạo) bằng DRL | Không hợp tác nhận thức |

**Ứng viên đóng góp (chưa chứng minh):** lập lịch message hợp tác có điều kiện theo trạng thái bên nhận dưới LA thực
tế, nơi giá trị một message phụ thuộc vào cái BS đã biết và vào kênh.
- Rủi ro novelty là thật: SchedCP và các bài tracking hợp tác có thể đã gần.
- Memo đầy đủ phải đọc kỹ chúng trước khi viết thuật toán.

### 5.5 Tái dùng

| Thành phần | Nguồn | Sửa |
|---|---|---|
| PHY LA, ARQ, OLLA, nhiễu, năng lượng | `env.py`, `env_la` | Gắn với lịch UAV3D (bước 0,2 s ↔ slot) |
| Scheduler JA, hàng đợi, deadline | `ja_sched.py`, `env_ev.py` | Thay "ảnh" bằng "message" có kích thước theo loại |
| Bộ hiệu chỉnh tốc độ từ ACK, MPC | `a9_mpc.py` | Hàm giá trị theo trạng thái bên nhận |
| Message box / KF, hợp nhất, sổ cái nhân quả, AMOTA | `swarmbridge/experiments/uav3d_comm` (chỉ thêm, không sửa code UAV3D cũ) | Thêm trạng thái bên nhận (bất định, tuổi) |
| Bộ đánh giá công bằng, bootstrap theo cảnh | `atc_summary.py` | |

Mới phải viết: trạng thái bên nhận, giá trị có điều kiện của từng loại message, giao diện UAV3D ↔ LA.

### 5.6 Rủi ro và điều sẽ bác bỏ hướng này

1. **H2 sai:** tham lam có điều kiện ≈ oracle nhìn xa. Khi đó không có lý do cho RL / MPC dài hạn; đóng góp thu về H1 +
   đánh giá dưới LA.
2. **H1 sai:** điều kiện theo trạng thái bên nhận không hơn luật kiểu Where2comm ở cùng airtime. Khi đó đích gốc không
   có cơ hội trên dữ liệu này.
3. **UAV3D mini quá nhỏ** (20 cảnh). Cần bản đầy đủ; nếu không tải được thì kết luận chỉ ở mức khám phá.
4. **Dữ liệu tổng hợp:** cần một kiểm tra trên dữ liệu thật trước khi công bố. Hiện chưa có bộ dữ liệu thật nào sạch về
   quyền dùng và có góc nhìn bổ sung mạnh.
5. **Novelty:** nếu SchedCP hoặc các bài tracking hợp tác đã có trạng thái bên nhận + kênh thực tế, phải thu hẹp
   đóng góp.

## 5b. Câu hỏi "sao không mở thêm biến trong bài toán khảo sát cũ?" (bổ sung sau tin nhắn mới)

Phản biện viết "chúng ta đã thử joint channel–compression, chưa thử joint mobility–communication–compression", dựa trên
báo cáo A-X2. **Điều này sai.** Action của `env.py` từ A2 (06/11) là (bay, kênh, công suất, mode), và PPO đã điều khiển
cả bốn ở A3.2, A4 và A5:

| Thí nghiệm | Bài toán | Kết quả joint (bay + kênh + công suất + mode) | Lợi ích đến từ đâu |
|---|---|---|---|
| A3.2 (07/11, `archive/uav-semcom-a/uav-semcom-a.md` §7) | 3 UAV, tải thấp | +2,6…+2,7 so với heuristic đã tune | Bay +1,0…+1,9 (bay lệch hướng thay vì hover để canh nhịp), kênh + công suất ≈ +1, mode 0 |
| A4 (07/11, §8.11) | Khảo sát ảnh gốc, 20 m/s | +4,9 so với heuristic chốt trước | Mode theo hàng đợi (+4,2) và cho 3 UAV dùng 2 kênh; **bay và công suất do RL quyết định không đóng góp**. Hai luật rút từ hành vi PPO đạt trong khoảng 0,5 so với PPO |
| A5 (08–10/11, `uav-semcom-a5.md` §4.1) | ref / K5 / phy_mcs_blind | joint −0,87 / +2,19 / −5,53 so với T:best; **thua PPO chỉ chọn mode ở cả 3** | Vì vậy từ A5 trở đi chỉ giữ PPO chọn mode |

**Vì sao bay gần như vô dụng trong bài toán khảo sát:**
- Điều UAV nhìn thấy không phụ thuộc vị trí: ảnh được rút từ một pool, nội dung không đổi theo nơi chụp.
- Waypoint khảo sát là cố định, phải bay qua hết.
- Vì thế bay chỉ còn hai tác dụng:
  - đổi **thời điểm chụp** (canh nhịp hàng đợi), mà luật "paced" làm được;
  - đổi **khoảng cách tới BS** (chất lượng kênh).
- Bay không đổi giá trị thông tin. Thêm lại biến bay vào đúng bài toán này rất có thể lặp lại A3–A5: lợi ích nhỏ, và
  heuristic rút ra được.
- **Thứ duy nhất thật sự mới so với A3–A5 là PHY LA thực tế**, nơi khoảng cách tới BS và nhiễu đồng kênh có thể quan
  trọng hơn. A-X7 cho thấy đổi cách bay (paced so với mission) thay đổi return khoảng 19. Đó là một phép thử rẻ có thể
  làm, nhưng nên kỳ vọng thấp.

**Novelty:** điều khiển chung quỹ đạo + chọn mode SemCom + tài nguyên cho UAV offloading ảnh đã có người làm bằng DRL:
- Si et al. Globecom 2023;
- Li et al. JSAC 2026 (mode của Huang + quỹ đạo UAV + WPT + DSAC; ghi chú 06/11).

Mở lại các biến cũ trong bài toán khảo sát là đi vào vùng đã bị chiếm.

**Kết luận:** "giữ bài toán, mở thêm biến" và "khôi phục đề xuất gốc" không mâu thuẫn, nếu biến được thêm là biến
**làm bay có giá trị thông tin**:
- **Giá trị quan sát phụ thuộc vị trí và thời gian:** vùng chưa phủ, vùng cần nhìn lại, vật thể bị che với UAV này
  nhưng không với UAV kia.
- **BS có trạng thái:** đã biết gì, biết từ lúc nào.

Đó vẫn là nhiệm vụ khảo sát nhiều UAV gửi về BS, không đổi đề tài, nhưng có hợp tác. Giai đoạn 2 ở mục 5.3 (bay với mô
hình cảm biến theo vị trí) chính là "mở biến bay" theo nghĩa này.

## 6. Quyết định cần từ bạn (và phản biện)

1. Chấp nhận khung khôi phục ở mục 5.1 (giai đoạn 1 không có quyết định bay)? Hay bay phải có ngay trong bài đầu?
2. Cho phép kiểm tra tính chất bài toán ở mục 5.2 (không train, vài cảnh UAV3D mini, mức khám phá) trước khi viết memo
   literature đầy đủ?
3. Có tải được UAV3D bản đầy đủ không (dung lượng, giấy phép)? Mình sẽ kiểm tra trang dữ liệu nếu bạn đồng ý.
4. SB: giữ làm ứng viên cho bộ hoàn thiện (chỉ khi có bất định đa giả thuyết thật), hay bỏ hẳn khỏi tên dự án?
