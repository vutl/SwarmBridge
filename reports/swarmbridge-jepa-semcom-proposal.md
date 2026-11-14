# Proposal: truyền thông ngữ nghĩa có dự đoán, đồng bộ trạng thái bên nhận, trên Town3 (JEPA là một ứng viên, không phải tiền đề)

Ngày 14/11/2026, theo đề nghị của người phản biện sau RQ1-E. Nguồn đầu vào: bản trao đổi "JEPA cho UAV SemCom" (`pdf/JEPA cho UAV SemCom.pdf`, 33 trang) và các báo cáo Town3 trong [archive/sb-town3/](archive/README.md#sb-town3). Chưa chạy GPU; đây là đề cương cho một nghiên cứu hệ thống.

## 1. Tóm tắt

- **Câu hỏi then chốt:** sau khi đã có một bộ truyền bản đồ có dự đoán rất mạnh, JEPA còn thêm được gì mà mô hình thời gian thông thường không làm được?
- **Trả lời thẳng [SUY LUẬN, sẽ được kiểm]:**
  - Với **một nhiệm vụ cố định** (BEV segmentation 3 lớp) và thông điệp là **bản đồ số**, JEPA **không có lý do kỹ thuật** để thắng.
  - Lý do: số bit cần gửi bằng −log p(bản đồ thật | trạng thái bên nhận). Mục tiêu đúng là một mô hình **xác suất có điều kiện** của bản đồ kế tiếp (huấn luyện bằng cross-entropy), dùng trực tiếp làm mô hình entropy.
  - JEPA chỉ học dự đoán trong không gian latent bằng sai số L2, không có likelihood, nên không cho mô hình entropy. Gắn tên JEPA ở thiết lập này là trang trí.
- **JEPA có vai trò kỹ thuật thật** khi bên nhận phải giữ một **trạng thái chung cho nhiều nhiệm vụ, hoặc cho nhiệm vụ chưa biết lúc truyền**. Khi đó thông điệp không thể là output của một nhiệm vụ.
  - Mục tiêu dự đoán trong latent của JEPA bỏ qua chi tiết không dự đoán được, vốn tốn bit nếu dùng mục tiêu tái tạo.
  - Đây là giả thuyết trung tâm của proposal, và nó kiểm chứng được trên Town3: đã có nhãn BEV, box 3D có ID theo vết, và pose.
- **Thiết kế hai tầng:**
  - **Tầng 1:** hệ thống mã hoá dự đoán cho output nhiệm vụ (bản đồ, box), dùng các bộ dự đoán học (ConvGRU, temporal U-Net, Mamba). Đây vừa là baseline mạnh nhất, vừa là một kết quả có ích tự thân.
  - **Tầng 2:** trạng thái latent có dự đoán cho đa nhiệm vụ, với một nhiệm vụ **giữ lại** không được thấy lúc huấn luyện.
- **Tên JEPA chỉ được giữ** nếu mục tiêu JEPA thắng mục tiêu dự đoán thông thường **cùng backbone** ở tầng 2 (thiết kế 2 × 2: backbone × mục tiêu).

## 2. Cơ sở thực nghiệm đã có (đối chiếu với báo cáo trong repo)

Town3 test, BEV segmentation 3 lớp (khác, đường, xe), teacher đạt mIoU 0,881.

**Khôi phục latent theo thời gian, gửi 15% block** ([sb-va-town3.md](archive/sb-town3/sb-va-town3.md)):

| Phương pháp | mIoU |
|---|---|
| Mapping từng frame | 0,609 |
| Bộ nhớ dịch theo pose (không học) | 0,679 |
| Temporal U-Net | 0,685 |

**Chọn block, 15%** ([temporal-block-selection-2026-11-03.md](archive/sb-town3/temporal-block-selection-2026-11-03.md)), mIoU / IoU xe:

| Cách chọn | Kết quả |
|---|---|
| Ngẫu nhiên | 0,676 / 0,220 |
| Trạm chọn theo bất định | 0,707 / 0,277 |
| UAV giữ bản sao trạng thái trạm (replica) | **0,760 / 0,419** |
| Biết trạng thái thật của trạm (chẩn đoán) | 0,763 / 0,420 |
| Replica khi mất toàn bộ ACK | 0,744 |

**Gửi bản đồ segmentation thay vì latent** ([step1-hardening-2026-11-03.md](archive/sb-town3/step1-hardening-2026-11-03.md), bit là độ dài mã lý tưởng theo mô hình ngữ cảnh, kênh số có mất gói, bên nhận không có mạng học):

| | Kết quả |
|---|---|
| Toàn bộ bản đồ | **1,0 kbit/frame** (test), mIoU 0,882 nếu không mất gói |
| Toàn bộ latent, lượng tử + nén entropy | 56,2 kbit/frame |
| Bộ nhớ + replica, khoảng 0,35 kbit/frame | **0,771 / 0,504** |
| Cùng chất lượng bằng latent | khoảng 8,4 kbit (khoảng 24 lần nhiều hơn) |

Hai nhánh khác kênh và cách lượng tử, nên tỉ lệ là chẩn đoán của thiết lập, không phải kết luận tổng quát.

**Bốn điều rút ra cho proposal:**
1. Phần lớn lợi ích thời gian đến từ **tái dùng frame trước**: bộ nhớ không học 0,679, U-Net chỉ thêm 0,6 điểm. Bộ dự đoán học chỉ có chỗ ở những gì bộ nhớ dịch theo pose làm sai: xe di chuyển, vùng mới thấy, mất gói liên tiếp.
2. Với kênh số và ACK, UAV mô phỏng **chính xác** trạng thái trạm. Phần "receiver-aware" đã giải xong trong thiết lập đó; khoảng trống cho mô hình niềm tin chỉ khoảng 1,5–2 điểm (khi mất toàn bộ ACK).
3. Output nhiệm vụ rẻ hơn latent hơn 20 lần. Mọi biểu diễn học phải so với thông điệp output nhiệm vụ, cùng kênh, cùng cách tính bit.
4. Vùng tốc độ đáng quan tâm là **0,08–0,6 kbit/frame**, nơi bộ nhớ + replica còn cách teacher 0,88 mIoU khá xa (0,58–0,81).

**Đã có sẵn:** dữ liệu `data/coperception_uav/` (bev_town3 với `rgb`, `obs`, `gt`, `seen` cho 5 UAV, `origin`, `axes`; bảng nuScenes-style với 67.576 box có ID); code `temporal/`, `temporal_select/`, `segmap_baseline/`; latent cache của teacher; môi trường `~/envs/sb` với `mamba-ssm` 2.3.2 đã kiểm chạy được (torch 2.12, RTX 3070).

**Giới hạn đã biết:** chỉ 17 scene (11 train / 2 val / 4 test), mô phỏng, UAV cách xa nhau (không có chồng lấn đáng kể giữa UAV). Giấy phép CoPerception-UAV chưa được kiểm trong đợt này.

## 3. JEPA thêm được gì? Lập luận kỹ thuật

### 3.1 Khi nhiệm vụ cố định và thông điệp là output nhiệm vụ: không

Gọi s_t là bản đồ thật, h_t là những gì bên nhận đã có (bản đồ trước đã căn pose, thông điệp đã nhận). Với mã entropy lý tưởng, số bit để gửi phần còn thiếu của s_t là

  R_t = −log₂ p_θ(s_t | h_t) (trên các block được gửi).

- Bộ dự đoán tốt nhất cho truyền thông là bộ cho **likelihood cao nhất** của bản đồ thật, tức một mô hình phân loại có điều kiện theo từng ô, huấn luyện bằng cross-entropy. Nó cho luôn cả bộ chọn block: những ô có entropy cao là chỗ có thông tin mới.
- Một predictor JEPA học ẑ ≈ z (L2 trong latent, target encoder EMA, không decoder) **không** cho p_θ(s_t | h_t). Muốn dùng để mã hoá phải gắn thêm đầu ra xác suất, và khi đó nó thành một bộ dự đoán có điều kiện thông thường với một bước tiền huấn luyện.
- Hệ quả: ở tầng 1, so sánh đúng là **backbone × mục tiêu**, không phải "JEPA so với U-Net". Nếu JEPA không thắng cùng backbone dùng cross-entropy, không có lý do gọi hệ thống là JEPA.

### 3.2 Khi bên nhận phải phục vụ nhiều nhiệm vụ, hoặc nhiệm vụ chưa biết: có thể

- Nếu nhiệm vụ ở BS không cố định, ví dụ thêm đếm xe theo hướng hay dự báo chiếm chỗ sau khi hệ thống đã triển khai, thì gửi output của từng nhiệm vụ không còn khả thi: không biết trước phải gửi output nào.
- Thông điệp phải là một **trạng thái chung** z_t, và bên nhận giữ ẑ_t dự đoán từ lịch sử.
- So với mục tiêu **tái tạo** (pixel / MAE): mục tiêu tái tạo buộc z_t mã hoá cả chi tiết không dự đoán được (kết cấu, nhiễu), vốn tốn bit mỗi frame. JEPA dự đoán **trong latent** và có xu hướng loại phần không dự đoán được khỏi mục tiêu. Đó chính là điều một bộ mã dự đoán cần: phần dư z_t − ẑ_t nhỏ ở những chỗ không có sự kiện.
- So với **học có giám sát đa nhiệm vụ**: latent học theo các nhiệm vụ đã biết có thể bỏ thông tin mà nhiệm vụ mới cần.
- Đây là giả thuyết **H-J** của proposal: *ở cùng số bit và cùng backbone, trạng thái latent học bằng mục tiêu JEPA phục vụ một nhiệm vụ giữ lại tốt hơn mục tiêu tái tạo và mục tiêu giám sát đa nhiệm vụ, và tốn ít bit hơn việc gửi output của mọi nhiệm vụ đã biết.* H-J có thể sai, và thiết kế dưới đây cho phép nó sai.

### 3.3 Những điều JEPA không tự làm

- Không tự biết thay đổi nào quan trọng cho nhiệm vụ (cần bộ chọn theo nhiệm vụ, áp như nhau cho mọi bộ dự đoán).
- Không tự giải đồng bộ khi mất gói (cơ chế replica và ACK áp như nhau cho mọi phương pháp).
- Không tự là codec (cần lượng tử và mô hình entropy, áp như nhau cho mọi thông điệp latent).

## 4. Kiến trúc hệ thống đề xuất

**UAV (bên gửi):**
1. Quan sát x_t và pose_t; bộ mã hoá E cho output nhiệm vụ (tầng 1) hoặc latent z_t (tầng 2). Bộ mã hoá dùng chung, đóng băng giữa các phương pháp được so.
2. **Replica** của bên nhận: giữ đúng trạng thái h_t mà BS có (chính xác với kênh số + ACK; ước lượng khi ACK mất hoặc trễ).
3. **Bộ dự đoán P** giống hệt BS: ŝ_t = P(h_t, pose_t), kèm phân phối hoặc độ bất định.
4. **Bộ chọn phần đổi mới theo nhiệm vụ:** chọn block có giảm mất mát nhiệm vụ kỳ vọng lớn nhất trên mỗi bit, dưới ngân sách B_t.
5. **Mã entropy** theo mô hình xác suất của chính bộ dự đoán (tầng 1) hoặc một prior học được dùng chung (tầng 2); header và chỉ số block được tính phí.

**Kênh:** số, block đến nguyên vẹn hoặc mất (xác suất p_erase, hiện thực kênh rút trước và dùng chung cho mọi phương pháp); ACK 1 bit/block trên đường xuống, có thể trễ d frame hoặc mất.

**BS (bên nhận):** cùng P → ŝ_t; ghép thông điệp nhận được → h_{t+1}; các đầu nhiệm vụ (BEV seg, phát hiện xe, nhiệm vụ giữ lại) đọc từ h_{t+1}.

## 5. Ba câu hỏi nghiên cứu (một nghiên cứu thống nhất)

- **Q1, dự đoán:** bộ dự đoán học (theo backbone và mục tiêu) dự đoán trạng thái kế tiếp tốt hơn bộ nhớ dịch theo pose **ở đâu**? Đo tách riêng ở ô có xe di chuyển, ô mới thấy (dùng mặt nạ `seen`), và sau chuỗi mất gói.
- **Q2, tốc độ – chất lượng:** lợi ích dự đoán có thành tiết kiệm bit ở cùng chất lượng không? Đo đường cong tốc độ – chất lượng và BD-rate so với bộ nhớ + replica, ở dải 0,08–1 kbit/frame (tầng 1) và ở tầng 2.
- **Q3, độ bền:** khi mất gói, ACK trễ hoặc trạng thái hai bên lệch, cơ chế cập nhật còn hoạt động không? Đo độ dốc suy giảm và thời gian hồi phục sau khi lệch.

## 6. Dữ liệu, nhiệm vụ và cách chia

- **Town3:** 17 scene × 20 mốc thời gian × 5 UAV = 1.700 lưới BEV; chia theo scene đã có: 11 / 2 / 4. Chọn mọi thứ trên val, chấm test một lần.
- Test chỉ có 4 scene (20 chuỗi UAV), nên báo kết quả **theo scene** và khoảng tin cậy bootstrap theo scene. Không coi seed huấn luyện là đơn vị độc lập.
- **Nhiệm vụ:**
  - **T1:** BEV segmentation 3 lớp (đã có teacher);
  - **T2:** phát hiện xe trong BEV từ box 3D (nhãn có sẵn: 67.576 box có ID, cần dựng nhãn BEV);
  - **T3 (giữ lại cho tầng 2):** một nhiệm vụ không dùng lúc huấn luyện bộ mã hoá / dự đoán, ví dụ hướng di chuyển hoặc vận tốc xe từ ID theo vết. Chỉ huấn luyện một đầu đọc nhỏ trên trạng thái ở BS, không truyền lại gì. Chốt T3 trước khi huấn luyện.
- **Giai đoạn sau, VisDrone-VID** (video UAV thật; split v1 theo nguồn đã kiểm toán trong R1b): thông điệp là detection, dự đoán theo thời gian bằng vết; nhiệm vụ là F1 phát hiện. Không so chỉ số trực tiếp với Town3.

## 7. Các phương pháp so sánh

Mọi phương pháp dùng **cùng loại thông điệp** trong cùng tầng, cùng codec, cùng hiện thực kênh, cùng ngân sách, cùng quy tắc replica và bộ chọn.

| Nhóm | Phương pháp | Ghi chú |
|---|---|---|
| Không học | B0: không nhớ (ngẫu nhiên / gửi hết) | mốc dưới |
| Không học | **B1: bộ nhớ dịch theo pose + replica** | baseline mạnh nhất hiện có (0,771 @ 0,35 kbit) |
| Không học | B2: B1 + ngoại suy chuyển động xe (vận tốc không đổi theo vết đã nhận) | kiểm tra "dự đoán chuyển động đơn giản" |
| Học, mục tiêu dự đoán thông thường | ConvGRU-CE, temporal U-Net-CE, Mamba-CE | cross-entropy trên bản đồ / box kế tiếp (tầng 1); tái tạo hoặc giám sát đa nhiệm vụ (tầng 2) |
| Học, mục tiêu JEPA | ConvGRU-JEPA, Mamba-JEPA | target encoder EMA, stop-gradient, L2 latent; đầu nhiệm vụ nhỏ cho đánh giá |
| Tham chiếu | gửi hết output mọi nhiệm vụ; teacher không mất gói | trần / chi phí đối chứng |

**Thiết kế 2 × 2 chính:** {ConvGRU, Mamba} × {mục tiêu thông thường, JEPA}, cùng số tham số (±10%), cùng dữ liệu, cùng số bước huấn luyện. Nó tách được lợi ích do **kiến trúc** và lợi ích do **mục tiêu học**.

## 8. Tính bit

- **Tầng 1 (output nhiệm vụ):** độ dài mã lý tưởng −log₂ p theo mô hình của chính phương pháp (B1 / B2 dùng mô hình ngữ cảnh đã fit như `segmap_baseline`), cộng chỉ số block (bitmap hoặc mã vị trí) và header.
- **Tầng 2 (latent):** lượng tử đều + mô hình entropy dạng prior phân tách học trên train, dùng chung cho mọi phương pháp latent; chi phí là độ dài mã lý tưởng.
- **Đường xuống:** ACK 1 bit/block, bitmap yêu cầu nếu có; báo riêng.
- **Kênh:** báo cả bit và số lần dùng kênh với một tỉ lệ mã cố định khai báo trước.
- **Độ trễ:** thời gian suy luận đo trên RTX 3070, ghi rõ là proxy, không phải phần cứng UAV.

## 9. Chỉ số

- Đường cong tốc độ – chất lượng: mIoU, IoU xe (T1), AP / F1 (T2), độ chính xác T3.
- BD-rate so với B1.
- Chất lượng tách theo vùng (Q1): ô xe di chuyển, ô mới thấy, sau mất gói.
- Độ bền (Q3): chất lượng theo p_erase và độ trễ ACK; số frame để hồi phục sau một lần lệch trạng thái.
- Chi phí tính toán: thời gian và số tham số của bộ dự đoán, chạy ở cả hai đầu.
- Mọi số liệu theo từng scene, kèm khoảng tin cậy bootstrap theo scene.

## 10. Các so sánh chính (khoá trước khi chấm test; không phải chuỗi cổng nhỏ)

Một nghiên cứu, báo đủ mọi kết quả thành công lẫn thất bại. Bốn so sánh chính:

1. **C1 (Q2, tầng 1):** bộ dự đoán học tốt nhất (chọn trên val) so với B1, BD-rate trên T1 ở dải 0,08–0,6 kbit/frame. *Có ý nghĩa* nếu tiết kiệm ≥ 10% và khoảng tin cậy theo scene không chứa 0.
2. **C2 (2 × 2, tầng 1 và 2):** JEPA so với mục tiêu thông thường cùng backbone. Tên JEPA được giữ chỉ khi JEPA thắng ở T3 giữ lại (tầng 2) với khoảng tin cậy không chứa 0, và không kém quá 1 điểm ở T1 / T2.
3. **C3 (tầng 2 so với tầng 1):** tổng bit để đạt chất lượng mục tiêu trên T1 + T2 của trạng thái latent so với gửi output của cả hai nhiệm vụ; cộng thêm khả năng phục vụ T3 mà không truyền lại.
4. **C4 (Q3):** độ dốc suy giảm theo p_erase và độ trễ ACK so với B1.

Các phân tích Q1 theo vùng là mô tả, không quyết định.

## 11. Kết luận nào sẽ rút ra trong từng kịch bản

| Kết quả | Kết luận |
|---|---|
| C1 không đạt | Trên Town3, bộ nhớ dịch theo pose đã gần tối ưu cho dự đoán bản đồ; bài viết (nếu có) là một nghiên cứu đánh giá về truyền bản đồ có dự đoán, không có bộ dự đoán học |
| C1 đạt, C2 không đạt | Bộ dự đoán học có ích, JEPA thì không; hệ thống mang tên theo cơ chế (predictive task-output coding), không gắn JEPA |
| C2 đạt, C3 không đạt | JEPA cho trạng thái chung tốt hơn các mục tiêu khác nhưng vẫn đắt hơn gửi output; chỉ đáng khi nhiệm vụ thật sự thay đổi |
| C1, C2, C3 đạt | Ứng viên đóng góp hệ thống: trạng thái ngữ nghĩa có dự đoán, phục vụ nhiều nhiệm vụ với ít bit; cần kiểm chứng độc lập trên VisDrone-VID |

## 12. Prior art phải rà trước khi tuyên bố gì

Chưa đọc trong đợt này; liệt kê từ bản trao đổi và kiến thức nền, phải xác minh:
- mã hoá video có điều kiện / dự đoán (DVC, DCVC và các biến thể): "dự đoán + gửi phần dư + mô hình entropy có điều kiện" là kỹ thuật chuẩn;
- CommJEPA (arXiv, 11/2026); TS-JEPA cho điều khiển tiết kiệm truyền thông; truyền thông hướng nhiệm vụ có world model JEPA cho mạng tác tử (2026);
- GOSC và truyền thông theo giá trị có vòng kín; Wyner–Ziv / thông tin phụ ở bên nhận;
- dự đoán BEV theo thời gian trong collaborative perception (bù trễ kiểu How2comm, CoBEVFlow).

Mục tiêu của đợt rà này là trả lời: đã có ai làm "trạng thái latent có dự đoán, phục vụ nhiệm vụ giữ lại, có tính bit" chưa? Không phải viết lại một bản tổng quan lớn.

## 13. Kế hoạch và tính toán

| Bước | Nội dung | Ước tính (RTX 3070) |
|---|---|---|
| 0 | Chạy lại B1 từ code có sẵn (khớp 0,771 @ 0,35 kbit); dựng nhãn BEV cho T2, chốt T3; khung tính bit chung; hiện thực kênh rút trước | 1–2 ngày công, không cần GPU nhiều |
| 1 | Tầng 1: B2, ConvGRU-CE, U-Net-CE, Mamba-CE; C1 trên val | vài giờ GPU mỗi mô hình (1.700 lưới, 192 × 192) |
| 2 | Tầng 2: 2 × 2 backbone × mục tiêu, prior entropy chung, đầu đọc T3; C2, C3 trên val | khoảng 1 ngày GPU |
| 3 | Q3 độ bền; khoá cấu hình; chấm test một lần | vài giờ |
| 4 | VisDrone-VID (thông điệp detection, dự đoán theo vết) | đề cương riêng sau bước 3 |

Mamba đã có môi trường chạy được. Không train lại V-JEPA 2 hay mô hình nền lớn; bộ mã hoá dùng lại hoặc nhỏ, đóng băng khi so các bộ dự đoán.

## 14. Rủi ro

- **Dữ liệu nhỏ** (4 scene test): khoảng tin cậy sẽ rộng; kết luận là khám phá cho tới khi có VisDrone-VID.
- **Baseline đã rất mạnh:** chỗ cho bộ dự đoán học nằm ở xe di chuyển và vùng mới thấy. Phải đo trước những ô này chiếm bao nhiêu phần lỗi của B1 (bước 0) để biết trần lợi ích. Không tạo thêm sự kiện nhân tạo để JEPA thắng.
- **Nhiệm vụ giữ lại T3** phải được chốt trước và không được chọn sau khi xem kết quả.
- **Chồng chéo prior art** với mã hoá video có điều kiện: đóng góp chỉ có thể nằm ở trạng thái đa nhiệm vụ có dự đoán và tính bit đầy đủ, không ở "dự đoán + phần dư".

## 15. Cần duyệt

1. Đồng ý cách đặt vấn đề: JEPA là một ứng viên cho trạng thái chung ở tầng 2, không phải tiền đề của toàn hệ thống.
2. Chốt T3 (đề xuất: hướng / vận tốc xe từ ID theo vết).
3. Duyệt bước 0 và bước 1 (GPU nội bộ, dữ liệu Town3 đã có, không tải gì mới).
