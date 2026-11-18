# Proposal v2.1: truyền thông ngữ nghĩa có dự đoán, thích ứng theo nhiệm vụ, cho nhiều UAV

**Sửa theo phản biện #25 (15/11/2026):** R1 lập luận JEPA; R2 mặt nạ chấm điểm; R4 bit đo từ codec thật; R6 nhãn AUDITED-TEST và tính nhân quả của geo-trax; R7 tách hai hướng nghiên cứu. Kết quả S1 chẩn đoán: [swarmbridge-songdo-s1-feasibility.md](swarmbridge-songdo-s1-feasibility.md). **Kết quả S1 (validation 6/10 PM1): tải cả đội max 11,2 kbit/s, nên truyền thông không bị giới hạn → S2/S3 trên Songdo và JEPA trên Songdo là NO-GO; các mục 9–12 dưới đây giữ lại làm hồ sơ, không còn là kế hoạch.**

Ngày 15/11/2026, sửa theo phản biện sau bản v1 (commit f1e54601, 050fc67a). Bản này thay thế v1. Chưa train mô hình nào; đã kiểm dữ liệu thật của Songdo Traffic (mục 2).

## 0. Thay đổi so với v1

- Dữ liệu chính chuyển sang **Songdo Traffic v2** (quỹ đạo xe thật từ 10 drone, CC BY 4.0). Town3 giữ làm baseline kỹ thuật; UAVid giữ làm kiểm chứng độc lập cho bản đồ ngữ nghĩa theo thời gian.
- Thêm **bộ chọn mode truyền** theo nhiệm vụ, trạng thái bên nhận và kênh (không chỉ theo kênh như SAGIN / UA-TSC).
- JEPA vẫn là một giả thuyết được kiểm, không phải tiền đề. Kiểm dữ liệu cho thấy rõ **ở đâu JEPA kiểm được và ở đâu không** (mục 2.5).

## 1. Tóm tắt

- **Songdo Traffic là bối cảnh nhiều UAV thật hợp với SwarmBridge hơn MDMT**, nhưng theo kiểu khác với dự kiến:
  - 10 drone bay **đồng thời**;
  - mỗi drone **tuần tra** luân phiên 2–3 giao lộ, nên mỗi giao lộ chỉ được quan sát ngắt quãng (khoảng trống dài nhất 4,7–11 phút);
  - vùng nhìn chồng nhau đồng thời giữa các drone **gần như không có**: chỉ 20 giây ở giao lộ R và 51 giây ở S, lúc bàn giao.
- **Bài toán tự nhiên của dữ liệu này:** BS giữ trạng thái giao thông của 20 giao lộ. Drone chỉ thấy một phần, theo lịch tuần tra, và cùng chia một đường lên có hạn. BS phải **dự đoán** những gì không được quan sát hay chưa được gửi; drone phải chọn **gửi gì, mode nào, khi nào**.
  - Đây là bài toán dự đoán + lập lịch nhiều UAV thật, không cần liên kết xuyên góc.
- **Về JEPA trên Songdo Traffic (sửa theo R1):**
  - Bản v2 viết rằng latent học từ trạng thái có cấu trúc "không có lý do tồn tại". **Lập luận đó sai.** Bất đẳng thức xử lý dữ liệu I(Z; Y) ≤ I(X; Y) đúng cho **mọi** latent học từ đầu vào X, kể cả I-JEPA / V-JEPA trên ảnh. Giá trị của một latent không nằm ở việc tạo thêm thông tin, mà ở nén, biểu diễn dự đoán và phục vụ nhiệm vụ dưới giới hạn bit.
  - JEPA trên chuỗi thời gian có cấu trúc đã có prior art: TS-JEPA cho điều khiển từ xa với đường lên hạn chế (arXiv 2406.04853, 2024) và CHARM (ICML 2026, embedding chuỗi cảm biến).
  - Kết luận sửa lại: **mã hoá trạng thái có cấu trúc theo xác suất là lựa chọn mặc định mạnh hơn trên Songdo; JEPA chưa có giá trị gia tăng được chứng minh cho T1–T3, nhưng không bị loại về mặt lý thuyết.** JEPA không được đưa vào S1–S3.
  - Songdo Vision **không** khớp được với quỹ đạo; chỉ 29 video mẫu (60 giây mỗi video) khớp được.
- **Ước lượng thô:** trạng thái có cấu trúc rất rẻ.
  - Mỗi giao lộ có trung bình 31–117 xe trong khung hình, vận tốc trung vị khoảng 1 km/h ở giờ cao điểm chiều (phần lớn đứng chờ).
  - Gửi đủ trạng thái 1 lần/giây tốn vài kbit/s mỗi giao lộ, so với 0,5–1 Mbit cho một khung 4K.
  - Gần như chắc chắn: với ba nhiệm vụ đã định, **gửi trạng thái có cấu trúc kèm dự đoán đã đủ**. Proposal ghi rõ điều đó thay vì tạo nhu cầu cho JEPA.

## 2. Kiểm tra khả thi dữ liệu (đã làm)

Cách làm: đọc metadata Zenodo và README; chỉ lấy **CSV phiên PM5 ngày 7/10/2022 của cả 20 giao lộ** (khoảng 360 MB nén, 2,6 GB giải nén) bằng yêu cầu theo dải byte; liệt kê tên file trong `sample_videos.zip` mà không tải video. Script: `scripts/songdo/audit_overlap.py` (v2, sửa theo R5) và `audit_timeline.py`; kết quả tổng hợp: `runs/songdo/audit_v2_2022-10-07_PM5_AUDITED-TEST.json` (ngày test, chỉ để kiểm khả thi) và `runs/songdo/audit_v2_2022-10-06_PM1_VALIDATION.json`. Dữ liệu thô nằm ở `data/songdo/` (ngoài git).

### 2.1 Songdo Traffic v2 (Zenodo 17924857, CC BY 4.0)

| Mục | Đã xác minh |
|---|---|
| Kích thước | 41 GB: 80 zip quỹ đạo (16–360 MB, mỗi zip = 1 giao lộ × 1 ngày × 10 phiên AM1–AM5, PM1–PM5), orthophoto 1,8 GB, master frames 0,25 GB, phân đoạn làn 25 KB, video mẫu 26,8 GB |
| Phạm vi | 20 giao lộ (A–U, không có D), 4 ngày (4–7/10/2022), 10 drone DJI Mavic 3, cao 140–150 m, 29,97 điểm/giây |
| Cột CSV | `Vehicle_ID` (duy nhất **trong một CSV**), `Local_Time`, `Drone_ID`, toạ độ ortho / EPSG:5186 / WGS84, dài–rộng xe, lớp (xe con, buýt, tải, mô tô), vận tốc, gia tốc, `Road_Section`, `Lane_Number`, `Visibility` |
| Phiên PM5 ngày 7/10 | 26,8 triệu dòng, 22.068 ID xe (theo CSV); bắt đầu 17:40–17:52, kết thúc 17:55–18:10 ở mọi giao lộ → **các drone bay đồng thời** |
| Tuần tra | Mỗi drone luân phiên 2–3 giao lộ, mỗi lượt 2–4 phút (ví dụ D1: A → B → C → B → A …); mỗi giao lộ có dữ liệu 171–1.042 giây, trải trên 171–1.673 giây (audit v2) |
| Nhiều drone trên một giao lộ | 8 giao lộ (A, F, J, L, M, N, R, S) có 2–3 drone, nhưng **thay phiên**: số giây có hai drone cùng lúc là 0, trừ R (20 s) và S (51 s) |
| Cùng một xe ở hai drone | Chỉ ở R (39 cặp) và S (150 cặp), ghép một-một (láng giềng gần nhất hai chiều, cùng lớp, ≤ 2 m, ≥ 3 giây) chỉ trên các giây cùng quan sát; tỉ lệ va chạm 0 (R) và 0,0002 (S). Khoảng cách trung vị trên các cặp được chấp nhận 0,14 m (p90 0,76–0,77 m), nhỏ đáng ngờ so với sai số phát hiện, nên có thể là dấu vết của bước xử lý; cần đối chiếu video. **Không có ID xuyên drone nào được cung cấp.** |
| Video mẫu | 29 video, 60 giây đầu của PM5 ngày 7/10, đặt tên `<giao lộ>_D<drone>`; 8 giao lộ có 2–3 video từ các drone khác nhau |

### 2.2 Songdo Vision v2 (Zenodo 21728430, CC BY 4.0)

- 5.419 ảnh 4K (4.335 train / 1.084 test), 274.190 box, 4 lớp; 16,1 GB.
- Các khung **lấy ngẫu nhiên** từ video (cộng 145 khung khó chọn tay), tên file là số thứ tự (`00001.jpg`), chia train / test **ngẫu nhiên theo khung**.
- Hệ quả:
  - **Không khớp được ảnh với quỹ đạo** từ thông tin đã phát hành (không có giao lộ, phiên hay thời điểm);
  - split train / test có thể rò rỉ (khung cùng phiên ở cả hai phía).
- Dùng được làm dữ liệu huấn luyện detector trên ảnh drone 4K nhìn thẳng xuống; không dùng làm nhãn cho nhiệm vụ theo thời gian.

### 2.3 Những điểm phải xử lý khi dùng Songdo Traffic

- **Tính nhân quả của từng trường** (đọc bài geo-trax, arXiv 2411.02136, mục 3.4–3.7 và phụ lục B):
  - **Vị trí:** không làm mượt, không nội suy ("we refrain from applying smoothing or interpolation techniques on the raw data"); bộ theo vết BoT-SORT chạy trực tuyến; mỗi khung căn theo khung đầu của video. → coi là nhân quả.
  - **Vận tốc / gia tốc:** nội suy tuyến tính rồi lọc Gauss σ = 14 khung, cửa sổ khoảng ±1,4 s. → **không nhân quả**; bên gửi tính lại vận tốc từ vị trí (sai phân lùi 1 s); giá trị làm mượt chỉ là đích đánh giá.
  - **Kích thước xe:** ước lượng trên cả vết. → không nhân quả; không dùng ở bên gửi.
  - **Master frame** để quy về toạ độ địa lý: chọn một lần mỗi giao lộ. → coi là tri thức tĩnh.
  - Kết luận: quỹ đạo Songdo được coi là **đầu ra cảm biến giả lập offline**; chỉ vị trí, lớp và làn được dùng như quan sát nhân quả.
- **Quỹ đạo là đầu ra tự động** (detector + tracker + ổn định hình + nội suy), không phải GT gán tay. Chúng vừa là "quan sát của UAV" vừa là "đích đánh giá", nên mọi kết quả là tương đối so với pipeline geo-trax. Ghi rõ đây là giới hạn.
- Đã biết có phân mảnh vết, nhập nhằng ID, gán làn sai cho xe cao (README).

### 2.4 Town3 và UAVid

- **Town3:** 17 scene × 20 mốc × 5 UAV, chia theo scene 11 / 2 / 4, có nhãn BEV, box 3D có ID và pose. Baseline mạnh đã đo:
  - bộ nhớ dịch theo pose 0,679 so với temporal U-Net 0,685 (gửi 15%);
  - toàn bộ bản đồ 1,0 kbit/frame đạt 0,882;
  - bộ nhớ + replica 0,771 @ 0,35 kbit (latent khoảng 8,4 kbit).
  - Nguồn: [sb-va-town3.md](../2026-10-30_sb-town3/sb-va-town3.md), [step1-hardening-2026-11-03.md](../2026-10-30_sb-town3/step1-hardening-2026-11-03.md).
- **UAVid:** video UAV thật, segmentation 8 lớp, 42 chuỗi (20 / 7 / 15), CC BY-NC-SA 4.0. Theo người phản biện, GT test 2020 đã công khai qua EOStore (tôi chưa tự kiểm cách truy cập). Kiểm chứng độc lập cho bản đồ ngữ nghĩa theo thời gian.

### 2.5 Mỗi bộ dữ liệu kiểm được câu hỏi nào

| Câu hỏi | Songdo Traffic | Video mẫu Songdo | Town3 | UAVid |
|---|---|---|---|---|
| Dự đoán trạng thái + mã hoá phần đổi mới | **có** (quỹ đạo) | có (ngắn) | có (bản đồ) | có (bản đồ) |
| Lập lịch nhiều UAV chung đường lên, quan sát ngắt quãng | **có** (10 drone tuần tra đồng thời) | một phần | không (UAV xa nhau) | không |
| Chọn mode theo nhiệm vụ | **có** cho M0 / M1 | có cho M3 | có | có |
| Latent / JEPA cho nhiệm vụ giữ lại | về nguyên lý có thể (kiểu TS-JEPA), nhưng chưa có giả thuyết hơn mã hoá trạng thái xác suất | có, nhưng chỉ 29 × 60 s | **có** | có |
| Hợp tác vùng nhìn chồng nhau | gần như không (R, S lúc bàn giao) | không đủ | không | không |

## 3. Bài toán (dựa trên dữ liệu)

- Một BS giữ trạng thái **S_t** của 20 giao lộ: các xe đang có, vị trí, vận tốc, lớp, làn, hàng chờ.
- 10 drone bay theo lịch tuần tra (BS biết lịch, coi như kế hoạch bay).
- Ở mỗi bước Δ (mặc định 1 giây), drone k đang ở giao lộ X_k(t) có quan sát o_{k,t} (trạng thái xe trong khung hình của nó).
- Đường lên chung có ngân sách B_t (bit / bước, chia giữa các drone); kênh từng drone có thể khác nhau.
- BS phục vụ nhiều nhiệm vụ (mục 5) từ trạng thái đã cập nhật Ŝ_t.
- Ở giao lộ chưa ai quan sát, BS chỉ có **dự đoán**.

## 4. Hệ thống nhân quả

**Trên drone k (bên gửi):**
1. Quan sát o_{k,t}: trạng thái xe trong khung hình. Vận tốc tính lại nhân quả từ vị trí; không dùng giá trị làm mượt.
2. **Replica** Ŝ^{BS}_t của BS cho giao lộ đang quan sát. Chính xác nếu kênh số + ACK; ước lượng nếu ACK mất hoặc trễ.
3. **Bộ dự đoán P** giống hệt ở BS: dự đoán trạng thái kế tiếp, kèm phân phối hoặc độ bất định.
4. **Bộ chọn mode** a_{k,t} = π(q_t, Ŝ^{BS}_t, c_{k,t}, B_t):
   - q_t: nhu cầu nhiệm vụ hiện tại của BS;
   - Ŝ^{BS}_t: trạng thái BS ước lượng nhân quả;
   - c_{k,t}: kênh;
   - B_t: phần ngân sách.
5. **Codec** cho mode được chọn; header, chỉ số và độ dài thông điệp đều tính phí.

**Kênh:** số, mất gói với xác suất p (hiện thực rút trước, dùng chung cho mọi phương pháp); ACK 1 bit / gói trên đường xuống, có thể trễ.

**Tại BS:** dự đoán P cho mọi giao lộ; cập nhật bằng thông điệp nhận được; các đầu nhiệm vụ đọc từ Ŝ_t.

**Vai trò từng trường dữ liệu:**

| Trường | Bên gửi biết lúc t? | Bên nhận biết miễn phí? | Vai trò |
|---|---|---|---|
| Vị trí, lớp, kích thước, làn, `Visibility` của xe trong khung hình | có (coi là đầu ra perception trên drone) | **không**; chỉ qua thông điệp | quan sát, được gửi có tính phí; đồng thời là đích đánh giá |
| Vận tốc / gia tốc làm mượt trong CSV | **không** (không nhân quả) | không | chỉ làm đích đánh giá |
| Vận tốc nhân quả (sai phân vị trí, lọc một phía) | có | không | quan sát |
| `Vehicle_ID` | có, như ID vết cục bộ trên drone | không | dùng trong thông điệp M1 (tính bit) |
| Bản đồ làn, orthophoto, lịch tuần tra | có | **có** (tri thức tĩnh / kế hoạch) | ngữ cảnh dùng chung |
| Xe ở giao lộ không được quan sát | không | không | không có đích đánh giá; chỉ chấm lúc drone quay lại |

## 5. Ba nhiệm vụ

- **T1, giám sát / đếm:** số xe theo lớp trên mỗi `Road_Section` (nhánh vào giao lộ) trong cửa sổ 10 giây, và độ dài hàng chờ (số xe vận tốc < 5 km/h theo làn). Chỉ số: MAE theo giao lộ – cửa sổ.
- **T2, theo vết / ước lượng trạng thái:** BS giữ danh sách xe đang ở các giao lộ đang được quan sát, kèm vị trí và vận tốc. Chỉ số: sai số vị trí trung bình trên các xe khớp được, tỉ lệ xe bị sót / thừa (kiểu MOTA giản lược), đo cả trong lúc không có thông điệp mới.
- **Mặt nạ chấm điểm (R2):** mỗi chỉ số chỉ được tính ở những thời điểm có nhãn.
  - **Đoạn có drone quan sát:** mô phỏng việc không gửi / mất gói, chấm với quỹ đạo đã ghi.
  - **Thời điểm drone quay lại:** chấm trạng thái BS dự đoán **trước khi** nhận thông điệp mới (số xe theo nhánh, ghép vị trí bằng Hungarian trong 5 m).
  - **Khoảng không có drone:** không có GT trực tiếp; chỉ báo dự đoán hoặc độ bất định, **không** tính là lỗi.
  - T3 chỉ chấm khi cả chân trời dự báo nằm trong đoạn có quan sát.
- **T3, dự báo ngắn hạn:** vị trí từng xe sau 1, 2 và 3 giây (ADE / FDE), và số xe vào mỗi nhánh trong 30 giây tới.
  - Mở rộng (không chính): dự báo trạng thái giao lộ **lúc drone quay lại** sau khoảng trống tuần tra, chấm bằng quan sát lúc quay lại.
- Phương án an toàn (dự phòng nếu chọn thay T3): phát hiện cặp xe có thời gian tới va chạm dưới 2 giây. Chỉ dùng nếu chốt trước khi chạy.

## 6. Các mode truyền

| Mode | Nội dung | Codec / bit |
|---|---|---|
| **M0: output nhiệm vụ** | số đếm theo lớp – nhánh, độ dài hàng chờ | mã entropy theo mô hình xác suất có điều kiện của số đếm (từ dự đoán) |
| **M1: cập nhật trạng thái có cấu trúc** | với mỗi xe: ID cục bộ, lớp, vị trí, vận tốc lượng tử; chỉ gửi khi sai số so với dự đoán của BS vượt ε (kiểu dead-reckoning), cộng sự kiện xe vào / ra | độ dài mã −log₂ p(trạng thái thật | dự đoán) với một mô hình sai số Gauss / Laplace, cộng bit ID và header |
| **M2: latent dùng chung** | biểu diễn học được của cảnh | lượng tử + prior entropy chung. **Chỉ áp dụng ở nơi có ảnh** (Town3, video mẫu Songdo) |
| **M3: crop / khung ảnh** | AVIF của crop hoặc khung | bit AVIF đo thật. Chỉ trên 29 video mẫu (dự phòng khi trạng thái không đủ) |

**Bit chính đo từ codec thật (R4):** các trường được lượng tử (vị trí 0,25 m, vận tốc 0,25 m/s), mã exp-Golomb có dấu, ghi ra bitstream và giải mã lại để đối chiếu; vòng đời ID (xe vào / cập nhật / xe ra), ACK và đồng bộ lại sau mất gói đều tính. Độ dài mã lý tưởng −log₂ p chỉ là chẩn đoán. Mọi mode dùng cùng khung tính bit: payload, header theo gói, ACK / yêu cầu đường xuống (báo riêng), số lần dùng kênh với tỉ lệ mã cố định. Độ trễ = truyền + suy luận (đo trên RTX 3070, ghi là proxy).

## 7. Bộ dự đoán và baseline

**Bộ dự đoán P** (dùng cho M1, và cho trạng thái BS ở giao lộ không quan sát):

| Loại | Phương pháp |
|---|---|
| Không học | giữ nguyên (xe đứng yên); vận tốc không đổi (dead-reckoning); đi theo làn với vận tốc không đổi; mô hình hàng chờ theo pha đèn ước lượng từ dữ liệu |
| Học (xác suất) | ConvGRU / Mamba / Transformer nhỏ trên tập xe của một giao lộ, đầu ra là phân phối (NLL), dùng thẳng làm mô hình entropy |
| Học (mục tiêu JEPA) | cùng backbone, mục tiêu dự đoán latent kiểu JEPA + đầu đọc. **Chỉ ở tầng có ảnh** (Town3, video Songdo); trên quỹ đạo thì đây là phân tích phụ, vì JEPA không cho likelihood (xem v1, mục 3.1) |

Thiết kế 2 × 2 {ConvGRU, Mamba} × {dự đoán thường, JEPA} giữ nguyên, chạy trên Town3 (tầng latent đa nhiệm vụ, có nhiệm vụ giữ lại). Trên Songdo, so sánh chính là **giữa các bộ dự đoán xác suất** về số bit M1 cần để đạt chất lượng T1–T3.

**Bộ chọn mode và lập lịch giữa 10 drone:**

| Loại | Phương pháp |
|---|---|
| Cố định | luôn M0; luôn M1 tần số cố định; luôn M3 (trên video mẫu) |
| Theo kênh (kiểu SAGIN / UA-TSC) | chọn mode / tốc độ chỉ theo SNR và ngân sách |
| Tuổi thông tin (AoI) | ưu tiên giao lộ có trạng thái cũ nhất |
| **Theo giá trị thông tin / bit** | ước lượng mức giảm lỗi nhiệm vụ kỳ vọng của mỗi mode trên mỗi bit, từ độ bất định của dự đoán và nhu cầu q_t; chọn tham lam dưới ngân sách |
| Oracle (đánh dấu) | biết trước lỗi nhiệm vụ thật của mỗi lựa chọn |

RL chỉ được xét nếu oracle hơn heuristic giá trị / bit đáng kể **và** lựa chọn hiện tại ảnh hưởng tương lai rõ rệt (ví dụ trước khoảng trống tuần tra dài).

## 8. Chia dữ liệu

- **Nhãn truy cập (R6):** phiên PM5 ngày 7/10 đã được mở để kiểm khả thi ngày 15/11 → **AUDITED-TEST**: không dùng để chọn thiết kế hay ngưỡng; báo riêng. Xác nhận cuối dùng các phiên khác của ngày 7/10 chưa mở.
- **Theo ngày:** train 4–5/10, val 6/10, test 7/10. Xe không có ID xuyên ngày nên không rò ID; nhưng hình học giao lộ lặp lại, nên thêm trục thứ hai bên dưới.
- **Theo giao lộ (kiểm tổng quát):** giữ riêng một cụm giao lộ cho test. Vì lịch tuần tra nối các giao lộ với nhau, phải giữ trọn **lãnh thổ của một nhóm drone** (ví dụ D9 + D10: R, S, T, U), không tách lẻ giao lộ.
- **Theo phiên:** AM và PM báo riêng (lưu lượng khác nhau).
- Chọn mọi siêu tham số trên val; chấm test một lần.
- **Video mẫu Songdo:** chỉ có PM5 ngày 7/10, nên chỉ dùng cho thử nghiệm M2 / M3 trên test, không huấn luyện trên chúng.

## 9. Một nghiên cứu, bốn so sánh chính (khoá trước khi chấm test)

1. **S1, trạng thái có cấu trúc đủ tới đâu (Songdo):** đường cong bit – chất lượng của M0 và M1 với từng bộ dự đoán, cho T1–T3. Báo rõ số bit / giây / giao lộ cần để đạt chất lượng gần trần. Nếu M1 với dead-reckoning đã đạt T1–T3 ở mức vài kbit/s, **ghi kết luận đó**.
2. **S2, bộ dự đoán học có giảm bit không (Songdo):** bộ dự đoán xác suất tốt nhất (chọn trên val) so với dead-reckoning / đi theo làn: BD-rate cho M1 trên T2 và T3, khoảng tin cậy bootstrap theo giao lộ – phiên.
3. **S3, lập lịch nhiều UAV (Songdo):** bộ chọn theo giá trị / bit so với cố định, theo kênh và AoI, ở cùng ngân sách chung cho 10 drone; chỉ số T1–T3 trên mọi giao lộ (kể cả giao lộ đang không được quan sát, chấm lúc quay lại).
4. **S4, JEPA cho trạng thái chung (Town3):** thiết kế 2 × 2 trên nhiệm vụ giữ lại; tên JEPA chỉ được giữ nếu JEPA thắng cùng backbone với khoảng tin cậy không chứa 0, và không kém quá 1 điểm ở nhiệm vụ đã biết.

**Kết luận theo kịch bản:**

| Kết quả | Kết luận |
|---|---|
| S1 cho thấy M1 rẻ và đủ; S2 không có lợi | Bài viết là về **lập lịch dự đoán nhiều UAV với trạng thái có cấu trúc** (S3); không có JEPA |
| S2 có lợi | Bộ dự đoán học có giá trị truyền thông đo được; Mamba / ConvGRU so sánh trực tiếp |
| S3: giá trị / bit hơn AoI và theo kênh | Đóng góp chính: quyết định theo nhiệm vụ + trạng thái bên nhận, khác SAGIN / UA-TSC |
| S4 đạt | JEPA có vai trò cho trạng thái đa nhiệm vụ ở nơi có ảnh; một phần độc lập của bài |

## 9b. Hai hướng nghiên cứu tách riêng (R7)

- **Hướng S (Songdo):** truyền thông trạng thái giao thông có cấu trúc, có dự đoán, cho nhiều UAV tuần tra. Câu hỏi: có tồn tại bài toán truyền thông không tầm thường sau khi đã có codec trạng thái mạnh không? S1–S3 thuộc hướng này.
- **Hướng J (Town3):** latent dùng chung có dự đoán cho nhiều nhiệm vụ. S4 thuộc hướng này. Hai hướng dùng chung code (bộ nhận có dự đoán, khung tính bit) nhưng **không** được ghép thành một bài trước khi có kết quả.
- **Nhiệm vụ giữ lại cho S4 (đề xuất, chưa chốt):** phân loại **loại phương tiện** (từ bảng `category` của Town3, 21 loại) tại các vị trí xe trong BEV. Nhãn lấy từ box 3D có category; **không** suy ra được từ bản đồ BEV 3 lớp (khác / đường / xe) 1 kbit; chỉ huấn luyện trên scene train. Baseline mạnh: bản đồ BEV + một token loại xe cho mỗi xe (có cấu trúc, tính bit) so với latent dùng chung.

## 10. Prior art phải rà (chưa đọc trong đợt này)

- **Dead-reckoning và quy tắc phát bản tin V2X** (ETSI CAM / CPM): "chỉ gửi khi sai số so với dự đoán vượt ngưỡng" là kỹ thuật chuẩn. S1 / S2 phải so với chúng.
- Lập lịch theo **AoI / VoI** cho nhiều nguồn chung kênh (Sun & Pappas, TAMP, Fresh2comm đã ghi trong rà soát trước).
- **Đa nhiệm vụ, chuyển mode theo dịch vụ** trong SemCom: "Multi-level semantic-aware communication for multi-task image transmission" (2025), "Multimodal Semantic Communication With Information-Theoretic Disentangled Representations" (2026); bài lai bit / semantic dùng DRL trong SAGIN (có trong `pdf/`); UA-TSC (TNSE 2026).
- **Mã hoá dự đoán / nén quỹ đạo** (trajectory compression, dự báo quỹ đạo xe).
- JEPA cho truyền thông và chuỗi thời gian: TS-JEPA (arXiv 2406.04853, đã xác minh tóm tắt), CHARM (ICML 2026, đã xác minh tóm tắt), CommJEPA (11/2026), TOC với world model JEPA (2026).
- Ước lượng từ xa theo sự kiện (event-triggered / send-on-delta remote estimation), ước lượng từ xa có nhận biết nguồn (source-aware): gần nhất với M1 + bộ chọn theo giá trị / bit.

## 11. Kế hoạch

| Bước | Nội dung | Cần gì |
|---|---|---|
| 0 | Tải **chỉ các zip quỹ đạo** theo split (khoảng 9,5 GB / ngày; có thể bắt đầu với PM của 4 ngày) + phân đoạn làn; dựng pipeline bước 1 giây, vận tốc nhân quả, nhãn T1–T3; khung tính bit chung; kênh rút trước | CPU, đĩa |
| 1 | S1: M0 / M1 với dự đoán không học; đo trần chất lượng và số bit | CPU |
| 2 | S2: bộ dự đoán xác suất ConvGRU / Mamba / Transformer | GPU, vài giờ mỗi mô hình |
| 3 | S3: bộ chọn mode + lập lịch 10 drone | CPU |
| 4 | S4: Town3, thiết kế 2 × 2 | GPU, khoảng 1 ngày |
| 5 | Video mẫu Songdo (M2 / M3) và UAVid | đề cương riêng |

Không cần tải 26,8 GB video, 16,1 GB Songdo Vision hay orthophoto cho bước 0–3.

## 12. Cần duyệt

1. Đồng ý trọng tâm Songdo: dự đoán + lập lịch nhiều UAV với trạng thái có cấu trúc; JEPA chỉ kiểm ở nơi có ảnh (Town3).
2. Chốt ba nhiệm vụ T1–T3 (và có dùng phương án an toàn thay T3 không).
3. Duyệt bước 0–1: tải các zip quỹ đạo (không video), CPU.
