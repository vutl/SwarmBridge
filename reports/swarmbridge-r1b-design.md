# SwarmBridge R1b: thiết kế v3 (đề xuất, chưa phải protocol khoá)

Ngày 12/11/2026. Bản v3 sửa v2 theo phản biện ngày 12/11; v1 và v2 xem trong lịch sử git. Ghi chú đọc paper ở `reports/r0-notes/r1b-feedback-video.md`.

**Trạng thái:**
- Chưa code R1b-0, chưa chạy pilot, chưa train RL, chưa mở R2.
- Chuẩn bị dữ liệu VisDrone-VID train / val: xem `reports/visdrone-vid-data-audit.md`.

**Thay đổi chính so với v2:**
1. Thí nghiệm đầu tiên là một **ablation có kiểm soát**: chỉ thay **thông tin phản hồi** mà UAV được thấy, mọi thứ khác giữ nguyên (mục 3). Hệ đầy đủ với codec video chỉ xây nếu ablation này có kết quả dương.
2. Ba giả thuyết H1 / H2 / H3 thay Q1–Q3:
   - các điều kiện stress khoá trước, không còn chữ "hoặc";
   - có định nghĩa INCONCLUSIVE.
3. Ngân sách: uplink và downlink giới hạn **từng chiều**. Bỏ nhánh "≤ 70% bit uplink" không tính downlink.
4. **Metadata chuyển động có giá.** Nếu gửi khi SKIP thì SKIP không còn 0 bit; nếu không gửi thì BS không biết homography.
5. **Giao thức:**
   - mã băm gắn với trạng thái chuẩn tại một thời điểm xác định;
   - số học tuần tự cho phiên bản và thời điểm quay vòng;
   - dự đoán bằng số học điểm cố định;
   - đóng gói, mất gói, deadline và phụ thuộc tham chiếu như nhau cho mọi hệ.
6. Cận oracle chỉ là cận **trong lớp policy đã cho**, không phải giá trị tối đa của phản hồi nói chung.

## 1. Câu hỏi và giả thuyết

**Câu hỏi.** BS có tracker và detector mạnh hơn UAV. Nếu UAV được biết (có trễ, có mất, có giá) BS đang tin gì, nó có quyết định **gửi gì, khi nào** tốt hơn không, theo nghĩa object-time F1 ở BS?

**Ba giả thuyết, kiểm theo thứ tự:**
- **H1. Giá trị gia tăng của phản hồi ngữ nghĩa.** Cùng policy, cùng menu hành động, cùng tracker, cùng ngân sách uplink. Phản hồi ngữ nghĩa có giá (C) cho object-time F1 cao hơn so với không có phản hồi ngữ nghĩa (A), và không thấp hơn gợi ý kiểu Glimpse (B).
- **H2. Độ vững.** Lợi ích của H1 còn lại dưới trễ và mất phản hồi, sau khi tính chi phí downlink.
- **H3. Hệ đầy đủ.** Bộ điều khiển biểu diễn (SKIP / TOK / khung video / ROI-QP) cạnh tranh được với x264 low-delay, Glimpse và AoIS ở cùng uplink, cùng giới hạn downlink, cùng tính toán và cùng deadline.

**Chỉ xây hệ của H3 nếu H1 không FAIL.** Nếu chỉ oracle (D) có lợi còn phản hồi có giá (C) không có lợi, thì báo rõ và **dừng** phát triển semantic ACK phức tạp (versioning, CRC, Kalman chung).

## 2. Dòng thời gian sự kiện (một UAV)

Mọi thời điểm theo **tick khung phát lại** f = 0, 1, 2, … ở nhịp khoá trước (đề xuất 10 khung/s, lấy mỗi 3 khung gốc). VisDrone-VID không công bố fps, nên nhịp là giả định và được báo cùng kết quả.

| Bước | Nơi | Việc | Được dùng thông tin gì |
|---|---|---|---|
| 1 | UAV | Chụp khung f | Khung của nó |
| 2 | UAV | YOLO11n (ms đo thật, R1a: 26 / 50 ms CPU proxy); ước lượng chuyển động nếu biến thể M1 / M2 (ORB + LK + RANSAC, ms đo thật) | Khung của nó |
| 3 | UAV | Tracker cục bộ → tập L_f | Khung, detection của nó |
| 4 | UAV | Ước lượng niềm tin của BS \hat T_f, tuỳ điều kiện thông tin (mục 3) | **Chỉ** các thông điệp downlink đã **tới** trước tick f |
| 5 | UAV | Điểm lệch g_f(L_f, \hat T_f), luật hành động, kiểm hạn mức uplink | Như trên |
| 6 | UAV → kênh | Đóng gói, xếp hàng uplink, mất gói p_ul, deadline D_max | — |
| 7 | BS | Giải mã nếu tham chiếu còn nguyên; YOLO11s trên phần ảnh; gộp với token (gộp hiệu chỉnh theo nguồn, khoá ở addendum R1a); cập nhật bảng track → phiên bản v+1 | Gói đã tới |
| 8 | BS → kênh | Phản hồi theo điều kiện thông tin: trễ d_dl, mất p_dl, có kịch bản lặp và đảo thứ tự | — |
| 9 | UAV | Áp dụng phản hồi hợp lệ; xác nhận phiên bản ở header uplink kế tiếp | — |

**Không rò rỉ:** UAV không bao giờ đọc bảng track thật của BS, kết quả khớp GT hay bit tương lai. Trạng thái thật của BS chỉ được ghi để tính chỉ số; riêng điều kiện D được đọc nó, và D không triển khai được.

## 3. Ablation thông tin phản hồi (thí nghiệm chính của H1 / H2)

**Giữ nguyên ở mọi điều kiện:**
- khung, detector UAV, detector BS, tracker BS, bộ gộp;
- menu hành động (pilot: SKIP / TOK / FRAME, mục 6);
- lớp policy và thủ tục tinh chỉnh;
- ngân sách uplink và mô hình kênh.

**Chỉ thay:** UAV được thấy gì về trạng thái BS, tức cách tạo \hat T_f.

| Điều kiện | UAV thấy | \hat T_f | Downlink (tính bit) |
|---|---|---|---|
| **A0. Không phản hồi** | Không gì | Bản sao "local-only": UAV tự giả định BS biết đúng những gì UAV đã gửi; với FRAME, giả định BS thấy đúng detection YOLO11n của UAV | 0 |
| **A1. ACK gói** | Gói nào đã tới | Như A0, nhưng chỉ tính phần đã được xác nhận tới | Mỗi khung có gửi: 1 ACK 32 bit |
| **B. Gợi ý kiểu Glimpse** | Danh sách detection BS tìm được trên khung đã xử lý gần nhất. Không id track, không phiên bản | Tracker cục bộ khởi tạo lại từ detection của BS, rồi theo vết bằng khung của UAV | ~ 24 bit header + 50 bit/box |
| **C0. Delta trạng thái, không versioning** | Delta bảng track BS (add / update / delete), best-effort | Bản sao áp delta khi tới | Như C1 trừ phiên bản / băm |
| **C1. Phản hồi ngữ nghĩa đầy đủ** | Delta có phiên bản + băm + snapshot (mục 4) | Bản sao đồng bộ + bộ dự đoán chung | Mục 4 |
| **D. Oracle** | Bảng BS **thật** tại tick f, không trễ, không mất | Chính bảng đó | 0 (không triển khai được) |

**Policy chung** (một lớp tham số, như nhau cho mọi điều kiện):
- **Điểm lệch** g_f = w_n·#mới + w_l·#mất + w_d·#trôi + w_u·diện tích không chắc.
  - "mới": track trong L_f không khớp \hat T_f (IoU < 0,3);
  - "mất": track trong \hat T_f không có trong L_f quá k khung;
  - "trôi": IoU khớp < 0,5;
  - "không chắc": box YOLO11n có conf trong [0,05; 0,25).
- **Luật:** nếu g_f ≥ θ_F và còn hạn mức thì FRAME; ngược lại nếu g_f ≥ θ_T thì TOK, chỉ gửi các đối tượng lệch; còn lại SKIP. Thêm làm mới bắt buộc mỗi N_max khung.
- **Hạn mức uplink:** token bucket ở tốc độ R_ul, độ sâu 1 s. Hành động vượt hạn mức hạ xuống hành động rẻ hơn.

**Tinh chỉnh công bằng:**
- Mỗi điều kiện được tinh chỉnh **riêng**, cùng thủ tục và cùng lưới (θ_T, θ_F, N_max, w, k) trên clip **train**, ở từng mức R_ul.
- Mục tiêu tinh chỉnh: F1 lớn nhất trên train, với bit uplink thực tế ≤ R_ul và downlink ≤ cap.
- Lý do tinh chỉnh riêng: dùng chung ngưỡng tinh cho C sẽ thiệt A, và ngược lại.
- Có thêm một phân tích độ nhạy với **ngưỡng chung**: ngưỡng của A áp cho mọi điều kiện.

**Giá trị gia tăng cần báo:**

| Hiệu | Ý nghĩa |
|---|---|
| F1(C1) − F1(A1) | Giá trị của phản hồi ngữ nghĩa so với ACK gói |
| F1(C1) − F1(B) | Giá trị của trạng thái track so với gợi ý detection |
| F1(C1) − F1(C0) | Giá trị của versioning / băm / snapshot |
| F1(D) − F1(A1) | Dư địa trong lớp policy này |
| F1(D) − F1(C1) | Phần mất do trễ, mất và giá |

Không gộp lợi ích của token, bỏ khung hay thích nghi kênh vào các hiệu này: những thứ đó như nhau ở mọi điều kiện.

**Về cận oracle:** F1(D) − F1(A1) là dư địa của **lớp policy ngưỡng này với thủ tục tinh chỉnh này**. Nó không phải giá trị tối đa của thông tin phản hồi; một lớp policy khác có thể tận dụng thông tin tốt hơn hoặc kém hơn.

## 4. Giao thức đồng bộ trạng thái (điều kiện C1)

**Trạng thái chuẩn** S(v) cho UAV k:
- Bảng track sắp theo id.
- Mỗi track có: id 10 bit, lớp 4 bit, box lượng tử 1/256 khung (32 bit), vận tốc lượng tử (16 bit), conf 3 bit, tuổi.
- Trạng thái **được dự đoán tới tick t_state** bằng bộ dự đoán tất định viết bằng **số học điểm cố định** (số nguyên), để UAV và BS ra **cùng bit**. Số thực dấu phẩy động có thể lệch giữa hai máy.
- **Mã băm** H(v) = CRC32 của chuỗi hoá chuẩn S(v) **tại t_state**. Hai bảng cùng phiên bản nhưng khác nội dung, hoặc cùng nội dung nhưng khác thời điểm, đều phân biệt được.

**Thông điệp downlink:**

```
msg_id(16) | uav_id(4) | base_version(16) | state_version(16) | t_state(16, tick) | frame_processed(16, tick)
| n_ops(8) | ops[...] | H(state_version)(32)                                   header 140 bit
op: track_id(10) | kind(2) | [lớp 4 | box 32 | vận tốc 16 | conf 3]   → 67 bit; xoá 12 bit
snapshot: base_version = 0, toàn bộ bảng
```

**Quay vòng:**
- msg_id, phiên bản và tick dùng **số học tuần tự** kiểu RFC 1982: a "mới hơn" b nếu 0 < (a − b) mod 2^16 < 2^15.
- Tick 16 bit ở 10 khung/s quay vòng sau 109 phút. Một episode giới hạn ≤ 30 phút, nên cửa sổ 2^15 tick (55 phút) luôn đủ.
- Phiên bản tăng tối đa một mỗi tick, nên cũng không vượt cửa sổ.

**Ở UAV:**

| Trường hợp | Xử lý |
|---|---|
| `base == applied` | Áp delta, dự đoán tới t_state, tính CRC. Khớp thì `applied := state_version`; lệch thì đánh dấu lệch và chờ snapshot |
| `state_version` không mới hơn `applied` (lặp, cũ) | Bỏ |
| `base` mới hơn `applied` (thiếu thông điệp giữa) | Giữ trong bộ đệm tối đa W_hold tick; quá hạn thì báo lệch |
| Snapshot | Thay toàn bộ bản sao nếu mới hơn `applied` |

- **Header uplink:** `applied_version(16) | H(applied)(32)`, 48 bit trên mọi gói uplink (kể cả TOK).
- BS so với bản lưu của phiên bản đó; lệch hoặc quá cũ (ngoài N_hist phiên bản) thì gửi snapshot.
- **Snapshot định kỳ** mỗi N_ref tick (đề xuất 20 tick = 2 s) bất kể có lệch.
- **Nội dung đã gửi nhưng chưa được xác nhận:** UAV chạy theo một giả định khoá trước, *thận trọng* (BS chưa có). Giả định *lạc quan* chỉ báo ở phân tích độ nhạy.

**Kiểm thử bắt buộc trước khi chạy** (unit test, commit cùng code):
1. mất / lặp / đảo thứ tự ngẫu nhiên 10⁵ thông điệp → bản sao luôn hội tụ về S(v) sau snapshot kế tiếp;
2. quay vòng 16 bit;
3. băm phát hiện 100% các sai lệch cấy vào (một box lệch 1 đơn vị lượng tử);
4. UAV và BS cùng bộ dự đoán cho cùng bit sau 10³ tick.

## 5. Chi phí, kênh, đóng gói, deadline (chung cho mọi hệ, kể cả x264)

- **Ngân sách theo từng chiều:**
  - uplink R_ul ∈ {3 mức, khoá sau kiểm tra x264 trên VID train};
  - downlink: giới hạn cứng DL_cap ∈ {16; 64} kbit/s.
- **Báo cáo:** bit uplink và downlink báo riêng; không cộng như thể thay thế được cho nhau. Kết quả trình bày dạng Pareto (UL, DL, F1).
- **Đóng gói:** payload chia gói ≤ 1200 byte, mỗi gói thêm 12 byte header ứng dụng + 48 bit header đồng bộ (C0 / C1).
  - Mất gói Bernoulli p_ul = 1% trên uplink; một khung hoặc thông điệp mất nếu mất bất kỳ gói nào (không FEC).
  - Downlink: d_dl, p_dl theo kịch bản (mục 7).
- **Hàng đợi và deadline:**
  - uplink tốc độ R_ul cố định (pilot), sau này dùng trace A5;
  - khung tới BS sau D_max = 300 ms kể từ t_cap thì bị bỏ, nhưng **vẫn tính bit**.
- **Phụ thuộc tham chiếu:**
  - Khung P (VF) chỉ giải mã được nếu tham chiếu đã giải mã.
  - Mất gói làm decoder mất đồng bộ cho tới khung I, hoặc tới khi encoder tham chiếu lại một khung mà BS đã báo giải mã được qua `frame_processed`.
  - x264 baseline dùng **cùng** quy tắc và cùng kênh. Riêng x264 có ACK `frame_processed` 32 bit/khung trên downlink, có tính bit.
- **Tính toán:**
  - ms UAV mỗi khung: YOLO11n, tracker, ước lượng chuyển động, mã hoá;
  - giới hạn ms/khung khoá trước, ví dụ ≤ 100 ms ở 10 khung/s;
  - ms BS.

## 6. Chuyển động camera (có giá)

| Biến thể | UAV gửi | BS biết | Giá |
|---|---|---|---|
| **M0. Không metadata** | Không | Chỉ homography ước lượng từ **các khung đã nhận** (quá khứ). Khi chỉ có SKIP / TOK, BS dự đoán trong toạ độ ảnh | 0 bit; sai số dự đoán tăng khi camera chạy |
| **M1. Metadata mỗi khung** | Homography khung trước → khung này, 8 × 16 bit + 8 bit header = 136 bit, **kể cả khi SKIP** | Chuyển động tới khung hiện tại | SKIP = 136 bit + header gói; ms ORB / LK / RANSAC trên UAV |
| **M2. Metadata khi cần** | Gửi khi dịch chuyển tích luỹ > ngưỡng (tinh chỉnh trên train) | Chuyển động tới lần gửi gần nhất | Như M1 nhưng thưa hơn |

**Quy tắc:**
- BS **không bao giờ** dùng homography của khung tương lai.
- **Pilot H1 dùng M0 cho mọi điều kiện.** Đây là biến thể rẻ nhất và giống nhau giữa các điều kiện. M1 / M2 chạy như phân tích phụ.
- Kết quả báo theo tam phân vị chuyển động camera. Ngưỡng tam phân vị tính trên clip train, bằng homography ước lượng offline, chỉ dùng để phân nhóm.

## 7. Cổng R1b-0 (khả thi, trên 7 clip val; không phải xác nhận thống kê)

**Dữ liệu:**
- tinh chỉnh trên clip VID **train**, trừ các clip bị loại ở kiểm toán rò rỉ;
- cổng chấm trên 7 clip **val**;
- **test-dev chưa tải**, giữ cho xác nhận cuối với protocol khoá lại.

**Kịch bản kênh khoá trước:**

| Mã | d_dl | p_dl | Ghi chú |
|---|---|---|---|
| N (danh định) | 100 ms | 1% Bernoulli | |
| S1 | 300 ms | 1% | |
| S2 | 100 ms | 10% Bernoulli | |
| S3 | 100 ms | 5% trung bình, Gilbert–Elliott, cụm trung bình 5 | Chỉ báo, không là tiêu chí |
| S4 | 100 ms | 1% + 5% lặp + đảo thứ tự | Chỉ báo, không là tiêu chí |

Mức ngân sách: 3 mức R_ul × DL_cap = 64 kbit/s cho tiêu chí; DL_cap = 16 kbit/s chỉ báo.

**H1 (kịch bản N).** "Thắng" một mức R_ul nghĩa là: ΔF1 gộp 7 clip ≥ +2,0 điểm **và** ΔF1 > 0 ở ≥ 5/7 clip.

| Kết luận | Điều kiện |
|---|---|
| **PASS** | Ở ít nhất 2/3 mức R_ul, C1 − A1 thắng **và** C1 − B ≥ 0 gộp **và** không mức nào C1 − A1 < −1 |
| **FAIL (không có dư địa)** | D − A1 < +2 ở mọi mức. Phản hồi không có giá trị ngay cả khi lý tưởng, trong lớp policy này |
| **FAIL (phản hồi có giá không dùng được)** | D − A1 ≥ +2 ở ít nhất một mức, nhưng C1 − A1 < +1 gộp ở mọi mức. Báo rõ và dừng semantic ACK |
| **INCONCLUSIVE** | Các trường hợp còn lại (ví dụ +1 ≤ ΔF1 < +2, hoặc gộp ≥ +2 nhưng < 5/7 clip) |

**H2 (chỉ chạy nếu H1 không FAIL).** Gọi G_N = C1 − A1 trong kịch bản N ở từng mức thắng.

| Kết luận | Điều kiện |
|---|---|
| **PASS** | C1 − A1 ≥ 0,5·G_N ở **cả** S1 **và** S2, ở cùng các mức đó; không kịch bản nào (kể cả S3, S4) có C1 < A1 − 1 |
| **INCONCLUSIVE** | Chỉ một trong S1 / S2 đạt |
| **FAIL** | Cả hai không đạt. Ghi rằng phản hồi ngữ nghĩa chỉ có ích khi downlink tốt |

**H3 (chỉ xây nếu H1 PASS; protocol riêng commit trước).**
- **So với:** biến thể x264 low-delay tốt nhất. Lưới fps × độ phân giải × ROI-QP được chọn trên train, có cùng kênh, đóng gói, deadline và phụ thuộc tham chiếu. Cùng với kiểu Glimpse đầy đủ và kiểu AoIS.
- **Ràng buộc:** cùng R_ul (bit thực tế ±5%), downlink ≤ DL_cap, ms UAV ≤ giới hạn.
- **PASS:** F1 ≥ baseline tốt nhất + 2 ở ≥ 1 mức, không mức nào kém > 2, và cùng dấu ở ≥ 5/7 clip.
- **PASS thay thế:** cùng F1 (±0,5) với uplink ≤ 70% baseline **và** downlink ≤ DL_cap **và** downlink tăng thêm ≤ 25% phần uplink tiết kiệm. Tiết kiệm uplink không được mua bằng downlink.
- **INCONCLUSIVE / FAIL:** định nghĩa tương tự H1.

**Báo cáo bắt buộc:**
- theo **từng clip**: object-time precision / recall / F1, IDF1, thời lượng track sai, độ trễ báo lần đầu (trung vị, p90), UL / DL bit, ms UAV / BS, tỉ lệ khung không giải mã được, belief error (chẩn đoán);
- gộp 7 clip, kèm CI bootstrap theo clip **chỉ để mô tả**.

**Về 5/7:** đó là một tín hiệu khả thi, **không** phải xác nhận thống kê. Kết luận định lượng cần protocol khoá lại và VID test-dev.

## 8. Đối chứng hệ đầy đủ (cho H3)

1. **x264 low-delay CBR** ở cùng bit uplink thực tế: đủ fps; lưới giảm fps / độ phân giải; ROI-QP từ YOLO11n. Cùng YOLO11s và tracker ở BS. SVT-AV1 nếu kịp.
2. **Kiểu Glimpse:** khung kích hoạt theo thay đổi + gợi ý từ server, gợi ý tính bit downlink.
3. **Kiểu MARLIN:** kích hoạt khi theo vết suy giảm hoặc có thay đổi ngoài box.
4. **Kiểu AoIS / AoII:** gửi khi f(tuổi)·g(độ lệch ước lượng) vượt ngưỡng; ước lượng độ lệch như A1.
5. **Token mỗi khung; bỏ K khung định kỳ.**

Mọi ngưỡng tinh chỉnh trên train và khoá trước khi chấm val.

## 9. Pilot tối giản đề xuất (R1b-0a, chỉ kiểm H1)

Mục tiêu: trả lời câu "BS có tracker mạnh hơn; phản hồi trạng thái của nó có làm UAV gửi tốt hơn không?" với ít code nhất.

**Phạm vi:**
- **Một UAV.** Nhịp phát lại 10 khung/s.
- **Dữ liệu:** 7 clip val; clip train sau loại rò rỉ, có thể lấy một tập con cố định theo id để giảm thời gian tinh chỉnh.
- **Menu hành động:**
  - SKIP;
  - TOK: token đối tượng lệch, 48 + 52 bit/box như R1a;
  - FRAME: ảnh độc lập AVIF ở **một** chất lượng khoá trên train.

  Không codec liên khung, nên không có mất đồng bộ decoder. Kết quả H1 vì vậy **chưa** chuyển sang codec video.
- **Chuyển động:** M0.
- **Điều kiện:** A1, B, C1, D. C0 và A0 là phân tích phụ.

**Cách chạy:**
1. **Tính trước một lần (GPU qua lock):** YOLO11n trên mọi khung phát lại; YOLO11s trên mọi khung ở chất lượng FRAME đã khoá; bit AVIF từng khung.
2. **Mô phỏng sự kiện trên CPU từ các bảng đã tính:** kênh, policy, tracker BS, giao thức. Rẻ, nên chạy được đủ ma trận.

**Ước lượng chi phí:**
- val khoảng 950 khung phát lại; train ≤ 8.000;
- detector khoảng 15 ms/khung trên GPU;
- tổng tính trước < 1 giờ GPU;
- mô phỏng vài phút mỗi ô.

Kết quả: một quyết định H1 theo mục 7. Chỉ khi H1 không FAIL mới viết protocol H2 / H3 và xây codec.

**Ma trận thí nghiệm của pilot:**

| Yếu tố | Mức |
|---|---|
| Điều kiện thông tin | A1, B, C1, D (+ A0, C0 phụ) |
| R_ul | 3 mức (khoá sau khi đo bit token / AVIF trên VID train) |
| DL_cap | 64 kbit/s (+ 16 phụ) |
| Kênh | N (+ S1–S4 nếu H1 không FAIL) |
| Chuyển động | M0 (+ M1, M2 phụ) |
| Clip | 7 val, báo từng clip |

## 10. Ma trận chồng lấn với Active Semantic Recovery (chờ người phụ trách)

Không đọc, không sửa nhánh ASR. Câu hỏi gửi người phụ trách, nguyên văn đề xuất của người dùng:

> Mình đang đối chiếu phạm vi SwarmBridge R1b với Active Semantic Recovery. Bên R1b dự kiến UAV chủ động chọn skip/token/video/ROI theo ước lượng trạng thái tracking của BS, với semantic feedback có delay, loss và communication overhead. Nhánh Active Recovery hiện có tracking theo video, receiver feedback về detections/tracks, semantic state synchronization, hay receiver-requested ROI/refinement không? Và đã có cơ chế tương đương trên nhiều UAV chưa? Mình chỉ cần xác nhận phạm vi hiện tại để tránh triển khai trùng nhau, chưa cần truy cập hoặc sửa code của nhánh.

| Cơ chế | C dùng? | ASR có? | Nếu trùng thì C điều chỉnh |
|---|---|---|---|
| ROI / tinh chỉnh do bên nhận yêu cầu | Không. C chủ động phía gửi; snapshot khi lệch | ? | Bỏ snapshot theo yêu cầu, chỉ giữ định kỳ |
| Theo dõi video / đồng bộ trạng thái | Có (trung tâm) | ? | C phải khác ở chi phí / lỗi phản hồi hoặc ở ablation thông tin |
| Chọn token / ảnh / hybrid | Có (H3) | ? | Giữ H1 / H2 làm đóng góp chính |
| Phản hồi detection / track từ BS | Có, có giá | ? | Như trên |
| Nhiều UAV | Chỉ ở R2 | ? | Không ảnh hưởng R1b |

**Chưa code R1b-0 cho tới khi phạm vi ASR được xác nhận.** Chuẩn bị dữ liệu được làm độc lập.

## 11. Checklist trước khi code / chạy R1b-0a

| # | Việc | Trạng thái |
|---|---|---|
| 1 | Câu trả lời ASR, điền mục 10; người dùng xác nhận phạm vi | Chờ |
| 2 | VID train / val: nguồn, checksum, clip id, schema nhãn, thống kê track, kiểm tra rò rỉ với DET (pHash + xem tay) | Xem `reports/visdrone-vid-data-audit.md` |
| 3 | Clip VID trùng DET-train bị loại khỏi val (hoặc thay bằng clip train sạch, ghi rõ); quy tắc loại khoá trước khi xem kết quả | Sau mục 2 |
| 4 | Ghi rõ giao thức huấn luyện: detector train trên VisDrone-DET-train, dùng trên VID. Đây là protocol nghiên cứu riêng, **không** claim tuân thủ challenge. Dữ liệu chỉ dùng cho nghiên cứu, không phân phối lại khung | Ghi trong báo cáo dữ liệu |
| 5 | Khoá: nhịp phát lại, chất lượng FRAME, 3 mức R_ul, DL_cap, D_max, p_ul, kịch bản N / S1–S4, giới hạn ms UAV | Sau khi đo bit trên VID train |
| 6 | Khoá: lớp policy, lưới tinh chỉnh, định nghĩa A0 / A1 / B / C0 / C1 / D, định dạng và số bit của mọi thông điệp downlink | Mục 3–4, chờ duyệt |
| 7 | Tracker BS và UAV (ByteTrack-kiểu IoU + Kalman điểm cố định), bộ dự đoán chung | Chưa code |
| 8 | Unit test chỉ số: GT làm detection cho F1 = IDF1 = 100; trễ tổng hợp cho độ trễ báo lần đầu đúng giá trị cấy | Chưa code |
| 9 | Unit test giao thức (mục 4, bốn phép thử) | Chưa code |
| 10 | Đo ms trên UAV: YOLO11n (đã có), ORB / LK / RANSAC (cho M1 / M2) | Chưa |
| 11 | Protocol R1b-0a (mục 7 + 9 đã khoá số) commit **trước** khi chạy | Chưa |
| 12 | Cho H3 (sau): x264 / ffmpeg low-delay, bit theo khung, chọn tham chiếu, ROI-QP trên WSL | Để sau H1 |

## 12. Phạm vi (nói thẳng)

- **R1b một UAV:** không có đóng góp swarm.
- **R2 với 5 luồng độc lập** chủ yếu là lập lịch nhiều nguồn chia kênh.
- Muốn thành nghiên cứu swarm, sau khi C qua pilot vẫn cần **một quyết định hợp tác thực sự** giữa các UAV, ví dụ góc nhìn chồng lấn (ứng viên B của R0).
- **Độ mới chưa được xác lập:** Glimpse, DDS / server-driven, AoIS, video analytics và hybrid semantic communication đều rất gần. Phần có thể mới là **giá trị gia tăng đo được của phản hồi trạng thái có giá** (H1 / H2). Phần này chỉ đáng claim nếu H1 / H2 PASS và lặp lại trên test-dev.
- **Kết quả R1a** (ngưỡng giao ~0,3 Mbit/ảnh) đo trên ảnh nén độc lập, **không** mang sang R1b; phải đo lại với codec liên khung.
