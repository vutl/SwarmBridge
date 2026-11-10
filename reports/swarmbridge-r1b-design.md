# SwarmBridge R1b: thiết kế v2 (đề xuất, chưa phải protocol khoá)

Ngày 11/11/2026. Bản v2 sửa bản v1 (cùng file, xem lịch sử git) theo hai đợt phản biện ngày 10–11/11. Ghi chú đọc paper ở `reports/r0-notes/r1b-feedback-video.md`.

**Chưa** tải VisDrone-VID, chưa code R1b, chưa train RL, chưa mở R2. Trước khi chạy pilot cần đủ điều kiện ở mục 9.

## 1. Câu hỏi nghiên cứu

> Khi BS đã có một trạng thái nhận thức (bảng track), nhưng trạng thái ấy có thể **sai hoặc lỗi thời**, và chỉ được đồng bộ **một phần, có trễ, tốn tài nguyên**: UAV có thể chọn một **bằng chứng nhỏ, đúng lúc, đủ tin cậy** (token, vùng ưu tiên trong khung, khung đầy đủ, hoặc không gửi) để sửa trạng thái đó, thay vì truyền lại cả luồng video không?

R1a cho thấy token và hybrid rất mạnh về recall khi bit chật, nhưng kém ảnh về AP/FP. R1b phải chứng minh lợi ích của **việc đồng bộ hiểu biết của BS**, không chỉ lợi ích quen thuộc của "gửi token thay vì ảnh".

## 2. Dòng thời gian sự kiện (một UAV k)

1. **t_cap:** chụp khung f. Nhịp phát lại được tuyên bố trước (ví dụ 10 khung/s; độ nhạy 5 / 25), vì VisDrone-VID không công bố fps.
2. **Suy luận trên UAV:** YOLO11n nếu còn hạn mức suy luận (mức năng lực: 5 / 10 / 20 lần/s), nếu không thì cập nhật bằng tracker rẻ.
   - Thời gian thật lấy từ R1a: u960 50 ms CPU proxy, u640 26 ms.
   - Ước lượng chuyển động camera (mục 4).
3. **Cập nhật tracker cục bộ** → tập đối tượng UAV thấy L_f.
4. **Dự đoán niềm tin của BS:** từ bản sao (T_v, t_v), cùng bộ dự đoán tất định với BS (mục 3), đưa về thời điểm t_cap.
5. **Độ lệch ước lượng** g_f giữa L_f và niềm tin dự đoán:
   - đối tượng mới chưa có ở BS;
   - track BS có lẽ đã mất (vật biến mất);
   - track trôi (IoU thấp);
   - vùng không chắc.
6. **Chọn hành động:** SKIP / TOK / VF-ROI / VF / KEY (mục 5), qua heuristic có ngưỡng tinh chỉnh trên train.
7. **Mã hoá, xếp hàng uplink, truyền** (kênh: một UAV ở R1b; mô hình A5 ở R2). Có trễ, có thể mất gói p_ul.
8. **BS nhận:** giải mã nếu chuỗi tham chiếu còn nguyên (mục 5), chạy YOLO11s trên phần ảnh, gộp với token bằng cách gộp đã hiệu chỉnh theo nguồn của addendum R1a, cập nhật bảng track → phiên bản v+1.
9. **BS gửi phản hồi ngữ nghĩa** (mục 3) trên downlink: trễ d_dl, mất p_dl, có thể lặp hoặc đến sai thứ tự.
10. **UAV áp dụng phản hồi** nếu hợp lệ, rồi xác nhận phiên bản đã áp dụng trong header của uplink kế tiếp.

## 3. Đồng bộ trạng thái (semantic ACK)

**Trạng thái chuẩn hoá ở BS cho UAV k:**
- Bảng track T, sắp theo id track.
- Mỗi track có: id, lớp, box lượng tử 1/256 khung, vận tốc lượng tử, mức conf, tuổi.
- Phiên bản v tăng đơn điệu, thời điểm trạng thái t_v.
- **Mã băm nội dung** H(T): CRC32 trên chuỗi hoá chuẩn của T, không chỉ trên số phiên bản.
- Hai bảng cùng phiên bản mà khác nội dung sẽ bị phát hiện.

**Thông điệp phản hồi (BS → UAV):**

```
msg_id(16) | uav_id(4) | base_version(16) | state_version(16) | t_state(16, ms mod) | frame_processed(16)
| n_ops(8) | ops[...] | H(T_state)(32)
op: track_id(10) | kind(2: add / update / delete) | [lớp 4 | box 32 | vận tốc 16 | conf 3]  → ~67 bit (xoá: 12 bit)
snapshot: như trên với base_version = 0 và toàn bộ bảng
```

- **Delta tích luỹ** từ phiên bản UAV đã xác nhận áp dụng (base_version) tới phiên bản mới. Mất một thông điệp không làm hỏng các thông điệp sau.
- BS giữ lịch sử phiên bản đủ để tính delta từ bất kỳ base nào trong N bản gần nhất. Quá xa thì gửi snapshot.

**Ở UAV:**

| Trường hợp | Xử lý |
|---|---|
| `base_version == applied` | Áp dụng; so H với băm tính lại. Khớp thì `applied := state_version`. Lệch thì đánh dấu lệch và yêu cầu snapshot |
| `state_version ≤ applied` (lặp, cũ) | Bỏ qua |
| `base_version ≠ applied` và mới hơn (đến sai thứ tự hoặc thiếu) | Giữ lại tới khi đủ, hoặc yêu cầu snapshot sau T_wait |

- Header uplink mang `applied_version(16) | H(bản sao)(32)`.
- BS so với bản lưu của phiên bản đó; lệch thì gửi snapshot.
- **Snapshot định kỳ** mỗi N_ref (ví dụ 2 s) dù không lệch.
- **Gửi đi chưa được xác nhận:** báo hai cách giả định của UAV, thận trọng (BS chưa áp dụng) và lạc quan (BS đã áp dụng).

**Chi phí:**
- Bit downlink (thông điệp + snapshot) và bit uplink (header + nội dung) **báo riêng từng chiều** và báo tổng, vì tài nguyên lên và xuống không hoán đổi 1:1.
- Kịch bản kênh: d_dl ∈ {20; 100; 300} ms; p_dl ∈ {0; 1; 5; 10}% (Bernoulli, thêm Gilbert–Elliott); có lặp và đảo thứ tự ở một kịch bản riêng.

**Không rò rỉ:** quan sát của UAV chỉ gồm khung của nó, detector của nó, và các thông điệp phản hồi đã nhận. Sai lệch thật giữa bản sao và BS (belief error) chỉ dùng để báo cáo.

## 4. Chuyển động của UAV (ego-motion)

UAV bay và quay làm vật đứng yên trên mặt đất dịch chuyển mạnh trong ảnh. Kalman vận tốc hằng trong toạ độ ảnh sẽ dự đoán sai, dễ gây gửi KEY liên tục.

**Ba biến thể, so trực tiếp:**
- **(a) Kalman trong toạ độ ảnh:** baseline ngây thơ.
- **(b) Bù chuyển động:** UAV ước lượng homography giữa các khung liên tiếp (ORB/LK + RANSAC) và gửi kèm tham số tích luỹ (8 số × 16 bit, có tính bit) để BS và bản sao biến đổi track về cùng hệ.
- **(c) Như (b) nhưng homography tính ở BS** từ các khung nhận được. Chỉ dùng được khi có khung, nên là cận.

**Báo riêng theo mức chuyển động camera:** chia clip theo trung vị độ dịch chuyển homography thành tam phân vị (đứng gần yên / vừa / nhiều). VisDrone-VID không có telemetry, nên không dùng pose GPS/IMU.

## 5. Biểu diễn, phụ thuộc liên khung và mất đồng bộ bộ giải mã

**Hành động:**
- **SKIP:** không gửi.
- **TOK:** token delta so với niềm tin dự đoán; độc lập, không phụ thuộc khung khác.
- **VF:** khung P của luồng x264 low-delay (không B-frame, zerolatency). Encoder chỉ nhận các khung được chọn (nhịp khung thay đổi), nên tham chiếu là khung gửi trước.
- **VF-ROI:** như VF với bản đồ QP từ box không chắc / vật nhỏ của YOLO11n. Không gửi ô ảnh độc lập.
- **KEY:** khung I.

**Phụ thuộc và mất gói:** một khung VF chỉ giải mã được nếu khung tham chiếu đã tới.
- Mất gói uplink p_ul làm bộ giải mã ở BS **mất đồng bộ** cho tới KEY kế tiếp, hoặc tới khi encoder tham chiếu lại một khung BS **đã xác nhận giải mã** (reference-picture selection theo phản hồi).
- Trường `frame_processed` trong phản hồi mang thông tin "khung giải mã được gần nhất".
- **Áp dụng như nhau** cho đối chứng x264, để so công bằng.
- Mã hoá ảnh độc lập (AVIF của A5/R1a) chỉ giữ làm đối chứng tham khảo, không phải đối thủ chính.

## 6. Chỉ số (bắt buộc)

| Chỉ số | Định nghĩa |
|---|---|
| **Object-time precision / recall / F1 ở BS** | Ở mỗi khung GT: danh sách track BS đang giữ (sau dự đoán) so với box GT, IoU ≥ 0,5, cùng lớp |
| **Thời lượng track sai** | Tổng giây một track BS tồn tại mà không khớp GT |
| **IDF1** | Id track BS so với id GT của VisDrone-VID |
| **Độ trễ báo lần đầu** | Mỗi track GT: trung vị, p90 |
| **Bit** | Uplink, downlink, tổng |
| **Tính toán** | ms trên UAV mỗi khung và tỉ lệ dùng hạn mức suy luận; ms ở BS |
| Belief error | Chẩn đoán |
| Tỉ lệ khung không giải mã được | Chẩn đoán |

Chấm theo **nhãn VisDrone**. Báo riêng vật nhỏ và theo tam phân vị chuyển động camera.

## 7. Đối chứng

1. **x264 low-delay CBR ở cùng bitrate uplink thực tế:**
   - đủ fps;
   - lưới giảm fps / độ phân giải;
   - ROI-QP từ YOLO11n.

   BS dùng cùng YOLO11s, cùng tracker, cùng mô hình mất gói. SVT-AV1 nếu kịp.
2. **Kiểu Glimpse:** khung kích hoạt theo thay đổi + gợi ý từ server. **Bit gợi ý tính ở downlink.**
3. **Kiểu MARLIN:** kích hoạt khi theo vết suy giảm hoặc có thay đổi ngoài box.
4. **Kiểu AoIS / AoII:** gửi khi f(tuổi) · g(độ lệch ước lượng) vượt ngưỡng.
5. **Token mỗi khung**, chỉ dùng ACK gói (local-only), và bỏ qua K khung định kỳ.
6. **Cận:**
   - **local-only:** UAV giả định BS biết đúng những gì UAV đã gửi;
   - **oracle-belief:** UAV biết chính xác bảng của BS, miễn phí.

   Khoảng cách giữa hai cận là giá trị tối đa của phản hồi ngữ nghĩa.

Mọi ngưỡng tinh chỉnh trên clip train, khoá trên val. Test-dev chỉ dùng cho R1b đầy đủ, **không** dùng trong pilot.

## 8. Ma trận chồng lấn với Active Semantic Recovery (điền khi có câu trả lời)

| Cơ chế | C dùng? | ASR có? (chờ người phụ trách) | Nếu trùng thì C điều chỉnh |
|---|---|---|---|
| ROI hoặc tinh chỉnh do bên nhận yêu cầu | Không; C chủ động phía gửi. Snapshot theo yêu cầu khi lệch | ? | Bỏ snapshot theo yêu cầu, chỉ giữ định kỳ; nhấn vào chọn phía gửi |
| Theo dõi video / đồng bộ trạng thái | Có (trung tâm) | ? | Nếu ASR có đồng bộ trạng thái, C phải khác ở chi phí/lỗi phản hồi hoặc ở họ biểu diễn |
| Chọn token / ảnh / hybrid | Có | ? | Nếu trùng, C giữ phần đồng bộ trạng thái làm đóng góp chính |
| Phản hồi detection/track từ BS | Có, có tính chi phí | ? | Như trên |
| Lập lịch nhiều UAV / uplink chung | Chỉ ở R2 | ? | Không ảnh hưởng R1b |

## 9. Cổng khả thi R1b-0 (pilot nhỏ trên video VAL)

Cổng chỉ dùng clip train (tinh chỉnh) và 7 clip val (đánh giá pilot), **không** dùng test-dev. Policy C ở pilot là heuristic đơn giản, khoá trước.

| Câu hỏi | PASS nếu (val, ở d_dl = 100 ms, p_dl = 1%) |
|---|---|
| **Q1.** Bản sao BS có giảm sai lệch so với local-only và gợi ý kiểu Glimpse không? | Belief error giảm ≥ 30% so với local-only, **và** object-time F1 ≥ local-only + 2 điểm ở cùng tổng bit (lên + xuống), **và** không thấp hơn kiểu Glimpse ở cùng bit |
| **Q2.** Lợi ích còn khi tính chi phí và trễ phản hồi không? | Ở d_dl = 300 ms **hoặc** p_dl = 10%, còn ≥ 50% phần lợi F1 của Q1 |
| **Q3.** So với x264 ở cùng bitrate thực tế và ngân sách tính toán, chọn biểu diễn có đáng đầu tư không? | F1 của C ≥ biến thể x264 tốt nhất + 2 điểm ở ít nhất 1 trong 3 mức bitrate, hoặc cùng F1 với ≤ 70% bit uplink |

- Vì val chỉ có 7 clip, mỗi điều kiện còn phải **cùng dấu ở ≥ 5/7 clip**. CI được báo nhưng không làm tiêu chí.
- **FAIL Q1 hoặc Q3:** dừng C trước R1b đầy đủ, báo nguyên nhân.
- **Chỉ FAIL Q2:** ghi rằng phản hồi chỉ có ích khi kênh xuống tốt, rồi quyết định cùng người dùng.

## 10. Điều kiện tối thiểu trước khi chạy pilot R1b-0

1. **Câu trả lời của nhóm ASR**, điền vào ma trận mục 8, và người dùng xác nhận phạm vi.
2. **Kiểm tra VisDrone-VID:**
   - điều khoản: chỉ có câu "research purpose", ghi rõ khi dùng;
   - link tải: Google Drive / Baidu trên README chính thức; chưa thử tải;
   - dung lượng: train 7,53 GB, val 1,49 GB, test-dev 2,14 GB; ổ WSL còn khoảng 870 GB;
   - định dạng nhãn: khung, id, box, score, lớp, truncation, occlusion.
3. **Kiểm tra rò rỉ:** perceptual hash giữa khung VID (train, val, test-dev) và ảnh VisDrone-DET-train (tập train của cả hai detector). Clip nào trùng thì loại, hoặc chỉ dùng cho train. Cần tải VID trước. Đề nghị tải **val và train** trước (khoảng 9 GB), test-dev sau, khi người dùng cho phép.
4. **Addendum R1a** đã chốt cách gộp có hiệu chỉnh theo nguồn (khoá trên val) để dùng ở BS.
5. **Protocol R1b-0** (mục 9, cùng nhịp phát lại, mức năng lực, kịch bản kênh) được commit **trước** khi chạy.
6. **Kiểm tra công cụ:** x264 / ffmpeg low-delay, bản đồ QP (addroi), đếm bit theo khung trên môi trường WSL.

## 11. Phạm vi (nói thẳng)

- **R1b chỉ một UAV:** không có đóng góp swarm.
- **R2 với 5 luồng video độc lập** chủ yếu là bài toán lập lịch nhiều nguồn chia kênh.
- Muốn thành nghiên cứu swarm intelligence, sau khi C qua pilot vẫn cần xác định **một quyết định hợp tác thực sự giữa các UAV** dựa trên dữ liệu có cơ sở, ví dụ góc nhìn chồng lấn (ứng viên B của R0).
- **Độ mới chưa được xác lập:** Glimpse, AoIS, video analytics và hybrid semantic communication đều rất gần.
