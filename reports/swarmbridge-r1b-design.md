# SwarmBridge R1b: thiết kế đã sửa (đề xuất, chưa phải protocol khoá)

Ngày 10/11/2026. Sửa ứng viên C của `reports/swarmbridge-next-scenario-r0.md` theo phản biện ngày 10/11:
- ACK gói ≠ ACK ngữ nghĩa;
- codec video liên khung phải là đối chứng;
- Glimpse/MARLIN đã có các cơ chế gần.

Ghi chú đọc paper (tiếng Anh, có số trang) ở `reports/r0-notes/r1b-feedback-video.md`.

**Chưa** tải VisDrone-VID, chưa code R1b, chưa train RL. Bản này cần người dùng duyệt, và cần hỏi nhóm ASR (mục 8), trước khi khoá protocol.

## 1. Câu hỏi nghiên cứu (đã sửa)

> Khi hiểu biết của BS chỉ được đồng bộ **một phần, có trễ và tốn tài nguyên**, UAV nên gửi **loại bằng chứng nào** (không gửi, token, khung có vùng ưu tiên, khung đầy đủ) và **khi nào**, để BS duy trì phát hiện và theo dõi mục tiêu tốt nhất, trong một mạng nhiều UAV tranh chấp kênh?

Hai điều đã có tiền lệ và **không** phải đóng góp:
- "UAV biết BS đã biết gì": Glimpse gửi gợi ý từ server về thiết bị; AoIS/AoII lập lịch theo độ sai của bên nhận.
- Khung kích hoạt theo thay đổi (Glimpse, MARLIN).

Phần có thể khác biệt (mục 6):
- phản hồi ngữ nghĩa **có tính chi phí, có trễ, có mất**, từ một bên nhận **mạnh hơn**;
- lựa chọn giữa **các họ biểu diễn**;
- nhiều UAV dùng chung kênh;
- chấm bằng ground-truth trên video drone thật.

## 2. Mô hình hệ thống đã sửa

**Bên nhận (BS):**
- Chạy YOLO11s trên nội dung ảnh nhận được (khung, vùng), dùng token như detection.
- Duy trì **bảng track T_v có số phiên bản v** cho mỗi UAV: Kalman vận tốc hằng + ghép IoU, tham số cố định.

**Phản hồi ngữ nghĩa (semantic ACK), thay cho ACK gói:**
- Sau mỗi lần cập nhật, BS gửi về UAV một **delta tích luỹ** so với phiên bản UAV đã xác nhận, nên áp dụng nhiều lần vẫn đúng khi mất gói:
  - header khoảng 48 bit (id UAV, phiên bản, khung uplink đã xử lý, cờ);
  - mỗi track thay đổi khoảng 52 bit (id track, thao tác thêm/sửa/xoá, lớp, box lượng tử 1/256, mức conf);
  - xoá khoảng 14 bit.
  - Cỡ điển hình 0,3–1,1 kbit mỗi thông điệp.
- Tối đa **một thông điệp đang bay**, cộng **làm mới toàn trạng thái** mỗi N_ref (ví dụ 1 s).
- UAV gửi kèm trên uplink một mã băm 16–32 bit của phiên bản nó tin. Nếu lệch, BS gửi làm mới.
- **Kênh xuống:** trễ d_dl ∈ {20; 100; 300} ms cộng thời gian suy luận ở BS; mất gói p_dl ∈ {0; 1; 5; 10}% (Bernoulli, thêm Gilbert–Elliott nếu kịp).
- **Bit xuống được tính** cùng ngân sách với bit lên. Báo cả hai và tổng.

**Bản sao ở UAV:**
- UAV chạy **cùng bộ dự đoán tất định** như BS (Kalman, tham số cố định) trên phiên bản T_v cuối đã nhận. Từ đó nó biết BS **đang tin gì ở thời điểm hiện tại** mà không cần chạy lại detector của BS.
- Các gửi đi chưa được xác nhận: báo cả hai cách giả định, **thận trọng** (BS chưa áp dụng) và **lạc quan** (BS đã áp dụng).
- **Không có ground-truth hay trạng thái ẩn của BS trong quan sát của UAV.** Sai lệch thật giữa bản sao và BS (belief error, ghép IoU) được đo, nhưng chỉ để báo cáo.

**Hành động uplink mỗi khung:**
- **SKIP:** không gửi.
- **TOK:** token YOLO11n, dạng delta so với bản sao.
- **VF(q):** khung video **mã hoá liên khung** bởi một encoder low-delay (x264, không B-frame, zerolatency). Encoder chỉ nhận các khung được chọn gửi, nên khung tham chiếu là khung gửi trước đó. Như vậy gửi khung đã hưởng dư thừa thời gian một cách công bằng.
- **VF-ROI(q_bg, q_roi):** như VF, nhưng bản đồ QP lấy từ box không chắc / vật nhỏ của YOLO11n (ROI trong cùng codec, kiểu AccMPEG/EAAR). **Không gửi ô ảnh độc lập**, vì DDS cho thấy trên video VisDrone ảnh vùng độc lập thua video H.264.
- **KEY:** khung khoá (I-frame) khi cảnh đổi lớn hoặc bản sao lệch nhiều.

**Tính toán trên UAV:**
- YOLO11n bị giới hạn tốc độ theo mức năng lực (ví dụ 5 / 10 lần/s, lấy từ số đo R1a).
- Giữa hai lần suy luận dùng tracker rẻ (LK / IoU).
- Thời gian suy luận và mã hoá tính vào deadline.

**Nhịp video:** VisDrone-VID không công bố fps. Phải **tuyên bố nhịp phát lại** (ví dụ 10 khung/s), với độ nhạy ở 5 và 25.

## 3. Chỉ số

- **Recall hiện thời ở BS:** tỉ lệ (đối tượng GT, khung) mà danh sách track BS **đang giữ** có box khớp IoU ≥ 0,5, cùng lớp. Báo riêng vật nhỏ.
- **Độ trễ báo lần đầu** của mỗi track GT: trung vị và p90.
- IDF1 nếu kịp.
- Track sai mỗi khung.
- Bit lên, bit xuống, tổng.
- ms tính toán trên UAV.
- Belief error.
- Chấm theo **nhãn VisDrone**, không theo detection của YOLO11s. DDS dùng DNN làm ground-truth, cách đó che lỗi của detector ở server.

## 4. Đối chứng (xếp theo mức nguy hiểm cho C)

1. **Codec video ở cùng bitrate thực tế:** x264 low-delay CBR, thêm SVT-AV1 nếu kịp. BS chạy YOLO11s + tracker trên mọi khung giải mã. Ba biến thể:
   - đủ fps;
   - lưới giảm fps / độ phân giải (kiểu AWStream);
   - ROI-QP từ box của YOLO11n.
2. **Kiểu Glimpse:** theo vết trên UAV, gửi khung khi dịch chuyển đặc trưng hoặc thay đổi pixel so với khung đã gửi vượt ngưỡng; BS gửi gợi ý (detection) về. **Bit gợi ý được tính.**
3. **Kiểu MARLIN:** kích hoạt khi tương quan trong box theo dõi < 0,3 hoặc có thay đổi ngoài box. Bộ phát hiện thay đổi đơn giản hoá; ghi rõ.
4. **Kiểu AoIS / AoII:** gửi khi f(tuổi) · g(độ lệch ước lượng) vượt ngưỡng. Ở R2 nhiều UAV: lập lịch tham lam theo AoIS lớn nhất.
5. **Định kỳ:** bỏ qua K khung; gửi mọi khung ở chất lượng cố định.
6. **Hai cận để đo giá trị của phản hồi:**
   - **local-only:** UAV giả định BS biết đúng những gì UAV đã gửi, chỉ dùng ACK gói;
   - **oracle-belief:** UAV biết chính xác trạng thái BS, miễn phí.

Mọi ngưỡng tinh chỉnh trên chuỗi train, khoá trên val, chấm test-dev một lần.

## 5. Dữ liệu

- **VisDrone-VID:** 96 clip.

  | phần | clip | khung | dung lượng tải |
  |---|---|---|---|
  | train | 56 | 24 198 | 7,53 GB |
  | val | 7 | 2 846 | 1,49 GB |
  | test-dev | 17 | 6 635 | 2,14 GB |

  - Nhãn có id track.
  - Chỉ có câu "dùng cho mục đích nghiên cứu", không có văn bản giấy phép.
  - Ổ WSL còn khoảng 870 GB.
- **Rò rỉ:** cả YOLO11s (BS) và YOLO11n (UAV) đều train trên VisDrone-DET-train. Không có tuyên bố chính thức nào về việc DET và VID có chung cảnh quay. **Trước khi chấm:** kiểm tra gần-trùng (perceptual hash) giữa khung VID val/test-dev và ảnh DET-train; loại clip trùng, ghi rõ.
- **Cỡ mẫu:** test-dev chỉ có 17 clip, nên CI theo clip sẽ rộng. Nói trước để không diễn giải quá mức.

## 6. Điều còn khác biệt (nói thẳng)

- Mọi thành phần đều có tiền lệ:
  - khung kích hoạt, gợi ý từ server, theo vết che độ trễ: Glimpse, MARLIN;
  - độ tươi có trọng số theo độ sai: AoII, AoIS (Han 2025; Zhou JWCN 2026);
  - ROI trong codec: AccMPEG, EAAR;
  - lập lịch nhiều UAV: Zhou JWCN 2026.
- Không tìm thấy trong các nguồn đã kiểm:
  - phản hồi ngữ nghĩa **được tính bit, có trễ, có mất**, từ một bên nhận mạnh hơn bên gửi. Glimpse, DDS, EAAR trả kết quả nhưng không tính chi phí. AoII/AoIS giả định ACK hoàn hảo. Zhou cho bên gửi biết trạng thái bên nhận miễn phí;
  - lựa chọn **họ biểu diễn** theo khung, chấm bằng ground-truth trên video drone.
- Đây là **tổ hợp**, độ mới trung bình.
- R1b chỉ có **một UAV**. R2 với 5 UAV phát lại 5 video độc lập vẫn là **tối ưu uplink dùng chung nhiều UAV**, chưa phải hợp tác nhận thức sâu. Phải nói rõ giới hạn này khi viết.

## 7. Cổng R1b (đề xuất, sẽ khoá cùng protocol)

**Bước 0: giá trị của phản hồi.** Trên val, ở 2–3 mức bit khớp nhau, đo oracle-belief, local-only và Glimpse-hint, cùng x264 ở cùng bit thực tế.
- Khoảng cách oracle-belief − local-only là **giá trị tối đa của phản hồi ngữ nghĩa**.
- Nếu khoảng cách < 1 điểm recall hiện thời: không cần thiết kế semantic ACK. C thu về local-only + chọn biểu diễn; ghi nhận và tiếp tục với câu hỏi chọn biểu diễn.

**PASS chính.** C dùng phản hồi có tính chi phí (d_dl = 100 ms, p_dl = 1%) phải hơn **cả** biến thể x264 tốt nhất **và** kiểu Glimpse ở cùng tổng bit (lên + xuống), theo một trong hai cách:
- recall hiện thời cao hơn ≥ +3 điểm;
- hoặc cùng recall với ≤ 70% bit.

Kèm điều kiện:
- p90 độ trễ báo lần đầu không tệ hơn;
- CI theo clip trên test-dev loại 0;
- ở p_dl = 10%, giữ ≥ một nửa lợi ích.

**FAIL:** không hơn x264 **hoặc** không hơn Glimpse ở cùng bit. Khi đó dừng trước R2 và trước mọi RL, báo nguyên nhân.

## 8. Câu hỏi gửi nhóm Active Semantic Recovery (trước khi khoá R1b)

> Bên mình đang thiết kế một hướng trong đó UAV duy trì ước lượng trạng thái theo dõi đối tượng của BS (qua phản hồi ngữ nghĩa có tính chi phí) và chủ động chọn bỏ qua / token / khung có vùng ưu tiên / khung đầy đủ trên video VisDrone-VID. Nhánh Active Semantic Recovery hiện có dùng: (1) yêu cầu ROI hoặc tinh chỉnh do bên nhận khởi xướng; (2) theo dõi video hoặc đồng bộ trạng thái; (3) chọn giữa token / ảnh / hybrid; (4) phản hồi detection/track từ BS; (5) lập lịch nhiều UAV hoặc uplink dùng chung không? Mình muốn biết phần chồng lấn trước khi triển khai để tránh làm trùng.
