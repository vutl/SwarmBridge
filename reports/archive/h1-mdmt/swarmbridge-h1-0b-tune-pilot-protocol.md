# SwarmBridge H1-CROP: protocol pilot 0B-TUNE (bản nháp để duyệt)

Ngày 12/11/2026. Theo phản biện thứ bảy: CROP được "GO có điều kiện" để chuẩn bị một pilot rất nhỏ trên TUNE, dùng detector và crop thật.

**Trạng thái:**
- Chỉ là protocol: chưa có code pilot, chưa tải ảnh, chưa chạy gì.
- **Bị chặn bởi quyền dùng ảnh MDMT** (mục 8).
- Không có HOLD, không có tuyên bố xác nhận.
- Không mở Gate 1, không viết simulator mới.
- A5/A5.5 (PPO) giữ nguyên làm nền RL.

## 1. Câu hỏi

1. **Giá trị nhiệm vụ thật:** khi UAV tự sinh ứng viên crop mà không dùng GT, mã hoá AVIF thật, và BS chạy YOLO11s thật trên crop, thì so với chỉ gửi token, F1 có tăng **sau khi tính FP**, ở ngân sách và thời gian tính toán thực tế không?
2. **Phối hợp hai UAV:** trong các crop ứng viên đó, có bao nhiêu là thừa vì UAV kia cũng đề xuất cùng vật, hoặc đã phát hiện chắc chắn vật đó? Nếu bỏ chúng thì F1 tăng bao nhiêu, khi liên kết vẫn lý tưởng nhưng có tính phí?

Câu 2 chỉ có ý nghĩa nếu câu 1 cho thấy crop có giá trị.

## 2. Dữ liệu và split

- **MDMT train**, chỉ khi có quyền (mục 8).
- **Quy tắc HOLD, ghi ngay bây giờ, trước khi thấy danh sách hay ảnh nào:** một chuỗi thuộc HOLD nếu `int(blake2b(tên_chuỗi, digest_size=8), 16) % 4 == 0`. HOLD không được mở trong pilot.
- **Pilot-TUNE:** 6 cặp chuỗi ngoài HOLD, chọn bằng hash nhỏ nhất theo cùng hàm. Lấy 1 epoch mỗi 10 khung, tối đa 60 epoch mỗi chuỗi, tức khoảng 720 ảnh.
- **Trước khi dùng ảnh:** kiểm pHash với VisDrone-DET train (cùng nhóm phát hành), như audit VID. Chuỗi trùng thì loại.
- **Nhãn GT chỉ dùng để chấm**, và để dựng oracle liên kết **có đánh dấu** cho câu 2.

## 3. Đường ống (nhân quả, không GT)

**Trên mỗi UAV, mỗi epoch:**
1. **Detector:** YOLO11n @640 (huấn luyện VisDrone, có sẵn ở R1a), ánh xạ lớp person / car / bicycle.
2. **Token:** detection có conf ≥ τ_hi = 0,25, mỗi token 52 bit (wire format R1b).
3. **Ứng viên crop (không GT):**
   - **P1 (chính):** mỗi detection không chắc (0,05 ≤ conf < 0,25) cho một cửa sổ bằng box × 1,5, tối thiểu 64 px. Các cửa sổ chồng nhau với IoU > 0,5 được gộp.
   - **P2 (độ nhạy):** ô 320 px chứa ít nhất một detection không chắc, theo cách chọn ô của R1a.
   - **Chưa làm:** đề xuất theo chuyển động, vì cần bù chuyển động camera. Ghi là giới hạn.
4. **Mã hoá crop:** AVIF thật (Pillow / libavif, speed 6, 4:2:0, như `r1b_precompute`) ở độ phân giải gốc, q ∈ {35, 50}. Đo bit thật và thời gian mã hoá CPU từng crop.
5. **Thứ tự gửi trong ngân sách:** token trước. Crop sắp theo conf giảm dần (vật không chắc nhưng conf cao có khả năng là thật hơn).

**Ở BS:**
6. YOLO11s (VisDrone, có sẵn ở A5) chạy trên **crop đã giải mã**, imgsz = cạnh crop làm tròn lên bội 32, tối đa 640. Box đưa về toạ độ ảnh gốc.
7. **Gộp từng góc:** token + detection từ crop, NMS theo lớp với IoU 0,6 (như R1a). Khai báo nếu conf ≥ τ_decl. τ_decl được chọn trên pilot-TUNE, và ghi rõ là đã chọn trên cùng tập.

**Thời gian truyền:**
- Không có hàng đợi và không có simulator mới.
- Báo **airtime** = bit / R_ul ở các mức tốc độ, và đánh dấu epoch nào có airtime vượt chu kỳ epoch.

## 4. Điều kiện so sánh (cùng ngân sách tổng B của 2 UAV)

| Mã | Mô tả | Vai trò |
|---|---|---|
| **T0** | Chỉ token, mỗi UAV độc lập; tỉ lệ chia ngân sách α ∈ {0,3; …; 0,7} chọn trên pilot-TUNE | Baseline token |
| **T0+** | Như T0, nhưng hạ ngưỡng: gửi thêm token cho detection không chắc (conf ≥ 0,05) thay vì crop | Đối thủ trực tiếp của crop: "gửi thêm token yếu" so với "gửi ảnh vùng đó" |
| **T1** | Token + crop P1, mỗi UAV độc lập | Kiểm câu 1 |
| **T1-P2** | Như T1 với ô P2 | Độ nhạy |
| **F** | Khung AVIF q35 toàn ảnh (như R1b), YOLO11s ở BS, khi ngân sách đủ | Baseline ảnh mạnh ở ngân sách cao |
| **T2-oracle** | **Oracle liên kết hoàn hảo có tính phí:** như T1, nhưng UAV2 bỏ crop của ứng viên mà GT cho biết (a) UAV1 cũng đề xuất cùng vật, hoặc (b) UAV1 đã có token conf ≥ 0,25 cho vật đó. Bản kê của cả hai UAV (24 bit/track) và bitmap được trừ vào B | Kiểm câu 2. Ứng viên không khớp GT nào (FP) không bao giờ bị bỏ |

Mức B được cố định bằng quy tắc, không chọn theo kết quả:
- đo dòng "token + mọi crop P1" của cả hai UAV trên pilot-TUNE;
- B = {0,25; 0,5; 1,0} × trung bình dòng đó mỗi epoch;
- thêm một mức đủ cho F.

## 5. Chỉ số

1. **F1 theo lớp, từng góc** (Hungarian theo lớp, IoU ≥ 0,5), kèm P và R. Không cần liên kết xuyên góc. Dùng cho câu 1.
2. **F1 đối tượng duy nhất theo lớp** (bộ chấm ở protocol v2, mục 3, với FP_wrong / FP_dup / FP_assoc), dùng **liên kết lý tưởng ở BS cho mọi điều kiện** và được ghi rõ như vậy. Dùng cho câu 2.
3. **Chi phí:**
   - bit uplink thực đạt (payload + header + bản kê);
   - bit downlink;
   - số **ứng viên sai** mỗi khung (không khớp GT nào) và bit của chúng;
   - thời gian CPU trên UAV mỗi epoch (YOLO11n + tổng mã hoá crop), đo một luồng, median và p90;
   - thời gian YOLO11s ở BS;
   - airtime.
4. **Thống kê:** theo chuỗi (6 chuỗi), trung vị và khoảng. **Không có CI xác nhận:** đây là sàng lọc.

## 6. Tiêu chí sàng lọc (đăng ký trước; không phải chân lý)

Các ngưỡng là **tiêu chí sàng lọc đã đăng ký**. Chúng cho biết pilot có đủ tín hiệu để đề xuất bước tiếp không, chứ không chứng minh crop vô dụng khi dưới ngưỡng.

| | Đo | Ngưỡng sàng lọc |
|---|---|---|
| C1 | Tỉ lệ GT (từng góc) không có detection conf ≥ 0,25 khớp | ≥ 15% |
| C2 | Recall của ứng viên P1 trên tập ở C1 (ứng viên chứa tâm GT, hoặc IoU ≥ 0,3); kèm số ứng viên sai mỗi khung | ≥ 50% |
| C3 | F1 từng góc: max_B [F1(T1) − max(F1(T0), F1(T0+))] | > 0 ở ít nhất một mức B, và không phải chỉ nhờ tăng recall kèm giảm precision mạnh (báo cả P và R) |
| C4 | Tỉ lệ ứng viên crop khớp GT mà ở góc kia cũng được đề xuất hoặc đã có token chắc; và F1 duy nhất (T2-oracle) − F1 duy nhất (T1) | ≥ 20%, và hiệu số > 0 ở ít nhất một mức B |
| C5 | Thời gian CPU trên UAV mỗi epoch (YOLO11n + mã hoá crop) | median ≤ chu kỳ epoch (khai báo 1 s); báo p90 |

**Đọc kết quả:**
- **C1–C3 và C5 qua:** crop có giá trị cho từng UAV. Nếu C4 cũng qua thì đề xuất 0B xác nhận trên HOLD (cần duyệt riêng) và Gate 1.
- **C1–C3 qua nhưng C4 trượt:** crop có ích cho từng UAV, nhưng phối hợp ít giá trị. Không mở hướng phối hợp crop.
- **C1, C2 hoặc C3 trượt:** dừng H1-CROP trong thiết lập này và báo nguyên nhân.
- **C5 trượt:** phải đưa ràng buộc tính toán vào thiết kế (bài học R1b), trước khi nói gì về giá trị.

## 7. Ngoài phạm vi pilot

- HOLD, CI xác nhận.
- Liên kết xuyên góc thật (Gate 1).
- Hàng đợi, mất gói, ARQ.
- RL / PPO.
- N ≥ 3 UAV.
- Đề xuất theo chuyển động.

## 8. Điều kiện để khởi động pilot

1. **Quyền dùng ảnh MDMT:** cần trả lời của tác giả về (i) dùng cho nghiên cứu, (ii) công bố kết quả, (iii) phát hành dữ liệu dẫn xuất. Mẫu thư ở phụ lục của [protocol v2](swarmbridge-h1-gate0-protocol-v2.md).
   - **Hiện chưa có.** Theo những gì tôi biết, thư chưa được gửi.
2. **Đường tải:** bản đầy đủ chỉ trên Baidu (25,8 GB), cần người dùng tải.
3. **Protocol này được duyệt**, kể cả các ngưỡng sàng lọc và quy tắc HOLD.
4. **Trong lúc chờ** (được phép theo phản biện): viết code pilot và test tổng hợp, gồm
   - sinh ứng viên từ detection;
   - hình học crop;
   - chọn theo ngân sách;
   - luật bỏ của T2-oracle;
   - bộ chấm F1 từng góc và F1 duy nhất;
   - kế toán bit (UL, DL, bản kê).

   Code chỉ được chạy trên dữ liệu tổng hợp.
5. **VisionTrack:** chỉ khi được duyệt riêng, `train` có thể dùng làm **đối chứng âm** cho C1–C2 (vật to, nên C1 có lẽ trượt). `test` vẫn đóng.


## 9. Cập nhật dữ liệu (12/11): bản sao MDMT trên HuggingFace

Link Baidu chính thức không tải được. Bản sao **không chính thức** `huggingface.co/datasets/mrhwl123/MDMT` (một zip 16,7 GB, license "other" trỏ về repo gốc) đã được tải về `data/mdmt/`. Thư mục này nằm trong gitignore.

**Kiểm tra:**
- CRC toàn bộ zip: OK.
- Trong zip có checkpoint `.pth` và 2 file config `.py` của mmdet. **Không load, không chạy.**
- **Nhãn:** XML (định dạng CVAT) của 4 chuỗi test (26, 31, 57, 73, cả hai góc) chứa **mọi** box của nhãn test chính thức trên GitHub, sai lệch ≤ 1 px. Khung và ID đánh số từ 0; XML có thêm khoảng 0,1% box.
- **Ảnh:** vẽ box chính thức lên ảnh chuỗi 26, khung 150: khớp ở cả hai góc; ID chung chỉ cùng một xe.

**Bản sao không đủ:** chỉ 33/88 chuỗi có ảnh.

| Tập | Chuỗi có ảnh cả hai drone | Ghi chú |
|---|---|---|
| train | **0/14** (chỉ có drone 1) | Không dùng được cho cặp góc nhìn |
| val | **5/5**: 22, 36, 46, 49, 72 (1.930 cặp khung) | Nhãn chưa ai trong nhóm nhìn |
| test | **12/14** (71, 73 thiếu drone 2) | Nhãn là EXPL (đã thấy ở Gate 0A) |

**Hệ quả:** pilot-TUNE theo mục 2 (6 cặp chuỗi MDMT train) **không làm được** với bản sao này. Quy tắc HOLD theo hash vẫn giữ cho bản đầy đủ sau này.

**Đề xuất sửa mục 2 (chờ duyệt):**
- **Phương án B (khuyến nghị):**
  - pilot sàng lọc trên **12 cặp chuỗi test (EXPL)**: cùng dữ liệu với Gate 0A, nên so được trực tiếp dư địa lý tưởng với giá trị thật trên cùng chuỗi;
  - **giữ 5 chuỗi val chưa mở** làm tập kiểm tra nhỏ;
  - tập xác nhận thật vẫn cần bản đầy đủ.
- **Phương án A:** pilot trên 5 chuỗi val (chưa thấy nhãn). Sạch hơn, nhưng nhỏ, và mất tập chưa mở duy nhất.

Quyền dùng dữ liệu vẫn **chưa rõ**: không có license, và đây là bản sao không chính thức. Đề nghị vẫn gửi thư hỏi tác giả.

## 10. Cập nhật 12/11 sau phản biện #8

- **Phương án B được áp dụng:** pilot chỉ trên chuỗi test (EXPL); 5 chuỗi val vẫn đóng (adapter chặn).
- **Quy tắc căn chỉnh** (ghi trước khi mở ảnh): chỉ dùng chuỗi mà ở cả hai góc, khung có nhãn lớn nhất bằng số ảnh − 1. Chuỗi 55 bị loại, còn **11 cặp**, 459 epoch.
- **Khung f ↔ ảnh thứ f theo thứ tự file**, không phải file f+1 (6 chuỗi có số file không bắt đầu từ 1).
- Chi tiết cài đặt và giới hạn: [báo cáo sẵn sàng](swarmbridge-h1-0b-pilot-readiness.md).
