# Hướng B: sửa so sánh cùng ngân sách và tính khả thi theo bộ dữ liệu

Ngày 13/11/2026, theo phản biện #19. Không chạy tầng 1 trên MDMT; không tải ảnh; không train.

## 1. Tóm tắt

- **Lỗi so sánh ngân sách của B1 đã sửa.** Mọi phương án giờ chịu cùng một giới hạn bit chung theo từng epoch, cố định từ nửa đầu chuỗi. 17 test tổng hợp qua (152 test cả bộ pilot).
- **MDMT:** sẵn sàng về kỹ thuật, vẫn chờ cơ sở dùng dữ liệu.
- **M3OT:** giấy phép CC BY 4.0 đã xác minh. Nhưng nó **không thay thế trực tiếp** được cho B1: chỉ 3/9 chuỗi qua sàng lọc ID xuyên UAV, vật thể rất nhỏ, một lớp, khung 640 × 512.
- **CrossTrack-UAV:** sát bài toán nhất về mô tả, nhưng phải ký thỏa thuận (cần chữ ký người hướng dẫn) và repo ghi dữ liệu chưa phát hành chính thức.
- **Đề xuất:** hai việc không tính toán, làm song song: xin xác nhận quyền dùng MDMT, và gửi thỏa thuận CrossTrack-UAV. M3OT chỉ nên dùng nếu đổi câu hỏi (bổ sung RGB–hồng ngoại), cần đăng ký riêng.

## 2. Sửa so sánh cùng ngân sách (Sửa đổi 1 của bản đăng ký)

Chi tiết quy tắc nằm ở [swarmbridge-a8-b1-preregistration.md](swarmbridge-a8-b1-preregistration.md), mục "Sửa đổi 1". Tóm tắt:

- **Giới hạn B chung** cho hai UAV, áp cho mọi phương án ở từng epoch.
- **B_ref** lấy từ nửa đầu, chỉ dùng số bit: chi phí lớn nhất của "một FRAME + TOKEN của góc kia". Quyết định ở B = B_ref; các mức 0,25 đến 2 × B_ref cho đường F1–bit.
- **Khi không vừa:** FRAME bị thay bằng TOKEN trong phần ngân sách còn lại. Số bit gửi không bao giờ vượt B; code dừng nếu vi phạm.
- **Thêm baseline `BIG`** (FRAME cho góc có FRAME lớn hơn, không nhìn nhãn) để kiểm trường hợp `VIEW` thắng chỉ vì tiêu nhiều bit hơn.
- **Báo thêm:** bit đạt được, tỉ lệ epoch gửi đúng như yêu cầu, và cách đọc đã định trước khi một quy tắc không cần GT đã bắt kịp `VIEW`.
- Ngưỡng GO (≥ 2 điểm F1, dương ở ≥ 73% chuỗi) không đổi.

Test hồi quy mới (dữ liệu tổng hợp, `test_h1_b1.py`):

| Test | Kiểm điều gì |
|---|---|
| FRAME hai góc lệch 3 lần | tái hiện lỗi: không có giới hạn thì hai phương án "một FRAME" tiêu bit khác nhau gần 3 lần |
| Giới hạn + fallback | FRAME lớn không vừa → gửi TOKEN; F1 và bit trùng với `TT`; không vượt B |
| `FF` dưới giới hạn một FRAME | chỉ một FRAME được gửi |
| Token bị cắt | khi cả token cũng không vừa, bit vẫn ≤ B |
| Bất biến ngẫu nhiên | 40 chuỗi × 6 mức B × 8 phương án: không epoch nào vượt B |
| B_ref | chỉ phụ thuộc bit của nửa đầu; không đổi khi bỏ nhãn hoặc khi FRAME nửa sau to gấp đôi |
| Chọn góc dưới giới hạn | ở 0,5 × B_ref, FRAME có ích không vừa nên `VIEW` không được hưởng giá trị của nó |
| Đối chứng `BIG` | khi `VIEW` trùng góc với `BIG`: không GO, và được ghi là "quy tắc không cần GT là đủ" |

## 3. Khảo sát bộ dữ liệu

### 3.1 MDMT

- Bản ghi pilot đã có; 8/11 chuỗi qua sàng lọc ID; tầng 1 chạy offline trong vài phút.
- **Chặn:** chưa có xác nhận quyền dùng từ tác giả hoặc lab / trường.

### 3.2 M3OT

Nguồn đã kiểm: bản ghi Figshare (DOI 10.6084/m9.figshare.28308887), bài mô tả trên Scientific Data (DOI 10.1038/s41597-025-06204-0), và các tệp nhãn. Tôi chỉ đọc mục lục của tệp zip và lấy 115 tệp nhãn (23 MB) bằng yêu cầu theo dải byte; **không tải ảnh nào**.

**Đã xác minh:**
- Giấy phép dữ liệu: CC BY 4.0 (ghi trên bản ghi Figshare); bài báo cũng CC BY 4.0.
- Cấu trúc: 2 UAV × 2 kiểu ảnh (RGB, hồng ngoại) × 9 chuỗi (7 train, 1 val, 1 test); 21.581 ảnh PNG, 10,7 GB.
- Khung 640 × 512, 10 khung/giây, chuỗi dài 300–1.365 khung; hai UAV có cùng số khung ở mọi chuỗi.
- Một lớp duy nhất: `vehicle`. Nhãn dạng MOT (`gt.txt`), kèm COCO JSON.
- Vật thể rất nhỏ: kích thước box trung vị khoảng 8–22 × 11–25 điểm ảnh. Số vật mỗi khung: từ 2–3 (chuỗi 03) đến 36–40 (chuỗi 09).
- Tập ID của hai UAV trùng nhau ở mọi chuỗi (ví dụ 83 / 83, 35 / 35); ID của RGB và hồng ngoại cũng trùng.
- RGB và hồng ngoại **không** khớp từng điểm ảnh: lệch trung vị 6–16 điểm ảnh giữa box cùng ID.

**Bài báo không nêu quy ước ID xuyên UAV.** Câu duy nhất liên quan: "We set a different ID number for each target and checked to make sure there were no duplicate ID labels." Vì vậy phải kiểm bằng dữ liệu, như với MDMT.

**Sàng lọc ID xuyên UAV** (chỉ nhãn; cùng phương pháp với MDMT; dung sai 13,3 điểm ảnh = 40 × 640 / 1920, cố định trước khi chạy; kết quả ở `runs/uav_semcom/h1/m3ot_id_screen_v1.json`):

| Chuỗi | Split | Vật chung mỗi khung | Inlier trung vị (ID giữ lại) | Bản tráo | Đạt |
|---|---|---|---|---|---|
| 01 | train | 6,4 | 0,00 | 0,00 | không |
| 02 | train | 4,3 | không đủ vật để kiểm | | không |
| 03 | test | 1,5 | không đủ vật để kiểm | | không |
| 04 | train | 7,7 | 0,58 | 0,00 | không |
| 05 | train | 10,3 | 0,00 | 0,00 | không |
| 06 | train | 7,4 | 0,71 | 0,00 | có |
| 07 | train | 8,3 | 0,73 | 0,00 | có |
| 08 | val | 7,6 | 0,00 | 0,00 | không |
| 09 | train | 33,6 | 1,00 | 0,00 | có |

- **3/9 chuỗi đạt**, đều thuộc train.
- Chẩn đoán thêm sau khi xem kết quả (không thuộc quy tắc định trước):
  - Ở 01, 05, 08, khớp hình học trên toàn bộ ID chung không khác bản tráo (0,31 so với 0,31; 0,50 so với 0,40; 0,30 so với 0,30), kể cả khi nới dung sai lên 40 điểm ảnh hay dịch thời gian 1–5 giây.
  - Nhưng cùng một ID có mặt cùng lúc ở hai UAV nhiều hơn hẳn cặp ID ngẫu nhiên ở chuỗi 01 (0,52 so với 0,00). Ở 05 và 08 chênh lệch nhỏ (0,53 so với 0,42; 0,45 so với 0,18).
  - Tôi **không kết luận được từ nhãn** rằng ID ở ba chuỗi này sai hay hai góc không cùng nhìn một mặt phẳng. Cần xem ảnh hoặc hỏi tác giả.

**Đánh giá cho B1:**
- Không cắm thẳng vào pilot được. Detector hiện có (YOLO11n / 11s train trên VisDrone) chưa được kiểm trên vật 10–20 điểm ảnh và ảnh hồng ngoại; gần như chắc phải train lại.
- Khung chỉ 640 × 512, nên chênh lệch bit giữa TOKEN và FRAME nhỏ hơn nhiều so với MDMT (1920 × 1080). Câu hỏi "cấp FRAME cho góc nào" có thể mất ý nghĩa ở kích thước này; chưa đo.
- Chỉ 3 chuỗi được xác nhận ID, ít hơn 8 của MDMT, nên không đủ đơn vị độc lập cho quy tắc "dương ở ≥ 73% chuỗi".
- Điểm mạnh riêng: hai kiểu ảnh. Một câu hỏi khác (chọn RGB hay hồng ngoại, theo UAV) có thể hợp hơn, nhưng đó là nghiên cứu mới cần đăng ký riêng.

### 3.3 CrossTrack-UAV

Nguồn đã kiểm: README của repo `ccs0302/CrossTrack-UAV` và mẫu thỏa thuận trong thư mục `license`.

**Đã xác minh:**
- Tác giả thuộc KAIST và ADD (Agency for Defense Development, Hàn Quốc). README ghi bài đã được nhận đăng ở IEEE RA-L.
- UAV cánh cố định, bối cảnh quy mô lớn, tầm xa; đánh giá theo dõi xuyên góc (CVMA, CVIDF), tức có ID xuyên góc.
- Truy cập: điền và ký thỏa thuận, gửi email cho tác giả; dữ liệu được gửi lại sau khi xác minh.
- Điều khoản chính của thỏa thuận: chỉ dùng cho học thuật / nghiên cứu / phi thương mại; được dùng cho bài báo và luận văn; cấm phân phối lại hoặc đăng công khai dữ liệu hay phần đáng kể của nó; cấm chia sẻ đường dẫn tải; phải trích dẫn. Cần chữ ký của người dùng **và của người hướng dẫn**.
- README ghi repo "work in progress" và mã nguồn cùng dữ liệu sẽ phát hành chính thức khi bài được đăng.

**Chưa xác minh được** (chỉ có trong hình hoặc trong tóm tắt của người phản biện): 4 UAV, 53 kịch bản, 1.272 track xuyên góc, metadata GPS / độ cao / hướng camera; độ phân giải; lớp vật thể.

**Đánh giá cho B1:** cơ sở pháp lý rõ nếu ký được. Rủi ro: thời gian chờ, và UAV cánh cố định tầm xa có thể cho vật thể rất nhỏ như M3OT. Vì repo SwarmBridge đang công khai, các bảng theo từng khung phải nằm ngoài git, như đang làm với MDMT.

## 4. Tính khả thi của hướng B

| Bộ dữ liệu | Quyền dùng | ID xuyên góc | Dùng được cho B1 tầng 1 | Việc cần |
|---|---|---|---|---|
| MDMT | chưa xác nhận | 8/11 chuỗi qua sàng lọc | có, ngay khi có quyền (vài phút) | xác nhận của tác giả hoặc lab / trường |
| M3OT | CC BY 4.0 | 3/9 chuỗi qua sàng lọc | không trực tiếp | đổi câu hỏi + train detector + đăng ký mới |
| CrossTrack-UAV | theo thỏa thuận ký | có theo mô tả, chưa kiểm | chưa biết | ký thỏa thuận (cần người hướng dẫn), chờ dữ liệu |

- Hướng B **không chết** nếu MDMT không có quyền, nhưng cũng chưa có bộ thay thế nào dùng được ngay.
- Bước rẻ nhất vẫn là MDMT: mọi thứ đã sẵn, chỉ thiếu xác nhận.
- Không mở nhánh thí nghiệm nào thêm theo yêu cầu của phản biện #19.

## 5. Tệp

- Code B1: `swarmbridge/experiments/uav_semcom/h1_b1_view.py`, `test_h1_b1.py`.
- M3OT: `scripts/uav_semcom/remote_zip_members.py` (đọc mục lục zip và lấy tệp nhãn theo dải byte), `m3ot_id_check.py` (quy tắc định trước), `m3ot_id_diag.py`, `m3ot_id_diag2.py` (chẩn đoán sau); kết quả `runs/uav_semcom/h1/m3ot_id_screen_v1.json`.
