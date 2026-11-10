# VisDrone-VID train / val: kiểm tra dữ liệu (chuẩn bị cho R1b; chưa train, chưa pilot)

Ngày 12/11/2026. Theo chỉ đạo ngày 12/11: tải train và val từ nguồn chính thức, **không** tải test-dev.

**Trạng thái (10/11): ĐÃ TẢI VÀ KIỂM TOÁN.**
- `gdown` bị chặn hạn mức suốt 5 lần thử. Người dùng tải bằng trình duyệt từ đúng hai link Drive chính thức.
- Kết luận chính (mục 5): **4/7 clip VID-val chứa khung trùng hệt ảnh DET-train**. Theo quy tắc khoá ở mục 4, cổng R1b-0 chuyển sang 7 clip VID-train sạch (G7).

## 1. Nguồn và điều khoản (đã kiểm)

| Mục | Nội dung | Nguồn |
|---|---|---|
| Link tải train | Google Drive, file id `1NSNapZQHar22OYzQYuXCugA3QlMndzvw`, ghi 7,53 GB; BaiduYun `pan.baidu.com/s/1kC3NTK6MPVv3D1CY9gXaCQ` | README github.com/VisDrone/VisDrone-Dataset |
| Link tải val | Google Drive `1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B`, 1,49 GB; BaiduYun `pan.baidu.com/s/12-A6Mg1Gg7hyS4WwG27dDw` | như trên |
| Giấy phép | README **không có** mục license hay terms of use. Câu duy nhất: "Researchers can use test-dev to publish papers" | như trên |
| Mục đích dùng | Toolkit: "This code library is for research purpose only". Trang evaluate: dữ liệu phát hành cho "academic purpose" | github.com/VisDrone/VisDrone2018-VID-toolkit; aiskyeye.com/evaluate/ |
| Trích dẫn | README: Zhu et al., "Detection and tracking meet drones challenge", TPAMI 44(11), 2021. Toolkit: "Vision Meets Drones: A Challenge", arXiv 1804.07437 | cả hai |
| Định dạng nhãn | `<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>`; target_id liên kết box qua các khung | toolkit README |
| Số liệu công bố | Toolkit: train 56 clip / 24.201 khung; val 7 / 2.819. README dataset (ghi chú R0): 24.198 / 2.846. **Hai nguồn chính thức lệch nhau**; sẽ đếm thực tế | cả hai |

**Quy tắc challenge về dùng dữ liệu giữa các task:**
- Tôi **không tìm thấy** quy tắc riêng cho VID về việc dùng dữ liệu DET để huấn luyện trên các trang đã đọc: README, toolkit, aiskyeye.com/challenge, /evaluate, /download, trang Task 1 năm 2022.
- Trang Task 1 (ảnh) ghi: "allow them to use additional training data" và "The use of external data and pre-trained models must be indicated during submission".
- Các trang khác của aiskyeye (submit-2023, faq, challenge-2023) chưa đọc, nên không loại trừ được một quy tắc ở đó.
- **Dù vậy:** detector của SwarmBridge train trên VisDrone-DET-train và dùng trên VID. Đây là **protocol nghiên cứu riêng**. Báo cáo sẽ mô tả chính xác dữ liệu huấn luyện và **không** claim tuân thủ challenge protocol.

**Cách dùng:**
- Chỉ cho nghiên cứu.
- Không phân phối lại khung; dữ liệu nằm ngoài git, ở `~/data/visdrone_vid` trong WSL.
- Ảnh ghép xem tay nằm ở `~/work/vid_audit/side`, cũng ngoài git.

## 2. Kiểm toán sẽ chạy (`swarmbridge/experiments/uav_semcom/vid_audit.py`)

1. **Cấu trúc:** clip id, số khung mỗi clip, độ phân giải, khung liên tục, chỉ số khung trong nhãn khớp số ảnh.
2. **Schema nhãn:**
   - 10 cột;
   - miền giá trị của score {0,1}, lớp 0–11, truncation, occlusion;
   - box không dương;
   - dòng (khung, target_id) lặp;
   - đếm theo lớp.
3. **Thống kê track** (lớp được chấm 1–10, score = 1): số track mỗi clip; độ dài track (trung vị, p10, max); track có khoảng hở và số khung hở; số vật mỗi khung (trung bình, max); khung không có vật; tỉ lệ vật nhỏ (< 32² px); số vùng bỏ qua.
4. **Rò rỉ với VisDrone-DET** (DET-train là tập huấn luyện của cả YOLO11n và YOLO11s; DET-val và DET-test-dev là các pool A5 / R1a):
   - (a) **Metadata:** số nguồn trong tên DET (`0000013_…_d_…`) so với số trong tên clip VID (`uav0000013_…`). Hai không gian tên khác nhau, nên trùng số chỉ là gợi ý.
   - (b) **pHash 64 bit** (DCT 32×32, viết bằng numpy): **mọi** khung VID so với **mọi** ảnh DET. Cặp có Hamming ≤ 10 là nghi vấn; xác nhận bằng tương quan chuẩn hoá ảnh xám 64×64 ≥ 0,9.
   - (c) **Xem tay:** ảnh ghép cạnh nhau, tối đa 3 cặp tốt nhất mỗi clip.
   - (d) **Đối chứng:** khung val VID so với khung train VID (mỗi khung thứ 5).
5. **Ghi:** nguồn, sha256 của file zip, kết quả vào `runs/uav_semcom/vid_audit/vid_audit.json`.

**Kiểm thử script** (đối chứng dương, ngày 12/11):
- Một clip giả gồm 5 ảnh DET-val, so với một tập "DET" chứa 3 ảnh đó và 20 ảnh khác.
- Kết quả: 3/3 ảnh trùng được bắt ở Hamming 0, NCC 1,0. Hai khung còn lại cùng chuỗi DET nhưng khác khung có Hamming 20–24, không bị gắn cờ. Khoảng hở 1 khung trong track được đếm đúng.

**Giới hạn của pHash:** bắt được **ảnh gần trùng** (cùng khung, nén lại, đổi cỡ), **không** bắt được "cùng địa điểm, khác thời điểm". Rò rỉ kiểu cảnh chỉ kiểm được một phần, bằng xem tay các cặp Hamming thấp nhất.

## 3. Tải

`scripts/uav_semcom/run_vid_download_audit.sh`:
- tải bằng `gdown` từ đúng file id của README;
- `unzip -t` kiểm tính toàn vẹn;
- sha256;
- giải nén vào `~/data/visdrone_vid`;
- chạy kiểm toán ngay sau đó.

**Thử lại mỗi 30 phút, tối đa 16 lần (8 giờ).** Script nhận luôn file zip đã có sẵn trong thư mục, hoặc trong `C:\Users\USER\Downloads` (tên khớp `*VID*val*.zip` / `*VID*train*.zip`), nếu file qua `unzip -t`.

Nếu Drive vẫn khoá, có hai cách:
- người dùng tải bằng trình duyệt đã đăng nhập Google vào thư mục Downloads;
- tải từ BaiduYun (cần tài khoản).

## 4. Quy tắc xử lý rò rỉ (khoá trước khi xem kết quả)

- **Clip val** có ≥ 1 khung trùng DET-train đã xác nhận (Hamming ≤ 10 và NCC ≥ 0,9, xem tay đồng ý): bị **loại khỏi cổng R1b-0**.
  - Nếu còn < 5 clip val sạch, cổng chuyển sang 7 clip lấy từ VID-train sạch, chọn theo id trước khi xem kết quả. Các clip này bị loại khỏi tập tinh chỉnh.
- **Clip train** trùng DET-train: vẫn được dùng để tinh chỉnh ngưỡng; ghi rõ.
- **Trùng với DET-val / DET-test-dev:** chỉ ghi lại, vì đó không phải dữ liệu huấn luyện detector.

**Sự cố trong script (đã sửa, 9679bee5):** WSL Ubuntu không có `unzip`. Vì vậy bước kiểm zip luôn thất bại và script âm thầm quay lại thử Drive, mất khoảng 1,5 giờ. Đã thay bằng `zipfile` của Python (`scripts/uav_semcom/zip_tool.py`). Script cũng nhận zip đặt trong `data/visdrone/` (git-ignored).

## 5. Kết quả (10/11/2026)

Số liệu đầy đủ ở `runs/uav_semcom/vid_audit/vid_audit.json`.

### 5.1 Nguồn và tính toàn vẹn

| File | Kích thước | sha256 | Kiểm CRC |
|---|---|---|---|
| VisDrone2019-VID-train.zip | 8.080.572.990 B | `3d7ddd2216524c5a12dcb55cee5bedf99f4625f7093bbf446072f40b8d9c1ea6` | 24.316 mục, 0 lỗi |
| VisDrone2019-VID-val.zip | 1.602.397.759 B | `0e1b44481ad701cec088089ab9630b957b2bf247a759c62eba1bec9a887d3865` | 2.863 mục, 0 lỗi |

Nguồn: hai file id Google Drive ở mục 1, tải bằng trình duyệt ngày 10/11. Dữ liệu giải nén ở `~/data/visdrone_vid` (WSL), ngoài git.

### 5.2 Cấu trúc và schema

| | train | val |
|---|---|---|
| Clip / khung | 56 / 24.201 | 7 / 2.846 |
| Khớp số công bố? | khớp toolkit (24.201); README ghi 24.198 | khớp README (2.846); toolkit ghi 2.819 |
| Khung / clip | 58–1.424 | 184–978 |
| Khung liên tục | 56/56 | 7/7 |
| Lỗi schema | 0 | 0 |
| Độ phân giải | 6 loại, 1344×756 đến 2720×1530 | 5 loại, 1344×756 đến **3840×2160** |
| Track (lớp 1–10, score 1) | 7.342 | 758 |
| Trung vị độ dài track (trung vị theo clip) | 126,5 khung | 138 khung |
| Track có khoảng hở | 506 (6,9%) | 86 (11%) |
| Vật / khung (trung bình theo clip) | 49,1 | 49,1 |
| Tỉ lệ vật nhỏ < 32² px | 0,21 | 0,19 |
| Khung không có vật | 3 | 0 |

Schema có 10 cột. Mọi giá trị nằm trong miền:
- score {0,1}, lớp 0–11, truncation và occlusion {0,1,2};
- không có box không dương;
- không có dòng (khung, target_id) lặp.

Các lỗi thường gặp trên dữ liệu VisDrone vì vậy không xuất hiện ở bản này.

**Thống kê 7 clip val:**

| Clip | Khung | Độ phân giải | Track | Trung vị độ dài track | Vật/khung | Vật nhỏ | Track có hở | Vùng bỏ qua |
|---|---|---|---|---|---|---|---|---|
| uav0000086_00000_v | 464 | 1344×756 | 82 | 322 | 49,8 | 0,29 | 2 | 780 |
| uav0000117_02622_v | 349 | 2720×1530 | 145 | 75 | 49,5 | 0,08 | 6 | 1.184 |
| uav0000137_00458_v | 233 | 2688×1512 | 180 | 140 | 103,4 | 0,01 | 26 | 273 |
| uav0000182_00000_v | 363 | 1344×756 | 155 | 96 | 48,7 | 0,49 | 9 | 769 |
| uav0000268_05773_v | 978 | 3840×2160 | 52 | 164,5 | 13,9 | 0,14 | 7 | 0 |
| uav0000305_00000_v | 184 | 1904×1071 | 69 | 87 | 34,7 | 0,11 | 33 | 0 |
| uav0000339_00001_v | 275 | 1904×1071 | 75 | 138 | 43,7 | 0,21 | 3 | 988 |

### 5.3 Rò rỉ với VisDrone-DET

**(a) Metadata.**
- Số nguồn trong tên DET và tên VID là **cùng một không gian**, chứ không chỉ trùng số ngẫu nhiên. Mọi cặp trùng hệt ở (b) đều có cùng số nguồn, ví dụ khung VID `uav0000339_00001_v/0000197` = DET `0000339_00393_d_0000163`. Ảnh DET là khung lấy từ cùng video gốc.
- 44/55 số nguồn VID có mặt trong DET-train (493 ảnh DET-train); 16 trong DET-val (116 ảnh); 1 trong DET-test-dev (nguồn 278, 4 ảnh).

**(b) pHash.** Mọi khung VID (27.047) so với mọi ảnh DET: 6.471 train, 548 val, 1.610 test-dev.

| Tập DET | Khung VID nghi (Hamming ≤ 10) | Xác nhận (NCC ≥ 0,9) | Ảnh DET bị trùng | Clip VID có khung xác nhận |
|---|---|---|---|---|
| DET-train | 6.655 | 3.653 | 108 | **39 train + 4 val** |
| DET-val | 1.424 | 630 | 23 | 7 train + 2 val |
| DET-test-dev | 1 | 0 | 0 | 0 |

Số khung xác nhận lớn hơn nhiều số ảnh DET, vì các khung liền nhau của video gần như giống hệt nên cùng khớp một ảnh DET.

**Đối chiếu (a) với (b) theo clip** (63 clip):

| | Có khung trùng | Không có khung trùng |
|---|---|---|
| Có số nguồn trong DET-train | 43 | 8 |
| Không có | 0 | 12 |

**(c) Xem tay ảnh ghép.** Đã xem các cặp tốt nhất của mọi clip val nghi vấn, cùng các ca biên.
- **Val 339, 182, 137, 305 (Hamming 0, NCC 0,99–1,0):** cùng một khung ảnh. Trùng thật.
- **Ca biên `train/266_04830` (NCC 0,87) và `train/329_04715` (NCC 0,74):** không trùng khung, nhưng **cùng camera, cùng giao lộ, khác thời điểm**. Đây là rò rỉ cảnh mà ngưỡng NCC 0,9 bỏ sót.
  - 8 clip "có số nguồn nhưng không có khung trùng" ở bảng trên thuộc loại này: đoạn khác của cùng video gốc.
- **`train/326_01035` (NCC 0,90):** cùng cảnh, xe đã đổi chỗ. Được xác nhận nhờ NCC, dù không phải cùng khung.
- **`train/295_02300` so với DET `0000348_…` (NCC 0,27):** hai cảnh khác hẳn. NCC loại đúng; pHash một mình cho dương tính giả.

**(d) Đối chứng VID-val so với VID-train:** chỉ val 339 có 7 khung gần trùng một khung train (clip này đã bị loại vì DET-train).

### 5.4 Áp quy tắc mục 4

1. **Loại khỏi cổng:** val **339, 182, 137, 305** (trùng hệt DET-train, xem tay đồng ý). Còn **3** clip val sạch: 086, 117, 268. 3 < 5, nên **cổng chuyển sang 7 clip VID-train sạch**.
2. **Định nghĩa "sạch" cho clip cổng — chặt hơn mục 4, ghi rõ:**
   - không có khung xác nhận trùng DET-train, **và**
   - không có số nguồn trong DET-train.

   Điều kiện thứ hai được thêm **sau khi** xem tay (mục 5.3c) cho thấy rò rỉ cảnh. Nó chỉ siết quy tắc theo hướng an toàn và không dùng kết quả R1b nào (R1b chưa chạy).

   Có 9 clip train đạt: 020_00406, 138_00000, 244_01440, 278_00001, 289_00001, 289_06922, 295_02300, 316_01288, 360_00001.
3. **G7 = 7 clip có id nhỏ nhất trong 9 clip đó** (quy tắc cơ học, không nhìn thống kê):
   - `uav0000020_00406_v`
   - `uav0000138_00000_v`
   - `uav0000244_01440_v`
   - `uav0000278_00001_v`
   - `uav0000289_00001_v`
   - `uav0000289_06922_v`
   - `uav0000295_02300_v`

   Hai clip của nguồn 289 cùng nằm trong G7, nên không nguồn nào vừa ở cổng vừa ở tập tinh chỉnh. Tổng cộng 3.453 khung (khoảng 1.150 tick ở 10 fps).
4. **Tinh chỉnh:** 49 clip train còn lại (47 clip trùng DET-train hoặc cùng nguồn + 316, 360), ghi rõ theo mục 4.
5. **3 clip val sạch** (086, 117, 268): chạy và báo từng clip như **kết quả phụ**, không quyết định cổng.
6. **Chỉ ghi lại** (không phải dữ liệu huấn luyện detector):
   - G7 có trùng DET-val: 244 (223 khung xác nhận), 289_00001 (103), 289_06922 (11), 295 (14);
   - 278 có số nguồn trong DET-test-dev (4 ảnh, không có khung trùng);
   - val 086 và 117 trùng DET-val (57 và 41 khung).

   DET-val là pool đánh giá A5/R1a, nên các con số R1a và R1b **không độc lập hoàn toàn về cảnh**.

**Hệ quả:**
- G7 gồm clip train, không phải split val chính thức. Kết quả R1b-0 vì vậy **không so trực tiếp được** với số liệu VID-val công bố.
- Vẫn còn rò rỉ cảnh kiểu "cùng thành phố, khác địa điểm" mà pHash không đo được.
- Kết luận định lượng cuối vẫn cần VID test-dev, và test-dev cũng phải qua đúng kiểm toán này trước khi dùng.
