# VisDrone-VID train / val: kiểm tra dữ liệu (chuẩn bị cho R1b; chưa train, chưa pilot)

Ngày 12/11/2026. Theo chỉ đạo ngày 12/11: tải train và val từ nguồn chính thức, **không** tải test-dev.

**Trạng thái: CHƯA TẢI ĐƯỢC.** Google Drive chính thức trả lỗi hạn mức tải ("Too many users have viewed or downloaded this file recently … may take up to 24 hours"). Một vòng thử lại đang chạy (mục 3). Kết quả kiểm toán sẽ điền vào mục 5 khi có dữ liệu.

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

## 5. Kết quả

*Chờ tải. Sẽ điền: sha256, số clip / khung thực tế so với số công bố, độ phân giải, lỗi schema, bảng thống kê track theo clip, kết quả rò rỉ (a)–(d) kèm nhận xét xem tay, danh sách clip bị loại theo mục 4.*
