# RQ1-E E0: kết quả

Ngày 14/11/2026, theo phản biện #24. Protocol: [swarmbridge-rq1e-preregistration.md](swarmbridge-rq1e-preregistration.md) (commit 936959b8, trước khi tính embedding). Ảnh MDMT thật, xử lý nội bộ; không train, không mở val / HOLD.

**Cơ sở dùng dữ liệu:** theo lời người dùng, giáo sư hướng dẫn cho phép lab dùng MDMT cho nghiên cứu nội bộ (xác nhận miệng, do người dùng thuật lại; chưa có văn bản; tác giả chưa trả lời; mirror không chính thức). Ảnh, crop và embedding nằm ngoài git (`~/work/h1_pilot0b/rq1e/`); repo chỉ có kết quả tổng hợp theo chuỗi.

## 1. Phán quyết

| Câu hỏi | Kết quả |
|---|---|
| **A.** Đặc trưng ngoại hình có cho phép liên kết không GT đáng tin cậy không? | **KHÔNG. E0 FAIL (0/8 chuỗi đạt) → NO-GO cho pipeline DINOv2-S zero-shot này trên MDMT** |
| **B.** Truyền có chọn lọc có cải thiện F1 vật thể duy nhất so với các phương án cùng chi phí không? | **KHÔNG ĐƯỢC ĐÁNH GIÁ.** E1 không chạy vì E0 FAIL, đúng protocol |

- Không thử backbone khác, không tìm ngưỡng khác, không học mô hình, đúng điều đã khoá.
- Kết quả này không nói gì về novelty truyền thông đa UAV.

## 2. Đo tốc độ thật (RTX 3070, CPU 6 lõi)

| Bước | Thời gian |
|---|---|
| Giải mã 616 khung JPEG từ zip | 9,2 s |
| Cắt và tiền xử lý 21.321 crop | 24,7 s |
| DINOv2-S trên GPU (21.321 crop) | 38,6 s (552 crop/s; batch theo khung) |
| **Tổng** | **72,6 s** (chưa tính nạp mô hình) |

Memo ước "dưới 1 phút GPU"; con số đó gần đúng cho riêng phần mô hình (38,6 s), còn toàn bộ quy trình mất 72,6 s.

**Ghi chú vận hành:** `uav3d_comm/gpulock_run.py` không chạy được vì trình Python Windows của `.venv` (uv trampoline) không khởi động. Tôi đã kiểm GPU rảnh, rồi giữ **đúng file khoá** `runs/.swarmbridge-gpu.lock` theo cùng cơ chế msvcrt, bằng một trình Python Windows khác. Code UAV3D không bị sửa.

## 3. E0 với embedding đầy đủ 384 chiều (nửa đầu; FREE-APPEARANCE)

| Chuỗi | Đăng ký hợp lệ | Độ chính xác ứng viên | Không kiểm được | Độ phủ ứng viên | Lý do thất bại | Đạt |
|---|---|---|---|---|---|---|
| 26 | 0% | 1,00 | 75% | 0,01 | < 8 ứng viên ở 15/15 epoch | không |
| 31 | 0% | 0,94 | 40% | 0,03 | < 8 ứng viên ở 18/18 | không |
| 34 | 0% | 0,83 | 0% | 0,06 | < 8 ứng viên ở 15/15 | không |
| 52 | 0% | 1,00 | 41% | 0,02 | < 8 ứng viên ở 18/18 | không |
| 56 | 0% | 0,89 | 10% | 0,04 | < 8 ứng viên ở 20/20 | không |
| 57 | 0% | 0,00 | 17% | 0,00 | < 8 ứng viên ở 24/24 | không |
| 61 | 0% | 0,86 | 22% | 0,06 | < 8 ứng viên ở 13/13 | không |
| 62 | 0% | 0,92 | 8% | 0,03 | < 8 ứng viên ở 30/30 | không |

- Không epoch nào có đủ 8 cặp ứng viên, nên không có homography, không có sai số chẩn đoán và không có ghép cuối.
- Ở những chỗ có ứng viên, chúng thường đúng (0,83–1,00, trừ chuỗi 57), nhưng chỉ phủ 0–6% số cặp lý tưởng. Đây đúng là trường hợp "vài cặp chính xác" mà tiêu chí độ phủ được đặt ra để chặn.

**Bản nén** (số chuỗi đạt): 64 bit 0/8; 128 bit 0/8; 256 bit 0/8; 3.072 bit 0/8. Không có độ rộng chính cho E1.

## 4. Chẩn đoán sau (post-hoc; không đổi phán quyết, không dùng ngưỡng mới)

Câu hỏi: nút thắt là phép thử tỉ số (ngưỡng 0,8) hay đặc trưng không phân biệt được?

| Chuỗi | Detection mỗi góc | Cặp lý tưởng / epoch | Ứng viên / epoch | Láng giềng hai chiều (không phép thử tỉ số) / epoch | Độ chính xác của chúng | Vật thật là láng giềng gần nhất | Trong 5 gần nhất | Thứ hạng trung vị |
|---|---|---|---|---|---|---|---|---|
| 26 | 45 / 25 | 11,1 | 0,3 | 6,6 | 0,21 | 32% | 65% | 3 |
| 31 | 50 / 63 | 35,3 | 1,7 | 13,3 | 0,51 | 18% | 39% | 11 |
| 34 | 13 / 44 | 6,0 | 0,4 | 3,7 | 0,46 | 47% | 76% | 2 |
| 52 | 55 / 85 | 45,6 | 1,2 | 13,2 | 0,41 | 23% | 47% | 7 |
| 56 | 10 / 21 | 9,1 | 0,5 | 3,6 | 0,49 | 32% | 70% | 3 |
| 57 | 23 / 31 | 20,2 | 0,3 | 5,5 | 0,19 | 13% | 42% | 8 |
| 61 | 24 / 68 | 16,2 | 1,4 | 9,5 | 0,52 | 40% | 63% | 2 |
| 62 | 17 / 18 | 11,8 | 0,4 | 4,7 | 0,58 | 41% | 83% | 2 |

- Vật tương ứng thật chỉ là láng giềng gần nhất trong 13–47% trường hợp.
- Kể cả khi bỏ phép thử tỉ số, các cặp láng giềng hai chiều chỉ đúng 19–58%, dưới tỉ lệ inlier 50% mà bộ đăng ký đòi hỏi.
- Vậy nút thắt là **đặc trưng**, không phải ngưỡng: CLS của DINOv2-S, trên crop nhỏ nhìn từ trên xuống (phóng to lên 224 px) của nhiều xe giống nhau, không phân biệt được cùng một vật giữa hai góc.

## 5. Điều kết quả này nói và không nói

**Nói:**
- Trên MDMT, cả hai cách liên kết không GT đã khoá đều không dùng được:
  - chỉ hình học, trên bản ghi cache: Phase 0, NOT IDENTIFIABLE;
  - ngoại hình zero-shot DINOv2-S: E0, NO-GO.
- Tiền đề của RQ1 (BS khó tự ghép hai góc) càng có cơ sở. Nhưng các công cụ không GT hiện có không đủ để **đo** được phần mất mát và giá trị của việc truyền thêm.

**Không nói:**
- Không nói rằng mọi đặc trưng ngoại hình đều thất bại. Một mô hình Re-ID được **học** cho drone (như ReID của MIA-Net, chưa phát hành) có thể khác, nhưng nó cần train và không được duyệt.
- Không đánh giá được câu hỏi B (truyền chọn lọc).

## 6. Đề xuất

- **Dừng RQ1 trên MDMT** với các công cụ zero-shot. Không thử backbone hay ngưỡng khác.
- Hai đường còn lại đều là quyết định mới, không tự mở:
  1. dữ liệu có metadata camera (CrossTrack-UAV, chờ thỏa thuận) cho liên kết theo pose;
  2. một mô hình Re-ID được học cho ảnh drone, cần duyệt train và một tập train không trùng chuỗi đánh giá.

## 7. Tệp

- Kết quả tổng hợp: `runs/uav_semcom/h1/rq1e_e0_v1.json` (E0), `runs/uav_semcom/h1/rq1e_e0_diag_v1.json` (chẩn đoán sau). Cả hai chỉ có số liệu theo chuỗi.
- Code: `swarmbridge/experiments/uav_semcom/h1_rq1e.py`, `test_h1_rq1e.py` (8 test tổng hợp). Script chẩn đoán không đưa vào repo vì nó là phân tích một lần sau khi đã có phán quyết; số liệu của nó nằm ở JSON trên.
- Ngoài git: embedding và thời gian đo ở `~/work/h1_pilot0b/rq1e/`; trọng số DINOv2-S ở `~/work/models/dinov2-small/`.
