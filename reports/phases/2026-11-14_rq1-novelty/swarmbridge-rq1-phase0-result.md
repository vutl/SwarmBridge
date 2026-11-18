# RQ1 Phase 0: kết quả và báo cáo quyết định

Ngày 14/11/2026, theo phản biện #23. Protocol: [swarmbridge-rq1-association-preregistration.md](swarmbridge-rq1-association-preregistration.md) v2 (commit 2c287e86, trước khi chạy). Chạy offline trên bản ghi pilot 0B, **chỉ các epoch nửa đầu**; không ảnh mới, không detector, không train. GT chỉ dùng làm thước đo chẩn đoán.

**Cơ sở dùng dữ liệu:** theo lời người dùng, giáo sư hướng dẫn cho phép lab dùng MDMT (xác nhận miệng, do người dùng thuật lại; chưa có văn bản; tác giả chưa trả lời; dữ liệu từ mirror không chính thức).

## 1. Quyết định

| Câu hỏi | Kết quả theo tiêu chí đã khoá |
|---|---|
| **RQ1-A** (liên kết có phải nút thắt?) | **NOT IDENTIFIABLE FROM EXISTING RECORDS**: chỉ 3/8 chuỗi đăng ký được (cần ≥ 6/8) |
| **RQ1-B** (crop thật có giải quyết được?) | **NOT IDENTIFIABLE FROM EXISTING RECORDS**: điều kiện tiên quyết RQ1-A không đạt (độ phủ crop thì đạt, 80% > 25%) |

- Phase 1 và Phase 2 **không chạy**, đúng protocol. Không có GO / NO-GO bằng số.
- Bộ ước lượng **không được chỉnh** sau khi xem kết quả.
- Đây **không phải NO-GO** cho RQ1: nó chỉ nói rằng bản ghi hiện có, cùng bộ liên kết không GT đã khoá, không đủ để đo câu hỏi.

## 2. Kết quả theo chuỗi (nửa đầu, TOKEN / TOKEN)

| Chuỗi | Epoch | Hợp lệ | Sai số chẩn đoán trung vị (px) | Epoch "hợp lệ" nhưng sai > 100 px | Token trung bình mỗi góc | Đăng ký được |
|---|---|---|---|---|---|---|
| 26 | 15 | 27% | 420 | 4 / 4 | 45 / 25 | không |
| 31 | 18 | 100% | 18 | 0 / 18 | 50 / 63 | **có** |
| 34 | 15 | 47% | 676 | 7 / 7 | 13 / 44 | không |
| 52 | 18 | 100% | 15 | 3 / 18 | 55 / 85 | **có** |
| 56 | 20 | 30% | 124 | 3 / 6 | 10 / 21 | không |
| 57 | 24 | 67% | 80 | 4 / 16 | 23 / 31 | không |
| 61 | 13 | 62% | 623 | 8 / 8 | 25 / 68 | không |
| 62 | 30 | 100% | 5 | 0 / 30 | 17 / 18 | **có** |

Sai số chẩn đoán: chiếu tâm đáy box GT góc 1 bằng homography không GT, lấy trung vị khoảng cách tới box GT góc 2. Ngưỡng đăng ký được: hợp lệ ở ≥ 50% epoch **và** sai số trung vị ≤ 40 px.

**Lý do thất bại** (ngoài các epoch hợp lệ):
- không đủ 8 cặp láng giềng hai chiều trong 40 px;
- homography suy biến.

**Độ phủ crop** của 4.130 detection mơ hồ (gộp nửa đầu): 80% nằm trong một crop P1 hoặc ô P2 đã tính sẵn; chỉ 27% nếu tính riêng crop P1.

## 3. Điều Phase 0 cho biết (khám phá, chẩn đoán)

- **Liên kết chỉ bằng hình học, không có cặp tương ứng, rất mong manh trên MDMT.** Bộ ước lượng không GT chỉ đăng ký đúng ổn định ở 3/8 chuỗi.
- **Kiểm tra hợp lệ không dùng nhãn không phát hiện được đăng ký sai.**
  - 29 trong 107 epoch "hợp lệ" có sai số chẩn đoán > 100 px.
  - Ở chuỗi 26, 34, 61, **mọi** epoch hợp lệ đều sai (sai số 420–676 px).
  - Nguyên nhân [SUY LUẬN]: cảnh dày đặc vật cùng lớp (chủ yếu xe) tạo ra ≥ 8 cặp láng giềng tình cờ dưới một phép biến đổi sai. Hệ thật dùng bộ liên kết này sẽ ghép nhầm hàng loạt mà không biết.
- Điều này **gợi ý** tiền đề của RQ1 (BS khó tự ghép hai góc) là thật. Nhưng nó **không đo** được bao nhiêu F1 bị mất, và một bộ ước lượng tốt hơn có thể làm khác. Không được dùng như bằng chứng cho RQ1.
- Hai chuỗi đăng ký được là các chuỗi có nhiều token ở cả hai góc (31, 52) hoặc hai góc gần giống nhau (62, sai số 5 px).

## 4. Điều gì sẽ làm RQ1 xác định được

| Cách | Cần gì | Trạng thái |
|---|---|---|
| Liên kết dựa trên **pose** (GPS / độ cao / hướng camera), không cần GT | dữ liệu có metadata camera | MDMT không có. CrossTrack-UAV được mô tả là có GPS, độ cao, hướng (**chưa xác minh**); thỏa thuận đã điền, chờ chữ ký giáo sư |
| Liên kết dựa trên **ngoại hình** (embedding từ ảnh) | xử lý ảnh mới để trích đặc trưng | phản biện #23 không duyệt; quyền dùng MDMT chỉ là xác nhận miệng |
| Một bộ liên kết hình học khác, mạnh hơn | đăng ký trước một nhánh mới | không tự mở; chỉnh bộ hiện tại sau khi thấy kết quả là không hợp lệ |

## 5. Đề xuất

- **Dừng RQ1 trên bản ghi MDMT hiện có.** Không chạy Phase 1 / 2, không chỉnh bộ ước lượng.
- **RQ1 vẫn còn mở**, và bước tiếp theo phụ thuộc dữ liệu: CrossTrack-UAV, nếu xác minh được có metadata camera, cho phép một bộ liên kết theo pose không cần GT. Khi đó RQ1-A có thể đăng ký lại.
- Không mở nhánh nghiên cứu nào khác.

## 6. Tệp

- Kết quả tổng hợp theo chuỗi: `runs/uav_semcom/h1/rq1_phase0_v1.json`. Bảng theo từng epoch nằm ngoài git (dữ liệu theo khung suy ra từ MDMT).
- Code: `swarmbridge/experiments/uav_semcom/h1_rq1_assoc.py`, 17 test tổng hợp ở `test_h1_rq1.py` (170 test cùng bộ pilot và B1; commit 2c287e86 ghi nhầm là 171).
