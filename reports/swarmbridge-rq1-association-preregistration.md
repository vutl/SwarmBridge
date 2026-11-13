# RQ1: phép thử khả thi "ngân sách lỗi do liên kết xuyên góc" (đăng ký trước)

Ngày 14/11/2026. Bản này được commit **trước** khi chạy trên bất kỳ bản ghi dataset nào. Câu hỏi và phép thử được đề xuất ở mục 5 và 7 của [swarmbridge-collab-perception-novelty-review-2026-11-13.md](swarmbridge-collab-perception-novelty-review-2026-11-13.md).

**Trạng thái duyệt:** code và test đã xong; **chưa chạy trên MDMT**. Chỉ chạy khi người dùng đồng ý.

## Câu hỏi

Khi BS phải tự ghép detection của hai UAV (không có ID, không có pose chính xác), F1 vật thể duy nhất mất bao nhiêu so với liên kết lý tưởng? Một oracle chỉ "gửi thêm một crop" cho các detection có liên kết mơ hồ hồi phục được bao nhiêu, với bao nhiêu bit?

- Nếu phần mất nhỏ: liên kết hình học đã đủ, RQ1 không có dư địa → NO-GO.
- Nếu phần mất lớn và oracle hồi phục rẻ: có dư địa cho một cơ chế "gửi để phân định" → GO cho bước thiết kế (vẫn chưa phải cơ chế triển khai được).

## Dữ liệu và luồng

- 8 chuỗi MDMT qua sàng lọc ID (26, 31, 34, 52, 56, 57, 61, 62); bản ghi pilot 0B có sẵn; không ảnh mới, không train.
- Mỗi chuỗi: **nửa đầu** dùng để hiệu chuẩn; **nửa sau** để chấm (như B1).
- Mỗi epoch, mỗi góc khai báo detection ở BS với ngưỡng τ_decl = 0,3:
  - **TT** (quyết định): cả hai góc gửi TOKEN;
  - **FV** (báo kèm): góc được quy tắc `NDET` của B1 chọn gửi FRAME, góc kia TOKEN.
- Nhãn chỉ dùng cho: hiệu chuẩn ở nửa đầu, chấm điểm, và oracle được đánh dấu.

## Các điều kiện liên kết

| Điều kiện | Cách ghép hai góc |
|---|---|
| `ideal` | theo ID của GT (như B1; FP trùng = FP liên kết = 0) |
| `fresh` | homography góc 1 → góc 2 khớp trên box GT **cùng epoch** (tâm đáy, RANSAC), rồi ghép không nhãn: cùng lớp, Hungarian trên sai số chiếu, ngưỡng 40 px. Cận trên kiểu "pose hoàn hảo" |
| **`stale`** (quyết định) | như `fresh`, nhưng homography khớp **một lần** ở epoch cuối của nửa đầu có ≥ 8 ID chung, rồi dùng cho cả nửa sau. Mô phỏng "hiệu chuẩn một lần rồi trôi" |
| `none` | không ghép gì (mỗi detection là một vật): mốc dưới |

**Detection mơ hồ:** có ứng viên cùng lớp ở góc kia trong vòng 80 px, **và** (có ≥ 2 ứng viên như vậy, hoặc ứng viên gần nhất xa hơn 20 px).

**Oracle phân định:** mọi detection mơ hồ được ghép đúng theo GT. Chi phí: một crop cho mỗi detection mơ hồ, giá bằng trung vị số bit của crop P1 (AVIF q50) ở nửa đầu chuỗi. Nếu không có crop nào để lấy giá, chi phí là vô hạn (oracle không bao giờ miễn phí).

**Đếm vật thể duy nhất sau hợp nhất:**
- Mỗi vật đã hợp nhất (một cặp đã ghép, hoặc một detection lẻ) nhận tối đa một ID của GT (ghép cực đại).
- Vật có ID nhưng mọi ID đã bị vật khác nhận → **FP trùng**.
- Vật không khớp ID nào → **FP sai**.
- Cặp ghép hai ID khác nhau → **FP liên kết** (chẩn đoán; một trong hai vật bị mất).
- Với `ideal`, cách đếm này trùng với B1 (có test).

## Quy tắc quyết định (mode TT, điều kiện `stale`)

ΔA = F1(`ideal`) − F1(`stale`), trung bình theo 8 chuỗi. Hồi phục = [F1(oracle) − F1(`stale`)] / ΔA.

- **NO-GO** nếu ΔA < 1 điểm F1.
- **GO** nếu đồng thời:
  - ΔA ≥ 3 điểm F1;
  - ΔA > 0 ở ≥ 6/8 chuỗi;
  - oracle hồi phục ≥ 50% ΔA;
  - bit thêm của oracle ≤ 10% số bit trung vị của một FRAME.
- **INCONCLUSIVE** trong các trường hợp còn lại.

Các tổ hợp khác (TT / `fresh`, FV / `stale`, FV / `fresh`) chỉ báo kèm. Không đổi tham số hay ngưỡng sau khi xem kết quả.

## Tham số (cố định)

τ_decl = 0,3; ngưỡng ghép 40 px (bằng dung sai của sàng lọc ID); dải mơ hồ 20–80 px; ≥ 8 ID chung để khớp homography; RANSAC 40 px; crop ở AVIF q50.

## Giới hạn biết trước

- Hiệu chuẩn `stale` dùng nhãn của nửa đầu; trong hệ thật cần pose / GPS hoặc một bước hiệu chuẩn khác. Đây là lựa chọn có lợi cho liên kết hình học, nên nếu ΔA vẫn lớn thì kết luận càng mạnh; nếu ΔA nhỏ, có thể chỉ vì hiệu chuẩn quá tốt.
- Homography chỉ đúng với mặt đất; MDMT bay cao nên giả định này hợp lý hơn so với dữ liệu bay thấp.
- Oracle dùng GT để biết cách ghép đúng; nó đo **dư địa**, không phải một cơ chế.
- Giá crop là trung vị crop P1 (crop quanh detection bất định), không phải crop đo riêng cho từng cặp.
- 8 chuỗi; nhãn EXPL đã dùng nhiều lần; quyền dùng MDMT là xác nhận miệng của giáo sư do người dùng thuật lại.

## Code

`swarmbridge/experiments/uav_semcom/h1_rq1_assoc.py`; 13 test tổng hợp ở `test_h1_rq1.py` (không dùng bản ghi dataset).
