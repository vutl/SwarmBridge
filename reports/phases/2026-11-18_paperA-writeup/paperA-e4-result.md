# Paper A E4: quy mô mạng (số UAV, số kênh), kết quả (22/11)

Pre-registration: [paperA-e4-e5-prereg.md](paperA-e4-e5-prereg.md), commit trước khi chạy (ee1dbd0d, phụ lục
2f11b15b). Seed test mới 13000–13099. LA danh nghĩa, tổng băng thông 1 MHz. H\* tune lại theo từng cấu hình trên seed
train: q3 ở mọi cấu hình, riêng N7x2 là q1. Bộ lập lịch JA giữ nguyên (chọn ở K5M2).

## Kiểm định đã đăng ký (chênh return, CI 95% ghép cặp, 100 seed; * = CI không chứa 0)

| Cấu hình | K/M | H1: cal − H\* | H2: cal − RobustMPC-style | H3: cal − danh nghĩa | Gộp − cal (ablation) |
|---|---|---|---|---|---|
| N3x2 | 1,5 | **+1,7*** [+0,6; +2,8] | **+1,7*** [+0,7; +2,8] | **+7,0*** | −1,7* |
| N5x2 (cấu hình chính) | 2,5 | +2,7 [−0,5; +5,8] | **+3,3*** [+0,2; +6,3] | **+19,5*** | −1,5 |
| N7x2 | 3,5 | **+12,6*** [+9,7; +15,5] | **+12,0*** [+9,1; +15,0] | **+17,9*** | −1,4 |
| N5x3 | 1,67 | +1,5 [−0,3; +3,2] | +1,7 [−0,2; +3,5] | **+16,0*** | −0,7 |
| N5x4 | 1,25 | **+2,7*** [+1,8; +3,6] | **+3,0*** [+2,1; +3,9] | **+10,6*** | −1,2* |
| **Số cấu hình đạt** | | **3/5** | **4/5** | **5/5** | |

**Theo quy tắc đã đăng ký** (H1 và H2 cùng đạt ở ≥ 4/5): **KHÔNG ĐẠT**, vì H1 chỉ đạt 3/5.
- Bài **không** được viết "lợi ích giữ được qua các quy mô mạng đã đánh giá".
- Chỉ được viết:
  - lớp MPC hiệu chỉnh tốt hơn MPC danh nghĩa ở cả 5 cấu hình;
  - tốt hơn RobustMPC-style ở 4/5 cấu hình;
  - tốt hơn incumbent không có MPC ở 3/5 cấu hình (N3x2, N7x2, N5x4);
  - ở N5x2 và N5x3 thì chênh dương nhưng CI chứa 0.

## Lưu ý quan trọng ở cấu hình chính N5x2

- Trên seed E1 (9900–9999), H\* + MPC hiệu chỉnh − H\* là khoảng +8,2. Trên seed mới E4 (13000–13099) chỉ còn +2,7, CI
  chứa 0.
- Hai lần đo dùng cùng cấu hình và cùng phương pháp, chỉ khác bộ seed. Mức chênh giữa hai bộ seed lớn hơn khoảng tin cậy
  của từng bộ.
- Vì vậy con số "MPC thêm bao nhiêu so với incumbent" ở cấu hình chính phải báo kèm cả hai bộ seed. Không được chọn bộ
  có lợi hơn.
- Phép so với MPC danh nghĩa (+19,5) và với RobustMPC-style (+3,3) vẫn có ý nghĩa thống kê trên seed mới.

## Chỉ số phụ (trung bình 100 seed)

| Cấu hình | Return / ảnh: H\* → +cal | Recall: H\* → +cal | Rớt ảnh: H\* → +cal | Lập lịch p99 | Quyết định tích hợp p99 (+cal) | Slot > 50 ms (+cal) |
|---|---|---|---|---|---|---|
| N3x2 | +0,061 → +0,072 | 55,0 → 54,7 | 1,7% → 1,5% | 1,5 ms | 40,9 ms | 0% |
| N5x2 | −0,131 → −0,121 | 52,1 → 51,0 | 6,8% → 5,4% | 4,2 ms | 45,2 ms | 0,5% |
| N7x2 | −0,351 → −0,316 | 35,3 → 37,2 | 2,0% → 2,1% | 34,9 ms | 71,8 ms | **27,7%** |
| N5x3 | +0,021 → +0,026 | 54,7 → 54,0 | 2,2% → 1,6% | 11,0 ms | 50,0 ms | 2,1% |
| N5x4 | +0,068 → +0,078 | 55,2 → 54,9 | 1,4% → 1,1% | 38,0 ms | 75,5 ms | **40,2%** |

**Recall:** MPC hiệu chỉnh làm recall giảm nhẹ ở 4/5 cấu hình (−0,3 đến −1,1 điểm, có ý nghĩa thống kê). Riêng N7x2
recall tăng +1,9. Return và recall lệch nhau, giống kết quả chính của bài.

**Độ trễ:**
- Ở N7x2 và N5x4, riêng bộ lập lịch JA (duyệt (M+1)^K phương án) đã tốn 35–38 ms mỗi slot.
- Khi thêm MPC, 28% / 40% số slot vượt 50 ms. **Không được tuyên bố đáp ứng thời gian thực** ở hai cấu hình này.
- Muốn mở rộng cần bộ lập lịch xấp xỉ. Đó là việc riêng, phải được dán nhãn rõ.

**RobustMPC-style ≈ H\* ở mọi cấu hình:** chênh ≤ 0,6. Ước lượng tốc độ bảo thủ của nó gần như không bao giờ khiến MPC
đổi mode của incumbent. Khớp với E2c (mỗi UAV chỉ có 5 quan sát, cộng hai lớp bảo thủ).

**Gộp − chia SNR:** bản gộp thấp hơn bản chia SNR ở cả 5 cấu hình (−0,7 đến −1,7; có ý nghĩa ở N3x2 và N5x4). Điều này
khác E2c, nơi chia SNR không tạo khác biệt. Phương pháp đóng băng (CalRates chia SNR) vì vậy được giữ nguyên. Bản gộp chỉ
báo như ablation, đúng như phản biện yêu cầu.

**E5:** kết quả trễ/mất phản hồi trong `e4.json` bị loại (lỗi bộ theo dõi đã sửa ở 2f11b15b). Kết quả chạy lại là
`e5.json`, báo riêng.

**File:** `runs/uav_semcom/paperA_r2/e4.json`, `e4.deliv.json.gz`, `summary_e4.json/.log`; script
`paperA_e4_summary.py`.

## E5: phản hồi tốc độ bị trễ hoặc mất (chạy lại sau khi sửa lỗi, 22/11)

Bộ hiệu chỉnh **và** bộ theo dõi sai số cùng nhận lịch trễ/mất báo cáo giống nhau (commit 2f11b15b, có test). Cấu hình
N5x2, cùng seed 13000–13099 với E4, ghép cặp. Một slot = 1 s.

| Báo cáo | − báo cáo tức thì | − MPC danh nghĩa | − H\* |
|---|---|---|---|
| Trễ 1 slot | +1,8 [−1,5; +5,1] | **+21,3*** | **+4,5*** |
| Trễ 3 slot | −0,6 [−3,8; +2,6] | **+18,9*** | +2,1 [−0,6; +4,7] |
| Trễ 10 slot | −1,4 [−4,5; +1,8] | **+18,1*** | +1,3 [−1,8; +4,3] |
| Trễ 1 slot, mất 30% | +1,2 [−2,1; +4,6] | **+20,7*** | **+3,9*** |
| Trễ 3 slot, mất 30% | −0,5 [−3,7; +2,8] | **+19,0*** | +2,2 [−0,8; +5,1] |

**Theo quy tắc đã đăng ký** (chỉ được nói "chịu được trễ d / mất p" với những thiết lập có CI so với MPC danh nghĩa
hoàn toàn trên 0):
- **Đạt ở cả 5 thiết lập.** Câu được phép viết: "trong các thiết lập E5 đã thử, lợi thế so với MPC danh nghĩa vẫn có ý
  nghĩa khi báo cáo throughput dùng cho lập kế hoạch trễ tới 10 s, hoặc trễ 1–3 s kèm mất 30%".
- So với báo cáo tức thì, **chưa phát hiện** khác biệt có ý nghĩa (CI chứa 0). Điều này **không** chứng minh hai điều
  kiện tương đương.
- So với H\* không có MPC, kết quả lẫn lộn, giống E4: ngay cả báo cáo tức thì cũng chỉ +2,7, CI chứa 0.

**Phạm vi và những gì chưa kiểm tra:**
- Chỉ báo cáo throughput dùng cho lập kế hoạch bị trễ/mất. ACK của ARQ và quan sát hàng đợi của bộ lập lịch giữ nguyên.
- **Chưa kiểm tra:**
  - mất ACK ở tầng truyền dẫn;
  - trễ 10 s kết hợp mất 30%;
  - chi phí overhead của báo cáo.
- Giả định phản hồi trong bài vì vậy **được nới một phần**, chưa được gỡ bỏ hoàn toàn.
- Kết quả `+mpcfb` cũ trong `e4.json` (lỗi bộ theo dõi) không được dùng.

## E6: tổng băng thông 0,5 / 1 / 2 MHz (K = 5, M = 2; 22/11)

Đăng ký trước trong phụ lục của pre-registration. Seed 13000–13099; điểm 1 MHz là N5x2 của E4 (ghép cặp). H\* tune lại
theo băng thông trên seed train: AVIF-20 ở 0,5 MHz, q3 ở 1 và 2 MHz.

| Băng thông | H1: cal − H\* | H2: cal − RobustMPC-style | H3: cal − danh nghĩa | Recall: cal − H\* | Gộp − cal |
|---|---|---|---|---|---|
| 0,5 MHz | **+9,7*** [+7,7; +11,8] | **+9,8*** | **+5,8*** | **+2,6*** | +1,1 |
| 1 MHz | +2,7 [−0,5; +5,8] | **+3,3*** | **+19,5*** | −1,1* | −1,5 |
| 2 MHz | **+4,4*** [+0,7; +8,1] | **+5,0*** | **+30,3*** | −0,6* | +0,0 |
| **Đạt** | **2/3** | **3/3** | **3/3** | | |

**Theo quy tắc đã đăng ký** (H1 và H2 đạt ở cả ba mức băng thông): **KHÔNG ĐẠT**.
- Lý do duy nhất: H1 ở 1 MHz, chính là điểm N5x2 của E4.
- Câu được phép viết:
  - MPC hiệu chỉnh tốt hơn MPC danh nghĩa và RobustMPC-style ở cả ba mức băng thông;
  - tốt hơn incumbent không có MPC ở 0,5 và 2 MHz, nhưng không có ý nghĩa ở 1 MHz.

**Ghi nhận khác:**
- Ở 0,5 MHz, MPC hiệu chỉnh tăng cả return (+9,7) lẫn recall (+2,6). Ở 1 và 2 MHz, recall giảm nhẹ. Lệch return/recall
  phụ thuộc mức tải.
- Lợi thế so với mô hình danh nghĩa tăng theo băng thông (+5,8 → +19,5 → +30,3).
- Không có slot nào vượt 50 ms ở cả ba mức băng thông (p99 41–45 ms).
- Gộp − chia SNR: CI chứa 0 ở cả ba mức.

**File:** `runs/uav_semcom/paperA_r2/e6.json`, `e6.deliv.json.gz`, `summary_e6.json/.log`; script
`paperA_e6_summary.py`.
