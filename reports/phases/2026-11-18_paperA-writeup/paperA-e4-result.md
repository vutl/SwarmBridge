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
