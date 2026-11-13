# A7-S0: kết quả cổng dư địa — họ TOKEN / CROP / FRAME trong môi trường A5

Ngày 13/11/2026. Thiết kế và cổng đã đăng ký trước ở [bản thiết kế](swarmbridge-a7s-semantic-family-design.md) (commit 67e4fc5e, trước khi chạy).

- **Code:** `swarmbridge/experiments/uav_semcom/a7s_semfam.py`. **Kết quả:** `runs/uav_semcom/h1/a7s_semfam_v1.json`.
- **Không train gì.** Không mở ảnh MDMT mới: bảng mode dựng từ 918 bản ghi pilot 0B đã có, và nằm ngoài git.
- **Khám phá:** dữ liệu nền là MDMT EXPL (quyền dùng chưa xác minh độc lập); 11 chuỗi chia hai fold; τ_decl = 0,3 chọn trên EXPL.

## 1. Bảng mode đo được

| Mode | Bit trung bình | Thời gian trên UAV | TP / FP mỗi khung (τ = 0,3) | Giá trị trung bình v |
|---|---|---|---|---|
| TOKEN | 2 kbit | 0,065 s (YOLO11n) | 24,7 / 6,5 | 0,83 |
| CROP (mọi P1, q50) | 205 kbit | +0,22 s | 31,9 / 22,9 | 0,81 |
| FRAME (q35) | 500 kbit | +1,07 s | 30,1 / 7,2 | 1,00 |

CROP bị FRAME lấn át trong thiết lập này: gửi mọi crop cho nhiều TP nhất nhưng cũng nhiều FP nhất, nên giá trị (TP − 0,5·FP) còn thấp hơn TOKEN.

## 2. Kết quả trên test (100 seed mỗi ô; ± là CI 95% ghép cặp theo seed)

Return là mục tiêu của môi trường (giá trị nhiệm vụ + thưởng đúng hạn − phạt); "Δ" là so với FIX.

| Fold, chế độ | FIX (mode cố định tốt nhất) | AGN: chỉ trạng thái kênh | CAU: thêm dự đoán nội dung | ORA: biết bit và giá trị thật |
|---|---|---|---|---|
| A, ref | TOKEN, R 58,5 | +0,0 ± 0,0 | −0,1 ± 0,3 | **+4,9 ± 0,2 (+8,5%)** |
| A, K5 | TOKEN, R 97,7 | +0,5 ± 0,1 | **−9,0 ± 0,7** | **+6,2 ± 0,3 (+6,4%)** |
| A, phy_mcs_blind | TOKEN, R 57,4 | +0,0 | −0,5 ± 0,1 | +0,3 ± 0,4 |
| B, ref | TOKEN, R 88,3 | +0,0 ± 0,5 | **+7,6 ± 0,4 (+8,7%)** | **+9,5 ± 0,3 (+10,8%)** |
| B, K5 | TOKEN, R 147,4 | +1,3 ± 0,3 | +2,0 ± 0,6 | **+14,1 ± 0,5 (+9,5%)** |
| B, phy_mcs_blind | TOKEN, R 87,2 | +0,0 | +0,8 ± 0,4 | +0,9 ± 0,1 |

Recall trên test:

| Fold, chế độ | FIX | AGN | CAU | ORA |
|---|---|---|---|---|
| A, ref | 0,431 | 0,431 | 0,524 | 0,530 |
| A, K5 | 0,430 | 0,438 | 0,519 | 0,526 |
| A, phy_mcs_blind | 0,431 | 0,431 | 0,456 | 0,517 |
| B, ref | 0,567 | 0,657 | 0,658 | 0,670 |
| B, K5 | 0,567 | 0,584 | 0,586 | 0,669 |
| B, phy_mcs_blind | 0,567 | 0,567 | 0,589 | 0,649 |

**Mode cố định:** tính theo return, TOKEN thắng ở mọi ô. FRAME cố định có điểm nhiệm vụ cao nhất (~1,0) nhưng return thấp hơn vì trễ và nghẽn: 53,0 so với 58,5 ở A-ref; 69,6 so với 97,7 ở A-K5; 25,2 so với 57,4 ở A-phy_mcs_blind.

## 3. Theo cổng đã đăng ký

- **G-S1 (ORA − FIX ≥ +2 điểm task_score và return cao hơn, cả hai fold): ĐẠT ở cả ba chế độ.** task_score của ORA hơn FIX 4–19 điểm. Tuy vậy ở phy_mcs_blind, lợi ích về return rất nhỏ (+0,3 và +0,9).
- **G-S2 (CAU lấy được bao nhiêu phần dư địa, tính theo task_score như đã đăng ký):**

| | A, ref | A, K5 | A, mcs_blind | B, ref | B, K5 | B, mcs_blind |
|---|---|---|---|---|---|---|
| Theo task_score (đăng ký) | 104% | 103% | 30% | 85% | 15% | 76% |
| Theo return (bổ sung sau, không đăng ký) | −3% | âm | âm | 80% | 15% | 89% |

- Theo chỉ số đăng ký, 2/6 ô dưới 70%. Nhưng chỉ số đó **gây hiểu lầm**: ở fold A, CAU đạt điểm nhiệm vụ cao bằng cách gửi 59–61% khung, và mất lại toàn bộ ở phần trễ (return bằng hoặc thấp hơn TOKEN, −9 ở K5). Tính theo return thì heuristic nhân quả chỉ lấy được phần lớn dư địa ở 2/6 ô (B-ref và B-phy_mcs_blind).
- **Kết luận theo cổng:** có dư địa phụ thuộc trạng thái (G-S1), và heuristic nhân quả chưa lấy được nó một cách ổn định (G-S2). Theo thiết kế, bước này dẫn tới một **đề xuất** pilot PPO mode-only nhỏ, **cần duyệt riêng**; tôi chưa chạy.

## 4. Đọc kết quả

1. **Khác G-L2.** Với các mức chất lượng ảnh, cận trên thông tin chỉ hơn mode cố định khoảng 1–2 điểm recall. Với họ TOKEN / CROP / FRAME, ORA hơn mode cố định tốt nhất **8–10 điểm recall và 6–11% return** ở ref và K5, ở cả hai fold. Khoảng chênh 250 lần về bit và 16 lần về CPU tạo ra một bài toán lựa chọn thật.
2. **"Trạng thái" ở đây là trạng thái nhận thức, không phải trạng thái vô tuyến.** AGN, chỉ dùng tốc độ kênh / hàng đợi / hạn chót và không nhìn nội dung, gần như bằng mode cố định (+0 đến +0,9% return). Dư địa nằm ở việc biết **khung nào** đáng gửi FRAME hay CROP, tức là dự đoán giá trị theo từng khung.
3. **Dự đoán nội dung từ đặc trưng rẻ không tổng quát giữa các nhóm chuỗi.** CAU tốt ở fold B (ref: 80% dư địa) nhưng vô ích hoặc có hại ở fold A (K5: −9%), nơi tập test chứa các chuỗi khó (59, 31, 61, 26) mà bộ dự đoán học trên các chuỗi dễ hơn.
4. **phy_mcs_blind:** dư địa về return gần bằng 0. FRAME cố định rất tệ ở đây (return 25–58), tức là gửi tải lớn dưới PHY không quan sát được nhiễu thì dễ mất gói; chính sách an toàn là gần như chỉ TOKEN.
5. **CROP** chỉ chiếm 11–23% lựa chọn của ORA; phần lớn dư địa đến từ việc trộn TOKEN và FRAME.

## 5. Điều chưa chắc

- **ORA dùng giá trị thật theo GT.** Bài học G-L2: một phần "dư địa oracle" là nhiễu của detector, không dự đoán được. Chưa có phép khử thiên lệch ở đây.
- CI theo seed môi trường **đánh giá thấp** bất định theo chuỗi (A5.5: CI theo cụm video rộng hơn 3–10 lần). Chỉ có hai fold với 5–6 chuỗi mỗi phía.
- Thời gian suy luận ở BS là giả định (0 / 0,05 s). Không có sàn PSNR. τ_decl cố định 0,3. CPU là desktop 1 luồng.
- Môi trường A5 là nhiệm vụ khảo sát ảnh rời, không phải video.

## 6. Đề xuất (cần duyệt; chưa làm)

Trước khi train bất cứ gì, một bước rẻ theo bài học G-L2:

1. **Khử thiên lệch dư địa oracle:** thay giá trị thật của khung bằng giá trị đo ở khung lân cận cùng chuỗi và cùng góc (bản lặp theo thời gian; chỉ dùng bản ghi đã có). Nếu dư địa ORA sụp, phần lớn là nhiễu → dừng.
2. Nếu dư địa còn: pilot **PPO mode-only** nhỏ trên nền A5.5, quan sát có thêm các đặc trưng trên UAV, so với FIX, CAU và ORA, chia theo chuỗi, nhiều seed. Tiêu chí thành công và số bước train phải được đăng ký trước.

Không mở HOLD, không mở MDMT-val, không chạy ảnh MDMT mới.
