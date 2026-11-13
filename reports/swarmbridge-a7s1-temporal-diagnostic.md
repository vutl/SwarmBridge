# A7-S1: chẩn đoán khả năng dự đoán theo thời gian (sau A7-S0)

Ngày 13/11/2026, theo phản biện #15 (duyệt bước chẩn đoán; **không** duyệt train PPO).

- **Code:** `swarmbridge/experiments/uav_semcom/a7s_semfam.py` (thêm các điều kiện thông tin). **Kết quả:** `runs/uav_semcom/h1/a7s1_lag_v1.json`.
- Chỉ dùng bản ghi đã có. Không train, không ảnh mới, không val / HOLD.
- Giữ nguyên hai fold, ba chế độ, seed train / test, họ chính sách và hệ số phần thưởng của A7-S0. Các ô FIX / AGN / CAU / ORA cho lại đúng số của A7-S0.
- **Phần thưởng luôn chấm bằng giá trị thật của khung hiện tại.** Chỉ thông tin dùng để quyết định mode bị thay.

## 1. Các điều kiện thông tin

| Ký hiệu | Giá trị dùng để quyết định | Bit dùng để quyết định | Ghi chú |
|---|---|---|---|
| FIX | — | — | mode cố định tốt nhất (TOKEN) |
| CAU | dự đoán từ đặc trưng hiện tại trên UAV | dự đoán | nhân quả |
| ORA | thật, khung t | thật, khung t | tham chiếu lý tưởng |
| TV_PB | thật, khung t | dự đoán | tách phần "giá trị thật" |
| HV_TB | dự đoán (như CAU) | thật, khung t | tách phần "bit thật" |
| LAG_TB / LAG_PB | **của khung t−1** (cùng chuỗi, cùng góc) | thật / dự đoán | chẩn đoán |
| SHUF_TB / SHUF_PB | của **một khung khác bất kỳ** cùng chuỗi và cùng góc | thật / dự đoán | đối chứng cho LAG |

- **Ánh xạ t−1:** bản ghi liền trước theo thứ tự epoch của MDMT, cùng chuỗi và cùng góc. Có assert kiểm chuỗi, góc và chỉ số khung. 22 bản ghi đầu (11 chuỗi × 2 góc) không có quá khứ và dùng giá trị trung bình theo mode. Không dùng khung tương lai.
- **Không điều kiện LAG / SHUF nào triển khai được:** giá trị của khung trước cũng là giá trị tính bằng GT. Thêm nữa, môi trường A5 rút ảnh ngẫu nhiên từ pool cho mỗi điểm chụp, nên hai nhiệm vụ liên tiếp trong mô phỏng không phải hai khung liên tiếp.

## 2. Độ ổn định của giá trị giữa khung t−1 và t (offline, 918 bản ghi)

| | Tương quan v (TOKEN / CROP / FRAME) | Tương quan của hiệu FRAME − TOKEN | Cùng mode tốt nhất | Cùng dấu FRAME − TOKEN |
|---|---|---|---|---|
| Khung t−1 | 0,98 / 0,97 / 0,99 | 0,90 (Spearman 0,80) | 71% | 78% |
| Khung tráo, cùng chuỗi và góc | 0,87 / 0,81 / 0,92 | 0,84 (Spearman 0,72) | 66% | 76% |

- Sau khi trừ trung bình của từng (chuỗi, góc), tương quan của hiệu FRAME − TOKEN giữa t−1 và t chỉ còn **0,36** (bản tráo: −0,05).
- **Ý nghĩa:** giá trị tuyệt đối rất ổn định, nhưng phần lớn sự ổn định đó là *đặc tính của cảnh* (chuỗi và góc nhìn). Tính liên tục từ khung sang khung có thật nhưng nhỏ.
- Mode tốt nhất theo giá trị: FRAME ở 68% bản ghi, TOKEN 23%, CROP 10%.

## 3. Kết quả trong môi trường A5 (test, 100 seed; Δ return so với FIX, ± CI 95% ghép cặp theo seed)

| Fold, chế độ | CAU | HV_TB | TV_PB | LAG_PB | SHUF_PB | ORA |
|---|---|---|---|---|---|---|
| A, ref | −0,1 ± 0,3 | −0,0 ± 0,3 | +4,9 ± 0,2 | +3,0 ± 0,2 | +2,4 ± 0,2 | +4,9 ± 0,2 |
| A, K5 | −9,0 ± 0,7 | −9,4 ± 0,8 | +5,7 ± 0,4 | +3,0 ± 0,3 | +1,8 ± 0,3 | +6,2 ± 0,3 |
| B, ref | +7,6 ± 0,4 | +7,6 ± 0,4 | +9,4 ± 0,4 | +7,9 ± 0,4 | +7,6 ± 0,4 | +9,5 ± 0,3 |
| B, K5 | +2,0 ± 0,6 | +7,1 ± 0,6 | +13,7 ± 0,5 | +11,2 ± 0,5 | +10,5 ± 0,5 | +14,1 ± 0,5 |

Phần dư địa ORA − FIX được giữ lại (theo return):

| Fold, chế độ | TV_PB | HV_TB | LAG_PB | SHUF_PB |
|---|---|---|---|---|
| A, ref | 99% | 0% | 61% | 49% |
| A, K5 | 91% | âm | 48% | 30% |
| B, ref | 99% | 80% | 83% | 80% |
| B, K5 | 98% | 50% | 79% | 74% |

- LAG_TB gần bằng LAG_PB ở mọi ô (chênh ≤ 0,2 return). SHUF_TB bằng SHUF_PB, trừ B-K5 (+7,1 so với +10,5).
- Recall của LAG_PB và SHUF_PB chỉ thấp hơn ORA khoảng 0,5–1,5 điểm (ví dụ A-ref: 0,521 / 0,520 / 0,530; FIX 0,431).
- **phy_mcs_blind:** dư địa ORA − FIX chỉ +0,3 và +0,9 return, nên mọi tỉ lệ đều vô nghĩa. Không coi là bằng chứng.

**Theo chuỗi** (Δ utility mỗi nhiệm vụ so với FIX; chế độ ref; trong ngoặc là tỉ lệ chọn FRAME):

| Chuỗi (fold test) | CAU | LAG_PB | SHUF_PB | ORA |
|---|---|---|---|---|
| 26 (A) | +0,066 (49%) | +0,067 (51%) | +0,066 (52%) | +0,077 (51%) |
| 31 (A) | −0,028 (99%) | +0,002 (42%) | −0,007 (43%) | +0,015 (38%) |
| 57 (A) | −0,018 (1%) | −0,008 (3%) | −0,012 (3%) | −0,004 (4%) |
| 59 (A) | −0,022 (81%) | +0,009 (17%) | +0,007 (16%) | +0,029 (15%) |
| 61 (A) | +0,098 (84%) | +0,100 (83%) | +0,095 (85%) | +0,109 (84%) |
| 34 (B) | +0,093 (48%) | +0,100 (51%) | +0,104 (49%) | +0,110 (49%) |
| 48 (B) | −0,014 (1%) | −0,009 (2%) | −0,008 (1%) | −0,004 (1%) |
| 52 (B) | +0,077 (51%) | +0,076 (51%) | +0,070 (51%) | +0,082 (50%) |
| 56 (B) | −0,009 (6%) | −0,004 (18%) | −0,010 (16%) | +0,006 (20%) |
| 62 (B) | −0,006 (2%) | −0,003 (19%) | −0,010 (18%) | +0,011 (16%) |
| 68 (B) | +0,311 (98%) | +0,304 (85%) | +0,304 (84%) | +0,320 (83%) |

## 4. Trả lời các câu hỏi của phản biện

**Dư địa đến từ bit thật hay giá trị thật?** Gần như toàn bộ từ **giá trị thật theo từng khung**.
- TV_PB (giá trị thật, bit dự đoán) giữ 91–99% dư địa.
- HV_TB (bit thật, giá trị dự đoán) không hơn CAU ở 3/4 ô. Chỉ ở B-K5, bit thật giúp thêm (+7,1 so với +2,0).
- Lý do: bit CROP dự đoán được rất tốt từ số cửa sổ P1 (tương quan 0,98 trên test).

**Khoảng cách theo thời gian có ổn định không?** **Ổn định một phần, và chủ yếu ở mức cảnh.**
- Giá trị của khung t−1 giữ lại 48–83% dư địa ở ref / K5.
- Nhưng giá trị của một khung *bất kỳ* cùng chuỗi và góc đã giữ 30–80%. Phần riêng của tính liên tục theo thời gian (LAG − SHUF) chỉ là +0,3 đến +1,2 return.
- Bảng theo chuỗi cho cùng một hình ảnh: lợi ích nằm ở vài cảnh cụ thể (26, 61, 34, 52, 68), nơi FRAME được chọn khoảng 50–85%. Tỉ lệ ~50% ở chuỗi 26, 34, 52 gợi ý FRAME có lợi ở một trong hai góc nhìn. Ở các cảnh còn lại, mọi chính sách đều gần TOKEN.
- Như vậy dư địa không phải chủ yếu là nhiễu theo từng khung của detector (điều đã xảy ra ở G-L2). Nó là hiểu biết **"ở cảnh / góc nhìn này, FRAME có đáng không"**.

**Vì sao CAU thất bại ở fold A?** Nó nhận nhầm cảnh. Ở chuỗi 31 và 59, CAU gửi FRAME 81–99% trong khi ORA chỉ gửi 15–38%, nên mất utility. "Detector trên UAV thiếu chắc chắn" không đồng nghĩa "FRAME đáng giá ở đây".

## 5. Kết luận cho PPO mode-only

**NO-GO trong môi trường A5 hiện tại.**
1. Dư địa chỉ đạt được khi có thông tin lấy từ GT (giá trị thật của khung này, khung trước, hoặc khung khác cùng cảnh). Không có tín hiệu nào trong số đó có sẵn trên UAV hay BS.
2. Môi trường A5 rút ảnh độc lập từ pool, không có cảnh kéo dài. Một chính sách học không có cách nào tích luỹ hiểu biết về cảnh, kể cả khi có tín hiệu.
3. Đặc trưng nhân quả của khung hiện tại (CAU) không ổn định giữa hai fold.
4. Lưu ý về AGN (theo phản biện): AGN ≈ FIX chỉ cho thấy *heuristic này* với trạng thái mạng hiện tại không tìm được lợi ích; nó không chứng minh trạng thái vô tuyến vô ích với mọi chính sách.

Giữ A5 / A5.5 làm hướng chính.

## 6. Điều kiện để mở lại (chỉ là phác thảo protocol; chưa làm gì)

Dư địa mức cảnh gợi ý một bài toán khác với "PPO chọn mode theo khung": **học giá trị của FRAME theo cảnh từ phản hồi của BS** (thăm dò rồi khai thác). Trước khi viết protocol RL đầy đủ, cần hai điều kiện tiên quyết, đều kiểm được offline trên bản ghi đã có và không cần train:

1. **Tín hiệu không cần GT ở BS.** Khi BS nhận một FRAME, nó có cả kết quả YOLO11s trên khung lẫn token của UAV. Hiệu số "số khai báo từ FRAME − số token" (hoặc số vật FRAME thêm được so với token) có thay được giá trị thật trong LAG / SHUF không? Cổng đề nghị: giữ ≥ 50% dư địa ORA − FIX ở ref và K5 trong cả hai fold, và không âm ở chuỗi nào quá mức của CAU.
2. **Môi trường có cảnh kéo dài.** Mỗi UAV (hoặc mỗi cụm điểm chụp) gắn với một (chuỗi, góc) và rút khung theo thứ tự thời gian, để phản hồi quá khứ có ý nghĩa.

Nếu cả hai đạt, một pilot nhỏ sẽ có dạng:
- **Quan sát:** quan sát A5.5 + đặc trưng YOLO11n của khung + thống kê phản hồi BS của cảnh (số FRAME đã gửi, lợi ích proxy trung bình).
- **Baseline:** FIX, CAU, heuristic thăm dò–khai thác đơn giản (ví dụ gửi FRAME thăm dò mỗi N khung rồi theo ngưỡng), và chính sách A5.5 hiện có với mode cố định.
- **Tiêu chí thành công (đăng ký trước):** hơn baseline thăm dò–khai thác tốt nhất ≥ 2% return ở ref hoặc K5, trong cả hai fold, với ≥ 5 seed train.
- **Giới hạn:** tối đa 2M bước mỗi seed; dừng sớm nếu sau 1M bước chưa vượt FIX trên val.

Tất cả cần duyệt riêng. Nếu điều kiện 1 trượt, đóng hẳn nhánh này.

## 7. Giới hạn

- Hai fold, 5–6 chuỗi mỗi phía; CI theo seed đánh giá thấp bất định theo chuỗi (bảng theo chuỗi ở mục 3 là để đọc trực tiếp).
- τ_decl = 0,3 và ngưỡng conf chọn trên EXPL; thời gian suy luận ở BS là giả định.
- Quyền dùng MDMT chưa xác minh độc lập; bảng theo khung nằm ngoài git.
