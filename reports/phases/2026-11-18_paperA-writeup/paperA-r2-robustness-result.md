# Paper A, vòng kiểm tra độ vững E1–E3 (+ E1b), 19/11/2026

- **Code:**
  - `paperA_r2.py` (lệnh e12, e3, e1b), `paperA_r2_summary.py`, `paperA_r2b_summary.py`;
  - `a9_mpc.HarmRates` (baseline kiểu RobustMPC, hậu tố `+mpcrh` trong `ax2_eval`).
- **Kết quả:** `runs/uav_semcom/paperA_r2/`.
- Không train mới. Các chính sách giữ nguyên như R1: T:best, PPO-S, PPO-LA, H\*_S = AVIF-50, H\*_L = mbt q3.

## Cách làm

- **E1, các biến thể LA** (test pool, seed mới 9900–9999, cùng seed cho mọi arm):
  - **L:** như R1;
  - **L_fade:** fading nhỏ rút lại mỗi block 0,1 s, thay vì đóng băng theo slot 1 s;
  - **L_bursty:** UAV dùng chung kênh chỉ gây nhiễu khi đang gửi dữ liệu;
  - **L_badfb:** báo cáo CQI sai số 3 dB, trễ 10 block (1 s);
  - **S** (Shannon) trên cùng seed.
- **E2, baseline kiểu RobustMPC** (Yin et al. 2015):
  - cùng planner, incumbent, ngân sách 50 ms và cổng;
  - dự báo tốc độ mỗi UAV = trung bình điều hòa 5 slot gần nhất / (1 + sai số tương đối lớn nhất của 5 dự báo gần
    nhất);
  - chỉ dựa vào lịch sử, không dùng vị trí hay SNR kỳ vọng.
- **E3, nội dung giữ lại:** pool test-dev (VisDrone2019-DET-test-dev, 49 video), S và L, seed mới 12000–12099.
  - Heuristic vẫn là H\* tune trên ảnh train, đúng protocol: không được tune trên nội dung test.
- **E1b, so công bằng:** tune lại H\* trên seed train 0–29 của từng biến thể LA (cùng họ 22 luật như R1), rồi chạy trên
  seed E1.
  - PPO-LA **không** được train lại cho từng biến thể, nên phép so này có lợi cho heuristic.

Mọi Δ là chênh lệch return ghép cặp theo seed, kèm CI 95 %.

## Kết quả

### 1. Đảo thứ hạng PPO-S so với T:best (khối A, cùng luật slot T:best)

| Cấu hình | PPO-S − T:best |
|---|---|
| S (seed mới) | +2,25 [+1,30, +3,20] |
| L | −23,41 [−26,93, −19,89] |
| L_fade | −38,98 [−42,34, −35,62] |
| L_bursty | −22,08 [−26,20, −17,96] |
| L_badfb | −37,88 [−41,98, −33,79] |
| test-dev, S | +13,99 [+12,96, +15,02] |
| test-dev, L | −59,67 [−64,46, −54,88] |

Đảo chiều giữ nguyên ở mọi biến thể LA và trên nội dung giữ lại.

### 2. Hiệu chỉnh tốc độ: theo SNR, so với kiểu RobustMPC, so với danh nghĩa (H\* làm incumbent)

| Cấu hình | MPC hiệu chỉnh − danh nghĩa | Hiệu chỉnh − kiểu RobustMPC | Kiểu RobustMPC − danh nghĩa |
|---|---|---|---|
| S | +0,57 [−0,92, +2,07] | −0,89 [−1,94, +0,16] | +1,46 [+0,13, +2,79] |
| L | +16,96 [+12,71, +21,22] | **+6,62** [+3,12, +10,13] | +10,34 [+5,85, +14,83] |
| L_fade | +36,43 [+31,19, +41,67] | **+10,12** [+7,28, +12,97] | +26,30 [+21,30, +31,30] |
| L_bursty | +20,92 [+16,40, +25,44] | **+8,16** [+4,35, +11,97] | +12,76 [+8,65, +16,87] |
| L_badfb | +28,22 [+21,12, +35,32] | **+13,73** [+9,10, +18,36] | +14,49 [+8,39, +20,59] |
| test-dev, S | +1,79 [+0,34, +3,23] | +12,05 [+9,97, +14,13] | −10,27 [−12,53, −8,00] |
| test-dev, L | +5,24 [+1,04, +9,43] | **+18,04** [+13,12, +22,95] | −12,80 [−18,01, −7,59] |

- Hiệu chỉnh theo phân phối thực nghiệm thắng baseline kiểu RobustMPC ở mọi cấu hình LA và trên test-dev. **Nhưng xem mục 5:** lợi thế này không đến từ việc chia theo SNR.
- Trên test-dev, baseline kiểu RobustMPC còn tệ hơn MPC danh nghĩa.
- Với PPO-LA làm incumbent, hiệu chỉnh − kiểu RobustMPC = +6,9 / +10,3 / +2,7 / +9,2 (L / fade / bursty / badfb) và
  +10,9 (test-dev).

### 3. PPO-LA so với heuristic

| Cấu hình | PPO-LA + MPC − H\*_L (đóng băng) + MPC | PPO-LA + MPC − H\*_V (tune lại) + MPC | H\* tune lại |
|---|---|---|---|
| L | +0,46 [−2,27, +3,18] | (giống cột trước) | q3 |
| L_fade | +61,21 [+57,15, +65,26] | **−31,35** [−35,30, −27,40] | **q1** |
| L_bursty | −9,22 [−12,35, −6,10] | −9,22 | q3 |
| L_badfb | +50,97 [+45,85, +56,10] | **−29,27** [−33,39, −25,16] | **q1** |
| test-dev, L | **+7,39** [+4,04, +10,73] | (không tune trên test-dev) | q3 (từ ảnh train) |
| test-dev, S (PPO-S + MPC − H\*_S + MPC) | +8,02 [+6,28, +9,75] | | AVIF-50 |

**Đọc kết quả (chỉ mô tả):**
- Mode tốt nhất của heuristic thay đổi theo điều kiện kênh: q3 → q1 khi fading đổi nhanh hoặc phản hồi kém. Tune lại
  cho đúng điều kiện cộng thêm +92,6 / +80,3.
- PPO-LA (train trên L gốc, không train lại) chuyển sang các điều kiện đó tốt hơn heuristic đóng băng (+61 / +51),
  nhưng kém heuristic được tune lại cho đúng điều kiện (−31 / −29).
- Trên nội dung ảnh mới (test-dev), khi heuristic chỉ được tune trên ảnh train, PPO hơn heuristic + MPC ở cả hai loại
  kênh (+7,4 trên L, +8,0 trên S).
- Chưa có phép so "cả hai cùng được thiết kế lại cho từng biến thể": muốn có phải train lại PPO cho từng biến thể.

### 4. Recall detector (tuân thủ Qmin / đúng hạn, %)

| Cấu hình | H\*_L + MPC | PPO-LA + MPC |
|---|---|---|
| L | 50,9 / 52,2 | 47,9 / 51,6 |
| L_fade | 35,0 / 35,9 | 38,4 / 44,3 |
| L_bursty | 50,7 / 52,3 | 45,9 / 50,7 |
| L_badfb | 32,5 / 33,2 | 34,5 / 40,7 |
| test-dev, L | 39,6 / 40,0 | 37,1 / 39,6 |

Return và recall không luôn cùng chiều.

## Hệ quả cho Paper A

- **Đóng góp 2 vững hơn nhiều:** bộ điều khiển thiết kế trên Shannon tụt ở mọi biến thể LA và trên nội dung mới; đảo
  thứ hạng PPO-S/T:best giữ nguyên ở 6/6 cấu hình LA.
- **Đóng góp 3 (đã sửa sau ablation E2b):** bộ hiệu chỉnh thắng dự báo kiểu RobustMPC ở mọi cấu hình LA (+3,4 đến
  +17,8 khi bỏ chia SNR), nhưng **chia theo SNR không đóng góp gì**. Chưa tách được phần nào của bộ hiệu chỉnh tạo ra
  lợi thế.
- **Câu "PPO không hơn heuristic + MPC" chỉ đúng ở LA gốc.** Kết quả đầy đủ:
  - bộ điều khiển nào cũng phụ thuộc vào điều kiện lúc thiết kế;
  - chính sách học chuyển sang điều kiện mới (ảnh mới, suy giảm kênh chưa thấy) tốt hơn heuristic chọn mode cố định
    đã đóng băng;
  - nhưng heuristic được tune lại cho đúng điều kiện thì tốt nhất trong các phép so đã có.

## 5. E2b: hiệu chỉnh không chia theo SNR (ablation do phản biện đề nghị)

- **Cấu hình:** cùng bộ hiệu chỉnh (cùng quan sát số bit đã ACK mỗi slot, cùng cách lấy mẫu phân phối thực nghiệm, cùng
  planner và cổng), chỉ gộp mọi quan sát của mọi UAV vào một ngăn (`a9_mpc.PoolRates`, hậu tố `+mpcpl`).
- **Seed:** giống E1 / E3, ghép cặp.

| Cấu hình | Incumbent | Chia SNR − không chia | Không chia − kiểu RobustMPC | Không chia − danh nghĩa |
|---|---|---|---|---|
| L | H\*_L | −0,05 [−3,10, +3,00] | +6,67 [+2,60, +10,75] | +17,01 |
| L | PPO-LA | +1,15 [−0,03, +2,33] | +5,76 [+4,40, +7,13] | |
| L_fade | H\*_L | −0,66 [−3,62, +2,30] | +10,78 [+8,06, +13,49] | +37,08 |
| L_fade | PPO-LA | +2,19 [+0,77, +3,62] | +8,14 [+6,58, +9,69] | |
| L_bursty | H\*_L | +1,62 [−1,71, +4,94] | +6,54 [+3,33, +9,75] | +19,30 |
| L_bursty | PPO-LA | −0,70 [−1,96, +0,57] | +3,37 [+1,77, +4,97] | |
| L_badfb | H\*_L | −0,38 [−5,44, +4,68] | +14,11 [+10,02, +18,21] | +28,60 |
| L_badfb | PPO-LA | −0,07 [−2,14, +2,01] | +9,29 [+7,15, +11,43] | |
| S | H\*_S | −0,53 [−1,23, +0,16] | −0,35 [−1,33, +0,62] | +1,11 (n.s.) |
| test-dev L | H\*_L | +0,25 [−3,10, +3,59] | +17,79 [+13,15, +22,43] | +4,99 |
| test-dev L | PPO-LA | +0,45 [−1,19, +2,08] | +10,47 [+8,55, +12,38] | |
| test-dev S | H\*_S | −0,74 [−1,88, +0,41] | +12,79 [+10,98, +14,59] | +2,52 |

**Kiểm tra số ngăn (20/11):** SNR kỳ vọng mà bộ hiệu chỉnh thấy nằm trong 10,8–18,9 dB (không có giá trị âm; 3 seed ×
S/L, script ghi nhận SNR trong `CalRates._snr_db`). Vì `BIN_DB = 1e6`, `PoolRates` đúng là một ngăn duy nhất. Với dải
SNR hẹp như vậy, ngăn 2 dB chỉ chia thành khoảng 5 ngăn, nên ablation này chỉ kiểm tra được trong dải đó.

**Kết luận:**
- Chia theo SNR kỳ vọng **không** tạo ra lợi ích: 11/12 CI chứa 0; chỉ PPO-LA trên L_fade +2,2.
- **Đính chính:** ở trên mình từng viết "hiệu chỉnh theo vị trí/SNR là cơ chế riêng". Điều đó **sai**.
- Lợi thế so với baseline kiểu RobustMPC đến từ chính bộ hiệu chỉnh theo phân phối thực nghiệm. Chưa tách được phần
  nào quyết định: lấy mẫu cả phân phối thay vì một giá trị, gộp dữ liệu nhiều UAV và nhớ lâu hơn, hay không có hệ số
  bi quan (1 + sai số lớn nhất).

## 6. Độ trễ quyết định (mọi arm có MPC, E1–E3, E1b, E2b)

- p99 trung bình theo episode: 39–48 ms.
- p99 lớn nhất của một episode: 44–64 ms. Riêng PPO-S + MPC trên L lên tới 64,0 ms.
- Tỉ lệ slot vượt 50 ms ≤ 0,31 %.
- Như R1: giả định các quyết định mode trong một slot chạy song song.
- Chi tiết trong `runs/uav_semcom/paperA_r2/summary_c.log`.

## 7. Test-dev: CI theo video (bootstrap hai chiều seed × 49 video, recall tuân thủ Qmin, điểm %)

| Kênh | Phép so | Δ recall [95 % CI] |
|---|---|---|
| S | PPO-S − T:best (khối A) | +4,24 [+1,13, +9,88] |
| S | H\* + MPC hiệu chỉnh − kiểu RobustMPC | −1,08 [−1,74, −0,64] |
| L | PPO-S − T:best (khối A) | +2,01 [+0,71, +3,15] |
| L | H\* + MPC hiệu chỉnh − kiểu RobustMPC | −0,54 [−1,31, +0,29] |
| L | PPO-LA + MPC − H\* + MPC | −2,48 [−3,50, −1,58] |

**Ghi chú:**
- Return là tổng của cả episode nên không tách được theo video; phân tích theo video vì vậy làm trên recall.
- Trên test-dev L, PPO-LA + MPC có return cao hơn (+7,4) nhưng recall tuân thủ Qmin thấp hơn (−2,5, có ý nghĩa
  thống kê theo cả seed lẫn video).
- PPO-S có return thấp hơn T:best trên L (−59,7) nhưng recall cao hơn (+2,0).
- Return và recall đi ngược chiều.

## 8. E2c: thành phần nào của bộ hiệu chỉnh tạo ra lợi thế (21/11)

**Câu hỏi** (phản biện sau bản viết lại 20/11): bộ hiệu chỉnh hơn baseline kiểu RobustMPC từ +6,5 đến +17,8. Lợi thế
đó đến từ đâu? Hai bộ dự báo khác nhau ở nhiều điểm cùng lúc.

**Cách làm:**
- Đi từ RobustMPC-style (`rh`) sang bộ hiệu chỉnh gộp (`pl`, = bộ hiệu chỉnh không chia SNR, E2b), mỗi bước đổi đúng
  một yếu tố (`a9_mpc.AblRates`, hậu tố `+mpcab0..7`).
- Incumbent H\*, cùng planner, cùng ngân sách, cùng seed với E1/E3 (ghép cặp), 7 cấu hình.
- **Kiểm tra cài đặt:**
  - `ab7` (cài lại `pl`) trùng `pl` ở 680/700 episode; chênh trung bình ≤ 0,06. Vài episode lệch do sai khác số thực
    trong xác suất lấy mẫu làm đổi một lần rút, sau đó episode rẽ nhánh.
  - Bỏ các slot tốc độ 0 khỏi RobustMPC-style (`ab0`): không đổi gì (mọi CI chứa 0, phần lớn trùng hệt), nên baseline
    không bị yếu giả do trung bình điều hòa gặp giá trị 0.

**Kết quả** (chênh return, CI 95% ghép cặp; * = CI không chứa 0):

| Bước (đổi một yếu tố) | S | L | L_fade | L_bursty | L_badfb | TD-S | TD-L |
|---|---|---|---|---|---|---|---|
| Bỏ hệ số bi quan | −1,0* | +0,3 | +7,2* | +0,3 | +0,8 | +2,1* | +2,7 |
| Trung bình cộng thay điều hòa | −1,2* | +1,4 | −2,7* | +1,9 | +0,0 | +7,5* | +8,7* |
| Lấy mẫu thay một giá trị (5 slot gần nhất) | +0,4 | +2,1 | −0,6 | +1,2 | +2,1* | −1,7* | +0,2 |
| Nhớ cả episode thay 5 slot | +1,8* | −1,1 | +4,9* | +0,0 | +0,0 | +1,0 | −0,2 |
| Gộp mọi UAV thay từng UAV | −0,5 | +1,8 | +3,9* | +4,0* | +9,0* | +4,3* | +2,9 |
| Chờ đủ 20 s dữ liệu | +0,2 | +2,2 | −1,8 | −1,0 | +2,2 | −0,4 | +3,6* |
| **Tổng (pl − rh)** | −0,4 | **+6,7*** | **+10,8*** | **+6,5*** | **+14,1*** | **+12,8*** | **+17,8*** |
| Lấy mẫu − trung bình (ở đầu đã gộp, `pl − ab6`) | +1,6* | −0,6 | +1,8 | +2,1 | +4,5 | −1,7* | −1,1 |
| Chia SNR (cal − pl) | −0,5 | −0,1 | −0,7 | +1,6 | −0,4 | −0,7 | +0,3 |

**Kết luận:**
- **Không có một yếu tố duy nhất.** Lợi thế cộng dồn từ vài bước nhỏ.
- **Yếu tố ổn định nhất là gộp quan sát của mọi UAV:** dương ở 6/7 cấu hình, có ý nghĩa ở 4 (+3,9 đến +9,0).
- **Thứ hai là tránh ước lượng quá bi quan:**
  - trung bình điều hòa → trung bình cộng: +7,5 / +8,7 trên ảnh test-dev;
  - bỏ hệ số bi quan: +7,2 trên L_fade.
  - Cả hai làm ngược lại trên S (−1,0 / −1,2) và L_fade (−2,7).
- **Lấy mẫu cả phân phối KHÔNG phải cơ chế:** so với dùng trung bình ở cùng điểm gộp, dấu đổi theo cấu hình (+1,6 trên
  S, −1,7 trên TD-S, còn lại CI chứa 0). Chia SNR cũng không (E2b).
- **Hệ quả cho Paper A** (theo quy tắc phản biện đã nêu trước khi chạy):
  - Mô tả phương pháp là **một bộ ước lượng tốc độ thực nghiệm đơn giản**: trung bình các tốc độ đã ACK, gộp mọi UAV,
    nhớ cả episode.
  - **Không** nhấn mạnh lập kế hoạch ngẫu nhiên (stochastic planning) hay điều kiện theo SNR.
  - Lý do RobustMPC-style kém hơn: mỗi UAV chỉ có 5 quan sát, cộng thêm hai lớp bảo thủ (điều hòa, hệ số bi quan), nên
    ước lượng nhiễu và thấp.

**File:** `runs/uav_semcom/paperA_r2/e2c_test*.json`, `e2c_testdev*.json`, `summary_e2c.json/.log`; script
`paperA_r2d_summary.py`.
