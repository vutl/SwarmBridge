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

- Hiệu chỉnh theo SNR thắng baseline kiểu RobustMPC ở mọi cấu hình LA và trên test-dev.
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
- **Đóng góp 3 có cơ chế riêng:** hiệu chỉnh theo SNR kỳ vọng tại vị trí UAV thắng dự báo chỉ dựa trên lịch sử (kiểu
  RobustMPC) ở mọi cấu hình LA (+6,6 đến +18,0).
- **Câu "PPO không hơn heuristic + MPC" chỉ đúng ở LA gốc.** Kết quả đầy đủ:
  - bộ điều khiển nào cũng phụ thuộc vào điều kiện lúc thiết kế;
  - chính sách học chuyển sang điều kiện mới (ảnh mới, suy giảm kênh chưa thấy) tốt hơn heuristic chọn mode cố định
    đã đóng băng;
  - nhưng heuristic được tune lại cho đúng điều kiện thì tốt nhất trong các phép so đã có.
