# Bước C (phản biện #46): bộ dự đoán số vật thể nhân quả và chẩn đoán Qmin, trên nội dung giữ lại

Ngày chạy: 2026-11-18.
- **Protocol:** [đăng ký trước](swarmbridge-atc-protocol.md), commit `d08c3bb3`, trước mọi lần chạy.
- **Lựa chọn dev đóng băng:** commit `7f8a4ee9` (`dev.json`), trước khi chạy test.

**Code:**
- `atc_predict.py`, `atc_run.py`, `atc_summary.py`.
- `at1_task.py`: nguồn `pA` / `pB` / `pC`, luật `rvq`, bảng test-dev, FP.
- `env.py`: `info_extra_s`, tính chi phí tính toán của bộ dự đoán trên UAV.
- `test_atc.py`: 6 test; cùng `test_at1.py` và `test_env.py`, 39 test đều pass.

**Kết quả** ở `runs/uav_semcom/atc/`: `pred.json`, `dev.json`, `dev.log`, `test.json`, `test.deliv.json.gz`,
`summary.log`, `summary.json`.

## Trả lời ngắn

**Cổng C1 đăng ký trước: FAIL (3 trong 8 điều kiện) → theo quy tắc, DỪNG hướng giá trị tác vụ.** Không thêm
detector lớn hơn, không thêm bộ học phức tạp hơn.

Lý do quan trọng hơn bản thân cổng: **trên nội dung mới (VisDrone test-dev, 49 video), chính oracle biết số vật thể
cũng không còn dư địa.**
- RV-count − C\* = **−0,97** [−2,64, +1,10]; ở A-T1 trên tập test cũ là +1,53.
- Cơ chế "ảnh ít vật thể gửi rẻ để giải phóng kênh" mà A-T1 tìm ra không tổng quát được sang nội dung mới.

**Bộ dự đoán được chọn (pC, detector nhẹ) − C\* = +0,20** [−0,94, +1,63]. Recall theo ảnh giảm 3,0 điểm.

**Điều còn đứng:** oracle biết kết quả detector thật (RV-gt) vẫn +2,17 [+1,13, +4,07] trên test-dev. Thông tin
"ảnh này cho bao nhiêu TP ở mỗi mode" có giá trị. Nhưng không bộ dự đoán nhân quả nào ở đây xấp xỉ được nó, và số
vật thể không phải là đại diện ổn định của nó.

## 1. Đã làm gì

**Bài toán (quyết định 1 của phản biện #46): detector-only.**
- Mục tiêu là recall detector đúng hạn. PSNR là chỉ số phụ, không phải ràng buộc.
- Đây là một **bài toán khác** A5. Không kết quả nào ở đây được báo là cải thiện bài toán A5.
- Qmin được giữ thành kịch bản riêng C0.

**Nội dung giữ lại thật.**
- Test là pool **test-dev**: VisDrone2019-DET-test-dev, 403 khung, 49 video. Đây là split VisDrone khác với các pool
  train / val (DET-val).
- Không trùng video và không trùng hash ảnh giữa các pool; code kiểm tra bằng assert.
- Detector được train trên DET-train.
- Test-dev từng được dùng chỉ để đánh giá ở A5 / A6-L1 / G-L2. Không lựa chọn nào của bước C nhìn vào nhãn test-dev.
- Seed kênh mới: 9600–9699. Tập test A-T1 (9200–9299) không dùng lại.

**Bộ dự đoán số vật thể.**
- Đích: log(1 + n_gt). Ridge, α chọn bằng GroupKFold theo video trên train.
- Dự đoán trên train là out-of-fold (dùng để tune λ).

| Bộ | Đặc trưng có trên UAV lúc quyết định | Chi phí tính vào env |
|---|---|---|
| pA | payload probe AVIF-5 (đã có ở mọi arm), số pixel | 0 |
| pB | pA + 13 thống kê ảnh G-L2 | +10 ms |
| pC | pA + 14 đặc trưng detector G-L2 (detector A5 ở imgsz 480) | +35 ms |

pC dùng **cùng trọng số với detector ở trạm mặt đất**, nên là kịch bản lạc quan.

**Luật điều khiển:** RV, chọn argmax_m U(m) − λ·b̂(m), với U = n̂ / n̄_train · μ_m. Đây cùng dạng với oracle số vật thể
của A-T1. Scheduler, cách bay và thông tin kênh giữ cố định như A-T1.

**Chọn theo từng bậc (payload trước):**
- tune λ trên train (seed 0–29);
- chọn P\* trên val (84 khung, 14 video, seed 3000–3019);
- đóng băng, rồi chạy test một lần.

## 2. Dev (train / val; đóng băng trước test)

**Độ chính xác đếm offline trên val** (Spearman với n_gt; RMSE của log(1 + n)):

| Bộ | Spearman val | RMSE val | Đoán hằng số (trung bình train) |
|---|---|---|---|
| pA | −0,02 | 0,93 | 0,62 |
| pB | 0,07 | 0,95 | 0,62 |
| pC | **0,62** | **0,29** | 0,62 |

- pA và pB tệ hơn đoán hằng số trên video mới. Out-of-fold trên train, pA có Spearman 0,46 và pB 0,52, nên chúng
  học được đặc điểm của các video train, không phải số vật thể.
- **Train tái lập A-T1:** q3 48,02; q3 + MPC 48,21; RV-gt 51,29; RV-count 50,02. Env không bị thay đổi khi thêm
  các dòng test-dev.
- **C\* = q3 + MPC** (A5 objective), giống A-T1. **Q-C\* = q3** (theo recall tuân thủ Qmin trên train).

**Recall đúng hạn trên val:**

| Arm | Recall val |
|---|---|
| C\* | 47,80 |
| RV-pA | 47,06 |
| RV-pB | 47,86 |
| **RV-pC** | **49,43** |
| RV-count (oracle) | 49,22 |

**Quy tắc chọn:**
- pA hồi phục −0,52 phần dư địa oracle số vật thể trên val (dư địa +1,42).
- pB không qua quy tắc +0,3 điểm và RMSE ≤ 0,9.
- pC được xét và qua → **P\* = pC**.

## 3. Test (test-dev, seed 9600–9699, chạy một lần)

Chỉ số tính trên toàn khối việc; khung trễ hoặc rớt đóng góp 0 TP và không có detection.

| Arm | Recall | Recall theo ảnh | FP / khung | AP50 | Recall tuân thủ Qmin | Tỉ lệ đúng hạn | Mbit | Năng lượng (kJ) | p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| q3 (= F_task = Q-C\*) | 40,19 | 46,62 | 9,56 | 26,89 | 40,07 | 0,872 | 75,3 | 78,4 | 3,5 |
| **C\*** = q3 + MPC | **39,77** | 46,18 | 9,73 | 26,41 | 39,36 | 0,911 | 68,0 | 74,4 | 42,8 |
| RV-mean | 40,60 | 47,61 | 9,85 | 27,16 | 38,38 | 0,901 | 66,1 | 74,2 | 3,5 |
| RV-pA | 40,67 | 47,03 | 10,03 | 27,08 | 39,09 | 0,922 | 64,1 | 73,2 | 3,7 |
| RV-pB | 39,49 | 46,07 | 9,58 | 26,55 | 38,05 | 0,916 | 65,3 | 74,1 | 3,7 |
| **RV-pC (P\*)** | **39,97** | 43,18 | 10,70 | 25,73 | 32,94 | 0,986 | 40,1 | 68,1 | 3,7 |
| RV-count (oracle) | 38,80 | 40,13 | 10,03 | 25,80 | 31,92 | 0,990 | 36,9 | 67,9 | 3,8 |
| RV-gt (oracle) | 41,94 | 45,92 | 12,17 | 27,27 | 33,07 | 0,989 | 36,7 | 67,6 | 3,7 |

**Cổng C1** (RV-pC so với C\*; CI chính là bootstrap hai chiều: seed môi trường × video):

| Điều kiện | Giá trị | Kết quả |
|---|---|---|
| 1. ΔR ≥ +1,0 | +0,20 (seed ghép cặp: +0,16 [−0,29, +0,62]) | **FAIL** |
| 2. Cận dưới CI hai chiều > 0 | [−0,94, +1,63] | **FAIL** |
| 3. ΔR ≥ 0,5·H_cnt | H_cnt = −0,97 [−2,64, +1,10] | qua (vì oracle âm) |
| 4. Recall theo ảnh giảm không quá 1 điểm | −3,00 | **FAIL** |
| 5. FP / khung tăng không quá 10 % | 10,70 so với 9,73 (+10,0 %) | qua (vừa sát) |
| 6. Tỉ lệ đúng hạn | +7,4 điểm | qua |
| 7. Năng lượng | −8,5 % | qua |
| 8. Độ trễ quyết định p99 | 3,7 ms | qua |

→ **DỪNG.**

**Các so sánh phụ** (điểm recall; CI hai chiều | CI theo seed):

| So sánh | Hai chiều | Theo seed | Recall theo ảnh |
|---|---|---|---|
| RV-pA − C\* | +0,90 [−0,57, +1,70] | +0,90 [+0,52, +1,27] | +0,85 |
| RV-mean − C\* | +0,82 [−0,09, +1,57] | +0,82 [+0,39, +1,26] | +1,43 |
| RV-pC − RV-pA | −0,70 [−2,16, +1,75] | −0,73 [−1,13, −0,34] | −3,85 |
| RV-count − RV-mean | −1,79 [−3,70, +0,79] | −1,83 [−2,35, −1,31] | −7,48 |
| RV-gt − C\* | **+2,17 [+1,13, +4,07]** | +2,14 [+1,70, +2,59] | −0,26 |
| RV-gt − RV-count | +3,14 [+2,23, +4,74] | +3,15 [+2,98, +3,33] | +5,79 |
| q3 + MPC − q3 | −0,42 [−1,26, +0,29] | −0,42 [−0,76, −0,08] | −0,44 |

**Độ chính xác đếm offline trên test-dev** (tính sau test):

| Bộ | Spearman | RMSE log(1 + n) |
|---|---|---|
| pA | 0,28 | 0,91 |
| pB | 0,21 | 1,06 |
| pC | 0,82 | 0,46 |

**Độ tập trung của RV-pC − C\*:**
- Theo mật độ vật thể: ảnh ít vật thể −7782 TP, trung bình −4298, nhiều vật thể +14 505.
- Theo mức tắc nghẽn của C\*: thấp −1,91, trung bình −0,19, **cao +2,74**.
- Theo video: 33 / 49 video có lợi. Video 9999973 kéo tổng xuống −4241 TP.

## 4. C0: Qmin là ràng buộc trong tập hành động (chẩn đoán, không có cổng)

Ở mỗi luật, chỉ các mode có PSNR dự đoán từ probe ≥ Qmin + δ được phép; λ và δ tune trên train theo recall tuân thủ
Qmin.

| Arm | H_Q: recall tuân thủ Qmin − Q-C\* (q3) | Recall detector-only của arm − C\* |
|---|---|---|
| RVQ-count (oracle) | +0,69 [−0,27, +2,34] | **+1,81 [+0,99, +3,05]** |
| RVQ-pC | −0,08 [−0,91, +1,09] | +0,51 [−0,10, +1,43] |
| RVQ-gt (oracle) | +2,36 [+1,46, +3,90] | +3,58 [+2,78, +4,89] |
| RVQ-mean | +0,03 [−0,75, +0,61] | +0,43 [−0,31, +1,09] |

**Đọc theo quy tắc đã đăng ký:** H_Q(count) có CI chứa 0, nên hai bài toán (detector-only và có sàn chất lượng) khác
nhau đáng kể.

**Điểm đáng chú ý (không thuộc quy tắc):**
- Có ràng buộc Qmin, oracle số vật thể lại cho recall detector-only **cao hơn** C\* (+1,81). Không ràng buộc thì nó
  thấp hơn (−0,97).
- Ràng buộc chất lượng ngăn luật hạ mode quá tay trên nội dung mới. Đây là dấu hiệu rằng λ tune trên train quá
  mạnh cho test-dev (xem mục 5), không phải bằng chứng cho một phương pháp mới.

## 5. Giải thích (đọc sau khi xem kết quả; không dùng để cứu hướng này)

**Vì sao oracle số vật thể mất dư địa trên test-dev.**
- Test-dev có trung bình **46,9** vật thể / khung, train có **68,6** (theo manifest).
- U = n̂ / n̄_train · μ_m, nên phần lớn khung test-dev trông như "ít giá trị". Với λ = 40 tune trên train, luật dồn
  sang mode rẻ: avif_5 / mbt q1 chiếm 59 % ở RV-count.
- Kết quả: tắc nghẽn gần như biến mất (đúng hạn 0,99), nhưng TP mất ở chính các khung bị hạ chất lượng.
- Recall theo ảnh giảm 7,5 điểm so với RV-mean.
- Đây là lỗi hiệu chỉnh theo phân phối, chính điều mà phản biện #46 yêu cầu kiểm tra bằng nội dung độc lập. Tập
  test A-T1 lấy cùng nguồn DET-val với train nên không lộ ra.

**Bộ dự đoán nhẹ (pC) đếm tốt hơn hẳn** (Spearman test-dev 0,82), nhưng thừa hưởng cùng lỗi.
- Lợi ích của nó chỉ xuất hiện khi kênh tắc (+2,74). Khi kênh không tắc, nó làm hại (−1,91).
- Recall theo ảnh −3,0; AP50 −0,68.

**Luật đơn giản đã ngang mức.** RV-mean và RV-pA cao hơn C\* khoảng +0,8…+0,9: CI theo seed > 0, nhưng CI hai chiều
chứa 0.
- q3 + MPC thấp hơn q3 −0,42 trên test-dev, nên C\* (chọn trên train) là baseline yếu hơn dự kiến.
- Nhưng đổi baseline sau khi thấy test là không hợp lệ, và dù đổi, pC cũng không vượt.

**Phần còn đứng là RV-gt +2,17.** Thông tin đúng là "ảnh này cho bao nhiêu TP ở mỗi mode", không phải "ảnh này có
bao nhiêu vật thể".
- G-L2 đã cho thấy thông tin đó không dự đoán được từ đặc trưng rẻ, và lần này cũng vậy.
- RV-gt có FP / khung cao nhất (12,2), và recall theo ảnh không tăng (−0,26).

## 6. Kết luận và những gì giữ lại

- **Theo quy tắc: DỪNG hướng giá trị tác vụ** (bộ dự đoán giá trị ảnh cho phân bổ airtime) trên menu mode và detector
  này.
- Không dùng detector lớn hơn, không thêm học phức tạp, không dùng lại tập test-dev để tune.
- **Kết luận A-T1 cần sửa** (đưa vào bảng sửa kết luận):
  - "+1,53 là biết số vật thể" chỉ đúng trên nội dung cùng nguồn với train.
  - Trên nội dung độc lập, oracle số vật thể là −0,97 (CI chứa 0).
  - Cơ chế giảm tắc nghẽn là có thật, nhưng phụ thuộc hiệu chỉnh λ theo phân phối số vật thể.
- **Giữ nguyên:**
  - A-T1 PASS theo protocol của nó.
  - A6-L1 và G-L2 (âm).
  - A-X9 (DRL dừng).
- **Không đề xuất thuật toán mới.** Bước tiếp theo là quyết định của phản biện. Mình không đề xuất mở lại hướng này
  bằng cách hiệu chỉnh λ theo phân phối: đó là cứu kết quả sau khi đã thấy test.

## Giới hạn

- Một detector ở một ngưỡng hoạt động; menu mode A5.
- Khung được rút độc lập từ pool.
- 49 video test-dev; độ rộng CI hai chiều cho thấy biến thiên giữa video lớn (khoảng ±1,3 điểm).
- pC lạc quan: dùng trọng số giống trạm mặt đất; chi phí 35 ms đo khi máy rảnh.
- AP50 tính từ detection đã lưu (hash khớp `sha256.txt`).
