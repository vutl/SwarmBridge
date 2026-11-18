# Paper A, R1: cùng chính sách đã đóng băng, kênh Shannon so với kênh LA thực tế

Ngày chạy: 18/11/2026.
- **Code:** `paperA_r1.py` (tune / test), `paperA_r1_summary.py`.
- **Kết quả:** `runs/uav_semcom/paperA_r1/`: `tune.json`, `test.json`, `test.deliv.json.gz`, `test.log`,
  `summary.json`, `summary.log`.

## Cách làm

- **Cùng điều kiện cho mọi chính sách:** workload (test pool), 5 UAV, 2 kênh.
- **Seed:** 100 seed mới 9800–9899, ghép cặp giữa các chính sách.
- **Không train mới.** Checkpoint PPO lấy nguyên từ A5 (PPO-S, train trên Shannon) và A-X9 (PPO-LA, train trên LA).
  Mỗi PPO có 3 checkpoint; lấy trung bình theo seed trước khi ghép cặp.
- **Tune duy nhất:** heuristic H\* được chọn trên seed train 0–29 của từng loại kênh:
  - H\*_S = `fixed_avif_50`;
  - H\*_L = `fixed_mbt2018-mean_q3`, trùng A-X9.
- **Hai khối:**
  - **A:** luật bay / kênh / công suất của A5 (T:best, bay theo nhiệm vụ). Đây là kiểu đánh giá của các bài DRL cho
    UAV SemCom, với các luật chưa tune làm đối chứng.
  - **B:** scheduler JA và cách bay pacedB3, đóng băng từ A-X9 stage 1.
- **Hai loại kênh:**
  - **S:** tốc độ Shannon (PHY tham chiếu của A5).
  - **L:** link adaptation mức block, gồm CQI trễ, OLLA 10 % BLER, ARQ, nhiễu đồng kênh bền, LA biết lịch.
- **Tách rõ hai loại chính sách chạy trên L:**
  - chính sách thiết kế trên S rồi chạy thẳng trên L (chuyển giao zero-shot);
  - chính sách tune / train trên chính L.
- **Chỉ số:**
  - return A5;
  - ok;
  - recQ: recall detector tuân thủ Qmin (đúng hạn và PSNR ≥ Qmin);
  - recOn: recall đúng hạn, bỏ qua PSNR; luật chưa tune không ghi cột này;
  - năng lượng;
  - p99 độ trễ quyết định tích hợp.

## Kết quả

### Kênh Shannon (S)

| Khối | Chính sách | Return | Δ [95 % CI] | ok | recQ | recOn | p99 ms |
|---|---|---|---|---|---|---|---|
| A | Random (chưa tune) | −110,00 | −172,70 [−175,42, −169,98] | 0,592 | 31,93 | — | — |
| A | Round robin (chưa tune) | −79,05 | −141,75 [−149,42, −134,09] | 0,679 | 40,68 | — | — |
| A | Max rate (chưa tune) | −112,26 | −174,96 [−178,21, −171,70] | 0,551 | 33,02 | — | — |
| A | Greedy (chưa tune) | −73,05 | −135,75 [−143,22, −128,27] | 0,716 | 42,91 | — | — |
| A | **T:best (tune trên S), tham chiếu** | 62,70 | 0 | 0,969 | 56,73 | 57,76 | 0,9 |
| A | **PPO-S** | 64,82 | **+2,12 [+1,39, +2,86]** | 0,990 | 57,88 | 58,19 | 5,2 |
| B | H\*_S | 66,91 | +0,34 [−0,67, +1,35] | 0,997 | 58,78 | 58,78 | 4,0 |
| B | **H\*_S + MPC hiệu chỉnh, tham chiếu** | 66,56 | 0 | 0,992 | 57,61 | 58,02 | 45,0 |
| B | H\*_S + MPC không hiệu chỉnh | 66,77 | +0,21 [−1,19, +1,61] | 0,991 | 58,39 | 58,57 | 45,1 |
| B | H\*_L (tune trên L) chạy trên S | 42,82 | −23,75 [−25,02, −22,47] | 0,972 | 55,88 | 56,76 | 3,9 |
| B | PPO-S | 63,87 | −2,69 [−3,47, −1,91] | 0,989 | 57,85 | 58,18 | 7,5 |
| B | PPO-S + MPC hiệu chỉnh | 62,78 | −3,79 [−4,56, −3,01] | 0,986 | 56,94 | 57,58 | 47,7 |

### Kênh LA thực tế (L)

| Khối | Chính sách | Return | Δ [95 % CI] | ok | recQ | recOn | p99 ms |
|---|---|---|---|---|---|---|---|
| A | Random (chưa tune) | −217,59 | −60,81 [−67,00, −54,63] | 0,397 | 20,66 | — | — |
| A | Round robin (chưa tune) | −286,06 | −129,28 [−135,03, −123,54] | 0,237 | 13,67 | — | — |
| A | Max rate (chưa tune) | −220,97 | −64,20 [−69,81, −58,59] | 0,354 | 21,09 | — | — |
| A | Greedy (chưa tune) | −288,50 | −131,72 [−137,63, −125,82] | 0,243 | 13,97 | — | — |
| A | **T:best (tune trên S), tham chiếu** | −156,77 | 0 | 0,619 | 31,90 | 37,25 | 0,9 |
| A | PPO-S (zero-shot) | −180,24 | −23,47 [−27,78, −19,16] | 0,605 | 33,47 | 35,23 | 4,1 |
| B | H\*_L | −39,91 | −7,98 [−11,31, −4,65] | 0,895 | 51,38 | 52,19 | 3,8 |
| B | **H\*_L + MPC hiệu chỉnh, tham chiếu** | −31,93 | 0 | 0,909 | 50,95 | 52,28 | 44,9 |
| B | H\*_L + MPC không hiệu chỉnh | −51,77 | **−19,84 [−24,36, −15,32]** | 0,873 | 49,22 | 50,24 | 45,4 |
| B | H\*_S (tune trên S) chạy trên L | −130,81 | **−98,88 [−106,20, −91,56]** | 0,748 | 43,80 | 43,80 | 4,1 |
| B | PPO-S (zero-shot) | −54,74 | −22,81 [−25,70, −19,93] | 0,859 | 48,94 | 50,77 | 7,3 |
| B | PPO-S (zero-shot) + MPC hiệu chỉnh | −38,43 | −6,50 [−9,21, −3,79] | 0,893 | 49,66 | 51,48 | 48,3 |
| B | PPO-LA (train trên L) | −38,02 | −6,09 [−9,05, −3,12] | 0,859 | 46,80 | 51,92 | 7,7 |
| B | PPO-LA + MPC hiệu chỉnh | −30,41 | **+1,51 [−1,55, +4,58]** | 0,888 | 47,95 | 51,66 | 48,2 |

## Đọc kết quả (chỉ mô tả số đo)

1. **Mô hình kênh dùng lúc thiết kế quyết định kết quả trên kênh LA, cho mọi loại phương pháp.**
   - Heuristic tune trên S, chạy trên L: −98,9 so với tham chiếu L.
   - PPO train trên S, chạy trên L: −22,8.
   - Luật bay / kênh của A5 (thiết kế trên S): toàn khối A trên L đều dưới −150.
   - Ngược lại, heuristic tune trên L chạy trên S: −23,8.
2. **Hiệu chỉnh online từ ACK:**
   - Trên L, MPC hiệu chỉnh so với chính MPC đó không hiệu chỉnh, cùng incumbent: +19,8.
   - MPC hiệu chỉnh kéo PPO-S zero-shot từ −22,8 lên −6,5.
   - Trên S (mô hình kênh của planner đúng), hiệu chỉnh không đổi kết quả: +0,2, n.s.
3. **Train / tune đúng loại kênh:** trên L, PPO-LA + MPC và H\*_L + MPC không khác nhau có ý nghĩa: +1,51 [−1,55, +4,58].
   - PPO-LA + MPC có recall tuân thủ Qmin thấp hơn (47,95 so với 50,95); recall đúng hạn gần bằng (51,66 so với 52,28).
4. **Đánh giá kiểu khối A trên S tái lập được kết quả thường thấy:** PPO-S hơn heuristic tune +2,1 và hơn các luật
   chưa tune 136–175.
5. **Độ trễ:** các chính sách có MPC có p99 45–48 ms, với giả định quyết định mode chạy song song trong ngân sách 50 ms
   (như A-E8 / A-X8). Chính sách không MPC: 1–8 ms.

## Ghi chú cho bản thảo (theo phản biện #49)

- Kết luận chỉ áp dụng cho các chính sách PPO, môi trường và baseline đã thử.
- PPO-S được train với luật slot T:best (khối A). Ở khối B nó chạy với scheduler JA, khác lúc train; ghi rõ điều này
  khi trình bày.
- "+19,8" là MPC hiệu chỉnh so với MPC không hiệu chỉnh cùng incumbent, không phải so với heuristic không MPC (cái đó
  là +7,98).
- Không có số PPO-LA trên S: quan sát của nó cần đặc trưng LA, kênh S không có.
