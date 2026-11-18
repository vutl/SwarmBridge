# A-X8 (phản biện #43): đối chứng công bằng cho tổ hợp lập lịch + bộ điều khiển mode dưới link adaptation thực tế

Ngày chạy: 2026-11-18. Không train; dùng 3 checkpoint PPO mode-only cũ của K5.

**Code:**
- `ax8_run.py`, `ax8_summary.py`.
- `ax2_eval.py`: thêm mode `h=` / `hc=`, MPC lấy heuristic làm incumbent, `--pool`, latency tích hợp, goodput/BLER.
- `test_ax8.py`: 5 test mới; tổng 40 test đều pass.

**Kết quả** ở `runs/uav_semcom/ax8/`: `tune.json`, `test.json`, `test.deliv.json.gz`, `summary.json`, `summary.txt`.

Tiếp nối [A-X7](swarmbridge-ax7-information-mode-result.md).

## Trả lời ngắn (cổng quyết định của phản biện #43)

**PPO–MPC không còn hơn heuristic mạnh nhất đã tune cùng điều kiện.**
- Cách bay paced, scheduler JA: **PPO–MPC − heuristic đã tune = −1,25 [−4,18, +1,68]** (theo checkpoint +0,2 / −5,9 /
  +2,0).
- Stack tốt nhất là **MPC chọn lọc hiệu chỉnh online, lấy heuristic làm incumbent**: hơn PPO–MPC +5,00 [2,47, 7,53].
- Lợi thế +13,27 của A-X7 là do baseline chưa tune: luật mode của heuristic chưa được chọn lại cùng JA và cách bay.

**Phần có ích, theo thứ tự độ bền:**
1. **Scheduler JA:** hơn least-slack +8,0…+15,9 dưới **mọi** bộ điều khiển mode, ở cả hai cách bay.
2. **Cách bay paced** (hover khi hàng đợi dài): mọi stack đều chọn paced trên seed train. Đổi lại năng lượng tăng khoảng
   12 kJ.
3. **MPC chọn lọc với goodput hiệu chỉnh online:** hơn chính incumbent của nó +3,75 [0,53, 6,97] (paced) và +12,33
   [10,43, 14,23] (mission). Đây là phần "lập kế hoạch với mô hình tốc độ được hiệu chỉnh".
4. **Hiệu chỉnh online cho luật mode thiển cận (greedy/budget): không giúp.** Luật heuristic tốt nhất ở cả hai cách bay
   là mode cố định, nên bộ hiệu chỉnh không có gì để sửa. Các luật greedy/budget có hiệu chỉnh còn kém hơn bản không
   hiệu chỉnh trên seed train (tốt nhất −87,5 so với −78,4).
5. **PPO làm incumbent không đóng góp:**
   - PPO zero-shot thua heuristic −13,1 (paced);
   - PPO–MPC thua MPC lấy heuristic làm incumbent −5,0 (paced), −1,6 n.s. (mission).

## Luồng làm

**Env:**
- K5_la, link adaptation biết lịch, nhiễu persistent, quan sát giống K5.
- Scheduler JA = `JA:stat:NR:a1:m0` (nhân quả).
- Hai cách bay:
  - **paced** = di chuyển và công suất của T:best@K5 (pacedB2);
  - **mission** = T:best@K5_la, bay theo lộ trình.
- Không dùng SRS.

**Tune:** pool train, seed 0–29, cho từng cách bay.

| Họ | Thành phần |
|---|---|
| H | 22 luật mode: 14 mode cố định, greedy, greedy + budget τ ∈ {0,5, 0,75, 1, 1,25, 1,5, 2, 3} |
| HC | greedy / budget với tốc độ kỳ vọng thay bằng **đúng tốc độ hiệu chỉnh của MPC** (`RateErrTracker.predict` dùng `CalRates` cold, theo slot, nhân channel share). Mode cố định không dùng tốc độ |
| PPO | zero-shot |
| PPO + MPC | MPC chọn lọc, hiệu chỉnh online cold (A-E8: N 8, H 4, ngân sách 50 ms) |
| HC* + MPC | MPC lấy heuristic HC* tốt nhất làm incumbent |

Mỗi stack chọn cách bay tốt nhất trên seed train.

**Test:** pool test, **seed 8900–8999 mới, chưa từng dùng**, cả hai cách bay. Cộng thêm scheduler least-slack (`ls` /
`tbest_env`) với cùng các mode, để đo tương tác.

## Kết quả tune (seed train)

| Cách bay | H* | HC* | PPO | PPO+MPC | HC*+MPC |
|---|---|---|---|---|---|
| paced | `fixed_mbt2018-mean_q3` −59,38 | (giống H*, mode cố định) | −80,62 | −61,92 | **−55,80** |
| mission | `fixed_avif_20` −71,46 | (giống H*) | −103,60 | −81,92 | −59,64 |

- Luật thích ứng tốt nhất là `greedy+budget1`: −78,4 (paced) / −78,6 (mission).
- Luật hiệu chỉnh tốt nhất: −87,5 / −96,6.
- Mọi stack chọn **paced**.

## Kết quả test (100 seed giữ riêng)

| Stack (cách bay) | Return | ok | Goodput (kbit/s) | Năng lượng (kJ) | Jain | UAV kém nhất | Recall detector | Mode p50 / p99 (ms) | Tích hợp p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| JA + fixed q3 (paced) | −37,61 | 0,923 | 400,7 | 79,0 | 0,997 | 0,858 | **0,530** | 0 / 0 | 1,7 |
| JA + fixed q3 + MPC (paced) | **−33,86** | 0,923 | 399,7 | 77,2 | 0,997 | 0,862 | 0,517 | 21,2 / 39,7 | 41,6 |
| JA + PPO (paced) | −50,67 | 0,894 | 421,1 | 81,5 | 0,996 | 0,816 | 0,512 | 1,7 / 3,4 | 5,3 |
| JA + PPO + MPC (paced) | −38,86 | 0,912 | 420,1 | 78,5 | 0,997 | 0,844 | 0,509 | 23,2 / 43,2 | 45,2 |
| least-slack + fixed q3 (paced) | −53,54 | 0,900 | 389,4 | 80,7 | 0,997 | 0,837 | 0,517 | 0 / 0 | 0,7 |
| JA + fixed avif_20 (mission) | −56,43 | 0,759 | 331,4 | 65,9 | 0,995 | 0,694 | 0,388 | 0 / 0 | 1,3 |
| JA + avif_20 + MPC (mission) | −44,10 | 0,817 | 359,9 | 66,7 | 0,996 | 0,752 | 0,420 | 21,9 / 40,7 | 42,4 |
| JA + PPO + MPC (mission) | −45,69 | 0,865 | 446,4 | 68,0 | 0,990 | 0,752 | 0,482 | 24,4 / 44,4 | 46,2 |

Trễ hạn bằng 0 ở mọi stack.

### So sánh theo cặp (Δ return, CI 95 %; cùng cách bay, cùng JA)

| So sánh | paced | mission |
|---|---|---|
| C1 hiệu chỉnh trong heuristic (HC* − H*) | 0 (H* là mode cố định) | 0 |
| **C2 cổng quyết định: PPO+MPC − HC*** | **−1,25 [−4,18, 1,68]** (ckpt +0,2 / −5,9 / +2,0) | +10,74 [6,14, 15,34] (ckpt +13,8 / +6,8 / +11,7) |
| C3 PPO+MPC − PPO | +11,81 [9,95, 13,67] | +17,03 [14,64, 19,42] |
| C4 HC*+MPC − HC* (MPC lấy heuristic làm incumbent) | **+3,75 [0,53, 6,97]** | **+12,33 [10,43, 14,23]** |
| **C5 incumbent học được: PPO+MPC − HC*+MPC** | **−5,00 [−7,53, −2,47]** (ckpt −3,5 / −9,7 / −1,8) | −1,59 [−5,58, 2,40] |
| C6 PPO − HC* | −13,06 [−16,08, −10,04] | −6,29 [−11,96, −0,62] |
| JA − least-slack, theo mode (H* / PPO / PPO+MPC) | +15,9 / +14,9 / +13,6 | +8,0 / +14,0 / +13,3 |
| **I1 tương tác so với heuristic hiệu chỉnh** ((JA−LS) với PPO+MPC − với HC*) | **−2,35 [−5,94, 1,25]** | **+5,37 [2,28, 8,47]** |

**Stack tốt nhất mỗi loại** (mọi stack chọn paced trên seed train):
- PPO+MPC − stack không học tốt nhất (HC*+MPC): −5,00 [−7,53, −2,47].
- PPO+MPC − heuristic hiệu chỉnh: −1,25 [−4,18, 1,68].

**Latency:**
- Lập lịch JA mỗi slot: p99 ≤ 2 ms.
- Mỗi quyết định mode có MPC: p99 40–44 ms.
- **Tích hợp** (lập lịch + quyết định mode lâu nhất trong cùng slot): p99 41–46 ms. Tỉ lệ vượt 50 ms ≤ 0,1 %.
- Nếu các quyết định mode trong cùng một slot phải chạy tuần tự, tổng mỗi slot là p99 145–180 ms. Mỗi quyết định có ngân
  sách 50 ms riêng, theo sự kiện ảnh.
- Hai thang thời gian được báo riêng như phản biện yêu cầu.

**Nhiệm vụ downstream** (recall detector trên các ảnh được giao):
- Paced: heuristic q3 đạt 0,530, PPO+MPC 0,509.
- Mission: PPO+MPC 0,482 so với avif_20 0,388, vì mode cố định avif_20 có chất lượng thấp.
- Ở paced, thêm MPC tăng return nhưng giảm nhẹ recall (0,530 → 0,517); PPO+MPC 0,509. Return (PSNR/deadline/năng
  lượng) và recall không hoàn toàn cùng chiều. Reward chưa phải mục tiêu nhiệm vụ, đúng như phản biện lưu ý.

## Diễn giải

1. **Luận điểm "PPO–MPC hơn heuristic" không đứng** khi heuristic được tune cùng JA và cùng tập cách bay. Đây là lần
   thứ hai một lợi thế lớn biến mất khi đối chứng được tune công bằng (lần trước: A-X2b). Kết quả A-X7 +13,27 được ghi
   lại là so sánh với baseline yếu.
2. **Thành phần có giá trị là lập kế hoạch MPC với tốc độ được hiệu chỉnh online**, đặt trên một incumbent tốt.
   - Nó cộng thêm +3,8 đến +12,3 cho bất kỳ incumbent nào.
   - Nó không cần PPO: với checkpoint zero-shot hiện có, PPO là incumbent kém hơn mode cố định đã tune.
   - Riêng việc hiệu chỉnh tốc độ cho luật mode thiển cận thì không giúp.
3. **JA là thành phần bền nhất:** +8…+16 dưới mọi mode.
   - Tương tác lập lịch × mode so với heuristic hiệu chỉnh không ổn định (−2,4 n.s. paced, +5,4 mission).
   - Vì vậy +4–5 ở A-X7 không phải một cơ chế phối hợp bền.
4. **Hệ quả cho Paper A:**
   - Với PHY thực tế và baseline công bằng, đóng góp đứng được là: một scheduler gán tổ hợp nhân quả + MPC chọn lọc,
     hiệu chỉnh online, có hạn thời gian, trên một incumbent rẻ.
   - Phần "DRL" (PPO incumbent) chưa có giá trị chứng minh được ở env này.
   - Câu hỏi còn lại cho DRL: một PPO mode-only **train trên LA** (với JA và paced) có thành incumbent tốt hơn mode cố
     định không? Checkpoint hiện có train trên Shannon.

## Giới hạn

- PPO chỉ zero-shot (train trên Shannon K5); chưa train trên LA, theo yêu cầu của phản biện.
- JA cố định một thành viên (tune ở A-X7 với workload mission / avif_20); chưa tune lại cùng cách bay paced.
- Heuristic incumbent là một mode cố định cho cả episode. Tuy vậy thêm MPC đã làm nó thích ứng.
- Hiệu chỉnh cold mỗi episode. Một env (K5_la, nhiễu persistent, LA biết lịch). 100 seed.
- Recall tính theo bảng fidelity của A5 (TP của detector trên ảnh giao đúng hạn / tổng GT của mọi ảnh đã chụp).

## Đề xuất cho phản biện #44 (chưa chạy)

1. **Ghi nhận kết quả âm:** PPO–MPC ≈ heuristic tune công bằng (paced). MPC lấy heuristic làm incumbent là stack tốt
   nhất.
2. **Một thí nghiệm có giới hạn cho phần DRL:** train PPO mode-only trên LA (JA + paced, 3 seed, cùng ngân sách 2M bước
   như A5).
   - So làm incumbent cho MPC với fixed q3 + MPC, trên seed giữ riêng.
   - Nếu không hơn, Paper A nên được viết lại quanh **lập kế hoạch có hiệu chỉnh online + lập lịch tổ hợp** thay vì DRL.
3. **Tune lại JA** cùng cách bay paced, và thêm vài cách bay paced (τ khác) cho cả hai họ.
4. Shadowing / bộ lọc vẫn hoãn.

Không viết lại Method trước khi có quyết định ở mục 2.
