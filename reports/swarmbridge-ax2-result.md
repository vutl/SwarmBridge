# A-X2 (phản biện #37) + baseline kiểu bài báo + PPO huấn luyện trực tiếp trên MCS

Ngày chạy: 2026-11-17. Commit code: a96ddbcc (A-X2), c172524a (baseline chuẩn); kết quả trong `runs/uav_semcom/ax2/`
và `runs/uav_semcom/ax3/`. Không viết lại Method của Paper A (theo #37).

## 0. Luồng thí nghiệm (đọc trước)

1. **A-X2.** Học **chọn kênh** (C-only, mode theo heuristic) hoặc **kênh + mode** (C+Mo). Di chuyển và công suất bị
   **khoá đúng** theo luật slot của T:best, tính lại trên trạng thái hiện tại ở mỗi slot (`ax2_env.GymSchedEv`).
   - Thiết lập: TimeMaskablePPO, cấu hình như A5; 2 regime (K5, ref) × 2 biến thể × 3 seed train = 12 run; chọn
     checkpoint theo val 3000–3019.
   - Eval: 100 seed mới 8800–8899, so với các baseline của #37.
   - Stress MCS: giữ controller K5 (train trên Shannon), chạy trên `K5_mcsg` (MCS biết nhiễu) và `K5_mcs` (MCS không
     biết nhiễu), 50 seed.
2. **(a) Baseline kiểu bài báo.** Thêm Random, Round Robin, Max Rate (= Nearest khi cùng công suất) và greedy chưa
   tune (`std_baselines.py`), đặt cạnh T:best và PPO A5 trên cùng 100 seed. Lý do: người dùng thấy các bài trong `pdf/`
   chỉ so RL với luật đơn giản.
3. **(b) PPO huấn luyện trực tiếp trên `K5_mcsg`.** Joint ×3 và mode-only ×3, 2M bước. Đây là phép so sánh công bằng
   "RL vs heuristic khi kênh thực tế hơn", vì heuristic cũng được tune trên chính `K5_mcsg`.
4. **Khảo sát độ phức tạp kịch bản** của 19 bài RL và 6 repo trong `pdf/`.

Kiểm tra cài đặt:
- 12 test `test_ax2.py`: mask và hành động thực thi đúng heuristic; policy không tác động được lên move/power; chọn
  đúng kênh/mode của T:best thì tái tạo T:best; harness khớp A5 và A-X1.
- 7 test `test_std_baselines.py`.
- Đối chiếu chéo: PPO mode-only − T:best trên K5 bằng +2,26 ở cả `ax2_eval` lẫn `a5_eval`.

## 1. A-X2: kết quả chính (return, CI 95% trên seed môi trường, kèm giá trị theo từng seed train)

| Đối chứng | K5 | Theo seed train K5 | ref | Theo seed train ref |
|---|---|---|---|---|
| X1 C-only − T:best | **+2,52 [+1,68, +3,35]** | 2,93 / 2,28 / 2,33 | **+3,07 [+2,54, +3,59]** | 3,03 / 3,14 / 3,02 |
| X2 C-only − SCH* tốt nhất | +2,51 [+1,67, +3,35] | | +3,08 [+2,55, +3,60] | |
| X3 C+Mo − T:best + PPO mode-only | +0,06 [−0,38, +0,50] | −3,03 / 4,39 / −1,18 | **+1,48 [+1,25, +1,72]** | 1,63 / 0,95 / 1,88 |
| X4 C+Mo − C-only | −0,20 [−0,85, +0,45] | −2,21 / 1,35 / 0,26 | **−1,76 [−2,09, −1,42]** | |
| X5 C+Mo − jc\|jm (head A5) | **−1,98 [−2,61, −1,36]** | | **+1,53 [+1,22, +1,85]** | |
| X6 PPO–MPC (cold) − PPO, trên T:best | +0,34 [−0,03, +0,72] | 0,04 / 1,11 / −0,12 | **+1,11 [+0,92, +1,30]** | |
| X7 C+Mo+MPC − C+Mo | **+0,42 [+0,12, +0,73]** | | **+1,05 [+0,86, +1,24]** | |
| X8 C+Mo+MPC − T:best+PPO+MPC | +0,14 [−0,42, +0,70] | −2,41 / 3,46 / −0,62 | **+1,42 [+1,19, +1,65]** | |
| I1 (OOD) kênh × head mode | +0,48 [+0,10, +0,87] | | +0,86 [+0,59, +1,13] | |
| I2 (OOD) | −0,88 [−1,70, −0,06] | | −1,87 [−2,40, −1,35] | |

Return tuyệt đối:

| | T:best | C-only | C+Mo | C+Mo+MPC | PPO mode-only | jc\|jm |
|---|---|---|---|---|---|---|
| K5 | 62,11 | 64,63 | 64,43 | 64,85 | 64,37 | 66,41 |
| ref | 58,94 | 62,01 | 60,25 | 61,30 | 58,76 | 58,72 |

Đọc:
- **Học chọn kênh, giữ mode heuristic (C-only) là thành phần ổn định duy nhất.** Mức lợi +2,5 / +3,1, cả 6 seed train
  cùng dấu. Cả họ 16 luật chọn kênh SCH (A-X1) cũng không đạt mức này.
- **Học thêm mode (C+Mo) thì tuỳ regime.**
  - K5: trung bình ngang PPO mode-only nhưng phân tán theo seed train (−3 đến +4), và thua head joint cũ của A5 (X5 −1,98).
  - ref: hơn PPO mode-only +1,48 nhưng thua C-only −1,76.
  - Ở ref, không cấu hình học mode nào thắng C-only + mode heuristic (PPO mode-only ≈ T:best: −0,18, xem mục 4).
  - Ở K5, chỉ head joint cũ của A5 với move/power heuristic (jc|jm, 66,41) cao hơn C-only (64,63). C+Mo train lại thì
    không.
- **PPO–MPC cold** thêm +0,3 đến +1,1, đều dương ở cả hai scheduler; ở K5 trên T:best chưa có ý nghĩa thống kê (X6).
- **Không kết luận "không kém hơn"** từ những CI có chứa 0 (X3 K5, X4 K5, X8 K5).
- I1/I2 dùng head mode được train dưới scheduler khác, nên chỉ là chẩn đoán OOD. I2 âm ở cả hai regime: khi đã học
  scheduler, ghép thêm PPO mode-only (train dưới T:best) làm giảm return.

Các chỉ số khác (K5 / ref):
- **ok rate:**
  - K5: 0,968 (T:best) so với 0,972 (C-only), 0,989 (C+Mo), 0,990 (PPO).
  - ref: 0,990 so với 0,996 và 0,999.
  - Tỉ lệ trễ bằng 0 ở mọi cấu hình.
- **Recall detector:** K5 0,565–0,582; ref 0,594–0,598. Chênh lệch ≤ 0,02 giữa mọi cấu hình, nên **không được nói A-X2
  cải thiện ngữ nghĩa**. Lợi ích về return chủ yếu đến từ PSNR, ok rate và chi phí thời gian.
- **Năng lượng** gần như bằng nhau: K5 65,9–66,9 kJ, ref 39,6–40,5 kJ.
- **Công bằng:** Jain ≈ 1. min-UAV ok: K5 0,919 (T:best) → 0,937 (C-only) / 0,966 (C+Mo).
- **Kênh:**
  - Thời gian đồng kênh: C-only gần như như T:best ở K5 (0,827 so với 0,839), C+Mo giảm còn 0,715.
  - Hệ số dùng kênh: 1,59 / 1,21 so với 1,64.

## 2. Các kiểm tra bắt buộc của #37

**Kiểm tra `_rates()`** (tỉ lệ tốc độ phục vụ thực tế / dự đoán = mô hình hiệu chỉnh × M/N_busy, tính trên tổng; NMAE
theo từng slot):

| | T:best | C-only | C+Mo | PPO mode-only |
|---|---|---|---|---|
| K5 | 1,48 / 0,83 | 1,41 / 0,84 | 1,01 / 0,58 | 1,05 / 0,57 |
| ref | 1,00 / 0,53 | 0,97 / 0,58 | 0,88 / 0,51 | 0,84 / 0,47 |
| K5_mcsg | 1,32 / 0,87 | 1,23 / 0,85 | 1,34 / 1,01 | 1,43 / 1,01 |
| K5_mcs | 0,81 / 1,68 | 0,98 / 1,74 | 1,15 / 1,82 | 1,73 / 2,41 |

- **Thiên lệch tổng phụ thuộc scheduler.** Với T:best ở K5, mô hình chia kênh đánh giá thấp tốc độ thực tế khoảng 48%
  (dùng chung kênh cho phép nhiều hơn M UAV phát cùng lúc). Dưới C+Mo thì gần như không thiên lệch (1,01). Ở ref, C+Mo
  và PPO lại đánh giá cao khoảng 12–16%.
- **Sai số từng slot lớn** (NMAE 0,5–0,9). `_rates()` chỉ dùng được như mô hình trung bình, không dùng được cho dự đoán
  từng slot.
- Cũng vì vậy, calibration online (cold) là bắt buộc khi đổi scheduler.

**Cold start:**
- Planner chạy lần đầu ở giây 8,6 / 8,8 / 10,1 (K5, phân vị 10/50/90) và 10,1 / 11,3 / 15,1 (ref), trên episode
  khoảng 75–80 s. Trước thời điểm đó controller chính là PPO.
- Tỉ lệ bị gate: K5 0,12; ref 0,17. Abort: 0,1–0,2% ở K5, 0 ở ref.

**Độ trễ tích hợp** (p99, ms):
- Scheduler học được mỗi slot: K5 3,6–4,0; ref 2,5–2,8 (T:best: 0,3–0,7).
- Head mode: khoảng 3 ms; head mode + MPC: 41,7–43,3, dưới mức 50.
- Số quyết định vượt 50 ms: ≤ 0,0001.

## 3. Stress MCS (controller K5 train trên Shannon, 50 seed)

| | K5_mcsg | K5_mcs |
|---|---|---|
| Heuristic tune cho môi trường đó (`fixed_mbt2018-mean_q3`, không dùng chung kênh) | −1,93 | −1,93 |
| T:best K5 (dùng chung kênh) | −15,91 | −340,65 |
| PPO mode-only | −10,42 | −315,51 |
| C-only | −11,37 | −244,25 |
| C+Mo | −9,83 | −262,96 |
| PPO + MPC (cold) | −1,48 | −316,10 |
| C+Mo + MPC (cold) | −0,79 | −262,16 |
| jc\|jm | −40,74 | −257,66 |

- **Lợi thế dùng chung kênh ở K5 phụ thuộc giả định Shannon lý tưởng.** Sang MCS, mọi controller train trên Shannon
  thua heuristic không dùng chung kênh (S2–S4: −8 đến −9 ở `K5_mcsg`, khoảng −240 đến −310 ở `K5_mcs`).
- **`K5_mcsg`:** PPO–MPC cold lấy lại +8,9 [+7,3, +10,6] (X6) và +9,0 (X7), ngang heuristic tốt nhất. Đây là bằng chứng
  cho vai trò của calibration online khi mô hình sai.
- **`K5_mcs`:** MPC không giúp được (X6 −0,60, chỉ 1–6 lần chuyển mode mỗi episode). Lỗi nằm ở việc dùng chung kênh,
  là quyết định MPC không điều khiển. Scheduler học được giảm thiệt hại (X1 +96) nhưng vẫn rất xa heuristic.

## 4. (a) Baseline kiểu bài báo (100 seed mới, test pool)

| Policy | K5 | ref | K5_mcsg |
|---|---|---|---|
| Random | −111,2 (ok 0,59) | −44,0 | −170,4 |
| Round Robin | −79,4 (0,68) | +44,6 (0,94) | −231,1 |
| Max Rate / Nearest | −110,5 (0,56) | +5,1 (0,79) | −176,2 |
| Greedy (heuristic gốc, chưa tune) | −74,2 (0,71) | +48,8 (0,96) | −226,8 |
| Mode cố định tốt nhất (đã tune) | +26,4 | +49,0 | −0,5 |
| **T:best (đã tune)** | **+62,1** | **+58,9** | **−0,5** |
| PPO joint (3 seed) | +63,5 | +58,2 | +0,1 (train trên mcsg) |
| PPO mode-only (3 seed) | +64,4 | +58,8 | −0,9 (train trên mcsg) |

PPO so với baseline chưa tune tốt nhất:
- K5: +138 [+131, +145].
- ref: +9,4 đến +10,0 (khoảng +20%), ngang mức lợi trong các bài báo.
- K5_mcsg: +170.

PPO so với T:best:
- K5: joint +1,43 [+0,48, +2,38]; mode-only +2,26 [+1,37, +3,15].
- ref: joint −0,78 [−1,39, −0,17]; mode-only −0,18 [−0,73, +0,38].

Các luật chưa tune thua chủ yếu vì chọn chất lượng ảnh cao (PSNR 43–44 dB). Hàng đợi bị dồn, ảnh trễ deadline, ok
rate chỉ còn 0,55–0,71 ở K5. Chỉ riêng việc tune mode đã kéo return lên rất nhiều (mode cố định tốt nhất +26 ở K5),
và phần lớn khoảng cách giữa "RL" và "luật đơn giản" là khoảng cách mà một heuristic được tune cũng đạt được.

## 5. (b) PPO huấn luyện trực tiếp trên K5_mcsg (100 seed mới)

| | Return | Theo seed train | ok | Năng lượng (kJ) | Thời gian nhiệm vụ |
|---|---|---|---|---|---|
| T:best của mcsg (`fixed_mbt2018-mean_q3`) | −0,51 | | 0,944 | 66,7 | 75 s |
| PPO joint | +0,11 | −4,45 / +7,87 / −3,08 | 0,981 | 96,0 | 108 s |
| PPO mode-only (slot của T:best) | −0,89 | −1,13 / −0,73 / −0,79 | 0,941 | 66,5 | 75 s |

So với T:best:
- Joint: +0,63 [−1,15, +2,40], không có ý nghĩa thống kê.
- Mode-only: −0,37 [−0,85, +0,11].

Đọc:
- **Trên kênh MCS biết nhiễu, RL train đúng môi trường chỉ ngang heuristic đã tune, không thắng.**
- **Mode-only** hội tụ về đúng mode cố định của heuristic (PSNR 31,9 so với 32,0).
- **PPO joint:**
  - Rất không ổn định giữa các seed train: một seed +7,9, hai seed âm.
  - Đổi lại dùng nhiều năng lượng hơn 44% và kéo dài nhiệm vụ thêm khoảng 33 s, nên ok rate cao hơn (0,981).
  - Cả 3 run joint đều có checkpoint tốt nhất ở bước cuối (2M), tức là **chưa hội tụ**. Train dài hơn có thể thay đổi
    kết quả; chưa kiểm tra.

## 6. Kịch bản của mình có "dễ" hơn các bài báo không? (19 bài RL + 6 repo trong `pdf/`)

| Khía cạnh | Các bài báo | Mình |
|---|---|---|
| Số UAV/agent | 1 UAV ở 7 bài; 2–5 ở 11 bài | 5 (hoặc 3) |
| Số người dùng | thường 3–50, có bài 200 | 1 trạm gốc |
| Tài nguyên tần số | thường 10–80 kênh con hoặc 10–20 MHz | 2 × 1 MHz, cho phép dùng chung kênh |
| Kênh | khoảng 12/19 có fading hoặc shadowing; 6 có nhiễu đồng kênh hoặc NOMA; một số có CSI cũ (Doppler), nhiều anten, ray tracing | LoS/NLoS theo xác suất, Rician K=10 dB / Rayleigh theo từng slot, chỉ có CSI thống kê, nhiễu đồng kênh; stress MCS |
| Không gian hành động | phần lớn nhỏ hơn hoặc liên tục | 11 × 3 × 3 mỗi UAV + 14 mode |
| Baseline | luật chưa tune (khoảng 10 bài), RL khác (khoảng 11), solver tham lam hoặc cận trên (khoảng 9); **trong 19 bài khảo sát, không bài nào tune họ heuristic** | heuristic tốt nhất trong một họ, tune trên pool riêng |
| Thống kê | 6/19 nêu số lần chạy; **trong 19 bài, không bài nào có CI** cho mức lợi | 100 seed mới, CI, độ dao động theo seed train |
| Mức lợi báo cáo | +5–12% so với RL khác; +15–90% so với luật đơn giản | +3–7% so với T:best ở K5 (mục 4) |

Ở những bài có baseline mạnh, RL hoà hoặc thua:
- Ghép cặp Gale–Shapley thắng MARL (*MARL Resource Allocation for UAV Networks*, tr. 22).
- DP thắng PPO ở quy mô nhỏ (*Adaptive Task-Oriented SemCom for UAVs…*, tr. 17–18).
- Round Robin thắng SchedCP về utility ở 500–600 kHz (*DRL-Based User Scheduling for Collaborative Perception*, tr. 10).

Phạm vi (sửa theo phản biện #38): các nhận xét trên chỉ áp dụng cho 19 bài + 6 repo có trong `pdf/`, không khái quát cho toàn bộ SOTA.

**Kết luận:** kịch bản của mình không đơn giản hơn trên các chiều quan trọng. Nó **ít người dùng và ít tài nguyên tần
số hơn**, không có CSI cũ hay người dùng di chuyển. Khoảng cách RL–heuristic nhỏ chủ yếu vì baseline đã được tune. Mục
4 xác nhận điều này: so với luật kiểu bài báo, PPO thắng +10 (ref) đến +138 (K5).

## 7. Hệ quả cho Paper A (để phản biện quyết định; chưa sửa Method)

1. **Một mình RL không phải là đóng góp.** Trên cả ba regime (Shannon K5, ref, MCS), RL chỉ hơn heuristic tune kỹ từ 0
   đến +3. Ở ref và `K5_mcsg` RL train đúng môi trường thì không thắng. Đây là kết quả nên báo cáo trung thực, kèm bảng
   baseline kiểu bài báo để người đọc thấy tại sao các con số khác văn liệu.
2. **Thành phần có bằng chứng ổn định:**
   - (i) Scheduler kênh học được: +2,5 / +3,1, ổn định qua seed train. Nhưng lợi ích này chưa được kiểm tra dưới MCS
     với scheduler train trên MCS.
   - (ii) Calibration online + MPC chọn lọc khi mô hình sai: +9 ở `K5_mcsg`, A-E5…A-E11. Không giúp khi lỗi nằm ở chọn
     kênh (`K5_mcs`).
3. **Không được nói:**
   - "Cải thiện ngữ nghĩa": recall gần như không đổi.
   - "Không kém hơn": từ CI chứa 0.
   - "Tối ưu di chuyển/công suất có hại": A-X1 C6 chỉ là bằng chứng đồng thích nghi.
   - "PPO joint kém trên MCS": run chưa hội tụ.
4. **Câu hỏi mở cho phản biện:**
   - Có nên train C-only trên `K5_mcsg` (3 seed, khoảng 1,3 giờ) để xem scheduler học được còn giữ +2,5 khi kênh thực tế?
   - Có nên kéo dài PPO joint trên mcsg (4M bước)?
   - Có nên thêm chiều khó mà văn liệu có còn mình thiếu (nhiều người dùng / kênh con, CSI trễ) thay vì làm kịch bản mới?

## Tái tạo

```
# A-X2 (WSL, 6 CPU): train 12 run, eval, tóm tắt
python ax2_train.py OUT --regime {K5,ref} --variant {c,cm} --seed {0,1,2}
python ax2_eval.py runs/uav_semcom/ax2/eval_K5.json --regime K5 --seeds 8800:8900 --procs 6 --combos ...
python ax2_eval.py runs/uav_semcom/ax2/stress_K5_mcsg.json --regime K5 --env-regime K5_mcsg --seeds 8800:8850 ...
python ax2_summary.py runs/uav_semcom/ax2/eval_K5.json
# (a), (b)
python a5_eval.py eval runs/uav_semcom/ax3/std_K5.json --regimes K5 --pool test --fresh --seeds 8800:8900 \
  --policies H:std_rand,H:std_rr,H:std_maxrate,H:greedy,T:fixed,T:best,<PPO specs>
python a5_train.py OUT --regime K5_mcsg --rl-part {joint,mode} --seed {0,1,2} [--heur T:best]
python ax3_summary.py
```

Nhật ký train (không có checkpoint): `runs/uav_semcom/ax2/train/`, `runs/uav_semcom/ax3/train/`.
