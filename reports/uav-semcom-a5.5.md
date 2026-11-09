# SwarmBridge hướng A — A5.5: kiểm toán độ bền của kết luận A5

Bắt đầu 2026-11-09 tối, sau A5 (báo cáo `reports/uav-semcom-a5.md`, commit `399d36b5`). A5 không bị sửa: mọi kết quả A5.5 nằm ở `runs/uav_semcom/a55/`.

**Câu hỏi:** ưu thế của PPO trong A5 (đặc biệt +12,3 ở K5 trên test-dev) có còn không khi:
1. đối thủ là một heuristic **tự hiệu chỉnh trực tuyến**;
2. độ bất định được tính đủ ba nguồn: seed huấn luyện, seed môi trường, và **chuỗi video**?

## 0. Quy tắc định trước (commit trước khi chạy bất kỳ đánh giá nào)

**Lưu ý về tính khám phá.** Thiết kế của A5.5 được làm *sau khi đã xem* kết quả test A5. Vì vậy:
- mọi phân tích trên seed test cũ 7000..7099 chỉ là **khám phá** (exploratory);
- kết luận chính chỉ lấy từ seed môi trường **mới 8000..8099**, chưa từng dùng ở A4/A5. Đây là đánh giá xác nhận (confirmatory).
- Ảnh test và test-dev thì đã được A5 dùng. Không còn ảnh có nhãn nào chưa dùng mà lại đến từ video khác: phần còn lại của test-dev là các khung khác của cùng 49 video. Bootstrap theo video ở 0.4 là cách đo độ nhạy theo nội dung.

### 0.1 Heuristic tự hiệu chỉnh (SC)

**Cấu trúc:** giống các heuristic A5 (điều nhịp theo backlog `pacedB`, chọn mode theo `budget`, tuỳ chọn `reuse`). Khác biệt duy nhất: ba đại lượng mà A5 lấy từ mô hình được **hiệu chỉnh trực tuyến** từ những gì UAV đã quan sát được (nhân quả, chỉ dữ liệu trong quá khứ).

1. **Tốc độ phục vụ thực tế.** A5 dùng `expected_rate × channel_share`, một tốc độ mô hình: Shannon, không tính nhiễu.
   - SC nhân tốc độ này với hệ số `ρ`, ước lượng từ các slot mà UAV *đang có backlog đủ lớn*:
     - `ρ = (Σ bit thực gửi + c) / (Σ bit mô hình dự đoán + c)`;
     - dùng chung cho cả đội;
     - trọng số mỗi slot giảm theo λ = 0,95 (hằng số thời gian khoảng 20 s);
     - prior c = 5·10⁶ bit, tương đương ρ = 1.
   - Bit thực gửi lấy từ các đoạn truyền đã kết thúc. Đây là thông tin mà bên phát biết được qua ACK.
2. **Payload thực tế.**
   - `ρ_b = (Σ bit thật + c_b) / (Σ bit dự đoán + c_b)`, lấy trên các ảnh **đã encode xong** (c_b = 2·10⁶).
   - Payload dự đoán của mọi mode được nhân với `ρ_b`.
3. **Hệ số an toàn `s` theo kết quả (AIMD).**
   - Mỗi ảnh của đội bị rớt hoặc trễ (biết được khi hết hạn): s ← min(1,3·s; 3).
   - Mỗi ảnh giao đúng hạn: s ← max(0,995·s; 1).
   - Thời gian truyền dự đoán được nhân với s.
   - Mức trễ mục tiêu ở trạng thái cân bằng khoảng 2%.
   - Các hằng số trên được cố định trước, không tinh chỉnh.

**Họ SC:**
- `sc:SLOT+budgetτ[+reuse]`, với SLOT ∈ {greedy, pacedB1.5, pacedB2, pacedB3} và τ ∈ {0,5; 0,75; 1; 1,25; 1,5; 2; 3}.
- Tổng cộng 56 thành viên, cùng lưới với các họ budget/reuse của A5.
- Tinh chỉnh trên pool **train**, seed 0..29, như A5. **T:SC** = thành viên có return train cao nhất.

**Kiểm tra code bắt buộc** (test):
- (a) Khi tắt cả ba hiệu chỉnh, SC cho hành động **trùng từng bước** với heuristic A5 cùng tên.
- (b) Tính nhân quả: đổi fading/giá trị của các slot **tương lai** thì hành động hiện tại không đổi.

### 0.2 Đối thủ "heuristic mạnh nhất không nhìn test" (H*)

Với mỗi regime, H* là thành viên có return **train** cao nhất trong {T:best của A5, T:SC}. Quy tắc chọn này không dùng test.

### 0.3 So sánh xác nhận (seed 8000..8099; pool test và test-dev; regime ref, K5, phy_mcs_blind)

| Δ chính | Ghi chú |
|---|---|
| **PPO mode-only (best-val, A5) − H*** | Giả thuyết chính H1 |
| PPO joint (best-val) − H* | |
| PPO joint K5 5M (best-val) − H* | Chỉ ở K5 |
| T:SC − T:best của A5 | SC có giúp heuristic không |

- Tất cả theo cặp, cùng seed môi trường.
- Model PPO: các checkpoint A5 đã chọn trên val; không train lại.
- Heuristic: cả 21 regime trên pool test (rẻ); PPO chỉ ở 3 regime.

### 0.4 Ba mức độ bất định

1. **Theo seed môi trường:**
   - trung bình 3 model trên mỗi seed;
   - t-test theo cặp trên 100 seed;
   - đây là CI của A5.
2. **+ theo video (cluster bootstrap có mô phỏng lại):**
   - B = 50 bản: mỗi bản lấy mẫu có hoàn lại các **chuỗi video** của pool (test: 28 chuỗi; test-dev: 49 chuỗi);
   - pool ảnh của bản đó = mọi khung của các chuỗi được chọn, giữ đúng số lần lặp;
   - chạy mọi policy trên 20 seed môi trường riêng của bản, lấy Δ trung bình của bản.
   - Seed của bản: ban đầu ghi là 8100 + 20b + i. **Điều chỉnh trước khi chạy:** với B = 50, dải đó lên tới 9099, đè lên seed 9000..9099 dành cho A6.0-L1. Vì vậy đổi thành 20000 + 20b + i.
3. **+ theo seed huấn luyện (bootstrap phân tầng):**
   - như (2), nhưng trong mỗi bản lấy thêm mẫu có hoàn lại 3 model PPO trong 3 model.

CI là phân vị 2,5–97,5% của các bản. Đơn vị độc lập được báo rõ: 3 model huấn luyện; 28 hoặc 49 chuỗi video; 100 seed môi trường (không phải 100 tập dữ liệu độc lập).

### 0.5 Quy tắc kết luận

Cho từng (regime, pool) của Δ = PPO mode-only − H*:

| Điều kiện | Kết luận |
|---|---|
| CI mức 3 không chứa 0 và Δ > 0 | **Ưu thế bền** |
| CI mức 1 không chứa 0, nhưng CI mức 3 chứa 0 | **Ưu thế chưa chắc** (phụ thuộc video hoặc seed huấn luyện) |
| CI mức 1 chứa 0 | **Không có ưu thế** |

Tương tự nếu Δ < 0. Kết luận được báo cáo nguyên trạng, kể cả khi SC bắt kịp PPO.

### 0.6 Kiểm tra lại phép tính task fidelity của A5.2

- Đo lại toàn bộ (ảnh, mode) bằng cùng detector và codec (bước GPU của A6.0-L1). Mức trùng khớp TP/số detection với các dòng A5.2 được báo cáo.
- Test đơn vị cho phép ghép detection–nhãn (class-aware, IoU 0,5, vùng ignore).
- Tính lại AP50 từ detection đã lưu, đối chiếu với `coco_val.json` của A5.

## 1. Code, test và cách chạy

| Thành phần | File |
|---|---|
| Heuristic tự hiệu chỉnh (SC), họ `sc:` / `tv:` | `swarmbridge/experiments/uav_semcom/a55_heur.py` |
| Tinh chỉnh, đánh giá, bootstrap theo video | `a55_eval.py` (tune / eval / boot) |
| Bảng tổng hợp (quy tắc 0.3–0.6) | `a55_summarize.py` |
| Test (19 qua) | `test_a55.py` |
| Hàng đợi | `scripts/uav_semcom/run_a55.sh` |

**Test:**
- (a) Tắt hiệu chỉnh thì SC trùng từng bước với A5 (5 tên × 3 regime).
- (b) Nhân quả: đổi fading và ảnh của tương lai trên một bản sao env thì hành động hiện tại không đổi.
- (c) Kiểm tra độ hợp lý: ở K5 có reuse, ρ của phy_mcs_blind < ρ của phy_mcs. Test này tự thêm, không thuộc quy tắc định trước. Phiên bản đầu so với ref và giả định sai rằng ρ ≈ 1 ở ref, nên đã thay.

**Kết quả thô:** `runs/uav_semcom/a55/{tune,final,boot}/`.

## 2. Tinh chỉnh SC trên train (seed 0..29)

- Trên 21 regime, T:SC chỉ có return train cao hơn T:best của A5 ở **3 regime**:
  - K5: +43,30 so với +42,51;
  - phy_mcs: +16,55 so với +16,02;
  - phy_mcs_blind: +16,55 so với +14,98.
- Vì vậy H* = T:SC ở 3 regime này và H* = T:best của A5 ở 18 regime còn lại.
- Ở 3 regime được chọn cho PPO: H* là `pacedB1.5+budget1.25+reuse` (ref), `sc:pacedB3+budget1+reuse` (K5), `sc:pacedB3+budget1.5` (phy_mcs_blind).

## 3. Heuristic tự hiệu chỉnh có giúp không? (T:SC − T:best của A5, seed xác nhận 8000..8099, pool test)

**Không.**
- Trên 21 regime, T:SC không tốt hơn có ý nghĩa ở regime nào: CI theo seed môi trường luôn chứa 0 hoặc nằm dưới 0.
- Kém hơn có ý nghĩa ở 11/21 regime, thường −0,2 đến −1,8; lệch nhiều nhất ở wt0 (−5,3), nơi T:best là một policy đứng yên kiểu "stag".
- Ở K5, nơi SC thắng trên train, mức chênh trên test là +0,76 [−0,21; +1,73]. Trên test-dev là −0,86 [−1,82; +0,10]. Thêm bootstrap theo video thì cả hai đều "không có khác biệt".
- Seed khám phá 7000..7099 cho cùng bức tranh: chỉ K5 test có +1,07 [+0,12; +2,03].

**Cách hiểu:**
- Heuristic A5 đã được tinh chỉnh ngưỡng trên train, nên đã hấp thụ phần lớn sai lệch giữa mô hình tốc độ và thực tế dưới dạng ngưỡng cố định.
- Ước lượng trực tuyến dường như chỉ thêm nhiễu: ở ref, SC cho PSNR thấp hơn 1,4 dB mà tỉ lệ đúng hạn không tăng. Giả thuyết (**chưa kiểm chứng** bằng ablation từng thành phần ρ, ρ_b, s) là hệ số an toàn AIMD tăng sau mỗi lần trễ làm heuristic chọn mode nhẹ hơn mức cần.
- Tức là khoảng cách giữa PPO và heuristic **không** đến từ việc heuristic thiếu hiệu chỉnh trực tuyến các đại lượng đó.

## 4. PPO so với H* (giả thuyết chính H1), ba mức độ bất định

Return tuyệt đối của H* (seed 8000..8099):

| | ref | K5 | phy_mcs_blind |
|---|---|---|---|
| test | 59,24 | 62,66 | 25,65 |
| test-dev | 45,93 | 40,61 | 13,14 |

**PPO mode-only (best-val, A5) − H*:**

| regime | pool | Δ [CI mức 1: seed env] | CI mức 2: + video | CI mức 3: + seed train | kết luận 0.5 |
|---|---|---|---|---|---|
| ref | test | −0,46 [−0,98; +0,06] | [−2,42; +1,61] | [−2,47; +1,79] | không có khác biệt |
| ref | test-dev | +2,63 [+2,06; +3,20] | [−0,01; +4,76] | [+0,06; +4,83] | bền, **sát biên** |
| K5 | test | +1,35 [+0,61; +2,10] | [−2,51; +5,22] | [−3,10; +5,78] | chưa chắc |
| K5 | test-dev | +12,36 [+11,23; +13,50] | [+7,62; +27,37] | [+7,16; +27,22] | **bền** |
| phy_mcs_blind | test | +5,73 [+4,91; +6,56] | [+3,80; +8,15] | [+3,68; +8,32] | **bền** |
| phy_mcs_blind | test-dev | +7,81 [+6,79; +8,83] | [+3,69; +10,05] | [+2,70; +10,76] | **bền** |

**PPO joint − H*:**

| regime | pool | Δ [CI mức 1] | CI mức 3 | kết luận |
|---|---|---|---|---|
| ref | test | −1,24 [−1,87; −0,60] | [−3,59; +1,87] | chưa chắc (−) |
| ref | test-dev | +1,68 [+1,01; +2,35] | [−3,33; +5,65] | chưa chắc (+) |
| K5 | test | +0,39 [−0,45; +1,22] | [−4,86; +7,09] | không có khác biệt |
| K5 | test-dev | +5,99 [+4,82; +7,16] | [+1,51; +14,87] | bền |
| K5 5M | test | +1,84 [+1,00; +2,69] | [−2,50; +8,25] | chưa chắc (+) |
| K5 5M | test-dev | +6,40 [+5,09; +7,71] | [+0,64; +13,32] | bền |
| phy_mcs_blind | test | −4,43 [−5,56; −3,29] | [−7,82; +0,22] | chưa chắc (−) |
| phy_mcs_blind | test-dev | −0,23 [−1,35; +0,88] | [−5,50; +3,82] | không có khác biệt |

**Từng seed huấn luyện, mode-only − H*:**

| | s0 | s1 | s2 |
|---|---|---|---|
| K5 test | +3,13 | **−1,89** | +2,82 |
| K5 test-dev | +11,5 | +12,5 | +13,1 |
| phy_mcs_blind test | +5,6 | +5,8 | +5,8 |
| phy_mcs_blind test-dev | +6,7 | +6,8 | +9,9 |
| ref test-dev | +2,6 | +3,3 | +2,0 |

**Đọc kết quả:**
1. **Ưu thế của PPO mode-only là bền ở phy_mcs_blind (cả hai pool) và ở K5 test-dev.** Đây là hai trường hợp heuristic có mô hình kênh sai lệch (MCS thật nhưng heuristic chỉ biết Shannon) hoặc kênh chia sẻ chật (K = 5). Heuristic tự hiệu chỉnh không thu hẹp được khoảng cách này.
2. **K5 trên pool test không bền.** Δ giảm từ +3,1 (A5, seed 7000, so với T:best của A5) xuống +1,35 (seed mới, so với SC), và CI theo video chứa 0. Một trong 3 model (s1) kém H* trên pool này.
3. **Mức +12,4 ở K5 test-dev là có thật nhưng độ lớn không chắc:** CI theo video [+7,6; +27,4] rất lệch. Test-dev có một chuỗi video chiếm 129/403 khung; bản bootstrap nào lấy chuỗi đó nhiều lần thì Δ lớn hơn nhiều.
4. **ref:**
   - Trên test thì không có khác biệt.
   - Trên test-dev thì quy tắc 0.5 cho "bền", nhưng CI mức 2 chạm 0 (−0,01) còn CI mức 3 vừa đủ dương (+0,06). Với B = 50, hai biên này nằm trong sai số Monte Carlo của chính bootstrap. Tôi xếp kết quả này là **sát biên, không đủ để kết luận PPO hơn ở ref**.
5. **Joint control vẫn không đáng tin**, trừ ở K5 test-dev. Kéo dài lên 5M bước chỉ giúp ở K5 test, và vẫn "chưa chắc".
6. **CI theo seed môi trường (mức 1) hẹp hơn CI có tính video (mức 2/3) 3–10 lần.** Các CI của A5 (chỉ mức 1) vì vậy đã đánh giá thấp độ bất định. Kết luận A5 "+3,1 ở K5 test" cần sửa: ở seed mới và tính video, đó là ưu thế chưa chắc.

## 5. Kết luận A5.5

| Câu hỏi | Trả lời |
|---|---|
| Heuristic tự hiệu chỉnh có bắt kịp PPO không? | Không. SC không tốt hơn heuristic A5 ở regime nào một cách có ý nghĩa; nó chỉ thay H* ở 3 regime với mức chênh trong sai số. |
| Ưu thế PPO mode-only còn không khi tính video và seed huấn luyện? | **Còn, ở phy_mcs_blind (2/2 pool) và K5 test-dev.** Không còn ở K5 test và ref test. ref test-dev sát biên. |
| Joint PPO? | Không bền, trừ K5 test-dev. |

**Giới hạn:**
- 3 model huấn luyện là quá ít để ước lượng phân phối theo seed huấn luyện: mức 3 chỉ lấy mẫu lại 3 giá trị.
- B = 50 cho biên CI có sai số Monte Carlo khoảng ±0,3 ở các Δ cỡ vài đơn vị.
- Test và test-dev là các khung của cùng những video mà A5 đã xem kết quả, nên đây không phải một tập ảnh mới hoàn toàn.

## 6. Kiểm tra lại task fidelity của A5.2 (quy tắc 0.6)

Đo lại toàn bộ (ảnh, mode) bằng cùng detector và codec (bước GPU của A6.0-L1, `runs/uav_semcom/a6l1/fidelity/`). Bảng do `a55_summarize.py fid` tạo.

| pool | cặp (ảnh, mode) của A5.2 | có trong lần đo mới | bytes, n_gt, TP, n_det trùng | max ΔTP | max ΔPSNR | cặp mới (mode sinh ảnh trên mọi ảnh) |
|---|---|---|---|---|---|---|
| val | 7424 | 7424 | 7424/7424 | 0 | 0 | 1344 |
| test-dev | 5239 | 5239 | 5239/5239 | 0 | 0 | 1209 |

- AP50 COCO của mọi mode trên tập con `all` (và `test`, `in_a4` ở val) trùng với A5.2 tới mọi chữ số đã lưu.
- Tập con `gen_subset` khác nhau vì định nghĩa khác: ở A5.2 là 100 ảnh, ở lần đo mới là toàn bộ ảnh. Đây không phải sai lệch đo.
- Test đơn vị cho phép ghép (class-aware, IoU 0,5, vùng ignore) và test "AP50 theo policy tái tạo COCOeval" đều qua (`test_a6l1.py`, 7/7).

**Kết luận:** phép đo task fidelity của A5.2 tái lập chính xác (tất định), nên kết quả detector-recall trong A5 không bị ảnh hưởng.
