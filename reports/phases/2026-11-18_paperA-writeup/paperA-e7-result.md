# Paper A E7: lớp MPC hiệu chỉnh trên incumbent từ ba họ RL, kết quả (23/11)

Pre-registration: [paperA-e7-rl-families-prereg.md](paperA-e7-rl-families-prereg.md) (61f2c489, phụ lục 1934380d, ghi
chú vận hành d1596b79). Train chạy lại từ đầu lúc 12:26 ngày 22/11 và xong lúc 03:14 ngày 23/11. Đánh giá xong khoảng
04:00 cùng ngày, EXIT 0, đủ 30 kết quả × 100 seed ở mỗi pool.

## 1. Cách đọc

**Bài toán:** mỗi slot, bộ lập lịch JA cố định chia kênh cho các UAV. Incumbent X chọn mode nén cho ảnh kế tiếp. Lớp
MPC có chọn lọc có thể thay mode của X:
- `+cal`: MPC dùng mô hình tốc độ hiệu chỉnh từ ACK (phương pháp của bài);
- `+nom`: cùng MPC nhưng dùng mô hình tốc độ danh nghĩa.

**Incumbent:**
- H\* = fixed mbt2018-mean q3;
- PPO-LA = 3 checkpoint của A-X9;
- QR-DQN và Discrete SAC: mới train trong E7, mỗi họ 3 seed.

**Thống kê:**
- Nhánh học được lấy trung bình 3 seed train cho mỗi seed môi trường, rồi tính CI 95% ghép cặp trên 100 seed môi trường.
  CI này chỉ đo biến động theo môi trường. Độ phân tán theo seed train báo riêng ở các dòng "seed s".
- \* = CI không chứa 0.

**Pool:**
- Test: seed 13000–13099, dùng chung với E4 N5x2.
- Test-dev: seed 14000–14099, nội dung held-out. Pool này đã dùng ở các vòng trước nên không phải dữ liệu hoàn toàn
  chưa chạm tới. Chỉ báo như bản lặp lại (replication).

## 2. Chọn siêu tham số và train (chỉ dùng validation)

| Họ | Thiết lập T1 (seed 0, val tốt nhất) | Chọn | Val tốt nhất của 3 seed (bước) |
|---|---|---|---|
| QR-DQN | lr 1e-4: −30,0; lr 5e-4: −53,7 | lr 1e-4 | −30,0 (1,8M); −28,6 (1,4M); −55,3 (1,2M) |
| Discrete SAC | r 0,3: −56,3; r 0,6: −73,4 | r 0,3 | −56,3 (1,6M); −36,0 (2,0M); −46,7 (1,8M) |
| PPO-LA (A-X9, không sweep) | | | −7,9 (0,8M); −3,5 (0,4M); −5,2 (0,4M) |

**Đường học validation** (mỗi 200k quyết định; đầy đủ trong `summary_e7.log`):
- QR-DQN và Discrete SAC dao động mạnh, không đơn điệu.
- Checkpoint được chọn nằm muộn (1,2M–2,0M). Với Discrete SAC seed 1, checkpoint chọn chính là checkpoint cuối.
- PPO-LA đạt đỉnh sớm (0,4M–0,8M) rồi giảm dần.
- Với ngân sách 2M quyết định và các thiết lập đã thử, hai họ mới có validation thấp hơn PPO-LA rõ rệt.

## 3. RQ-B (kiểm định đăng ký, quyết định câu chữ của claim)

B1 = X + cal − X. B2 = X + cal − X + nom. Đơn vị: return.

| Incumbent | Test B1 | Test B2 | Test-dev B1 | Test-dev B2 |
|---|---|---|---|---|
| (tham chiếu) H\* | +2,3 [−1,0; +5,5] | **+20,4*** | **+16,4*** | **+6,5*** |
| PPO-LA | **+6,2*** [+4,9; +7,6] | **+20,2*** [+17,5; +22,9] | **+10,7*** | **+10,8*** |
| QR-DQN | **+23,9*** [+22,0; +25,8] | **+10,6*** [+7,3; +14,0] | **+24,8*** | **+4,8*** |
| Discrete SAC | **+29,2*** [+27,3; +31,1] | **+9,0*** [+6,1; +11,9] | **+35,1*** | +1,2 [−1,4; +3,8] |
| **Số CI > 0** | | **6/6** | | **5/6** |

**Theo quy tắc đã đăng ký** (cả sáu CI trên pool test phải nằm trên 0): **ĐẠT**.
- Câu được phép viết: "với ngân sách train và các thiết lập đã đánh giá, lớp MPC hiệu chỉnh cải thiện incumbent từ ba
  họ RL (actor-critic on-policy, value-based phân phối, actor-critic off-policy maximum-entropy), cả so với incumbent
  gốc lẫn so với cùng MPC dùng mô hình danh nghĩa".
- **Bản lặp lại trên test-dev** đạt 5/6. Discrete SAC B2 = +1,2, CI chứa 0, phải ghi rõ.

**Theo từng seed train** (B1 / B2, CI theo seed môi trường):

| Incumbent | Seed | Test B1 | Test B2 | Test-dev B1 | Test-dev B2 |
|---|---|---|---|---|---|
| PPO-LA | 0 | +7,5* | +21,0* | +9,8* | +12,9* |
| | 1 | +5,5* | +20,6* | +12,6* | +9,8* |
| | 2 | +5,7* | +19,1* | +9,8* | +9,6* |
| QR-DQN | 0 | +16,0* | +13,8* | +18,6* | +5,5* |
| | 1 | +20,3* | +17,5* | +20,9* | +12,1* |
| | 2 | +35,5* | **+0,7 [−3,6; +4,9]** | +34,8* | **−3,3 [−7,5; +1,0]** |
| Discrete SAC | 0 | +35,8* | +8,5* | +37,9* | **+0,5 [−4,1; +5,0]** |
| | 1 | +20,1* | +13,5* | +28,8* | +5,7* |
| | 2 | +31,7* | +5,1* | +38,4* | **−2,6 [−7,0; +1,8]** |

- B1 dương có ý nghĩa ở cả 18 cặp (seed × pool × họ).
- B2 không có ý nghĩa ở 4/18:
  - QR-DQN seed 2 ở cả hai pool;
  - Discrete SAC seed 0 và seed 2 trên test-dev.
- Đó đều là những seed có incumbent yếu nhất, nơi bản thân MPC danh nghĩa đã kéo return lên nhiều.

## 4. Policy yếu được "cứu" không làm hệ thống cuối tốt hơn (phụ lục pre-registration)

Return tuyệt đối (trung bình; trong ngoặc là từng seed train):

| Incumbent | Pool | X | X + cal | X + nom |
|---|---|---|---|---|
| H\* | test | −34,0 | −31,8 | −52,2 |
| PPO-LA | test | −38,4 | −32,2 (−30,8 / −33,3 / −32,5) | −52,4 |
| QR-DQN | test | −69,2 (−58,8 / −59,0 / −90,0) | −45,3 | −56,0 |
| Discrete SAC | test | −76,9 (−84,7 / −64,5 / −81,4) | −47,6 | −56,7 |
| H\* | test-dev | −67,2 | −50,8 | −57,3 |
| PPO-LA | test-dev | −53,5 (−50,1 / −57,2 / −53,1) | −42,8 | −53,5 |
| QR-DQN | test-dev | −76,2 (−70,6 / −63,6 / −94,4) | −51,4 | −56,2 |
| Discrete SAC | test-dev | −90,9 (−94,7 / −81,5 / −96,5) | −55,9 | −57,1 |

- B1 lớn ở QR-DQN và Discrete SAC (+24 đến +35) phần lớn là MPC sửa lỗi của incumbent yếu.
- Hệ thống cuối vẫn thua H\* + cal trên test:
  - QR-DQN + cal − H\* + cal = **−13,6*** [−17,3; −9,9];
  - Discrete SAC + cal − H\* + cal = **−15,9*** [−19,4; −12,4].
- Trên test-dev:
  - QR-DQN + cal − H\* + cal = −0,6 (CI chứa 0);
  - Discrete SAC + cal − H\* + cal = −5,1\*.
- MPC danh nghĩa đưa mọi incumbent về gần cùng một mức (−52 đến −57 trên test, −53 đến −57 trên test-dev). Tức là MPC
  danh nghĩa ghi đè lựa chọn của incumbent rất thường xuyên. Vì vậy B2 nhỏ hơn khi incumbent càng yếu.

## 5. RQ-A (mô tả, không suy rộng ra ngoài các agent đã train)

| Đối chiếu | Test | Test-dev |
|---|---|---|
| PPO-LA − H\* | **−4,4*** [−8,0; −0,8] | **+13,7*** [+9,1; +18,2] |
| PPO-LA + cal − H\* + cal | −0,4 [−3,6; +2,7] | **+8,1*** [+4,0; +12,2] |
| QR-DQN − H\* | **−35,2*** | **−9,0*** |
| QR-DQN + cal − H\* + cal | **−13,6*** | −0,6 [−5,4; +4,2] |
| QR-DQN − PPO-LA | **−30,9*** | **−22,7*** |
| Discrete SAC − H\* | **−42,9*** | **−23,7*** |
| Discrete SAC + cal − H\* + cal | **−15,9*** | **−5,1*** |
| Discrete SAC − PPO-LA | **−38,5*** | **−37,4*** |

**Theo quy tắc câu chữ đã đăng ký:**
- Không họ mới nào thắng H\* + MPC với CI > 0 trên cả hai pool. Câu so sánh học được với heuristic trong bài **không
  thay đổi** vì E7.
- Hai họ mới thua PPO-LA rõ rệt trên cả hai pool, với ngân sách và thiết lập đã đánh giá. Bài không được viết RL nói
  chung là kém hay vượt trội.

**PPO-LA so với H\* đổi dấu giữa hai pool:**
- Trên test, PPO-LA − H\* = −4,4\*, và PPO-LA + cal − H\* + cal = −0,4 (CI chứa 0). Kết quả này khớp với A-X9, nơi chênh
  là −0,14 (CI chứa 0) trên seed 8300–8399.
- Trên test-dev, PPO-LA thắng H\* (+13,7\*), và PPO-LA + cal thắng H\* + cal (+8,1\*). Ở pool này, H\* có tỉ lệ rớt ảnh
  13,3% (6,8% trên test).
- Bài phải báo **cả hai pool**. Không được chọn pool có lợi hơn. Thứ hạng giữa H\* và PPO-LA phụ thuộc nội dung ảnh.

**Theo từng seed train:**
- QR-DQN − H\* trên test-dev: −3,4 / +3,6 (cả hai CI chứa 0) / −27,3\*.
- QR-DQN + cal − H\* + cal trên test-dev: −1,2 / **+8,2\*** / −8,8\*. Một seed QR-DQN có cộng MPC thắng H\* + cal trên
  test-dev. Seed đó vẫn thua trên test (−7,0\*), nên không phải phát hiện dương theo quy tắc.

## 6. Recall (Qmin-compliant) và các chỉ số khác

| Incumbent | Recall test: X → + cal → + nom | Recall test-dev: X → + cal → + nom | B1 recall test | B1 recall test-dev |
|---|---|---|---|---|
| H\* | 52,1 → 51,0 → 49,1 | 39,8 → 39,2 → 38,4 | **−1,2*** | **−0,6*** |
| PPO-LA | 46,9 → 47,9 → 47,8 | 36,4 → 36,9 → 37,7 | **+0,9*** | **+0,6*** |
| QR-DQN | 39,1 → 43,6 → 47,1 | 32,0 → 34,5 → 37,2 | **+4,5*** | **+2,6*** |
| Discrete SAC | 36,7 → 42,8 → 47,2 | 29,5 → 33,6 → 37,3 | **+6,1*** | **+4,1*** |

**Return và recall lệch nhau (vẫn là kết quả chính của bài):**
- Với QR-DQN và Discrete SAC, MPC danh nghĩa có return **thấp hơn** MPC hiệu chỉnh nhưng recall **cao hơn** (47 so với
  43 trên test).
- Với H\*, MPC hiệu chỉnh tăng return nhưng giảm recall, giống E4.
- Recall của mọi nhánh học được vẫn thấp hơn H\* cùng loại. Ví dụ trên test, PPO-LA + cal − H\* + cal = −3,1\*.

**Thời gian quyết định:**
- Chỉ incumbent, p99: PPO-LA 3,2 ms, QR-DQN 0,43 ms, Discrete SAC 0,36 ms. PPO dùng sb3 MaskablePPO; hai họ mới là MLP
  PyTorch tự viết, nên đây không phải so sánh độ phức tạp thuật toán.
- Quyết định tích hợp với MPC, p99: 42–47 ms.
- Số slot vượt 50 ms ≤ 0,1% ở mọi nhánh.

## 7. Kiểm tra ghép cặp với E4 và khả năng tái lập

Ba nhánh H\* của E7 trên test được đối chiếu với E4 N5x2, cùng seed và cùng PHY:

| Nhánh | E4 | E7 | Chênh (CI) | Số seed khác nhau |
|---|---|---|---|---|
| H\* | −34,01 | −34,01 | 0 | 0/100 |
| H\* + cal | −31,35 | −31,75 | −0,40 [−1,29; +0,49] | 16/100 |
| H\* + nom | −50,86 | −52,19 | −1,34 [−3,16; +0,49] | 21/100 |

- **Nguyên nhân:** RNG không phải nguyên nhân, vì MPC được tạo lại cho mỗi episode với seed cố định. Ngân sách 50 ms tính
  theo đồng hồ thật: phần lập kế hoạch dừng khi ước lượng thời gian vượt ngân sách. Khi tải CPU khác nhau, một số slot bị
  cắt ở chỗ khác, và episode rẽ nhánh từ đó.
- **Hệ quả:** nhánh có MPC không tái lập được từng bit giữa các lần chạy.
  - Chênh trung bình giữa hai lần chạy nhỏ, CI chứa 0.
  - B1 của H\* là +2,66 ở E4 và +2,26 ở E7. Tương quan của B1 theo từng seed giữa hai lần chạy là 0,96.
  - Độ lệch theo từng episode thì không nhỏ: SD 4,5 cho + cal, 9,2 cho + nom.
- **Bài phải ghi:** kết quả có MPC phụ thuộc thời gian chạy thật, và lần chạy lại cho chênh trung bình trong khoảng
  ±1,3.

## 8. Tóm tắt cho phần viết bài

- **Claim đa họ RL ĐẠT** theo quy tắc: test 6/6. Test-dev lặp lại 5/6; Discrete SAC B2 chứa 0.
- B1 đạt ở mọi seed train và mọi pool. B2 không đạt ở 4/18 cặp seed × pool × họ, đều ở các seed incumbent yếu.
- MPC không biến policy yếu thành hệ thống tốt nhất: QR-DQN + cal và Discrete SAC + cal vẫn thua H\* + cal trên test.
- Lợi ích so với H\* không có MPC (B1 của H\*) thay đổi theo pool:
  - E4/E7 test: +2,7 / +2,3, CI chứa 0;
  - E1: +8,2;
  - E7 test-dev: +16,4\*.
- So sánh học được với heuristic không đổi: PPO-LA + cal ≈ H\* + cal trên test, thắng trên test-dev; hai họ mới thua.
- Return và recall lệch nhau thêm ở một dạng mới: MPC danh nghĩa có return thấp hơn nhưng recall cao hơn khi incumbent
  yếu.
- Kết quả có MPC không tái lập từng bit (ngân sách theo đồng hồ thật). Chênh giữa hai lần chạy ≤ 1,3 trung bình.
- **Theo stop rule:** phạm vi Paper A đóng băng từ đây. Không thêm vòng thuật toán nào.

**File:**
- `runs/uav_semcom/paperA_r2/e7_test.json`, `e7_testdev.json` (kèm `.deliv.json.gz`), `e7.log`;
- `summary_e7.json` / `.log`; script `paperA_e7_summary.py`;
- `runs/uav_semcom/paperA_rl/selection.json`, `ckpts.json`, `train/*.train.json`, `logs/`, `queue.log`.

Trọng số không có trong git (`/home/uservutl/work/e7`).
