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
   - chạy mọi policy trên 20 seed môi trường riêng của bản (8100 + 20b + i), lấy Δ trung bình của bản.
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
