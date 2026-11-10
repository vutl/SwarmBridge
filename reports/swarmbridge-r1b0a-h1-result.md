# SwarmBridge R1b-0a: kết quả cổng H1

Ngày 11/11/2026. Protocol: [swarmbridge-r1b0a-protocol.md](swarmbridge-r1b0a-protocol.md). G7 được chấm **một lần**, với tham số đã commit trước (`6d2247f1`).

## Kết luận

**H1 = FAIL (không có dư địa).** Theo luật đã đăng ký: D − A1 < +2 ở cả ba mức tiêu chí (1×, 2×, 4× R_F), và C1 không thắng ở mức nào.

Hai điều cần đọc cùng kết luận này:
1. **Ở 1×R_F, C1 hơn A1 đều nhưng nhỏ:** +1,24 điểm F1 (CI bootstrap theo nguồn [0,75; 1,74]; 6/7 nguồn dương; 5/5 seed dương). Mức này dưới ngưỡng +2 đã đăng ký, nên **không** thắng.
2. **D (biết đúng trạng thái BS) không phải chặn trên.** D thua A1 2,3 điểm ở 1×R_F. Nhãn "không có dư địa" vì vậy chỉ có nghĩa: với menu hành động heuristic này, biết trạng thái BS không giúp tới +2. Nó không chứng minh phản hồi vô ích trong mọi thiết kế.

Theo protocol (mục 8, "Sau cổng"): **dừng mở rộng phản hồi ngữ nghĩa** trên nhánh này. Không chạy H2/H3. Không đổi reward, ngưỡng hay thêm PPO để cứu kết quả.

## 1. Thí nghiệm hỏi gì (luồng)

1. Mỗi tick (100 ms), UAV chạy detector nhỏ (YOLO11n) và tracker cục bộ.
2. UAV chọn một trong ba hành động:
   - gửi **FRAME** (ảnh AVIF);
   - gửi **TOK** (vài box lệch);
   - **SKIP**.
3. BS chạy detector lớn (YOLO11s) trên FRAME, cập nhật tracker. **Bản đồ tracker ở BS là thứ được chấm** (object-time F1 theo lớp).
4. Các điều kiện chỉ khác nhau ở việc UAV biết gì về bản đồ BS khi quyết định:

| Điều kiện | UAV biết gì về bản đồ BS |
|---|---|
| A1 | Tự đoán, cập nhật khi nhận ACK |
| B / B+ | Gợi ý box (B+ có thêm lớp và độ tin cậy) |
| C1 | Bản sao trạng thái có version |
| D | Biết đúng, miễn phí |

5. H1 hỏi: phản hồi trạng thái có tính phí (C1) có giúp bản đồ BS tốt hơn ACK đơn thuần (A1) không?

## 2. Bảng chính (G7, gộp 7 nguồn × 5 seed, F1 theo lớp)

| Mức R_ul | A1 | B | B+ | **C1** | D | A0 | C0 | C1 − A1 [CI95] | D − A1 | C1 − max(B, B+) | nguồn dương | thắng |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0,5× (chỉ báo) | 19,72 | 12,61 | 13,55 | **20,70** | 17,08 | 16,51 | 20,04 | +0,98 [0,60; 1,49] | −2,64 | +7,15 (B+) | 7/7 | – |
| 1× | 18,75 | 12,61 | 13,36 | **19,99** | 16,48 | 15,40 | 19,14 | +1,24 [0,75; 1,74] | −2,26 | +6,63 (B+) | 6/7 | không |
| 2× | 22,12 | 22,09 | 22,13 | **22,11** | 22,14 | 22,11 | 22,11 | −0,01 [−0,10; 0,04] | +0,02 | −0,02 (B+) | 1/7 | không |
| 4× | 22,46 | 22,45 | 22,50 | **22,45** | 22,51 | 22,48 | 22,46 | −0,00 [−0,01; 0,00] | +0,05 | −0,05 (B+) | 0/7 | không |

- Baseline gợi ý (max(B, B+)) được chọn theo F1 trên TUNE: B+ ở mọi mức.
- CI chỉ để mô tả.
- **Ổn định qua seed:** ở mọi mức, ΔF1 gộp theo từng seed đều < +2, nên 0/5 seed cho kết luận "thắng". Ở 1×: +0,69 / +0,89 / +1,72 / +1,28 / +1,66.

**F1 class-agnostic và IDF1 (theo lớp)** cho cùng thứ tự. Ví dụ ở 1×:

| | A1 | C1 | D | B+ |
|---|---|---|---|---|
| F1 class-agnostic | 21,93 | 23,23 | 19,29 | 16,03 |
| IDF1 | 17,7 | 18,2 | 16,7 | 10,1 |

### Theo nguồn, 1×R_F (F1 trung bình 5 seed)

| Nguồn | A1 | B | B+ | C1 | D | A0 | C0 |
|---|---|---|---|---|---|---|---|
| 020_00406 | 33,1 | 32,8 | 32,7 | 32,5 | 33,4 | 32,6 | 32,6 |
| 138_00000 | 19,4 | 18,6 | 19,1 | 21,2 | 18,2 | 17,5 | 20,2 |
| 244_01440 | 29,6 | 14,0 | 15,9 | 30,2 | 26,9 | 23,4 | 29,5 |
| 278_00001 | 12,7 | 8,7 | 9,2 | 13,7 | 11,1 | 10,4 | 13,1 |
| 289_00001 | 30,0 | 15,1 | 16,0 | 31,2 | 28,3 | 23,4 | 30,8 |
| 295_02300 | 12,1 | 9,4 | 9,8 | 13,9 | 9,3 | 9,3 | 12,6 |
| 316_01288 | 30,6 | 17,7 | 19,1 | 30,7 | 28,3 | 30,7 | 30,1 |

Ở 2× và 4×, bảy điều kiện gần như trùng nhau ở từng nguồn: chênh ≤ 0,7 điểm. Chi tiết trong `runs/uav_semcom/r1b/gate_v1_tables.json`.

## 3. Vì sao không có dư địa (chẩn đoán)

### 3.1. Từ 2×R_F trở lên: giới hạn là CPU, không phải đường lên

- R_F được khoá theo định nghĩa đã đăng ký: tốc độ tối thiểu để một FRAME trung vị kịp deadline khi hàng đợi rỗng, khoảng 151 ms phát sóng.
- Một FRAME tốn 29 + 2 + 104 ms tính toán trên UAV. Với mô hình worker tuần tự, tick kế tiếp bị bỏ, nên UAV chỉ gửi được tối đa 1 FRAME mỗi 200 ms.
- Ở 2× và 4×, mọi điều kiện đều gửi FRAME ở mọi cơ hội: 98–100% quyết định là FRAME, 50% tick bị rơi vì CPU.
- Không còn quyết định nào để phản hồi tác động, nên mọi điều kiện trùng nhau.
- Đây là lỗi thiết kế của protocol: tôi chọn mức R_ul theo thời gian phát một FRAME, không kiểm tra tốc độ FRAME tối đa mà CPU cho phép. Lỗi này lộ ra khi chẩn đoán trên TUNE, **trước** khi chấm G7 (xem mục 6). Tôi không sửa, vì sửa sẽ là đổi thiết kế sau khi thấy kết quả tinh chỉnh.

### 3.2. Ở 1× và 0,5×: luật kích hoạt không bao giờ chọn SKIP

| Mức | FRAME | TOK | SKIP |
|---|---|---|---|
| 1× | 10% | 90% | 0% |
| 0,5× | 0% | ≈ 100% | 0% |

- Tracker cục bộ dùng detection conf ≥ 0,05, nên luôn có "track mới". Điểm g luôn vượt ngưỡng TOK, và UAV gửi TOK ở mọi tick.
- Phản hồi chỉ còn ảnh hưởng tới **nội dung** TOK (box nào bị coi là lệch), không ảnh hưởng tới việc có gửi hay không.

### 3.3. Biết đúng trạng thái BS (D) không giúp được

Trên TUNE, D thua A1 khoảng 2 điểm ở 0,5× và 1× **với mọi bộ tham số** trong lưới. Kiểm tra từng clip (`scripts/uav_semcom/r1b_tune_diag_D.py`):
- số FRAME, TOK, SKIP của D và A1 gần như giống nhau;
- 15/48 clip cho kết quả trùng khít;
- chênh lệch đến từ nội dung TOK. Khi đối chiếu với bản đồ BS thật, vốn đầy track sai, D gửi các box làm FP ở BS tăng thêm (ví dụ clip 243: FP +8.199, TP −442).

Kết luận: chất lượng bản đồ BS **không đơn điệu theo độ chính xác niềm tin** của UAV trong menu này. Đây đúng là "giới hạn menu" đã khai báo ở protocol; không có lỗi code.

### 3.4. Bản đồ BS có precision rất thấp

| Mức | Precision | Recall |
|---|---|---|
| 1× | ≈ 14% | ≈ 30% |
| 2× | ≈ 21% | ≈ 23% |

- Track xác nhận từ detection conf thấp chiếm phần lớn.
- Chẩn đoán track ma:
  - ở 1× có khoảng 6.700–8.500 giây-track ma mỗi phút (cộng dồn trên nhiều track);
  - 75% thời gian track ma không có detection nào của UAV ở gần.
- Điều này áp dụng chung cho mọi điều kiện. Nó làm F1 kém nhạy, nhưng không đảo thứ tự.

## 4. Gửi đi có tới nơi không (FRAME / TOK)

Tỉ lệ tới BS kịp deadline và nguyên vẹn (kịch bản chính, p_ul = 1% mỗi gói, mất kiểu tất cả hoặc không):

| Mức | FRAME tới được | TOK tới được |
|---|---|---|
| 1× | 0,68–0,71 | 0,99 |
| 2×, 4× | 0,72–0,73 | ≈ 1,00 (rất ít TOK) |

- FRAME khoảng 31 gói, nên mất trọn khoảng 27–30% FRAME.
- Mô hình **không có FEC hay truyền lại**; không khái quát cho mọi mạng.

## 5. Độ nhạy (chỉ báo; không đổi kết luận)

Tất cả trên G7, cùng tham số đã khoá:

| Biến thể | 1×: C1 − A1 | 1×: D − A1 | 2×: C1 − A1 | 4×: C1 − A1 | Ghi chú |
|---|---|---|---|---|---|
| Chính | +1,24 | −2,26 | −0,01 | −0,00 | |
| Ngưỡng chung của A1 | +1,25 | −2,26 | −0,01 | −0,00 | kết quả không do tinh chỉnh riêng |
| Tính toán pipelined | +1,26 | −2,20 | −0,06 | +0,02 | F1 tuyệt đối 2× tăng lên 30,8 |
| DL_cap 16 kbit/s | **+4,18** | −2,27 | −0,02 | −0,00 | xem ghi chú bên dưới |
| p_ul = 0 | +1,24 | −2,23 | −0,00 | −0,00 | FRAME tới 88% (1×), 100% (2×) |
| p_ul = 5% | +1,13 | −2,27 | +0,06 | −0,00 | FRAME tới 22–28%; F1 ở 2× chỉ còn 7,9 |

**Ghi chú DL16:**
- Downlink nhỏ hơn lại làm C1 **tốt hơn**: 22,93 so với 19,99 với DL 64 kbit/s.
- Đây là thêm một dấu hiệu menu không đơn điệu; giả thuyết (chưa kiểm) là bản sao cũ hơn thay đổi nội dung TOK. Đó **không** phải bằng chứng phản hồi có ích.
- Đây là phân tích phụ đã khai báo, không dùng để đổi kết luận.

**AUX (3 clip val):** cùng hình ảnh. Ở 1×: A1 20,57, C1 21,93, D 19,45, B+ 16,61. Ở 2×/4×, các điều kiện cách nhau ≤ 1,2 điểm (3 clip, nhiễu lớn hơn G7).

**A0 / C0:**
- A0 (feed lúc gửi, không chờ ACK) kém A1 3,4 điểm ở 1×: ACK có giá trị so với đoán mù.
- C0 (delta không version) kém C1 0,85 điểm ở 1×.
- C1 có 793 tick lệch đồng bộ và 742 snapshot (G7, 5 seed); C0 có 270 snapshot.

## 6. Ghi chép thủ tục (trung thực)

- **Tham số đã khoá trước khi chấm:** `tune_v1.json` (commit `6d2247f1`). Tinh chỉnh dùng lưới 64 điểm và seed 1000, theo amendment v1.1.
  - Các trường `tune_seeds` và `grid` trong `lock_v1.json` vẫn ghi bản cũ (2 seed, 256 điểm). Code dùng amendment v1.1. File lock không được sửa lại, để giữ đúng thứ đã commit.
- **Đã thấy trên TUNE trước khi chấm G7:**
  - D < A1 ở 0,5× và 1×;
  - các điều kiện trùng nhau ở 2× và 4×;
  - nguyên nhân CPU (mục 3).

  Nghĩa là tôi đã đoán trước H1 có khả năng FAIL. Tôi không đổi gì sau khi thấy những điều này.
- **Lệnh chấm G7 được gọi ba lần.** Hai lần đầu dừng vì lỗi shell khi chuyển đường dẫn từ Git Bash sang WSL:
  - lần 1 đọc `/lock_v1.json` rỗng và thoát ngay khi đọc lock;
  - lần 2 không tìm thấy script.

  Cả hai không chạy job mô phỏng nào và không ghi file kết quả. Lần 3 là lần chấm duy nhất: 4.025 lượt, ghi `gate_v1.json`.
- Chưa chạy H2/H3, PPO/SAC, R2, VID test-dev hay x264.

## 7. Đặt cạnh tiền lệ nội bộ

Kết quả cùng chiều với hai lần trước, nơi phản hồi có tính phí không tạo lợi ích rõ:
- Town3 `temporal_select` (`91179ca1`): bản sao latent, có mất ACK;
- UAV3D v1.1 (`70486c1b`): phản hồi có tính phí cho +0,005–0,008 AMOTA, cỡ nhiễu seed.

Kiểm toán ASR (Codex, chỉ đọc) xác nhận ASR không có cơ chế tương đương C1 (thiết kế v3 mục 10.1).

Lưu ý đã cam kết trước: kể cả nếu H1 PASS, đó cũng **không** phải novelty. Ở đây H1 FAIL nên câu hỏi novelty không đặt ra.

## 8. Đọc kết quả thế nào

**Kết luận được phép:**
- Trong thiết lập này (VisDrone-VID, YOLO11n/s, CPU 6 lõi, AVIF q35, mất gói tất cả hoặc không, menu FRAME / TOK / SKIP heuristic), phản hồi trạng thái có tính phí không nâng bản đồ BS đủ ngưỡng +2.
- Từ 2×R_F trở lên, nó không có tác dụng gì, vì CPU là nút thắt.

**Không được kết luận:**
- "Phản hồi vô ích nói chung". Oracle D không phải chặn trên của menu này, và hai lớp hiệu ứng làm lu mờ tác động của thông tin:
  - luật kích hoạt luôn bật;
  - bản đồ BS precision thấp.
- "C1 có ích +1,24": dưới ngưỡng đăng ký; DL16 cho thấy hiệu ứng có thể đến từ tương tác với nội dung TOK.

**Việc tiếp theo theo protocol:** dừng nhánh phản hồi ngữ nghĩa. Mọi thiết kế lại (đặt mức R_ul theo nút thắt CPU, ngưỡng detection cho tracker, chính sách có SKIP thật) sẽ là một **thí nghiệm mới** với protocol mới và split mới. Đó không phải là chạy lại H1.

## Tệp

| Tệp | Nội dung |
|---|---|
| `runs/uav_semcom/r1b/gate_v1.json` | Kết luận, thống kê theo mức |
| `runs/uav_semcom/r1b/gate_v1_rows.json` | 4.025 dòng: mỗi clip × seed × điều kiện × mức × biến thể |
| `runs/uav_semcom/r1b/gate_v1_tables.json` | Bảng mô tả, tạo bằng `scripts/uav_semcom/r1b_gate_tables.py` |
| `runs/uav_semcom/r1b/tune_v1.json`, `tune_v1_grid.json` | Tham số đã khoá và toàn bộ lưới TUNE |
| `scripts/uav_semcom/r1b_tune_diag_D.py` | Chẩn đoán D so với A1 trên TUNE |
