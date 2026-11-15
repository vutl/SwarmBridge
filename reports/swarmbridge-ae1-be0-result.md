# A-E1 và B-E0: kết quả prototype đầu tiên

Ngày 15/11/2026, theo phản biện #27 (GO cho A-E1 và B-E0). Giai đoạn khám phá: báo cái gì chạy, tốn bao nhiêu, thấy gì, vướng gì.

## A-E1: chi phí ra quyết định của PPO mode-only, H* và SC (Paper A)

**Cách đo:**
- Dùng đúng đường code của A5.5 (`a55_eval.job`). PPO mode-only dùng phần slot của heuristic A5 T:best, checkpoint best-val của A5.
- Seed 8000..8019, pool test, ba regime ref / K5 / phy_mcs_blind.
- Một luồng torch, 3 tiến trình song song. Bỏ 50 quyết định đầu (warm-up), trước đó chạy một episode warm-up riêng.
- Phần cứng: Intel i5-14400F (WSL2), Python 3.12.
- Code: [a9_latency.py](../swarmbridge/experiments/uav_semcom/a9_latency.py). Kết quả: `runs/uav_semcom/a9/ae1_latency.json`.

**Kiểm tra:** return của cả 18 tổ hợp (regime × chính sách) **trùng tuyệt đối** với file xác nhận A5.5 trên cùng seed (sai lệch lớn nhất 0,0). Vòng đo không làm thay đổi hành vi.

| Regime | Chính sách | Return (20 seed) | Quyết định mode: TB / p95 / p99 (µs) | Forward thuần (µs) | Quyết định slot TB / p95 (µs) |
|---|---|---|---|---|---|
| ref | PPO mode (3 model) | 58,2–58,5 | 1.167–1.177 / 1.827–1.864 / ~2.040 | 916–927 | 146–148 / ~290 (heuristic T:best) |
| ref | H* (= T:best) | 58,12 | 63 / 104 / 133 | – | 114 / 228 |
| ref | SC | 57,38 | 39 / 69 / 82 | – | 55 / 95 |
| K5 | PPO mode (3 model) | 59,5–64,5 | 1.648–1.699 / 2.630–2.684 / ~2.980 | 1.347–1.392 | 210–219 / ~430 |
| K5 | H* (= SC) | 62,20 | 52 / 94 / 136 | – | 228 / 421 |
| phy_mcs_blind | PPO mode (3 model) | 30,0–30,5 | 1.174–1.219 / 1.868–1.925 / ~2.170 | 919–957 | 155–160 / ~300 |
| phy_mcs_blind | H* (= SC) | 26,14 | 46 / 80 / 93 | – | 129 / 250 |

- **Số quyết định mỗi episode:** 3.070 quyết định mode (K = 3) hoặc 5.150 (K = 5); khoảng 1.500 quyết định slot.
- **Kích thước mạng:** MLP [256, 256]; 288k tham số (quan sát 271 chiều) và 377k tham số (K5, 429 chiều).

**Đọc kết quả:**
- **PPO chậm hơn heuristic khoảng 20–35 lần mỗi quyết định mode:** 1,2–1,7 ms so với 40–60 µs.
  - Khoảng 80 % thời gian là forward của mạng.
  - Phần còn lại là dựng quan sát, mask và overhead của SB3.
- **So với ràng buộc thời gian của hệ thống, cả hai đều không đáng kể:**
  - slot dài 1 s, hạn chót ảnh 4–12 s;
  - encode ảnh tốn hàng chục ms đến hàng giây (bảng mode A5).
  - p99 của PPO khoảng 3 ms.
- **Hệ quả cho Paper A:** đúng như dự đoán ở phản biện #26, so với heuristic thì PPO **không** có lợi thế chi phí. Luận điểm chi phí chỉ có nghĩa khi so với một bộ tối ưu online đắt (A-E2). Ưu thế return của PPO ở phy_mcs_blind (+3,9 đến +4,4 trên 20 seed này so với H*) có được với chi phí 1–2 ms mỗi quyết định.

**Thiết kế A-E2 / A-E3 (sửa theo phản biện #27), chưa chạy:**
- **So sánh chính:** MPC **chỉ chọn mode**, cùng slot policy với PPO mode-only (phần slot của T:best), cùng quan sát (mode_info `predicted`), cùng ràng buộc và hạn chót. MPC không biết fading tương lai: nó dùng kỳ vọng hoặc lấy mẫu từ phân phối kênh tại thời điểm quyết định.
- **Hai biến thể mô hình tốc độ:** đúng (MCS) và sai (Shannon), trên phy_mcs_blind.
- **So sánh phụ** (báo riêng): MPC joint (mode + kênh + công suất) so với PPO joint.

## B-E0: pipeline truyền token có dự đoán (Paper B)

**Luồng xử lý:**
1. Khung VisDrone-VID, thu về 308 × 546.
2. DINOv2-S đóng băng (fp16) cho 22 × 39 = 858 token, mỗi token 384 chiều.
3. Bộ dự đoán "giữ nguyên" chạy y hệt ở hai đầu.
4. Chọn k patch có sai số dự đoán lớn nhất.
5. Lượng tử từng token: scale fp16 + 384 số q bit.
6. Ghi bitstream thật (header + chỉ số patch bằng bitmap hoặc exp-Golomb, chọn cái ngắn hơn) và giải mã.
7. Bên nhận thay token vào bản đồ của mình, rồi chạy đầu nhiệm vụ.

**Đồng bộ và khởi tạo:**
- Bên gửi cập nhật bản sao chỉ bằng token **đã giải mã**. `assert` kiểm bản sao trùng từng bit với trạng thái bên nhận ở mọi khung.
- Khung 0 là thông điệp khởi tạo (mọi patch), được tính bit riêng.
- Kênh không mất gói trong B-E0.

**Nhiệm vụ:**
- Mỗi patch có / không có 4 lớp: người, xe con, xe lớn, xe 2–3 bánh. Nhãn lấy từ box VID; vùng `ignored` bị loại khỏi chấm.
- Đầu tuyến tính train trên 4 clip TUNE (211 khung).
- Đánh giá trên 2 clip `gate` (nguồn train giữ riêng) và 1 clip `aux_val` (VID-val không trùng DET-train); 371 khung, lấy 1 khung trong 3.
- Chỉ số: mAP theo patch.
- Code: [b0_pipeline.py](../swarmbridge/experiments/jepa_semcom/b0_pipeline.py), [test_b0.py](../swarmbridge/experiments/jepa_semcom/test_b0.py) (19 test tổng hợp, đều qua). Kết quả: `runs/jepa_semcom/b0_run.json`, `b0_extract_stats.json`, `b0_avif_bits.json`. Token nằm ngoài git (`~/work/jepa_semcom/`).

**Chi phí đo được:**
- **Trích token** (7 clip, 582 khung; RTX 3070, fp16, batch 16): GPU 138–315 khung/giây, bộ nhớ đỉnh 199 MB.
  - Giải mã JPEG + resize chậm hơn GPU (khoảng 100 khung/giây trên một luồng).
  - Toàn bộ 27k khung ước tính dưới 10 phút.
- **Mô phỏng codec + chấm** cho 8 cấu hình × 3 clip: 42 s trên CPU.

**Kết quả** (trung bình 3 clip đánh giá; mAP với token gốc = 0,588):

| k / khung | q | Bit / khung sau khởi tạo | mAP | % so với token gốc | cos(token nhận, token thật) |
|---|---|---|---|---|---|
| 0 (đóng băng khung đầu) | 4 | 32 | 0,114 | 19 % | 0,64 |
| 16 | 4 | 25,0 k | 0,351 | 60 % | 0,74 |
| 64 | 4 | 99,7 k | 0,525 | 89 % | 0,87 |
| 128 | 4 | 199 k | 0,558 | 95 % | 0,93 |
| 256 | 4 | 398 k | 0,579 | 98 % | 0,96 |
| 858 (gửi hết) | 4 | 1.333 k | 0,582 | 99 % | 0,98 |
| 64 | 2 | 50,5 k | 0,243 | 41 % | 0,44 |
| 64 | 8 | 198 k | 0,528 | 90 % | 0,89 |

- Thông điệp khởi tạo: 1,33 Mbit (q = 4).
- Tham chiếu kích thước AVIF của chính các khung đó ở 308 × 546, mỗi khung mã độc lập (chỉ đo bit, chưa đo nhiệm vụ):

| Chất lượng AVIF | 20 | 50 | 80 |
|---|---|---|---|
| kbit / khung | 21–44 | 73–142 | 201–359 |

**Đọc kết quả:**
1. **Pipeline chạy trọn vẹn và đồng bộ.** Đầu nhiệm vụ dùng được token tái tạo: chỉ cần gửi lại 30 % patch mỗi khung đã đạt 98 % mAP của token gốc.
2. **Dự đoán "giữ nguyên" yếu vì camera di chuyển.** Không gửi gì thì mAP rơi từ 0,59 xuống 0,11. Đây là chỗ một predictor tốt hơn (bù chuyển động trong latent, hoặc predictor học, gồm JEPA) có dư địa rõ. Câu hỏi B-1 đặt đúng chỗ.
3. **Token thô quá đắt:** mỗi token q = 4 tốn 1.552 bit.
   - Với k = 64 (89 % mAP), thông điệp khoảng 100 kbit/khung, cùng cỡ AVIF q = 50 của cả khung.
   - Muốn SemCom có lợi thì phải nén chiều token: PCA / bottleneck học được, mã entropy thay cho lượng tử đều. Đây là việc kỹ thuật cần làm trước B-3, không phải lý do dừng.
4. q = 8 không hơn q = 4; q = 2 kém hẳn. 4 bit mỗi chiều là đủ với lượng tử đều.

**Giới hạn:**
- Đầu nhiệm vụ train trên 211 khung, nên trần mAP thấp (0,44–0,80 tuỳ clip).
- 3 clip đánh giá. Kênh không mất gói.
- AVIF mới đo bit, chưa đo nhiệm vụ (cần chạy DINOv2 trên khung giải mã: B-3).

**Việc tiếp theo cho Paper B** (theo phản biện #27):
- **Nén token:** bottleneck tuyến tính / PCA xuống khoảng 32–64 chiều, mã entropy, đo lại đường cong bit – mAP.
- **Predictor không JEPA trong cùng không gian latent:** dịch / warp token theo chuyển động ước lượng, và một predictor học (ConvGRU / transformer nhỏ) hồi quy MSE trên token DINOv2. Cùng lịch sử đầu vào, cùng ngân sách tham số.
- **Định nghĩa rõ mục tiêu JEPA trước khi nói về JEPA:** context encoder + target encoder EMA học được, dự đoán latent của vùng bị che / khung tương lai. Hồi quy token kế tiếp trên DINOv2 đóng băng chỉ là "latent predictive regression", không phải đóng góp JEPA riêng.
- **B-3:** AVIF + DINOv2 bên nhận ở cùng bit, chấm cùng nhiệm vụ.

## Vướng mắc

- Không có vướng mắc chặn.
- `gpulock_run.py` vẫn hỏng (python của `.venv`); GPU lock được giữ bằng Windows python như các lần trước.
- WSL chỉ có 7 GB RAM: khi trích toàn bộ VID phải ghi token theo clip (đã làm vậy).
