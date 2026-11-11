# SwarmBridge H1: kết quả Gate 0A (khám phá)

Ngày 12/11/2026.
- **Protocol:** [swarmbridge-h1-gate0-protocol-v2.md](swarmbridge-h1-gate0-protocol-v2.md), mục 10 (v2.1, khoá ở `74e047c3`).
- **Code và test** (14 test qua) commit trước khi chạy: `6f3e1949`, sửa hiệu năng ở `8297b558`.
- **Kết quả số:** `runs/uav_semcom/h1/gate0a_v21.json`.

**Đây là phân tích khám phá:**
- dữ liệu là 14 cặp chuỗi nhãn MDMT-test công khai, đã thấy khi chọn dataset;
- báo cáo hoàn hảo (box GT);
- không ảnh, không detector.

Kết quả **không phải bằng chứng xác nhận**. Gate 0B chưa được duyệt.

## 0. Tóm tắt

| Giả thuyết | Kết luận theo luật đã khai báo (Δ_free ≥ 5 và Δ_real ≥ 0,5·Δ_free) | Vùng có dư địa (tổng 2 UAV mỗi epoch) | Cỡ lợi ích |
|---|---|---|---|
| **H1-TOK** (token) | **Có dư địa, nhưng chỉ trong một dải hẹp** | 3–8 kbit/epoch, tức **0,55–1,4 lần** dòng token gửi hết (5,6 kbit/epoch) | Δ_free +5,5 đến +8,7; **thực tế (bitmap) +2,8 đến +6,3** điểm recall |
| **H1-CROP** (mảnh ảnh) | **Có dư địa, ở dải ngân sách thực tế hơn** | 100–500 kbit/epoch | Δ_free +5,1 đến +11,8; **thực tế +4,1 đến +8,9** |

Quy đổi sang tốc độ (tổng 2 UAV) theo tần số làm tươi r:

| Nhánh | r = 1 epoch/s | r = 2 | r = 5 | r = 10 |
|---|---|---|---|---|
| TOK | 3–8 kbit/s | 6–16 kbit/s | 15–40 kbit/s | 30–80 kbit/s |
| CROP | 100–500 kbit/s | 0,2–1 Mbit/s | 0,5–2,5 Mbit/s | 1–5 Mbit/s |

**Đọc nhanh:**
- **TOK:** phối hợp chỉ có ích khi liên kết dành cho token có cỡ xấp xỉ chính dòng token. Với uplink cỡ Mbit/s, hoặc khi ngân sách dư, phối hợp không giúp, thậm chí làm mất khoảng 1 điểm do lỗi trễ.
- **CROP:** dải có dư địa rơi vào tốc độ uplink UAV thông thường. Nhưng kết quả **giả định mảnh ảnh luôn cho detection đúng.** Giá trị thật phụ thuộc giả thuyết "detector trên UAV bỏ sót vật nhỏ", mà 0A không kiểm được.
- **Downlink không phải gửi lại box:**
  - bitmap trên danh sách track của UAV2 chỉ tốn **103 bit/epoch** DL, so với 1.668 bit nếu gửi lại box;
  - trong khi đó, phần trùng mà UAV gửi độc lập lãng phí là 1.658 bit/epoch (TOK) và 209 kbit/epoch (CROP).

  Phản biện đúng ở điểm này: giả định "DL tốn ngang phần tiết kiệm" của bản v2 quá bi quan.

## 1. Thiết lập (nhắc lại)

- **Dữ liệu và epoch:** 14 cặp chuỗi, cứ 10 khung lấy một epoch: 590 epoch, 41.410 lượt đối tượng duy nhất, 45% có mặt ở cả hai góc.
- **Ngân sách B:** tổng của 2 UAV mỗi epoch, gồm header, cộng ρ × bit DL điều phối; chính ρ = 1.
- **Token:** 52 bit; header 48 bit + 96 bit mỗi gói 1.200 byte.
- **Mảnh ảnh:** 1,5 lần box, tối thiểu 64 px; bit = 2.400 + 0,65 × diện tích.
- **Baseline độc lập:**
  - IND chia đều hoặc chia theo tỉ lệ α tốt nhất (chọn ngay trên EXPL, nên có lợi cho baseline);
  - Q: BS báo hạn mức, 22 bit;
  - TOK chọn ngẫu nhiên (kỳ vọng chính xác); CROP chọn rẻ nhất trước.
- **BS khử trùng bằng danh tính lý tưởng ở mọi chính sách.** Phối hợp chỉ có thể tiết kiệm ngân sách uplink.
- **Cơ chế phối hợp:**

  | Mã | Cơ chế | Ghi chú |
  |---|---|---|
  | S0 | Oracle miễn phí | |
  | S1 | Gửi lại box | |
  | S2 | Bitmap trên danh sách track của UAV2 | |
  | S2b | Bloom filter | |
  | S3 | Mặt nạ lưới 8×8 hoặc 16×9 | |
  | S3s | Mặt nạ tĩnh | |

  S1–S3 dùng thông tin của **epoch trước**, nên có lỗi trễ: bỏ nhầm đối tượng UAV1 đã mất, và gửi trùng đối tượng mới thành chung.

## 2. H1-TOK: kết quả chính (ρ = 1, s = 10)

| B (bit/epoch) | Baseline tốt nhất | S0 (oracle) | S2 bitmap | S3 lưới 8×8 | S3s tĩnh | S1 gửi lại box | Δ_free | Δ_real | Có dư địa |
|---|---|---|---|---|---|---|---|---|---|
| 1.000 | 17,6 | 18,6 | 15,9 | 15,5 | 17,8 | 0,9 | 1,0 | 0,2 | không |
| 2.000 | 38,2 | 43,0 | 40,6 | 39,5 | 40,0 | 13,6 | 4,9 | 2,5 | không |
| 3.000 | 53,8 | 59,3 | 56,6 | 56,3 | 55,6 | 27,5 | 5,5 | 2,8 | **có** |
| 4.000 | 64,7 | 71,4 | 68,9 | 68,0 | 67,0 | 39,3 | 6,7 | 4,2 | **có** |
| 5.000 | 73,7 | 81,3 | 79,0 | 77,7 | 76,3 | 51,1 | 7,7 | 5,3 | **có** |
| 6.000 | 80,9 | 89,6 | 87,2 | 85,4 | 83,7 | 62,3 | 8,7 | 6,3 | **có** |
| 8.000 | 91,5 | 97,6 | 95,9 | 93,8 | 92,7 | 80,8 | 6,1 | 4,4 | **có** |
| gửi hết | 100 | 100 | 99,0 | 96,8 | 96,1 | 99,0 | 0 | **−1,0** | không |

**Độ nhạy (các dải "có dư địa"):**

| Biến thể | Dải B có dư địa | Ghi chú |
|---|---|---|
| ρ = 0 | 3–8k | Δ_real 4,6–7,2; S1 tốt ngang S2 vì DL miễn phí |
| ρ = 0,5 | 3–8k | |
| s = 1 | 3–8k | |
| s = 30 (trễ lớn) | chỉ 5–6k | Δ_real 3,8–4,5 |
| bỏ box bị che | 3–6k | |

**Đọc kết quả:**
1. **Ngân sách thấp** (≤ 2k): phối hợp gần như vô ích. Với ít token thì xác suất hai UAV cùng chọn trúng một đối tượng trùng đã nhỏ sẵn.
2. **Ngân sách dư** (≥ dòng gửi hết): phối hợp không còn gì để tiết kiệm. Lỗi trễ (0,72 đối tượng bị bỏ nhầm mỗi epoch với S2) còn làm mất khoảng 1 điểm.
3. **Lợi ích chỉ có quanh ngân sách ≈ dòng token.** Nghĩa là một cơ chế triển khai được phải **biết khi nào nên phối hợp**, theo tỉ lệ ngân sách so với dòng token, và tắt khi dư.
4. **Gửi lại box (S1) với ρ = 1 tệ hơn không phối hợp**, vì DL tốn 1.668 bit/epoch. Cơ chế rẻ (bitmap 103 bit) mới đáng xét.

## 3. H1-CROP: kết quả chính (ρ = 1, s = 10)

| B (bit/epoch) | Baseline tốt nhất | S0 | S2 bitmap | S3 8×8 | Δ_free | Δ_real | Có dư địa |
|---|---|---|---|---|---|---|---|
| 50k | 10,2 | 12,6 | 12,4 | 12,2 | 2,4 | 2,2 | không |
| 100k | 20,7 | 25,8 | 24,8 | 24,5 | 5,1 | 4,1 | **có** |
| 200k | 38,3 | 47,3 | 45,2 | 44,2 | 9,0 | 6,9 | **có** |
| 500k | 71,5 | 83,4 | 80,5 | 78,6 | 11,8 | 8,9 | **có** |
| 1M | 96,2 | 99,9 | 98,0 | 95,8 | 3,7 | 1,8 | không |
| gửi hết (≈ 698k trung bình) | 100 | 100 | 99,0 | 96,8 | 0 | −1,0 | không |

- **Độ nhạy:**
  - bpp 0,3: dải 100–500k, Δ_real 5,0–8,3;
  - không overhead container (c0 = 0): dải 100–200k, Δ_real 6,0–8,1;
  - ρ = 0 hoặc 0,5, s = 30, bỏ box bị che: dải vẫn 100–500k.
- **Khi gửi độc lập, 30% bit mảnh ảnh là trùng** (209 kbit trên 698 kbit mỗi epoch). Bit điều phối (≈ 100 bit) không đáng kể so với phần này.
- **Giới hạn quan trọng:**
  - Ở đây mọi đối tượng đều được coi là cần gửi mảnh ảnh, và mảnh ảnh luôn cho detection đúng.
  - Trong hệ thật, mảnh ảnh chỉ nên gửi cho vật mà detector UAV bỏ sót hoặc không chắc; khi đó tập cần gửi nhỏ hơn và phần trùng có thể khác.
  - Điều đó chỉ đo được ở 0B, với detector thật.

## 4. Theo độ chồng lấn từng chuỗi

- **Tương quan giữa tỉ lệ chồng lấn chuỗi (0,23–0,83) và lợi ích là yếu:**
  - TOK ở 4k: r = 0,21 / 0,24;
  - CROP ở 200k: r = 0,50 / 0,28.
- **Yếu tố quyết định là ngân sách nằm ở đâu so với dòng của từng chuỗi:**
  - Chuỗi 31 (chồng lấn 0,83, khoảng 96 đối tượng mỗi góc) đạt +22,5 / +18,2 điểm ở TOK 6k.
  - Chuỗi 57 (0,68, ít đối tượng) có lợi ích ≈ 0, vì ngân sách đã đủ gửi hết.
- **Hệ quả:** với ngân sách cố định, lợi ích tập trung ở các chuỗi mà dòng gửi hết vượt ngân sách vừa phải. Chồng lấn cao là điều kiện cần, nhưng không đủ.

## 5. Giới hạn (phải đọc cùng kết quả)

1. **EXPL đã thấy:** dữ liệu đã dùng để chọn hướng, và α của baseline được chọn trên chính dữ liệu này.
2. **Báo cáo hoàn hảo:** không detector, không FP. Khử trùng ở BS dùng danh tính lý tưởng.
3. **Danh tính có trễ vẫn là oracle:** S1–S3 dùng GT ID của epoch trước. Lỗi liên kết thật (Gate 1) chưa có.
4. **Không có thời gian truyền hay hàng đợi:** đây là nhận thức offline theo ngân sách bit, không phải "đúng hạn".
5. **fps của MDMT không được ghi.** s = 10 khung là quy ước; độ nhạy s = 1 và 30 không đổi kết luận TOK về chiều, chỉ thu hẹp dải ở s = 30.
6. **Chỉ có 2 UAV.** Lợi ích với N ≥ 3 chưa được đo.

## 6. Ghi chép thủ tục

- **Lần chạy đầu treo 8,4 giờ CPU vì lỗi hiệu năng:** hàm tính số token tối đa giảm từng đơn vị từ 10¹⁰ khi ngân sách là 10¹² ("gửi hết"). Không có kết quả nào được ghi ra.
- **Bản sửa:** giới hạn theo số đối tượng; thêm 1 test. Commit `8297b558` **trước khi** xem kết quả.
- Lần chạy thứ hai mất 38 giây.

## 7. Đề xuất bước tiếp (cần duyệt; chưa làm)

1. **Nhánh CROP có lý do để vào 0B,** vì dải dư địa trùng tốc độ uplink thực tế. Câu hỏi 0B: khi chỉ gửi mảnh ảnh cho vật mà YOLO11n trên UAV bỏ sót hoặc không chắc, phối hợp còn giữ được lợi ích không?
   - Cần ảnh MDMT, nên phụ thuộc quyền dữ liệu.
   - **Chưa** mở VisionTrack test.
2. **Nhánh TOK chỉ đáng giữ cho kênh tốc độ rất thấp** (vài chục kbit/s ở 10 Hz). Không nên là trọng tâm.
3. **Cơ chế cần nghiên cứu nếu đi tiếp:**
   - **(a) Khi nào phối hợp:** bật hoặc tắt theo tỉ lệ ngân sách / dòng gửi hết. Lợi ích chỉ có quanh tỉ lệ ≈ 1; dư ngân sách thì phối hợp gây hại nhẹ. Quyết định này phụ thuộc trạng thái mạng, nên là chỗ nối tự nhiên với nền A5/A5.5, nơi PPO có ích khi trạng thái mạng dự đoán được. Đây là giả thuyết, chưa kiểm.
   - **(b) Liên kết xuyên góc không dùng GT** (Gate 1). Có thể thử trước trên VisionTrack `train` (góc nhìn rất khác nhau) mà không đụng tập xác nhận.
4. **Không thay đổi hướng RL:** A5/A5.5 vẫn là nền. H1 là thành phần hợp tác bổ sung.
