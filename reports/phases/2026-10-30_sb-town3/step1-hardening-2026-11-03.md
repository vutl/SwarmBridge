# Bước 1 — củng cố kết quả chọn block (03/11/2026)

Hai kiểm tra rủi ro cho kết quả [temporal-block-selection-2026-11-03.md](temporal-block-selection-2026-11-03.md). Chỉ eval, không train.

## A. Mất phản hồi ACK

Code: `temporal_select/run.py --ack-loss p`. Với xác suất p, toàn bộ ACK của một frame bị mất. Khi đó bản mô phỏng của UAV giả định mỗi block đã gửi đến nơi với xác suất 1 − p_erase. Kết quả ở `runs/temporal_select_ackloss/p*/`. Lần chạy với p = 0 tái tạo đúng từng chữ số kết quả v1.

mIoU, 3 seed, ngân sách 15% / 30%:

| | random | ens_task (trạm) | replica_task (UAV) | ideal |
|---|---|---|---|---|
| Val, p = 0 | 0.682 / 0.759 | 0.730 / 0.813 | 0.774 / 0.834 | 0.778 / 0.836 |
| Val, p = 0.3 | | | 0.771 / 0.833 | |
| Val, p = 0.6 | | | 0.765 / 0.831 | |
| Val, p = 1 (không ACK) | | | 0.759 / 0.829 | |
| Test, p = 0 | 0.676 / 0.750 | 0.707 / 0.799 | 0.760 / 0.829 | 0.763 / 0.830 |
| Test, p = 1 (không ACK) | | | 0.744 / 0.823 | |

**Kết luận A:**
- Kể cả **không có phản hồi nào**, UAV chỉ mất khoảng 1.5 điểm ở mức 15%, và vẫn hơn hẳn trạm tự chọn (+3.7 trên test).
- Việc UAV không chắc trạm đã nhận gì chỉ là vấn đề **nhỏ** trong setup này. Khoảng trống cho một mô hình niềm tin (SB hay khác) về "trạm biết gì" chỉ khoảng 1.5–2 điểm.

## B. Baseline gửi bản đồ segmentation (tính bit chính xác)

Code: `swarmbridge/experiments/segmap_baseline/run.py`. Kết quả: `runs/segmap_baseline_v1/{val,test}.json`.

**Cách làm:**
- **Thứ được gửi:** UAV gửi nhãn BEV của teacher (192×192, 3 lớp) theo block 16×16 px, tương ứng 1 block latent.
- **Nén:** mỗi block được mã hóa độc lập bằng mô hình ngữ cảnh p(nhãn | trái, trên, trên-trái), fit trên tập train. Chi phí tính bằng độ dài mã lý tưởng.
- **Kênh:** digital. Block đến nguyên vẹn hoặc mất theo p_erase, cùng dải và cùng cách rút với các thí nghiệm khác.
- **Trạm:** dịch bản đồ trước theo pose. Pixel chưa từng biết điền lớp "other". **Không có mạng học nào.**

**Chi phí mỗi lưới:**

| | Val | Test |
|---|---:|---:|
| Toàn bộ bản đồ (đạt mIoU 0.882 nếu không mất gói) | 1.16 kbit | 1.00 kbit |
| Toàn bộ latent, lượng tử bước 0.5 + nén entropy | 57.1 kbit | 56.2 kbit |

**Test, mIoU / IoU xe theo kbit thực gửi mỗi frame (3 lần rút kênh):**

| kbit/frame | không nhớ + random | nhớ + random | **nhớ + UAV mô phỏng trạm** |
|---:|---|---|---|
| ~0.08 | 0.263 / 0.038 | 0.399 / 0.120 | 0.578 / 0.154 |
| ~0.17 | 0.314 / 0.101 | 0.518 / 0.236 | 0.686 / 0.340 |
| ~0.35 | 0.414 / 0.218 | 0.648 / 0.373 | **0.771 / 0.504** |
| ~0.6 | 0.537 / 0.349 | 0.743 / 0.487 | 0.805 / 0.568 |
| ~0.75 | 0.618 / 0.448 | 0.784 / 0.542 | 0.816 / 0.589 |
| gửi hết (~0.9–1.0) | 0.683 / 0.524 | 0.823 / 0.602 | 0.823 / 0.602 |

**So với pipeline latent tốt nhất (temporal U-Net + UAV mô phỏng trạm), test:**

| Chất lượng | Gửi latent | Gửi bản đồ | Chênh lệch |
|---|---|---|---|
| mIoU ≈ 0.76–0.77 | 15% block ≈ **8.4 kbit** | ≈ **0.35 kbit** | ~24 lần ít bit hơn |
| mIoU ≈ 0.82–0.83 | 30% block ≈ 16.9 kbit | ≈ 0.75–0.9 kbit | ~20 lần |

**Lưu ý khi so sánh:**
- Hai bên dùng kênh khác nhau. Latent đi kênh analog AWGN + mất gói; bản đồ đi kênh digital, cần mã kênh. Mã kênh có làm tăng số lần dùng kênh lên 2–3 lần thì bản đồ vẫn rẻ hơn khoảng 7–10 lần.
- Tính theo số lần dùng kênh cũng ra cùng chiều. Một block latent cần 256 ký hiệu thực. Một block bản đồ trung bình chỉ khoảng 7–12 bit, tức vài đến vài chục lần dùng kênh, kể cả khi SNR thấp.
- Số bit latent là ước lượng có lợi cho latent, vì chưa trừ phần chất lượng mất đi do lượng tử hóa.

**Kết luận B:**
1. Với **một nhiệm vụ cố định** (BEV segmentation) trên dữ liệu này, gửi latent 16×48×48 **lãng phí khoảng 20 lần** so với gửi chính kết quả nhiệm vụ. Lỗ hổng này nằm ở cách đặt bài toán "khôi phục latent", không riêng ở SB.
2. **Ý tưởng hệ thống vẫn đứng vững và còn mạnh hơn khi đổi sang gửi bản đồ:**
   - trạm có trí nhớ đã căn pose: gửi hết thì 0.683 → 0.823;
   - UAV mô phỏng trạm để chọn block: ở 0.35 kbit thì 0.648 → 0.771, IoU xe 0.373 → 0.504.

   Với kênh digital, UAV mô phỏng **chính xác** trạng thái trạm.
3. Latent chỉ có lý do tồn tại khi cần **đa nhiệm vụ** hoặc **ghép đặc trưng giữa nhiều UAV** (kiểu Where2comm). Town3 không kiểm tra được cả hai.

## Hệ quả cho hướng nghiên cứu

- Trục chính nên là **truyền bản đồ ngữ nghĩa có dự đoán**:
  - trạm nhớ và dự đoán;
  - UAV giữ bản sao trạng thái trạm và chỉ gửi block trạm đang sai;
  - phân bổ ngân sách bit giữa nhiều UAV.
- Chỗ mô hình học (và có thể là mô hình sinh) còn đất dùng là **trạm dự đoán bản đồ tốt hơn**. Hiện trạm chỉ dịch bản đồ cũ và điền "other" vào chỗ trống, chưa xét xe di chuyển hay vùng mới vào tầm nhìn. Dự đoán tốt hơn thì UAV cần gửi ít hơn.
- SB cho khôi phục latent (Q1, decoder) mất cơ sở dưới phép tính bit này. SB cho niềm tin về việc trạm biết gì chỉ còn khoảng trống khoảng 1.5–2 điểm (phần A).
