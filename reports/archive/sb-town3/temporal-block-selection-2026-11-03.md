# Chọn block trên receiver temporal — pilot 03/11/2026

Protocol: [review-block-selection-proposal-2026-11-03.md](review-block-selection-proposal-2026-11-03.md) (Codex). Code: `swarmbridge/experiments/temporal_select/run.py`. Kết quả: `runs/temporal_select_v1/` (`s{seed}_{val,test}.json`, `summary_{val,test}.md`). Chỉ eval, dùng checkpoint `runs/temporal_v1`, không train gì mới.

## Thiết kế

- Mỗi frame trong rollout 20 frame được ngân sách tổng b ∈ {15, 30, 50}% block trong footprint, chia làm 2 vòng:
  - **Vòng 1:** b/2 block ngẫu nhiên, cùng block và cùng hiện thực kênh cho mọi chính sách.
  - **Vòng 2:** b/2 block **mới**, do chính sách chọn, dùng chung một hiện thực kênh (fading/nhiễu/erasure rút theo seed của frame).
- Bộ thu luôn là temporal model `temporal_v1` cùng seed. Mỗi chính sách cuộn trên trạng thái của chính nó, nên chỉ khác nhau ở cách chọn vòng 2.
- Tập chính sách chốt trước khi chạy. Val chạy trước; test chạy đúng một lần với code không đổi.

| Chính sách | Bên quyết định | Thông tin được dùng |
|---|---|---|
| random | — | không |
| entropy | trạm | entropy của bản giải mã sau vòng 1 |
| ens_latent / ens_task | trạm | bất đồng giữa 3 temporal model (latent / xác suất lớp) |
| replica_latent / replica_task | UAV | latent sạch + **bản mô phỏng trạm do UAV tự giữ**, chỉ biết ACK từng block và SNR trung bình, không biết fading/nhiễu |
| ideal_latent / ideal_task | chẩn đoán | latent sạch + trạng thái **thật** của trạm. Không triển khai được chỉ với ACK, không phải cận trên task |

`_task` = 1 − xác suất của lớp mà teacher giải mã từ latent sạch (UAV có latent sạch nên tính được, không dùng nhãn GT).

## Kết quả test (3 seed; mIoU / IoU xe; Δ so với random, điểm mIoU)

| Chính sách | 15% | Δ | 30% | Δ | 50% | Δ |
|---|---|---:|---|---:|---|---:|
| random | 0.676 / 0.220 | | 0.750 / 0.364 | | 0.793 / 0.467 | |
| entropy | 0.700 / 0.261 | +2.4 | 0.789 / 0.461 | +3.9 | 0.830 / 0.561 | +3.7 |
| ens_latent | 0.689 / 0.279 | +1.3 | 0.785 / 0.461 | +3.5 | 0.826 / 0.559 | +3.3 |
| ens_task | 0.707 / 0.277 | +3.1 | 0.799 / 0.485 | +4.9 | 0.833 / 0.569 | +4.0 |
| replica_latent | 0.755 / 0.415 | +7.9 | 0.822 / 0.555 | +7.2 | 0.843 / 0.602 | +5.0 |
| **replica_task** | **0.760 / 0.419** | **+8.4** | **0.829 / 0.568** | **+7.8** | **0.846 / 0.607** | **+5.3** |
| ideal_task | 0.763 / 0.420 | +8.7 | 0.830 / 0.571 | +8.0 | 0.847 / 0.607 | +5.4 |

Mọi chính sách thắng random ở 3/3 seed, mọi ngân sách, cả val và test. Val cho cùng thứ tự với mức tăng lớn hơn một chút (15%: replica_task +9.2, ens_task +4.8, entropy +3.5).

## Đọc kết quả

1. **Còn rất nhiều chỗ cải thiện ở khâu chọn block.** Chọn đúng block ở vòng 2 tăng mIoU khoảng +8 điểm ở mức 15%, và IoU xe gần gấp đôi (0.22 → 0.42).
2. **Bản mô phỏng của UAV gần bằng bản ideal** (chênh ≤ 0.3 điểm). Trong mô hình kênh này, ACK cộng SNR trung bình đủ để UAV theo kịp trạm, dù không biết nhiễu. Chưa kiểm tra trường hợp mất ACK, ước lượng SNR sai, hay kênh lệch.
3. **Trạm tự chọn theo uncertainty có ích, nhưng chỉ lấy được một phần.** Ở 15%, ens_task đạt +3.1 so với +8.4 của UAV (khoảng 37%). Ở 30–50% thì đạt 62–75%. Uncertainty theo **xác suất lớp** tốt hơn theo latent. Entropy của một model rẻ hơn ensemble mà đã đạt khoảng 75–80% mức của ensemble.
4. **So theo cùng số block:**
   - replica gửi 15% (0.760) hơn random gửi 30% (0.750).
   - replica gửi 30% (0.829) hơn random gửi 50% (0.793) và hơn cả tham chiếu "trí nhớ sạch + random 30%" của temporal_v1 (0.809).
   - Đây là tiết kiệm số block, chưa phải bitrate thực.

## Giới hạn

- **Chi phí phản hồi chưa trừ vào ngân sách.**
  - ACK: 1 bit/block đã gửi. Yêu cầu của trạm: bitmap 144 bit/frame.
  - So với 2048 bit/block (theo quy ước 8 bit/giá trị), ACK chưa tới 0.1% và bitmap tương đương khoảng 7% của một block mỗi frame.
  - Chưa tính độ trễ của vòng phản hồi.
- **Chi phí tính toán trên UAV.** UAV phải chạy temporal model; bản `_task` cần thêm decoder. Bản `_latent` không cần decoder và chỉ kém khoảng 0.5 điểm.
- **Phạm vi dữ liệu.** Test có 4 scene / 20 chuỗi; 3 seed không phải scene độc lập. Kênh là mô hình đơn giản.
- **Tính mới.** Ý tưởng "encoder giữ bản sao decoder" là chuẩn trong video coding. Lập lịch closed-loop theo giá trị đã có trong GOSC. Kết quả này cho thấy hướng predictive communication có lợi lớn trên dữ liệu của ta; chưa xác nhận tính mới.

## Hệ quả cho SB

- **Khi có phản hồi**, UAV tự chọn từ latent sạch, không cần posterior phía trạm, và đã gần mức ideal.
- **Vai trò còn lại cho một posterior tốt hơn (SB hay khác)** là khi trạm phải tự quyết mà không có bản mô phỏng ở UAV:
  - không có phản hồi;
  - UAV không đủ tính toán;
  - trạm cần xin từ nhiều UAV mà chưa nhận gì.
- **Khoảng có thể giành được** nằm giữa ens_task và replica: khoảng 5 điểm ở mức 15%, khoảng 3 điểm ở 30%, khoảng 1.3 điểm ở 50%.
- **Muốn chứng minh SB có ích ở đây**, SB phải vượt entropy/ensemble trong khoảng đó, với chi phí tính toán được báo cáo.
- Thí nghiệm này không kiểm tra SB, và cũng không thay thế phép kiểm tra sampler trên bài toán có posterior giải tích.
