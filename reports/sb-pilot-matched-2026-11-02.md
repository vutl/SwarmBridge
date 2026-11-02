# Pilot SB với dữ liệu train khớp lúc đánh giá — kết quả (02/11/2026)

Code: `swarmbridge/experiments/sb_pilot/run.py`. Kết quả: `runs/sb_pilot_matched_v1/` (`summary.md`, `results.json`). Hàng đợi chạy 31/10 23:15 → 01/11 04:37, không lỗi.

## Thiết lập

- **Bộ sinh dữ liệu train mới** (áp dụng cho mọi model): chọn mức gửi đều trong {15, 30, 50, 70, 100}% → chọn ngẫu nhiên block trong footprint → kênh `EVAL_CFG` (erasure 0–0,6). Tỉ lệ lưới train nhận được ≤ 15% footprint: 21% (sampler cũ: 6,3%).
- Backbone, loss (MSE + 1×CE qua decoder đông cứng), optimizer, EMA, lịch VE c=0,4 k=2,6 giữ như audit `sb_audit_ve_v1`. 6.000 updates, seed 0/1/2, đánh giá EMA một bước trên 200 lưới val, cùng protocol audit.
- Model: `map_x` (mapping, input z_recv), `map_c` (mapping, input [z_recv, z_recv, r]), `sb_x_pXX` (SB x_t-only, XX% mẫu tại endpoint), `sb_c_pXX` (SB input [x_t, z_recv, r]). `sb_x_p70` là biến thể thăm dò, thêm sau khi thấy kết quả seed 0; không thuộc luật đặt trước.
- Gate đặt trước: SB − mapping cùng input ≥ −0,005 mIoU ở cả 15% và 30%, và IoU xe trung bình 70/100% ≥ −0,005.

## Kết quả (trung bình 3 seed, mIoU một bước)

| Model | 15% | 30% | 50% | 70% | 100% | IoU xe 70/100% |
|---|---:|---:|---:|---:|---:|---:|
| map_x | 0,6082 | 0,6931 | 0,7499 | 0,7890 | 0,8343 | 0,5266 |
| **map_c** | **0,6102** | **0,6962** | **0,7541** | **0,7929** | **0,8360** | **0,5342** |
| sb_x_p10 | 0,5688 | 0,6681 | 0,7364 | 0,7811 | 0,8316 | 0,5244 |
| sb_x_p50 | 0,5983 | 0,6872 | 0,7474 | 0,7884 | 0,8349 | 0,5309 |
| sb_x_p70 | 0,6019 | 0,6888 | 0,7495 | 0,7887 | 0,8352 | 0,5298 |
| sb_c_p10 | 0,5736 | 0,6739 | 0,7397 | 0,7832 | 0,8324 | 0,5281 |
| sb_c_p50 | 0,5982 | 0,6885 | 0,7491 | 0,7901 | 0,8353 | 0,5333 |

Gate: **không biến thể SB nào qua, ở cả 3 seed** (chi tiết theo seed trong `summary.md`).

## Phân rã khoảng cách SB–mapping tại mức gửi 15% (x_t-only, 6k)

| Bước | Khoảng cách | Thay đổi |
|---|---:|---:|
| Audit cũ (sampler cũ, 10% endpoint) | −4,56 | |
| Sửa sampler (vẫn 10% endpoint) | −3,94 | +0,6 |
| 50% endpoint | −0,99 | +3,0 |
| 70% endpoint (thăm dò) | −0,63 | +0,4 |
| Thêm conditioning [z_recv, r] (ở 50%) | −0,80 so với map_c tương ứng | không giúp |

## Kết luận

1. **Root chính là phân bổ huấn luyện endpoint** (≈3 điểm trên 4,5), phần nhỏ là sampler train/eval lệch (≈0,6). Không phải lỗi công thức, và conditioning không giúp.
2. Khi tăng tỉ lệ endpoint, SB tiến dần về mapping (p=1 đúng là mapping). Lợi thế xe ở 70/100% của SB so với map_x (+0,3–0,4) giảm cùng lúc với khoảng cách ở vùng gửi ít, tức đây là một đường đánh đổi, không có điểm nào thắng cả hai.
3. **So với mapping mạnh nhất (`map_c`), mọi biến thể SB đều kém ở mọi mức gửi, kể cả IoU xe.**
4. Sửa sampler giúp cả mapping (+0,7 điểm tại 15% so với sampler cũ); nên dùng sampler này cho decoder Q2.

Giới hạn: chỉ 6k updates. Audit 30k cho thấy mapping bị overfit còn SB tiếp tục tăng, nên ở 30k cân bằng có thể dịch về phía SB tại mức gửi cao; một phép so công bằng khi đó cần mapping có early stopping. Theo luật đặt trước, kết quả này là căn cứ để dừng đầu tư vào SB-decoder với recipe hiện tại.
