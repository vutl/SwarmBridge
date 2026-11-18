# Review công việc Claude — 02/11/2026

Đối chiếu transcript người dùng cung cấp, code hiện tại và JSON kết quả. Chỉ đọc và phân tích trên CPU; không khởi chạy training, không sửa code hay báo cáo của Claude. Số liệu tổng hợp: `runs/claude_review_2026-11-02/summary.json`; kiểm tra chuỗi: `sequence_checks.json` trong cùng thư mục.

## Kết luận

Claude đã làm một bước hữu ích: xây baseline có trí nhớ theo thời gian, có căn pose và đánh giá bằng rollout trên chính dự đoán trước đó. Lợi ích của trí nhớ xuất hiện cả trên test. Tuy nhiên, đây chưa phải thí nghiệm SB theo thời gian. Kết quả không chứng minh SB sẽ xử lý được khoảng cách với trí nhớ sạch, cũng không chứng minh lọc tuần tự là ứng dụng duy nhất SB có thể thắng.

Nên giữ baseline này và nghiên cứu tiếp giả thuyết SB có điều kiện theo thời gian. Tiêu chuẩn thành công phải là vượt các baseline có cùng thông tin quá khứ, cùng quan sát hiện tại và ngân sách tính toán được báo cáo.

## 1. Việc gì đã thực sự hoàn thành?

- Pilot matched sampler: 7 cấu hình × 3 seed × 6.000 updates, đã có kết quả. Các cấu hình SB không qua ngưỡng pilot đặt trước. `p70` là nhánh thăm dò được thêm sau khi đã xem một phần kết quả.
- Temporal: đủ 3 seed, 6.000 updates/seed, đủ JSON val và test; log kết thúc bằng `TEMPORAL DONE`.
- Model temporal hiện tại là U-Net mapping, không có SB, particle filter hay RL.
- Code chính: `swarmbridge/experiments/temporal/run.py`.

Luồng hiện tại:

```text
Ước lượng frame trước do chính bộ thu tạo
             + pose cũ/mới
                    ↓ căn tọa độ
Trí nhớ đã căn + latent vừa nhận + độ tin cậy + mask hợp lệ
                    ↓ U-Net
          Ước lượng frame hiện tại
                    ↓ lưu làm trí nhớ
```

Điều này dùng lại vùng quan sát của cùng UAV qua thời gian, phù hợp với UAV phân chia vùng bao phủ. Không đòi các UAV phải nhìn trùng nhau. Việc phân bổ truyền giữa nhiều UAV vẫn là phần tiếp theo, chưa được giải quyết bởi baseline temporal.

## 2. Kết quả test, trung bình ba seed

Các cột là mIoU trên toàn bộ rollout 20 frame; phần trăm là tỷ lệ block chọn gửi trước erasure của kênh.

| Chọn gửi | Mapping từng frame | Fill từ trí nhớ | U-Net temporal | Temporal với frame trước sạch |
|---|---:|---:|---:|---:|
| 15% | 0.60938 | 0.67862 | 0.68487 | 0.79003 |
| 30% | 0.69729 | 0.74198 | 0.75117 | 0.80846 |
| 50% | 0.75142 | 0.78339 | 0.78837 | 0.82118 |
| 70% | 0.79064 | 0.81443 | 0.81971 | 0.83821 |
| 100% | 0.83497 | 0.84639 | 0.85152 | 0.85792 |

Ở 15%, temporal hơn mapping 7,55 điểm phần trăm mIoU, nhưng hơn fill chỉ 0,62 điểm. IoU xe temporal = 0.23167, fill = 0.23439; temporal thấp hơn ở 2/3 seed. Do đó phần lớn mức tăng ở chế độ này đã có từ việc giữ và căn trí nhớ, chưa cần mô hình sinh. Ở 30–100%, temporal có cải thiện trung bình cả mIoU lẫn IoU xe so với fill.

Temporal gửi 30% đạt 0.75117, gần mapping gửi 50% đạt 0.75142. Điều này gợi ý giảm khoảng 40% số block chọn gửi tại mức chất lượng đó. Chưa phải phép đo giảm 40% tổng bit hoặc băng thông hệ thống: chưa hạch toán đầy đủ mã hóa, pose, điều khiển, phản hồi, và chi phí inference.

## 3. Những kiểm tra code đã qua

- Quan hệ `prev` giữ đúng scene và UAV. Train/val/test lần lượt có 55/10/20 chuỗi, mỗi chuỗi 20 phần tử; toàn bộ 1.100/200/400 grids được phủ đúng một lần, không có predecessor dùng chung sai cấu trúc hay chu trình.
- Quy ước đổi tọa độ trong temporal phù hợp `build_bev.py`: world = origin + res × (col × axis_col + row × axis_row). Đây là căn hình học từ metadata, không tối ưu alignment bằng latent sạch hiện tại.
- Các phương pháp nhận cùng hiện thực kênh ở mỗi frame trong một lượt rollout.
- `fill` và `temporal` có state riêng, cập nhật từ chính đầu ra của chúng. Không thấy đường đưa latent sạch quá khứ/tương lai vào bản temporal triển khai được. Nhánh dùng quá khứ sạch được tách riêng.

Phạm vi kiểm tra: đối chiếu code, metadata, đồ thị chuỗi và JSON; chưa kiểm chứng toàn bộ phép chiếu BEV bằng ground truth hình học độc lập. Warp latent bằng nội suy là một xấp xỉ vì encoder CNN không bảo đảm equivariance chính xác theo pose.

## 4. Các kết luận cần chỉnh

### Khoảng cách với trí nhớ sạch không phải phần hiệu năng SB được bảo đảm thu hồi

Ở 15%, chênh lệch 0.79003 − 0.68487 = 10,52 điểm mIoU trên test. Thay state bằng latent sạch cung cấp thông tin mà bộ thu chưa từng nhận, đồng thời đổi phân phối đầu vào. Khoảng cách gồm thiếu thông tin, sai số dự đoán quá khứ, lệch phân phối và có thể cả tích lũy sai số. Không thể quy toàn bộ cho tích lũy sai số.

JSON theo frame cũng chưa cho thấy suy giảm tích lũy rõ ở 15%: temporal trung bình frame đầu 0.57590, frame cuối 0.69055; trung bình frame 6–10 là 0.68851, năm frame cuối là 0.69675. Đây không chứng minh không có lỗi lan truyền, nhưng không hỗ trợ câu chuyện suy giảm dần chỉ bằng đường mIoU hiện có. Các frame khác nhau về nội dung nên cần đối chứng reset/rollout theo độ dài để tách nguyên nhân.

Nhánh `temporal_oracle_prev` nên được gọi là tham chiếu dùng quá khứ sạch. Nó không phải cận trên toán học, vì cùng mạng đã được train với quá khứ khôi phục và có thể phản ứng khác với đầu vào sạch.

### Train chưa khớp hoàn toàn rollout

Trong `TemporalTrainData.batch`, trí nhớ train do mapping từng frame đông cứng tạo ra. Khi test, trí nhớ do chính temporal tạo và lặp lại nhiều lần. Đây là một lệch phân phối cụ thể còn tồn tại. Nó không làm mất giá trị kết quả đã có, nhưng nên được kiểm tra bằng baseline được train với rollout trước khi quy phần yếu còn lại cho giới hạn của mapping.

### Không thể kết luận SB chỉ thắng trong lọc tuần tự

`sb-huong-moi-2026-11-02.md` nói quá rộng về literature. Chẳng hạn [SBM, Table 1](https://arxiv.org/html/2510.16834v2) báo VoiceBank-Demand ESTOI 0.891 so với Mamba-base 0.831, bên cạnh PESQ 3.503 so với 2.612. ESTOI là chỉ số intelligibility, nên không đúng khi gộp mọi lợi thế thành chỉ số cảm nhận. Tác giả dùng cùng họ backbone và dữ liệu, nhưng SBM có thêm timestep embedding, 3.93M so với 3.61M tham số; không nên gọi đó là phép so hoàn toàn đồng nhất.

Một benchmark lọc Lorenz là động lực nghiên cứu, không phải bằng chứng rằng bộ lọc SB sẽ thắng trên latent BEV cao chiều của ta. Các mệnh đề “SB chỉ có chỗ đứng khi…” nên được giữ như giả thuyết định hướng, không phải định lý hay kết luận tổng quát từ literature.

### Root của SB-decoder chỉ được xác định trong phạm vi đã thử

Pilot ủng hộ việc phân bổ ví dụ endpoint và độ khớp corruption train/eval có ảnh hưởng lớn. Nó chưa chứng minh có một root duy nhất, hay mọi cách dùng SB-decoder đều thất bại. Kết luận chắc hơn: recipe 6k và các cấu hình đã thử chưa đạt tiêu chí tiếp tục đặt trước.

Lưu ý 10% endpoint là 10% ví dụ, không phải 10% optimizer updates. Với batch hiệu dụng 32 và lấy độc lập, khoảng 96,57% updates vẫn có ít nhất một ví dụ endpoint; số ví dụ endpoint thấp hơn mapping mới là sự khác biệt cần bàn.

Hai chi tiết báo cáo pilot cần sửa khi cập nhật:

- `sb_c_p50` ở 15%: 0.59819 − `map_c` 0.61023 = −1,20 điểm mIoU; bảng hiện ghi −0,80.
- “Mọi biến thể SB kém map_c ở mọi metric” không chính xác tuyệt đối: vehicle IoU ở 100% của `sb_c_p50` = 0.587392, `map_c` = 0.587320, gần như hòa. Điều này không thay đổi kết luận không qua gate.

## 5. Trước khi chạy SB temporal

1. **Định nghĩa đúng bài toán.** Cần prior có điều kiện theo quá khứ, pose và khoảng thời gian: p(z_t | lịch sử). Ghép hai phân phối biên frame trước/sau bằng optimal transport không tự xác định được động lực thật của scene. Các cặp cùng scene phải được bảo toàn, và cần so với transition predictor được train trên cùng cặp.
2. **Định nghĩa đúng likelihood.** `z_p` là thống kê LMMSE đã co về prior, không phải phép đo AWGN trực tiếp của z. Với kênh age=0 hiện tại, điều kiện theo r và phần quan sát, code cho `z_p = r*z + noise`, phương sai noise = r*(1-r). Tương đương, ở r>0, dùng `z_p/r` như quan sát với phương sai `(1-r)/r`. r=0 không có quan sát. Coi `z_p` là z cộng noise phương sai 1-r sẽ đưa sai likelihood vào particle weights. Đây là yêu cầu cho code bộ lọc sắp viết, chưa phải bug của temporal mapping hiện tại.
3. **Giữ đối thủ đủ mạnh.** So SB với fill, temporal U-Net, và baseline uncertainty rẻ như ensemble hoặc predictor dị phương sai. Nhiều mẫu dự đoán không mặc nhiên cho uncertainty được hiệu chuẩn; các particle có thể cùng sai. Với latent 16×48×48, phải theo dõi suy biến trọng số/ESS và chi phí K×NFE.
4. **Đánh giá đúng mục tiêu dùng phân phối.** Ngoài mIoU/IoU xe, đo xem uncertainty có xếp đúng block đáng xin thêm và giúp tăng chất lượng trên cùng số bit không. Nếu lợi ích chỉ đến từ thêm trí nhớ, chưa thể ghi công cho SB.
5. **Chốt protocol trước khi xem kết quả mới.** Test hiện đã được xem; nên chọn recipe bằng val, ghi rõ số lần khảo sát test và giới hạn chỉ có bốn scene test. Ba seed tối ưu không tạo thêm scene độc lập. Lưu cấu hình, code hash, dữ liệu và checkpoint nguồn để resume không trộn recipe.

Timestamp dataset có độ lớn phù hợp nanosecond; tài liệu [AirSim](https://github.com/microsoft/AirSim/blob/main/docs/apis.md) ghi timestamp của trạng thái theo nanosecond. Nếu dataset giữ quy ước này, median khoảng cách frame là 1,51s train / 1,26s val / 1,38s test, và có khoảng cách tới 11,08s trên train. Cần xác minh đường export trước khi dùng số này làm dt của dynamics; không nên giả định mọi frame cách nhau một giây. Temporal hiện tại chưa dùng dt.

## Quyết định đề nghị

Giữ baseline temporal và tiếp tục kiểm tra SB như mô hình dự báo có điều kiện/bộ lọc. Câu hỏi cần trả lời là: với cùng lịch sử, bằng chứng nhận được và ngân sách truyền, biểu diễn phân phối bằng SB có giúp cập nhật trạng thái hoặc chọn yêu cầu truyền tốt hơn baseline temporal và ensemble không? Lượt vừa hoàn thành xác nhận trí nhớ hữu ích; nó chưa trả lời câu hỏi đó.
