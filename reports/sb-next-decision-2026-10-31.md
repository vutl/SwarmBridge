**Quyết định tiếp theo cho SB, sau chẩn đoán checkpoint — 31/10/2026**

Đề xuất: tiếp tục Q2 với mapping làm decoder đối chứng chính; giữ nhánh nghiên cứu SB bằng một phép thử khôi phục thiếu có điều kiện, giới hạn ngân sách và thay từng yếu tố. Chưa có căn cứ mở rộng sang Mamba hoặc distillation của một teacher chưa thắng đối chứng trên mục tiêu cần cải thiện.

**Phép chẩn đoán đã chạy, không huấn luyện lại**

Đã load checkpoint EMA 30k của mapping/SB, đủ ba seed, đánh giá cùng 200 lưới validation tại 15/50/100% block chọn gửi. Dùng lại đúng channel state, mask và noise của audit cũ. Sai lệch mIoU lớn nhất so kết quả cũ là 1,14e-5 khi đổi eval batch từ 32 sang 16; phù hợp sai khác số học AMP.

Tách các vị trí theo reliability của thông tin thực sự nhận được: `r=0` không có bằng chứng; `0<r<0,5` có bằng chứng yếu; `r>=0,5` có bằng chứng mạnh. Mask latent được upsample nearest sang kích thước nhãn. Chỉ tính pixel có nhãn hợp lệ; đã xác minh không có pixel hợp lệ ngoài footprint UAV. Decoder dùng thông tin lân cận, nên đây là phân tích theo vị trí đầu ra, không phải ba nhiệm vụ độc lập.

Khi chọn gửi 15%, sau mất gói còn khoảng 10,59% pixel hợp lệ có bằng chứng; khoảng 89,41% không có bằng chứng. Không đồng nhất 15% được chọn gửi với 15% thông tin nhận được.

| Vị trí đầu ra, chọn gửi 15% | Mapping mIoU | SB mIoU |
|---|---:|---:|
| Toàn bộ nhãn hợp lệ | 0,62115 | 0,58882 |
| Không có bằng chứng | 0,59356 | 0,55672 |
| Bằng chứng yếu | 0,80778 | 0,80292 |
| Bằng chứng mạnh | 0,87557 | 0,87207 |

Latent MSE trong vùng không có bằng chứng là mapping 1,4717 và SB 1,6231. Điều này bổ sung vị trí của thất bại, không xác định nguyên nhân trong training.

Đã kiểm tra thêm hai can thiệp lên latent dự đoán trước decoder, không thêm nhãn thật:

| Nguồn latent, chọn gửi 15% | mIoU toàn bộ |
|---|---:|
| Toàn bộ từ mapping | 0,62115 |
| Toàn bộ từ SB | 0,58882 |
| Vùng đã nhận lấy mapping; vùng thiếu lấy SB | 0,58962 |
| Vùng đã nhận lấy SB; vùng thiếu lấy mapping | 0,62136 |

Khi thay riêng phần thiếu của SB bằng phần thiếu do mapping ước lượng, gần như toàn bộ khoảng cách được khôi phục. Thay phần đã nhận thì không có tác dụng tương tự. Can thiệp có tương tác không gian qua decoder và tạo ranh giới latent mới; không phải chứng minh nhân quả về lý do SB học kém và không phải model hybrid đã được chọn để triển khai.

Ở mức chọn gửi 100%, SB nhỉnh hơn mapping trong cả ba vùng (mIoU vùng thiếu 0,74495 vs 0,74103). Vì vậy kết luận phù hợp là SB hiện gặp khó hơn khi thiếu bằng chứng trên diện rộng, không phải cứ có một block mất là SB luôn kém.

Script: `scripts/diagnose_sb_bev_regions.py`. Chi tiết: `runs/sb_bev_region_diagnostic/results.json` và `summary.json`. Phép đánh giá đã kết thúc; không còn job GPU nền do phép chẩn đoán này.

**Pilot đã sửa theo phản biện, chưa khởi chạy from scratch**

**Cập nhật sau audit nguyên nhân:** đã phát hiện sampler train đang gửi toàn bộ latent rồi erase 0–90%, trong khi eval 15% chọn gửi trước rồi mới chịu channel loss. Nếu sửa sampler để mô phỏng đúng cả hai bước trước pilot, phải train lại cả hai baseline x-only: tổng năm model trong bảng, thay vì chỉ ba model mới. Xem số liệu và cách phân biệt distribution mismatch với endpoint allocation trong [kiểm toán nguyên nhân](sb-training-root-audit-2026-10-31.md). Bảng bên dưới mô tả đối chứng khi giữ sampler cũ; không tái sử dụng baseline cũ để gán nguyên nhân sau khi đổi sampler.

Có hai giả thuyết riêng: thiếu conditioning về bằng chứng nhận được và thiếu tập trung huấn luyện tại endpoint. Thiết kế ban đầu chỉ kiểm tra giả thuyết đầu tiên; chưa đủ để kết luận nguyên nhân nếu thất bại. Xem [kiểm toán công thức, loss/gradient và can thiệp endpoint](sb-training-root-audit-2026-10-31.md).

| Model | Input / endpoint | Đối chứng |
|---|---|---|
| Mapping x-only | z_received; t=1 | Đã có seed 0, 6k |
| SB x-only | x_t; 10% endpoint | Đã có seed 0, 6k |
| SB x-only mới | x_t; 50% endpoint | So với hai dòng trên |
| Mapping có điều kiện mới | [z_received,z_received,r]; t=1 | Đối chứng input 33 kênh |
| SB có điều kiện mới | [x_t,z_received,r]; 10% endpoint | So với mapping 33 kênh |

Giữ cùng backbone, MSE+CE, khởi tạo theo seed, dữ liệu/kênh, effective batch, optimizer và EMA. Chạy lần lượt ba model mới, một seed, 6k updates mỗi model. Ước tính 1–1,5 giờ theo tốc độ đã đo. Mỗi SB phải so với mapping có input family tương ứng; không gán khác biệt input 16/33 kênh cho riêng endpoint allocation.

Gate mở rộng: mIoU_SB − mIoU_mapping_matched >= −0,005 tại cả 15% và 30% chọn gửi; đồng thời trung bình IoU xe tại 70/100% không thấp hơn mapping_matched quá 0,005. Đây là 0,5 điểm phần trăm, không phải 0,5 mIoU tuyệt đối. Mọi so sánh cùng seed 0, cùng 6k; không lấy lợi thế xe ở 30k làm chuẩn cho model 6k. Gate này dùng để cân nhắc ba seed, không phải bằng chứng SB thắng hay kiểm định thống kê equivalence.

Nếu cần, bổ sung có điều kiện + 50% endpoint để kiểm tra tương tác. Hai thay đổi riêng lẻ không giúp vẫn chưa chứng minh mọi tổ hợp hoặc mọi cách dùng SB đều không phù hợp. Dừng mở rộng có thể là quyết định ngân sách, cần nói đúng phạm vi kết luận. Recipe cũ của Claude đã có source/CSI nhưng khác clock, residual và endpoint nên không thay thế đối chứng giữ các yếu tố kia cố định.

**Quyết định về vai trò SB**

Kết quả speech/vocoder xác nhận có những cách dùng SB phục hồi tốt từ đầu vào suy giảm; chưa xác nhận lợi ích riêng so mapping tại máy. Việc khôi phục thông tin không nhận được cần prior có ích và bám đúng nhiệm vụ. Cộng stochasticity vào latent không tự mang thêm bằng chứng về cảnh thật. Đây là giới hạn thông tin chung của mapping lẫn SB, không phải lý do để miễn kiểm chứng cho bất kỳ model nào.

Mục tiêu nghiên cứu vẫn là đạt chất lượng nhiệm vụ với ít tài nguyên truyền hơn. Các UAV có thể quan sát những vùng khác nhau và cùng chia sẻ ngân sách; không cần đổi dataset chỉ để tạo chồng vùng nhằm giữ vai trò SB. Decoder hiện tại, conditional inference/uncertainty cho lựa chọn truyền và codec translation là các câu hỏi khác nhau. Chỉ mở vai trò mới khi có nhu cầu và đối chứng tương ứng; chưa gộp chúng vào một queue lớn.
