**Kiểm toán công thức SB theo đề xuất của Claude — 30/10/2026**

Mục tiêu là kiểm tra liệu công thức huấn luyện bridge khác có cải thiện phục hồi latent Town3 so với mapping khi giữ backbone, loss nhiệm vụ và ngân sách cập nhật. Thí nghiệm này không giả định các UAV phải nhìn trùng vùng; nó kiểm tra decoder phục hồi thông tin của từng UAV trong thiết lập đang có. Chưa thêm RL hoặc Mamba vào phép so.

**Phép so chính trên Town3**

| Thành phần | Mapping đối chứng | SB được kiểm toán |
|---|---|---|
| Backbone | U-Net width 64, input 16 kênh | Cùng kiến trúc, số tham số, khởi tạo theo seed |
| Đầu vào | Latent nhận được, t=1 | Chỉ x_t và embedding thời gian; không thêm latent nguồn/CSI |
| Đích dự đoán | Latent sạch | Latent sạch |
| Loss | MSE latent + 1 × segmentation CE qua decoder đông cứng | Giống mapping |
| Thời gian train | t=1 | 90% uniform [0,0001; 1], 10% đúng t=1 |
| Lịch bridge | Không có | VE, c=0,4; k=2,6 |
| Ngân sách | 30.000 optimizer updates × 3 seed | Giống mapping |
| Suy luận chính | Một lần gọi mạng | Một lần gọi mạng tại t=1 |

Với η(t)=c[k^(2t)−1]/[2 log(k)], marginal là:

    x_t = [1 − η(t)/η(1)] x_clean + [η(t)/η(1)] x_source
          + sqrt[η(t)(η(1) − η(t))/η(1)] ε.

Phải đổi cả hệ số nội suy và phương sai theo η, không chỉ thay độ lớn nhiễu. Phương sai cực đại khoảng 0,301 trên dữ liệu latent đã chuẩn hóa. Tham số c,k lấy từ [Jukić et al., Table 1](https://www.isca-archive.org/interspeech_2024/jukic24_interspeech.pdf). Tỉ lệ endpoint 10% là lựa chọn kiểm toán của chúng ta; không khẳng định đó là hyperparameter của paper. Đây là bản thích nghi công thức cho BEV, không tái lập trọn vẹn kiến trúc/loss speech của [SBM](https://arxiv.org/html/2510.16834v2).

Có thêm `mapping_csi_residual`: mapping dùng CSI và đầu ra residual như baseline mạnh trước đây, được train cùng số bước và loss. Biến thể này có input khác; không gộp nó vào cặp kiểm soát kiến trúc chính.

Optimizer AdamW lr 3e-4, weight decay 0,01; clip gradient norm 10; EMA 0,999; effective batch 32, microbatch 16, tích lũy hai lần. Dùng cùng stream lấy mẫu dữ liệu/kênh theo seed và một stream ngẫu nhiên riêng cho bridge. AMP bfloat16 trên GPU. Mỗi model dùng seed 0,1,2. Lưu và đánh giá EMA tại 6k, 15k, 30k; kết luận chính theo checkpoint 30k đã quy định, không chọn checkpoint đẹp nhất trên val.

Đánh giá cùng 200 lưới validation tại ngân sách chọn gửi 15/30/50/70/100%, cùng điều kiện kênh. Chỉ số: mIoU, IoU xe và MSE latent. Tỉ lệ chọn gửi không phải tỉ lệ nhận thành công. So SB một bước với mapping một bước là phép so chính; SB bốn bước stochastic và bốn bước mean-path là kiểm tra phụ. Mean-path là heuristic bỏ nhiễu, không gọi nó là probability-flow ODE. Chưa dùng test set để chọn công thức; sau audit cần đóng băng lựa chọn trước khi đánh giá test.

**Kiểm tra ảnh nhỏ độc lập**

Dùng CIFAR-10: chia cố định 40k train / 10k validation từ official training split, đánh giá 1.000 ảnh validation. Official test được tải nhưng không dùng. Khoét vùng trung tâm 16×16 trên ảnh 32×32, điền Gaussian noise; cả mapping và bridge nhận cùng ảnh hỏng và mask. Loss chỉ tính trong vùng thiếu; khi suy luận giữ nguyên các pixel đã biết ở mọi bước.

Hai model dùng cùng U-Net, input, batch 64, optimizer và 10.000 updates × 3 seed. Mapping dự đoán ảnh sạch; bridge dùng noise-label `(x_t−x_clean)/sqrt(η(t))` như [I²SB runner](https://github.com/NVlabs/I2SB/blob/master/i2sb/runner.py), Brownian clock q=1. Đây là chẩn đoán I²SB-style riêng, không phải bản VE của thí nghiệm BEV. Bridge được đo ở 1/4/16 bước stochastic. Lưu preview ảnh sạch/ảnh hỏng/ảnh phục hồi; đo PSNR toàn ảnh và vùng thiếu, SSIM, LPIPS AlexNet sau resize 64×64, cùng sai số pixel đã biết.

Đây không phải tái lập benchmark ImageNet dùng pretrained ADM và lịch beta đối xứng của paper I²SB. Thấy ảnh sắc hơn không chứng minh posterior đúng; nếu không thấy lợi ích cũng không bác bỏ kết quả paper. Kiểm tra công thức độc lập và thực nghiệm chất lượng trả lời các câu hỏi khác nhau.

**Chạy và khôi phục**

    .venv\Scripts\python.exe -m unittest discover -s tests -p test_sb_audit.py -v
    powershell -ExecutionPolicy Bypass -File scripts/start_sb_audit.ps1

Launcher chạy cửa sổ ẩn, ghi PID và đường dẫn log tại `runs/sb_audit_ve_v1/launcher.json`. Queue chạy các chẩn đoán ảnh trước, sau đó Town3 theo từng mốc 6k → 15k → 30k, lần lượt đủ ba model và ba seed ở mỗi mốc. Chạy lại cùng lệnh sẽ đọc checkpoint, tự bỏ qua phần đã hoàn thành và tiếp tục phần còn thiếu.

Theo dõi `runs/sb_audit_ve_v1/status.json` và `summary.md`. Mỗi job lưu `last.pt` mỗi 500 updates, gồm model gốc, EMA, optimizer, bước và các trạng thái RNG; có thể phải chạy lại tối đa 499 updates nếu dừng đột ngột. Checkpoint EMA ở từng mốc được lưu riêng. Manifest lưu cấu hình, hash code và teacher/cache; từ chối resume nếu chúng thay đổi để không trộn các thí nghiệm khác nhau.

Sáu kiểm tra độc lập đã qua: VE closed form và endpoints; reverse kernel so với tích hai Gaussian; Monte Carlo pinned marginal; train endpoint-only tương đương mapping cả loss và gradient; chuyển đổi noise-label và giữ pixel đã biết; resume khôi phục optimizer, EMA và RNG cho cập nhật tiếp theo giống chạy liền. Năm test nền trong `tests/test_q1_core.py` cũng đã chạy lại và đều qua. Phụ thuộc bổ sung `lpips==0.1.4`, CIFAR-10 và weights AlexNet đã được cài/tải trong workspace.

Smoke run `runs/sb_audit_smoke` dùng 20 updates/model, 8 lưới BEV và 16 ảnh, chỉ xác nhận luồng chạy. Không diễn giải các chỉ số smoke thành kết quả nghiên cứu. Với tốc độ nóng máy quan sát khoảng 0,034 giây/update cho ảnh và 0,23 giây/update cho BEV, tổng 60k updates ảnh + 270k updates BEV có thể mất khoảng 18–24 giờ tính cả đánh giá và I/O; đây là ước lượng, không cam kết thời gian hoàn thành.

Một kết quả SB thắng/thua sau audit chỉ kết luận về công thức, dữ liệu và ngân sách đã thử. Không suy ra chắc chắn rằng mọi cách dùng SB đều phù hợp hoặc không phù hợp với SwarmBridge.

**Trạng thái khởi chạy thực tế**

Smoke run đã hoàn tất cả năm cấu hình, gồm huấn luyện, checkpoint và đánh giá. Đợt chính thức bắt đầu lúc 23:50:47 ngày 30/10/2026, Asia/Seoul; đã xác nhận job đầu `image_mapping_s0` vượt 1.800/10.000 updates và checkpoint 1.500 được ghi thành công. Đây là ảnh chụp trạng thái lúc khởi chạy, không phải tiến độ cập nhật liên tục; đọc `status.json` để xem trạng thái hiện tại. Chưa có kết luận về chất lượng SB từ đợt chính thức.
