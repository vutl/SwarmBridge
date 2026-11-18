# Đối chiếu Paper A với 20 paper gần nhất trong `pdf/` (19/11/2026)

Một agent đọc toàn văn bằng pypdf, có ghi số trang. Đây không phải khảo sát hệ thống: chỉ gồm các paper có trong thư
viện của người dùng. Văn bản trích xuất nằm trong scratchpad của phiên làm việc, không đưa lên git.

## Mỗi bài dùng mô hình tốc độ kênh nào

| Mô hình tốc độ | Số bài | Bài |
|---|---|---|
| Công thức Shannon / dung lượng log2(1+SINR) | 13 | Huang JSAC 2025 (eq.3); Si GLOBECOM 2023 (eq.9); Zhao TNSE 2026 (eq.6); Cross-Modal hierarchical DRL; Joint adaptive semantic compression (eq.14); Collaborative computation & migration (eq.16); CCSRA (eq.21); Airtalking (eq.21); Attention-based RL (eq.5); Cui MARL (eq.13); Wang/Hanzo MADRL-MEC (eq.10, "không xét MCS"); DDPG UAV-MEC (eq.2); SchedCP (eq.7) |
| Hàm khớp SNR/SINR → độ chính xác ngữ nghĩa hoặc ngưỡng SINR, không có MCS | 2 | Multi-modal SemCom UAV (TCOM 2025); Semantic-driven multi-UAV CoMP (HALO) |
| Không có mô hình liên kết tường minh | 3 | PASC (theo độ lợi kênh); SemanticRL (kênh AWGN/fading của codec); Decision Transformers |
| Mức liên kết / gói tin | 2 | HARQ cho cooperative perception (LDPC + QAM + OFDM + HARQ, MCS cố định, không có bộ điều khiển); UavNetSim (ngưỡng SINR, 2 Mbps cố định, CSMA/CA, ACK) |

**Không bài nào** mô hình hóa đồng thời việc chọn MCS từ CQI trễ, BLER, OLLA, ARQ cùng với một bộ điều khiển tài nguyên
học được hoặc tối ưu.

## So với ba kết quả của Paper A

1. **Bộ điều khiển thiết kế trên Shannon tụt mạnh dưới LA: không bài nào cho thấy.** Không bài nào đánh giá cùng một
   bộ điều khiển đã đóng băng dưới hai mô hình PHY. Gần nhất chỉ là đổi tham số trong cùng một mô hình kiểu Shannon:
   - SchedCP: train ở 300 kHz, test ở 200–600 kHz;
   - SemanticRL: đổi SNR;
   - Decision Transformers: đổi số mũ suy hao, có fine-tune;
   - PASC: sai số dự đoán kênh.
2. **PPO train trên LA không hơn heuristic đã tune + MPC hiệu chỉnh: chưa bài nào làm phép thử này.**
   - Mọi bài DRL đều so với baseline chưa tune hoặc đơn giản: random, round robin, nearest, max rate, cấp phát cố định,
     hoặc DRL khác dùng chung siêu tham số.
   - Bộ tối ưu dựa trên mô hình (PPDO, SCA, Lyapunov, NSGA-II) luôn được giả định có mô hình hoàn hảo.
   - Chỉ có tín hiệu yếu: ở Cui 2018 (tr. 23), matching có đủ thông tin thắng MARL; DP/exhaustive search đạt mức cao
     nhất ở Zhao (tr. 17) và Joint compression (tr. 13). Không bài nào có kiểm định thống kê hay dùng LA.
3. **Hiệu chỉnh mô hình tốc độ của MPC từ bit đã giao / ACK: không có trong thư viện.**
   - Gần nhất: PASC dự báo độ lợi kênh bằng GRU; Decision Transformers đưa tốc độ bước trước vào state.
   - "Calibration" trong CCSRA là hiệu chỉnh độ tin cậy của detector, không phải của tốc độ kênh.
   - **Ngoài thư viện:** nguyên lý dự báo và hiệu chỉnh throughput cho MPC đã có trong adaptive video streaming
     (RobustMPC, Yin et al. SIGCOMM 2015). Đóng góp 3 vì vậy là chuyển và đánh giá nguyên lý đó cho UAV SemCom, không
     phải phát minh.

## Giới hạn của đối chiếu này

- Chỉ gồm 20 paper trong thư viện, không phải tìm kiếm có hệ thống.
- Học tăng cường cho chính bài toán link adaptation (chọn MCS bằng RL) là một chủ đề đã có ngoài thư viện. Điểm của
  Paper A khác: đánh giá các bộ điều khiển ở tầng trên PHY khi mô hình PHY thay đổi.
