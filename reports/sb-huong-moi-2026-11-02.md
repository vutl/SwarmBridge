# Hướng mới cho SB trong SwarmBridge — tổng hợp từ thư viện paper (02/11/2026)

Nguồn: 4 nhóm đọc song song, khoảng 32 paper và 5 file kế hoạch trong `pdf/`. Mỗi nhóm được giao câu hỏi "SB nên đóng vai trò gì khác, khi vai trò decoder khôi phục đã thất bại". Bản trích text nằm trong scratchpad của phiên này.

## 1. Quy luật chung trong tài liệu

Mọi paper có đo chỉ số distortion hoặc task đều cho kết quả giống ta:

- **Conditional Simulation (CDSB), Table 3:** SB cải thiện FID (30,9 → 15,0) nhưng làm giảm PSNR (24,83 → 24,34).
- **I3SB, Table 1:** 1 bước cho PSNR tốt nhất, 100 bước cho FID tốt hơn.
- **SDDBMs:** sampler ODE tốt nhất về PSNR, SDE tốt nhất về FID/LPIPS.
- **BridgeCodec, Tab. 1–2:** NFE=1 tốt nhất trên STOI/MEL. Không có baseline regression.
- **SBM-Mamba (yang26e):** one-step SB thắng mapping cùng backbone chủ yếu trên chỉ số perceptual. Chạy một lần, không có khoảng tin cậy.
- **Optimally Bridging Semantics and Data (SB-SemCom):** cấu trúc giống hệt decoder ta đã thử. Thắng các baseline sinh ảnh về FID; không có ablation so với regression refiner cùng backbone.

**Thắng rõ duy nhất của SB trước phương pháp không-sinh** là lọc tuần tự: Lorenz-63, RMSE 0,178 so với 0,354 của EnKF (CDSB, Table 4). Ở đó, phân phối được **lan truyền theo thời gian và dùng lại**, chứ không bị thu gọn thành một điểm.

Quy tắc rút ra: SB chỉ có chỗ đứng khi
- (a) cả **phân phối** được dùng cho một quyết định, hoặc được mang sang bước sau;
- (b) **không có dữ liệu cặp**;
- (c) cần **ghép bằng chứng mới lúc suy luận** mà không train lại.

## 2. Trạng thái các vai trò SB trong file kế hoạch

| Vai trò | Trạng thái |
|---|---|
| Decoder SB z_recv → z*; one-step SB kiểu SBM; độ phân tán mẫu SB làm uncertainty (Q1b) | **Đã bác bỏ** bằng thí nghiệm 3 seed |
| Swarm-conditioned bridge, cần chồng lấn giữa các UAV | Không kiểm tra được trên dữ liệu hiện có; về bản chất vẫn là decoder cặp |
| Wireless-aware reference (R1/R2) | Bị chặn bởi gate của chính kế hoạch (SB isotropic đã thua mapping) |
| Transport energy, SB mô phỏng belief sau message, multi-marginal theo vòng, dịch codec giữa UAV khác loại, SB vật lý/điều khiển | Chưa kiểm tra |

## 3. Phát hiện mới trên dữ liệu: chồng lấn theo thời gian rất mạnh

Mỗi UAV có 17 cảnh × 20 frame liên tiếp. Giữa hai frame, UAV di chuyển trung vị 2,7 m, nên hai frame trùng khoảng 97% diện tích BEV.

Đo trên 570 cặp frame liên tiếp (val+test), dùng `map_c` seed 0 của pilot:

| | Gửi 15% | 30% | 50% | 100% |
|---|---:|---:|---:|---:|
| Chỉ phần nhận được | 0,303 | 0,387 | 0,488 | 0,664 |
| Mapping tốt nhất `map_c` | 0,615 | 0,696 | 0,754 | **0,833** |
| Nhận được + **frame trước bù chỗ mất** (không có mạng) | **0,742** | **0,751** | **0,763** | 0,779 |

Giải mã riêng frame trước (đã căn vị trí) đã đạt 0,734; teacher trên frame hiện tại đạt 0,882.

**Lưu ý, các con số trên là lạc quan:**
- dùng latent **sạch** của frame trước (thực tế trạm chỉ có bản khôi phục của nó);
- căn vị trí bằng cách dò dịch chuyển nguyên ô khớp nhất với frame hiện tại (thực tế dùng pose);
- phép bù là thô, không có mạng.

Dù vậy, khoảng cách +12,7 điểm ở mức gửi 15% đủ lớn để kết luận: **thông tin theo thời gian là nguồn lợi lớn nhất hiện có**, lớn hơn mọi cải tiến decoder đã thử. Script: `scratchpad/temporal_gonogo.py`.

## 4. Hướng mới được đề xuất

### H1 (chính). Temporal SwarmBridge: SB làm prior động học trong bộ lọc ngữ nghĩa

- **Luồng:**
  - Trạm giữ một tập mẫu (particles) về latent của mỗi UAV ở frame t−1.
  - SB đưa từng mẫu sang z_t (dự báo).
  - Các block nhận được ở frame t "kéo" các mẫu về phía bằng chứng: đây là likelihood, theo Reich / BridgeTwist, kênh xóa được mô hình bằng phép chọn tọa độ.
  - Độ phân tán giữa các mẫu cho biết chỗ nào trạm **chưa đoán được**. UAV chỉ cần gửi những chỗ đó (logic AoII / value-of-information).
- **SB nối:** z_{t−1} → z_t, hai đầu đều là dữ liệu thật. Đây là vùng mạnh nhất của SB, và khác hẳn decoder z_recv → z* đã thất bại.
- **Vì sao cần mô hình phân phối:**
  - Belief được mang qua nhiều frame. Một ước lượng điểm không ghép đúng với bằng chứng mới, và sai số tích lũy dần.
  - Xe di chuyển làm tương lai đa mode.
  - Mức xóa, SNR, số lần truyền lại thay đổi từng gói; likelihood hấp thụ được mà không phải train lại.
- **Đối thủ trung thực:**
  - U-Net nhận [z_recv, r, frame trước đã căn] cộng heteroscedastic, chạy cuộn nhiều frame bằng **bản khôi phục của chính nó** (không dùng latent sạch);
  - lấy frame trước bù chỗ mất;
  - EnKF.
- **Đóng góp hệ thống (không phụ thuộc SB thắng):** bộ thu dự báo được nên UAV chỉ gửi phần "bất ngờ". Đây là predictive semantic communication cho multi-UAV, kèm reward nội dung thật (ΔmIoU) cho RL.
- **Rủi ro:**
  - U-Net có frame trước có thể hòa với SB về mIoU. Khi đó lợi ích của SB phải nằm ở uncertainty, hiệu quả truyền lại và độ bền khi kênh lệch.
  - Particle sụp trong 36.864 chiều, cần tempering.
  - Chi phí K×N lần gọi mạng.
  - ~1.100 lưới có thể bị học thuộc.

### H2. SB mô phỏng phản thực tế để tính value-of-information cho RL

- **Ý tưởng:** lấy mẫu "UAV j sẽ gửi gì", giải mã, ước lượng E[ΔmIoU] cho bất kỳ hành động nào (tập block, ngân sách).
- **Đối chiếu bắt buộc:** regressor ΔmIoU trực tiếp, và Monte Carlo từ Gaussian heteroscedastic.
- **Rủi ro cao:** sample spread của SB đã thua heteroscedastic một lần (Q2 progressive). GOSC Prop. 2 cũng cho thấy scheduler chỉ cần xếp hạng thô.
- **Gợi ý:** nên gộp vào H1, vì particles của bộ lọc chính là bộ mô phỏng này.

### H3. Dịch latent giữa UAV khác loại encoder, ít hoặc không có cặp (BridgeCodec + FSBM / α-DSBM)

- **Ý tưởng:** UAV với camera hoặc encoder khác gửi latent; SB dịch sang không gian của decoder leader.
- **Vì sao cần SB:** chỉ khi thiếu cặp; nếu có đủ cặp thì U-Net làm được (BridgeCodec NFE=1).
- **Thí nghiệm:** train encoder B riêng; N_pairs ∈ {0, 16, 64, 256, all}; so linear/Procrustes, U-Net, SB bán giám sát.
- **Rủi ro:**
  - Hai không gian latent độc lập không cùng hình học, nên OT thuần không cặp dễ dịch sai ngữ nghĩa.
  - Phải tự mô phỏng tính "khác loại".

### Không khuyên làm

- Các biến thể decoder mới (SDDBM, I3SB, DDSB, CDSB-C): vẫn là decoder khôi phục.
- Fusion nhiều UAV (CoGMP SGFF, multi-marginal): cần chồng lấn, dữ liệu không có.
- Branched SB: không hợp với latent nhiều chiều.
- Transport energy kiểu GMF: quy về residual của regression.

## 5. Cảnh báo baseline và việc không cần SB

- **Gửi bản đồ segmentation đã nén** (kiểu Grassucci / late fusion của CoSDH) có thể rẻ hơn gửi latent 16×48×48 trên mỗi bit. Phải đo trước khi tuyên bố tiết kiệm bit.
- RL nên:
  - dùng ΔmIoU thật làm reward; đây là điểm khác biệt rõ nhất so với các paper UAV-SemCom-RL, vốn dùng đường cong fit offline;
  - lấy luật GOSC "value > price × c*(SNR)" làm baseline phải thắng;
  - bắt đầu bằng contextual bandit, rồi mới chuyển sang constrained MDP khi có ngân sách năng lượng;
  - tính độ trễ decoder vào reward.
- **Môi trường RL:** custom gymnasium trên latent town3 cache, fading Gauss-Markov. UavNetSim chỉ dùng sau này để lấy trace SINR.

## 6. Thứ tự đề xuất

1. **Baseline thời gian thực tế (~1 ngày GPU):**
   - U-Net nhận thêm frame trước, căn bằng pose, dùng **bản khôi phục của chính nó** từ frame trước, cuộn qua 20 frame của mỗi cảnh;
   - đo mIoU theo bit, so với gửi segmentation map.
   - Nếu lợi thế thời gian còn lớn trong điều kiện thực tế này, nó là trục chính của paper.
2. **SB z_{t−1} → z_t làm prior** (particles + likelihood), so với U-Net ở bước 1, đo trên uncertainty / truyền lại / kênh lệch, 3 seed.
3. **Môi trường bandit/RL:** quyết định UAV nào gửi phần "bất ngờ" nào, reward ΔmIoU.
4. H3 chỉ khi cần câu chuyện đội UAV khác loại.
