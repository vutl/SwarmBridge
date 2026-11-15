# Hai hướng song song: Paper A (DRL + UAV + SemCom) và Paper B (JEPA-SemCom)

Ngày 15/11/2026, theo phản biện #26. Giai đoạn: **khám phá (Stage A)**. Mục tiêu là có prototype chạy được, chưa phải kiểm chứng để đăng. Chỉ những lỗi làm thí nghiệm vô nghĩa (rò rỉ train/test, dùng tương lai, so sánh bất công) mới chặn tiến độ.

## Nền tảng chung

- **Một giao diện thông điệp chung:** `mode → (bytes thật, chất lượng nhiệm vụ, thời gian encode/decode)`. Bảng mode của A5 (`imcodecs.py`, `a5_profile*`) đã đúng kiểu này. Codec JEPA của Paper B sẽ xuất cùng dạng bảng, nên sau này có thể trở thành một mode mà PPO chọn được.
- **Cầu nối giữa hai bài:** A5 rút ảnh **ngẫu nhiên** từ VisDrone-DET, nên không có chuỗi thời gian; vì vậy JEPA (dự đoán từ lịch sử) chưa gắn vào A5 được. Cách nối là thay nguồn ảnh của A5 bằng **các clip VisDrone-VID liên tiếp, mỗi UAV một clip**. Việc này chỉ làm ở bước tích hợp, không phải điều kiện của bài nào.

## Paper A: DRL cho truyền thông ngữ nghĩa UAV khi mô hình kênh sai và thời gian quyết định có hạn

**Bằng chứng đã có** (A5.5, seed xác nhận, CI có tính video và seed huấn luyện):
- PPO mode-only hơn heuristic mạnh nhất H* một cách bền ở `phy_mcs_blind`: +5,7 (test), +7,8 (test-dev).
- Cũng bền ở K5 test-dev: +12,4.
- Đây là các regime mà heuristic dùng mô hình tốc độ sai (Shannon trong khi kênh thật dùng MCS) hoặc kênh chung chật.
- Chưa có: cơ chế, đối thủ là bộ tối ưu, và chi phí ra quyết định.

**Câu hỏi khoa học:** khi bộ tối ưu online dùng mô hình kênh sai lệch và phải ra quyết định trong một hạn thời gian, chính sách DRL học từ trải nghiệm có đạt chất lượng gần bộ tối ưu (hoặc hơn) với chi phí ra quyết định thấp hơn nhiều bậc không?
- Đây là luận điểm "amortized optimization + bền với sai mô hình".
- Không phải luận điểm "PPO chọn mode", vì phần đó đã có CCSRA, UA-TSC và bài PPO + SAC phân cấp năm 2026.

**Dùng lại:**
- `env.py`, `env_ev.py`, `gym_env.py`, `gym_env_ev.py`, `channel_a4.py`, `regime.py` (các regime ref, K5, phy_mcs_blind);
- heuristic: `policies.py`, `mode_heur.py`, `a55_heur.py` (H*, SC);
- huấn luyện / đánh giá: `a5_train.py`, `a5_eval.py`, `a55_eval.py`;
- 21 checkpoint trong `runs/uav_semcom/a5/train/`.
- Không đổi dữ liệu, không train lại để bắt đầu.

**Thêm:**
1. `a9_latency.py`: đo thời gian một quyết định (trung bình, p95) của PPO, H*, SC trên cùng CPU, cùng trạng thái.
2. `mpc_policy.py`: bộ tối ưu online tầm nhìn H bước.
   - Tìm kiếm có cắt tỉa / quy hoạch động trên (mode × kênh × công suất), dùng mô hình tốc độ của bộ điều khiển.
   - Có tham số ngân sách tính toán (số nút hoặc mili giây).
   - Hai phiên bản: dùng mô hình đúng (MCS) và mô hình sai (Shannon).
3. Đường cong **return theo thời gian quyết định** cho MPC (nhiều ngân sách) và PPO (một forward pass).

**Ba thí nghiệm tối thiểu** (CPU, seed 8000..8099 như A5.5):

| # | So sánh | Kết quả muốn thấy |
|---|---|---|
| A-E1 | PPO vs H* vs SC: return + ms / quyết định | Bảng chi phí; không cần PPO thắng về tốc độ so với heuristic |
| A-E2 | PPO vs MPC (mô hình đúng) ở nhiều ngân sách tính toán | PPO đạt bao nhiêu % return của MPC, và rẻ hơn bao nhiêu lần |
| A-E3 | PPO vs MPC (mô hình sai) trên `phy_mcs_blind` | Kiểm cơ chế: ưu thế của PPO có đến từ việc học kênh thật mà bộ tối ưu dùng mô hình sai không làm được không |

**Nếu kết quả tốt:** bước kế là một cơ chế mới cho Paper A, chẳng hạn PPO có ước lượng độ tin cậy của mô hình kênh, hoặc PPO dùng MPC làm hướng dẫn (lai). Chưa chốt trước khi có A-E1 đến A-E3.

**Rủi ro đã biết:**
- A7-S1 cho thấy dư địa của các mode nội dung (TOKEN / CROP / FRAME) trong A5 đến từ kiến thức GT cấp cảnh. Điều đó không ảnh hưởng tới kết quả PPO mode-only ở trên, nhưng Paper A không nên dựa vào các mode nội dung đó.
- Nếu MPC mô hình sai vẫn hơn PPO, luận điểm bền với sai mô hình yếu đi; khi đó chỉ còn luận điểm chi phí ra quyết định.

## Paper B: JEPA-SemCom, truyền phần đổi mới ngữ nghĩa theo dự đoán đồng bộ với bên nhận

**Câu hỏi khoa học:** một predictor dự đoán trong không gian latent kiểu JEPA, chạy đồng bộ ở cả hai đầu, có cho đánh đổi **bit – chất lượng nhiệm vụ** tốt hơn một predictor thời gian không theo JEPA cùng cỡ không? Đặc biệt khi có sự kiện ngữ nghĩa bất ngờ (vật thể xuất hiện / biến mất).

**Dữ liệu (đã có trên máy):**
- VisDrone2019-VID ở `~/data/visdrone_vid`: train 56 clip / 24.201 khung, val 7 clip / 2.846 khung, có box theo khung.
- Dùng VID-train (theo `r1b_split.py`) để huấn luyện và VID-val để đánh giá.
- Không dùng detector train trên DET, nên vấn đề rò rỉ DET-train ↔ VID-val (audit 10/11) không áp dụng.

**Hệ thống:**
1. **Encoder đóng băng:** DINOv2-S (đã có ở `~/work/models/dinov2-small`, Apache-2.0), token theo patch.
2. **Predictor nhỏ** (transformer 4 lớp hoặc ConvGRU, vài triệu tham số): dự đoán token khung t+1 từ các token mà **bên nhận** đang giữ. Bên gửi chạy bản sao y hệt, nên biết bên nhận dự đoán gì.
3. **Thông điệp đổi mới:** chọn k patch có sai số dự đoán lớn nhất (hoặc theo ngân sách bit); lượng tử token; ghi bitstream thật và giải mã lại (cách làm như codec Songdo).
4. **Bên nhận:** thay token đã nhận vào dự đoán, rồi chạy đầu nhiệm vụ.
5. **Nhiệm vụ:**
   - nhiệm vụ chính: bản đồ có / không có vật thể theo lớp trên lưới patch, nhãn từ box VID;
   - phụ: đếm theo lớp;
   - đầu nhiệm vụ nhẹ, train trên VID-train.

**Ba so sánh chính:**

| # | So sánh | Trả lời câu hỏi |
|---|---|---|
| B-1 | predictor latent kiểu JEPA vs predictor cùng kiến trúc nhưng dự đoán **pixel** (khung kế) rồi mới encode | JEPA có hơn dự đoán thông thường ở cùng số bit không |
| B-2 | dự đoán + đổi mới vs gửi token của các patch thay đổi **không dự đoán** (giữ token cũ ở bên nhận) | dự đoán có giảm bit không |
| B-3 | vs codec ảnh thông thường (AVIF trong `imcodecs.py`) ở cùng bit | SemCom có hơn gửi ảnh không |

Kèm phân tích riêng trên các khung có vật thể mới xuất hiện.

**Tính toán:**
- trích token cho 27k khung: vài phút trên RTX 3070 (đã đo DINOv2-S ở RQ1-E: 21k crop / 72 s);
- train predictor: dưới vài giờ GPU;
- GPU đi qua lock như mọi lần.

**Prior art phải đặt cạnh khi viết** (không chặn prototype): WirelessJEPA, CommJEPA (11/2026), TS-JEPA, conditional replenishment / predictive video coding, các phương pháp lọc token theo ngữ nghĩa trong SemCom.

## Tích hợp (sau khi cả hai chạy)

- Môi trường A5 lấy khung từ clip VID, mỗi UAV một clip.
- Thêm mode `JEPA-innovation` với bảng bit / chất lượng đo từ Paper B; PPO chọn giữa ảnh, token và JEPA.
- Đây là phần mở rộng, không phải đóng góp chính của bài nào.

## Thứ tự đề xuất

1. A-E1 (rẻ, chỉ CPU) và B-E0 (trích token DINOv2-S cho VID, kiểm pipeline round-trip với predictor "giữ nguyên") chạy ngay.
2. A-E2 / A-E3 và B-1..B-3 tiếp theo.
3. Phản biện lớn chỉ sau khi có kết quả prototype của cả hai.
