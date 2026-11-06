# Hướng A — UAV image delivery + hybrid bit/semantic modes + DRL

Bắt đầu 2026-11-06. Hướng do người dùng chọn (theo đề xuất ChatGPT): triển khai trực tiếp một framework UAV + SemCom + RL
trên dữ liệu thật, không thêm vòng "gate" quyết định có nên nghiên cứu. Huang Step 0 (`reports/huang-step0.md`) chỉ
còn là tài liệu tham khảo.

Thứ tự: A1 (bảng mode thật theo từng ảnh) → A2 (môi trường có topology nhất quán) → A3 (RL thật + baseline công bằng).

## 0. Prior art gần nhất (đọc toàn văn 2026-11-06)

| Bài | Topology | Mode semantic / chất lượng | Biến quyết định | Solver / baseline |
|---|---|---|---|---|
| Huang et al., JSAC 2025 | LEO → UAV relay → user | BLIP-2 caption, caption + VQ latent → diffusion; **1 ảnh ví dụ**, bảng PSNR/bit cố định | mode, pairing UAV, quỹ đạo, ISL | DSAC; chỉ MAPPO |
| Si et al., GLOBECOM 2023 | user → UAV (thu thập, uplink), NOMA | Q(η) = hàm log khớp từ bài FAST (CIFAR); **η không đổi số bit gửi** | model scale η, kênh, công suất, quỹ đạo | hybrid PPO (2 agent); EP, triple PPO |
| Li et al., JSAC 2026 (arXiv 2606.00668) | 1 UAV (sạc WPT từ EHBS) → user mặt đất | Lấy nguyên mode của Huang; **không nêu dataset**, R_k lấy từ 1 ảnh ví dụ | quỹ đạo 3D, chọn EHBS, chọn user, mode, tỉ lệ thời gian sạc β | DSAC + decision assistant; MAPPO, DSAC, fixed β, hovering; **không greedy/fixed-mode** |
| Truong et al. (UCLab), IoT-J 13(17) 2026 | user → UAV relay → BS; UAV cố định | Learned codec ADSC (TCM + hyperprior), 6 mô hình λ, **PSNR đo theo ảnh CLIC** | 1 λ chung mỗi slot, chia băng thông | Giải tách + dạng đóng, không RL; QG/TG-DSC, EBW |
| Do et al. (UCLab), JNCA 2026 | multi-UAV relay, multi-flow routing | đường cong fidelity DeepSC (BLEU) | định tuyến | Ant colony; thắng nhẹ Dijkstra, thua dưới evaluator nhiễu thực |

Hệ quả cho novelty: "Huang + Si" tự nó **không mới** — Li et al. 2026 đã ghép mode Huang với quỹ đạo UAV + DSAC.
Khoảng trống quan sát được (chưa phải contribution, cần kết quả để xác nhận):
1. Không bài nào trong nhóm RL dùng chất lượng **đo theo từng ảnh** của mode generative; tất cả dùng bảng cố định
   hoặc hàm khớp. Truong đo theo ảnh nhưng chỉ cho learned codec và không có RL/mobility.
2. Không bài nào so với **greedy / fixed-mode / learned codec** mạnh; mode bit luôn là JPEG/JPEG2000.
3. Không bài nào đưa độ trễ giải mã diffusion vào T_k (Li bỏ qua; Truong "ignore processing time").

## 1. A1 — bảng mode thật theo từng ảnh

**Thiết lập.**
- Ảnh: 100 ảnh VisDrone2019-DET-val, lấy cố định (sắp theo tên, cứ 5 ảnh lấy 1). Resize Lanczos rồi cắt giữa 512×512, giống kích thước ảnh của Huang.
- Mỗi cặp (ảnh, mode) đều encode thật ra byte rồi decode thật. Đo PSNR, MS-SSIM, LPIPS (AlexNet) và CLIP (ViT-B-32; CLIP chỉ là chỉ số phụ), cùng thời gian encode và decode.
- Phần cứng: RTX 3070 và i5-14400F. Learned codec còn được đo riêng trên CPU, coi như UAV không có GPU.
- Code: `swarmbridge/experiments/uav_semcom/{imcodecs,metrics,profile_a11,profile_a12,summarize_a1}.py`.
- Dữ liệu: `runs/uav_semcom/{a11,a12}/profile.jsonl`; bảng đầy đủ ở `runs/uav_semcom/a1_summary.md`.

**Các mode.**
- A1.1, codec cổ điển (Pillow): JPEG, JPEG2000 (codestream J2K thô, chế độ rate) và AVIF (libavif), mỗi loại 6 mức.
- A1.1, learned codec: CompressAI `bmshj2018-hyperprior` và `mbt2018-mean`, q1–8. Bitstream rANS thật, payload gồm cả 13 byte header.
- A1.2, generative:
  - `cap_t2i`: caption BLIP-large (byte UTF-8) → SD-Turbo text-to-image, 4 bước. Tương ứng mode 2 của Huang.
  - `capthumb{S}_i2i_t{T}`: caption + thumbnail S×S nén AVIF → SD-Turbo img2img với strength T. Bản thay thế cho mode 3 của Huang, vì mô hình VQ + diffusion của Huang không công bố.
  - `thumb{S}_up`: thumbnail phóng to bicubic, làm đối chứng không sinh ảnh.
  - Mọi mode generative đều có decoder thật dùng đúng các byte đã truyền cộng một seed dùng chung.

**Kết quả chính.** Trung bình trên 100 ảnh; trong ngoặc là p10–p90.

| mode | KB | PSNR dB | MS-SSIM | LPIPS ↓ | enc / dec (ms) |
|---|---|---|---|---|---|
| cap_t2i | 0.1 | 9.7 [8.2–10.9] | 0.06 | 0.84 | 228 / 303 (GPU) |
| thumb64_up | 0.8 | 22.0 | 0.75 | 0.82 | 8 / 3 |
| cap_thumb64_i2i_t0.6 | 0.9 | 19.1 | 0.56 | 0.78 | 236 / 292 (GPU) |
| thumb128_up | 2.1 | 24.1 | 0.87 | 0.62 | 18 / 3 |
| cap_thumb128_i2i_t0.3 | 2.2 | 21.9 | 0.76 | 0.64 | 247 / 248 (GPU) |
| jpeg2000_400 | 2.0 | 22.8 | 0.76 | 0.71 | 66 / 5 |
| avif_5 | 2.9 [2.4–3.5] | 25.3 [22.8–27.6] | 0.90 | 0.44 | 38 / 3 |
| mbt2018-mean_q1 | 4.5 [2.8–6.5] | 27.4 | 0.93 | 0.36 | 19 / 24 GPU; 125 / 152 CPU |
| jpeg2000_25 | 31.4 | 30.7 [26.7–34.9] | 0.96 | 0.19 | 65 / 9 |
| mbt2018-mean_q8 | 52.8 [35–73] | 41.5 | 0.997 | 0.03 | 29 / 38 GPU; 236 / 272 CPU |
| avif_80 | 53.1 | 40.2 | 0.996 | 0.011 | 101 / 6 |

Đọc kết quả:
1. **Đối chiếu với Huang.**
   - JPEG2000 31 KB cho 30,7 dB, khớp với M1 của Huang (39 KB, 32,2 dB).
   - `cap_t2i` cho 9,7 dB, khớp với M2 của Huang (11,3 dB).
   - Ngay theo số của chính Huang, M3_3 (4,1 KB) đạt 23,2 dB, thấp hơn `avif_5` (2,9 KB, 25,3 dB) và `mbt q1` (4,5 KB, 27,4 dB).
   - Vậy mode bit trong Huang và Li (JPEG/JPEG2000) là baseline yếu. Thay bằng AVIF hay learned codec thì mode semantic thua ở mọi mức bit từ khoảng 2 KB trở lên.
2. **Mode generative** chỉ có chỗ dưới khoảng 1–2 KB, và chỉ khi đo bằng LPIPS:
   - Ở 1 KB, `cap_thumb64_i2i` thắng `thumb64_up` về LPIPS (0,78 so với 0,82) nhưng thua 3 dB PSNR.
   - Ở 2 KB, img2img thua thumbnail phóng to trên cả PSNR, LPIPS và CLIP.
   - Kiểm tra bằng mắt (`runs/uav_semcom/a1_grid.png`): ảnh sinh **bịa ra xe và đám đông ở vị trí sai**. Với ảnh giám sát từ UAV, đây là lỗi nội dung, không chỉ là lỗi thẩm mỹ.
3. **Độ trễ sinh ảnh** khoảng 250–300 ms/ảnh trên RTX 3070, dù SD-Turbo chỉ chạy 4 bước. SD 50 bước sẽ mất cỡ vài giây. Huang, Li và Truong đều không tính khoản này.
4. **Phụ thuộc vào ảnh: có.**
   - Số byte (codec VBR) dao động khoảng 2× giữa p10 và p90. Với JPEG2000 ở bitrate cố định, PSNR dao động 22–31 dB.
   - Ở cùng ngân sách byte, mode tốt nhất khác nhau tuỳ ảnh: 6–11 mode thắng khác nhau, mode thắng phổ biến nhất chỉ chiếm 34–64% số ảnh.
   - Nhưng chênh lệch giữa "mode tốt nhất cho từng ảnh" và "mode cố định tốt nhất" nhỏ, khoảng ±1 dB PSNR hoặc 0,02 LPIPS. Xem `a1_summary.md`, mục "Does the best mode depend on the image?".

Giới hạn của A1:
- SD-Turbo và BLIP không được huấn luyện trên ảnh UAV.
- Chưa thử các codec diffusion chuyên dụng như PerCo, DiffEIC hay CDC. Chúng có thể thu hẹp khoảng cách ở bitrate cực thấp.
- Các mode thumbnail được ghi `device=cuda` trong file kết quả, nhưng thực tế chạy trên CPU.
- Ảnh 512×512, chưa phải độ phân giải gốc.

## 2. A2 — môi trường (topology khớp dữ liệu)

UAV tự chụp ảnh nên topology là **camera trên UAV → encoder trên UAV → kênh không đối đất → BS mặt đất**. Không có vệ tinh, relay hay NOMA.

Code ở `swarmbridge/experiments/uav_semcom/env.py`, có 4 test. Các cơ chế trong môi trường:

- **Nhiệm vụ.**
  - 3 UAV, mỗi chiếc khảo sát một khu cách BS 1–2,5 km theo 6 waypoint kiểu lawn-mower.
  - Mỗi waypoint chụp 4 tile 512². Mỗi tile là một ảnh VisDrone thật lấy từ tập test (chia 70/30).
  - Waypoint không kịp tới trước khi hết episode thì mỗi ảnh của nó tính −1.
- **Kênh.**
  - Xác suất LoS theo góc ngẩng (Al-Hourani, đô thị), cộng FSPL và excess loss 1/20 dB.
  - Fading Rician K=10 dB khi LoS, Rayleigh khi NLoS, rút lại mỗi slot.
  - 1 MHz chia làm M=2 kênh trực giao cho K=3 UAV; UAV dùng chung kênh thì gây nhiễu cho nhau.
- **Hành động mỗi UAV mỗi slot (1 s).**
  - Bay: đứng yên, 8 hướng ở 20 m/s, hoặc 2 hành động vĩ mô "tới waypoint kế" và "về BS".
  - Kênh: im lặng hoặc 1..M.
  - Công suất: 3 mức (0,02 / 0,08 / 0,2 W).
  - Mode cho ảnh đầu hàng đợi: 14 mode lấy từ A1.
- **Năng lượng.** Công suất đẩy rotary-wing (Zeng et al. 2019) cộng năng lượng phát, có trọng số trong reward.
- **Utility mỗi ảnh** giống SCE của Huang/Li: ξq·clip((PSNR−Qmin)/15) + ξd·(Dmax−delay)/Dmax, trừ 1 nếu vi phạm.
  - Qmin ~ U[22,30] dB, Dmax ~ U[4,12] s.
  - Delay gồm cả thời gian encode trên UAV (CPU) và decode ở BS.

## 3. A3 — PPO và baseline trên cùng môi trường (seed train 0, 30 episode test, ảnh test chưa từng thấy)

PPO (SB3): 1 controller tập trung, action space MultiDiscrete, MLP 256×256, 3 triệu bước, khoảng 26 phút trên CPU.

| chính sách | utility / ảnh | tỉ lệ đạt | ghi chú |
|---|---|---|---|
| fixed cap_t2i (≈ Huang M2) | −1.23 | 0.00 | caption chỉ đạt khoảng 10 dB, luôn dưới Qmin |
| fixed cap_thumb128_i2i | −0.84 | 0.08 | |
| fixed jpeg2000_25 (≈ Huang M1) | +0.07 | 0.74 | |
| fixed avif_80 / mbt q5 | +0.31 / +0.30 | 0.78 / 0.84 | |
| greedy (mode theo từng ảnh) | +0.31 | 0.78 | |
| **PPO** | **+0.645** | **0.95** | |
| **paced_q0** (heuristic mới) | **+0.624** | 0.93 | |

**PPO có lợi thế ở đâu.** Phân tích hành vi trên 10 episode (`runs/uav_semcom/a3_s0/analysis.json`):
- PPO **giãn nhịp nhiệm vụ**: lần chụp cuối ở giây 59,6, so với 37 s của greedy.
- Nhờ vậy hàng đợi ngắn hơn (0,37 so với 0,58) và độ trễ thấp hơn (3,5 so với 5,2 s).
- Mode PPO chọn gần như giống greedy (avif_80 hoặc mbt q8).

Heuristic `paced_q0` làm đúng việc đó: chọn mode greedy, nhưng đứng chờ tới khi hàng đợi trống mới bay tiếp, miễn còn đủ thời gian.

| | utility | PPO − chính sách này (khoảng tin cậy 95% ghép cặp, 30 ep) |
|---|---|---|
| greedy | +0.308 | +0.336 [+0.310, +0.362] |
| paced_q0 | +0.624 | +0.020 [+0.001, +0.040] |
| paced_q0, chọn mode theo bảng trung bình (không theo từng ảnh) | +0.618 | +0.027 [+0.008, +0.045] |

Đọc kết quả:
1. Cơ chế dài hạn có thật trong môi trường này là **nhịp chụp ↔ hàng đợi**: bay nhanh làm ảnh dồn ứ.
2. Một heuristic khoảng 20 dòng lấy lại khoảng 94% lợi thế của PPO. PPO còn dư +0,02, có ý nghĩa thống kê nhưng mới có 1 seed train.
3. Thông tin riêng từng ảnh chỉ thêm khoảng +0,006.
4. Lợi thế này tồn tại vì môi trường **không tính chi phí thời gian nhiệm vụ**: 200 s cho một nhiệm vụ khoảng 40 s. Si et al. tối thiểu hoá chính thời gian này.


### 3.1 Thêm seed, thêm chi phí thời gian, và baseline được tinh chỉnh (2026-11-06)

Lần chạy mới:
- 2 seed PPO nữa ở w_time = 0.
- PPO với chi phí thời gian nhiệm vụ w_time = 0,1 và 0,3 (reward −w_time mỗi giây, mỗi UAV chưa chụp xong).
- Mỗi lần 3 triệu bước, chỉ CPU.

So sánh bằng **return của episode**, chính là đại lượng PPO tối đa hoá: tổng utility, trừ năng lượng, trừ thời gian.

**Baseline công bằng.** Họ heuristic được tinh chỉnh trên **ảnh train và 30 seed train** (0..29, không trùng seed test 1000..1029). Chọn thành viên có return train cao nhất, rồi so với PPO trên 30 seed test. Họ heuristic gồm:
- greedy theo ảnh, hoặc greedy *biết hàng đợi* (tối đa utility cả hàng đợi, không chỉ ảnh đầu);
- điều nhịp theo ngưỡng hàng đợi q ∈ {0..5};
- điều nhịp theo thời gian xả hàng đợi dự đoán τ ∈ {0,5..6} s, có tính tốc độ link;
- **chia lượt theo số kênh** (`staggered`, thêm sau khi phân tích PPO, xem bên dưới).

Code: `tune_heur.py`, `ablate_policy.py`, `policies.py`. Kết quả: `runs/uav_semcom/a3_*/tune.json`, `ablate.json`.

**Bước 1. Chỉ dùng họ điều nhịp (chưa có `staggered`).** PPO vẫn hơn ở cả 3 cấu hình:

| cấu hình | heuristic tốt nhất (chọn trên train) | PPO − heuristic, return [KTC 95%] | PPO − heuristic, utility |
|---|---|---|---|
| w_time 0, seed 1 | pacedT0.5 | +2,77 [+1,30, +4,24] | +0,039 |
| w_time 0,1 | paced_q2_qa | +3,93 [+2,50, +5,36] | +0,042 |
| w_time 0,3 | paced_q3_qa | +3,18 [+1,30, +5,07] | +0,060 |

Greedy biết hàng đợi không giúp gì (chênh dưới 0,01).

**Bước 2. Tách thành phần (ablation).** Mỗi phần hành động {di chuyển, kênh + công suất, mode} lấy từ PPO (P) hoặc từ heuristic (H). Đo trên 30 seed test:

| di chuyển / kênh+công suất / mode | w_time 0,3 (H = paced_q3) | w_time 0, seed 1 (H = paced_q0) |
|---|---|---|
| PPP (PPO) | +2,23 | +45,05 |
| **PHH** | **+2,74** | **+46,04** |
| HPP | −0,33 | +41,46 |
| HHH (heuristic) | −0,26 | +42,90 |

Toàn bộ lợi thế nằm ở **di chuyển**. Mode và kênh của PPO còn kém greedy một chút: PHH tốt hơn PPP.

**Bước 3. PPO di chuyển thế nào.** Thời điểm chụp waypoint (giây) của 3 UAV, seed 1000, PPO seed 1:
- PPO: UAV 1 và 2 chụp ở `[10, 15, 21, 26, 31, 37]`, bay hết tốc độ; UAV 0 **chờ ở điểm xuất phát** rồi chụp ở `[42, 52, 58, 64, 69, 75]`.
- paced_q0: cả 3 UAV cùng chụp ở khoảng `[10, 20, 31, 40, 50, 60]`.

Hệ quả:
- P(có hơn M = 2 UAV cùng có hàng đợi): PPO 0,1–1,8%, heuristic 5–8%.
- P(một UAV khác chụp trong vòng 3 s): PPO 0,68–0,72, heuristic 0,93–0,98.

PPO học được luật: **chỉ M UAV khảo sát cùng lúc, mỗi UAV gần như có một kênh riêng; UAV thứ K chờ tới lượt.** Đây là luật phối hợp truy cập kênh nhiều UAV (K = 3 > M = 2). Nó không phải lựa chọn semantic.

**Bước 4. Viết luật đó thành heuristic (`staggered`).**
- Tối đa M UAV "đang bận", tức đã bắt đầu và chưa (chụp xong và xả xong hàng đợi).
- UAV chưa tới lượt đứng chờ, nhưng chỉ khi vẫn còn đủ thời gian cho cả nhiệm vụ.
- Thứ tự vào lượt: idx / near / far. Có thể kèm điều nhịp q.

Tinh chỉnh trên train như trên:

| cấu hình | heuristic chọn trên train | return heuristic | return PPO | PPO − heuristic [KTC 95%] |
|---|---|---|---|---|
| w_time 0, seed 1 | stag_far_q0 | +46,75 | +45,05 | −1,71 [−2,94, −0,47] |
| w_time 0, seed 2 | stag_far_q0 | +46,75 | +44,80 | −1,96 [−3,31, −0,61] |
| w_time 0,1 | stag_idx | +31,20 | +29,94 | −1,25 [−2,47, −0,04] |
| w_time 0,3 | stag_idx | +1,34 | +2,23 | +0,89 [−0,30, +2,08] (không có ý nghĩa) |

**Kết luận A3 trên phiên bản môi trường này:**
1. Heuristic chia lượt (khoảng 30 dòng, tinh chỉnh 1 tham số rời rạc trên train) **bằng hoặc hơn** PPO ở cả 4 lần chạy. Lợi thế của PPO đã được giải thích hết.
2. PPO có ích như một **công cụ khám phá**: nó tìm ra luật chia lượt, mà luật này không có trong họ heuristic ban đầu. Nhưng ở đây nó không phải lời giải tốt nhất.
3. Phần semantic (chọn mode) **không** là nơi tạo khác biệt:
   - greedy theo ảnh đã đủ;
   - mode sinh (caption + diffusion) không bao giờ được chọn, vì với utility tính theo PSNR chúng bị AVIF / learned codec áp đảo ở mọi bitrate (bảng A1).
4. Hệ quả cho hướng A: muốn mode semantic có vai trò thì thước đo chất lượng phải là **nhiệm vụ của người nhận**, không phải PSNR. Ví dụ: phát hiện hoặc đếm đối tượng trên ảnh nhận được (VisDrone có nhãn box). Lúc đó mới biết mode sinh có lúc nào đáng chọn hay không. Ngoài ra, phối hợp truy cập nhiều UAV (K > M) là cơ chế thật của môi trường này và đáng giữ.
