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

> Mục này mô tả env **v1**. v1 có lỗi (mục 3.2). Env **v2** và MDP đầy đủ ở **mục 4**.

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

> **KHÔNG CÒN HIỆU LỰC.** Mọi kết quả ở mục 3 và 3.1 chạy trên env v1, vốn có lỗi "1 ảnh/slot" (mục 3.2). PPO chưa được train lại trên env v2.

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

> **KHÔNG CÒN HIỆU LỰC** (env v1, xem 3.2).

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

### 3.2 ĐÍNH CHÍNH (2026-11-06): kết luận A3 / A3.1 là hệ quả của một lỗi mô phỏng

**Lỗi.** `env.step` phục vụ **tối đa 1 ảnh / UAV / slot 1 s**. Gửi xong ảnh đầu hàng đợi thì phần còn lại của slot bị bỏ phí, và ảnh tiếp theo chỉ bắt đầu encode ở slot sau. ChatGPT phát hiện lỗi này khi đọc code. Tôi đã kiểm tra lại và xác nhận.

Trong khi đó, link thực ra dư rất nhiều. Tốc độ kỳ vọng ở các waypoint là trung vị 2,2 Mb/s (p10 1,8, p90 2,9). Thời gian phát của mode tốt nhất:

| mode | kbit | thời gian phát (s) | PSNR |
|---|---|---|---|
| avif_80 | 418 | 0,19 | 40,3 |
| mbt q8 | 415 | 0,19 | 41,5 |
| jpeg2000_25 | 251 | 0,11 | 30,8 |
| avif_5 | 23 | 0,01 | 25,3 |

Như vậy nút thắt là **số slot**, không phải số bit. Greedy bay 20 m/s, cứ khoảng 6 s mỗi UAV chụp 4 ảnh. 3 UAV tạo ra 2 ảnh/s, đúng bằng sức chứa 2 kênh × 1 ảnh/slot, nên hàng đợi tắc.

**Kiểm tra.** `check_multipkt.py` dùng env cho phép phục vụ tiếp hàng đợi trong cùng slot (encode ảnh kế tiếp ngay sau ảnh trước). Lần chạy này cũng lấy gain nhiễu từ đúng mẫu fading của UAV gây nhiễu. Kết quả trên 30 seed test, chỉ heuristic, không train lại:

| chính sách | env cũ: return / ok | env sửa: return / ok | env cũ, w_time 0,3: return | env sửa, w_time 0,3: return |
|---|---|---|---|---|
| greedy_img | +20,2 / 0,78 | +48,7 / 0,97 | −12,3 | +16,3 |
| paced_q0 | +42,9 / 0,93 | +53,3 / 0,99 | −12,5 | +7,5 |
| paced_q3 | +40,1 / 0,92 | +53,2 / 0,99 | −0,3 | +14,7 |
| stag_idx | +46,1 / 0,95 | +54,3 / 0,99 | +1,3 | +10,1 |
| fixed avif_80 | +20,0 / 0,78 | +50,3 / 0,98 | −12,4 | **+17,9** |

**Hệ quả:**
- Câu chuyện "hàng đợi tắc → điều nhịp → PPO học chia lượt theo số kênh" ở mục 3 và 3.1 là **sản phẩm của lỗi này**, không phải tính chất của hệ thống.
- Khi đã sửa và có phạt thời gian, một mode cố định (avif_80) là tốt nhất; điều nhịp và chia lượt còn làm hại.
- Sau khi sửa, môi trường với tham số hiện tại gần như **không còn trade-off**: link dư khoảng 5 lần so với mode chất lượng cao nhất. Vì vậy chọn mode, công suất hay quỹ đạo đều ít quan trọng.
- Docstring "calibrated so that payloads matter" của `env.py` là sai.

**Các vấn đề khác đã xác nhận trong code** (theo rà soát của ChatGPT, tôi kiểm tra lại từng điểm):
- PPO không quan sát thời gian `t`, trong khi heuristic dùng `env.t`.
- Một RNG chung cho cả task và fading, nên các chính sách khác nhau gặp chuỗi ảnh và deadline khác nhau. KTC ghép cặp vẫn hợp lệ thống kê, nhưng mất lợi ích giảm phương sai.
- Nhiễu dùng một mẫu fading khác với link của chính UAV gây nhiễu.
- Không có giới hạn pin. Hover 168,5 W, mà 40 s hover chỉ bị phạt khoảng 0,13.
- UAV tiêu tốn năng lượng tới T = 200 s, nên năng lượng gần như bằng nhau ở mọi chính sách.
- Công suất phát không có trade-off: năng lượng phát khoảng 0,2 W × 0,2 s, không đáng kể.
- Không có action masking.
- Quan sát chứa bits/PSNR của mọi mode cho ảnh đầu hàng đợi. UAV có thể tự tính được, vì nó giữ ảnh gốc, nhưng phải encode và decode đủ 14 mode, và chi phí đó đang không được tính. Phải khai báo rõ đây là giả định.

## 4. A2 v2 — sửa simulator (Bước 1) và system model + MDP (Bước 2), 2026-11-07

**Trạng thái:**
- Env v2 đã viết xong và có 19 unit test, tất cả đều qua.
- Phân tích chế độ tải đã xong.
- Đã chạy kiểm tra các heuristic, không học gì.
- **PPO chưa được train lại**, chờ review mục này.

Code: `swarmbridge/experiments/uav_semcom/` — `env.py` (môi trường), `gym_env.py` (wrapper và action mask), `policies.py` (heuristic), `test_env.py`, `regime.py`, `sanity_v2.py`, `regime_native_probe.py`.

Kết quả: `runs/uav_semcom/regime_v2.md`, `runs/uav_semcom/v2_sanity/`, `runs/uav_semcom/regime_native_probe.jsonl`.

### 4.1 Lỗi đã sửa và test tương ứng

| # | Lỗi | Cách sửa trong v2 | Test kiểm tra |
|---|---|---|---|
| 1 | Mỗi UAV chỉ phục vụ được 1 ảnh/slot (v1) | Bộ phát phục vụ hàng đợi FIFO liên tục trong slot; encoder chạy song song với bộ phát (encoder nối tiếp → bộ phát nối tiếp) | `test_several_images_per_slot`, `test_bits_time_energy_conservation` |
| 2 | Bản sửa thử `check_multipkt.py` dùng mode của ảnh đầu cho cả các ảnh sau trong cùng slot (ChatGPT phát hiện) | **Một quyết định mode ứng với đúng một ảnh**: action có một vector mode, mỗi phần tử cho một ảnh đang chờ | `test_mode_vector_applies_per_image`, `test_mode_vector_length_enforced` |
| 3 | Gain nhiễu lấy một mẫu fading khác | Mỗi UAV một gain mỗi slot, dùng cả cho link riêng lẫn khi gây nhiễu | `test_interference_uses_same_gain_as_own_link` |
| 4 | Một RNG chung, nên các chính sách khác nhau gặp kịch bản khác nhau | Danh sách ảnh (ảnh, deadline, Qmin theo UAV × waypoint × ảnh) và biến ngẫu nhiên fading (theo slot × UAV) được rút trước lúc reset, từ các stream riêng | `test_scenario_independent_of_policy` |
| 5 | Năng lượng phát tính p × cả phần còn lại của slot; UAV tiêu năng lượng tới 200 s | Năng lượng phát = p × thời gian phát thực; hết sortie (khảo sát xong và hàng đợi rỗng) thì UAV ngừng tiêu năng lượng và ngừng chịu phí thời gian | `test_bits_time_energy_conservation`, `test_no_energy_after_sortie_end` |
| 6 | Ảnh quá hạn vẫn được phát tới cuối slot | Deadline là **hạn cứng**: hết hạn (kể cả thời gian decode) thì ảnh bị bỏ ngay, không tốn thêm thời gian phát | `test_deadline_expiry_spends_no_airtime` |
| 7 | Ngân sách thông tin lệch: PPO không thấy thời gian; bit/PSNR mặc định là oracle | Observation có thời gian, quãng đường khảo sát còn lại và chi tiết hàng đợi. Mức thông tin về mode chọn bằng cấu hình `mode_info` (oracle / predicted / mean); heuristic đọc qua đúng hàm `env.mode_info` đó | `test_observation_contains_time_and_remaining_path`, `test_mode_info_levels` |
| 8 | Không có action mask | `GymUavSemcom.action_masks()` | `test_action_masks_shape` |
| 9 | Không có giới hạn vùng bay và pin | `area_radius` 3 km (cứng), `battery_j` (cứng; với nhiệm vụ này không bao giờ chạm, xem 4.6) | (chạy trong mọi test episode) |

Ghi chú: bảng 3.2 (fixed avif_80 > greedy) chạy bằng `check_multipkt.py`, nên dính lỗi #2. Không dùng bảng đó để đánh giá cách chọn mode.

### 4.2 System model

- **Thực thể.**
  - 1 BS ở gốc toạ độ, cao 25 m.
  - K UAV bay ở độ cao cố định H. Mỗi UAV có một khu khảo sát riêng, cách BS 1–2,5 km: W = 6 waypoint kiểu lawn-mower, xuất phát cách hàng đầu tiên 1,5 khoảng waypoint.
  - Các khu nằm cách nhau ít nhất khoảng 60° theo góc. Va chạm không được mô hình hoá.
- **Dữ liệu.**
  - Tới mỗi waypoint, UAV chụp L = 4 ảnh. Mỗi ảnh là một ảnh VisDrone-DET-val đã tiền xử lý như A1: thu cạnh ngắn về 512, cắt giữa 512×512.
  - **Phạm vi:** ảnh được rút ngẫu nhiên từ tập test; nội dung ảnh **không** phụ thuộc vị trí UAV. Đây là mô phỏng tải giao ảnh, **không** phải bài toán active sensing.
  - Ảnh 512² là bản thu nhỏ rồi cắt của một khung, không phải một tile ở độ phân giải gốc. Con số "4 ảnh / waypoint" là giả định.
- **Trên UAV.** Một encoder nối tiếp, mã hoá từng ảnh một. Độ trễ encode đo ở A1:
  - codec Pillow và learned codec: CPU i5-14400F;
  - caption BLIP và các mode sinh: GPU RTX 3070, mạnh hơn nhiều so với máy trên UAV (giả định cần review).
- **Link.**
  - Uplink UAV → BS, M kênh trực giao, mỗi kênh B/M.
  - Fading khối theo slot. SINR có nhiễu đồng kênh. Tốc độ Shannon (mã hoá lý tưởng, không có lỗi gói).
  - Bộ điều khiển chỉ biết CSI thống kê (theo vị trí), không biết fading của slot.
- **BS.** Giải mã. Độ trễ decode đo ở A1; decoder sinh chạy trên RTX 3070.

### 4.3 Thời gian và thời điểm quyết định

- Slot dt = 1 s, horizon T = 200 s. Episode kết thúc khi mọi sortie xong hoặc t = T.
- **Theo slot** (áp dụng cho cả slot): hướng bay, kênh, mức công suất.
- **Theo ảnh**:
  - Mode được chọn **đúng một lần cho mỗi ảnh**, ở thời điểm quyết định đầu tiên sau khi chụp.
  - Phần tử i của vector mode áp cho ảnh thứ i chưa có mode, theo thứ tự chụp.
  - Mỗi lần quyết định có tối đa L ảnh đang chờ, vì khoảng waypoint 120 m > 2·(15 + 20) m. Điều này có `assert` trong code.
- Vị trí dùng để tính kênh là vị trí đầu slot (UAV coi như đứng yên trong slot, như Si et al.). Bay và chụp được cập nhật cuối slot.

### 4.4 MDP

**Trạng thái đầy đủ s_t** (`UavSemcomEnv`, `UAV`, `Task`):
- Thời gian t.
- Mỗi UAV k:
  - vị trí x_k và chỉ số waypoint tiếp theo (danh sách waypoint cố định trong episode);
  - hàng đợi các ảnh (t_cap, Dmax, Qmin, ảnh, mode, thời điểm encode xong, số bit còn lại);
  - thời điểm encoder rảnh E_k, năng lượng đã dùng, cờ sortie còn chạy.
- Phần ngoại sinh: các ảnh sẽ chụp ở những waypoint sau, và các biến ngẫu nhiên fading.

**Observation o_t** (`env.obs()`, 514 chiều; là thứ bộ điều khiển thật sự biết):
- Toàn cục: t/T.
- Mỗi UAV:
  - còn chạy hay không; vị trí; vector tới waypoint kế;
  - tỉ lệ waypoint còn lại; quãng đường khảo sát còn lại;
  - SNR kỳ vọng tại vị trí; năng lượng đã dùng;
  - độ dài hàng đợi; thời gian encoder còn bận.
- Mỗi ảnh trong 6 ảnh đầu hàng đợi: slack tới deadline, Qmin, đã có mode chưa, số bit (chính xác nếu đã encode xong, nếu chưa thì lấy theo `mode_info`), thời gian còn lại tới khi encode xong.
- Mỗi ảnh đang chờ quyết định mode (tối đa L): slack, Qmin, và **bit, PSNR của từng mode theo `mode_info`**.
- Bộ điều khiển **không** thấy: ảnh tương lai, fading của slot, và bit/PSNR thật (trừ khi `mode_info = oracle`).

Ba mức `mode_info` (`ModeTable.fit_info`, `predict`):
- **oracle**: bảng đo của chính ảnh đó. Đây là giả định "biết trên mức triển khai được": UAV giữ ảnh gốc nên tính được, nhưng phải encode và decode đủ 14 mode; chi phí đó không được tính.
- **predicted**: UAV encode thử bằng avif_5 (tốn khoảng 0,04 s, **có tính** vào thời gian encoder). Sau đó dự đoán log bit và PSNR của từng mode bằng hồi quy tuyến tính fit trên ảnh train. Trên ảnh test: R² của log bit khoảng 0,6–0,9 với AVIF, CompressAI và thumbnail; R² của PSNR khoảng 0,6–0,8; sai số tuyệt đối PSNR 0,5–1,7 dB (bảng trong `regime_v2.md`).
- **mean**: trung bình theo mode trên ảnh train, không có thông tin riêng từng ảnh.

**Action a_t** (`step`, `gym_env.decode_action`). Với mỗi UAV k:

$$a_t^k = \big(m_k \in \{0..10\},\; c_k \in \{0..M\},\; \ell_k \in \{0,1,2\},\; (\mu_{k,1},\dots,\mu_{k,L}) \in \{0..13\}^L\big)$$

- m: hover, 8 hướng ở v_max, hành động vĩ mô "tới waypoint kế", hành động vĩ mô "về BS".
- c: kênh (0 = im lặng). ℓ: mức công suất p ∈ {0,02; 0,08; 0,2} W. μ: mode cho từng ảnh đang chờ.
- MultiDiscrete gồm 21 thành phần. Mask loại bỏ: hành động khi UAV đã hết sortie; macro 9 khi đã khảo sát xong; kênh khi hàng đợi rỗng; các μ thừa.

**Chuyển trạng thái** (`step`, các bước (1)–(7) theo đúng thứ tự trong code):
1. Ảnh i chờ mode nhận μ_i.
   - Encode bắt đầu ở e_i = max(t, E_k), xong ở r_i = e_i + τ_probe + τ_enc(μ_i); sau đó E_k ← r_i.
   - Payload b_i = bits(ảnh, μ_i), lấy từ bảng đo.
2. Gain của slot.
   - LoS với xác suất P_LoS(θ) (Al-Hourani): β_L |h|², h ~ Rician(K).
   - Ngược lại: β_N X, X ~ Exp(1). β gồm FSPL cộng excess loss η.
3. Phát trong slot.
   - Tập UAV đang phát TX = {k : c_k > 0, ảnh đầu hàng đợi encode xong trước t + dt}.
   - $R_k = \frac{B}{M}\log_2\!\big(1 + \frac{p_k g_k}{N_0 B/M + \sum_{j\in TX, j\ne k, c_j=c_k} p_j g_j}\big)$. Nhiễu tính cho cả slot (giả định bảo thủ).
   - Ảnh được phát theo FIFO, ảnh i bắt đầu ở max(lúc ảnh trước phát xong, r_i).
   - Giao xong ở thời điểm end: delay D_i = end + τ_dec − t_cap.
4. Hết hạn: nếu end + τ_dec > t_cap + Dmax thì ảnh bị bỏ ngay ở thời điểm đó.
5. Bay.
   - x ← x + Δ với |Δ| ≤ v_max·dt, rồi kẹp trong bán kính 3 km.
   - Năng lượng: E += P_prop(|v|)·dt + p·airtime (P_prop theo Zeng 2019).
   - Cách waypoint kế ≤ 15 + v_max·dt/2 m thì chụp L ảnh mới, t_cap = t + dt.
6. Khảo sát xong và hàng đợi rỗng thì sortie kết thúc. Hết pin thì sortie bị huỷ, các ảnh còn lại tính thất bại.
7. Trừ chi phí năng lượng và thời gian.

**Reward:**

$$r_t = \sum_{i \in \text{xong trong slot}} U_i \;-\; w_E\,E_{\text{slot}}[\text{kJ}] \;-\; w_T\,dt\cdot\#\{\text{UAV còn sortie}\}$$

trong đó $U_i = \xi_q\,\mathrm{clip}\big(\frac{Q_i - Q_{\min}}{15}, -1, 1\big) + \xi_d \frac{\max(D_{\max} - D_i, 0)}{D_{\max}} - \mathbb{1}[\text{vi phạm}]$.
- Ảnh bị bỏ, hoặc waypoint chưa chụp khi hết horizon: −1 mỗi ảnh.
- Mục tiêu: max E[Σ_t r_t] (episode hữu hạn; PPO dùng γ = 0,99).

### 4.5 Ràng buộc

| loại | ràng buộc | thực thi |
|---|---|---|
| cứng | tốc độ ≤ v_max | step (5) |
| cứng | trong bán kính 3 km quanh BS | step (5) |
| cứng | công suất ∈ {0,02; 0,08; 0,2} W ≤ p_max | không gian action |
| cứng | mỗi UAV ≤ 1 kênh; M kênh, tổng băng thông B | không gian action, công thức tốc độ |
| cứng | encoder nối tiếp; bộ phát FIFO; bộ phát không chạy trước khi encode xong | step (1), (3) |
| cứng | deadline: ảnh quá hạn bị bỏ, coi như thất bại | step (3), (4) |
| cứng | pin: hết pin thì huỷ sortie | step (6) |
| mềm (phạt) | PSNR ≥ Qmin: vi phạm bị −1 trong U_i (cách của Huang/Li) | `utility` |
| mềm | năng lượng (w_E), thời gian nhiệm vụ (w_T) | step (7) |
| không mô hình | tránh va chạm, vùng cấm bay, lỗi gói / truyền lại, ngưỡng SINR, ngân sách tính toán / nhiệt, tổng công suất nhiều UAV, chặng bay về | — |

### 4.6 Nguồn tham số

| tham số | giá trị | loại | nguồn / ghi chú |
|---|---|---|---|
| K UAV, M kênh | 3, 2 | giả định | Huang dùng N = 3 UAV |
| B_total | 1 MHz | **giả định** | các paper dùng: Si 5 MHz, Huang 10 MHz, Li 20 MHz, Truong 100 MHz |
| fc | 2,4 GHz | giả định | |
| nhiễu | −174 dBm/Hz + NF 7 dB | chuẩn / giả định NF | |
| p_max | 0,2 W | paper | Huang: UAV phát 0,2 W |
| LoS đô thị (a, b, η_LoS, η_NLoS) | 9,61; 0,16; 1 dB; 20 dB | paper | Al-Hourani et al. 2014 |
| Rician K | 10 dB | paper | Li |
| H | 100 m | giả định | dưới giới hạn 120 m; Huang và Li dùng 45–70 m |
| v_max, dt | 20 m/s, 1 s | paper | Li (Huang, Si: 10 m/s) |
| công suất đẩy | Zeng 2019, hover 168,5 W | paper | |
| pin | 321 kJ (89,2 Wh) | spec | pin cỡ Phantom 4 Pro (~1,4 kg, cùng hạng ~2 kg với mô hình Zeng: hover 168 W → ~32 phút). Sửa ngày 07/11 từ 1,97 MJ (pin M300, máy bay 6 kg, không khớp mô hình). Nhiệm vụ tham chiếu dùng 21–35 kJ (7–11%) nên **vẫn không chạm**; không thu nhỏ pin cho tới khi nó có tác dụng |
| khu khảo sát, waypoint | 1–2,5 km; 6 waypoint cách 120 m; 4 ảnh/waypoint | giả định | |
| Dmax | U(4, 12) s | giả định | gần Huang U(2, 11); Li U(0,5, 2) |
| Qmin | U(22, 30) dB | giả định | Huang U(10, 30); Li U(10, 25) |
| ξ_q, ξ_d | 0,5; 0,5 | paper | Huang |
| q_span, w_E, w_T | 15 dB; 0,02/kJ; 0 hoặc 0,3/s | giả định | trọng số thiết kế |
| bit, PSNR, τ_enc, τ_dec theo ảnh × mode | đo | **đo** | A1: 100 ảnh VisDrone 512²; CPU i5-14400F (codec), RTX 3070 (BLIP, SD-Turbo) |

### 4.7 Chế độ tải (offered load / service capacity)

Đầy đủ ở `runs/uav_semcom/regime_v2.md`.

Khi bay khảo sát hết tốc độ, mỗi UAV tạo L / (120 m / 20 m/s) = 0,67 ảnh/s; cả 3 UAV là 2 ảnh/s. Tốc độ mỗi kênh (công suất max, không nhiễu) có trung vị 2,25 Mb/s (p10 1,81).

| | ρ_link của mode chất lượng cao nhất (mbt q8 / avif_80, ~0,42 Mbit) |
|---|---|
| env hiện tại (1 MHz) | 0,18–0,19, thời gian phát 0,19 s/ảnh |
| băng thông của Si / Huang / Li (5 / 10 / 20 MHz) | 0,07 / 0,05 / 0,04 |
| khung gốc VisDrone (đo trên 20 ảnh, đa số 960×540 và 1360×765; avif_80 ≈ 1,0 Mbit, 44 dB), 1 khung/waypoint | 0,11 |
| tải encoder (mbt q8, CPU) | ρ_enc 0,16 |
| số UAV trên 2 kênh để mbt q8 làm đầy link | K ≈ 16 |

~~Các paper có trade-off về bit vì payload của họ lớn so với băng thông: Huang và Li dùng ảnh gốc 3,5 MB với 10–20 MHz.~~ **Câu này sai, rút lại** (xem 4.9).
- Hình 5 của Huang cho thấy payload thật sự truyền đi: Mode 1 (JPEG2000) **314 kb** cho một ảnh 512², 32,2 dB. Con số này cùng cỡ với payload của chúng ta.
- Truong: phương án không nén (UAV-WoSem) mất 2,4 s ở 100 MHz. Đây là baseline của họ, không phải payload của mode nén.

**Kiểm tra heuristic trên v2** (`runs/uav_semcom/v2_sanity/sanity_v2.md`, 30 seed test, không học):
- Với w_T = 0, các chính sách tốt đạt tỉ lệ đạt yêu cầu ≈ 1,00 và PSNR ≈ 41 dB:
  - greedy theo ảnh +61,2;
  - fixed mbt q8 +60,6;
  - paced / staggered +62,8 / +63,6. Chờ chỉ làm giảm thành phần delay (1,7 → 1,2 s).
- Với w_T = 0,3: greedy +25,4 là tốt nhất; chờ hay chia lượt gây hại (+16,6).
- Mức thông tin oracle / predicted / mean chỉ đổi return dưới 0,5.

**Đọc kết quả.** Với payload đã đo và tham số link có nguồn, việc giao ảnh tĩnh của 3 UAV **không bị giới hạn bởi băng thông**. Hệ quả:
- Chọn mode rút về "lấy mode chất lượng cao nhất", gần như không cần biết thông tin riêng từng ảnh.
- Bay, kênh và công suất còn rất ít thứ để tối ưu.

Đây là tính chất của kịch bản (chế độ tải), không phải kết luận về RL hay về SemCom.

### 4.8 Những điểm cần quyết khi review (trước khi train lại PPO)

Đây là các lựa chọn kịch bản cho **cùng hệ thống**, không phải hướng mới:

1. **Chế độ tải.** Giữ nguyên, hay chọn một kịch bản có tải cao vì lý do vật lý? Mỗi phương án đều phải đo hoặc có nguồn, không chỉnh để RL thắng:
   - nhiều UAV dùng chung ít kênh hơn, tức K/M lớn;
   - tần suất chụp theo yêu cầu chồng ảnh của photogrammetry, với khung gốc (phải đo đủ mode ở độ phân giải gốc);
   - khoảng cách xa hơn (BVLOS).
2. **Mức thông tin chính:** oracle (cận trên) hay predicted (triển khai được), hay báo cáo cả hai.
3. **Qmin và deadline:** giữ phạt mềm như Huang/Li, hay chuyển sang CMDP với ràng buộc cứng.
4. **Trọng số w_E, w_T:** cố định, hay quét để vẽ Pareto.
5. **Độ trễ mode sinh** đo trên RTX 3070: giữ kèm ghi chú, đo lại trên thiết bị nhúng, hay loại khỏi tập mode trên UAV.
6. **Thời điểm chọn mode:** ngay sau khi chụp (hiện tại), hay ngay trước khi encode (late binding).

### 4.9 Review vòng 1 (ChatGPT, 2026-11-07): đã xử lý gì

**Cấu hình nghiên cứu được chốt cho kịch bản tham chiếu:**

| vấn đề | chốt |
|---|---|
| thông tin codec | `predicted` là thiết lập chính; `oracle` là tham chiếu "biết trên mức triển khai được" |
| deadline / PSNR | deadline là hạn cứng (ảnh quá hạn bị bỏ); PSNR < Qmin bị phạt mềm (như v2) |
| năng lượng / thời gian | w_E = 0,02/kJ; w_T ∈ {0; 0,3} |
| độ trễ mode sinh | giữ số đo trên RTX 3070, ghi rõ đây là giả định về thiết bị tính toán, không phải máy tính phổ thông trên UAV |
| thời điểm chọn mode | mỗi ảnh chọn một lần, ngay sau khi chụp |
| kịch bản | v2 hiện tại là tham chiếu; **không** tăng K lên 16 chỉ để làm đầy link |

**1. Action mask giờ được dùng thật.**
- `train_a3.py` dùng `sb3_contrib.MaskablePPO`; `--no-mask` cho PPO thường để làm ablation. Lúc đánh giá cũng truyền mask.
- Test `test_maskable_ppo_consumes_masks` chạy 512 bước học: MaskablePPO lấy **0** sub-action bị mask. Cùng test đó với PPO thường: vi phạm ở **512/512** bước. Như vậy test đủ nhạy.
- 20/20 test qua.

**2. Pin.** Giá trị cũ 1,97 MJ là pin của M300 (máy bay 6 kg), không khớp mô hình công suất đẩy của Zeng (hạng ~2 kg, hover 168 W).
- Đã đổi sang 89,2 Wh = 321 kJ (pin Phantom 4 Pro; với 168 W cho khoảng 32 phút bay).
- Nhiệm vụ tham chiếu dùng 21–35 kJ (7–11%), nên ràng buộc pin vẫn không chạm. Tôi không thu nhỏ pin để nó có tác dụng.
- Pin chỉ chạm khi mỗi UAV phải khảo sát khoảng 1,5 km² (khoảng 25 km đường bay ở 10 m/s).

**3. Rút lại câu về payload của Huang.** Hình 5 của Huang ghi kích thước thật sự truyền cho một ảnh 512²:

| mode của Huang | payload | PSNR |
|---|---|---|
| Mode 1 (JPEG2000) | 314 kb | 32,2 dB |
| Mode 2 (text) | 0,376 kb | 11,3 dB |
| Mode 3 | 0,9 / 8,6 / 33 kb | 16,3 / 18,8 / 23,2 dB |

- Các số này cùng cỡ với payload đo ở A1 (jpeg2000_25: 251 kbit, 30,8 dB; cap_t2i: 0,5 kbit, 9,7 dB).
- "3,5 MB" chỉ là "kích thước ảnh gốc" trong bảng tham số. Bài không nói rõ con số nào đi vào độ trễ của Mode 1.
- Vì vậy câu "Huang/Li có trade-off vì payload 3,5 MB" của tôi là không có căn cứ. Huang có trade-off nhờ đâu (link vệ tinh, độ trễ tính toán 1,33 s của các mode semantic so với deadline 2–11 s, …) thì chưa xác định được chỉ từ bài báo.

**4. Kịch bản khảo sát theo yêu cầu thực: tính tải trước khi dựng.**

Nguồn cho từng thông số:
- Camera cỡ Phantom 4 Pro: FOV chéo 84°, cảm biến 3:2, 20 MP.
- Ở H = 100 m, footprint là 150 × 100 m (cạnh ngắn dọc theo hướng bay).
- Chồng ảnh khuyến nghị (Pix4D): 75% theo hướng bay, 60% giữa các tuyến. Từ đó: cứ 25 m chụp một khung, các tuyến cách nhau 60 m.
- Ở 10 m/s là một khung mỗi 2,5 s; ở 20 m/s là 1,25 s.

Khung được gửi là khung VisDrone ở độ phân giải gốc của dataset. Tập val có 408 ảnh 1360×765, 121 ảnh 960×540, 19 ảnh 1920×1080; trung bình 0,96 MP. Probe đo được avif_80 là 1,16 bpp (44 dB) và avif_50 là 0,59 bpp (36 dB).

| khung gửi đi | payload avif_80 | ρ_link (3 UAV, 2 kênh, 1 MHz) ở 10 m/s | ở 20 m/s |
|---|---|---|---|
| khung gốc dataset (~0,96 MP) | ~1,1 Mbit | **0,30** | **0,60** |
| khung camera đầy đủ 20 MP (*ngoại suy theo bpp, chưa đo*) | ~23 Mbit | ~6 | ~12 |

**Đọc kết quả.** Khảo sát theo yêu cầu chồng ảnh thực, với khung ảnh của chính dataset, cho **tải vừa (ρ 0,3–0,6), chưa phải tải cao**. Tải chỉ cao khi yêu cầu là gửi gần như **toàn bộ độ phân giải camera** gần thời gian thực. Đó là một yêu cầu ứng dụng cần quyết định, không phải tham số để chỉnh; dataset hiện có cũng không có ảnh 20 MP để đo.

**5. Đã chạy xong (xem 4.10).** A3 v2 trên kịch bản tham chiếu: MaskablePPO, seed 0, 2 triệu bước, cho {predicted, oracle} × w_T {0; 0,3}. Baseline dùng chung 30 seed test. Smoke test 200k bước đã chạy hết pipeline (PPO +55,0; greedy +60,9; staggered +63,1 ở 200k bước, chưa hội tụ).

### 4.10 A3 v2 — kết quả trên kịch bản tham chiếu (2026-11-07)

Cấu hình: MaskablePPO, seed 0, 2 triệu bước (khoảng 37 phút mỗi lần, chỉ CPU).

So sánh trên 30 seed test chung. "Heuristic tinh chỉnh" là thành viên của họ heuristic có return cao nhất trên **ảnh train và seed train 0..29**. Họ heuristic gồm: mọi mode cố định, greedy, paced q0..5, staggered (thứ tự idx / near / far, có hoặc không kèm điều nhịp).

Kết quả: `runs/uav_semcom/a3v2_*/{eval_s0,tune,ablate,analysis}.json`.

| mode_info | w_T | PPO | greedy | heuristic tinh chỉnh (tên) | PPO − tinh chỉnh, return [KTC 95%] |
|---|---|---|---|---|---|
| predicted | 0 | +62,67 | +60,79 | stag_near_q3 | −0,41 [−0,68, −0,14] |
| oracle | 0 | +63,27 | +61,23 | stag_near | −0,22 [−0,45, +0,00] |
| predicted | 0,3 | +26,44 | +24,92 | paced_q4 | **+1,21 [+0,80, +1,62]** |
| oracle | 0,3 | +26,83 | +25,43 | paced_q4 | **+1,19 [+0,86, +1,53]** |

Mọi chính sách ở đây đều giao 100% ảnh ở khoảng 41 dB.

**Ablation** (predicted, w_T = 0,3; H = paced_q4). Ghép từng phần {di chuyển, kênh + công suất, mode} từ PPO (P) hoặc heuristic (H):

| di chuyển / kênh+công suất / mode | return |
|---|---|
| PPP (PPO) | +26,44 |
| PPH | +26,67 |
| PHH | +26,56 |
| HPP | +24,38 |
| HHH (heuristic) | +25,23 |

Lợi thế nằm ở di chuyển. Kênh, công suất và mode của PPO không hơn heuristic.

**PPO bay khác gì** (10 seed):
- Gần như luôn bay thẳng tới waypoint; thỉnh thoảng hover hoặc lệch hướng khoảng 1 s.
- Lần chụp cuối chậm hơn khoảng 1,7 s (39,0 s so với 37,3 s).
- Nhờ vậy độ trễ trung bình giảm từ 1,75 xuống 1,30 s và hàng đợi ngắn hơn (1,03 so với 1,32 ảnh), ở cùng PSNR (41,0 dB).

Tức là PPO giãn nhịp chụp rất nhẹ, đủ để bớt dồn ảnh mà gần như không kéo dài nhiệm vụ. Heuristic theo ngưỡng hàng đợi không làm được kiểu điều chỉnh mịn như vậy.

**Đọc kết quả:**
1. Trong kịch bản tham chiếu tải thấp (ρ ≈ 0,19), mọi chính sách đều gần trần. Khác biệt giữa các chính sách chỉ khoảng 1–2 return trên khoảng 60 (≈ 0,02 utility/ảnh).
2. Không phạt thời gian: heuristic chia lượt nhỉnh hơn PPO một chút (−0,2 đến −0,4). Có phạt thời gian: PPO hơn heuristic tinh chỉnh khoảng +1,2, với khoảng tin cậy dương ở cả hai mức thông tin. Phần hơn này đến từ việc điều chỉnh nhịp bay mịn, **không** đến từ chọn mode.
3. Mức thông tin predicted và oracle chỉ chênh khoảng 0,4–0,6 return.
4. Chỉ có 1 seed train cho mỗi cấu hình. Đây là kết quả của kịch bản tham chiếu, không phải kết luận chung về RL hay SemCom.

## 5. A2 v2.1 — nhiễu theo thời gian thực và thông tin `predicted` có tính nhân quả (2026-11-07)

Đây là bước đầu theo chỉ đạo mới: A2 v2.1 → A3.2 (nhiều seed) → A4 (khung VisDrone gốc, tần suất chụp theo photogrammetry). Kết quả A3 v2 (mục 4.10, commit d812c7e7) vẫn giữ làm tham chiếu.

### 5.1 Thay đổi so với v2

Bản v2 được giữ nguyên trong `legacy_env_v2.py` để chạy regression test.

1. **Nhiễu tính theo thời điểm truyền thật.**
   - Trong mỗi slot, việc truyền được mô phỏng theo sự kiện: một ảnh phát xong, một ảnh encode xong, hoặc một deadline hết.
   - Giữa hai sự kiện, tập UAV đang phát là cố định nên tốc độ là hằng số. Một UAV chỉ bị nhiễu bởi các UAV **cùng kênh và đang phát đúng lúc đó**.
   - v2 thì tính nhiễu từ mọi UAV cùng kênh có ảnh sẵn trong slot, cho cả slot.
   - Mỗi đoạn truyền ghi lại (UAV, kênh, bắt đầu, kết thúc, nhiễu, tốc độ).
2. **Thông tin `predicted` có tính nhân quả.**
   - Ngay khi chụp, probe avif_5 được xếp vào hàng đợi encoder của UAV, các probe chạy nối tiếp nhau.
   - Một ảnh chỉ được chọn mode ở thời điểm quyết định **đầu tiên sau khi probe của nó xong**: chụp → probe → có dự đoán → chọn mode → encode → phát.
   - Trước thời điểm đó, observation không chứa dự đoán của ảnh này, và vector mode không áp dụng cho nó.
   - Thời gian probe không còn bị cộng vào thời gian encode của mode đã chọn.
   - Với `oracle` và `mean`, thông tin có ngay lúc chụp (không đổi).

Mọi công thức khác giữ nguyên.

### 5.2 Test (27/27 qua)

Test mới:
- `test_interference_non_overlapping`: UAV A phát 0–0,2 s, UAV B phát từ 0,6 s. Không UAV nào bị nhiễu. (v2 sẽ tính nhiễu cho cả hai suốt cả slot.)
- `test_interference_overlapping`: hai UAV cùng phát từ 0 s.
  - Trong lúc cả hai cùng phát, mỗi UAV chịu nhiễu đúng bằng p·g của UAV kia, với cùng mẫu fading mà UAV kia dùng cho link của chính nó.
  - Khi một UAV phát xong, UAV còn lại phát không nhiễu.
  - Tổng bit gửi đi khớp payload.
- `test_predicted_info_is_causal`:
  - Các probe chạy nối tiếp ngay sau khi chụp (info_t = t_cap + i·τ_probe).
  - Ngay lúc chụp chưa có ảnh nào được quyết định, và observation không lộ dự đoán.
  - Ở thời điểm quyết định kế tiếp, mode được gán; encode bắt đầu ≥ info_t.
- `test_oracle_info_available_at_capture`.
- `test_regression_vs_v2_without_cochannel` (4 cấu hình × 2 mode): khi thông tin là oracle và không có truyền đồng kênh, v2.1 cho kết quả **trùng khớp** v2 (return, utility, delay, năng lượng, thời gian nhiệm vụ).

Các test cũ được sửa theo định dạng segment mới.

### 5.3 Ảnh hưởng

`check_v21.py` chạy cùng các chính sách trên cả v2 và v2.1, trên 30 seed test, không train lại. PPO ở đây là model A3 v2 đã train trên v2. Kết quả: `runs/uav_semcom/check_v21.json`.

| cấu hình | chính sách | return v2 → v2.1 | delay (s) v2 → v2.1 | tỉ lệ airtime có nhiễu (v2.1) |
|---|---|---|---|---|
| oracle, w_T 0,3 | PPO | +26,83 → +26,93 | 1,27 → 1,25 | 7,9% |
| oracle, w_T 0,3 | greedy / paced_q4 | không đổi | không đổi | 0 |
| oracle, w_T 0 | PPO | +63,27 → +63,37 | | 4,9% |
| predicted, w_T 0,3 | PPO | +26,44 → +21,23 | 1,27 → 2,15 | 5,2% |
| predicted, w_T 0,3 | greedy | +24,92 → +18,30 | 1,81 → 2,72 | 0 |
| predicted, w_T 0,3 | paced_q4 | +25,23 → +18,47 | 1,75 → 2,69 | 0 |
| predicted, w_T 0 | PPO / stag_near | +62,67 → +57,88 / +63,20 → +58,53 | 1,32 → 2,27 / 1,27 → 2,21 | 16,4% / 0 |

PPO trừ heuristic tốt nhất (cặp theo seed, KTC 95%):

| cấu hình | trên v2 | trên v2.1 |
|---|---|---|
| oracle, w_T 0,3 | +1,19 [+0,86, +1,53] | +1,30 [+0,99, +1,62] |
| oracle, w_T 0 | −0,22 [−0,45, 0,00] | −0,12 [−0,33, +0,09] |
| predicted, w_T 0,3 | +1,21 [+0,80, +1,62] | +2,76 [+1,89, +3,63] (*PPO chưa train lại trên v2.1*) |
| predicted, w_T 0 | −0,52 [−0,73, −0,32] | −0,65 [−0,89, −0,42] |

**Đọc kết quả:**
1. **Nhiễu theo thời gian thực.** Các heuristic luôn cho mỗi UAV một kênh riêng, nên không bị ảnh hưởng. PPO có dùng chung kênh (5–16% airtime) và được lợi rất ít (+0,1 return). Giả định bảo thủ của v2 **không** phải nguồn gốc lợi thế của PPO, cũng không làm lợi thế đó nhỏ đi đáng kể.
2. **Probe có tính nhân quả** làm mọi chính sách ở mức `predicted` mất 5–7 return, vì delay tăng khoảng 0,9 s.
   - Nguyên nhân: probe xong sau thời điểm chụp (khoảng 0,04 s mỗi ảnh), nên quyết định mode phải đợi tới **thời điểm quyết định kế tiếp, tức 1 s sau**.
   - Đây là hệ quả của việc quyết định theo slot 1 s, không phải giá trị thật của thông tin.
   - Vì vậy khoảng cách oracle − predicted (khoảng 5 return) hiện chủ yếu đo độ thô của nhịp quyết định.
   - Nếu muốn so giá trị thông tin cho đúng, cần cho quyết định mode xảy ra ngay khi probe xong (một semi-MDP / quyết định theo sự kiện). Đây là một lựa chọn mô hình cần quyết, chưa làm.
3. Model PPO cũ (train trên v2) giữ được nhiều hơn heuristic dưới quy tắc mới (+2,76). Nhưng phải train lại trên v2.1 trước khi kết luận (A3.2).

## 6. A2 v2.2 — điều khiển hai thang thời gian: chọn mode theo sự kiện (2026-11-07)

Đây là lựa chọn đã chốt khi review: **chọn mode ngay khi probe xong**. Bay, kênh và công suất vẫn quyết định theo slot 1 s. Bản v2.1 (`env.py`) được giữ nguyên làm baseline cho các quyết định theo slot; A3.2 đang train trên bản đó. Code mới nằm ở các module riêng nên không đụng vào `env.py` trong lúc train.

### 6.1 Thiết kế

`env_ev.py` (`UavSemcomEnvEv`, kế thừa v2.1):
- Mỗi slot là một vòng lặp sự kiện theo thời gian liên tục: truyền, encode xong, deadline, thông tin về.
- Vòng lặp **dừng lại mỗi khi có một ảnh cần quyết định mode**, rồi chạy tiếp sau `decide(m)`. Policy không thấy gì xảy ra sau thời điểm quyết định.
- API:
  - `begin_slot([(move, ch, power)])` chỉ nhận phần slot, không nhận mode;
  - `advance()` trả về `("mode", k, ảnh)` hoặc `("slot", done)`;
  - `decide(m)` gán mode cho đúng ảnh đang chờ.
- Encoder FIFO, nối tiếp. Probe được xếp vào hàng đợi ngay lúc chụp, mỗi ảnh một probe, nối tiếp nhau. Lệnh encode theo mode được xếp vào lúc quyết định, sau những việc encoder đang giữ.

Bốn mức thông tin:

| mức | thông tin | có từ lúc nào |
|---|---|---|
| `oracle` | chính xác | lúc chụp, không tốn chi phí (cận trên) |
| `oracle_probe` | chính xác | sau khi probe chạy xong (**tách chi phí thu thập thông tin**) |
| `predicted` | dự đoán từ probe | sau khi probe chạy xong (triển khai được) |
| `mean` | trung bình trên ảnh train | lúc chụp |

Vì vậy: oracle → oracle_probe là chi phí thu thập thông tin; oracle_probe → predicted là sai số dự đoán.

**Interface RL** (`gym_env_ev.py`): **một policy, hai loại bước quyết định**.
- Bước slot ở đầu mỗi slot: action là (hướng bay, kênh, công suất) cho mỗi UAV; phần mode bị mask về 0.
- Bước mode khi thông tin của một ảnh về, có thể giữa slot: action là mode của **đúng ảnh đó**; phần slot bị mask.
- Observation = observation của env tại thời điểm quyết định + cờ loại bước + đặc trưng của ảnh đang được quyết định (UAV one-hot, slack, Qmin, bit/PSNR từng mode theo mức thông tin tại thời điểm đó, backlog encoder và backlog phát của UAV đó).
- Action space: MultiDiscrete([11, M+1, P] × K + [n_modes]). Reward của một bước là phần reward tích luỹ tới quyết định kế tiếp.
- γ áp dụng theo bước quyết định. Các bước mode gần như không tốn thời gian; đây là một xấp xỉ, đã ghi lại.
- Vector mode cũ (L phần tử chọn ở đầu slot) không được dùng ở đây, nên không có đường rò rỉ thông tin tương lai.

### 6.2 Test (`test_env_ev.py`, 12 test; cùng `test_env.py` 39/39 qua)

- Mode được quyết định **đúng tại** info_t = t_cap + (i+1)·τ_probe; encode bắt đầu ≥ thời điểm quyết định.
- Tại mọi thời điểm quyết định, policy chỉ thấy thông tin của những ảnh đã probe xong.
- Bảo toàn bit, thời gian, năng lượng; deadline; mỗi ảnh được tính đúng một lần. Chạy cho cả 4 mức thông tin.
- Encoder nối tiếp: các khoảng probe và encode không chồng nhau.
- **Regression:** với oracle, env theo sự kiện cộng greedy-theo-sự-kiện cho kết quả **trùng khớp** v2.1 cộng greedy theo slot (3 cấu hình).
- Với mode cố định: quyết định theo sự kiện không bao giờ làm ảnh nào encode hoặc giao muộn hơn, và ảnh đầu mỗi waypoint encode sớm hơn ít nhất 0,5 s.
- Interface RL:
  - một episode ngắn chạy đủ, số bước mode bằng đúng số ảnh;
  - MaskablePPO lấy 0 action bị mask.

### 6.3 Ảnh hưởng (chỉ heuristic, 30 seed test; `runs/uav_semcom/check_v22.json`)

Return (delay trung bình). Phần slot lấy từ heuristic v2; phần mode dùng greedy theo sự kiện.

| w_T | policy | theo slot: oracle | theo slot: predicted | sự kiện: oracle | sự kiện: oracle_probe | sự kiện: predicted | sự kiện: mean |
|---|---|---|---|---|---|---|---|
| 0,3 | greedy | +25,43 (1,73 s) | +18,30 (2,72 s) | +25,43 | +24,36 (1,91 s) | +24,36 (1,91 s) | +25,38 (1,73 s) |
| 0,3 | paced_q4 | +25,63 | +18,47 | +25,63 | +24,61 | +24,56 | +25,51 |
| 0 | greedy | +61,23 | +54,98 | +61,23 | +60,30 | +60,35 | +61,15 |
| 0 | stag_near | +63,49 | +58,53 | +63,49 | +62,73 | +62,68 | +63,40 |

**Đọc kết quả:**
1. Quyết định theo sự kiện lấy lại **+5 đến +6 return** so với quyết định theo slot ở mức predicted. Gần như toàn bộ khoảng mất ở v2.1 là do chờ nhịp quyết định 1 s.
2. Khoảng cách oracle − predicted còn khoảng 1 return. Phân rã:
   - **chi phí thu thập (probe): ≈ 0,8–1,1**, vì 4 probe chạy nối tiếp khoảng 0,16 s và encode phải xếp sau chúng;
   - **sai số dự đoán: ≈ 0** (oracle_probe ≈ predicted).
3. Mức `mean` (không probe, không có thông tin riêng từng ảnh) ≈ oracle. Trong kịch bản tham chiếu tải thấp, thông tin riêng từng ảnh gần như **không có giá trị**, nên chạy probe còn thiệt hơn không chạy. Đây là một đặc điểm của chế độ tải, không phải kết luận chung. Nó sẽ được kiểm tra lại ở A4, khi tải cao hơn.

## 7. A3.2 — 3 seed huấn luyện trên v2.1, điều khiển theo slot (2026-11-07)

Đây là baseline cho điều khiển theo slot: env v2.1 (`env.py`, không đổi trong lúc train), mode chọn ở đầu slot, mức thông tin `predicted`, w_T = 0,3.

Cấu hình train:
- MaskablePPO, 2 triệu bước mỗi seed, seed huấn luyện 0, 1, 2 (khoảng 37 phút mỗi seed, chỉ CPU);
- checkpoint cuối cùng được cố định từ trước, không chọn theo kết quả.

Lưu ý: `train_a3.py` theo dõi tiến trình train trên 10 seed đầu của tập test, nhưng không dùng kết quả đó để chọn hay dừng gì. Script mới `train_ev.py` theo dõi trên seed validation riêng (2000..2009).

Heuristic được tinh chỉnh trên ảnh train và seed train 0..29. Họ heuristic gồm:
- mọi mode cố định, greedy, greedy_detour;
- paced q0..5;
- staggered (idx / near / far, có hoặc không kèm điều nhịp);
- **pacedB τ ∈ {0,25 … 4}**: điều nhịp theo backlog tính bằng giây, mịn hơn ngưỡng số ảnh.

Chọn trên train, thành viên tốt nhất là **pacedB1.5**.

Kết quả lưu ở `runs/uav_semcom/a32_predicted_wt0.3_s{0,1,2}/{eval_s*,tune,ablate,analysis}.json`. Script: `scripts/uav_semcom/{run,tune,ablate,analyze}_a32.sh`.

### 7.1 Return trên 30 seed test chung

| policy | seed 0 | seed 1 | seed 2 | trung bình (độ lệch chuẩn giữa các seed) |
|---|---|---|---|---|
| **PPO** | +21,26 | +21,28 | +21,10 | **+21,21 (0,10)** |
| pacedB1.5 (tinh chỉnh) | +18,54 | (giống) | (giống) | +18,54 |
| greedy_detour | +18,57 | | | |
| paced_q4 | +18,47 | | | |
| greedy | +18,30 | | | |
| pacedB2 / pacedB1 / pacedB0,5 | +18,21 / +11,10 / +8,88 | | | |
| fixed avif_80 (mode cố định tốt nhất) | +17,68 | | | |
| fixed mbt2018-mean_q8 | +17,47 | | | |

Heuristic không phụ thuộc seed huấn luyện, nên chỉ có một giá trị.

PPO − heuristic tinh chỉnh, cặp trên seed test [KTC 95%]:

| seed huấn luyện | Δ return | Δ utility/ảnh | Δ thời gian nhiệm vụ |
|---|---|---|---|
| 0 | **+2,72 [+1,69, +3,75]** | +0,021 | −1,4 s |
| 1 | **+2,74 [+1,82, +3,65]** | +0,021 | −1,3 s |
| 2 | **+2,56 [+1,52, +3,60]** | +0,017 | −1,4 s |

Ba lần train cho kết quả gần như giống nhau. Độ lệch chuẩn giữa các seed là 0,10 return, nhỏ hơn nhiều so với độ rộng KTC trên seed test (khoảng ±1). Cả ba đều hội tụ ở khoảng 1M bước (utility theo dõi được 0,79 → 0,81).

Mọi chính sách đều giao khoảng 99–100% ảnh, PSNR khoảng 41 dB.

### 7.2 Ablation thành phần hành động (H = pacedB1.5)

| di chuyển / kênh+công suất / mode | seed 0 | seed 1 | seed 2 | trung bình |
|---|---|---|---|---|
| PPP (PPO) | +21,26 | +21,28 | +21,10 | +21,21 |
| PPH | +21,38 | +21,32 | +21,14 | +21,28 |
| PHP | +20,36 | +20,10 | +19,49 | +19,99 |
| PHH | +20,47 | +20,20 | +19,56 | +20,08 |
| HPP | +19,78 | +19,95 | +19,77 | +19,83 |
| HPH | +19,88 | +20,05 | +20,06 | +20,00 |
| HHP | +18,71 | +18,71 | +18,43 | +18,62 |
| HHH (heuristic) | +18,54 | +18,54 | +18,54 | +18,54 |

Mỗi thành phần, so cặp trên seed test (seed 0 / 1 / 2):
- **Di chuyển** (PHH − HHH): +1,9 ± 1,0 / +1,7 ± 1,0 / +1,0 ± 1,1.
- **Kênh + công suất** khi di chuyển là của PPO (PPH − PHH): +0,9 ± 0,4 / +1,1 ± 0,6 / +1,6 ± 0,5. Khi di chuyển là của heuristic (HPP − HHH): +1,2 đến +1,4.
- **Mode**: PPH ≥ PPP ở cả 3 seed. Mode của PPO **không** hơn mode của heuristic.

Khác với A3 v2 (mục 4.10), lần này kênh + công suất của PPO cũng đóng góp khoảng +1 return.

Giả thuyết, **chưa kiểm chứng riêng**: v2.1 tính nhiễu theo thời gian thực, nên hai UAV có thể phát chung một kênh trong lúc một UAV có độ lợi kênh yếu. Heuristic không bao giờ dùng chung kênh; PPO A3 v2 có dùng (5–16% airtime, mục 5.3). PPO cũng thỉnh thoảng hạ công suất (khoảng 10% số slot có phát).

### 7.3 PPO làm gì khác (10 seed test, mỗi seed huấn luyện)

| | PPO (s0 / s1 / s2) | pacedB1.5 |
|---|---|---|
| tốc độ trung bình | 19,2 / 19,1 / 19,2 m/s | 16,7 m/s |
| số slot hover / đi theo hướng la bàn (1–8) | 51 / 58 / 48 hover; 10–14% slot theo hướng la bàn | 219 hover, 0 hướng la bàn |
| lần chụp cuối | 39,0 / 39,6 / 38,5 s | 40,6 s |
| độ trễ trung bình | 2,24 s | 2,50 s |
| PSNR | 41,1–41,2 dB | 40,97 dB |
| tỉ lệ mbt q8 / avif_80 | 538:182 / 515:205 / 470:249 | 402:318 |
| năng lượng | 21,9 kJ | 22,5 kJ |

**Đọc kết quả:**
- pacedB1.5 điều nhịp bằng cách **hover** chờ hàng đợi vơi.
- PPO gần như không hover. Thay vào đó, đôi khi nó **bay lệch hướng** 1–2 slot, vẫn chậm lại tương đương mà không đứng yên. Theo mô hình Zeng, bay ở tốc độ vừa phải tốn ít công suất hơn hover, nên năng lượng thấp hơn khoảng 0,6 kJ.
- PPO vẫn chụp xong sớm hơn heuristic, độ trễ ngắn hơn, và nhờ hàng đợi thoáng hơn nên dùng mbt q8 (PSNR cao hơn) nhiều hơn.

Phần năng lượng chỉ đóng góp khoảng 0,01 return (w_E = 0,02/kJ). Phần hơn chủ yếu là utility: +0,02 × 120 ảnh ≈ +2,5.

### 7.4 Kết luận A3.2 (kịch bản tham chiếu, điều khiển theo slot)

1. Với 3 seed huấn luyện, PPO hơn heuristic tinh chỉnh tốt nhất (kể cả heuristic điều nhịp mịn hơn là pacedB) khoảng **+2,6 đến +2,7 return**, tức +0,02 utility/ảnh. KTC dương ở cả 3 seed; độ lệch giữa các seed nhỏ (0,10).
2. Phần hơn đến từ **di chuyển** (khoảng +1,0 đến +1,9) và **kênh + công suất** (khoảng +1), **không** đến từ chọn mode.
3. Đây vẫn là kịch bản tải thấp (ρ ≈ 0,19); mọi chính sách đều giao gần như 100% ảnh. Biên độ khác biệt nhỏ so với tổng return.
4. Điều khiển theo sự kiện (mục 6) lấy lại khoảng +6 return cho mọi chính sách ở mức predicted. Train PPO trên env hai thang thời gian (`train_ev.py`) là bước tiếp theo trên kịch bản tham chiếu, chạy cùng A4.

## 8. A4 — khảo sát photogrammetry trên ảnh VisDrone gốc (2026-11-07, đang chạy)

So với kịch bản tham chiếu, chỉ **workload** và **bảng mode** thay đổi. Topology (K = 3 UAV, M = 2 kênh, 1 MHz, BS), kênh truyền, encoder/transmitter, deadline (4–12 s), Qmin (22–30 dB), utility, reward và ràng buộc giữ nguyên như v2.1/v2.2. Điều khiển hai thang thời gian như mục 6. Mức thông tin chính là `predicted`, w_T = 0,3.

### 8.1 Bảng mode ở độ phân giải gốc (bước A4.1)

Cùng 100 ảnh VisDrone2019-DET-val như A1, nhưng **không resize**: 1360×765, 960×540 hoặc 1920×1080. Kết quả lưu ở `runs/uav_semcom/a4_native/{a11,a12}/`.
- Bitstream thật, PSNR/LPIPS trên ảnh gốc.
- Thời gian encode/decode đo trên GPU RTX 3070. Learned codec đo thêm trên CPU (i5-14400F, 10 ảnh), coi như máy tính trên UAV; CPU không chạy việc khác trong lúc đo.
- Learned codec: ảnh có cạnh không chia hết cho 64 được pad mép rồi crop lại. Payload ảnh 512² vẫn trùng từng byte với trước.
- Mode tạo sinh: sinh ở cạnh ngắn 512, giữ tỉ lệ khung hình, rồi resize về kích thước gốc. Thời gian resize tính vào decode.

| mode | kB (gốc) | bpp | PSNR (gốc) | encode (CPU đối với mbt) | kB (512²) | PSNR (512²) |
|---|---|---|---|---|---|---|
| avif_5 / 20 / 50 / 80 | 9,2 / 18,9 / 73,0 / 147,2 | 0,08 / 0,16 / 0,62 / 1,26 | 26,0 / 28,3 / 35,4 / 44,3 | 0,07–0,14 s | 2,9 / 5,8 / 20,3 / 53,1 | 25,3 / 27,6 / 33,3 / 40,2 |
| mbt2018-mean q1 / 3 / 5 / 8 | 14,8 / 35,2 / 75,4 / 172,2 | 0,13 / 0,30 / 0,65 / 1,48 | 28,0 / 31,4 / 35,5 / 42,0 | 0,45 / 0,45 / 0,87 / 0,84 s | 4,5 / 10,9 / 23,1 / 52,8 | 27,4 / 30,9 / 35,1 / 41,5 |
| jpeg2000 100 / 25 | 28,5 / 114,1 | 0,24 / 0,96 | 26,9 / 31,8 | 0,20 s | 7,8 / 31,4 | 25,9 / 30,7 |
| thumb64_up / thumb128_up | 1,3 / 3,6 | 0,01 / 0,03 | 21,7 / 23,7 | 0,02–0,03 s | 0,8 / 2,1 | 22,0 / 24,1 |
| cap_t2i | 0,1 | 0,001 | 9,7 | 0,21 s | 0,1 | 9,7 |
| cap_thumb64 / 128 i2i | 1,3 / 3,7 | 0,01 / 0,04 | 18,8–21,5 | 0,23–0,24 s | 0,9 / 2,2 | 19,1–21,9 |

- Payload ở độ phân giải gốc lớn gấp khoảng **3 lần** ảnh 512²; số pixel trung bình gấp khoảng 4 lần.
- Encode của learned codec trên CPU tăng lên 0,45–0,87 s mỗi khung hình.
- Mode tạo sinh vẫn chỉ cho PSNR 10–22 dB, dưới Qmin, nên hầu như không bao giờ đáng chọn.

### 8.2 Workload và chế độ tải (bước A4.2; chỉ heuristic, 30 seed test; `a4_native/check_a4_wt0.3.json`)

Mỗi UAV bay lawn-mower 4 tuyến × 300 m, chụp **52 khung hình**. Thông số:
- cách nhau 25 m dọc tuyến (chồng ảnh 75%) và 60 m giữa các tuyến (chồng 60%);
- camera lớp Phantom 4 Pro ở H = 100 m;
- khung hình chụp đúng lúc UAV đi qua điểm chụp, kể cả giữa slot.

Tốc độ bay khảo sát là 10 hoặc 20 m/s, tức λ = 0,4 hoặc 0,8 khung hình/s mỗi UAV. Tốc độ kênh ở các điểm chụp: median 2,18 Mb/s (p10 1,84, p90 2,94), công suất tối đa, chưa tính nhiễu.

Mức tải của các mode chất lượng cao: ρ_link = K·λ·bit / (min(K, M)·R), ρ_enc = λ·t_enc.

| | avif_80 | mbt q8 | mbt q5 | avif_50 | jpeg2000_25 |
|---|---|---|---|---|---|
| ρ_link ở 10 / 20 m/s | 0,32 / **0,65** | 0,38 / **0,76** | 0,17 / 0,34 | 0,16 / 0,32 | 0,25 / 0,50 |
| ρ_enc ở 10 / 20 m/s | 0,05 / 0,10 | 0,34 / **0,67** | 0,35 / **0,70** | 0,05 / 0,11 | 0,08 / 0,16 |

Kịch bản tham chiếu có ρ ≈ 0,19. Ở 20 m/s, cả kênh truyền lẫn encoder đều có tải đáng kể.

Return của heuristic (tỉ lệ giao đúng hạn):

| v | policy | oracle | oracle_probe | predicted | mean |
|---|---|---|---|---|---|
| 10 | greedy | +3,6 (1,00) | +3,0 | +2,4 | +3,2 |
| 10 | pacedB1.5 | +4,2 (1,00) | +3,9 | +3,3 | +4,2 |
| 10 | fixed avif_80 | +3,3 (1,00) | +2,3 | +2,3 | +3,3 |
| 20 | greedy | +31,8 (0,93) | +32,0 | +25,4 (0,91) | +26,5 |
| 20 | paced_q4 | +38,0 (0,95) | +36,9 | +31,3 | +32,0 |
| 20 | pacedB1.5 | **+48,1 (0,99)** | +47,2 | +46,1 | +46,8 |
| 20 | stag_near | +44,3 (0,99) | +43,8 | +43,4 | +43,8 |
| 20 | fixed avif_50 | +45,7 (1,00) | +45,1 | +45,1 | +45,7 |
| 20 | fixed mbt q5 | +34,2 (0,99) | +33,0 | +33,0 | +34,2 |

Return ở 20 m/s cao hơn ở 10 m/s chủ yếu vì nhiệm vụ ngắn hơn (khoảng 77–99 s so với khoảng 146 s), nên phạt thời gian nhỏ hơn. Return giữa hai tốc độ **không** so trực tiếp được với nhau.

**Đọc kết quả:**
1. **10 m/s vẫn là tải thấp.** Mọi heuristic điều nhịp đều giao gần như 100% ảnh, và thông tin riêng từng ảnh gần như không có giá trị (mean ≈ oracle), giống kịch bản tham chiếu.
2. **20 m/s là tải cao.**
   - greedy (chọn mode tốt nhất cho từng ảnh, không điều nhịp) rớt 7–9% ảnh.
   - Điều nhịp theo backlog (pacedB) hoặc giảm chất lượng cố định (avif_50) tránh được rớt ảnh.
   - Khác biệt giữa các heuristic lên tới khoảng 20 return, so với khoảng 1–2 ở kịch bản tham chiếu.
3. **Thông tin riêng từng ảnh có giá trị ở 20 m/s, nhưng chỉ khi điều khiển không điều nhịp.**
   - Với greedy: oracle − mean ≈ +5. Sai số dự đoán (oracle_probe → predicted) ≈ −6,5, tức với greedy, `predicted` còn kém `mean` khoảng 1.
   - Với pacedB1.5: oracle − mean ≈ +1,3, sai số dự đoán ≈ −1,2.
   - Nghĩa là bộ dự đoán tuyến tính từ probe avif_5 (fit trên ảnh train) chưa đủ chính xác ở độ phân giải gốc để policy tham lam tận dụng.
4. Heuristic dùng ở đây chưa được tinh chỉnh. Việc tinh chỉnh trên seed train (`tune_ev.py`, cùng họ heuristic như A3) chạy trước khi train PPO, và kết quả đó là baseline để so sánh.

### 8.3 Kế hoạch chạy

Đợt 1 (`run_a4_batch.sh`, đang chạy). Heuristic tinh chỉnh trên train ở 20 m/s là **pacedB1.5** (train +50,3, test +46,1). Mỗi tốc độ (20 rồi 10 m/s):
- tinh chỉnh heuristic trên seed train;
- MaskablePPO hai thang thời gian, **joint** (RL quyết định cả slot lẫn mode), seed 0, 1, 2, mỗi seed 2M bước;
- theo dõi trên seed validation 2000..2009, đánh giá cuối một lần trên seed test 1000..1029.

Tổng 6 lần train, mỗi lần khoảng 40 phút.

Đợt 2 (`run_a4_batch2.sh`, tự chạy sau đợt 1; ablation): **chỉ bay** (mode do greedy theo sự kiện quyết định) và **chỉ mode** (slot do heuristic tinh chỉnh quyết định), 3 seed mỗi tốc độ, cùng cấu hình.

### 8.4 Đợt 1: PPO hai thang thời gian, joint (RL quyết định cả slot lẫn mode), 3 seed mỗi tốc độ

Cấu hình: MaskablePPO, 2M bước, `predicted`, w_T = 0,3. Checkpoint cuối cố định; theo dõi trên seed validation; đánh giá trên 30 seed test. Kết quả lưu ở `runs/uav_semcom/a4_native/v{20,10}_joint_s{0,1,2}/`.

**Bổ sung baseline sau khi xem hành vi PPO (ghi rõ để minh bạch):**
- Phân tích (mục 8.5) cho thấy PPO ở 20 m/s **bay liên tục và hạ mode theo từng ảnh**, thay vì hover chờ như heuristic.
- Họ heuristic cũ chưa có luật nào làm như vậy. Rút kinh nghiệm A3 (staggered), tôi thêm một họ heuristic mode mới: **`budget_mode(τ)`** (`mode_heur.py`).
  - Mode policy chọn mode tốt nhất theo đúng tiêu chí của greedy, nhưng chỉ trong số các mode giữ backlog phát ≤ τ giây.
  - Với τ = ∞ nó **trùng khớp** event-greedy (có test).
  - Ghép với slot heuristic {greedy, pacedB1.5, pacedB2, pacedB3} và τ ∈ {0,5 … 3}.
- Họ heuristic được tinh chỉnh trên seed train, **hoàn toàn không dùng** kết quả PPO.
- Lỗi của tôi trong lúc làm, đã sửa trước khi lấy kết quả: bản đầu của `budget_mode` loại hẳn các mode trễ hạn. Greedy thì cho chúng −1, và −1 có thể tốt hơn một mode kịp hạn nhưng vi phạm Qmin (utility < −1). Test phát hiện ra lỗi này; kết quả tinh chỉnh của bản lỗi đã bỏ.

"Heuristic tinh chỉnh" dưới đây là thành viên tốt nhất **trên train** của họ hợp (họ v2 + họ budget).

| v | heuristic tinh chỉnh (train → test) | PPO s0 / s1 / s2 | trung bình (độ lệch chuẩn giữa các seed) | PPO − tinh chỉnh [KTC 95%] |
|---|---|---|---|---|
| 20 | pacedB1.5+budget1.5 (+52,2 → **+46,7**; ok 0,979; 80 s) | +52,89 / +53,19 / +52,06 | **+52,71 (0,59)** | **+6,2 [+4,0, +8,4] / +6,5 [+4,8, +8,2] / +5,4 [+3,3, +7,4]** |
| 10 | pacedB2+budget3 (+4,4 → **+3,2**) | +2,28 / +2,98 / +3,23 | +2,83 (0,49) | −0,9 [−1,4, −0,3] / −0,2 [−0,8, +0,4] / +0,1 [−0,5, +0,7] |

Ở 20 m/s, nếu chọn heuristic tốt nhất **ngay trên test** (lạc quan cho heuristic): pacedB1.5+budget2 đạt +47,3. PPO vẫn hơn khoảng +5.

Chỉ dùng họ cũ (trước khi thêm budget), tinh chỉnh chọn pacedB1.5 (+46,1), và PPO hơn +6,0 đến +7,1.

### 8.5 PPO làm gì ở 20 m/s (10 seed test; `v20_joint_s*/analysis.json`)

Phân rã return = Σ utility − w_T · (UAV-giây hoạt động) − w_E · năng lượng:

| | PPO (s0 / s1 / s2) | pacedB1.5 (họ cũ) |
|---|---|---|
| Σ utility | 120,3 / 120,9 / 118,6 | 128,8 |
| chi phí thời gian | 66,6 / 66,6 / 66,2 | 80,4 |
| chi phí năng lượng | 0,8 | 0,9 |
| thời gian nhiệm vụ | 74 s | 89 s |
| tỉ lệ giao đúng hạn | 1,00 | 0,99 |
| mode | avif_50 65–81%, avif_80 19–32%, vài mbt q3/q5 | avif_80 100% |
| PSNR / dư so với Qmin | 37–38 dB / +11–12 dB | 44,3 dB / +18 dB |
| slot hover / bay theo tuyến | 4–7 / 2161 | 520 / 2160 |
| độ trễ | 1,3–1,4 s | 2,4 s |

**Đọc kết quả:**
- PPO **chấp nhận PSNR thấp hơn** (vẫn trên Qmin khoảng 12 dB), mất khoảng 8 utility, để **bay khảo sát liên tục**, tiết kiệm khoảng 14 chi phí thời gian.
- Nó không hạ đều: khoảng 20–30% ảnh vẫn được gửi avif_80, có lẽ khi backlog cho phép.
- Heuristic budget làm cùng ý tưởng nhưng kém hơn: hoặc rớt nhiều ảnh hơn (ok 0,98), hoặc phải bay chậm hơn (80–88 s).

Phần đóng góp riêng của di chuyển và của chọn mode sẽ được tách ra ở đợt 2 (ablation chỉ bay / chỉ mode).

**Đọc kết quả A4 đợt 1:**
1. **10 m/s (tải thấp):** PPO ≈ heuristic tinh chỉnh, chênh −0,9 đến +0,1. Giống kịch bản tham chiếu: khi tải thấp, gần như không còn gì để học.
2. **20 m/s (tải cao):** PPO hơn heuristic tinh chỉnh tốt nhất khoảng **+5,4 đến +6,5 return**, KTC dương ở cả 3 seed, độ lệch giữa các seed 0,59. Đây là khoảng cách lớn nhất cho đến nay, và lần đầu tiên **chọn mode theo trạng thái** có vẻ đóng góp đáng kể.
3. Lợi thế này phụ thuộc vào trọng số thời gian w_T = 0,3: PPO đổi chất lượng ảnh lấy thời gian nhiệm vụ. Với w_T khác, cách đánh đổi đúng sẽ khác. Kết quả này **không** nói rằng PPO cho ảnh tốt hơn.
