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
| pin | 1,97 MJ (2 × 274 Wh) | giả định | một pin UAV thương mại; nhiệm vụ chỉ dùng 20–30 kJ nên **không bao giờ chạm** |
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

Các paper có trade-off về bit vì payload của họ lớn so với băng thông:
- Huang và Li: ảnh gốc 3,5 MB với 10–20 MHz.
- Truong: ảnh 2K **không nén** với 100 MHz; truyền không nén mất 2,4 s.

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
