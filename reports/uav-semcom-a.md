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
