# Bước 0 theo bài Huang (JSAC 2025): bài toán chọn mode có cần RL không? (06/11)

Nguồn gốc: Huang et al., "DRL-Based Resource Allocation for Hybrid Bit and Generative Semantic Communications in
SAGINs", IEEE JSAC 43(12), 2025 (`pdf/Deep Reinforcement Learning-Based Resource Allocation ...pdf`). Đây là
**môi trường đơn giản hoá theo Huang**, không phải tái lập simulator của bài: những chỗ bài không ghi (hệ số
Rician, kích thước vùng, cách chuẩn hoá SCE, cách xếp hàng đợi) đều được chọn và ghi rõ trong
`swarmbridge/experiments/huang_step0/model.py` (nhãn PAPER / ASSUMPTION). Chỉ dùng CPU, không train gì.

## Cách bài Huang hoạt động (tóm tắt)
LEO có ảnh → UAV chuyển tiếp → người dùng mặt đất. Mỗi ảnh chọn 1 trong 5 mode (Hình 5 của bài):

| Mode | PSNR (dB) | Kích thước gửi | Thời gian tính |
|---|---|---|---|
| M1 JPEG2000 + LDPC | 32.2 | 314 kbit | 0.1323 s |
| M2 chỉ text → sinh ảnh | 11.26 | 0.376 kbit | 1.33 s |
| M3_1 / M3_2 / M3_3 text + embedding | 16.3 / 18.8 / 23.2 | 0.897 / 8.575 / 33.151 kbit | 1.33 s |

Điểm SCE = 0.5·(chất lượng − mức tối thiểu) + 0.5·(deadline − độ trễ), đã chuẩn hoá; vi phạm ràng buộc thì phạt.
DSAC chọn mode cùng ghép UAV–vệ tinh, UAV–người dùng, quỹ đạo và ISL. Mô hình sinh ảnh chỉ đi vào bài toán RL qua
bảng số trên (bit, chất lượng, thời gian tính).

Đọc kỹ các hình của bài:
- Hình 7–8: các đường SCE của 5 mode **song song, thứ tự không đổi**; M3_3 luôn cao nhất ở mọi deadline và mọi
  ngưỡng chất lượng được vẽ.
- Hình 9–11: DSAC hơn MAPPO / "UAV đứng yên" / "bỏ bớt mode" khoảng 0.001–0.002 SCE trên nền ≈ 0.03, không có thanh
  sai số. Hình 6 so hai episode.
- Không có đối chứng greedy hay heuristic cùng không gian hành động.

## 0A — với các task độc lập, chọn mode chỉ là tra bảng (`step0a.py`, `runs/huang_step0/step0a/`)
Khi các task không dùng chung tài nguyên, phần thưởng là tổng các điểm của từng task, và điểm mỗi task chỉ phụ thuộc
mode của chính nó và tốc độ link lúc đó. Vì vậy chọn mode tốt nhất cho từng task (greedy) **là tối ưu**; không cần RL.
Câu hỏi còn lại: với chính số liệu của bài, mode nào thắng và ở dải tốc độ nào việc chọn còn có ý nghĩa?

Link LEO→UAV tính đúng theo công thức (1)–(3) của bài (750 km, 25 GHz, ăng-ten 40 dBi, 1 W, 10 MHz):

| Nhiễu | SNR thiên đỉnh / góc 40° | Tốc độ trung bình |
|---|---|---|
| −130 dBm (bài ghi) | 22.1 / 18.8 dB | 65.8 / 55.4 Mbit/s |
| k·χ·B với χ = 354.81 K của chính bài (≈ −103 dBm) | −4.8 / −8.1 dB | 3.8 / 2.0 Mbit/s |

- Với M1 = 314 kbit (Hình 5): **M1 vừa đẹp hơn vừa nhanh hơn mọi mode sinh ảnh khi tốc độ > 0.52 Mbit/s**. Cả hai
  mức nhiễu đều vượt xa ngưỡng này → luôn chọn M1, chọn mode không còn ý nghĩa.
- Nếu M1 phải gửi ảnh gốc 3.5 MB (28 Mbit): M1 thống trị khi tốc độ > 46.8 Mbit/s; dưới ngưỡng đó M3_3 thắng gần như
  mọi task. Thứ tự "M3_3 cao nhất" ở Hình 7 chỉ khớp với trường hợp này cộng với link chậm.
- Trên phân bố task của bài (Qmin U[10,30], Dmax U[2,11] s), **greedy theo task bằng mode cố định tốt nhất** ở gần như
  mọi tốc độ; chỉ khác nhau trong dải chuyển tiếp hẹp (ví dụ 0.16 Mbit/s với payload Hình 5: 0.307 so với 0.173;
  15.8 Mbit/s với ảnh gốc: 0.343 so với 0.234).
- Hệ số outdated CSI δ = J0(2π f_D T) ≈ 0.006 ở Ka-band (Doppler LEO ~650 kHz) → kênh coi như Rayleigh hoàn toàn.

**Kết luận 0A:** trong thiết lập của bài, chọn mode theo từng task là bài toán tra bảng; một mode cố định đã gần tối
ưu ngoài vài dải tốc độ hẹp. Nếu RL có lợi thì phải đến từ ràng buộc nối các task theo thời gian (0B), không phải
từ việc chọn mode.

## Kiểm tra FloodNet-VQA (metadata, không tải ảnh; `runs/huang_step0/floodnet_audit/`)
Ứng viên tôi nêu trước đó ("câu hỏi đến sau khi truyền") — ChatGPT phản biện rằng ontology hẹp. Kiểm tra trên file
câu hỏi chính thức (Track 2, CC BY-NC-ND; file gốc không đưa vào git):
- Train 4 511 cặp hỏi–đáp trên 1 448 ảnh; valid 1 415 / 450 ảnh và test 1 429 / 450 ảnh **không có đáp án**.
- **Chỉ 15 câu hỏi khác nhau.** Loại: tình trạng 2 315 (3 mẫu câu), có/không 867 (2 mẫu), đếm đơn 636 (4 mẫu),
  đếm phức 693 (6 mẫu); mọi câu chỉ hỏi về nhà, đường hoặc cả ảnh.
- Một bit "ảnh có ngập" khớp đáp án tình trạng đường 96% và đáp án có/không 96%. Câu đếm đếm nhà (ngập / không ngập)
  → tính được từ mask phân đoạn (đếm vùng liên thông), sai khi các nhà dính nhau (chưa đo).
- Chỉ 398/1 448 ảnh train VQA có mask Track 1; ảnh valid/test không có mask.

**Kết luận:** FloodNet-VQA lặp lại đúng kiểu thất bại cũ — một bản tóm tắt vài byte do UAV tính sẽ trả lời gần hết
câu hỏi. **Loại** làm testbed cho "phải gửi ảnh". Lời phản biện của ChatGPT đúng; đề xuất của tôi trước đó vội.

## 0B — ràng buộc theo thời gian (`sim.py`, `policies.py`, `run_step0b.py`, `runs/huang_step0/step0b/`)
Mô phỏng: 3 UAV, 8 người dùng, 5 LEO bay qua (750 km, 7.8 km/s, góc ≥ 40°), 100 ảnh/episode đến theo Poisson.
Ba nguồn nối các task theo thời gian, bật/tắt từng cái:
- **hàng đợi**: UAV nhận từ một LEO và phát cho một người dùng tại một thời điểm; người dùng giải mã từng ảnh một;
- **tầm nhìn vệ tinh**: LEO nguồn khuất thì đi ISL (công thức 4) sang LEO đang thấy;
- **nhiễu**: các UAV phát cùng lúc gây nhiễu cho nhau (công thức 6).

Lưới: nhiễu {−130 dBm, k·χ·B} × payload M1 {Hình 5, ảnh gốc} × tải λ {0.5, 1, 2, 4 ảnh/s} × {đủ, tắt từng nguồn}
= 64 ô; 10 seed train để chỉnh λ của luật giá, 30 seed test. Policy nhân quả (chỉ biết tốc độ kỳ vọng, không biết
fading và task tương lai): 5 mode cố định, greedy (mode + UAV), greedy UAV theo vùng, rẻ nhất còn khả thi, giá trị −
giá (kiểu GOSC, λ chỉnh trên seed train), rollout có mô hình (lấy mẫu task tương lai). Mốc tham khảo: tìm kiếm
hindsight (coordinate descent, mô phỏng lại chính xác).

Kết quả chính (ô "đủ", trung bình utility/task trên 30 seed test):

| nhiễu | M1 | λ | mode cố định tốt nhất | greedy | causal tốt nhất | hindsight biết hết − causal tốt nhất |
|---|---|---|---|---|---|---|
| −130 dBm | Hình 5 | 0.5–4 | M1 0.578–0.581 | bằng | bằng | 0.000 |
| −130 dBm | gốc | 2 | M3_3 −0.049 | 0.093 | greedy vùng 0.210 | +0.070 [0.039, 0.101] |
| −130 dBm | gốc | 4 | M3_3 −0.123 | −0.174 | rollout 0.086 | +0.148 [0.094, 0.201] |
| k·χ·B | Hình 5 | 4 | M1 0.274 | 0.353 | rollout 0.382 | +0.193 [0.107, 0.278] |
| k·χ·B | gốc | 1 | M3_3 −0.042 | −1.485 | giá −0.042 | +0.102 [0.090, 0.113] |

1. Khi M1 nhỏ và link nhanh (đúng tham số bài ghi): **một mode cố định = greedy = hindsight**. Không có gì để tối ưu.
2. **Nguồn nối theo thời gian quan trọng là hàng đợi chung.** Tắt hàng đợi, greedy chỉ kém hindsight-biết-hết
   0.01–0.04 ở ba ô đầu; nhiễu gần như không đổi kết quả. Ngoại lệ: k·χ·B / ảnh gốc / λ = 1 vẫn chênh 0.13 so với
   causal tốt nhất khi tắt hàng đợi — phần này đến từ việc biết fading (gửi M1 đúng lúc kênh LEO tốt), không từ
   hàng đợi.
3. Khi tải cao và M1 lớn, **greedy cận thị sụp** (một ảnh M1 chặn hàng đợi của các ảnh sau; ví dụ −1.49 so với −0.04
   của luật giá). Các luật rẻ (giá, rollout, chọn UAV theo vùng) cứu được phần lớn, nhưng **không luật nào thắng ở mọi
   ô** — rollout, giá, greedy theo vùng, "rẻ nhất khả thi" lần lượt đứng đầu ở các ô khác nhau.
4. Khoảng cách tới hindsight-biết-hết còn 0.07–0.19 ở các ô tải cao, nhưng mốc này biết cả fading thực tế — điều không
   policy nào có (CSI LEO lỗi thời hoàn toàn, δ ≈ 0.006) — nên chưa đo được phần RL có thể lấy.

**Sai sót của tôi trong cổng đã đăng ký trước.** Cổng so với "hindsight chỉ biết task tương lai" — một kế hoạch lập một
lần, dùng tốc độ kỳ vọng, không lập lại khi kênh thực khác dự đoán. Dưới fading thật, kế hoạch đó gây trễ dây chuyền và
thường **kém hơn cả heuristic** (đến −2.3), nên không phải cận trên; kết quả "0/64 ô qua cổng" vì vậy **không có giá
trị**. Sửa ở 0B-2 bên dưới (ghi rõ là sửa sau khi chạy).

## 0B-2 — mốc vòng kín biết trước task (sửa mốc, sau khi chạy; `run_step0b2.py`, `runs/huang_step0/step0b2/`)
Planner "foresight" biết **10 task thật tiếp theo** (giờ đến, nguồn, người nhận, deadline, ngưỡng chất lượng), dùng tốc
độ kỳ vọng, lập lại kế hoạch ở mỗi quyết định từ trạng thái thực. Đối chứng cùng cấu trúc: rollout nhân quả lấy mẫu
task tương lai. 16 ô "đủ", 30 seed test, ~31 phút CPU.

- Ở 12 ô Hình 5 / −130 dBm-gốc / k·χ·B-Hình 5: foresight − causal tốt nhất nằm trong **[−0.05, +0.02]**, mọi khoảng tin
  cậy chứa 0 hoặc âm. **Biết trước 10 task thật không thêm gì đo được.**
- Ở 4 ô k·χ·B / ảnh gốc: foresight **kém** causal tốt nhất 0.16–0.38, vì nó tiếp tục bằng greedy — đúng chính sách sụp ở
  vùng này. Nghĩa là mốc này cũng **không phải cận trên** ở đây; chỉ cho thấy biết trước task không cứu được một
  continuation kém.
- 0/16 ô qua cổng.

## Kết luận Bước 0
1. Chọn mode như bài Huang là **tra bảng** khi các task độc lập (0A). Theo đúng tham số bài ghi, JPEG2000 thống trị;
   mode sinh ảnh chỉ thắng khi link chậm hơn ~0.5 Mbit/s (payload Hình 5) hoặc ~47 Mbit/s (nếu M1 phải gửi ảnh gốc).
2. Bài toán nhiều bước chỉ xuất hiện khi **nhiều task tranh hàng đợi chung** và M1 lớn. Ở đó greedy cận thị sụp, nhưng
   luật rẻ (giá trị − giá, rollout có mô hình, chọn UAV theo vùng) cứu được.
3. **Không tìm thấy bằng chứng còn ≥ 0.02 utility để RL lấy** so với luật tốt nhất: biết trước task tương lai không
   giúp; phần cách biệt tới hindsight-biết-hết (0.07–0.19) chưa tách được khỏi việc biết fading. Điểm duy nhất ủng hộ
   học: **không luật nào tốt nhất ở mọi ô** — nhưng một luật chọn heuristic theo tải/link cũng có thể làm việc đó.
4. Giới hạn: môi trường đơn giản hoá (không có quỹ đạo UAV — UAV đứng yên; không chọn ISL; ghép LEO theo luật cố
   định); hai mốc tham khảo đều không phải cận trên chặt; 30 seed; chưa so DSAC/SAC thật.

**Hệ quả cho hướng nghiên cứu:** đưa RL vào bài toán chọn mode kiểu Huang không có cơ sở từ phép thử này. Nếu đi tiếp
theo Huang, phần "thật sự cần quyết định theo thời gian" phải được tạo ra có chủ đích (băng thông khan hiếm, nhiều
task tranh tài nguyên, di chuyển UAV làm đổi kênh tương lai) và kiểm lại bằng cùng phép thử này trước khi train.
