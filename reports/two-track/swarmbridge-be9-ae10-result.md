# B-E9 (hạn mức event) và A-E10 (warm-start lệch regime): kết quả (review #35)

Ngày: 2026-11-16.

Code:
- `swarmbridge/experiments/jepa_semcom/b4_roi.py`: thêm `evq`, `gtq`, `imp_event`, `offsets_quota`.
- `swarmbridge/experiments/uav_semcom/a10_hybrid.py`: thêm tùy chọn `calreg=`.
- `swarmbridge/experiments/uav_semcom/ae10_summary.py`.

Số liệu:
- `runs/jepa_semcom/b9/` (`b9_dev_eval.json`, `b9_eval.json`, `dev_run_stats.json`, `eval_run_stats.json`).
- `runs/uav_semcom/a16/` (`ae10_*.json`, `ae10_summary.json`).

Mục Method của Paper A đã sửa theo review #35 ở `reports/swarmbridge-paperA-method-draft.md` (v2, có changelog).

---

## 1. B-E9: hạn mức block riêng cho sự kiện, đổi lớp được tính

### 1.1 Luồng làm

Mọi thứ giữ nguyên như B-E8:
- adapter x265 (offset QP cho từng block 16×16);
- cấu hình HEVC low-delay;
- receiver DINOv2-S và head cố định;
- so sánh ở cùng số bit thực: quét CRF 30–45, nội suy log(bit) theo từng clip;
- 10 clip đánh giá.

Có hai thay đổi, đúng như review yêu cầu:

1. **Bản đồ sự kiện theo từng lớp.**
   - Công thức:
     $$
     E_t(i) = \max_c \left| p_{t,c}(i) - \tilde p_{t-1,c}(i) \right|
     $$
     - $p_c$: sigmoid của từng lớp;
     - $\tilde p_{t-1}$: bản đồ của frame trước, đã dịch theo chuyển động camera toàn cục (phase correlation, chỉ dùng
       frame ≤ t).
   - Một patch đổi từ car 0,9 sang person 0,9 cho $E \approx 0{,}9$; bản đồ cũ của `taskev` cho 0. Test
     `test_event_map_sees_class_swap_and_quota_is_disjoint` xác nhận điều này.
   - Test nhân quả: đổi frame hoặc score tương lai không làm thay đổi bản đồ quá khứ.
2. **Hạn mức event riêng (`evq`).**
   - Tập task giữ như cũ: top 15 % block theo $p_t$, A = 4.
   - Cộng thêm tối đa q·700 block **nằm ngoài tập task**, xếp theo $E_t$ (điều kiện $E > 0{,}1$).
   - Cả hai tập nhận −A; các block còn lại nhận +A·k/(n−k), nên offset trung bình bằng 0.
   - Không phát sinh bit phụ (offset nằm trong bitstream).

Thêm một bản chẩn đoán **`gtq` (dùng GT, không triển khai được)**: cùng hạn mức, nhưng block event là các patch có nhãn
GT 4 lớp khác frame trước, tức đúng các patch mà `ap_change` chấm. Bản này cho biết hạn mức kiểu này đạt tối đa được bao
nhiêu nếu biết chính xác patch nào thay đổi.

**Giao thức.**
- Chỉ thử 2 cấu hình, q ∈ {0,05; 0,10}, với A và f giữ như task.
- Chọn trên 4 clip dev (TUNE) theo `ap_change` trung bình ở 6/9/12k; cả `evq` và `gtq` đều chọn được q = 0,10.
- Sau đó chạy một lần trên 10 clip đánh giá, cùng với uniform, task (A4 f0,15) và taskev cũ (A4 f0,15).
- Kiểm tra tái lập: ở 9k, AP của uniform và task trùng tới từng chữ số với B-E8, nên encoder là tất định.

### 1.2 Hạn mức event có thực sự được dùng không

| (eval, 10 clip) | evq q = 0,10 | gtq q = 0,10 (GT) |
|---|---|---|
| block task / frame | 15 % | 15 % |
| block event thêm / frame | **9,2 %** | 4,2 % |
| tỉ lệ event trong các block được ưu tiên | **38 %** | 20 % |
| frame có block event | 99 % | 90 % |
| top-E đã nằm sẵn trong tập task | 46 % | 57 % |

Lần này cơ chế sự kiện được kiểm nghiệm thật: khoảng 38 % block được ưu tiên đến từ hạn mức event. B-E8 thì khác: taskev
cho bitstream trùng task trên 7/10 clip.

### 1.3 Kết quả trên tập đánh giá

Hiệu so với task-only, cùng số bit thực, CI t-test trên clip:

| | 4,5k | 6k | 9k | 12k |
|---|---|---|---|---|
| **evq − task, ap_change** | +0,008 [−0,000, +0,016] | **+0,009 [+0,001, +0,017]** | +0,000 | −0,002 |
| gtq − task, ap_change (GT) | +0,002 | **+0,008 [+0,002, +0,015]** | +0,006 [−0,000, +0,012] | +0,003 |
| evq − task, ap_event | +0,003 | +0,008 [−0,000, +0,015] | +0,001 | |
| evq − task, AP | −0,003 | +0,002 | −0,000 | |
| taskev (cũ) − task, ap_change | +0,006 | +0,003 | −0,002 | −0,001 |

So với HEVC đồng đều:
- **evq ap_change:** +0,011 (4,5k, CI chứa 0); **+0,011 [+0,001, +0,020] (6k)**; +0,003 (9k); +0,001 (12k).
- **task ap_change:** +0,002 đến +0,004, CI đều chứa 0 (giống B-E8).
- **evq ap_event:** +0,013 [+0,002, +0,024] (4,5k); +0,016 [+0,008, +0,024] (6k).
- **evq AP:** +0,010 / +0,008 (CI > 0), ngang task.

**Tiết kiệm bit ở cùng ap_change.** Chỉ tính các clip có đường cong đơn điệu (5–7/10 clip):

| | 6k | 9k |
|---|---|---|
| evq | **15 % [9, 22]** | **14 % [0, 27]** |
| task | ≈ 0 % | ≈ 0 % |
| gtq | 12 % | 18 % |

**PSNR** của evq thấp hơn task 0,1 dB và thấp hơn uniform khoảng 0,3 dB.

**Chi phí phía gửi.**
- Bản đồ event: 1,6 ms/frame (numpy).
- Cộng thêm DINOv2-S + head: 8–10 ms/frame (RTX 3070, batch 1, fp16).

### 1.4 Đọc kết quả, kể cả mặt tiêu cực

1. **Ở mức bit thấp, hạn mức event có tác dụng trên patch thay đổi.**
   - ap_change cao hơn task khoảng +0,009 ở 6k (CI > 0), tương đương tiết kiệm khoảng 15 % bit ở cùng ap_change.
   - Mức này ngang bản chẩn đoán dùng nhãn GT (`gtq`: +0,008).
   - Nghĩa là bản đồ nhân quả theo lớp đã bắt được gần hết phần có thể khai thác bằng kiểu phân bổ này.
2. **Trần của cách tiếp cận này rất thấp.**
   - Ngay cả khi biết chính xác patch nào thay đổi (GT), lợi ích chỉ khoảng +0,01 ap_change.
   - Để so sánh: chênh lệch latent-vs-video ở B-E7 là 0,04–0,09.
   - Lý do: 57 % patch thay đổi theo GT đã nằm sẵn trong ROI của task, vì vật thể di chuyển thì đã được ưu tiên. Phần
     còn lại ít, và QP chỉ chỉnh được tới đó.
3. **Không bền.**
   - Lợi ích biến mất từ 9k trở lên.
   - Trên tập dev, evq còn **kém** task về ap_change (−0,002 ở 6k, −0,011 ở 9k), trong khi gtq ≈ task.
   - Dấu dev và eval ngược nhau. Hai mức bit và ba chỉ số được xem cùng lúc, nên CI > 0 ở một điểm 6k cần đọc thận
     trọng (chưa hiệu chỉnh so sánh bội).
4. Tổng AP không đổi so với task, và PSNR giảm thêm 0,1 dB.

**Kết luận theo quy tắc của review #35:**
- Cơ chế sự kiện giờ đã được kiểm nghiệm thật.
- Lợi ích có thật nhưng nhỏ (khoảng 0,01), chỉ ở mức bit thấp, và đúng bằng trần của bản GT.
- Đề xuất **dừng tối ưu bản đồ QP và chuyển sang hướng (a)**: lớp tăng cường ngữ nghĩa trên nền video.
- `evq` nên giữ làm base layer / baseline mạnh nhất cho (a), vì nó là phương án "chỉ dùng video" tốt nhất ở mức bit
  thấp.

Giới hạn:
- 10 clip đánh giá, mỗi clip ≤ 100 frame; 4 clip dev;
- chỉ 2 cấu hình q; A và f cố định;
- block 16 px, trong khi nhóm lượng tử của x265 là 32×32;
- chỉ bù chuyển động toàn cục với bước nguyên;
- head được fit trên TUNE (điểm dev lạc quan);
- tiết kiệm bit chỉ tính trên 5–7 clip.

---

## 2. A-E10: warm-start lệch regime (bài stress tùy chọn của review #35)

### 2.1 Luồng làm

Bài này kiểm tra trực tiếp điểm phụ thuộc lớn nhất mà A-E9 tìm ra.

- **Tùy chọn mới `calreg=NAME`.** 10 episode hiệu chỉnh của giao thức "pre" chạy trong môi trường của **regime khác**.
  Heuristic, tập ảnh val và seed 9000–9009 giữ nguyên. Không có cơ chế mới và không tinh chỉnh gì.
- **Hai trường hợp:**
  - hiệu chỉnh ở ref (PHY Shannon), chạy ở phy_mcs_blind (MCS);
  - hiệu chỉnh ở phy_mcs_blind, chạy ở ref.
- **Bộ điều khiển:** bản chính cals np-any 50 ms. Ở phy chạy thêm Shannon np-pess.
- **Seed:** 8500–8519 như A-E9, ghép cặp theo seed với các lượt A-E9 (PPO, warm-start đúng regime, cold). PPO cho
  return trùng từng giá trị với A-E9.

### 2.2 Kết quả

Return so với PPO, CI theo seed môi trường:

| Test regime | Bộ điều khiển | Warm-start đúng (A-E9) | Cold (A-E9) | **Warm-start lệch** | Lệch − đúng |
|---|---|---|---|---|---|
| phy_mcs_blind | cals np-any | +2,18 [+1,30, +3,06] | +0,65 | **−6,37 [−8,33, −4,41]** | −8,55 [−10,4, −6,7] |
| phy_mcs_blind | Shannon np-pess | +1,32 | – | **−7,93 [−9,95, −5,92]** | −9,25 |
| ref | cals np-any | +1,91 [+1,42, +2,39] | +0,96 | **−12,10 [−14,5, −9,7]** | −14,00 [−16,4, −11,6] |

- **Đồng nhất:** cả 3 checkpoint đều âm (phy: −6,3 / −6,8 / −6,0; ref: −13,2 / −10,5 / −12,6).
- **Số lần chuyển khỏi PPO tăng mạnh:** 704 → 1956 (phy); 504 → 995–1216 (ref).
- **Thời gian quyết định không đổi:** p99 37–44 ms.

**Tracker không phát hiện ra mô hình đã sai.**
- Hiệu chỉnh ở ref, chạy ở phy: bias median −0,03, gần như "mô hình đúng".
- Hiệu chỉnh ở phy, chạy ở ref: bias median +0,15, phát hiện đúng hướng (thực tế tốt hơn dự đoán), nhưng vẫn để
  chuyển mode sai.

### 2.3 Đọc kết quả

1. **Đây là điểm yếu nghiêm trọng của phương pháp hiện tại, không chỉ là một lưu ý nhỏ.**
   - Warm-start lệch regime làm bộ điều khiển kém hơn PPO 6–12 điểm.
   - Nó còn kém hơn hẳn bản cold (−7,0 và −13,1 so với cold, CI < 0).
   - Cổng quyết định không chặn được.
2. **Nguyên nhân dễ hiểu từ thiết kế.**
   - Tracker và calibrator **gộp chung** dữ liệu hiệu chỉnh đã đóng băng (10 episode, hàng nghìn báo cáo) với vài
     trăm báo cáo trực tuyến của episode đang chạy. Dữ liệu cũ áp đảo, nên bias và se phản ánh regime cũ.
   - Khoảng nhạy luôn chứa thang 1, tức mô hình danh định cũ.
   - Sai lệch giữa MCS và Shannon không phải hệ số nhân đều: MCS bão hòa ở CQI cao nhất. Vì vậy việc nhân rate theo
     một thang chung không bao được sai số.
3. **Hệ quả cho bài báo.**
   - Phương pháp chỉ an toàn khi warm-start được hiệu chỉnh **trên cùng cấu hình PHY/kênh** với lúc chạy.
   - Nếu không bảo đảm được điều đó, bản cold an toàn hơn (+0,65 / +0,96, không bao giờ kém PPO ở mức seed).
   - Phải ghi rõ trong Method và Limitations. Câu "robust với feedback trễ/mất" ở A-E9 cũng phải đọc kèm: nó chỉ đúng
     khi warm-start khớp regime.
4. **Cách sửa tự nhiên nhưng CHƯA làm, vì review #35 bảo không thêm cơ chế:**
   - quên dần dữ liệu cũ (trọng số giảm theo thời gian);
   - hoặc tách dữ liệu warm-start và dữ liệu trực tuyến, rồi bỏ warm-start khi hai nguồn mâu thuẫn rõ.

   Việc có làm hay không là quyết định của phản biện.

Giới hạn:
- Mới thử hai cặp regime khác nhau rất xa (Shannon ↔ MCS), là trường hợp xấu nhất.
- Chưa thử lệch nhỏ, ví dụ khác môi trường đô thị hoặc khác hệ số K của fading.
