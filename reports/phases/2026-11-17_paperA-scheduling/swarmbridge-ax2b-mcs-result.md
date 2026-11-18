# A-X2b (phản biện #38): scheduler kênh học được, train trực tiếp trên MCS

Ngày chạy: 2026-11-17. Kết quả trong `runs/uav_semcom/ax2/`: `eval_K5_mcsg*`, `stress_mcsgC_on_K5_mcs*`,
`train/K5_mcsg_c_s*.json`. Heuristic tune: `runs/uav_semcom/a5/tune/K5_mcs_l{75,50,25}.json`,
`runs/uav_semcom/ax1/tune/K5_mcsg.json`. Tiếp nối [swarmbridge-ax2-result.md](swarmbridge-ax2-result.md).

## Luồng

1. **Train.** C-only như A-X2: policy chọn kênh, còn di chuyển, công suất và mode khoá theo T:best của `K5_mcsg`
   (`fixed_mbt2018-mean_q3`, không dùng chung kênh). Train trên `K5_mcsg` (MCS với link adaptation biết nhiễu), 3
   seed, 2M bước, chọn checkpoint theo val.
2. **Baseline chọn kênh công bằng.** Tune họ SCH (A-X1, 16 luật) trên `K5_mcsg`; luật được chọn là `SCH:a0:s1:none`.
3. **Eval.** 100 seed mới 8800–8899.
4. **Thang robustness, zero-shot.**
   - Cùng controller, chạy trên `mcs_partial` với λ = 0,75 / 0,5 / 0,25. Link adaptation chỉ thấy λ phần nhiễu đồng
     kênh; block hỏng khi SINR thực thấp hơn ngưỡng CQI đã chọn.
   - Chạy thêm trên `K5_mcs` (λ = 0, hoàn toàn không biết nhiễu).
   - Heuristic được tune lại cho từng λ (họ A5 đầy đủ, 30 seed train). Ở mọi λ, luật tốt nhất vẫn là mode cố định,
     không dùng chung kênh.

## Kết quả (return, CI 95% trên 100 seed môi trường)

| Môi trường eval | T:best tune cho môi trường đó | C-only (train trên K5_mcsg) | Δ [CI] | Theo seed train |
|---|---|---|---|---|
| **K5_mcsg (λ = 1, link adaptation lý tưởng)** | −0,51 | **+10,77** | **+11,28 [+9,76, +12,79]** | +11,9 / +9,2 / +12,8 |
| λ = 0,75 | −0,51 | −32,98 | −32,47 [−34,57, −30,36] | −16,1 / −61,9 / −19,3 |
| λ = 0,50 | −0,51 | −106,23 | −105,72 [−109,64, −101,79] | −52,1 / −191,0 / −74,1 |
| λ = 0,25 | −0,51 | −125,62 | −125,10 [−129,45, −120,76] | −61,8 / −220,8 / −92,7 |
| λ = 0 (K5_mcs) | −0,51 | −131,75 | −131,24 [−135,77, −126,71] | −63,5 / −231,6 / −98,7 |

Các chỉ số khác:
- **Trên K5_mcsg so với SCH\*:** +11,46 [+9,94, +12,97].
- **ok rate:** 0,954 so với 0,944; ở λ = 0,75 tụt còn 0,880.
- **Recall detector:** K5_mcsg 0,547 so với 0,541; λ = 0,75 còn 0,503; λ = 0 còn 0,371.
- **Năng lượng** ngang nhau (66–69 kJ).
- **Tỉ lệ thời gian phát có dùng chung kênh:** khoảng 0,70–0,74 ở mọi λ, nghĩa là controller không đổi hành vi khi
  điều kiện kênh đổi. Heuristic: 0.
- **Độ trễ:** scheduler mỗi slot p99 3,0–3,7 ms.

So với các cách học khác trên `K5_mcsg` (cùng seed, từ báo cáo trước):

| | Δ so với T:best |
|---|---|
| PPO joint | +0,63 [−1,15, +2,40] |
| PPO mode-only | −0,37 [−0,85, +0,11] |
| **C-only (scheduler học được)** | **+11,28 [+9,76, +12,79]** |

## Đọc kết quả

1. **Với link adaptation lý tưởng, học chọn kênh cho lợi ích lớn và ổn định** (+11, cả 3 seed). Mọi heuristic dùng
   chung kênh đều thua khi tune; scheduler học được tìm ra cách dùng chung kênh có lợi. Đây là bằng chứng mạnh nhất
   từ trước đến nay cho một thành phần học được trong Paper A. Nó cũng lớn hơn nhiều so với học mode hoặc học joint.
2. **Nhưng lợi ích này hoàn toàn dựa vào giả định link adaptation lý tưởng.**
   - Chỉ cần link adaptation bỏ sót 25% nhiễu (λ = 0,75), controller đã thua heuristic không dùng chung kênh tới −32.
   - Controller không tự nhận ra điều kiện đã đổi: tỉ lệ dùng chung kênh giữ nguyên khoảng 0,7.
   - Heuristic không dùng chung kênh thì không bị ảnh hưởng ở bất kỳ λ nào.
3. **Giới hạn của thang λ.** λ < 1 là mô hình thô: luôn đánh giá thấp nhiễu theo một tỉ lệ cố định và không có biên an
   toàn kiểu outer-loop link adaptation (OLLA, nhắm BLER 10%), cũng không có HARQ. Hệ thống thực nằm giữa λ = 1 và mô
   hình này. Vì vậy **không được** kết luận "dùng chung kênh vô dụng trong thực tế". Kết luận đúng là: lợi ích của
   scheduler học được phụ thuộc vào mô hình link adaptation, và phải được kiểm chứng với link adaptation thực tế hơn.
4. **Nối với nhận xét của phản biện #38.**
   - MPC sửa được việc chọn mode khi mô hình tốc độ sai, nhưng không sửa được việc dùng chung kênh sai.
   - Giờ thêm: chính scheduler học được cũng không thích ứng khi điều kiện link adaptation đổi.
   - Đây là động lực cụ thể để Paper A xét **tính thích ứng của resource scheduling**, không chỉ của việc chọn mode.

## Đề xuất bước 3 (chưa làm; cần phản biện duyệt)

**Vật lý cần lưu ý.**
- Ở 2,4 GHz và 20 m/s, tần số Doppler khoảng 160 Hz. Fading nhỏ (small-scale) mất tương quan sau vài ms, nên với slot
  1 s, CSI trễ một slot không có giá trị cho fading nhỏ. Đây là lý do hiện fading được rút độc lập theo slot.
- Thứ có tương quan theo giây là **thành phần quy mô lớn**: shadowing và trạng thái LoS/NLoS, với khoảng cách mất tương
  quan cỡ 20–50 m (tương đương 1–2,5 s ở 20 m/s), cùng với **nhiễu đồng kênh do lịch phát của slot trước**.
- Vì vậy "CSI trễ" có nghĩa nên được mô hình như sau, trong simulator hiện có, không dựng lại:
  - (i) Shadowing có tương quan theo khoảng cách (mô hình Gudmundson), và trạng thái LoS có tương quan theo thời gian.
  - (ii) Link adaptation dùng CQI/SINR đo ở slot trước (bao gồm nhiễu của lịch phát slot trước), cộng biên OLLA nhắm
    BLER 10%.
  - (iii) Controller quan sát được SINR/ACK của slot trước.

**Thí nghiệm đề xuất.**
- (A) Đánh giá lại T:best, SCH\* và C-only hiện tại trên môi trường link adaptation thực tế nói trên (zero-shot), để
  biết điểm vận hành thực nằm ở đâu trên thang λ.
- (B) Train C-only trên môi trường đó, kèm ngẫu nhiên hoá mức độ thực tế của link adaptation (λ hoặc biên OLLA) theo
  episode, để scheduler học được **khi nào** dùng chung kênh an toàn. Eval trên cả thang λ.
- (C) Chỉ sau (A)/(B), so sánh thuật toán RL phù hợp với hành động gán kênh có cấu trúc tổ hợp: PPO phân rã hiện tại,
  so với multi-agent PPO chia sẻ tham số theo UAV, so với policy tự hồi quy (gán lần lượt từng kênh). Chọn một hoặc
  hai thuật toán, cùng ngân sách train.

Không viết lại Method trước khi có (A)/(B).
