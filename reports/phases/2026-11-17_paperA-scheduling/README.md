# Paper A, giai đoạn lập lịch + LA thực tế + tiện ích tác vụ (17–18/11, phản biện #37–#47)

**Trạng thái: ĐÓNG** (phản biện #47).
- Bài toán là offloading ảnh độc lập: 5 UAV, 2 kênh, BS.
- Bên nhận không có trạng thái, các UAV không hợp tác.
- Tài sản giữ lại: bộ vô tuyến LA (`env_la`), scheduler JA, bộ hiệu chỉnh tốc độ từ ACK.

**Câu hỏi:** dưới link adaptation thực tế, RL lập lịch / chọn mode có hơn heuristic tune công bằng + MPC không? Thông
tin tiện ích từng ảnh có giá trị không?

**Kết luận:**
- Dưới LA thực tế, PPO (kể cả train đúng PHY) chỉ ngang heuristic + MPC (A-X9: −0,14, dừng DRL).
- Tiện ích từng ảnh có dư địa trên nội dung cùng nguồn (A-T1: +1,71), nhưng không tổng quát sang video mới, và không
  bộ dự đoán nhân quả nào lấy lại được (bước C).

## Dòng thời gian

| Vòng (phản biện) | Câu hỏi | Kết quả chính | Báo cáo |
|---|---|---|---|
| A-X2 (#37) | Học chọn kênh (C-only) | +2,52 (K5) / +3,07 (ref); học thêm mode không ổn định; luật kiểu bài báo thua +10…+170; khảo sát 19 bài: khác biệt chính là baseline đã tune | [ax2](swarmbridge-ax2-result.md) |
| A-X2b (#38) | C-only train trên MCS biết nhiễu | +11,28; λ = 0,75 → −32, λ = 0 → −131 | [ax2b](swarmbridge-ax2b-mcs-result.md) |
| A-X5 (#39) | Link adaptation thực tế + train ngẫu nhiên hoá | Không dùng chung kênh tốt nhất ở cả 7 điều kiện; C-only ngẫu nhiên hoá −42,9 (dùng chung kênh khoảng 0,86, không thích ứng); C-only danh nghĩa ≈ T:best; cổng NACK cứu nhưng vẫn thua | [ax5](swarmbridge-ax5-la-result.md) |
| A-X6 (#41) | Sửa PHY, link adaptation biết lịch, cơ hội lập lịch | Sửa block không đổi A-X5 (≤ 2 điểm); chọn UAV (JA, không dùng chung kênh) +8,2 nhân quả; dùng chung kênh −5…+6 tuỳ giả định; biết trước kênh +12,5 (không nhân quả); K5_mcsg: JA không dùng chung kênh +12,4 ≈ RL A-X2b | [ax6](swarmbridge-ax6-opportunity-result.md) |
| A-X7 (#42) | Tách cơ hội thông tin, kênh tương quan, bộ điều khiển mode trên PHY mới | Oracle chỉ LoS khoảng 0 (khoảng cách oracle là fading nhỏ); bộ lọc LoS tương quan +0; SRS +3; JA ≥ mọi RL lập lịch; PPO–MPC hiệu chỉnh +12…+17 so với PPO; tương tác lập lịch × mode +4…+5; JA + PPO–MPC hơn heuristic tốt nhất +13,3 (heuristic chưa tune cùng JA) | [ax7](swarmbridge-ax7-information-mode-result.md) |
| A-X8 (#43) | Đối chứng công bằng cho tổ hợp lập lịch + mode | PPO–MPC ≈ heuristic tune cùng JA và cách bay (−1,25 n.s.); MPC lấy heuristic làm incumbent tốt nhất (+5,0 so với PPO–MPC); JA +8…+16 dưới mọi mode; hiệu chỉnh luật thiển cận không giúp; tương tác không ổn định; latency tích hợp p99 ≤ 46 ms | [ax8](swarmbridge-ax8-fair-baseline-result.md) |
| A-X9 (#44) | Phép thử DRL cuối (protocol đăng ký trước): PPO train trên LA làm incumbent cho MPC | (PPO-LA + MPC) − (heuristic + MPC) = −0,14 [−2,83, +2,56], recall −0,031 → cổng FAIL 4/5, **dừng nhánh DRL**; PPO-LA hơn PPO-Shannon +16,8 nhưng ≈ H\*; hiệu chỉnh +17 so với MPC danh nghĩa | [ax9](swarmbridge-ax9-final-drl-test-result.md) |
| A-T1 (#45) | Thông tin tiện ích detector từng ảnh đáng bao nhiêu trong mạng (oracle, kênh nhân quả) | Kiểm chéo +1,71 [1,29, 2,14] so với q3 + MPC → cổng qua; ≈ toàn bộ là biết số vật thể (+1,53), qua giảm tắc nghẽn; vượt Qmin (−2,15 recall tuân thủ Qmin); MPC mục tiêu TP không dùng được thông tin | [at1](swarmbridge-at1-task-utility-result.md) |
| Bước C (#46) | Bộ dự đoán số vật thể nhân quả (payload / thống kê ảnh / detector nhẹ) trên test-dev; chẩn đoán Qmin | Cổng FAIL: pC +0,20 [−0,94, +1,63], recall theo ảnh −3,0; oracle số vật thể −0,97 trên nội dung mới; oracle detector +2,17 → dừng hướng | [atc](swarmbridge-atc-step-c-result.md) |

## Tóm tắt theo phản biện (chép từ README chính ngày 18/11)

**Paper A** = DRL cho truyền ảnh ngữ nghĩa từ nhiều UAV, khi mô hình kênh sai và thời gian quyết định có hạn. Có hai phần.

**(a) Bộ điều khiển mode** (Method v2): PPO chọn mode, kết hợp MPC hiệu chỉnh trực tuyến, chạy có chọn lọc trong
ngân sách 50 ms.
- Hơn PPO +1,2…+2,2 return trên seed mới (A-E9, A-E11), bền với feedback trễ/mất và, sau khi sửa ở A-E11, với warm-start lệch.
- Chỉ đã kiểm trên PHY Shannon / MCS cấp slot. **Chưa thử dưới link adaptation thực tế.**

**(b) Lập lịch kênh** (A-X):
- Scheduler học được hơn heuristic tune khi PHY lý tưởng: +2,5 / +3,1 (Shannon), +11,3 (MCS biết nhiễu, A-X2b).
- **A-X6 đọc lại A-X2b:** một heuristic gán tổ hợp kênh, nhân quả, không dùng chung kênh, đạt +12,4 trên K5_mcsg (ngang
  RL). Lợi ích RL ở đó chủ yếu là chọn UAV nào phát, không phải dùng chung kênh.
- Với link adaptation thực tế:
  - mù lịch: dùng chung kênh luôn hại;
  - biết lịch (nhân quả): lợi ích riêng của dùng chung kênh nhỏ và không ổn định (−5…+6);
  - chọn UAV tốt hơn: +6…+8;
  - biết trước kênh của slot tới (không nhân quả, vì fading và LoS rút độc lập mỗi slot): thêm +12,5.

**A-X7 (phản biện #42):**
- Khoảng +12,5 của oracle gần như toàn bộ là fading nhỏ bị đóng băng theo slot (oracle chỉ LoS: khoảng 0). LoS chỉ
  khoảng 3 % ở hình học này.
- LoS tương quan theo quãng bay và bộ lọc nhân quả không thêm return. Probe SRS giúp +3 cho mọi policy.
- JA (có dùng chung kênh) ngang hoặc hơn mọi kết quả RL lập lịch (A-X2 K5 +1,3, ref +0,8; A-X2b +4,2).
- **Bộ điều khiển mode trên PHY mới:**
  - PPO–MPC hiệu chỉnh online hơn PPO zero-shot +12…+17;
  - tương tác lập lịch × mode +4…+5;
  - JA + PPO–MPC hơn tổ hợp heuristic tốt nhất +13,3, **nhưng heuristic mode chưa được tune lại cùng JA**.

**A-X8 (phản biện #43, đối chứng công bằng):**
- Khi heuristic mode được tune cùng JA và cùng tập cách bay: PPO–MPC − heuristic tốt nhất = −1,25 (n.s., paced).
- Stack tốt nhất là MPC chọn lọc hiệu chỉnh online lấy heuristic (mode cố định) làm incumbent; PPO làm incumbent thua
  −5,0.
- JA bền: +8…+16 dưới mọi mode.
- Hiệu chỉnh cho luật mode thiển cận không giúp.
- Tương tác lập lịch × mode không ổn định (−2,4 / +5,4).
- Lợi thế +13,3 của A-X7 là so với baseline yếu.

**A-X9 (phản biện #44, protocol đăng ký trước, seed test mới 8300–8399): DỪNG nhánh DRL.**
- PPO mode-only train trên LA (3 seed, 2M bước, quan sát LA nhân quả, cùng JA / cách bay / hàm thưởng) làm incumbent
  cho MPC hiệu chỉnh: (PPO-LA + MPC) − (heuristic + MPC) = **−0,14 [−2,83, +2,56]** (theo seed +0,75 / −0,71 / −0,45).
- Recall detector thấp hơn −0,031 [−0,034, −0,027]. Cổng FAIL 4/5.
- Train đúng PHY sửa sai lệch phân phối (+16,8 so với PPO-Shannon) nhưng chỉ ngang heuristic tốt nhất.
- Return và recall không đồng hướng: PPO-LA cùng return với H\* nhưng −4,9 điểm recall (mode rẻ, ít drop).
- MPC không hiệu chỉnh tệ hơn cả không dùng MPC (−16,95 so với có hiệu chỉnh).
- Lập lịch đóng băng: JA:stat:R:a1:m3 @ pacedB3 (dùng chung kênh lề 3 dB, chỉ +0,5…+1,7 so với không dùng chung).
- Theo quy tắc: không SAC / DSAC / PPO tự hồi quy / MAPPO. Không viết lại Method.
- JA + MPC chưa phải tính mới. Cần chọn một cơ chế mới có giá trị cho tác vụ trước; ứng viên trong
  [báo cáo A-X9](swarmbridge-ax9-final-drl-test-result.md#quyết-định-theo-quy-tắc-đã-đăng-ký).

**Phản biện #45 (18/11): rà soát tính mới + A-T1.**
- **Rà soát tính mới** ([bản rà soát](swarmbridge-task-utility-novelty-review-2026-11-18.md)):
  - Không có khoảng trống thuật toán nào được bảo vệ chắc chắn.
  - MPC hiệu chỉnh online, giới hạn tính toán đã có ở ABR (RobustMPC, Oboe, Fugu).
  - Ước lượng giá trị ảnh không nhãn đã có (ORIC, DAO).
  - Còn lại: giả định LA thay cho Shannon, và độ chặt của đánh giá.
- **A-T1** ([báo cáo](swarmbridge-at1-task-utility-result.md)): oracle biết TP thật từng ảnh, kênh vẫn
  nhân quả, seed mới 9200–9299.
  - So với q3 + MPC: +3,00; kiểm chéo bằng ảnh lật **+1,71 [1,29, 2,14]**.
  - Cổng đăng ký trước qua cả 4 điều kiện → bước C.
  - Phần lặp lại được ≈ biết **số vật thể** của ảnh (oracle chỉ biết số vật thể: +1,53). Ảnh ít vật thể được gửi rẻ,
    kênh được giải phóng; lợi nhất khi tắc nghẽn (+5,1).
  - "Mode nào hợp ảnh nào" vẫn không lặp lại được.
  - Lợi ích vượt sàn PSNR: recall tuân thủ Qmin −2,15.
  - MPC với mục tiêu TP đúng hạn không dùng được thông tin tiện ích (+0,2 kiểm chéo).
  - Bước C (bộ dự đoán số vật thể nhân quả) chờ phản biện quyết định về Qmin và chỉ số.

**Phản biện #46 (18/11): bước C** ([protocol](swarmbridge-atc-protocol.md),
[báo cáo](swarmbridge-atc-step-c-result.md)).
- Bài toán detector-only (khác A5). Test trên nội dung độc lập: VisDrone test-dev, 49 video, seed mới 9600–9699.
- **Cổng FAIL → dừng hướng giá trị tác vụ.**
  - Bộ dự đoán được chọn (detector nhẹ, pC) so với q3 + MPC: +0,20 [−0,94, +1,63]; recall theo ảnh −3,0.
  - **Oracle số vật thể cũng mất dư địa trên test-dev: −0,97** (A-T1: +1,53). Test-dev ít vật thể hơn (46,9 so với
    68,6 / khung), nên λ tune trên train hạ mode quá tay.
  - Payload probe và thống kê ảnh không đếm được vật thể trên video mới (Spearman val −0,02 / 0,07).
- Oracle kết quả detector thật vẫn +2,17 [+1,13, +4,07], nhưng không đặc trưng rẻ nào xấp xỉ được.
- C0: có ràng buộc Qmin, oracle số vật thể +0,69 [−0,27, +2,34] → hai bài toán khác nhau đáng kể.

**Phản biện #47 (18/11): đóng nhánh tối ưu offloading ảnh độc lập.**
- Chấp nhận DỪNG bước C.
- Không A-T2, không bộ dự đoán mới, không hiệu chỉnh λ sau test, không quay lại RL cho bài toán này, không viết lại
  Method.
- Hệ thống LA (PHY, CQI/OLLA, hàng đợi, deadline, scheduler) giữ làm tài sản nghiên cứu.
- Việc tiếp theo là memo đặt lại bài toán:
  - (A) bài đánh giá LA thực tế;
  - (B) cảm biến ngữ nghĩa phối hợp nhiều UAV.
- **Rà soát thiết kế (18/11, sau bản sửa #48):** [swarmbridge-design-review-2026-11-18.md](../2026-11-18_design-reset/swarmbridge-design-review-2026-11-18.md).
  - Đề xuất gốc (29/10): truyền bằng chứng bổ sung theo giá trị có điều kiện cho bên nhận, giữa nhiều UAV.
  - Từ 06/11 (hướng Huang), hợp tác và trạng thái bên nhận bị bỏ.
  - Điều khiển chung bay + kênh + công suất + mode **đã thử** ở A3.2 / A4 / A5. Lợi ích nhỏ; bay không đóng góp
    vì nội dung ảnh không phụ thuộc vị trí.
  - Đề xuất: ghép LA + pipeline hợp tác UAV3D, thêm trạng thái bên nhận. Bay ở giai đoạn 2, với giá trị quan sát theo
    vị trí.
- Chưa được duyệt train hay mô phỏng mới.
- Hai ghi chú tài liệu đã thêm vào [báo cáo bước C](swarmbridge-atc-step-c-result.md#ghi-chú-sau-phản-biện-47-không-chạy-thêm-không-đổi-quyết-định).

Tổng quan mọi giai đoạn: [reports/README.md](../../README.md). Ngữ cảnh ngắn: [CONTEXT.md](../../CONTEXT.md).
