# SwarmBridge: báo cáo

Cập nhật 18/11/2026 (sau phản biện #46: bước C, bộ dự đoán số vật thể trên test-dev → dừng hướng giá trị tác vụ). Trang này là điểm bắt đầu:
- **Mục 1** cho biết đang ở đâu.
- **Mục 2–3** là dòng thời gian của hai hướng.
- **Mục 4** liệt kê mọi kết luận đã bị sửa ở báo cáo sau. Đọc mục này trước khi trích một con số cũ.

Kết quả số: `runs/uav_semcom/`, `runs/jepa_semcom/`. Code: `swarmbridge/experiments/uav_semcom/`,
`swarmbridge/experiments/jepa_semcom/`. Bảng chuyển đường dẫn: [metadata/](metadata/README.md) (lần mới nhất
[18/11](metadata/report-moves-2026-11-18.json)).

## Cấu trúc thư mục

| Thư mục / file | Nội dung |
|---|---|
| [swarmbridge-paperA-method-draft.md](swarmbridge-paperA-method-draft.md) | Bản nháp Method của Paper A (v2, phản biện #35): PPO + MPC chọn lọc, chỉ chọn mode. Chưa viết lại theo yêu cầu phản biện |
| [scheduling/](scheduling/) | **Đang mở.** Nhánh lập lịch kênh của Paper A: A-X2, A-X2b, A-X5, A-X6, A-X7, A-X8, A-X9, A-T1, rà soát tính mới và bước C (phản biện #37–#46) |
| [two-track/](two-track/) | Kế hoạch hai hướng (phản biện #26) và các vòng A-E1…A-E11 / B-E0…B-E9 (phản biện #27–#36). Mỗi file chứa cả phần A và phần B của một vòng |
| [archive/](archive/README.md) | Các nhánh đã đóng, chia theo giai đoạn (30/10–15/11) |
| [figures/](figures/), [metadata/](metadata/README.md) | Hình; JSON kiểm toán và bảng chuyển đường dẫn |
| [wireless-abc.md](wireless-abc.md) | Tổng hợp hướng A/B/C ngày 06/11. Để nguyên ở đây vì `runs/active_recovery/protocol-v0.1.json` (nhánh Active Semantic Recovery do Codex làm) tham chiếu đúng đường dẫn này |

## 1. Đang ở đâu (18/11)

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
  [báo cáo A-X9](scheduling/swarmbridge-ax9-final-drl-test-result.md#quyết-định-theo-quy-tắc-đã-đăng-ký).

**Phản biện #45 (18/11): rà soát tính mới + A-T1.**
- **Rà soát tính mới** ([bản rà soát](scheduling/swarmbridge-task-utility-novelty-review-2026-11-18.md)):
  - Không có khoảng trống thuật toán nào được bảo vệ chắc chắn.
  - MPC hiệu chỉnh online, giới hạn tính toán đã có ở ABR (RobustMPC, Oboe, Fugu).
  - Ước lượng giá trị ảnh không nhãn đã có (ORIC, DAO).
  - Còn lại: giả định LA thay cho Shannon, và độ chặt của đánh giá.
- **A-T1** ([báo cáo](scheduling/swarmbridge-at1-task-utility-result.md)): oracle biết TP thật từng ảnh, kênh vẫn
  nhân quả, seed mới 9200–9299.
  - So với q3 + MPC: +3,00; kiểm chéo bằng ảnh lật **+1,71 [1,29, 2,14]**.
  - Cổng đăng ký trước qua cả 4 điều kiện → bước C.
  - Phần lặp lại được ≈ biết **số vật thể** của ảnh (oracle chỉ biết số vật thể: +1,53). Ảnh ít vật thể được gửi rẻ,
    kênh được giải phóng; lợi nhất khi tắc nghẽn (+5,1).
  - "Mode nào hợp ảnh nào" vẫn không lặp lại được.
  - Lợi ích vượt sàn PSNR: recall tuân thủ Qmin −2,15.
  - MPC với mục tiêu TP đúng hạn không dùng được thông tin tiện ích (+0,2 kiểm chéo).
  - Bước C (bộ dự đoán số vật thể nhân quả) chờ phản biện quyết định về Qmin và chỉ số.

**Phản biện #46 (18/11): bước C** ([protocol](scheduling/swarmbridge-atc-protocol.md),
[báo cáo](scheduling/swarmbridge-atc-step-c-result.md)).
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
- **Rà soát thiết kế (18/11, sau bản sửa #48):** [swarmbridge-design-review-2026-11-18.md](swarmbridge-design-review-2026-11-18.md).
  - Đề xuất gốc (29/10): truyền bằng chứng bổ sung theo giá trị có điều kiện cho bên nhận, giữa nhiều UAV.
  - Từ 06/11 (hướng Huang), hợp tác và trạng thái bên nhận bị bỏ.
  - Điều khiển chung bay + kênh + công suất + mode **đã thử** ở A3.2 / A4 / A5. Lợi ích nhỏ; bay không đóng góp
    vì nội dung ảnh không phụ thuộc vị trí.
  - Đề xuất: ghép LA + pipeline hợp tác UAV3D, thêm trạng thái bên nhận. Bay ở giai đoạn 2, với giá trị quan sát theo
    vị trí.
- Chưa được duyệt train hay mô phỏng mới.
- Hai ghi chú tài liệu đã thêm vào [báo cáo bước C](scheduling/swarmbridge-atc-step-c-result.md#ghi-chú-sau-phản-biện-47-không-chạy-thêm-không-đổi-quyết-định).

**Paper B** (JEPA-SemCom trên VisDrone-VID) **tạm dừng** từ phản biện #35.
- Codec video liên khung (HEVC/AV1) → DINOv2 thắng codec ngữ nghĩa ở ≤ 12 kbit (B-E7).
- ROI-QP trong HEVC có trần khoảng +0,01 AP (B-E8, B-E9).
- Hướng còn lại được đề xuất: (a) lớp tăng cường ngữ nghĩa trên nền video. Chưa duyệt.

## 2. Paper A: dòng thời gian

| Vòng (phản biện) | Câu hỏi | Kết quả chính | Báo cáo |
|---|---|---|---|
| A5 / A5.5 (07–10/11) | PPO chọn mode có hơn heuristic mạnh nhất H\* không | +5,7 (test) / +7,8 (test-dev) ở phy_mcs_blind; K5 test-dev +12,4. Cơ chế chưa rõ | [archive/uav-semcom-a/](archive/README.md#uav-semcom-a) |
| Kế hoạch (#26) | Tách hai hướng A / B | Paper A: amortized optimization + bền với sai mô hình | [two-track/plan](two-track/swarmbridge-two-track-plan.md) |
| A-E1 (#27) | PPO tốn bao nhiêu | 1,2–1,7 ms / quyết định so với 40–60 µs của heuristic; không đáng kể so với slot 1 s | [two-track/ae1-be0](two-track/swarmbridge-ae1-be0-result.md) |
| A-E2/E3 (#28) | PPO so với MPC | MPC mô hình đúng +2,7 nhưng đắt 30–100 lần; MPC Shannon (sai) −8,6 | [two-track/ae2-be1](two-track/swarmbridge-ae2-be1-result.md) |
| A-E4 (#29) | Rollout phi tiên tri, hiệu chỉnh từ ACK, kết hợp | MPC phi tiên tri ≈ PPO (+0,6); hiệu chỉnh ACK sửa sai mô hình (13,98 → 30,94); PPO + MPC đúng +2,0 | [two-track/ae4-be3](two-track/swarmbridge-ae4-be3-result.md) |
| A-E5 (#30) | Cổng theo sai số mô hình | Log throughput lệch −12 % → hiệu chỉnh theo slot (`cals`); cổng chặn Shannon; +1,69 / +1,86 | [two-track/ae5-be4](two-track/swarmbridge-ae5-be4-result.md) |
| A-E6 (#31) | Cổng theo độ tin cậy của lợi ích quyết định | +2,82 / +2,23; giá trị riêng là chống bias (Shannon +1,30 thay vì −8,67) | [two-track/ae6-be5](two-track/swarmbridge-ae6-be5-result.md) |
| A-E7 (#32) | Giảm chi phí | Hai tầng (sàng lọc H₁ = 1 → MPC 3 mode) +2,39 / +2,22 ở 25 / 20 ms | [two-track/ae7-be6](two-track/swarmbridge-ae7-be6-result.md) |
| A-E8 (#33) | Ngân sách 50 ms | Giữ hạn (p99 40–44 ms); ứng viên bi quan giúp khi mô hình lệch | [two-track/ae8-be7](two-track/swarmbridge-ae8-be7-result.md) |
| A-E9 (#34) | Seed mới, feedback trễ/mất | +2,18 / +1,91; feedback 1–3 slot, mất 30–60 % không hại (nhờ warm-start) | [two-track/ae9-be8](two-track/swarmbridge-ae9-be8-result.md) |
| A-E10 (#35) | Warm-start lệch regime | −6,4 / −12,1 so với PPO; tracker không phát hiện. Method v2 | [two-track/be9-ae10](two-track/swarmbridge-be9-ae10-result.md) |
| A-E11 (#36) | Sửa A-E10 | Cửa sổ 30 quan sát: tệ nhất −0,14; warm-start chỉ hơn cold +0,4 / +0,9 | [two-track/ae11-ax1](two-track/swarmbridge-ae11-ax1-result.md) |
| A-X1 (#36) | Kênh × mode có tương tác không | Có ở K5; 16 heuristic chọn kênh không hơn T:best | [two-track/ae11-ax1](two-track/swarmbridge-ae11-ax1-result.md) |
| A-X2 (#37) | Học chọn kênh (C-only) | +2,52 (K5) / +3,07 (ref); học thêm mode không ổn định; luật kiểu bài báo thua +10…+170; khảo sát 19 bài: khác biệt chính là baseline đã tune | [scheduling/ax2](scheduling/swarmbridge-ax2-result.md) |
| A-X2b (#38) | C-only train trên MCS biết nhiễu | +11,28; λ = 0,75 → −32, λ = 0 → −131 | [scheduling/ax2b](scheduling/swarmbridge-ax2b-mcs-result.md) |
| A-X5 (#39) | Link adaptation thực tế + train ngẫu nhiên hoá | Không dùng chung kênh tốt nhất ở cả 7 điều kiện; C-only ngẫu nhiên hoá −42,9 (dùng chung kênh khoảng 0,86, không thích ứng); C-only danh nghĩa ≈ T:best; cổng NACK cứu nhưng vẫn thua | [scheduling/ax5](scheduling/swarmbridge-ax5-la-result.md) |
| A-X6 (#41) | Sửa PHY, link adaptation biết lịch, cơ hội lập lịch | Sửa block không đổi A-X5 (≤ 2 điểm); chọn UAV (JA, không dùng chung kênh) +8,2 nhân quả; dùng chung kênh −5…+6 tuỳ giả định; biết trước kênh +12,5 (không nhân quả); K5_mcsg: JA không dùng chung kênh +12,4 ≈ RL A-X2b | [scheduling/ax6](scheduling/swarmbridge-ax6-opportunity-result.md) |
| A-X7 (#42) | Tách cơ hội thông tin, kênh tương quan, bộ điều khiển mode trên PHY mới | Oracle chỉ LoS khoảng 0 (khoảng cách oracle là fading nhỏ); bộ lọc LoS tương quan +0; SRS +3; JA ≥ mọi RL lập lịch; PPO–MPC hiệu chỉnh +12…+17 so với PPO; tương tác lập lịch × mode +4…+5; JA + PPO–MPC hơn heuristic tốt nhất +13,3 (heuristic chưa tune cùng JA) | [scheduling/ax7](scheduling/swarmbridge-ax7-information-mode-result.md) |
| A-X8 (#43) | Đối chứng công bằng cho tổ hợp lập lịch + mode | PPO–MPC ≈ heuristic tune cùng JA và cách bay (−1,25 n.s.); MPC lấy heuristic làm incumbent tốt nhất (+5,0 so với PPO–MPC); JA +8…+16 dưới mọi mode; hiệu chỉnh luật thiển cận không giúp; tương tác không ổn định; latency tích hợp p99 ≤ 46 ms | [scheduling/ax8](scheduling/swarmbridge-ax8-fair-baseline-result.md) |
| A-X9 (#44) | Phép thử DRL cuối (protocol đăng ký trước): PPO train trên LA làm incumbent cho MPC | (PPO-LA + MPC) − (heuristic + MPC) = −0,14 [−2,83, +2,56], recall −0,031 → cổng FAIL 4/5, **dừng nhánh DRL**; PPO-LA hơn PPO-Shannon +16,8 nhưng ≈ H\*; hiệu chỉnh +17 so với MPC danh nghĩa | [scheduling/ax9](scheduling/swarmbridge-ax9-final-drl-test-result.md) |
| A-T1 (#45) | Thông tin tiện ích detector từng ảnh đáng bao nhiêu trong mạng (oracle, kênh nhân quả) | Kiểm chéo +1,71 [1,29, 2,14] so với q3 + MPC → cổng qua; ≈ toàn bộ là biết số vật thể (+1,53), qua giảm tắc nghẽn; vượt Qmin (−2,15 recall tuân thủ Qmin); MPC mục tiêu TP không dùng được thông tin | [scheduling/at1](scheduling/swarmbridge-at1-task-utility-result.md) |
| Bước C (#46) | Bộ dự đoán số vật thể nhân quả (payload / thống kê ảnh / detector nhẹ) trên test-dev; chẩn đoán Qmin | Cổng FAIL: pC +0,20 [−0,94, +1,63], recall theo ảnh −3,0; oracle số vật thể −0,97 trên nội dung mới; oracle detector +2,17 → dừng hướng | [scheduling/atc](scheduling/swarmbridge-atc-step-c-result.md) |

## 3. Paper B: dòng thời gian (tạm dừng)

| Vòng | Kết quả chính | Báo cáo |
|---|---|---|
| B-E0 (#27) | Pipeline token DINOv2 + giữ nguyên + top-k + bitstream chạy, đồng bộ; token thô đắt cỡ AVIF | [two-track/ae1-be0](two-track/swarmbridge-ae1-be0-result.md) |
| B-E1/E2 (#28) | PCA nén token được; predictor học / kiểu JEPA không hơn "giữ nguyên"; AVIF → DINOv2 tốt hơn từ khoảng 25 kbit | [two-track/ae2-be1](two-track/swarmbridge-ae2-be1-result.md) |
| B-E3 (#29) | Bottleneck giữ nhiệm vụ; mã học được hơn AVIF 10/10 clip; JEPA-32 +0,02 mAP nhờ mã ổn định hơn | [two-track/ae4-be3](two-track/swarmbridge-ae4-be3-result.md) |
| B-E4 (#30) | 3 seed ghép cặp: JEPA +0,025 / +0,013 / +0,005, CI chứa 0 | [two-track/ae5-be4](two-track/swarmbridge-ae5-be4-result.md) |
| B-E5 (#31) | Chỉ làm mượt theo thời gian ≥ JEPA ở cùng bit → lợi ích JEPA là do ổn định theo thời gian | [two-track/ae6-be5](two-track/swarmbridge-ae6-be5-result.md) |
| B-E6 (#32) | Surrogate số bit exp-Golomb: +0,030–0,036 mAP so với task ở 9–18 kbit | [two-track/ae7-be6](two-track/swarmbridge-ae7-be6-result.md) |
| B-E7 (#33) | HEVC/AV1 liên khung → DINOv2 thắng codec ngữ nghĩa ở ≤ 12 kbit và trên patch thay đổi → Paper B dạng hiện tại không đứng | [two-track/ae8-be7](two-track/swarmbridge-ae8-be7-result.md) |
| B-E8 (#34) | ROI-QP trong HEVC theo DINOv2 +0,01 AP (~11 % bit); ap_change không tăng; số hạng sự kiện không kích hoạt | [two-track/ae9-be8](two-track/swarmbridge-ae9-be8-result.md) |
| B-E9 (#35) | Hạn mức event riêng: ap_change +0,011 ở 6 kbit = trần GT; không bền → đề xuất hướng (a) | [two-track/be9-ae10](two-track/swarmbridge-be9-ae10-result.md) |

## 4. Các lần sửa kết luận

Cột trái là điều từng được báo. Cột giữa là báo cáo đã sửa nó. Cột phải là điều hiện được coi là đúng.

| Kết luận cũ | Sửa ở | Hiện được coi là đúng |
|---|---|---|
| A8: thang sai lệch mô hình cho thấy PPO bền (+5,73) | A8 (archive) | **Không hợp lệ**: không có nhiễu đồng kênh với các chính sách được so, nên thang không tác động |
| A-E2: MPC mô hình đúng hơn PPO +2,7 | A-E4 | Phần lớn do rollout lạc quan (mode tương lai riêng từng đường mẫu). Phi tiên tri chỉ +0,6 (n.s.) ở phy_mcs_blind; ref vẫn +2,7 |
| A-E2/E3: PPO bền hơn planner có mô hình | A-E4 | Không đứng: planner hiệu chỉnh từ ACK cũng bền như vậy. Còn lại luận điểm chi phí và kết hợp |
| A-E4: hiệu chỉnh theo thời gian phát | A-E5 | Log throughput theo thời gian phát lệch thấp −12 % với mô hình đúng → dùng hiệu chỉnh theo slot (`cals`) |
| A-E5: cổng sai số +1,69 | A-E6 | Kết quả gắn với giao thức `warm`; khi warm-start riêng (`pre`), cổng dư thừa |
| A-E6: kiểm tra hai đầu khoảng | A-E7 | Có đảo thứ tự bên trong khoảng (12 % / 22 %); gọi đúng tên là "kiểm tra đầu mút"; lưới 5 điểm không tăng return |
| A-E9: bền với feedback trễ/mất | A-E9, A-E10 | Bền nhờ warm-start (cold chỉ +0,65); warm-start lệch regime thì −6 / −12 |
| A-E10: warm-start là nguồn lợi ích | A-E11 | Sửa bằng cửa sổ 30 quan sát; lợi ích chủ yếu từ hiệu chỉnh trong episode (warm chỉ +0,4 / +0,9 so với cold) |
| A-X2: học chọn kênh +2,5 / +3,1 | A-X2 stress | Lợi thế dùng chung kênh ở K5 phụ thuộc Shannon lý tưởng |
| A-X2b: C-only trên MCS +11,3 | A-X2b (λ), A-X5 | Chỉ khi link adaptation biết trước nhiễu; với link adaptation thực tế không còn dùng chung kênh có lợi |
| A-X5: PHY link adaptation đã kiểm chứng | Audit #40, A-X6 | Bit gửi trước khi block bị lật sang NACK vẫn được tính (0,3–0,5 % số bit). Đã sửa (`la_atomic`); chạy lại ghép cặp: thay đổi ≤ 2 điểm, kết luận A-X5 giữ nguyên |
| A-X5: không dùng chung kênh là tốt nhất ở cả 7 điều kiện | A-X6 | Chỉ đúng trong tập luật đã thử. Một scheduler không dùng chung kênh mạnh hơn (JA, nhân quả) hơn luật đó +6…+8; dùng chung kênh có kiểm soát dưới link adaptation biết lịch: −5…+6 |
| A-X2b: lợi ích +11,3 của RL đến từ dùng chung kênh (link adaptation lý tưởng) | A-X6 | Heuristic JA không dùng chung kênh đạt +12,4 (RL hơn nó −1,1, n.s.); dùng chung kênh chỉ thêm +3,2. RL thắng vì họ heuristic đã tune quá yếu, không vì heuristic dùng mô hình tốc độ sai |
| A-X6: khoảng +12,5 thông tin kênh có thể lấy lại phần nào bằng LoS tương quan | A-X7 | Oracle chỉ LoS cho khoảng 0; khoảng cách gần như toàn bộ là fading nhỏ bị đóng băng theo slot (giả định simulator, không nhân quả); LoS tương quan + bộ lọc: +0 |
| A-X2 (K5 / ref): học chọn kênh +2,5 / +3,1 so với T:best | A-X7 | JA (có dùng chung kênh) hơn RL +1,3 / +0,8: lợi ích có thật nhưng heuristic mạnh làm được |
| A-X7 (chạy đầu): tương tác lập lịch × mode +19 | A-X7 (chạy lại) | Lẫn với khác biệt cách bay (paced vs mission); so cùng cách bay: +4,3…+4,9 |
| A-X7: JA + PPO–MPC hơn tổ hợp heuristic tốt nhất +13,3; tương tác lập lịch × mode +4…+5 | A-X8 | Heuristic chưa tune cùng JA và cách bay. Tune công bằng: PPO–MPC − heuristic = −1,25 n.s.; MPC lấy heuristic làm incumbent hơn PPO–MPC +5,0; tương tác so với heuristic hiệu chỉnh: −2,4 (paced) / +5,4 (mission) |
| Paper A (A5…A-E11): PPO là thành phần đóng góp của bộ điều khiển mode | A-X8, A-X9 | Dưới LA thực tế, PPO kể cả khi train đúng PHY chỉ ngang mode cố định đã tune; kèm MPC thì −0,14 so với MPC lấy heuristic làm incumbent, và recall thấp hơn. Lợi ích A-E đo so với PPO, không so với heuristic + MPC |
| G-L2: dư địa tiện ích từng ảnh khoảng +1 điểm (tĩnh, khớp bit) | A-T1 | Trong simulator mạng có tắc nghẽn, dư địa kiểm chéo +1,71, chủ yếu từ biết số vật thể; phần "mode nào hợp ảnh nào" vẫn không lặp lại được. Recall A-X9 dùng file fidelity thiếu dòng mode sinh ảnh (ảnh hưởng không đáng kể); A-T1 dùng file L1 đầy đủ |
| A-T1: dư địa lặp lại được ≈ biết số vật thể (+1,53) | Bước C | Chỉ đúng trên nội dung cùng nguồn với train (DET-val). Trên test-dev (49 video) oracle số vật thể −0,97 [−2,64, +1,10]; λ tune theo phân phối số vật thể của train hạ mode quá tay. Oracle kết quả detector vẫn +2,17 |
| B-E3: JEPA-32 hơn non-JEPA +0,02 | B-E4, B-E5 | CI chứa 0 qua 3 seed; bằng với chỉ làm mượt theo thời gian |
| B-E3: mã học được hơn AVIF | B-E7 | Đúng so với AVIF ảnh đơn, nhưng thua codec video liên khung ở ≤ 12 kbit |
| B-E8: số hạng sự kiện | B-E9 | Bản B-E8 gần như không kích hoạt (lỗi thiết kế phân bổ theo hạng); B-E9 sửa, trần khoảng +0,01 |

## 5. Nhánh đã đóng

| Nhánh | Kết luận | Lưu trữ |
|---|---|---|
| R0 → R1b trên VisDrone (10–11/11) | H1 FAIL: không có dư địa | [archive/r1-visdrone/](archive/README.md#r1-visdrone) |
| H1 trên MDMT: Gate 0, pilot 0B (11–13/11) | Lợi ích crop nhỏ (+1–3 F1), không ổn định | [archive/h1-mdmt/](archive/README.md#h1-mdmt) |
| A7 / A8 / B1 (13–14/11) | A7: dư địa đến từ kiến thức GT cấp cảnh; A8 không hợp lệ; B1 NO-GO | [archive/a7-a8-b1/](archive/README.md#a7-a8-b1) |
| Rà soát novelty + RQ1 (13–14/11) | PPO / MAPPO chọn TOKEN / CROP / FRAME đã có người làm; RQ1 NOT IDENTIFIABLE, RQ1-E E0 FAIL 0/8 | [archive/rq1-novelty/](archive/README.md#rq1-novelty) |
| Songdo + proposal JEPA (14–15/11) | Truyền thông không bị giới hạn (tải max 11,2 kbit/s so với ≥ 60 kbit/s) → NO-GO | [archive/songdo-jepa/](archive/README.md#songdo-jepa) |
| UAV-SemCom A, A5, A5.5, A6-L1 (07–10/11) | Nền của Paper A; PPO mode-only có lợi ở một số cấu hình | [archive/uav-semcom-a/](archive/README.md#uav-semcom-a) |
| Huang JSAC, proposal A/B/C (04–06/11) | | [archive/huang-abc/](archive/README.md#huang-abc) |
| UAV3D (đóng băng 04/11) | | [archive/uav3d/](archive/README.md#uav3d) |
| SB / Town3 (30/10–03/11) | | [archive/sb-town3/](archive/README.md#sb-town3) |
| Tài liệu, paper, repo | | [archive/tai-lieu/](archive/README.md#tai-lieu) |

**Chưa có bằng chứng thực nghiệm** cho RL hợp tác bầy đàn (MARL). Mọi kết quả RL ở trên là một bộ ra quyết định tập trung.
