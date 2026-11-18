# SwarmBridge: báo cáo

Cập nhật 18/11/2026. Cách đọc:
1. **[CONTEXT.md](CONTEXT.md)**: tóm tắt một trang để dán vào phiên mới hoặc gửi phản biện (mục tiêu, trạng thái, điều
   đã biết, ràng buộc).
2. **[HISTORY.md](HISTORY.md)** (tiếng Anh): toàn bộ lịch sử, từng giai đoạn từ mới tới cũ kèm số liệu chính. Gửi
   file này cho phản biện (ChatGPT) để nó biết mọi thứ đã làm.
3. Trang này:
   - mục 1: đang ở đâu;
   - mục 2: các giai đoạn **từ mới tới cũ**, mỗi giai đoạn một thư mục có README riêng;
   - mục 3: mọi kết luận đã bị sửa. **Đọc mục 3 trước khi trích một con số cũ.**

Kết quả số: `runs/uav_semcom/`, `runs/jepa_semcom/`. Code: `swarmbridge/experiments/uav_semcom/`,
`swarmbridge/experiments/jepa_semcom/`. Đường dẫn cũ → mới: [metadata/](metadata/README.md) (lần mới nhất
[18/11b](metadata/report-moves-2026-11-18b.json)).

## Cấu trúc thư mục

| Thư mục / file | Nội dung |
|---|---|
| [CONTEXT.md](CONTEXT.md) | Ngữ cảnh ngắn cho phiên mới |
| [HISTORY.md](HISTORY.md) | Lịch sử đầy đủ cho phản biện (tiếng Anh), mới → cũ |
| [phases/](phases/) | Mỗi giai đoạn một thư mục `YYYY-MM-DD_tên` (ngày bắt đầu), nên sắp theo tên là theo thời gian. README của mỗi thư mục tóm tắt câu hỏi, kết luận và danh sách tài liệu |
| [literature/](literature/README.md) | Danh mục paper / repo, bản đồ bằng chứng, các lần tải tài liệu (30/10–04/11) |
| [figures/](figures/), [metadata/](metadata/README.md) | Hình; JSON kiểm toán và bảng chuyển đường dẫn |
| [wireless-abc.md](wireless-abc.md) | Tổng hợp hướng A/B/C ngày 06/11. Để nguyên ở đây vì `runs/active_recovery/protocol-v0.1.json` (nhánh Active Semantic Recovery do Codex làm) tham chiếu đúng đường dẫn này |

## 1. Đang ở đâu (18/11)

- **Đang làm:** [viết Paper A](phases/2026-11-18_paperA-writeup/paperA-outline.md) từ dữ liệu đã có. Hướng hợp tác nhiều UAV là Paper 2, làm sau.
- **Thiết kế cho Paper 2:** [đặt lại bài toán](phases/2026-11-18_design-reset/README.md).
  - Rà soát thiết kế: so đề xuất SwarmBridge gốc (29/10, truyền bằng chứng bổ sung giữa nhiều UAV theo giá trị có
    điều kiện cho bên nhận) với hệ thống đã chạy.
  - Đề xuất khôi phục: bộ vô tuyến LA + pipeline hợp tác UAV3D + trạng thái bên nhận; quyết định bay ở giai đoạn 2.
  - Chờ người dùng quyết định (mục 6 của tài liệu).
- **Paper A** (offloading ảnh độc lập, DRL / MPC chọn mode và lập lịch): **đóng** ở phản biện #47.
  - Dưới LA thực tế, PPO chỉ ngang heuristic tune công bằng + MPC hiệu chỉnh (A-X9).
  - Tiện ích từng ảnh không tổng quát sang video mới (bước C).
  - Giữ lại bộ vô tuyến LA làm tài sản.
- **Paper B** (JEPA-SemCom trên VisDrone-VID): tạm dừng từ phản biện #35. Codec video liên khung thắng codec ngữ nghĩa
  ở ≤ 12 kbit.
- **Chưa có bằng chứng thực nghiệm** cho RL hợp tác bầy đàn (MARL).
  - Mọi kết quả RL trong thư mục này là một bộ quyết định tập trung, trên bài toán không có hợp tác nhận thức.
  - Điều khiển chung bay + kênh + công suất + mode đã thử ở A3.2 / A4 / A5.

## 2. Các giai đoạn (mới → cũ)

| Giai đoạn | Thời gian (phản biện) | Câu hỏi | Kết luận | Trạng thái |
|---|---|---|---|---|
| [2026-11-18_paperA-writeup](phases/2026-11-18_paperA-writeup/README.md) | 18/11 → | Viết Paper A với luận điểm đánh giá (LA thực tế + baseline công bằng xóa ưu thế DRL; MPC hiệu chỉnh từ ACK) | Khung bài + việc còn thiếu (R1 bảng Shannon so với LA; R2 tái lập phương pháp đã công bố) | **Đang làm** |
| [2026-11-18_design-reset](phases/2026-11-18_design-reset/README.md) | 18/11 (#47–#48) | Đề xuất gốc bị lệch ở đâu; khôi phục thế nào | Hợp tác và trạng thái bên nhận bị bỏ từ 06/11; đề xuất LA + UAV3D + trạng thái bên nhận | **Đang mở** |
| [2026-11-17_paperA-scheduling](phases/2026-11-17_paperA-scheduling/README.md) | 17–18/11 (#37–#47) | Lập lịch kênh, LA thực tế, đối chứng công bằng, DRL cuối, tiện ích tác vụ | JA heuristic ≥ RL; PPO-LA + MPC ≈ heuristic + MPC (−0,14); tiện ích từng ảnh không tổng quát | Đóng |
| [2026-11-15_two-track](phases/2026-11-15_two-track/README.md) | 15–17/11 (#26–#36) | Paper A: PPO + MPC hiệu chỉnh khi mô hình kênh sai; Paper B: JEPA-SemCom | MPC hiệu chỉnh từ ACK + PPO +1,2…+2,2 so với PPO (đo so với PPO, không so heuristic); Paper B: codec video thắng | Đóng / tạm dừng |
| [2026-11-14_songdo-jepa](phases/2026-11-14_songdo-jepa/README.md) | 14–15/11 | C&C trên Songdo; proposal JEPA | Kênh không bị giới hạn → NO-GO | Đóng |
| [2026-11-14_rq1-novelty](phases/2026-11-14_rq1-novelty/README.md) | 13–14/11 | Rà soát novelty hợp tác nhận thức; liên kết xuyên góc (RQ1) trên MDMT | RL chọn TOKEN / CROP / FRAME đã có người làm; RQ1 không nhận dạng được, E0 FAIL 0/8 | Đóng |
| [2026-11-13_a7-a8-b1](phases/2026-11-13_a7-a8-b1/README.md) | 13–14/11 | Họ TOKEN / CROP / FRAME trong A5; thang sai lệch A8; chọn góc nhìn B1 | Dư địa = kiến thức GT cấp cảnh; A8 không hợp lệ; B1 NO-GO (góc tốt cố định) | Đóng |
| [2026-11-11_h1-mdmt](phases/2026-11-11_h1-mdmt/README.md) | 11–13/11 | Báo cáo hợp tác theo vật thể duy nhất trên MDMT (2 drone) | Crop +1–3 F1, chủ yếu một chuỗi | Đóng |
| [2026-11-10_r1-visdrone](phases/2026-11-10_r1-visdrone/README.md) | 10–11/11 | G-L2 tiện ích từng ảnh; R0–R1b phản hồi ngữ nghĩa | G-L2 FAIL; R1b H1 FAIL (không dư địa) | Đóng |
| [2026-11-06_uav-semcom-a](phases/2026-11-06_uav-semcom-a/README.md) | 06–10/11 | Hướng Huang: UAV gửi ảnh về BS, PPO chọn bay / kênh / công suất / mode | Joint PPO lợi nhỏ, bay chỉ canh nhịp; A5: PPO chọn mode +3…+12 ở vài cấu hình (sau đó bị A-X8/A-X9 đọc lại) | Đóng |
| [2026-11-04_uav3d](phases/2026-11-04_uav3d/README.md) | 04/11 | 5 UAV đồng bộ: gửi box hay đặc trưng, trễ / mất gói | Hợp tác +15 mAP / AMOTA; trễ 2 khung xóa lợi ích. Trên 3 cảnh dev: "chỉ gửi cái bên nhận chưa có" sai cho tracking (giá trị là xác nhận); feedback bên nhận không hơn luật bỏ trùng; còn dư địa thứ tự gửi +0,03–0,04 AMOTA chưa lấy được. Gác do chuyển hướng 06/11 | Đóng băng |
| [2026-11-04_huang-abc](phases/2026-11-04_huang-abc/README.md) | 04–06/11 | Proposal A/B/C; Huang JSAC bước 0 | Chọn mode của Huang là tra bảng; không thấy dư địa cho RL ở bước 0 | Đóng |
| [2026-10-30_sb-town3](phases/2026-10-30_sb-town3/README.md) | 30/10–03/11 | Schrödinger Bridge khôi phục latent BEV (CoPerception-UAV Town3); chọn block theo thời gian | SB ≤ ánh xạ; thiết lập một nguồn mỗi vùng (N = 1), chưa thử bằng chứng từ UAV khác | Đóng |

Đề xuất gốc (29/10) và các bản tóm tắt nằm trong `pdf/` (không đưa lên git):
- `Swarm-Conditioned Schrödinger Bridge cho Task-Oriented Multi-UAV Semantic Communications.pdf`;
- `Tóm tắt nghiên cứu.pdf`, `Tóm tắt điều hành.pdf`, `Executive summary.pdf`, `Tóm tắt báo cáo.pdf`;
- `1. Tình hình nghiên cứu (State-of-the-art).pdf`.

## 3. Các lần sửa kết luận

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
| A-X2 (lời phản biện #48 sau đó): "chưa thử điều khiển chung bay + truyền thông + nén" | Rà soát thiết kế 18/11 | Đã thử ở A3.2 / A4 / A5 (bay, kênh, công suất, mode do PPO quyết định). Lợi ích nhỏ; bay do RL quyết định không đóng góp ở A4, vì nội dung ảnh không phụ thuộc vị trí |
