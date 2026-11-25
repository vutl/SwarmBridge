# SwarmBridge: ngữ cảnh ngắn

Cập nhật 25/11/2026. Dán file này vào đầu phiên mới. Chi tiết:
- **[CHATGPT_HANDOFF.md](CHATGPT_HANDOFF.md)** (tiếng Anh): bản tóm tắt đầy đủ để bắt đầu một phiên ChatGPT phản biện
  mới. Dán nguyên văn file này;
- [HISTORY.md](HISTORY.md): lịch sử đầy đủ bằng tiếng Anh, để gửi phản biện;
- [README.md](README.md): các giai đoạn từ mới tới cũ, bảng sửa kết luận;
- `phases/*/README.md`: từng giai đoạn.

## Mục tiêu

**Multi-UAV + semantic communication + swarm intelligence**, hướng tới một bài Q1 có đóng góp thuật toán.

**Đề xuất gốc (29/10):** *Can a distributed UAV swarm communicate only the complementary evidence required to reduce
the receiver's task uncertainty?*
- Nhiều UAV có góc nhìn bổ sung cho nhau.
- Giá trị của một message có điều kiện theo những gì bên nhận đã biết.
- Có chi phí vô tuyến.
- Vòng kín: bất định → xin bằng chứng → hoàn thiện.

Schrödinger Bridge ban đầu là ứng viên cho bộ hoàn thiện, không phải lõi.

## Trạng thái (25/11)

- **Paper A: đánh giá mở rộng đã XONG** (23–25/11, [phases/2026-11-23_paperA-fulldata](phases/2026-11-23_paperA-fulldata/README.md)).
  - Quy mô: 25.200 episode mới, toàn bộ đăng ký trước. Kiểm tra coverage và checkpoint đều đạt.
  - **Đứng:**
    - lớp MPC chọn lọc tăng return so với heuristic tune (cấu hình chuẩn + 5/6 cấu hình mạng);
    - tăng return so với cả 3 họ RL (PPO / QR-DQN / DSAC);
    - **dải bền vững** là thành phần cần thiết.
  - **Không đứng:**
    - cổng và tầm nhìn H = 4 không cần thiết;
    - hiệu chỉnh ACK không thêm return ở cấu hình chuẩn (−0,10 n.s. so với danh định cùng thiết kế), chỉ thêm với RL
      và ở N5x3 / N5x4;
    - hiệu chỉnh **giảm recall** ở mọi nơi;
    - planner không bảo vệ có recall cao nhất, tức là return và recall lệch nhau;
    - chạy tuần tự vượt 50 ms ở 44–82 % số slot;
    - không họ RL nào hơn heuristic;
    - không có thành phần thuật toán mới.
  - **Khuyến nghị C:** viết lại Paper A thành đóng góp ứng dụng / thực nghiệm.
  - **Bản thảo `paper/paperA/main.tex` (method letter 23/11) đã lỗi thời.** Tiêu đề và abstract của nó ("hiệu chỉnh
    hơn danh định 12/12") bị kết quả matched design bác bỏ. Không sửa trước khi phản biện chốt hướng và venue.
  - **Đang chờ ChatGPT (phiên mới) quyết định:**
    1. C / B / D;
    2. có làm phân rã return nc so với ng (hậu kiểm, không mô phỏng) không;
    3. venue và số trang;
    4. dàn ý bài viết lại.
- **Paper 2** làm sau khi nộp Paper A.
- **Thiết kế Paper 2:** đặt lại bài toán ([rà soát thiết kế](phases/2026-11-18_design-reset/swarmbridge-design-review-2026-11-18.md)).
  - Đề xuất khôi phục: bộ vô tuyến LA đã kiểm chứng + pipeline hợp tác UAV3D (5 UAV đồng bộ) + trạng thái bên nhận
    (bất định / tuổi track).
  - Giai đoạn 2: quyết định bay, với giá trị quan sát phụ thuộc vị trí (mô hình cảm biến hiệu chỉnh trên UAV3D;
    CARLA/AirSim chỉ khi cần).
  - **Chờ người dùng quyết định:**
    1. chấp nhận khung giai đoạn 1 / 2;
    2. cho một phép kiểm tra rẻ "tham lam có trạng thái so với oracle nhìn xa" trên UAV3D mini;
    3. có tải được UAV3D đầy đủ không;
    4. giữ hay bỏ SB.
- **Đã đóng:**
  - Hướng thuật toán RL cho Paper A (offloading ảnh độc lập; RL / MPC chọn mode và lập lịch) ở phản biện #47.
    Paper A tiếp tục dưới dạng bài đánh giá / ứng dụng.
  - Paper B (JEPA-SemCom) tạm dừng từ #35.

## Điều đã biết (đứng vững sau các lần sửa)

1. **Hợp tác có giá trị khi có góc nhìn bổ sung thật.**
   - UAV3D mini: một UAV tự nhìn → hợp nhất box: mAP 0,512 → 0,665; AMOTA 0,542 → 0,690 (KF).
   - Trễ 2 khung xóa gần hết lợi ích; mất 50 % gói −5,5 AMOTA.
   - Cũng trên UAV3D (3 cảnh dev, chưa có CI, kênh tổng hợp):
     - "chỉ gửi cái bên nhận chưa có" là sai cho tracking (giá trị chủ yếu là xác nhận track);
     - feedback "bên nhận đã chắc" không hơn luật bỏ trùng giữa các UAV khi tính bit feedback;
     - message lũy tiến có hạn chót ≈ oracle biết kênh.
     Chi tiết: mục 5c của bản rà soát thiết kế.
   - MDMT (2 drone): góc tốt hầu như cố định, lợi ích crop nhỏ.
   - Town3 từng chạy với một nguồn mỗi vùng, nên chưa thử hợp tác.
2. **PHY thực tế đổi kết luận.**
   - Chính sách train trên Shannon thua 16,8 khi chạy dưới LA (CQI trễ, OLLA, ARQ, nhiễu đồng kênh).
   - Lợi thế dùng chung kênh biến mất dưới LA.
3. **Baseline phải được tune công bằng.**
   - Luật kiểu bài báo không tune thua heuristic tune từ +10 đến +170.
   - Scheduler heuristic JA ≥ mọi RL lập lịch đã thử.
   - Đây là bằng chứng từ các paper và baseline mình đã tái lập, không phải khẳng định về cả lĩnh vực.
4. **Bài toán gửi ảnh độc lập gần như chỉ cần nhìn trước mắt.**
   - PPO (kể cả train đúng PHY) ≈ heuristic + MPC hiệu chỉnh: −0,14 [−2,83, +2,56] (A-X9).
   - Điều khiển chung bay + kênh + công suất + mode đã thử (A3.2 / A4 / A5): lợi ích nhỏ, bay không đóng góp, vì nội
     dung ảnh không phụ thuộc vị trí.
5. **MPC hiệu chỉnh online từ ACK hơn MPC danh nghĩa *không bảo vệ*** khoảng +17. Nhưng (full-data 25/11) nếu MPC
   danh nghĩa có cùng cổng + dải bền vững thì khác biệt gần như mất ở cấu hình chuẩn (−0,10 n.s.): phần lớn lợi ích là
   do **dải**, không do hiệu chỉnh. MPC giới hạn tính toán có hiệu chỉnh đã có ở ABR (RobustMPC, Oboe, Fugu).
6. **Giá trị nội dung từng ảnh không dự đoán được từ đặc trưng rẻ, và không tổng quát.**
   - Oracle số vật thể +1,53 trên nội dung cùng nguồn, −0,97 trên test-dev.
   - Oracle kết quả detector +2,17 nhưng không xấp xỉ được.
7. **Khi tác vụ bên nhận là detection cố định, gửi danh sách box thường thắng gửi đặc trưng / latent.** Thiết kế
   message phải tính điều này.
8. **Novelty:**
   - "RL chọn mode SemCom + tài nguyên (+ quỹ đạo) cho UAV" đã có (Si 2023, Huang JSAC 2025, Li JSAC 2026).
   - "Chọn ai gửi gì" cho collaborative perception đã có (Where2comm, CoSDH, CodeFilling, SchedCP; hầu hết dùng kênh
     Shannon / CSI hoàn hảo).
   - "Gửi khi giá trị > chi phí" đã có (GOSC, AoII).
   - Không tuyên bố tính mới từ việc ghép các thành phần này.

## Quy trình làm việc đã thống nhất

- **Giai đoạn khám phá:** prototype nhanh với vài baseline có ý nghĩa; chỉ lỗi chí mạng mới chặn (rò rỉ dữ liệu, dùng
  thông tin tương lai, so sánh không công bằng). Cổng đăng ký trước và CI nhiều tầng để dành cho giai đoạn đánh giá.
- **Không thu hẹp bài toán đến mức mất cơ chế đang nghiên cứu.** Thu nhỏ quy mô (ít UAV, ít cảnh) để debug thì được.
- **Kiểm chứng mọi khẳng định của phản biện với báo cáo trong repo trước khi đổi hướng.**
- Giữ mọi kết quả âm và bảng sửa kết luận.
- Báo cáo bằng tiếng Việt, giải thích luồng trước khi đưa số.

## Ràng buộc

- **Repo công khai.** Không đưa dữ liệu, trọng số, ảnh / crop / embedding MDMT lên git.
- **MDMT:** chỉ có đồng ý miệng của giáo sư; bản tải từ mirror không chính thức. Không coi là đã được phép công bố.
- **CrossTrack-UAV:** cần ký thỏa thuận.
- **Không đụng nhánh Active Semantic Recovery** (`runs/active_recovery/`, do Codex làm). Để `reports/wireless-abc.md`
  đúng chỗ.
- PPO chạy CPU. Việc dùng GPU phải qua khóa GPU (`gpulock`).
- **Seed test đã dùng** (không dùng lại cho test mới): 7000–7099, 8000–8999 (A-E9 trở đi), 8300–8399, 8800–8999,
  9000–9099, 9200–9299, 9600–9699, 9800–9999, 12000–12099, 13000–13099, 14000–14099, 20000–26099 (full-data).
  Train 0–29; val 3000–3049.
- **Job dài:** khởi chạy tách rời qua WMI. Sau khi chạy và khi quay lại, phải kiểm tra tiến trình và `uptime` của WSL,
  không tin dòng "start" trong log. MPC không tái lập bit-for-bit (giới hạn thời gian thật), nên không chạy lại pool
  đã xong.

## Ở đâu

- **Code:**
  - `swarmbridge/experiments/uav_semcom/`: env, `env_la`, JA, MPC, A-X, A-T1, bước C.
  - `swarmbridge/experiments/uav3d_comm/`: pipeline hợp tác UAV3D.
  - `swarmbridge/experiments/jepa_semcom/`.
- **Kết quả:** `runs/uav_semcom/`, `runs/jepa_semcom/`.
- **Dữ liệu cục bộ** (không trên git): UAV3D mini (`data/v1.0-mini`), CoPerception-UAV Town3 (`data/coperception_uav`),
  VisDrone DET val (`data/visdrone`); VisDrone test-dev / VID trong môi trường WSL.
- **Tài liệu:** `pdf/` (đề xuất gốc, các bản tóm tắt, khoảng 90 paper); danh mục ở [literature/](literature/README.md).
