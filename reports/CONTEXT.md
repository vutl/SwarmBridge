# SwarmBridge: ngữ cảnh ngắn

Cập nhật 18/11/2026. Dán file này vào đầu phiên mới hoặc gửi kèm khi nhờ phản biện. Chi tiết:
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

## Trạng thái (18/11)

- **Đang mở:** đặt lại bài toán ([rà soát thiết kế](phases/2026-11-18_design-reset/swarmbridge-design-review-2026-11-18.md)).
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
  - Paper A (offloading ảnh độc lập; RL / MPC chọn mode và lập lịch) ở phản biện #47.
  - Paper B (JEPA-SemCom) tạm dừng từ #35.

## Điều đã biết (đứng vững sau các lần sửa)

1. **Hợp tác có giá trị khi có góc nhìn bổ sung thật.**
   - UAV3D mini: một UAV tự nhìn → hợp nhất box: mAP 0,512 → 0,665; AMOTA 0,542 → 0,690 (KF).
   - Trễ 2 khung xóa gần hết lợi ích; mất 50 % gói −5,5 AMOTA.
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
5. **MPC hiệu chỉnh online từ ACK sửa được mô hình kênh sai**: hiệu chỉnh +17 so với MPC danh nghĩa. Nhưng MPC
   giới hạn tính toán có hiệu chỉnh đã có ở ABR (RobustMPC, Oboe, Fugu).
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
  9000–9099, 9200–9299, 9600–9699. Train 0–29; val 3000–3049.

## Ở đâu

- **Code:**
  - `swarmbridge/experiments/uav_semcom/`: env, `env_la`, JA, MPC, A-X, A-T1, bước C.
  - `swarmbridge/experiments/uav3d_comm/`: pipeline hợp tác UAV3D.
  - `swarmbridge/experiments/jepa_semcom/`.
- **Kết quả:** `runs/uav_semcom/`, `runs/jepa_semcom/`.
- **Dữ liệu cục bộ** (không trên git): UAV3D mini (`data/v1.0-mini`), CoPerception-UAV Town3 (`data/coperception_uav`),
  VisDrone DET val (`data/visdrone`); VisDrone test-dev / VID trong môi trường WSL.
- **Tài liệu:** `pdf/` (đề xuất gốc, các bản tóm tắt, khoảng 90 paper); danh mục ở [literature/](literature/README.md).
