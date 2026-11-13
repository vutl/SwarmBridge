# A8 (thang sai lệch mô hình) và B1 (cơ hội hợp tác theo góc nhìn): đăng ký trước

Ngày 13/11/2026, theo phản biện #18: duyệt cả hai phép kiểm tra GO / NO-GO; **không train PPO**; B có điều kiện.

Bản này được commit **trước** khi chạy bất kỳ phần nào.

## A8: lợi thế của PPO có tăng theo mức sai lệch mô hình không?

**Giả thuyết:** PPO mode-only thắng heuristic đã điều chỉnh vì mô hình kênh của heuristic sai. Nếu đúng, Δ(PPO − H\*) phải tăng khi sai lệch tăng, và về gần 0 khi không có sai lệch.

**Thang sai lệch (5 mức, cố định trước):** PHY kiểu MCS của A5, trong đó bộ thích ứng liên kết chọn CQI theo SINR ước lượng

SINR_est = S / (N + λ·I), với I là nhiễu đồng kênh thật.

Khối truyền hỏng (tốc độ 0) nếu SINR thật thấp hơn ngưỡng của CQI đã chọn.

| Mức sai lệch 1 − λ | λ | Tên chế độ |
|---|---|---|
| 0 | 1 | `phy_mcs` (đã có; biết SINR thật) |
| 0,25 | 0,75 | `phy_mcs_l75` (mới) |
| 0,5 | 0,5 | `phy_mcs_l50` (mới) |
| 0,75 | 0,25 | `phy_mcs_l25` (mới) |
| 1 | 0 | `phy_mcs_blind` (đã có; mù nhiễu) |

Có test kiểm rằng λ = 1 và λ = 0 cho đúng tốc độ của `phy_mcs` và `phy_mcs_blind`.

**Chính sách PPO (đóng băng, không fine-tune):** 3 mô hình mode-only train ở `phy_mcs_blind` trong A5 (seed 0, 1, 2; checkpoint tốt nhất theo val đã chọn ở A5), dùng **luật slot lúc train** (`pacedB2+budget1`) ở mọi mức. Đây là chính sách duy nhất thuộc họ MCS đã được train; nó không nhìn thêm thông tin nào khi đổi mức.

**Baseline H\*(mức):** điều chỉnh lại **cho từng mức** trên pool train, seed 0..29, bằng đúng thủ tục của A5 / A5.5: tốt hơn giữa A5 T:best (132 thành viên) và T:SC (56 thành viên) theo return train. Heuristic chỉ dùng thông tin như ở A5, không thấy λ hay nhiễu thật. Cách này có lợi cho heuristic, vì PPO không được điều chỉnh lại.

**Đánh giá:** cùng workload, môi trường, phần thưởng. Seed môi trường 8000..8099 (đã khoá ở A5.5). Pool **test** là chính; test-dev báo kèm. Bootstrap theo cụm video có mô phỏng lại (B = 50, 20 seed mỗi bản lặp), và mức 3 lấy mẫu lại 3 mô hình, như A5.5.

**Kiểm tra tái lập:** tại `phy_mcs_blind`, Δ trên test phải khớp A5.5 (+5,73) trong sai số số học, vì cùng chính sách, cùng H\*, cùng seed.

**Quy tắc quyết định (pool test):**
- **GO** nếu (i) Δ không giảm theo mức sai lệch (cho phép lùi tối đa 0,5 return giữa hai mức liền nhau), **và** (ii) CI mức 3 loại trừ 0 về phía dương ở ít nhất 2 mức.
- **NO-GO** nếu lợi thế chỉ có ở mức 1 (đúng chế độ đã train), hoặc không có xu hướng.
- **INCONCLUSIVE** nếu có xu hướng nhưng (ii) không đạt.
- Không đổi định nghĩa mức hay ngưỡng sau khi xem kết quả.

**Giới hạn biết trước:**
- PPO chỉ được train ở mức 1, nên ở các mức thấp nó bị lệch khỏi chế độ train. Một kết quả "PPO thua ở mức 0" không bác bỏ RL nói chung.
- 3 seed train; ảnh test đã dùng ở A5 / A5.5.

## B1: cơ hội hợp tác theo góc nhìn (có điều kiện)

**Câu hỏi:** ở cùng ngân sách bit, cấp FRAME cho góc nhìn tốt hơn (góc kia gửi TOKEN) có tăng F1 **vật thể duy nhất sau hợp nhất** so với các phân bổ không phân biệt góc không? Đây là sàng lọc khả thi **có trợ giúp GT** (chọn góc bằng GT, liên kết lý tưởng); không phải cơ chế triển khai được.

### Điều kiện 1 (làm trước, chỉ dùng nhãn): ID của GT có thống nhất giữa hai UAV không?

`unique_counts()` gộp ID bằng phép hợp, nên phép chấm chỉ đúng nếu cùng một ID ở hai góc là cùng một vật. Kiểm bằng ba nguồn độc lập với bộ chấm:
1. **Tài liệu của dataset** (README / bài báo MDMT) về quy ước ID xuyên góc.
2. **Nhất quán lớp:** với các ID xuất hiện ở cả hai góc của một chuỗi, tỉ lệ cùng lớp; so với mức ngẫu nhiên khi tráo ID.
3. **Nhất quán hình học:** với mỗi cặp khung, ước lượng homography giữa hai góc từ tâm đáy box của các ID chung (RANSAC), rồi đo tỉ lệ ID chung là inlier, và sai số chiếu lại trên các ID *không* dùng để fit (chia đôi ngẫu nhiên). Đối chứng: tráo ID giữa các box của góc 2. Nếu ID không thống nhất, tỉ lệ inlier sẽ sụp về mức của bản tráo.

**Tiêu chí:** một chuỗi **đạt** nếu cùng lớp ≥ 95% và tỉ lệ inlier trung vị ≥ 70% và cao hơn bản tráo ít nhất 40 điểm. B1 chỉ chấm trên các chuỗi đạt; nếu dưới 8/11 chuỗi đạt thì B1 = INCONCLUSIVE, không tính F1 duy nhất.

Lưu ý: homography chỉ đúng với mặt phẳng đất, nên tỉ lệ inlier dưới 100% là bình thường; ngưỡng đặt theo đó.

### Điều kiện 2: cơ sở dùng dữ liệu

Phần chấm tầng 1 chỉ chạy khi cơ sở dùng dữ liệu MDMT được xác nhận (tác giả, hoặc lab / trường). Cho đến lúc đó chỉ có thiết kế, code và test tổng hợp. Điều kiện 1 là kiểm tra cấu trúc trên nhãn và được làm trước.

### Tầng 1 (offline, bản ghi pilot đã có; không ảnh mới)

- **Chia thời gian:** với mỗi chuỗi, nửa đầu số epoch dùng để chọn; nửa sau để chấm. Không dùng GT của nửa sau để chọn.
- **Phân bổ so sánh**, mỗi epoch một lựa chọn mode cho từng góc:
  - `TT`: cả hai TOKEN;
  - `FF`: cả hai FRAME;
  - `F1T`, `TF2`: FRAME cố định cho UAV1 hoặc UAV2;
  - `ALT`: luân phiên FRAME giữa hai góc theo epoch (đối xứng, cùng bit trung bình với một-FRAME);
  - **`VIEW`**: FRAME cho góc được chọn trên nửa đầu (góc có F1 duy nhất cao hơn khi nó gửi FRAME và góc kia gửi TOKEN), góc kia TOKEN.
- **Cùng ngân sách bit thực:** so sánh chính ở mức "một FRAME + một TOKEN mỗi epoch": `VIEW` so với phương án tốt nhất trong {`TT`, `ALT`, và phương án cố định góc **được chọn trên nửa đầu theo quy tắc không nhìn GT**: góc có nhiều detection YOLO11n hơn}. `FF` dùng gấp đôi bit và chỉ được báo như tham chiếu, không dùng để quyết định. Bit thực đạt được báo cho mọi phương án.
- **Chấm:** khai báo ở τ_decl = 0,3 (như A7); F1 / P / R vật thể duy nhất với liên kết lý tưởng theo ID của GT; FP trùng và FP liên kết bằng 0 theo cấu trúc (liên kết lý tưởng), báo rõ như vậy; kết quả theo từng chuỗi.
- **Quy tắc quyết định (đã đăng ký ở bản review):** GO tầng 1 nếu `VIEW` hơn phương án tốt nhất cùng bit ≥ 2 điểm F1 duy nhất (trung bình theo chuỗi) **và** dương ở ≥ 8/11 chuỗi (hoặc ≥ 73% số chuỗi đạt điều kiện 1). Ngược lại NO-GO.
- GO chỉ có nghĩa là có dư địa để nghiên cứu liên kết và lập lịch; không tự động mở Gate 1 hay PPO.

## Không làm

Không train; không ảnh MDMT mới; không val / HOLD; không Gate 1; không A7-S3; không đổi ngưỡng sau khi xem kết quả.
