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

## Sửa đổi 1 cho B1 (13/11/2026, theo phản biện #19; trước khi chạy tầng 1 trên bất kỳ bản ghi dataset nào)

**Lý do:** bản gốc yêu cầu "cùng ngân sách bit thực" nhưng code chỉ báo số bit đạt được. FRAME của hai UAV có kích thước khác nhau, nên "một FRAME + một TOKEN mỗi epoch" không phải là so sánh cùng bit.

**Thay đổi (ngưỡng GO giữ nguyên):**
- **Giới hạn chung B cho mọi phương án, theo từng epoch.** B là ngân sách UL chung của hai UAV.
- **B_ref** = giá trị lớn nhất, trên các epoch của **nửa đầu**, của chi phí "một FRAME (góc nào cũng được) + TOKEN của góc kia". Chỉ dùng số bit, không dùng nhãn.
- **Các mức:** B = κ · B_ref với κ ∈ {0,25; 0,5; 0,75; 1; 1,5; 2}. Quyết định ở **κ = 1**; các mức khác chỉ để báo đường F1–bit.
- **Quy tắc gửi (không dùng nhãn):** góc ở mode TOKEN được phục vụ trước; nếu token không vừa thì chia đều B và đóng gói theo quy tắc của pilot. Sau đó mỗi FRAME được yêu cầu chỉ được gửi nếu vừa phần còn lại (FRAME nhỏ hơn trước); nếu không vừa, góc đó gửi TOKEN trong phần còn lại.
- **Bất biến:** số bit gửi ≤ B ở mọi epoch; code dừng nếu vi phạm.
- **VIEW chọn góc dưới cùng giới hạn B** trên nửa đầu.
- **Thêm baseline `BIG`:** FRAME cho góc có FRAME trung bình lớn hơn ở nửa đầu (không nhìn nhãn). Đây là đối chứng "tiêu nhiều bit hơn". Tập baseline cùng ngân sách thành {`TT`, `ALT`, `NDET`, `BIG`}; thêm baseline chỉ làm cổng khó hơn.
- **Báo thêm cho mỗi phương án:** bit đạt được (trung bình, lớn nhất), tỉ lệ epoch gửi đúng như yêu cầu, F1 / P / R; `FF` không giới hạn chỉ là tham chiếu.
- **Cách đọc đã định trước:** nếu `VIEW` vượt ngưỡng so với {`TT`, `ALT`} nhưng không vượt so với `NDET` hoặc `BIG`, kết luận là "có cơ hội theo góc nhìn, nhưng một quy tắc không cần GT đã lấy được nó"; vẫn **không** phải GO cho `VIEW`.
- Kết quả chỉ áp dụng cho các chuỗi đạt sàng lọc ID (8/11); ngưỡng tỉ lệ dương ≥ 73% giữ nguyên (≥ 6/8).

Code: `swarmbridge/experiments/uav_semcom/h1_b1_view.py`; 17 test tổng hợp ở `test_h1_b1.py` (không dùng bản ghi dataset).

### Trạng thái sau phản biện #20: Sửa đổi 1 được chấp nhận, protocol B1 đóng băng

- `B_ref` = max nửa đầu và mức quyết định κ = 1 giữ nguyên; không đổi sang trung vị.
- Đây là so sánh **cùng giới hạn bit** (matched cap), **không** phải cùng số bit đạt được. Báo cáo phải kèm tỉ lệ dùng ngân sách, tỉ lệ bit `VIEW` / từng baseline, tỉ lệ fallback, và độ nhạy trên mọi κ đã đăng ký.
- `BIG` nằm trong cổng quyết định. So sánh với tập gốc {`TT`, `ALT`, `NDET`} được báo kèm (`original_baselines`) nhưng **không** quyết định.
- `BIG` được chọn theo kích thước FRAME của nửa đầu; không mặc định nó tiêu nhiều bit nhất ở nửa sau.
- Baseline của cổng là phương án tốt nhất trong tập **trên nửa được chấm**: một đối chứng bảo thủ, không phải cách chọn chính sách triển khai được. Nếu `VIEW` không vượt, kết luận là "không vượt tập đối chứng mạnh của protocol", không phải "mọi chiến lược hợp tác đều vô ích".
- Tham số `--data-use-basis` chỉ ghi lại nguồn xác nhận; nó không xác minh và không tạo ra quyền dùng dữ liệu.
- Không thêm thay đổi nào vào protocol trước khi chạy. Chưa chạy trên MDMT.

