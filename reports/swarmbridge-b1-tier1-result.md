# B1 tầng 1: cơ hội hợp tác theo góc nhìn trên MDMT (sàng lọc có trợ giúp GT)

Ngày 13/11/2026. Protocol: [swarmbridge-a8-b1-preregistration.md](swarmbridge-a8-b1-preregistration.md), mục B1 và Sửa đổi 1 (đóng băng sau phản biện #20). Chạy offline trên bản ghi pilot 0B đã có; không ảnh mới, không train.

**Cơ sở dùng dữ liệu:** theo lời người dùng ngày 13/11/2026, giáo sư hướng dẫn đã cho phép lab dùng MDMT cho nghiên cứu này. Đây là xác nhận miệng do người dùng thuật lại; chưa có văn bản, tác giả bộ dữ liệu chưa trả lời, và bản dữ liệu lấy từ một mirror không chính thức. Dòng này chỉ ghi nguồn, không phải xác minh.

**Tính chất kết quả:** khám phá. Góc được chọn bằng GT của nửa đầu chuỗi, liên kết giữa hai góc là lý tưởng (theo ID của GT), nhãn thuộc tập EXPL đã dùng nhiều lần. Chỉ áp dụng cho 8 chuỗi qua sàng lọc ID (26, 31, 34, 52, 56, 57, 61, 62); ba chuỗi 48, 59, 68 bị loại.

## 1. Kết luận

- **NO-GO cho `VIEW`** theo cổng đã đăng ký: `VIEW` − baseline tốt nhất = **0,0 điểm F1**, dương ở **0/8 chuỗi**.
- Lý do: quy tắc không cần nhãn `NDET` (cấp FRAME cho góc có nhiều detection trên UAV hơn ở nửa đầu) chọn **đúng cùng một góc với `VIEW` ở cả 8 chuỗi**.
- Theo cách đọc đã định trước: **có cơ hội theo góc nhìn, nhưng một quy tắc không cần GT đã lấy được nó.** So với hai phương án không phân biệt góc, `VIEW` hơn `ALT` 2,5 điểm F1 (dương ở 7/8 chuỗi) và hơn `TT` 6,5 điểm.
- Hệ quả: chưa có lý do xây dựng cơ chế ước lượng giá trị góc nhìn phức tạp hơn (học, phản hồi từ BS, PPO) trên dữ liệu này.

## 2. Kết quả ở mức quyết định (κ = 1)

Trung bình trên 8 chuỗi, nửa sau của mỗi chuỗi. F1 / P / R là của vật thể duy nhất sau hợp nhất.

| Phương án | F1 | P | R | Bit đạt được (kbit / epoch) | Tỉ lệ dùng ngân sách | Gửi đúng yêu cầu |
|---|---|---|---|---|---|---|
| `TT` (cả hai TOKEN) | 0,638 | 0,747 | 0,565 | 5 | 0,01 | 1,00 |
| `ALT` (luân phiên) | 0,678 | 0,765 | 0,618 | 445 | 0,70 | 0,90 |
| `BIG` (góc có FRAME to hơn) | 0,688 | 0,775 | 0,628 | 478 | 0,73 | 0,80 |
| `NDET` (góc nhiều detection hơn) | **0,703** | 0,787 | 0,644 | 485 | 0,76 | 0,85 |
| **`VIEW`** (chọn bằng GT) | **0,703** | 0,787 | 0,644 | 485 | 0,76 | 0,85 |
| `F1T` (FRAME cố định UAV 1) | 0,649 | 0,740 | 0,587 | 351 | 0,56 | 0,82 |
| `TF2` (FRAME cố định UAV 2) | 0,703 | 0,785 | 0,645 | 536 | 0,83 | 0,97 |
| `FF` không giới hạn (tham chiếu) | 0,717 | 0,780 | 0,671 | 1.013 | — | — |

Đây là so sánh **cùng giới hạn bit**, không phải cùng số bit đạt được: `VIEW` dùng nhiều hơn `ALT` 9% bit và bằng `NDET`.

**`VIEW` trừ từng phương án (điểm F1; số chuỗi dương):**

| So với | Trung bình | Dương | Theo chuỗi 26 / 31 / 34 / 52 / 56 / 57 / 61 / 62 |
|---|---|---|---|
| `NDET` | 0,0 | 0/8 | 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 |
| `BIG` | +1,5 | 2/8 | 0 / 0 / 0 / 0 / +7,0 / +4,8 / 0 / 0 |
| `ALT` | +2,5 | 7/8 | −0,0 / +0,4 / +5,8 / +4,2 / +4,4 / +1,6 / +1,3 / +2,1 |
| `TT` | +6,5 | 7/8 | 0 / +3,5 / +12,8 / +9,0 / +7,2 / +5,3 / +9,1 / +5,0 |
| `F1T` | +5,4 | 7/8 | 0 / +0,9 / +14,4 / +9,0 / +7,0 / +4,8 / +2,4 / +4,6 |

- **Cổng đã sửa đổi** ({`TT`, `ALT`, `NDET`, `BIG`}): baseline tốt nhất là `NDET` → không đạt.
- **Tập baseline gốc** ({`TT`, `ALT`, `NDET`}, chỉ báo kèm): baseline tốt nhất cũng là `NDET` → không đạt. `BIG` không phải lý do của NO-GO.
- **Chỉ so với phương án không phân biệt góc** ({`TT`, `ALT`}): đạt cả hai ngưỡng (+2,5 điểm; 7/8 chuỗi).

## 3. Độ nhạy theo ngân sách

| κ | `TT` | `ALT` | `NDET` | `BIG` | `VIEW` | `VIEW` − baseline tốt nhất | Bit `VIEW` (kbit) | `VIEW` gửi đúng yêu cầu |
|---|---|---|---|---|---|---|---|---|
| 0,25 | 0,638 | 0,638 | 0,638 | 0,638 | 0,638 | 0,0 | 5 | 0,00 |
| 0,5 | 0,638 | 0,638 | 0,638 | 0,638 | 0,638 | −0,0 | 5 | 0,00 |
| 0,75 | 0,638 | 0,647 | 0,644 | 0,638 | 0,656 | +0,9 (so với `ALT`; dương 4/8) | 219 | 0,50 |
| **1** | 0,638 | 0,678 | 0,703 | 0,688 | 0,703 | 0,0 | 485 | 0,85 |
| 1,5 | 0,638 | 0,683 | 0,713 | 0,698 | 0,713 | 0,0 | 604 | 1,00 |
| 2 | 0,638 | 0,683 | 0,713 | 0,698 | 0,713 | 0,0 | 604 | 1,00 |

- Dưới 0,75 × B_ref, FRAME gần như không bao giờ vừa: mọi phương án rơi về TOKEN.
- Ở κ = 1, khoảng 15% epoch của `VIEW` phải rơi về TOKEN vì FRAME nửa sau lớn hơn mức lớn nhất của nửa đầu. Ở κ ≥ 1,5 không còn fallback và kết luận không đổi (`VIEW` = `NDET`; `VIEW` − `ALT` = +3,0, dương 8/8).
- Một FRAME đặt đúng góc (κ = 1,5: 0,713 với 604 kbit) đạt gần bằng hai FRAME (0,717 với 1.013 kbit).

## 4. Hai điều cần đọc cẩn thận

**Góc tốt hơn gần như luôn là UAV 2.**
- `VIEW` và `NDET` chọn UAV 2 ở 7/8 chuỗi. Phương án cố định `TF2` vì thế cho F1 trung bình gần bằng `VIEW` ở κ = 1 (0,7025 so với 0,7026).
- Ngoại lệ là chuỗi 26: cả `VIEW` lẫn `NDET` chọn UAV 1. Ở κ = 1 FRAME của UAV 1 không vừa ngân sách nên không có lợi ích; ở κ = 1,5 nó vừa và `VIEW` hơn `TF2` 7,0 điểm ở chuỗi này.
- Vì chỉ có một chuỗi như vậy, dữ liệu này **không tách được** "`NDET` nhận ra góc tốt hơn" khỏi "trong MDMT, UAV 2 thường là góc tốt hơn".

**Lợi ích là thuộc tính tĩnh của cặp góc, không phải thứ thay đổi theo thời gian.**
- Góc chọn ở nửa đầu vẫn đúng ở nửa sau trong mọi chuỗi. Điều này khớp với A7-S1 (giá trị FRAME là kiến thức cấp cảnh) và A7-S2 (giá trị FRAME là thuộc tính của từng góc).
- Không có dấu hiệu nào ở đây cho thấy cần một chính sách thích ứng theo epoch.

## 5. Giới hạn

- 8 chuỗi; nhãn EXPL đã dùng ở các bước trước; không có khoảng tin cậy (các hiệu theo chuỗi được liệt kê đầy đủ ở mục 2).
- Liên kết lý tưởng: FP trùng và FP liên kết bằng 0 theo cấu trúc. Hệ thống thật sẽ kém hơn.
- Baseline của cổng là phương án tốt nhất trong tập trên nửa được chấm: đối chứng bảo thủ, không phải cách chọn triển khai được.
- FRAME mã hoá AVIF mất khoảng 1,17 s trên CPU của UAV (pilot 0B), dài hơn một epoch 1 s. Chi phí thời gian này không nằm trong phép chấm.
- Kết quả "không vượt tập đối chứng của protocol" không chứng minh mọi chiến lược hợp tác đều vô ích.

## 6. Ý nghĩa cho hướng B

- **Điều được củng cố:** phân bổ bất đối xứng giữa hai UAV có giá trị thật. Đặt FRAME đúng góc hơn đặt sai góc 5,4 điểm F1 và hơn luân phiên 2,5 điểm, ở cùng giới hạn bit.
- **Điều bị loại:** nhu cầu về một bộ ước lượng giá trị góc nhìn dựa trên GT, phản hồi, hay RL. Số detection trên UAV ở vài chục giây đầu đã đủ, trên dữ liệu này.
- **Điều còn mở:** quy tắc `NDET` có đứng vững khi góc tốt hơn thay đổi giữa các cảnh hay trong một chuyến bay không. MDMT không trả lời được (7/8 chuỗi cùng một phía). Cần dữ liệu khác, ví dụ CrossTrack-UAV.
- Theo bản đăng ký, kết quả này **không** mở Gate 1, val / HOLD hay PPO.

## 7. Tệp

- Kết quả: `runs/uav_semcom/h1/b1_view_tier1_v1.json` (tổng hợp theo chuỗi; không có bảng theo từng khung).
- Code: `swarmbridge/experiments/uav_semcom/h1_b1_view.py`; sàng lọc ID: `runs/uav_semcom/h1/b1_id_check_v1.json`.

## 8. Quyết định sau phản biện #21 (13/11/2026)

**NO-GO được chấp nhận. Nhánh chọn góc nhìn tĩnh trên MDMT đóng lại; không thêm thí nghiệm nào trên nhánh này.**

- B1 được giữ như một kết quả âm mang tính khám phá.
- Phân bổ bất đối xứng có ích, nhưng ưu tiên góc tĩnh trên MDMT không đủ làm đóng góp "swarm intelligence": không train PPO / MAPPO, không xây proxy mới, không mở Gate 1 cho bài toán này.
- Quan sát "một FRAME đúng góc gần bằng hai FRAME với ít hơn khoảng 40% bit" được giữ lại, nhưng **không** được gọi là cải thiện hiệu quả đầu-cuối: thời gian mã hoá FRAME (1,17 s) vượt chu kỳ 1 s và B1 không xét hạn chót.
- Câu hỏi hợp tác đa UAV tổng quát vẫn mở, nhưng cần thiết lập khác: giá trị tương đối của các góc **thay đổi theo thời gian**, vùng quan sát chồng lấn, chất lượng quan sát thay đổi, và một thước đo triển khai được cho thông tin bổ sung.

**Bước tiếp theo là tiếp cận dữ liệu và thiết kế nghiên cứu, không phải tính toán:**
1. Theo thủ tục xin CrossTrack-UAV (thỏa thuận học thuật có chữ ký).
2. Trước khi chạy detector trên bất kỳ bộ dữ liệu mới nào, kiểm xem nó có nhiều lần **đảo chiều** góc tốt hơn và có đánh giá vật thể chồng lấn có nghĩa không.
3. So với góc cố định, `NDET`, heuristic động đơn giản và một oracle GT được đánh dấu, trước khi xét bất kỳ thuật toán RL nào. Đo trước xem oracle động có thật sự thắng heuristic động không.
4. Chỉ đăng ký cổng mới sau khi đã xác lập tính khả thi của dữ liệu và rà soát các công trình collaborative perception liên quan.

A5 / A5.5 được giữ như bằng chứng RL hiện có, không được trình bày như bằng chứng về hợp tác bầy đàn.

