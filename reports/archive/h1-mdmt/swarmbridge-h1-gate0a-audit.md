# SwarmBridge H1: kiểm toán phương pháp Gate 0A (hiệu chỉnh khám phá)

> **Hiệu chỉnh v2 (12/11, theo phản biện thứ bảy; mục 8):**
> - Chế độ lịch sử và snapshot TOK của v1 chỉ trừ bản kê của UAV2. v2 trừ bản kê của **cả hai** UAV, cộng bitmap cho mỗi UAV được điều phối.
> - Kết quả mới ở `gate0a_audit_v2.json`; v1 giữ nguyên.
> - **S2m24 / S2m52 là "oracle liên kết hoàn hảo có tính phí"**: trừ bit bản kê, nhưng **không** thực hiện liên kết xuyên góc. Chúng không phải cơ chế triển khai được, và không cho thấy cải thiện thực tế.
> - Ở mục 0–3, cột "Δ có tính bản kê" của chế độ lịch sử là số v1; số đúng ở mục 8.

Ngày 12/11/2026. Viết theo phản biện thứ sáu về [swarmbridge-h1-gate0a-result.md](swarmbridge-h1-gate0a-result.md).

- **Code:** `swarmbridge/experiments/uav_semcom/h1_gate0a_audit.py`.
- **Test:** 56 test, đều qua; commit trước khi chạy (`c24b27d2`, sửa `5c738721`).
- **Kết quả:** `runs/uav_semcom/h1/gate0a_audit_v1.json`. Đây là phiên bản riêng; `gate0a_v21.json` giữ nguyên.

**Phạm vi:** chỉ EXPL (nhãn MDMT-test công khai), chỉ báo cáo hoàn hảo. Đây là **hiệu chỉnh khám phá, không phải xác nhận**. Chưa chạy Gate 0B, chưa mở VisionTrack test, chưa train.

## 0. Kết luận

| Mục | Kết luận |
|---|---|
| **S2 bitmap** | Là **oracle danh tính có trễ**, không phải cơ chế triển khai được. Nó dùng GT ID đầy đủ của hai UAV ở epoch trước (`Cp = Ap & Bp`), kể cả đối tượng chưa từng được gửi. |
| **S2 khi tính chi phí bản kê** | Để BS biết danh sách chung, mỗi UAV phải gửi một bản kê (manifest) các track hiện có. Khi tính bit uplink của bản kê (liên kết vẫn lý tưởng):<br>• **TOK: phối hợp không bao giờ dương:** 0 ở 250 bit, từ −5,6 đến −42 điểm ở các mức cao hơn, vì bản kê tốn gần bằng chính token;<br>• **CROP: phối hợp giữ được 69–82% lợi ích lý tưởng**, vì bản kê rẻ so với mảnh ảnh. |
| **S0 của 0A không tối ưu** | Có hai phản ví dụ, kiểm bằng vét cạn. S0 chính xác (liệt kê cho TOK, MILP cho CROP) cho thấy:<br>• TOK: S0 cũ thấp hơn tối ưu ở 29–575/590 epoch tuỳ ngân sách, nên **Δ_free bị đánh giá thấp** (ví dụ 1k: 1,0 → 5,1);<br>• CROP: greedy gần như tối ưu (≤ 3/590 epoch khác). |
| **Baseline mạnh hơn** (mỗi UAV nhớ lịch sử gửi, chỉ gửi đối tượng BS đang có box cũ; α cố định) | Dư địa CROP **vẫn còn**:<br>• oracle hơn baseline +4,1 (100k), +9,7 (200k), +12,4 (500k);<br>• có tính chi phí bản kê: +3,8, +8,8, +12,2.<br>TOK có tính bản kê vẫn âm. |
| **Nhánh CROP dùng GT** | Đúng như phản biện: mảnh ảnh được cắt theo box GT và được coi là luôn cho detection đúng. Dư địa CROP vì vậy là **cận lý tưởng**. Các điều kiện cần để nó thành giá trị thật ở mục 4. |
| **Khuyến nghị** | **TOK: NO-GO.** **CROP: GO có điều kiện** cho một phép thử nhỏ **chỉ trên TUNE**, với detector thật, khi có quyền dùng ảnh MDMT. **Không chạy HOLD.** |

## 1. S2: đây là oracle danh tính có trễ, cần thông tin gì để thành thật

**Code 0A:**
```
Cp = Ap & Bp          # GT ID đầy đủ của hai UAV ở epoch trước
sup = set(B) & Cp     # UAV2 bỏ các đối tượng thuộc Cp
```

- BS ngoài đời chỉ biết những gì đã được gửi. Đối tượng UAV2 chưa từng gửi thì BS không biết nó tồn tại, nên cũng không biết nó trùng với UAV1.
- Con số 103 bit DL/epoch của 0A không gồm chi phí cho BS biết danh sách track, cho UAV2 biết bitmap ứng với track nào, và sửa lỗi khi ID không khớp.

**Thông tin tối thiểu để một S2 thật hoạt động** (giao thức tuần tự trong một epoch):
1. Mỗi UAV gửi **bản kê**: một dòng cho mỗi track hiện có, gồm ID cục bộ và vị trí đủ để liên kết. Ví dụ 24 bit (ID 10 bit + tâm box lượng tử 2 × 7 bit), hoặc token đầy đủ 52 bit.
2. BS **liên kết** bản kê hai bên. Ở đây liên kết vẫn lý tưởng; lỗi liên kết thật là việc của Gate 1. BS gửi lại bitmap đánh chỉ số theo **thứ tự trong bản kê cùng epoch**, kèm số thứ tự bản kê.
   - Vì bitmap gắn với bản kê vừa nhận, không cần đồng bộ danh sách track qua nhiều epoch.
   - ID không khớp chỉ còn là lỗi liên kết.
3. UAV gửi payload trừ các mục bị đánh dấu.

Giao thức này thêm một vòng UL → DL trong mỗi epoch. Độ trễ đó **chưa** được mô hình (0A là offline).

**Kết quả khi tính chi phí bản kê** (ρ = 1, liên kết lý tưởng, không trễ). Δ so với baseline độc lập tốt nhất:

| Nhánh | B | Δ oracle chính xác | Δ S2 trễ (0A) | **Δ S2m24** | Δ S2m52 |
|---|---|---|---|---|---|
| TOK | 2k | +7,4 | +2,5 | **−24,4** | −33,5 |
| TOK | 4k | +8,3 | +4,2 | **−25,2** | −40,0 |
| TOK | 6k | +9,7 | +6,3 | **−17,8** | −42,4 |
| TOK | 8k | +6,3 | +4,4 | **−9,6** | −36,1 |
| CROP | 100k | +5,1 | +4,1 | **+3,5** | +2,8 |
| CROP | 200k | +9,0 | +6,9 | **+6,7** | +6,0 |
| CROP | 500k | +11,8 | +8,9 | **+9,7** | +9,3 |

**Đọc kết quả:**
- **Với TOK, phối hợp tự triệt tiêu.** Muốn biết cái gì trùng, BS phải nhận một mô tả của mọi đối tượng, và mô tả đó gần bằng chính token.
- **Với CROP,** bản kê vài kbit/epoch là nhỏ so với mảnh ảnh hàng trăm kbit, nên phần lớn lợi ích còn lại. Ở 500k, S2m24 còn hơn S2 trễ, vì nó dùng thông tin của epoch hiện tại.

## 2. S0: phản ví dụ và oracle chính xác

**Hai phản ví dụ** (test `test_counterexample_*`, kiểm bằng vét cạn):
1. **TOK, B = 250 bit, A = {1, 2}, B = {3}:**
   - S0 của 0A tính trước header cho cả hai UAV (288 bit) nên gửi 0 token.
   - Tối ưu: UAV1 gửi 2 token (48 + 96 + 2 × 52 = 248 bit), tức 2 đối tượng.
2. **CROP, hai mảnh 100 bit ở UAV1, một mảnh 90 bit ở UAV2, B = 200 + 144:**
   - Greedy chọn mảnh rẻ nhất trước, mở hai thông điệp, trả hai header, nên được 1 đối tượng.
   - Tối ưu: gửi cả hai mảnh từ UAV1, được 2 đối tượng.

**Oracle chính xác:**
- TOK: liệt kê số token mỗi UAV gửi.
- CROP: MILP, có biến số gói của từng UAV.
- Cả hai đều khớp vét cạn trên 25 ví dụ ngẫu nhiên mỗi loại.

| Nhánh | B | S0 0A | S0 chính xác | Số epoch chính xác > 0A | Δ_free 0A → chính xác |
|---|---|---|---|---|---|
| TOK | 1k | 18,6 | 22,8 | 572/590 | 1,0 → 5,1 |
| TOK | 2k | 43,0 | 45,5 | 352 | 4,9 → 7,4 |
| TOK | 4k | 71,4 | 73,0 | 220 | 6,7 → 8,3 |
| TOK | 6k | 89,6 | 90,5 | 131 | 8,7 → 9,7 |
| CROP | 100k–1M | không đổi | không đổi | 0–3 | không đổi |

- Lỗi này làm **trần lý tưởng của TOK thấp hơn thật**, không làm lợi ích bị phóng đại.
- Với oracle chính xác, luật đọc của 0A cho TOK chỉ còn "có dư địa" ở 4–8k, và chỉ khi tính bằng S2 trễ. Nhưng S2 trễ là oracle, nên điều này không còn ý nghĩa thực tế (mục 1).

## 3. Baseline không phối hợp mạnh hơn: nhớ lịch sử của chính mình

**Chế độ thời gian:**
- BS giữ box cuối cùng nhận được cho mỗi (góc, ID). Một đối tượng được tính "biết đúng" nếu ở một góc nó đang hiện diện, box BS có IoU ≥ 0,5 với GT hiện tại.
- Mỗi UAV biết mình đã giao gì (ACK lý tưởng). Nó gửi trước những đối tượng mình **chưa từng gửi**, rồi đến những đối tượng có box cũ.
- Tỉ lệ chia ngân sách α được **cố định** cho mọi epoch; chọn α tốt nhất trong 5 giá trị, chọn trên EXPL (có lợi cho baseline).
- Phối hợp chỉ còn tác dụng là bỏ các lần gửi lại từ góc kia.
- S0-hist chính xác **từng epoch** (cận cận thị, không phải tối ưu nhiều epoch).

Dòng dữ liệu khi không giới hạn ngân sách: TOK 4,47 kbit/epoch (so với 5,6 ở chế độ snapshot); CROP 569 kbit/epoch. Vật di chuyển, nên trong 10 khung box thường đã cũ.

| Nhánh | B | Baseline lịch sử tốt nhất | S0-hist | S2m24-hist | S2m52-hist | Δ oracle | **Δ có tính bản kê** |
|---|---|---|---|---|---|---|---|
| TOK | 1k | 24,5 | 29,4 | 2,6 | 0,1 | +4,9 | **−21,9** |
| TOK | 2k | 46,3 | 53,4 | 18,7 | 8,3 | +7,1 | **−27,6** |
| TOK | 4k | 71,4 | 80,8 | 44,8 | 26,8 | +9,4 | **−26,6** |
| TOK | 6k | 83,6 | 93,1 | 71,7 | 41,0 | +9,5 | **−11,9** |
| CROP | 50k | 13,7 | 16,7 | 16,3 | 15,7 | +3,0 | **+2,6** |
| CROP | 100k | 27,0 | 31,1 | 30,8 | 30,2 | +4,1 | **+3,8** |
| CROP | 200k | 44,4 | 54,1 | 53,2 | 52,7 | +9,7 | **+8,8** |
| CROP | 500k | 75,8 | 88,2 | 88,0 | 87,9 | +12,4 | **+12,2** |

Baseline nhớ lịch sử **không** xoá được dư địa của CROP. Dư địa TOK vẫn có ở mức oracle nhưng không đạt được khi phải tính chi phí bản kê.

## 4. Nhánh CROP: rò rỉ GT và điều kiện cần để thành giá trị thật

**Rò rỉ trong 0A:**
- Mọi đối tượng GT đều được coi là cần mảnh ảnh.
- Vị trí cắt lấy từ box GT.
- Mảnh ảnh luôn cho detection đúng.

Ngoài đời, nếu YOLO11n trên UAV đã bỏ sót một vật, UAV không biết chỗ nào để cắt.

**Cách tạo ứng viên khả thi (không dùng GT):**
1. Vùng quanh **detection độ tin cậy thấp** (0,05 ≤ conf < τ). R1a trên VisDrone-DET cho thấy chọn ô theo detection không chắc chỉ kém ô chọn bằng nhãn 0,2–0,5 điểm recall. Nhưng lợi ích recall của ROI ở R1a đến kèm nhiều FP, và AP thấp hơn ảnh đầy đủ.
2. Vùng **chuyển động** (khác biệt giữa khung sau khi bù chuyển động camera).
3. **Ô cố định** trong vùng chồng lấn, chọn theo tỉ lệ detection yếu.

Mọi ứng viên sai vẫn tốn bit và tốn thời gian mã hoá trên UAV.

**Điều kiện cần (đề xuất ngưỡng loại, chờ duyệt).** Mỗi điều kiện đo được trên **TUNE** với detector thật:

| | Điều kiện | Đo bằng | Đề xuất ngưỡng loại |
|---|---|---|---|
| C1 | Detector UAV bỏ sót hoặc không chắc một phần đáng kể đối tượng | Tỉ lệ GT không có detection conf ≥ 0,25 khớp (YOLO11n @640) | < 15% → CROP gần như không có gì để thêm |
| C2 | Ứng viên phủ được phần bỏ sót | Recall của ứng viên trên tập bỏ sót, cùng số ứng viên sai mỗi khung | Recall < 50% → NO-GO |
| C3 | BS phục hồi được đối tượng từ mảnh ảnh mà precision chấp nhận được | Recall và precision của YOLO11s trên mảnh ảnh | Lợi ích F1 so với chỉ token ≤ 0 → NO-GO |
| C4 | Ứng viên mảnh ảnh của hai UAV thật sự trùng nhau, hoặc một UAV đã chắc chắn, nên mảnh ảnh của UAV kia thừa | Tỉ lệ ứng viên có cùng GT ID ở góc kia (chỉ để đo) | < 20% → phối hợp ít giá trị (mảnh ảnh vẫn có thể có ích cho từng UAV) |
| C5 | UAV kịp mã hoá mảnh ảnh trong thời gian epoch | Thời gian mã hoá AVIF trên CPU cho tổng mảnh ảnh mỗi epoch | > chu kỳ epoch → phải tính vào ràng buộc (bài học L1 của R1b) |
| C6 | Liên kết xuyên góc không GT với bản kê 24–52 bit | Gate 1 | Để sau |

**Ghi chú về C4:** nếu UAV1 đã phát hiện chắc một vật mà UAV2 không chắc, mảnh ảnh của UAV2 là thừa. Đây là dạng phối hợp "bổ sung độ tin cậy xuyên góc", khác với khử trùng thuần tuý, và có thể là nguồn lợi ích chính. Chỉ đo được khi có detector.

## 5. Thống kê và chọn cơ chế

- **Không chọn cơ chế tốt nhất trên EXPL nữa:** bảng ở mục 1 và 3 dùng **S2m24 cố định**. Con số "best real" của 0A (chọn cơ chế tốt nhất theo từng ngân sách trên chính EXPL) lạc quan, và không phải bằng chứng tổng quát hoá.
- **Biến thiên theo chuỗi** (Δ so với IND chia đều; 14 chuỗi):

  | Nhánh | B | S0 chính xác: trung vị [IQR], số chuỗi > 0 | S2 trễ | **S2m24** |
  |---|---|---|---|---|
  | TOK | 4k | +9,5 [+2,0; +14,3], 13/14 | +5,0 [+1,4; +7,8], 12/14 | **−18,8 [−38,0; −0,1], 3/14** |
  | TOK | 6k | +14,1 [0; +23,0], 9/14 | +8,3 [−0,3; +18,8], 9/14 | **0,0 [−22,5; 0], 3/14** |
  | CROP | 100k | +5,6 [+1,4; +10,8], 14/14 | +3,7 [+1,4; +8,2], 14/14 | **+3,3 [+0,7; +7,6], 14/14** |
  | CROP | 200k | +12,6 [+8,2; +16,8], 14/14 | +9,1 [+4,7; +11,5], 14/14 | **+8,8 [+4,5; +11,5], 14/14** |
  | CROP | 500k | +14,2 [+3,7; +27,4], 12/14 | +11,6 [+2,5; +23,9], 11/14 | **+12,3 [+3,3; +24,8], 12/14** |

- Ngưỡng của 0A **không** được đổi. Kết luận 0A được giữ như bản ghi khám phá, kèm hiệu chỉnh này.

## 6. Ghi chép thủ tục

- **Lỗi thống kê theo chuỗi:** lần chạy kiểm toán đầu có số liệu S2 trễ theo chuỗi luôn bằng 0, vì dòng cộng dồn chưa được thêm (một lệnh sửa chuỗi trong code không khớp và không báo lỗi).
  - Đã sửa, và thêm kiểm tra lúc chạy: tổng theo chuỗi phải bằng tổng gộp (`5c738721`).
  - Số gộp không bị ảnh hưởng. Mọi số trong báo cáo này lấy từ lần chạy sau khi sửa.
- **Thời gian chạy:** 5 phút 19 giây; khoảng 17.000 bài MILP, mỗi bài giới hạn 30 giây, không bài nào chạm giới hạn.

## 7. Khuyến nghị GO / NO-GO cho Gate 0B

1. **H1-TOK: NO-GO.** Dư địa chỉ có ở mức oracle. Khi tính chi phí để BS biết cái gì trùng, phối hợp âm ở mọi ngân sách. Giữ làm kết quả âm có giải thích.
2. **H1-CROP: GO có điều kiện**, cho **0B-TUNE pilot**:
   - chỉ trên MDMT train, khi đã có quyền dùng ảnh;
   - detector thật: YOLO11n trên UAV, YOLO11s ở BS;
   - đo C1–C5 ở mục 4; ngưỡng loại khoá trước khi chạy;
   - **không chạy HOLD**, không có tuyên bố xác nhận.
3. **Nếu chưa có quyền dùng ảnh MDMT:** không có tập thay thế tốt. VisionTrack `train` có vật to, nên C1 nhiều khả năng thất bại. Có thể dùng nó làm **đối chứng âm** cho C1–C2 nếu được duyệt, nhưng không làm bằng chứng chính.
4. **Liên hệ với A5/A5.5:**
   - Lợi ích phối hợp tập trung ở vùng ngân sách xấp xỉ dòng dữ liệu, nên quyết định "khi nào phối hợp" phụ thuộc trạng thái mạng.
   - Đây vẫn chỉ là giả thuyết; chỉ đáng đưa PPO vào sau khi C1–C4 cho thấy giá trị nhiệm vụ thật với detector thật.


## 8. Hiệu chỉnh v2: chi phí bản kê của cả hai UAV (phản biện thứ bảy)

**Lỗi:**
- Trong `_hist_run()`, S2m chỉ trừ `m · |B|`, tức bản kê của UAV2.
- Ở snapshot TOK, tôi giả định token của UAV1 kiêm bản kê. Điều đó chỉ đúng khi UAV1 gửi hết token, điều không được bảo đảm dưới ngân sách.
- Ở chế độ lịch sử, oracle chọn đối tượng cho **cả hai** UAV, nhưng chỉ trừ một bitmap.

**Giao thức nhân quả dùng cho v2:**
1. Mỗi UAV gửi bản kê m bit/track của mọi track hiện có, mỗi bản kê là một thông điệp riêng có header.
2. BS liên kết. Ở đây liên kết vẫn lý tưởng, theo GT; đây là phần oracle.
3. BS gửi bitmap cho mỗi UAV mà lựa chọn được điều phối: snapshot chỉ UAV2, lịch sử cả hai. Mỗi bitmap có 1 bit/mục cộng 48 bit header, ρ = 1.
4. UAV gửi payload.

Chi phí nằm trong một hàm chung `coord_costs()`. Có 4 test kế toán ngân sách mới, gồm một test sẽ rớt nếu chỉ trừ bản kê UAV2. Tổng cộng 74 test qua.

**Kết quả v2:** Δ so với baseline độc lập tốt nhất, điểm recall đối tượng duy nhất; báo cáo hoàn hảo; EXPL.

| Chế độ | Nhánh | B | Δ oracle chính xác | Δ S2m24 v1 | **Δ S2m24 v2** | Δ S2m52 v2 |
|---|---|---|---|---|---|---|
| Lịch sử | CROP | 50k | +2,9 | +2,6 | **+2,0** | +1,1 |
| Lịch sử | CROP | 100k | +4,0 | +3,8 | **+3,5** | +2,3 |
| Lịch sử | CROP | 200k | +9,8 | +8,8 | **+8,6** | +8,0 |
| Lịch sử | CROP | 500k | +12,4 | +12,2 | **+12,1** | +11,8 |
| Lịch sử | TOK | 2k | +7,1 | −27,5 | **−39,8** | −46,2 |
| Lịch sử | TOK | 4k | +9,4 | −26,6 | **−45,5** | −57,7 |
| Snapshot | CROP | 100k / 200k / 500k | +5,1 / +9,0 / +11,8 | +3,5 / +6,7 / +9,7 | **không đổi** | +2,8 / +6,0 / +9,3 |
| Snapshot | TOK | 4k | +8,3 | −25,2 | **−40,0** | −53,4 |

- **Theo chuỗi** (snapshot CROP, S2m24): 14/14 chuỗi dương ở 50k–200k, 12/14 ở 500k. Snapshot TOK: 0–1/14.
- **Cách đọc đúng của các số S2m:** nếu BS liên kết hoàn hảo và chỉ phải trả thêm bit bản kê, phối hợp CROP còn lợi bao nhiêu. Chúng **không** trả lời được câu "bản kê 24 bit có đủ để liên kết hai góc nhìn rất khác nhau không". Câu đó thuộc Gate 1, chưa mở.
- **GT trong baseline nhớ lịch sử:**
  - box là box GT (báo cáo hoàn hảo);
  - track ID theo góc là GT ID (tracker cục bộ hoàn hảo);
  - "box cũ" được đo bằng IoU với box GT hiện tại;
  - ACK lý tưởng (không mất, không trễ).
- **Kết luận không đổi:**
  - TOK: phối hợp bằng giao thức bản kê này là NO-GO. Đây **không** có nghĩa token vô dụng làm một hành động trong hệ SemCom có RL.
  - CROP: còn dư địa ở mức oracle có tính phí; đáng làm pilot với detector thật.
