# A7-S2: phản hồi của BS không dùng GT có thay được hiểu biết về cảnh không?

Ngày 13/11/2026, theo phản biện #16 (duyệt phân tích offline; **không** duyệt train PPO).

- **Code:** `swarmbridge/experiments/uav_semcom/a7s_semfam.py`. **Kết quả:** `runs/uav_semcom/h1/a7s2_bsproxy_v1.json`.
- Chỉ dùng bản ghi YOLO11n / YOLO11s / AVIF đã có. Không ảnh mới, không val / HOLD, không train.
- Giữ nguyên fold, chế độ, seed, họ chính sách và hệ số phần thưởng. Phần thưởng luôn chấm bằng giá trị thật của khung hiện tại.
- Các hằng số của cơ chế thăm dò (số FRAME thăm dò mỗi cảnh ∈ {1, 2, 3}, trễ phản hồi 1 s, 64 bit DL mỗi phản hồi) được ghi trong code trước khi chạy.

**Kết luận ngắn: NO-GO cho cơ chế phản hồi đang thử** (proxy đếm detection). Kết luận này chỉ nói về proxy đó, không suy rộng.

## 1. Proxy và thời điểm có thông tin

**BS có gì, và khi nào:** chỉ sau khi một FRAME được giao đúng hạn và giải mã, BS mới có detection của YOLO11s trên khung đó. FRAME mang kèm token của UAV; bit token đã được cộng vào bit của FRAME (khoảng +0,4%).

**Proxy không dùng nhãn**, tính từ hai tập khai báo ở conf ≥ 0,3:
- `new` = số detection của FRAME không khớp token nào (khớp theo lớp, IoU ≥ 0,5);
- `lost` = số token không được FRAME xác nhận;
- proxy hiệu chỉnh = a + b·new + c·lost, fit tuyến tính với hiệu giá trị thật v_FRAME − v_TOKEN **chỉ trên các chuỗi train** của mỗi fold. GT của test chỉ dùng để đánh giá.

## 2. Proxy khớp giá trị thật đến đâu (tập test của mỗi fold)

| | Fold A | Fold B |
|---|---|---|
| Tương quan theo bản ghi: `new` / `new − lost` / hiệu chỉnh | 0,70 / 0,56 / 0,71 | 0,91 / 0,82 / 0,92 |
| Tương quan trung bình theo cảnh (chuỗi-góc) | 0,80 (10 cảnh) | 0,96 (12 cảnh) |
| Tương quan trong cùng cảnh (đã trừ trung bình cảnh) | 0,44 | 0,51 |
| Cùng dấu ở mức cảnh | 80% | 75% |

Trung bình theo cảnh, giá trị thật / proxy:

| Cảnh | Thật | Proxy | | Cảnh | Thật | Proxy |
|---|---|---|---|---|---|---|
| 26-1 | +0,55 | +0,59 | | 34-1 | −0,08 | **+0,17** |
| 26-2 | +0,04 | +0,10 | | 34-2 | +0,65 | +0,44 |
| 31-1 | +0,13 | **+0,36** | | 48-1 | +0,00 | +0,06 |
| 31-2 | +0,22 | **+0,61** | | 48-2 | −0,01 | +0,05 |
| 57-1 | +0,03 | −0,04 | | 52-1 | −0,01 | +0,11 |
| 57-2 | +0,06 | −0,00 | | 52-2 | +0,56 | +0,39 |
| 59-1 | +0,20 | +0,13 | | 56-1 | +0,01 | +0,04 |
| 59-2 | +0,08 | +0,14 | | 56-2 | +0,14 | +0,11 |
| 61-1 | +0,30 | +0,27 | | 62-1 | +0,02 | +0,06 |
| 61-2 | +0,54 | +0,58 | | 62-2 | +0,17 | +0,13 |
| | | | | 68-1 | +0,29 | +0,34 |
| | | | | 68-2 | +1,38 | +0,76 |

Proxy xếp hạng cảnh khá tốt, nhưng **đánh giá quá cao** ở các cảnh đông (31-1, 31-2, 34-1): ở đó FRAME cho nhiều detection "mới", nhưng phần lớn không phải lợi ích ròng (FP, hoặc không bù được chi phí).

## 3. Trong môi trường A5 (test, 100 seed; Δ return so với FIX; trong ngoặc là phần dư địa ORA − FIX giữ được)

| Fold, chế độ | CAU | SCN: trung bình cảnh theo GT | PRX0: trung bình proxy của cảnh, biết sẵn | **EE: thăm dò–khai thác nhân quả** | ORA − FIX |
|---|---|---|---|---|---|
| A, ref | −0,1 (−3%) | +2,7 (55%) | +1,0 (20%) | **+0,4 ± 0,3 (8%)** | 4,9 |
| A, K5 | −9,1 | +1,8 (30%) | −3,3 (−53%) | **−5,6 ± 0,7 (−91%)** | 6,2 |
| B, ref | +7,6 (80%) | +8,1 (85%) | +7,6 (80%) | **+6,7 ± 0,4 (71%)** | 9,5 |
| B, K5 | +1,9 (14%) | +11,2 (80%) | +3,0 (21%) | **+2,6 ± 0,5 (18%)** | 14,0 |

- **SCN** là tham chiếu, dùng GT: bộ điều khiển chỉ biết *một con số cho mỗi cảnh* (trung bình thật của v_FRAME − v_TOKEN).
- **PRX0** là chẩn đoán: trung bình proxy của cả cảnh, biết sẵn không mất phí. Nó tách chất lượng proxy khỏi chi phí thăm dò.
- **EE** nhân quả: với mỗi cảnh, gửi FRAME cho đến khi có N phản hồi từ những FRAME đã thực sự giao (phản hồi đến sau 1 s); sau đó dùng trung bình proxy của các phản hồi đó. Mỗi episode bắt đầu không biết gì. Giả định: UAV biết mình đang ở cảnh nào (như biết vị trí và góc nhìn).
- Chi phí phản hồi DL của EE: 1,3–8,3 kbit mỗi episode, không đáng kể. Chi phí thật là các FRAME thăm dò: EE gửi 14–84 Mbit mỗi episode so với 0,3–0,5 Mbit của FIX.
- phy_mcs_blind: dư địa gần 0, không diễn giải (số có trong JSON).

**Theo cổng phác thảo ở A7-S1** (proxy giữ ≥ 50% dư địa ở ref và K5, cả hai fold): PRX0 đạt 1/4 ô, EE đạt 1/4 ô → **trượt**.

## 4. Theo từng cảnh (chuỗi-góc), chế độ ref: Δ utility mỗi nhiệm vụ so với FIX (tỉ lệ chọn FRAME)

| Cảnh | SCN | EE | ORA |
|---|---|---|---|
| 26-1 | +0,158 (99%) | +0,153 (99%) | +0,162 (93%) |
| 26-2 | −0,008 (0%) | −0,025 (25%) | −0,004 (11%) |
| 31-1 | −0,009 (20%) | **−0,044 (98%)** | +0,001 (25%) |
| 31-2 | +0,004 (78%) | **−0,010 (99%)** | +0,029 (50%) |
| 57-1 | −0,009 (0%) | −0,017 (8%) | −0,004 (2%) |
| 57-2 | −0,009 (0%) | −0,017 (9%) | −0,004 (6%) |
| 59-1 | +0,012 (70%) | −0,002 (31%) | +0,042 (27%) |
| 59-2 | −0,008 (0%) | −0,025 (40%) | +0,017 (3%) |
| 61-1 | +0,048 (92%) | +0,037 (76%) | +0,061 (68%) |
| 61-2 | +0,154 (99%) | +0,151 (99%) | +0,156 (100%) |
| 34-1 | −0,008 (0%) | **−0,069 (55%)** | −0,005 (0%) |
| 34-2 | +0,216 (100%) | +0,215 (100%) | +0,219 (95%) |
| 48-1 | −0,008 (0%) | −0,015 (9%) | −0,004 (1%) |
| 48-2 | −0,007 (0%) | −0,014 (9%) | −0,005 (1%) |
| 52-1 | −0,009 (0%) | −0,026 (25%) | −0,002 (6%) |
| 52-2 | +0,169 (100%) | +0,168 (100%) | +0,174 (97%) |
| 56-1 | −0,007 (0%) | −0,016 (12%) | −0,005 (0%) |
| 56-2 | +0,001 (38%) | −0,006 (19%) | +0,016 (38%) |
| 62-1 | −0,008 (0%) | −0,016 (10%) | −0,004 (2%) |
| 62-2 | −0,005 (57%) | −0,009 (28%) | +0,027 (31%) |
| 68-1 | +0,039 (93%) | +0,034 (94%) | +0,058 (65%) |
| 68-2 | +0,571 (100%) | +0,570 (100%) | +0,572 (100%) |

## 5. Đọc kết quả

1. **Giá trị của FRAME là đặc tính của góc nhìn** (trả lời yêu cầu báo theo từng UAV view).
   - Lợi ích nằm ở 26-1, 61-1, 61-2, 34-2, 52-2, 68-2 (và nhẹ ở 68-1). Ở đó ORA chọn FRAME 65–100%.
   - Ở góc còn lại của cùng chuỗi (26-2, 34-1, 52-1), ORA chọn gần 0%.
   - Vậy tỉ lệ ~50% theo chuỗi ở A7-S1 là do **một trong hai UAV** gần như luôn cần FRAME, không phải hai UAV đổi mode theo thời gian.
2. **EE nhận đúng các cảnh "đáng gửi"**: ở 26-1, 61-2, 34-2, 52-2, 68-2 nó đạt gần bằng ORA.
3. **EE thua ở hai chỗ:**
   - **Proxy báo dương giả** ở cảnh đông (31-1, 31-2, 34-1): EE gửi FRAME 55–99% trong khi ORA gửi 0–50%, và mất utility. Dưới nghẽn (A-K5) điều này làm return âm.
   - **Chi phí thăm dò** ở các cảnh FRAME không đáng: mỗi cảnh mất khoảng −0,01 đến −0,02 utility mỗi nhiệm vụ.
4. **B-K5:** EE nhận đúng cảnh nhưng chỉ gửi FRAME 19% ở các cảnh tốt, vì heuristic được điều chỉnh trên train (các chuỗi của fold A, nơi proxy gây hại) chọn cấu hình dè dặt. Đây là hệ quả của proxy thiếu tin cậy, không phải lỗi riêng.
5. **Ngay cả hiểu biết cảnh theo GT (SCN) cũng chỉ giữ 30–85%** dư địa. Một con số cho mỗi cảnh không đủ ở A-K5.

## 6. Kết luận

- **NO-GO cho cơ chế phản hồi BS dựa trên đếm detection (`new`, `lost`)**, với heuristic thăm dò–khai thác đã thử: chỉ có lợi rõ ở 1/4 ô (B-ref) và gây hại ở A-K5.
- Vì điều kiện 1 không đạt, tôi **không** viết thiết kế môi trường có cảnh kéo dài và không đề xuất PPO.
- **Điều không được suy ra:** kết quả này không nói rằng mọi tín hiệu phản hồi đều vô ích. Proxy xếp hạng cảnh tốt (tương quan 0,80 / 0,96); chỗ hỏng cụ thể là nó không phân biệt detection mới *đúng* với detection mới *sai* ở cảnh đông.
- **Một quan sát để ghi lại, chưa phải đề xuất:** thứ còn thiếu là một cách kiểm tra detection mới mà không cần nhãn. Hai UAV nhìn cùng khu vực là một nguồn như vậy (xác nhận chéo giữa hai góc), và đó cũng là chỗ duy nhất trong chuỗi kết quả này mà hợp tác đa UAV có vai trò tự nhiên. Nhưng nó cần liên kết xuyên góc thật, thứ chưa có (Gate 1 chưa mở).
- A5 / A5.5 vẫn là hướng chính.

## 7. Giới hạn

- Hai fold, 5–6 chuỗi mỗi phía; ± là CI theo seed và đánh giá thấp bất định theo chuỗi; bảng theo cảnh ở mục 4 để đọc trực tiếp.
- Môi trường A5 rút ảnh độc lập: "cảnh" ở đây chỉ là nhãn (chuỗi, góc) của ảnh được rút, thứ tự thời gian trong cảnh không được giữ. PRX0 và SCN dùng thông tin của cả cảnh (kể cả khung "tương lai") nên chỉ là chẩn đoán; chỉ EE là nhân quả.
- Thời gian suy luận ở BS và trễ phản hồi là giả định (0,05 s và 1 s).
- **Bit phản hồi DL chỉ được báo cáo, chưa được mô phỏng (phản biện #17):** `FB_DL_BITS` không đi qua bộ lập lịch downlink, không chiếm tài nguyên vô tuyến và không ảnh hưởng return. Mô hình chỉ có trễ phản hồi 1 s và bit token gửi kèm FRAME. Nếu tính đủ chi phí DL thì EE chỉ có thể kém hơn, nên kết luận NO-GO không đổi; không chạy lại.
- τ_decl = 0,3 chọn trên EXPL. Quyền dùng MDMT chưa xác minh độc lập; bảng theo khung nằm ngoài git.
