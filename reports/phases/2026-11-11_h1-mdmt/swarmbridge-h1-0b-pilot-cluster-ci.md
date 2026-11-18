# SwarmBridge H1-CROP: độ tin cậy của pilot 0B (bootstrap theo chuỗi + bỏ từng chuỗi)

Ngày 13/11/2026, theo phản biện #13.

- **Đầu vào:** chỉ dùng `runs/uav_semcom/h1/pilot0b_expl_v1.json`. Không chạy YOLO, không đọc ảnh, không train.
- **Code:** `scripts/uav_semcom/h1_pilot0b_cluster_ci.py`. **Kết quả:** `runs/uav_semcom/h1/pilot0b_cluster_ci_v1.json`.

**Đây không phải kiểm chứng độc lập.** α và τ_decl được chọn trên chính 11 chuỗi EXPL này. Quyền dùng dữ liệu vẫn chưa được xác minh độc lập (xem báo cáo kết quả).

## 1. Cách tính

- **Đơn vị là chuỗi (11 cụm).** Hiệu ứng của mỗi chuỗi = F1 của điều kiện A − F1 của điều kiện B trong chuỗi đó, ở cấu hình đã chọn trong pilot. Hiệu ứng tổng = trung bình 11 chuỗi (macro).
- **Bootstrap ghép cặp theo cụm:** lấy mẫu lại 11 chuỗi có hoàn lại, 20.000 lần, cho khoảng tin cậy 95% theo phân vị. Không bootstrap 459 epoch như các mẫu độc lập.
- **Bỏ từng chuỗi (LOSO):** tính lại trung bình sau khi bỏ lần lượt từng chuỗi, và xem chuỗi 59 riêng.
- **Các hiệu ứng được tính:**
  - **C3:** F1 từng góc, T1 − T0 (T0+ trùng T0).
  - **C4:** F1 duy nhất, T2 − T1 và T2v − T1 (ρ = 0).
  - **Thêm:** F − T0 ở mức ngân sách đủ cho khung.

## 2. Kết quả

Đơn vị là điểm F1 (×100). "Phần của 59" là tỉ lệ tổng lợi ích đến từ chuỗi 59.

**C3: crop (T1) so với token (T0), F1 từng góc**

| q, B (kbit) | Trung bình [CI 95%] | Trung vị | Số chuỗi dương | Phần của 59 | Bỏ 59: trung bình [CI 95%], số dương |
|---|---|---|---|---|---|
| q35, 76 | +1,2 [−0,5; +3,6] | +0,4 | 7/11 | 85% | +0,2 [−0,7; +1,2], 6/10 |
| q35, 153 | +1,5 [−0,9; +5,0] | −0,5 | 5/11 | 99% | +0,0 [−1,2; +1,3], 4/10 |
| q35, 305 | **+2,6 [+0,1; +6,5]** | −0,0 | 5/11 | 70% | +0,8 [−0,2; +1,9], 4/10 |
| q50, 76 | +1,2 [−0,3; +3,3] | +0,9 | 7/11 | 81% | +0,3 [−0,5; +0,9], 6/10 |
| q50, 153 | +1,6 [−0,6; +4,9] | +0,1 | 6/11 | 93% | +0,1 [−1,0; +1,2], 5/10 |
| q50, 305 | **+2,8 [+0,2; +7,0]** | +0,8 | 8/11 | 73% | +0,8 [−0,1; +1,8], 7/10 |

**C4: phối hợp có trợ giúp GT (T2) so với T1, F1 duy nhất**

| q, B (kbit) | Trung bình [CI 95%] | Trung vị | Số chuỗi dương | Bỏ 59: trung bình [CI 95%] |
|---|---|---|---|---|
| q35, 76 | +0,3 [−0,4; +0,8] | +0,4 | 9/11 | +0,3 [−0,5; +0,9] |
| q35, 153 | **+0,5 [+0,2; +0,9]** | +0,2 | 10/11 | +0,6 [+0,2; +0,9] |
| q35, 305 | **+0,7 [+0,2; +1,2]** | +0,3 | 9/11 | +0,8 [+0,3; +1,4] |
| q50, 76 | **+0,4 [+0,2; +0,5]** | +0,4 | 11/11 | +0,4 [+0,2; +0,5] |
| q50, 153 | **+0,6 [+0,3; +1,0]** | +0,3 | 9/11 | +0,6 [+0,2; +1,0] |
| q50, 305 | **+0,7 [+0,2; +1,2]** | +0,3 | 8/11 | +0,7 [+0,2; +1,3] |

T2v cho kết quả gần như giống hệt T2 (chi tiết trong JSON).

**Thêm: cả khung (F) so với T0, ở mức ngân sách khung (khoảng 1,8 Mbit)**

| Trung bình [CI 95%] | Trung vị | Số chuỗi dương | Bỏ 59 |
|---|---|---|---|
| **+8,3 [+5,3; +11,5]** | +5,7 | 10/11 | +7,2 [+4,6; +9,8] |

Ở mọi hiệu ứng, bỏ bất kỳ một chuỗi nào cũng **không đổi dấu** của trung bình.

## 3. Đọc kết quả

**1. Lợi ích của crop (C3) do một chuỗi chi phối.**
- Chuỗi 59 chiếm 70–99% tổng lợi ích.
- CI 95% chỉ loại trừ 0 ở mức B cao nhất (305 kbit), và chỉ vừa sát: cận dưới +0,1 đến +0,2 điểm.
- Bỏ chuỗi 59 thì **mọi** CI đều chứa 0; trung bình còn +0,0 đến +0,8 điểm, chỉ 4–7/10 chuỗi dương.
- Trung vị theo chuỗi khoảng 0 (từ −0,5 đến +0,9).
- **Kết luận:** chưa có bằng chứng rằng crop giúp một chuỗi "điển hình". Bằng chứng chỉ có ở chuỗi mà detector trên UAV gần như mù.

**2. Phối hợp (C4) nhỏ nhưng ổn định.**
- +0,3 đến +0,7 điểm F1 duy nhất; CI loại trừ 0 ở 5/6 ô; dương ở 8–11/11 chuỗi; không phụ thuộc chuỗi 59.
- Nhưng đây là **heuristic có trợ giúp GT** (liên kết hoàn hảo nhờ ID của GT), không triển khai được và không phải cận trên. Điều duy nhất có thể nói: có một dư địa phối hợp nhỏ và đều ở mức đề xuất. Với liên kết xuyên góc thật, dư địa đó sẽ nhỏ hơn hoặc mất.

**3. Cả khung (F) là hiệu ứng mạnh và vững nhất** (+8,3 điểm, CI [+5,3; +11,5], 10/11 chuỗi, vẫn +7,2 khi bỏ 59).
- Nhưng F chỉ vừa ở ngân sách khoảng 1,8 Mbit mỗi epoch.
- Mã hoá AVIF cả khung mất 1,17 s trung vị trên 1 luồng, vượt chu kỳ 1 s.
- Đây là so sánh ở ngân sách khác, không phải cùng ngân sách.

**4. Kết luận, không đổi ngưỡng:** Gate 0B vẫn là PASS khám phá theo quy tắc. Về độ tin cậy, lợi ích của crop **không vững** khi xét theo chuỗi: nó phụ thuộc vào một chuỗi khó. Không có gì trong phân tích này ủng hộ việc mở HOLD hay train PPO cho lựa chọn TOKEN / CROP.
