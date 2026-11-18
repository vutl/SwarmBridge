# A8 và B1: kết quả và báo cáo quyết định

Ngày 13/11/2026, theo phản biện #18. Đăng ký trước: [swarmbridge-a8-b1-preregistration.md](swarmbridge-a8-b1-preregistration.md) (commit b5389600, trước khi chạy). Không train; không ảnh MDMT mới.

## 1. Tóm tắt

- **A8 (thang sai lệch mô hình): KHÔNG HỢP LỆ.** Thang tôi thiết kế không tác động lên hai chính sách được so. Cả 5 mức cho cùng một kết quả, trùng nhau từng episode. Quy tắc đã đăng ký, áp máy móc, ra "GO"; tôi **không** tuyên bố GO, vì 5 mức thực chất là một thí nghiệm lặp lại 5 lần.
- **Điều A8 vẫn cho biết:** lợi thế +5,73 của PPO mode-only có nguyên vẹn khi bộ thích ứng liên kết biết SINR thật. Vậy lợi thế này **không** do "PHY không quan sát được nhiễu". Câu diễn giải đó trong bản Research Decision Review là sai và đã được ghi chú sửa.
- **B1 (cơ hội hợp tác theo góc nhìn):** điều kiện 1 (ID xuyên góc) đạt ở đúng mức tối thiểu, 8/11 chuỗi. Tầng 1 **chưa chạy** vì chưa có cơ sở dùng dữ liệu MDMT. Code và 7 test tổng hợp đã sẵn.
- **Đề xuất:** không cấp thêm ngân sách cho hướng A như hướng chính. Hướng B là hướng duy nhất còn câu hỏi mở đúng mục tiêu, nhưng bước tiếp theo của nó là một việc không tính toán: có được xác nhận quyền dùng MDMT.

## 2. A8

### 2.1 Đã chạy gì

Đúng như đăng ký:
- 5 mức: `phy_mcs`, `phy_mcs_l75`, `phy_mcs_l50`, `phy_mcs_l25`, `phy_mcs_blind`.
- 3 mô hình PPO mode-only train ở `phy_mcs_blind`, đóng băng, luật slot `pacedB2+budget1`.
- H\* điều chỉnh lại cho từng mức trên pool train, seed 0..29.
- Seed môi trường 8000..8099; pool test (chính) và test-dev; bootstrap theo cụm video B = 50 × 20 seed.

### 2.2 Kết quả

Δ = PPO (trung bình 3 mô hình) − H\*. CI mức 3 = cụm video + lấy mẫu lại mô hình.

| Mức sai lệch | H\* | H\* return | PPO return | Δ test [CI mức 3] | Δ test-dev [CI mức 3] |
|---|---|---|---|---|---|
| 0 (`phy_mcs`) | sc:pacedB3+budget1.5 | 25,65 | 31,38 | +5,73 [+3,68; +8,32] | +7,81 [+2,70; +10,76] |
| 0,25 | như trên | 25,65 | 31,38 | +5,73 [+3,68; +8,32] | +7,81 [+2,70; +10,76] |
| 0,5 | như trên | 25,65 | 31,38 | +5,73 [+3,68; +8,32] | +7,81 [+2,70; +10,76] |
| 0,75 | như trên | 25,65 | 31,38 | +5,73 [+3,68; +8,32] | +7,81 [+2,70; +10,76] |
| 1 (`phy_mcs_blind`) | như trên | 25,65 | 31,38 | +5,73 [+3,68; +8,32] | +7,81 [+2,70; +10,76] |

- Kiểm tra tái lập đạt: tại `phy_mcs_blind`, Δ khớp A5.5 (+5,73 và +7,81).
- Return từng episode của PPO và H\* thay đổi tối đa 7·10⁻¹⁵ dọc thang (sai số dấu phẩy động).

### 2.3 Vì sao thang vô hiệu

- Tham số λ chỉ đổi cách bộ thích ứng liên kết nhìn **nhiễu đồng kênh**.
- Trong môi trường A5, nhiễu đồng kênh chỉ có khi hai UAV phát trên cùng một kênh (`env.py`, dòng 420).
- Luật slot của PPO mode-only (`pacedB2+budget1`) và của H\* (`sc:pacedB3+budget1.5`) cấp mỗi kênh cho một UAV. Không có dùng chung kênh, nên nhiễu luôn bằng 0 và λ không có gì để tác động.
- Với các heuristic **có** dùng chung kênh, thang hoạt động đúng. Cùng một thành viên `pacedB2+budget1+reuse`, return train theo mức: +16,0 / +2,2 / −23,9 / −29,0 / −31,2. Bản không reuse: +14,98 ở cả 5 mức.

Đây là lỗi thiết kế của tôi, và dấu hiệu đã có sẵn: A5.5 mục 2 ghi T:SC = +16,55 ở cả `phy_mcs` lẫn `phy_mcs_blind`. Bản đăng ký thiếu một bước kiểm tra thao tác (xác nhận return của các chính sách được so có đổi theo mức) trước khi chạy.

### 2.4 Phán quyết

- **Quy tắc áp máy móc:** Δ không giảm (vì phẳng) và CI mức 3 loại 0 ở 5/5 mức → "GO".
- **Phán quyết tôi báo cáo: KHÔNG HỢP LỆ.** Lý do:
  - Tiền đề của quy tắc (các mức khác nhau) không đúng; 5 mức không phải 5 bằng chứng.
  - Giả thuyết đã đăng ký còn dự đoán Δ về gần 0 khi không có sai lệch. Ở mức 0 của thang này Δ vẫn là +5,73.
- Tôi đã thêm bước kiểm tra thao tác vào `scripts/uav_semcom/a8_summarize.py` **sau** khi xem kết quả. Bước này chỉ có thể rút lại một GO, không tạo ra GO. Kết quả máy móc vẫn được lưu ở trường `rule_output`.

### 2.5 Điều rút ra được và không rút ra được

**Rút ra được:**
- Lợi thế của PPO mode-only trong họ MCS không phụ thuộc việc bộ thích ứng liên kết có thấy nhiễu hay không. Tên chế độ `phy_mcs_blind` gây hiểu lầm cho phép so này: với các chính sách không dùng chung kênh, nó trùng với `phy_mcs`.
- Ở `phy_mcs` trên test, heuristic tốt nhất *chọn sau khi xem test* là T:best có reuse (27,40). So với nó PPO còn +3,98. Đây là phép so có lợi cho heuristic.

**Không rút ra được:**
- Lợi thế có tăng theo sai lệch mô hình hay không. Câu hỏi này chưa được kiểm.
- Sai lệch thật sự còn lại giữa heuristic và môi trường ở họ MCS là: heuristic dự đoán tốc độ bằng công thức Shannon, còn PHY thật là MCS bậc thang với BLER 10% và ngưỡng cắt. Chưa có thang nào cho chiều này.

## 3. B1

### 3.1 Điều kiện 1: ID của GT có thống nhất giữa hai UAV không? (đã làm, chỉ dùng nhãn)

- **Tài liệu:** README của MDMT không nêu quy ước ID xuyên góc. Vì vậy phải kiểm bằng dữ liệu.
- **Kiểm hình học và lớp** (`scripts/uav_semcom/b1_id_check.py`, kết quả ở `runs/uav_semcom/h1/b1_id_check_v1.json`):

| Chuỗi | Cùng lớp | Inlier trung vị (ID không dùng để fit) | Bản tráo ID | Đạt |
|---|---|---|---|---|
| 56 | 1,000 | 0,90 | 0,00 | có |
| 34 | 0,997 | 1,00 | 0,00 | có |
| 52 | 0,998 | 0,95 | 0,00 | có |
| 68 | 0,947 | 0,94 | 0,03 | **không** (cùng lớp < 0,95) |
| 48 | 0,979 | 0,50 | 0,00 | **không** (inlier < 0,70) |
| 62 | 1,000 | 0,88 | 0,00 | có |
| 26 | 1,000 | 0,80 | 0,00 | có |
| 57 | 1,000 | 1,00 | 0,00 | có |
| 59 | 0,965 | 0,62 | 0,00 | **không** (inlier < 0,70) |
| 31 | 1,000 | 0,95 | 0,01 | có |
| 61 | 0,983 | 0,94 | 0,00 | có |

- ID chung khớp hình học ở mọi chuỗi (0,50–1,00 so với 0,00–0,03 khi tráo), nên ID thật sự chỉ cùng một vật ở hai góc.
- Theo ngưỡng đã đăng ký, **8/11 chuỗi đạt**, đúng mức tối thiểu. Không đổi ngưỡng. B1 sẽ chỉ chấm 8 chuỗi này; ngưỡng "dương ở ≥ 73% số chuỗi" thành ≥ 6/8.
- Chuỗi 59 bị loại. Đây là chuỗi đóng góp 70–99% lợi ích crop trong pilot 0B. (Sửa theo phản biện #19: bản trước viết rằng việc loại này làm phép chấm khắt khe hơn; điều đó chưa được chứng minh và đã rút lại.)
- Đây là **sàng lọc** tính nhất quán ID, không chứng minh mọi ID trong 8 chuỗi đều đúng. Kết quả B1 sau này chỉ áp dụng cho 8 chuỗi đạt; 3 chuỗi không đạt vẫn được báo trong bảng trên.
- Phản biện #19 phát hiện code tầng 1 chưa cưỡng chế cùng ngân sách bit. Đã sửa: xem Sửa đổi 1 trong bản đăng ký và [swarmbridge-b-feasibility-datasets.md](swarmbridge-b-feasibility-datasets.md).

### 3.2 Điều kiện 2 và tầng 1

- Cơ sở dùng dữ liệu MDMT **chưa được xác nhận** bởi tác giả hoặc lab / trường. Tầng 1 **không chạy**.
- Đã có: `swarmbridge/experiments/uav_semcom/h1_b1_view.py` và 7 test tổng hợp (không dùng bản ghi dataset nào). Script từ chối chạy nếu thiếu tham số `--data-use-basis`.
- Khi có xác nhận, tầng 1 chạy offline trên bản ghi pilot đã có, trong vài phút.

## 4. Quyết định đề xuất

### Hướng A

- **Bằng chứng:** không đổi về kích thước (+5,73 test, +7,81 test-dev, CI mức 3 loại 0), nhưng **mất lời giải thích**. Cơ chế "heuristic thua vì mù nhiễu" đã bị loại.
- **Các dữ kiện làm yếu thêm** (đã có từ trước):
  - Heuristic tự hiệu chỉnh tốc độ (T:SC) không hơn bản không hiệu chỉnh trên test (25,65 so với 26,49), nên sai lệch kiểu "tốc độ thật = hệ số × Shannon" không giải thích được khoảng cách.
  - Trên test-dev của A5, một mode cố định chọn sau khi xem kết quả thắng PPO 2,6.
- **Vì sao một thang khác cũng khó kết luận:** PPO chỉ được train ở một đầu thang. Với PPO đóng băng, "sai lệch tăng" và "gần chế độ đã train hơn" luôn đi cùng chiều, nên một xu hướng tăng không tách được hai cách giải thích. Phép thử dứt khoát cần train PPO theo từng mức, tức cần ngân sách train.
- **Đề xuất: không cấp thêm ngân sách cho A như hướng chính.** Giữ kết quả A5.5 như một kết quả phụ về phương pháp đánh giá.
- Nếu người phản biện vẫn muốn một lần thử nữa, phương án là thang giữa PHY Shannon và PHY MCS (tốc độ = (1 − μ)·Shannon + μ·MCS), với bước kiểm tra thao tác trên seed train **trước** khi đăng ký, và chấp nhận trước rằng kết quả với PPO đóng băng chỉ có tính gợi ý. Tôi không chạy phương án này: nó là một thí nghiệm mới, cần đăng ký và duyệt riêng.

### Hướng B

- **Bằng chứng:** chưa có bằng chứng về lợi ích. Chỉ có: (i) ID xuyên góc dùng được ở 8/11 chuỗi; (ii) A7-S2 cho thấy giá trị của FRAME là thuộc tính của từng góc nhìn.
- **Cơ hội khoa học:** còn mở, và là hướng duy nhất có hợp tác thật giữa các UAV.
- **Điểm nghẽn:** quyền dùng MDMT. Không có bộ dữ liệu hai góc có ID chung nào khác đã được xác định là dùng được.
- **Đề xuất:** bước tiếp theo đáng làm duy nhất là có xác nhận quyền dùng dữ liệu; sau đó chạy tầng 1 (vài phút, không train). Nếu không thể có xác nhận, hướng B không thực hiện được trên MDMT, và khi đó cả hai hướng đều không có bước tiếp theo đủ căn cứ.

## 5. Tệp

- Kết quả A8: `runs/uav_semcom/a8/` (eval, boot, `a8_summary_v1.json`); điều chỉnh heuristic cho 3 mức mới: `runs/uav_semcom/a5/tune/phy_mcs_l*.json`, `runs/uav_semcom/a55/tune/phy_mcs_l*.json`.
- Script: `scripts/uav_semcom/run_a8.sh`, `scripts/uav_semcom/a8_summarize.py`.
- B1: `scripts/uav_semcom/b1_id_check.py`, `runs/uav_semcom/h1/b1_id_check_v1.json`, `swarmbridge/experiments/uav_semcom/h1_b1_view.py`, `swarmbridge/experiments/uav_semcom/test_h1_b1.py`.
