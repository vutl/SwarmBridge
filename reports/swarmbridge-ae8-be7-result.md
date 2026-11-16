# A-E8 và B-E7: lập kế hoạch có ngân sách thời gian; codec video liên khung làm đối chứng

Ngày 16/11/2026, theo phản biện #33. Vẫn là giai đoạn khám phá:
- A: 20 seed (8000..8019), 3 model PPO, giao thức `pre`, mode-only, checkpoint PPO giữ nguyên.
- B: 10 clip đánh giá (7 gate + 3 aux_val), cùng frame (stride 3, tối đa 100 frame) và cùng head với B-E6.

Mã nguồn:
- `swarmbridge/experiments/uav_semcom/a9_mpc.py`: `Deadline`, `stacked_ctx`, `selective`, rollout theo nhóm.
- `a10_hybrid.py`: biến thể `ahyb`, cờ `--recall-diag`.
- `swarmbridge/experiments/jepa_semcom/b3_video.py`.
- Test: `test_a9_mpc.py` (21 test), `test_b3.py` (5 test).

## Tóm tắt

- **A-E8.** Bộ điều khiển chọn lọc có ngân sách 50 ms:
  - Mỗi quyết định được ước lượng trước chi phí và có biên dự trữ. PPO luôn sẵn làm phương án dự phòng.
  - Tính cả thời gian PPO, ngưỡng 50 ms được giữ ở **99,97–100 %** quyết định; p99 ≈ 40–44 ms.
  - Return không giảm so với bản không có ngân sách.
- **Sàng lọc có tính độ bất định** (PPO + ứng viên tốt nhất theo tốc độ danh định + ứng viên tốt nhất theo đầu thấp của khoảng nhạy cảm):
  - Có ích khi mô hình tốc độ lệch mạnh. Shannon dưới MCS: +1,90 so với PPO, thay vì +0,71 khi chọn top-2 danh định; hiệu ghép cặp +1,19 [+0,18; +2,21].
  - Recall ứng viên tăng.
  - Ở regime ref, khi mô hình gần như không lệch, nó kém top-2 khoảng 0,26–0,30 (CI chứa 0).
- **Với cùng ngân sách 50 ms**, planner đầy đủ dạng anytime bị cắt ở 66–68 % quyết định và mất một phần lợi ích (ref: +1,90 thay vì +2,23). Bản chọn lọc đạt mức tương đương hoặc tốt hơn với khoảng 60 % thời gian.
- **B-E7.** HEVC / AV1 liên khung, cấu hình độ trễ thấp, nhân quả → DINOv2 → cùng head **tốt hơn codec ngữ nghĩa tốt nhất (`rateonly03`)**:
  - ở 6–12 kbit/frame, hơn 0,02–0,09 mAP;
  - trên **mọi** mức bit đã thử với patch có thay đổi, hơn 0,04–0,09.
  - Chỉ ở khoảng 18 kbit/frame hai bên ngang nhau trên toàn bộ patch.
  - Đây là đối chứng phản biện #33 lo nhất, và **kết quả bất lợi cho Paper B ở dạng hiện tại**.
- **Surrogate tốc độ trong vòng kín:**
  - xếp hạng model gần như hoàn hảo (Spearman 0,98–0,99 trong từng clip);
  - theo dõi tốc độ trung bình theo clip tốt (0,95–0,98);
  - chỉ tương quan vừa phải theo từng frame (0,69–0,80).
  - Dạng surrogate tính trên residual thật (mc, top-k) đạt 0,99.

## Phần A (A-E8)

### 1. Cơ chế

**Ngân sách thời gian (`Deadline`).**
- Đồng hồ quyết định bắt đầu *trước* khi PPO suy luận.
- Trước mỗi đơn vị công việc (dựng tập mẫu tốc độ, đánh giá một mode, đánh giá kiểm tra đầu mút), bộ điều khiển chỉ làm tiếp nếu `đã dùng + số đơn vị × chi phí ước lượng + biên 3 ms ≤ 50 ms`.
- Chi phí mỗi đơn vị (một ảnh được rollout trong một lần đánh giá mode) được học trực tuyến: phân vị 95 % của 300 lần đo gần nhất cùng loại, giá trị ban đầu 1 ms.
- Ước lượng học được: sàng lọc khoảng 0,7–0,8 ms/đơn vị, đánh giá đầy đủ khoảng 1,2 ms/đơn vị, dựng mẫu khoảng 0,3 ms.

**Dự phòng khi hết ngân sách.** PPO luôn có sẵn. Ba trường hợp:
- không đủ thời gian để sàng lọc (`skip_screen`) hoặc sàng lọc bị ngắt giữa chừng (`int_screen`) → dùng mode của PPO;
- không đủ thời gian cho mode của PPO + 1 ứng viên + phần kiểm tra (`skip_full`) → dùng mode của PPO;
- đủ cho một phần ứng viên (`trunc_full`) → chọn trong số đã đánh giá, rồi vẫn kiểm tra đầu mút. Thời gian cho bước kiểm tra luôn được giữ trước.

**Giới hạn.** Đây **không phải bảo đảm thời gian thực cứng**: Python không ngắt được một lần đánh giá đang chạy. Một đơn vị chạy lâu hơn ước lượng, do hệ điều hành lập lịch hoặc tranh chấp CPU, vẫn gây trễ hạn. Số lần trễ hạn được đo và báo cáo.

**Đánh giá gộp (stacking).** Chi phí gần như không phụ thuộc số mẫu, nên bộ điều khiển đặt các mẫu tốc độ ở mức danh định và ở các đầu mút (hoặc ở [1, lo] khi sàng lọc) vào cùng một trục mẫu.
- Rollout nhân quả chọn mode riêng cho từng nhóm, nên mỗi nhóm cho đúng kết quả của một lần đánh giá riêng.
- Có test kiểm chứng điều này. Kiểm tra trên các seed thử cũng cho quyết định và return giống hệt.
- Kiểm tra đầu mút vì vậy gần như không tốn thêm thời gian.

**Sàng lọc có tính độ bất định (`cand=np`).**
- Một lượt sàng lọc gộp với tầm nhìn H1 = 1, ở mức danh định và ở `lo` (đầu thấp của khoảng nhạy cảm).
- Ứng viên: mode tốt nhất theo danh định và mode tốt nhất theo `lo`. Nếu hai mode trùng nhau thì lấy mode danh định thứ hai. Tổng cộng vẫn chỉ 3 mode: PPO + 2 ứng viên.
- Điều kiện chạy planner đầy đủ (`trig`): `any` nếu ít nhất một trong hai lượt sàng lọc cho thấy có mode hơn PPO; `pess` nếu chỉ tính lượt ở `lo`.

**Tái tạo kết quả cũ.**
- `ahyb cand=all` (không ngân sách) cho đúng return từng seed của dhyb A-E6/A-E7.
- `ahyb cand=n2` cho đúng return của shyb A-E7.

### 2. Return và thời gian

Gộp 3 model PPO, 60 cặp (model × seed) so với PPO cùng model. Thời gian tính cả PPO, đo với 6 tiến trình trên 6 nhân.

**phy_mcs_blind** (PPO: return 30,27, 1,5 ms)

| Bộ điều khiển | Mô hình tốc độ | Hơn PPO | ms TB | p95 | p99 | Lớn nhất | Vượt 50 ms | MPC đầy đủ | Dự phòng / cắt |
|---|---|---|---|---|---|---|---|---|---|
| dhyb (đủ 14 mode), không ngân sách | cals | +2,82 [+1,92; +3,71] | 50,5 | 75 | 90 | – | 48 % | 100 % | – |
| shyb (top-2), không ngân sách | cals | +2,39 [+1,33; +3,44] | 24,6 | 45 | 55 | – | 2,2 % | 76 % | – |
| **shyb, 50 ms, gộp** | cals | +2,39 (giống hệt) | **22,7** | 38 | 40 | 47,5 | **0** | 76 % | 0,0 % |
| **np-any, 50 ms** | cals | **+2,86** [+2,01; +3,72] | 24,5 | 40 | 43 | 47,5 | 0 | 79 % | 0,1 % |
| np-pess, 50 ms | cals | +2,59 [+1,62; +3,55] | 23,5 | 40 | 43 | 47,3 | 0 | 71 % | 0,1 % |
| dhyb anytime, 50 ms | cals | +2,70 [+1,84; +3,57] | 40,1 | 46 | 47 | 48,9 | 0 | 100 % | cắt 66 % |
| np-any, **25 ms** | cals | +0,06 [−0,35; +0,47] | 16,3 | 22 | 22 | 23,9 | 0 | 66 % | 78 % |
| dhyb, không ngân sách | Shannon | +1,30 [+0,28; +2,33] | 51,6 | 77 | 87 | – | 52 % | 100 % | – |
| shyb, không ngân sách | Shannon | +1,03 [−0,03; +2,10] | 29,8 | 47 | 57 | – | 3,1 % | 95 % | – |
| shyb, 50 ms | Shannon | +0,71 [−0,29; +1,71] | 25,3 | 39 | 42 | 50,2 | 0,01 % | 95 % | 0,6 % |
| np-any, 50 ms | Shannon | +1,69 [+0,76; +2,62] | 27,3 | 42 | 44 | 53,4 | 0,03 % | 97 % | 1,1 % |
| **np-pess, 50 ms** | Shannon | **+1,90** [+0,92; +2,89] | 24,5 | 41 | 43 | 54,0 | 0,02 % | 73 % | 0,6 % |
| np-any, 25 ms | Shannon | −0,24 [−0,80; +0,33] | 16,8 | 22 | 22 | 32,9 | 0,01 % | 78 % | 91 % |

**ref** (PPO: return 58,33, 1,5 ms)

| Bộ điều khiển | Mô hình | Hơn PPO | ms TB | p99 | Lớn nhất | Vượt 50 ms | MPC đầy đủ |
|---|---|---|---|---|---|---|---|
| dhyb, không ngân sách | cals | +2,23 [+1,79; +2,66] | 49,4 | 85 | – | 45,5 % | 100 % |
| **shyb, 50 ms, gộp** | cals | **+2,22** [+1,71; +2,72] | **19,7** | 40 | 45,4 | 0 | 47 % |
| np-any, 50 ms | cals | +1,96 [+1,45; +2,46] | 21,5 | 42 | 46,2 | 0 | 48 % |
| np-pess, 50 ms | cals | +1,91 [+1,44; +2,39] | 21,2 | 42 | 103,2* | 0,01 % | 46 % |
| dhyb anytime, 50 ms | cals | +1,90 [+1,46; +2,33] | 40,3 | 47 | 48,5 | 0 | 100 % (cắt 68 %) |
| np-any, 25 ms | cals | +0,28 [+0,04; +0,52] | 16,4 | 22 | 23,9 | 0 | 39 % |
| dhyb, không ngân sách | Shannon | +2,42 [+1,90; +2,94] | 48,8 | 83 | – | 43,7 % | 100 % |
| shyb, 50 ms | Shannon | +2,18 [+1,66; +2,70] | 20,1 | 40 | 45,2 | 0 | 52 % |
| np-any, 50 ms | Shannon | +2,33 [+1,82; +2,83] | 22,3 | 42 | 47,0 | 0 | 56 % |
| np-pess, 50 ms | Shannon | +2,06 [+1,52; +2,60] | 21,1 | 42 | 44,8 | 0 | 46 % |

\* Một quyết định duy nhất trong 9 360, có khả năng do hệ điều hành lập lịch: đơn vị công việc đang chạy không ngắt được.

**Hiệu ghép cặp** (60 cặp):

| So sánh | phy cals | phy Shannon | ref cals | ref Shannon |
|---|---|---|---|---|
| np-any − shyb (50 ms) | +0,48 [−0,34; +1,29] | +0,98 [−0,11; +2,07] | −0,26 [−0,66; +0,14] | +0,15 [−0,25; +0,54] |
| np-pess − shyb (50 ms) | +0,20 [−0,79; +1,19] | **+1,19 [+0,18; +2,21]** | −0,30 [−0,70; +0,09] | −0,12 [−0,60; +0,36] |
| np-any − dhyb (không ngân sách) | +0,05 [−0,91; +1,00] | +0,39 [−0,40; +1,18] | −0,27 [−0,66; +0,12] | −0,09 [−0,57; +0,39] |
| np-any 50 ms − dhyb anytime 50 ms | +0,16 [−0,76; +1,08] | – | +0,06 [−0,38; +0,50] | – |

**Nhận xét**

1. **Ngân sách 50 ms giữ được trong thực tế**, tính cả thời gian PPO:
   - bản `cals` không trễ hạn lần nào;
   - bản Shannon trễ hạn ở 0,01–0,03 % quyết định (lớn nhất 54 ms);
   - ở ref có một ngoại lệ 103 ms.
   p99 giảm từ 55 ms (shyb A-E7) xuống 40–44 ms; ở bản `cals` return không đổi.
2. **Planner đầy đủ không vừa ngân sách 50 ms.** Bản anytime bị cắt ở hai phần ba số quyết định và mất 0,1 (phy) đến 0,3 (ref) return. Bản chọn lọc dùng khoảng 60 % thời gian của nó mà return không thấp hơn.
3. **Sàng lọc có tính độ bất định có ích đúng ở chỗ phản biện dự đoán:** mô hình lệch mạnh (Shannon dưới MCS, lệch khoảng −46 %).
   - np-pess +1,90, cao hơn dhyb đầy đủ (+1,30), dù CI của hiệu vẫn chứa 0 (+0,60 [−0,35; +1,55]). Trigger `pess` còn giảm tỉ lệ chạy MPC đầy đủ từ 95 % xuống 73 %.
   - Khi mô hình gần đúng (`cals`, ref) thì không có ích, thậm chí hơi kém. Đầu thấp của khoảng có thể quá bi quan và đẩy ra một ứng viên kém hơn top-2 danh định.
   - Dù vậy, `np-any` với `cals` ở phy đạt +2,86, bằng dhyb đầy đủ với một nửa thời gian.
4. **Ở 25 ms, cơ chế an toàn nhưng gần như mất hết lợi ích**: 52–63 % quyết định bỏ qua MPC đầy đủ. Với kích thước bài toán hiện tại, planner cần khoảng 25–35 ms mới có giá trị.
5. **Mức nhiễu.** Shannon shyb có ngân sách chỉ khác bản không ngân sách ở 0,6 % quyết định (bị cắt hoặc bỏ qua), nhưng return lệch −0,32. Lệch nhỏ trong quyết định làm luồng ngẫu nhiên của các bước sau phân kỳ. Vì vậy chênh lệch dưới khoảng 0,3–0,5 giữa các biến thể nằm trong mức nhiễu của 20 seed.

### 3. Recall ứng viên (model PPO 0, toàn bộ 20 seed, không ngân sách, gộp)

Mỗi khi planner có thể chạy, một planner tham chiếu cũng ra quyết định. Planner này xét đủ 14 mode, dùng cùng mô hình, cùng calibrator, cùng kiểm tra đầu mút, không có ngân sách, nhưng **dùng mẫu độc lập**.
- "Recall": tỉ lệ mode đích của planner tham chiếu (khi nó đổi khỏi PPO) nằm trong tập mode bộ điều khiển đã đánh giá.
- Dòng dhyb cho **mức sàn do nhiễu mẫu**: chính dhyb chỉ đồng ý với một bản chạy lại với mẫu khác ở 81 % (phy) / 88 % (ref) quyết định.

| | Planner đầy đủ đổi mode | Recall ứng viên | Trigger bị bỏ lỡ | Đồng ý | Lần đổi của bộ điều khiển trùng planner đầy đủ |
|---|---|---|---|---|---|
| phy cals dhyb (sàn) | 24,7 % | 1 | 0 | 80,9 % | 58 % |
| phy cals top-2 | 23,6 % | 0,822 | 6,8 % | 82,6 % | 63 % |
| phy cals np-any | 24,2 % | **0,879** | 4,4 % | 82,9 % | 64 % |
| phy cals np-pess | 23,3 % | 0,852 | 6,2 % | 83,1 % | 62 % |
| phy Shannon dhyb (sàn) | 25,4 % | 1 | 0 | 80,8 % | 62 % |
| phy Shannon top-2 | 25,3 % | 0,957 | 0,5 % | 82,1 % | 64 % |
| phy Shannon np-any | 24,7 % | **0,988** | 0 | 81,9 % | 63 % |
| phy Shannon np-pess | 24,1 % | 0,945 | 4,7 % | 81,8 % | 61 % |
| ref cals dhyb (sàn) | 19,2 % | 1 | 0 | 87,5 % | 66 % |
| ref cals top-2 | 20,5 % | 0,869 | 10,8 % | 87,1 % | 70 % |
| ref cals np-any | 18,7 % | 0,894 | 8,9 % | 87,5 % | 66 % |
| ref Shannon np-any | 21,5 % | 0,987 | 0,3 % | 91,6 % | 81 % |

- Ứng viên từ đầu thấp tăng recall ở mọi cấu hình: 0,82 → 0,88 (phy cals), 0,957 → 0,988 (phy Shannon), 0,869 → 0,894 (ref cals).
- Phần recall còn mất chủ yếu do **trigger** (sàng lọc tầm nhìn 1 bước không thấy lợi ích), không do bỏ sót ứng viên.
- Mức đồng ý của mọi bộ điều khiển chọn lọc với planner đầy đủ (82–92 %) bằng mức planner đầy đủ đồng ý với chính nó. Vậy sàng lọc gần như không làm mất thông tin so với nhiễu mẫu sẵn có.

## Phần B (B-E7)

### 1. Thiết lập

**Frame.** Đúng các frame của các token đã trích. Mã số frame được kiểm tra khớp với file token. Token tính lại từ frame gốc khớp với token đã lưu (sai khác score tối đa 4·10⁻⁶).

**Encode.** Mỗi clip là một luồng video nhân quả, độ trễ thấp: frame đầu intra, sau đó chỉ P/inter frame.
- **HEVC (libx265 (so 215, gói Ubuntu 26.04), preset slow, tune zerolatency):** keyint vô hạn, bframes 0, rc-lookahead 0, scenecut 0, tắt SEI phiên bản; luồng Annex-B thô. Điểm CRF 22..51.
- **AV1 (SVT-AV1 2.3):** pred-struct 1 (low delay), lookahead 0, scd 0; luồng OBU thô. Lưu ý: ở low delay, SVT-AV1 **tự đổi sang preset 7 và QP cố định (CQP)** thay cho CRF, theo log của bộ mã hóa. Điểm QP 30..63.
- Test kiểm chứng: thay đổi các frame sau t₀ thì các frame đã giải mã ≤ t₀ **và packet của chúng giữ nguyên từng bit**, với cả hai codec. Vậy bộ thu không dùng frame tương lai.

**Bit.** Toàn bộ file bitstream (header, keyframe, inter frame, overhead NAL/OBU) chia cho số frame. Giống phía ngữ nghĩa, nơi tính cả bản tin khởi tạo.

**Bộ thu.** Giải mã → DINOv2-S (fp16, cùng chuẩn hóa) → cùng head tuyến tính → AP, ap_change, ap_event như B-E6.

**So sánh.** Nội suy theo log(bit thật) trong từng clip. Codec ngữ nghĩa lấy trung bình 3 seed huấn luyện; chỉ số từ `b6_eval.json` (dự đoán mc, k = 16..256). Khoảng t trên 10 clip.

### 2. Điểm vận hành của codec (trung bình 10 clip)

| | kbit/frame | Frame đầu (kbit) | PSNR | AP | ap_change |
|---|---|---|---|---|---|
| HEVC CRF 32 | 20,6 | 140 | 30,3 | 0,449 | 0,448 |
| HEVC CRF 37 | 9,1 | 77 | 27,7 | 0,412 | 0,417 |
| HEVC CRF 42 | 3,9 | 39 | 25,3 | 0,339 | 0,359 |
| AV1 QP 50 | 14,7 | 142 | 29,7 | 0,437 | 0,433 |
| AV1 QP 55 | 9,6 | 101 | 28,3 | 0,409 | 0,405 |
| AV1 QP 60 | 6,2 | 69 | 27,1 | 0,394 | 0,392 |
| Token gốc (không nén) | – | – | – | 0,475 | 0,472 |
| Codec ngữ nghĩa, không truyền (trần) | – | 124 (khởi tạo) | – | 0,486–0,489 | – |

Encode tốn khoảng 1–2 s cho 100 frame (preset chậm, đa luồng; không phải encoder thời gian thực cho UAV).

### 3. mAP ở cùng số bit thật (mc, trung bình 10 clip)

| kbit/frame | HEVC | AV1 | task | smooth | **rateonly03** | AVIF intra |
|---|---|---|---|---|---|---|
| 6 | 0,371 | 0,383 | 0,285 | 0,284 | 0,306 | – |
| 9 | 0,405 | 0,413 | 0,334 | 0,343 | 0,366 | – |
| 12 | 0,422 | 0,428 | 0,366 | 0,382 | 0,402 | – |
| 18 | 0,440 | 0,445 | 0,413 | 0,437 | 0,449 (9 clip) | 0,367 (3 clip) |
| 25 | 0,450 | 0,451 | 0,450 | 0,433 | 0,462 (3 clip) | 0,339 |

**rateonly03 trừ codec video** (khoảng t trên các clip):

| Chỉ số | Mức bit | − HEVC | − AV1 |
|---|---|---|---|
| AP | 6 k | −0,065 [−0,109; −0,022] | −0,085 [−0,140; −0,030] |
| AP | 9 k | −0,039 [−0,070; −0,007] | −0,051 [−0,084; −0,018] |
| AP | 12 k | −0,020 [−0,044; +0,005] | −0,026 [−0,048; −0,004] |
| AP | 18 k | +0,005 [−0,020; +0,030] | −0,001 [−0,026; +0,023] |
| ap_change | 6 k | −0,090 [−0,112; −0,068] | −0,086 [−0,127; −0,045] |
| ap_change | 12 k | −0,071 [−0,092; −0,051] | −0,071 [−0,100; −0,042] |
| ap_change | 18 k | −0,048 [−0,061; −0,035] | −0,047 [−0,066; −0,028] |
| ap_event | 9 k | −0,065 [−0,098; −0,032] | −0,075 [−0,114; −0,036] |
| ap_event | 18 k | −0,020 [−0,044; +0,004] | −0,027 [−0,049; −0,005] |

**Trạng thái ổn định** (bỏ frame đầu ở cả hai phía: keyframe của video, bản tin khởi tạo 124 kbit của codec ngữ nghĩa; `b7_eval_steady.json`): kết luận không đổi.
- AP: −0,039 / −0,060 ở 6 k; −0,010 / −0,018 ở 12 k (CI chứa 0); +0,008 / +0,002 ở 18 k.
- ap_change: −0,04 đến −0,08 ở mọi mức, CI < 0.

**Nhận xét**

1. Ở mức bit thấp đến trung bình (≤ 12 kbit/frame), một codec video tiêu chuẩn, nhân quả, rồi trích DINOv2 tại bộ thu, **tốt hơn mọi biến thể codec ngữ nghĩa của chúng ta**.
2. Trên patch có vật xuất hiện, biến mất hoặc di chuyển, video tốt hơn ở **mọi** mức bit đã thử. Đây đúng là chỗ B-E6 từng thấy rateonly03 hơn smooth: rateonly03 có tốt hơn các biến thể ngữ nghĩa khác, nhưng vẫn kém video xa.
3. Codec ngữ nghĩa chỉ ngang video quanh 18 kbit/frame, và chỉ trên toàn bộ patch. Trần của nó cao hơn token gốc (0,489 so với 0,475, nhờ huấn luyện với loss nhiệm vụ). Lợi thế ở mức bit cao có thể có nhưng chưa đo được: dải k hiện tại chỉ phủ 3/10 clip ở 25 kbit.
4. Giải thích hợp lý, chưa kiểm chứng: codec video có bù chuyển động dưới pixel, mã hóa entropy thích nghi ngữ cảnh và truyền *mọi* vùng ở chất lượng thấp. Codec ngữ nghĩa chỉ cập nhật top-k patch trên lưới 22 × 39, với bù chuyển động nguyên patch và mã exp-Golomb không có mô hình ngữ cảnh. Các patch thay đổi nhưng không lọt top-k bị giữ nguyên giá trị cũ.
5. Về vận hành, phương án video còn đơn giản hơn: UAV chỉ cần encoder video, DINOv2 chạy ở trạm mặt đất.

**Hệ quả cho Paper B.** Câu "codec latent tiết kiệm bit hơn" không đứng được trước video liên khung ở dải bit chính. Phản biện #33 đã nêu khả năng này. Hướng có thể đi, cần phản biện chọn:
- (a) Đưa codec ngữ nghĩa thành **lớp tăng cường trên nền video**: truyền hiệu chỉnh token chỉ ở các patch mà DINOv2 trên video giải mã sai, nhất là patch có thay đổi.
- (b) Giữ khung "task-aware, rate-aware" nhưng áp vào **chính codec video**: bản đồ QP theo nhiệm vụ / thay đổi, chọn QP theo đường cong tốc độ–tiện ích nhiệm vụ.
- (c) Thay bộ mã hóa entropy và bù chuyển động của codec ngữ nghĩa (hyperprior học được, bù chuyển động dưới patch). Việc này lớn hơn và không chắc đuổi kịp.

Đường cong tốc độ–mAP của HEVC/AV1 trên ảnh UAV tuần tự đo được ở đây cũng là dữ liệu thật đầu tiên cho giao diện tốc độ–chất lượng chung giữa A và B, đúng điều kiện "cùng nhiệm vụ, chuỗi frame" mà phản biện đặt ra.

### 4. Surrogate tốc độ trong vòng kín (`b7_surrdiag.json`)

Truyền vòng kín với `b1_pipeline.transmit`, dự đoán mc. Gồm 4 biểu diễn (task, smooth, rateb03, rateonly03) × 3 seed × 10 clip, ở k = 64 và 128. Với mỗi frame t ≥ 1 có hai dạng surrogate:
- **(a) Dạng huấn luyện:** trung bình log₂(1 + |z_t − z_{t−1}|/0,25) trên mọi token, với lịch sử thật.
- **(b) Dạng vòng kín:** cùng công thức, nhưng tính trên residual thật được truyền (top-k token, dự đoán mc từ trạng thái bộ thu).

| | k = 64 | k = 128 |
|---|---|---|
| Pearson theo frame trong clip, dạng (a) | 0,69 (p10–p90 0,54–0,85) | 0,80 (0,67–0,93) |
| Spearman theo frame, dạng (a) | 0,69 | 0,79 |
| Pearson theo frame, chỉ frame sự kiện, dạng (a) | 0,68 | 0,80 |
| Pearson theo frame, dạng (b) | 0,992 | 0,995 |
| Tương quan trung bình theo clip (120 chuỗi model × clip), dạng (a) | 0,95 | 0,98 |
| Xếp hạng 12 model trong từng clip, dạng (a) (Spearman) | 0,984 | 0,987 |

- Surrogate huấn luyện **xếp đúng model nào rẻ hơn** và theo dõi tốc độ trung bình theo clip, nhưng chỉ giải thích vừa phải biến động từng frame.
- Phần chênh đến từ việc chọn top-k và bù chuyển động. Bằng chứng: cùng công thức tính trên residual vòng kín đạt 0,99.
- Vậy surrogate dựa trên residual MC (đề xuất cũ) sẽ khớp chặt hơn nhiều. Tuy vậy, với kết quả mục 3, cải tiến này khó đủ để đuổi kịp video.

## Hạn chế

- A: 20 seed khám phá. Ngân sách là cơ chế hợp tác trong Python, không phải thời gian thực cứng. Thời gian đo với 6 tiến trình trên 6 nhân của một máy. Chênh lệch dưới khoảng 0,3–0,5 return nằm trong mức nhiễu. Chưa có ACK trễ hoặc mất.
- B:
  - 10 clip, tối đa 100 frame mỗi clip, nên chi phí keyframe được chia cho ít frame (đã kèm phân tích trạng thái ổn định).
  - Encoder dùng preset chậm và CRF/QP cố định, không phải cấu hình thời gian thực có kiểm soát tốc độ.
  - Ở low delay, AV1 thành CQP preset 7.
  - Head tuyến tính học trên token gốc, không học lại cho token từ video giải mã. Điều này nếu có thì bất lợi cho video.
  - Codec ngữ nghĩa không có điểm k > 256, nên mức bit cao phủ kém.

## Kết quả lưu

- `runs/uav_semcom/a14/ae8_{phy_mcs_blind,ref}.json`, `ae8_recall_{phy_mcs_blind,ref}.json`
- `runs/jepa_semcom/b7/extract_stats.json` (bit, PSNR, packet theo clip và điểm), `b7_eval.json`, `b7_eval_steady.json`, `b7_surrdiag.json`
- Score theo frame và bitstream trung gian chỉ nằm trong WSL (`~/work/jepa_semcom/b7_video`), không đưa vào git.
