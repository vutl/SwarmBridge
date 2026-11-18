# SwarmBridge: Research Decision Review (13/11/2026)

Theo phản biện #17. Bản này **không có thí nghiệm mới**: chỉ tổng hợp từ các báo cáo đã có (A5, A5.5, A6.0-L1 và audit, G-L2, R1a, R1b-0a, Gate 0A và audit, Pilot 0B, A7-S0 / S1 / S2, hai bản novelty audit). Số liệu lấy nguyên từ các báo cáo đó; không kết quả cũ nào bị sửa.

> **Đính chính (13/11/2026, sau A8):** cụm từ "PHY không quan sát được nhiễu" dưới đây và ở mục 6 là diễn giải sai. Với các chính sách không dùng chung kênh (PPO mode-only và H\*), `phy_mcs_blind` trùng với `phy_mcs`: lợi thế +5,73 có nguyên vẹn khi bộ thích ứng liên kết biết SINR thật. Số liệu không đổi; chỉ lời giải thích bị rút lại. Xem [swarmbridge-a8-b1-result-decision.md](swarmbridge-a8-b1-result-decision.md).

## 1. Tóm tắt một trang

- **Bằng chứng mạnh nhất hiện có:** PPO mode-only trên nền A5 thắng heuristic đã điều chỉnh tốt nhất khi **PHY không quan sát được nhiễu** (phy_mcs_blind: +5,7 return trên test, +7,8 trên test-dev; khoảng tin cậy mức 3 đều loại trừ 0) và khi **nghẽn kèm lệch phân phối** (K5 test-dev: +12,4). Ở chế độ ref và K5 test, không có lợi thế.
- **Mọi lợi ích "semantic" và "hợp tác" đo được đến nay** hoặc nhỏ, hoặc chỉ tồn tại với thông tin lấy từ GT.
- **Môi trường A5 không có hợp tác giữa các UAV.** Các UAV chỉ chia sẻ tài nguyên vô tuyến; một bộ điều khiển PPO tập trung điều khiển tất cả.
- **Đề xuất hai hướng** (mục 6), mỗi hướng có một phép kiểm tra GO / NO-GO rẻ. Hướng A dựa trên bằng chứng đã có nhưng không phải "swarm". Hướng B đúng mục tiêu ban đầu nhưng mọi bằng chứng hiện tại đều dựa trên GT.

## 2. Kết quả nào có bằng chứng thực nghiệm mạnh?

"Mạnh" ở đây nghĩa là: đăng ký trước, seed mới, và khoảng tin cậy có tính bất định theo video / chuỗi.

| Kết quả | Bằng chứng | Mức |
|---|---|---|
| PPO mode-only > heuristic tốt nhất (H\*) ở phy_mcs_blind | A5.5, seed mới 8000..8099: test +5,73, CI mức 3 [+3,68; +8,32]; test-dev +7,81, [+2,70; +10,76] | **Mạnh** |
| PPO mode-only > H\* ở K5 test-dev | +12,36, CI mức 3 [+7,16; +27,22]; độ lớn không ổn định vì một video chiếm 129/403 khung | Mạnh về dấu, yếu về độ lớn |
| Heuristic tự hiệu chỉnh online không thay được PPO | A5.5: không tốt hơn ở chế độ nào trong 21 chế độ, kém hơn có ý nghĩa ở 11/21 | Mạnh |
| Phần thưởng nhiệm vụ không tạo hiệu ứng động; chính sách tốt nhất là mode cố định ở 2/3 chế độ | A6.0-L1 + audit (kết luận C2, C4, C5 đứng vững) | Mạnh (kết quả âm) |
| Dự đoán mode theo từng ảnh (codec cả ảnh) không hơn luật theo payload | G-L2: FAIL ở cả ba mức bit; không ô nào ≥ +1 điểm so với baseline triển khai được | Mạnh (kết quả âm) |
| Phản hồi ngữ nghĩa kiểu R1b không có dư địa | R1b-0a: H1 FAIL theo luật đã đăng ký | Mạnh (kết quả âm, trong thiết kế đó) |
| Token / hybrid thắng ảnh nén ở ngân sách thấp | R1a: +9 đến +20 điểm recall ở 0,08 Mbit / khung; nhưng phán quyết đăng ký là INCONCLUSIVE, và cơ chế "phát hiện trước khi nén" đã được biết | Trung bình |

## 3. PPO A5 / A5.5 thắng baseline nào, ở đâu, bao nhiêu?

**Thiết lập:** 3 UAV (ref) hoặc 5 UAV (K5), 2 kênh trực giao, nhiệm vụ khảo sát ảnh, hạn chót 4–12 s. PPO mode-only chỉ học mode nén cho từng ảnh; di chuyển, kênh, công suất do luật slot của heuristic tốt nhất đảm nhận. Train 2M bước, 3 seed. H\* = heuristic có return train cao hơn giữa họ A5 đã điều chỉnh (132 thành viên) và họ tự hiệu chỉnh (56 thành viên).

**PPO mode-only − H\*** (A5.5, seed môi trường mới; return mỗi episode):

| Chế độ | Pool | Δ [CI mức 1: seed môi trường] | CI mức 3: + video + seed train | Phán quyết |
|---|---|---|---|---|
| ref | test | −0,46 [−0,98; +0,06] | [−2,47; +1,79] | không khác |
| ref | test-dev | +2,63 [+2,06; +3,20] | [+0,06; +4,83] | sát ngưỡng, chưa đủ kết luận |
| K5 | test | +1,35 [+0,61; +2,10] | [−3,10; +5,78] | không chắc |
| K5 | test-dev | +12,36 [+11,23; +13,50] | [+7,16; +27,22] | vững |
| phy_mcs_blind | test | +5,73 [+4,91; +6,56] | [+3,68; +8,32] | **vững** |
| phy_mcs_blind | test-dev | +7,81 [+6,79; +8,83] | [+2,70; +10,76] | **vững** |

**Những gì không đứng vững:**
- Con số "+3,1 ở K5 test" của A5 hạ xuống +1,35, không chắc.
- PPO joint (học cả di chuyển, kênh, công suất) không đáng tin, trừ K5 test-dev; ở phy_mcs_blind test là −4,43.
- Lợi thế recall của PPO ở A5 giảm một nửa đến hai phần ba khi đếm mọi khung đúng hạn (audit L1, khám phá).
- Ở phy_mcs_blind test-dev, một mode cố định chọn sau khi thấy test (mbt q3) thắng PPO 2,6 (A5; so sánh có lợi cho heuristic).

**Bất định:**
- CI theo seed môi trường hẹp hơn CI theo cụm video 3–10 lần.
- Chỉ có 3 seed train, và chỉ 3 trong 21 chế độ được train.
- Ảnh test / test-dev đã được dùng ở A5, nên A5.5 không phải bộ ảnh mới.

**Diễn giải có căn cứ nhất** (chưa kiểm chứng trực tiếp): PPO có ích khi *mô hình của heuristic sai* (nhiễu không quan sát được; tải lệch khi nghẽn), không phải ở điều kiện danh định.

## 4. Đóng góp nào mới chỉ là giả thuyết, hoặc dựa trên GT?

| Phát hiện | Vì sao chưa phải kết quả |
|---|---|
| Gate 0A: dư địa phối hợp CROP +4 đến +12 điểm | Báo cáo hoàn hảo cắt từ box GT; liên kết xuyên góc bằng ID của GT |
| Pilot 0B: C1–C5 đạt | Khám phá; crop +1,2 đến +2,8 điểm F1, 70–99% đến từ chuỗi 59; bỏ chuỗi 59 thì mọi CI chứa 0 |
| Pilot 0B: T2 (bỏ trùng giữa hai UAV) +0,3 đến +0,7 điểm F1 duy nhất | Ổn định (8–11/11 chuỗi) nhưng dùng ID của GT; không triển khai được |
| Cổng crop một ngưỡng (LOSO) | Tiết kiệm 22–37% bit so với luôn crop; so với chỉ token chỉ +0,5 đến +1,1 điểm khi bỏ chuỗi 59 |
| A7-S0: cận trên thông tin +6 đến +11% return ở ref / K5 | Dùng giá trị thật của từng khung theo GT |
| A7-S1: giá trị FRAME ổn định theo cảnh | Giá trị khung trước / khung cùng cảnh cũng lấy từ GT |
| A7-S2: phản hồi BS không nhãn | NO-GO: giữ 8% / âm / 71% / 18% dư địa; dương giả ở cảnh đông |
| Giá trị FRAME khác nhau theo góc nhìn (26, 34, 52: một UAV 93–97%, UAV kia 0–11%) | Quan sát từ cận trên dùng GT, chấm riêng từng góc; **chưa** đo F1 vật thể duy nhất sau khi BS hợp nhất |
| Gửi cả khung +8,3 điểm F1, CI theo chuỗi [+5,3; +11,5] | Đo thật và vững theo chuỗi, nhưng ở ngân sách ~1,8 Mbit và mã hoá 1,17 s > chu kỳ 1 s; khám phá; quyền dữ liệu chưa xác minh |

Tất cả kết quả trên MDMT là khám phá (nhãn EXPL đã thấy; tham số chọn trên cùng tập), và quyền dùng dữ liệu chưa được xác minh độc lập.

## 5. Bài toán multi-UAV hiện tại có hợp tác thật không?

**Không.** Trong môi trường A5:
- Các UAV chỉ **chia sẻ tài nguyên vô tuyến**: K UAV dùng chung M kênh, có nhiễu đồng kênh.
- Mỗi UAV khảo sát vùng riêng. Ảnh được rút ngẫu nhiên từ pool và **không phụ thuộc vị trí UAV**; báo cáo gốc ghi rõ đây là mô phỏng tải truyền ảnh, không phải cảm biến chủ động.
- BS chỉ giải mã; không hợp nhất, không khử trùng. Utility là tổng theo từng ảnh.
- PPO là **một bộ điều khiển tập trung, single-agent** (không gian hành động MultiDiscrete cho cả K UAV), không phải MARL.

Hai chỗ duy nhất có hai UAV nhìn cùng vật là Gate 0A và Pilot 0B (T2), và cả hai đều cần liên kết xuyên góc bằng GT. Liên kết xuyên góc thật (Gate 1) chưa từng được mở.

**Hệ quả:** hiện SwarmBridge có (i) một kết quả RL về phân bổ tài nguyên, không có "swarm", và (ii) vài quan sát về hợp tác, chưa có cơ chế triển khai được.

## 6. Hai hướng đề xuất

### Hướng A: "Khi nào RL thắng heuristic đã điều chỉnh trong truyền thông ngữ nghĩa UAV" (đánh giá có kiểm soát + cơ chế)

- **Nội dung:** biến diễn giải ở mục 3 thành giả thuyết kiểm chứng được: lợi thế của PPO tăng theo mức sai lệch mô hình của heuristic (nhiễu không quan sát được, tải lệch), và bằng 0 khi mô hình đúng.
- **Tính mới** (theo hai bản novelty audit, chỉ trong các nguồn đã kiểm): mức trung bình.
  - Không bài nào trong 24 bài của thư viện so RL với heuristic được điều chỉnh trên tập riêng và có biến thể tự hiệu chỉnh.
  - CI ba mức theo cụm video, và các đối chứng không nhìn nội dung của G-L2, cũng chưa thấy.
  - Đây là đóng góp về **phương pháp đánh giá và cơ chế**, không phải thuật toán mới.
  - RL chọn codec dưới hạn chót (Huang JSAC 2025) và phần thưởng theo detector trên VisDrone đa UAV (CCSRA) **không** mới.
- **Điểm nghẽn:** chỉ 3 seed train và 3 chế độ; ảnh đã dùng lại; không có hợp tác; khó là đóng góp chính của một bài Q1 nếu đứng một mình.
- **Dữ liệu:** VisDrone-DET đã có, không vướng quyền như MDMT. Không cần ảnh mới.
- **Phép kiểm tra GO / NO-GO** (bước đầu không cần train):
  1. Đăng ký trước một thang sai lệch mô hình có ≥ 4 mức giữa `phy_mcs` (thích ứng theo SINR thật) và `phy_mcs_blind` (mù nhiễu).
  2. Chạy các chính sách PPO **đã train** và H\* trên thang đó (zero-shot), seed mới, CI mức 2.
  - **GO** nếu Δ(PPO − H\*) tăng đơn điệu theo mức sai lệch và CI loại trừ 0 ở ≥ 2 mức. Khi đó mới đề xuất train thêm seed.
  - **NO-GO** nếu lợi thế chỉ xuất hiện đúng tại chế độ đã train: nó là hiện tượng riêng của một cấu hình, không phải cơ chế.

### Hướng B: Lập lịch theo góc nhìn với kiểm chứng chéo giữa hai UAV (hợp tác thật)

- **Nội dung:** BS quyết định *UAV nào* được cấp tài nguyên để gửi biểu diễn giàu (FRAME / CROP), dựa trên việc thông tin mới của góc đó có đúng và có bổ sung cho góc kia không. Kiểm chứng detection mới bằng sự nhất quán giữa hai góc, thay cho nhãn. Đây là giả thuyết rút ra từ A7-S2: BS cần phân biệt thông tin mới có ích với detection mới nhưng sai.
- **Tính mới:** chưa đánh giá. Phải rà soát kỹ dòng collaborative perception (Where2comm, CoSDH, How2comm, HyComm, CCSRA, SchedCP) trước khi tuyên bố gì; hai bản audit hiện có không bao phủ đúng câu hỏi này.
- **Điểm nghẽn:**
  - Liên kết xuyên góc không dùng GT (Gate 1) chưa có, và là rủi ro lớn nhất.
  - Mọi bằng chứng hiện tại dựa trên GT.
  - Dữ liệu hai góc có ID chung chỉ có MDMT: 11 cặp EXPL đã dùng, 5 cặp val chưa mở, **quyền dùng chưa xác minh**. VisionTrack chỉ có người đi bộ.
  - Môi trường A5 không có vùng nhìn chồng lấn, nên cần môi trường khác.
- **Dữ liệu cần:** quyền dùng MDMT rõ ràng (thư tác giả hoặc xác nhận của lab / trường), hoặc một bộ hai góc khác.
- **Phép kiểm tra GO / NO-GO** (offline, trên bản ghi pilot đã có; không ảnh mới; không train), theo hai tầng:
  1. **Tầng 1, với liên kết lý tưởng theo GT** (để biết có gì đáng theo đuổi không): ở cùng ngân sách bit, so F1 **vật thể duy nhất sau hợp nhất** của "cấp FRAME cho góc tốt hơn, góc kia gửi TOKEN" với phân bổ đối xứng tốt nhất (cả hai TOKEN; cả hai FRAME; chia đều). Chọn góc bằng nửa đầu của chuỗi, đánh giá trên nửa sau.
     - **NO-GO ngay** nếu lợi ích < 2 điểm F1 duy nhất, hoặc dương ở < 8/11 chuỗi. Nếu liên kết hoàn hảo còn không cho lợi ích thì liên kết thật càng không.
  2. **Tầng 2, chỉ khi tầng 1 đạt:** thay liên kết GT bằng một phép khớp không nhãn đơn giản, và đo phần lợi ích còn giữ. Đây chính là Gate 1, cần duyệt riêng.
  - Tầng 1 dùng nhãn EXPL đã thấy; kết quả vẫn là khám phá.

### So sánh nhanh

| | Hướng A | Hướng B |
|---|---|---|
| Bằng chứng nền | Mạnh (A5.5) | Chỉ GT / khám phá |
| Đúng mục tiêu "swarm + semantic" | Không | Có |
| Rủi ro chính | Đóng góp chỉ là phương pháp đánh giá | Liên kết xuyên góc; quyền dữ liệu |
| Bước quyết định đầu tiên | Zero-shot trên thang sai lệch, không train | Tầng 1 offline, không train, không ảnh mới |

**Nhận định của tôi:** hai phép kiểm tra đều rẻ và không cần train, nên có thể làm cả hai trước khi chọn. Nếu chỉ chọn một: tầng 1 của hướng B trả lời đúng câu hỏi mà phản biện đặt ra (có bài toán hợp tác thật hay không), và nếu NO-GO thì SwarmBridge nên được viết như hướng A, không gắn nhãn "swarm".

## 7. Việc không làm trong bản này

Không train; không mở ảnh MDMT, val hay HOLD; không A7-S3; không Gate 1; không sửa kết quả cũ.
