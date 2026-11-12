# SwarmBridge H1-CROP: báo cáo sẵn sàng của pilot 0B-TUNE

Ngày 12/11/2026. Theo phản biện thứ tám: duyệt viết code pilot 0B-TUNE và test tổng hợp. Chưa duyệt HOLD, Gate 1 hay huấn luyện RL mới.

> **v2 (phản biện #9, cùng ngày):** T2 sửa lại thành tuần tự, chỉ bỏ dựa trên những gì UAV1 **thực sự gửi**, và đổi nhãn thành heuristic, không phải oracle. CPU chính tính mọi ứng viên được mã hoá. UL và DL được tách. Báo cả F1 micro và macro. Xem **mục 8**; mục 8 thay thế mục 1–3 ở những chỗ mâu thuẫn.
>
> **v3 (phản biện #10):** sửa bộ chấm Hungarian (trước đây có thể không tối đa số TP) và bất biến ngân sách UL của T2. Xem **mục 9**.

**Trạng thái:**
- Code, test, adapter và cấu hình đã xong.
- **Chưa chạy trên ảnh MDMT nào.** Adapter từ chối trả ảnh khi chưa có ghi chú cơ sở quyền dùng dữ liệu (mục 6).
- Model thật mới chỉ chạy trên 2 ảnh mẫu đi kèm ultralytics.
- A5/A5.5 PPO giữ nguyên làm nền RL.

## 1. Đã xây dựng gì

| File | Vai trò |
|---|---|
| `swarmbridge/experiments/uav_semcom/mdmt_adapter.py` | Đọc zip MDMT tại chỗ, không giải nén. Có chốt split (chỉ `test` = EXPL; `val` đóng), chốt quyền (không trả ảnh nếu thiếu `rights_note`), tách GT (`image()` chỉ trả pixel; `labels()` là lời gọi riêng). Chỉ mở member `.jpg` / `.xml`; không bao giờ đọc `.pth` hay `.py` trong zip. |
| `swarmbridge/experiments/uav_semcom/h1_pilot0b.py` | Lõi thuần (numpy / scipy): ứng viên P1 / P2 không dùng GT, hình học crop, lượng tử token, chọn theo ngân sách, oracle T2 có đánh dấu, gộp ở BS, bộ chấm, các phép đo C1, C2, C4, C5. |
| `swarmbridge/experiments/uav_semcom/h1_pilot0b_precompute.py` | Model thật, chạy một lần cho mỗi (epoch, góc): YOLO11n trên CPU 1 luồng (có đo giờ) → P1 / P2 → AVIF thật (q35, q50; khung q35) → YOLO11s trên crop **đã giải mã** (GPU, qua `gpulock_run.py`). Không đọc nhãn. |
| `swarmbridge/experiments/uav_semcom/h1_pilot0b_eval.py` | Chỉ CPU. Lưới điều kiện × B × α × τ_decl, bảng kết quả, tiêu chí C1–C5. Nhãn đọc qua adapter, không lấy từ bản ghi. |
| `swarmbridge/experiments/uav_semcom/h1_pilot0b_config.json` | Cấu hình benchmark: split, q, bước epoch, hai giai đoạn (`smoke_expl`, `pilot_expl`), ngưỡng, quy tắc B. |
| `swarmbridge/experiments/uav_semcom/test_h1_pilot0b.py` | 57 test tổng hợp. |

**Các điều kiện**, đều dùng cùng ngân sách tổng B của 2 UAV (protocol mục 4):

| Mã | Mô tả |
|---|---|
| T0 | Token chắc |
| T0+ | Token chắc, rồi token yếu |
| T1 | Token chắc, rồi crop P1 |
| T1P2 | Token chắc, rồi ô P2 |
| F | Khung AVIF; nếu không vừa thì quay về T0 |
| T2 | ~~Oracle liên kết hoàn hảo có tính phí~~ → **heuristic bỏ trùng có trợ giúp GT** (v2, mục 8). Dùng ID GT, không triển khai được, không phải cận trên |
| T2v | Như T2, nhưng chỉ tính vật mà BS thực sự nhận ra từ báo cáo của UAV1 (v2) |

## 2. Các chi tiết được chốt khi viết code (khai báo; khác hoặc cụ thể hơn protocol)

1. **Token 55 bit**, theo định dạng hint B+ của R1b:
   - gồm lớp 4, box 4 × 12 trên lưới 4096, và conf 3 bit (giải mã ở giữa ô);
   - Gate 0A dùng 52 bit vì không có conf; BS cần conf để gộp với detection từ crop và áp τ_decl;
   - box và conf của token được lượng tử thật trước khi chấm.
2. **Chọn theo ngân sách:**
   - tham lam theo thứ tự protocol (token chắc trước, rồi crop theo conf giảm dần);
   - món không vừa thì bỏ qua và thử món sau;
   - chi phí mỗi UAV = payload + header theo MTU (`h1_gate0a.msg_overhead`).
3. **T2:**
   - UAV2 bỏ ứng viên P1 nếu vật GT của nó cũng được UAV1 đề xuất, hoặc UAV1 đã có token chắc khớp đúng lớp;
   - ứng viên không khớp GT nào không bao giờ bị bỏ;
   - phí phối hợp dùng `coord_costs` của audit v2:
     - bản kê UL 24 bit / món của cả hai UAV (UAV1: token chắc + ứng viên; UAV2: ứng viên);
     - bitmap DL tới UAV2;
     - tất cả trừ vào B trước khi chia α.
4. **F:** gửi khung nếu vừa phần ngân sách của UAV, nếu không thì gửi T0. Báo tỉ lệ epoch thực sự gửi khung.
5. **Ánh xạ lớp** VisDrone → MDMT, cố định trước khi thấy ảnh:
   - pedestrian, people → person;
   - car, van, truck → car;
   - bicycle, motor → bicycle;
   - bus → bus;
   - bỏ tricycle;
   - có báo F1 không phân lớp để kiểm tra ánh xạ.
6. **Khớp:**
   - Hungarian theo lớp: tối đa số cặp IoU ≥ 0,5, rồi tối đa IoU (có test so với vét cạn);
   - ứng viên khớp GT nếu chứa tâm GT hoặc IoU ≥ 0,3.
7. **F1 duy nhất:** liên kết lý tưởng ở BS cho **mọi** điều kiện. Một vật thấy ở cả hai góc chỉ tính một lần; mọi khai báo không khớp là FP_wrong.
8. **Tinh chỉnh:**
   - α ∈ {0,3 … 0,7} và τ_decl ∈ {0,05 … 0,5} được chọn riêng cho từng (điều kiện, q, B) theo F1 từng góc;
   - chọn trên chính các chuỗi được báo cáo (khám phá), mọi điều kiện được tinh chỉnh như nhau.
9. **Quy tắc B:**
   - {0,25; 0,5; 1,0} × dòng trung bình mỗi epoch của "token chắc + mọi crop P1 ở q35, cả 2 UAV, có header";
   - thêm một mức luôn vừa 2 khung đầy đủ (= 2 × khung đắt nhất).
10. **Đo CPU:**
    - YOLO11n trên 1 luồng torch, làm nóng 3 lần;
    - AVIF dùng `max_threads=1` trong 5 tiến trình song song, mỗi worker được làm nóng (smoke test cho thấy lần mã hoá đầu tiên chậm gấp 4: 83 ms so với 21 ms);
    - đây là CPU desktop thay cho phần cứng UAV.
11. **Airtime:** tổng bit UL / R với R ∈ {2, 5, 10, 20} Mbit/s; báo tỉ lệ epoch vượt chu kỳ 1 s.

## 3. Test (131 qua: 57 mới + 74 của Gate 0A / audit)

Test chỉ dùng dữ liệu tổng hợp:
- **Adapter:**
  - zip giả cùng cấu trúc;
  - số file không bắt đầu từ 1;
  - box `outside` bị bỏ;
  - nhãn lạ thì báo lỗi;
  - `val` bị chặn; pixel cần `rights_note`;
  - không đọc `.pth` / `.py`;
  - quy tắc HOLD bằng hash.
- **Hình học:**
  - kích thước crop, cạnh tối thiểu, dịch vào trong ảnh;
  - crop luôn chứa box (200 ca ngẫu nhiên);
  - gộp cửa sổ; ô P2; đổi về toạ độ ảnh;
  - ánh xạ lớp.
- **Token và ngân sách:**
  - sai số lượng tử ≤ 1,5 ô lưới;
  - header theo MTU; bỏ món không vừa;
  - thứ tự món của từng điều kiện;
  - F vừa / không vừa.
- **T2:**
  - đúng luật bỏ (a), (b);
  - token yếu hoặc sai lớp không được tính;
  - không bao giờ bỏ FP;
  - phí khớp đúng `coord_costs`;
  - ngân sách chỉ đủ cho phí phối hợp thì không gửi được gì;
  - với ngân sách dư, T2 không làm giảm TP duy nhất.
- **Chấm:**
  - NMS theo lớp;
  - gộp token với detection từ crop;
  - Hungarian bằng vét cạn (15 seed);
  - một vật ở hai góc chỉ tính một lần;
  - τ và chế độ không phân lớp;
  - bản vector hoá khớp vòng lặp gốc (10 seed).
- **Sàng lọc:** C1 / C2 / ứng viên sai; C4 theo cả hai chiều; CPU (tất cả so với đã gửi); quy tắc B.
- **End-to-end trên đồ chơi:**
  - crop khôi phục được vật mà token bỏ sót, nên T1 > T0 và T1 > T0+;
  - F = 1;
  - C1–C5 tính đúng.
- **Tách GT:** phía UAV (`Pipeline.record`) không nhận và không đọc nhãn.

## 4. Smoke test với model thật (2 ảnh mẫu của ultralytics, không phải MDMT)

Mỗi ảnh được dùng làm cả hai góc (góc 2 là ảnh lật ngang). Chạy YOLO11n R1a, YOLO11s A5, AVIF thật.

**Đường ống chạy thông:**
- Mọi điều kiện cho ra khai báo và bit.
- T2 cộng đúng phí bản kê và bitmap (ví dụ DL 52 bit).

**Số đo, chỉ để kiểm code, không có giá trị kết luận:**
- YOLO11n CPU 1 luồng: 34–96 ms mỗi ảnh (810 × 1080 và 1280 × 720).
- Crop P1:
  - 5–49 kbit ở q35;
  - mã hoá 7–101 ms (crop lớn nhất cao 720 px).
- Khung đầy đủ q35: 136–418 kbit.

Ảnh mặt đất không phải miền của VisDrone, nên detector gần như không thấy gì.

## 5. Adapter trên zip MDMT thật (chỉ tên file và nhãn EXPL; không mở ảnh nào)

**Phát hiện lỗi căn chỉnh trước khi chạy:**
- Ở 6/12 chuỗi test, số file ảnh **không bắt đầu từ 1**. Ví dụ: 52 có file 191..550; 55 có 550..700; 56, 57, 61, 68 tương tự.
- Nhãn XML luôn là 0..n−1 **theo thứ tự ảnh**.
- Quy tắc cũ "khung f ↔ file f+1" (đúng ở chuỗi 26, nơi tôi đã kiểm bằng mắt) sẽ lệch nhãn ở 6 chuỗi này. Ví dụ chuỗi 55 và 68 sẽ có 0 nhãn ở mọi epoch.
- **Đã sửa:** khung f ↔ ảnh thứ f theo thứ tự. Hai góc phải có cùng số file ở mỗi vị trí.

**Quy tắc mới, ghi trước khi mở bất kỳ ảnh nào của các chuỗi này:**
- Chỉ dùng chuỗi mà ở cả hai góc, khung có nhãn lớn nhất (tính cả box `outside`) bằng số ảnh − 1.
- Chuỗi 55 bị loại: góc 1 có 151 ảnh nhưng nhãn chỉ tới khung 149, không biết lệch ở đầu hay cuối.

**Kết quả:**
- **11 cặp chuỗi** dùng được. Thứ tự pilot (hash nhỏ nhất trước): 56, 34, 52, 68, 48, 62, 26, 57, 59, 31, 61.
- **459 epoch** (mỗi 10 khung một epoch, tối đa 60 epoch mỗi chuỗi), **45.148 box GT** (2 góc).
- `val` bị chặn; pixel bị chặn khi thiếu `rights_note`. Đã kiểm trên zip thật.

**Còn phải kiểm:** căn chỉnh của các chuỗi có số file lệch cần xem bằng mắt (vẽ nhãn lên ảnh chuỗi 52 và 68). Đây là bước đầu tiên của `smoke_expl` khi được phép mở ảnh.

## 6. Điều kiện để chạy trên ảnh thật, và lệnh chạy

Theo phản biện #8: "chỉ chạy trên ảnh thật khi đã xác định việc sử dụng phù hợp với quyền và điều khoản; không coi bản mirror bên thứ ba là giấy phép."

**Cơ chế:**
- `precompute --source mdmt` bắt buộc có `--rights-note "<cơ sở>"`.
- Ghi chú này được ghi vào mọi file kết quả (`provenance`).

**Ai quyết định:** tôi không tự quyết cơ sở này. Có hai khả năng:
- **(a)** tác giả MDMT trả lời thư;
- **(b)** người dùng và ChatGPT quyết định rằng việc dùng nội bộ, khám phá, không công bố, không phát hành dữ liệu dẫn xuất là chấp nhận được trong lúc chờ.

**Khi có cơ sở, các bước theo thứ tự** (GPU qua `gpulock_run.py`, env WSL `~/envs/semcom-a`):
1. `smoke_expl`:
   - 2 chuỗi đầu (56, 34) × 10 epoch;
   - vẽ nhãn lên ảnh của 52 và 68 để kiểm căn chỉnh;
   - kiểm số ứng viên và thời gian;
   - **không đọc tiêu chí**.
2. `pilot_expl`:
   - 11 chuỗi, 459 epoch;
   - precompute ước tính khoảng 20–40 phút; đánh giá khoảng 2 phút (đo trên dữ liệu tổng hợp cỡ thật).
3. Báo GO / NO-GO cho giá trị CROP theo C1–C5, kèm P, R, FP, bit, CPU và airtime. Mọi kết quả đều là **khám phá (EXPL)**.

```
python h1_pilot0b_precompute.py h1_pilot0b_config.json OUT --source mdmt --zip data/mdmt/...zip \
    --stage smoke_expl --rights-note "<cơ sở>" --onboard ~/work/r1a/.../best.pt --server ~/work/a5_det/.../best.pt
python h1_pilot0b_eval.py h1_pilot0b_config.json OUT result.json --zip data/mdmt/...zip
```

## 7. Giới hạn đã biết

- Nhãn test (EXPL) đã được dùng ở Gate 0A, và α, τ_decl được chọn trên chính các chuỗi đó. Kết quả là **sàng lọc**, không phải xác nhận.
- Hai detector được huấn luyện trên VisDrone, chưa tinh chỉnh trên MDMT. Miền gần nhau (cùng nhóm, cùng loại cảnh drone) nhưng không giống hệt; ánh xạ lớp là một nguồn sai.
- CPU desktop thay cho phần cứng UAV. Tiến trình AVIF chạy song song nên thời gian có nhiễu do tranh tài nguyên.
- Chưa có đề xuất theo chuyển động, hàng đợi, mất gói, hay liên kết xuyên góc thật.
- F1 duy nhất dùng liên kết lý tưởng cho mọi điều kiện.

## 8. Readiness v2: sửa theo phản biện #9

### 8.1. Lỗi logic của T2 (v1)

**Lỗi:** v1 cho UAV2 bỏ crop nếu UAV1 **đề xuất** cùng vật, hoặc **phát hiện** nó bằng token chắc, kể cả khi UAV1 không đủ ngân sách để gửi. BS khi đó có thể không nhận được báo cáo nào về vật đó, tức là "oracle" tự làm mất TP duy nhất.

**Sửa:** T2 giờ chạy tuần tự.
1. **UAV1 lập lịch trước**, theo thứ tự T1, trong phần ngân sách của nó. Bản kê 24 bit cho mỗi món **đã gửi** nằm trong phần ngân sách đó: hàm `send_with_manifest` bỏ dần món cuối cho đến khi payload + bản kê vừa.
2. **UAV2 chỉ bỏ** ứng viên P1 khớp vật GT mà UAV1 **thực sự gửi**: một crop đã gửi phủ vật đó, hoặc một token chắc đã gửi (box đã lượng tử) khớp đúng lớp.
3. Ứng viên không khớp GT không bao giờ bị bỏ.
4. Nếu UAV2 không có ứng viên, không có phối hợp: T2 = T1 và chi phí bằng 0.

**Thêm T2v:** vật chỉ được coi là đã giao nếu BS **nhận ra** nó từ báo cáo của UAV1 (giải mã thật, khớp đúng lớp, conf ≥ 0,05). Biến thể này tách "đã gửi" khỏi "đã gửi và đủ tốt để BS nhận diện".

**Nhãn:**
- T2 và T2v là **heuristic bỏ trùng có trợ giúp GT**, với liên kết hoàn hảo có tính phí.
- Chúng **không** tối ưu F1 và **không** là cận trên.
- T2 − T1 ≤ 0 **không** có nghĩa là không còn dư địa hợp tác. Muốn có cận trên cần một bộ tối ưu riêng trên cùng không gian hành động; việc đó chưa làm.
- C4 đổi tên phép đo trùng lặp thành **trùng lặp ở mức đề xuất** (trước ngân sách).

### 8.2. Mô hình tài nguyên UL / DL

**B là ngân sách uplink chung của 2 UAV.**

| Thành phần | Kênh | Tính vào đâu |
|---|---|---|
| Bản kê của UAV1 (24 bit / món đã gửi) | UL | Phần ngân sách của UAV1 |
| Bản kê của UAV2 (24 bit / ứng viên) | UL | Phần ngân sách của UAV2 |
| Bitmap (1 bit / ứng viên UAV2 + header DL) | **DL, kênh riêng** | Không trừ vào B |

**Thiết lập chính, ρ = 0:**
- DL được báo bằng bit và theo thời gian phối hợp T_coord = (bản kê UAV1 + bitmap) / R, ở R ∈ {2, 5, 10, 20} Mbit/s, giả định R_DL = R_UL.
- Có thể đặt trần DL `B_dl`. Nếu bitmap không vừa thì không bỏ gì, nhưng bản kê vẫn phải trả.

**Độ nhạy, ρ ∈ {1, 4}:** ngân sách chia sẻ thời gian. ρ = R_UL / R_DL quy bit DL ra bit UL tương đương: B_eff = B − ρ · DL.

### 8.3. CPU

- **Phép đo chính:** UAV chỉ biết số bit AVIF chính xác sau khi mã hoá, nên chi phí xử lý = YOLO11n + mã hoá **mọi** ứng viên mà điều kiện có thể gửi:
  - T1 / T2 / T2v: mọi P1;
  - T1P2: mọi P2;
  - F: cả khung;
  - T0 / T0+: chỉ YOLO11n.
- Báo median và p90 cho từng điều kiện. C5 dùng con số này.
- `cpu_sent_median_supplementary` chỉ là con số phụ, lạc quan: nó chỉ đúng nếu đoán được số bit mà không cần mã hoá. Không dùng nó để tuyên bố khả thi.

### 8.4. Chỉ số và tiêu chí

- **Mỗi dòng kết quả gồm:**
  - F1 / P / R từng góc, bản **macro** (trung bình theo chuỗi) và **micro** (gộp số đếm);
  - F1 duy nhất, macro và micro;
  - F1 của từng chuỗi;
  - FP mỗi góc mỗi epoch;
  - bit UL thực đạt, bản kê, DL;
  - số crop đã gửi và crop sai;
  - CPU chính và CPU phụ;
  - airtime và T_coord.
- **Chọn α, τ_decl:** theo F1 macro, trên chính các chuỗi EXPL. Mỗi dòng đều ghi rõ là khám phá.
- **C3, quy tắc đăng ký ở v2 trước khi có kết quả MDMT:**
  - T1 − max(T0, T0+) > 0 **ở cả macro và micro**;
  - precision micro giảm không quá 0,05;
  - báo kèm ΔP, ΔR, ΔFP, chênh lệch từng chuỗi và số chuỗi dương.
- **C4:**
  - trùng lặp mức đề xuất ≥ 20%;
  - và F1 duy nhất của T2 hoặc T2v (ρ = 0) − T1 > 0 ở cả macro và micro tại ít nhất một B;
  - báo các mức ρ và ΔFP.

### 8.5. Test

**138 qua** (64 test của pilot + 74 test cũ). Test mới:
- **Phản ví dụ của phản biện:** UAV1 đề xuất X nhưng không đủ ngân sách gửi; UAV2 phải gửi X, và BS phải thấy X. Test này trượt với logic v1.
- Token phát hiện được nhưng không gửi thì không bỏ.
- Token yếu hoặc sai lớp không tính.
- T2v: crop đã gửi nhưng BS không nhận ra thì không bỏ.
- Kế toán UL / DL:
  - UL = payload + bản kê của cả hai UAV;
  - DL không trừ vào B khi ρ = 0;
  - B_eff = B − DL khi ρ = 1;
  - trần DL tắt việc bỏ nhưng vẫn tính bản kê.
- Không có ứng viên UAV2 thì T2 = T1.
- `send_with_manifest` luôn vừa ngân sách (100 ca ngẫu nhiên).
- CPU chính theo từng điều kiện.
- C3 chặn precision và báo từng chuỗi.

**Thời gian đánh giá** (dữ liệu tổng hợp cỡ thật): khoảng 3 phút cho 459 epoch.

### 8.6. Việc chưa thay đổi

- Vẫn **chưa mở ảnh MDMT nào**.
- Khi có cơ sở quyền dùng ảnh (thư của tác giả, hoặc xác nhận của lab / trường cho việc dùng nội bộ bản công khai qua mirror; tự điền `rights_note` không tạo ra quyền):
  1. chạy `smoke_expl`;
  2. kiểm căn chỉnh 52 và 68 bằng mắt;
  3. rồi chạy `pilot_expl` trên 11 chuỗi đã đăng ký.
- `val` vẫn đóng. Không HOLD, không Gate 1, không huấn luyện PPO.

## 9. Readiness v3: sửa theo phản biện #10

Cả hai lỗi đã được tái hiện trên code v2 trước khi sửa.

### 9.1. Hungarian chưa tối đa số TP

**Lỗi:** với trọng số 1 + IoU, đường tăng cường thay k cặp trùng khít bằng k + 1 cặp IoU 0,5 bị thua khi k ≥ 4 (2k > 1,5(k + 1)).
- Phản ví dụ của phản biện: 5 GT, 5 detection. Code v2 cho 4 TP, trong khi có thể ghép 5.

**Sửa:** trọng số cạnh hợp lệ = BONUS + IoU, với BONUS = min(n_det, n_gt) + 1, lớn hơn mọi tổng IoU có thể có. Như vậy thứ tự là: tối đa số cặp trước, rồi mới tối đa IoU.

**Test hồi quy:**
- phản ví dụ n = 5;
- 60 đồ thị lưới dạng chuỗi (bước 20 px, rộng 60: cặp trùng IoU 1, cặp lệch một bước IoU 0,5), đối chiếu với ghép cực đại Kuhn, một thuật toán độc lập. Kuhn lại được đối chiếu với vét cạn trên 30 đồ thị nhỏ.

**Kiểm tra rằng test có tác dụng:** chạy lại bộ test với trọng số cũ thì 3 test trượt (phản ví dụ + 2 seed lưới).

Sửa này làm **trước** mọi đánh giá MDMT, không phải sửa theo kết quả.

### 9.2. T2 vượt ngân sách UL

**Lỗi:** v2 cộng bản kê của UAV2 vào UL kể cả khi phần ngân sách của UAV2 không đủ trả. Phản ví dụ: B = 100, α = 0,5, một ứng viên, bản kê 168 bit → UL báo 168 > B.

**Sửa:** quy tắc fallback, đăng ký trước, hàm `coord_feasible`. Phối hợp chỉ được thử khi mọi thứ cần thiết vừa **từ trước**; các đại lượng đều biết trước khi gửi:
- UAV2 có ít nhất một ứng viên;
- B − ρ·DL > 0;
- bitmap vừa trần B_dl;
- phần ngân sách của UAV2 trả được bản kê.

Nếu một điều kiện trượt: epoch chạy như T1 với toàn bộ B, **không** gửi và không tính bất kỳ thông điệp phối hợp nào. Lý do fallback được ghi lại (`coord_reasons`).

Hệ quả:
- DL đã thử = DL đã giao (bằng 0 khi fallback);
- trường hợp trần DL không còn "trả bản kê mà không bỏ được gì".

**Bất biến:** UL + ρ·DL ≤ B, và DL ≤ B_dl, cho mọi điều kiện và mọi epoch.
- Test thuộc tính: 6 seed × 60 cặp bản ghi ngẫu nhiên × 7 điều kiện, với B từ 0 đến 1e6, α ∈ {0,3; 0,5; 0,7}, ρ ∈ {0, 1, 4}, B_dl ∈ {∞, 10, 80}.
- Đánh giá cũng kiểm bất biến này lúc chạy và dừng ngay nếu vi phạm.

### 9.3. Phân loại T2v

T2v là **heuristic oracle kết quả đã xác minh**: dùng GT để biết BS đã nhận ra đúng vật chưa, và giả định suy luận ở BS cùng phản hồi tới UAV2 xong trong epoch. Không triển khai được. T_coord chỉ gồm thời gian vô tuyến; thời gian suy luận ở BS chưa nằm trong phân tích deadline.

### 9.4. Trạng thái

- **207 test qua.**
- Đánh giá khoảng 3 phút cho 459 epoch (đo trên dữ liệu tổng hợp).
- Không đổi thiết kế nào khác.
- Bước tiếp theo, khi có cơ sở dùng ảnh: `smoke_expl` + kiểm căn chỉnh 52 / 68 bằng mắt → `pilot_expl` trên 11 chuỗi EXPL → báo GO / NO-GO trung thực cho câu hỏi: T1 có tốt hơn T0 / T0+ ở cùng ngân sách không?
- val đóng. Không HOLD, Gate 1, PPO.

## 10. Trạng thái 13/11: readiness v3 được duyệt (phản biện #11)

- **Code readiness v3: APPROVED.** Không còn vòng audit hay thiết kế nào trước pilot.
- Protocol, cấu hình và 11 chuỗi EXPL giữ nguyên. Ngưỡng không được đổi sau khi thấy kết quả.

**Đã chuẩn bị để chạy ngay khi có cơ sở dùng ảnh** (từ tác giả, hoặc lab / trường):
- `h1_pilot0b_align_check.py`: vẽ nhãn của hai góc lên ảnh ở epoch đầu, giữa và cuối của chuỗi 52 và 68; cùng ID thì cùng màu. Có test trên zip tổng hợp.
- `scripts/uav_semcom/h1_pilot0b_run.sh`:
  - `smoke`: chạy test, vẽ ảnh căn chỉnh, precompute `smoke_expl`, rồi **dừng** để xem ảnh bằng mắt;
  - `pilot`: precompute `pilot_expl` + đánh giá.

**Thứ tự báo cáo khi có kết quả:**
1. **C3 trước:** T1 so với T0 / T0+, gồm F1 macro / micro, P, R, FP, bit UL thực đạt, recall ứng viên, CPU p90, từng chuỗi.
2. Sau đó mới đến C4, tách riêng. T2 / T2v không được dùng để tuyên bố lợi ích hợp tác triển khai được.
3. Kết luận **GO / NO-GO / INCONCLUSIVE**.

Vẫn đóng: MDMT-val, VisionTrack-test, HOLD, Gate 1, PPO / MAPPO.
