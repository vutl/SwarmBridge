# SwarmBridge H1-CROP: báo cáo sẵn sàng của pilot 0B-TUNE

Ngày 12/11/2026. Theo phản biện thứ tám: duyệt viết code pilot 0B-TUNE và test tổng hợp. Chưa duyệt HOLD, Gate 1 hay huấn luyện RL mới.

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
| T2 | **Oracle liên kết hoàn hảo có tính phí** (dùng ID GT, không triển khai được) |

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
