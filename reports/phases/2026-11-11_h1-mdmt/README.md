# H1 trên MDMT: Gate 0 và pilot 0B (11–13/11)

Đánh giá lại hướng, báo cáo hợp tác theo vật thể duy nhất, Gate 0A, pilot crop 0B; lợi ích crop nhỏ, không ổn định.

| # cũ | Tài liệu | Nội dung | Trạng thái |
|---|---|---|---|
| 10 | [swarmbridge-direction-reassessment.md](swarmbridge-direction-reassessment.md) | Đánh giá lại hướng sau R1b: postmortem, prior art, 3 hướng, novelty / dataset matrix, cổng 0 trên MDMT. Ghi chú nguồn: [reassess-notes/](reassess-notes/) | H1 duyệt ở mức ứng viên; Gate 0 thay bằng #11 |
| 11 | [swarmbridge-h1-gate0-protocol-v2.md](swarmbridge-h1-gate0-protocol-v2.md) | Protocol Gate 0 v2 cho H1: phân loại oracle, đặc tả bộ chấm F1 đối tượng duy nhất, ma trận baseline, split chống thiên lệch chọn, quyết định quyền dữ liệu, checklist | v2.1 khoá cho 0A; 0B chưa duyệt |
| 12 | [swarmbridge-h1-gate0a-result.md](swarmbridge-h1-gate0a-result.md) | Gate 0A (khám phá, nhãn MDMT-test, báo cáo hoàn hảo): dư địa phối hợp theo ngân sách, TOK và CROP riêng | TOK: dư địa hẹp (≈ dòng token); CROP: dư địa ở 100–500 kbit/epoch |
| 13 | [swarmbridge-h1-gate0a-audit.md](swarmbridge-h1-gate0a-audit.md) | Kiểm toán 0A: S2 là oracle có trễ; oracle chính xác (MILP); chi phí bản kê; baseline nhớ lịch sử; điều kiện C1–C6 cho CROP | TOK NO-GO; CROP GO có điều kiện; v2 sửa chi phí bản kê (mục 8) |
| 14 | [swarmbridge-h1-0b-tune-pilot-protocol.md](swarmbridge-h1-0b-tune-pilot-protocol.md) | Protocol pilot 0B-TUNE cho CROP: ứng viên không GT, AVIF thật, YOLO11n/s thật, F1 có FP, CPU, airtime, tiêu chí sàng lọc C1–C5 | phương án B (11 cặp EXPL) theo #15; chạy ảnh thật chờ cơ sở quyền dữ liệu |
| 15 | [swarmbridge-h1-0b-pilot-readiness.md](swarmbridge-h1-0b-pilot-readiness.md) | Sẵn sàng pilot 0B: adapter MDMT (chốt split / quyền / GT), lõi pilot, precompute model thật, đánh giá C1–C5, 57 test, smoke test model thật; sửa lỗi căn chỉnh số file | v3 duyệt (#11); đã chạy pilot, xem #16 |
| 16 | [swarmbridge-h1-0b-pilot-expl-result.md](swarmbridge-h1-0b-pilot-expl-result.md) | Pilot 0B trên 11 chuỗi MDMT EXPL với YOLO11n / AVIF / YOLO11s thật: C1–C5, T1 so với T0 / T0+, F, T2 | PASS khám phá, hiệu ứng yếu (+1–3 điểm, chủ yếu chuỗi 59); khung +8 điểm nhưng vượt CPU. Mục 6: cổng 'crop khi detector thiếu chắc chắn' dự đoán được (LOSO) nhưng thưởng nhỏ → chưa train PPO cho TOKEN/CROP. Quyền dữ liệu chưa xác minh |
| 17 | [swarmbridge-h1-0b-pilot-cluster-ci.md](swarmbridge-h1-0b-pilot-cluster-ci.md) | Độ tin cậy pilot 0B: bootstrap theo 11 chuỗi + bỏ từng chuỗi (từ JSON đã lưu) | Crop: CI chỉ loại 0 ở B 305k, 70–99% lợi ích từ chuỗi 59, bỏ 59 thì mọi CI chứa 0; T2 (GT) +0,3–0,7 ổn định; khung +8,3 [5,3; 11,5] |

Ghi chú đánh giá lại hướng: [h1-mdmt/reassess-notes/](reassess-notes/).

Tổng quan mọi giai đoạn: [reports/README.md](../../../README.md).
