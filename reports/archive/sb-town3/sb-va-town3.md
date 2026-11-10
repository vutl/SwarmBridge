# SB và Town3 — kết quả, kiểm toán và quyết định

Tổng hợp các report từ 30/10 đến 03/11/2026. Đây là nhánh lịch sử của SwarmBridge; trạng thái hiện tại nằm ở [reports/README](../../README.md). Các cấu hình và số liệu đầy đủ được giữ trong [archive/sb-town3](../README.md#sb-town3).

## Những điều đã rút ra

1. Trong các recipe đã thử, SB-decoder không vượt mapping mạnh nhất theo các chỉ số nhiệm vụ và luật pilot đặt trước. Tăng tỷ lệ endpoint và sửa phân phối train/eval thu hẹp khoảng cách, chưa tạo được lợi thế đồng thời ở các chế độ gửi.
2. Trí nhớ có căn pose và chọn block từ phía UAV có ích trên Town3; phần lớn lợi ích temporal ở mức gửi thấp đã xuất hiện với baseline điền từ trí nhớ đơn giản.
3. Với segmentation ba lớp cố định, truyền nhãn nhiệm vụ là đối chứng bắt buộc. Các ước tính chi phí cho thấy cần xét lại việc truyền latent, nhưng chưa phải phép so codec/kênh hoàn toàn đồng nhất.
4. Town3 hiện tại chưa tạo đủ tình huống nhiều UAV cung cấp bằng chứng bổ sung trên cùng đối tượng cho câu hỏi fusion/communication đã đặt ra. Điều đó là giới hạn của thiết lập, không có nghĩa UAV luôn phải bay chồng vùng.

## SB-decoder: pilot train/eval khớp nhau

Pilot 6.000 updates, ba seed, cùng loss nhiệm vụ và dữ liệu nhận; cột là mIoU một bước. Nhánh p70 là thăm dò, không thuộc luật pilot đặt trước.

| Model | Gửi 15% | Gửi 30% | Gửi 100% |
|---|---:|---:|---:|
| Mapping x-only | 0.6082 | 0.6931 | 0.8343 |
| Mapping có conditioning | **0.6102** | **0.6962** | **0.8360** |
| SB x-only, endpoint 10% | 0.5688 | 0.6681 | 0.8316 |
| SB x-only, endpoint 50% | 0.5983 | 0.6872 | 0.8349 |
| SB x-only, endpoint 70% | 0.6019 | 0.6888 | 0.8352 |
| SB có conditioning, endpoint 50% | 0.5982 | 0.6885 | 0.8353 |

Không biến thể SB nào qua gate pilot ở ba seed. Trong những thay đổi đã kiểm tra, tăng endpoint là yếu tố thu hẹp khoảng cách lớn nhất. Điều này không chứng minh mọi cách dùng SB đều thua; càng không phải chứng minh SB chỉ có thể có ích trong lọc tuần tự.

Nguồn chính: [pilot matched](sb-pilot-matched-2026-11-02.md), [root audit](sb-training-root-audit-2026-10-31.md), [review độc lập](review-claude-2026-11-02.md).

## Temporal, chọn block và ACK

Trên test Town3 ở mức gửi 15%, U-Net temporal đạt mIoU khoảng 0.685 so với mapping từng frame khoảng 0.609; baseline fill từ trí nhớ đã đạt khoảng 0.679. Các số này thuộc thí nghiệm temporal, không phải SB.

Pilot chọn block hai vòng, cùng ngân sách 15%, test ba seed:

| Chính sách | mIoU | IoU xe |
|---|---:|---:|
| Random | 0.676 | 0.220 |
| Trạm chọn theo ensemble task | 0.707 | 0.277 |
| UAV mô phỏng trạm, chọn theo task | 0.760 | 0.419 |
| Chẩn đoán biết trạng thái thật của trạm | 0.763 | 0.420 |

Khi mất toàn bộ ACK, UAV-replica còn khoảng 0.744 mIoU ở mức 15%. Đây là kết quả trong mô hình kênh đã thử; không suy rộng thành “không chắc receiver biết gì luôn là vấn đề nhỏ”. Chi phí feedback, hai vòng và tính toán phải được tính khi so hệ thống.

Nguồn: [temporal review](review-claude-2026-11-02.md), [chọn block](temporal-block-selection-2026-11-03.md), [hardening](step1-hardening-2026-11-03.md).

## Baseline truyền bản đồ

Hardening ghi mIoU test khoảng 0.771 ở khoảng 0.35 kbit/frame cho bộ nhớ cộng replica trên bản đồ; pipeline latent tương ứng được ước tính khoảng 8.4 kbit. **Bit bản đồ là độ dài mã lý tưởng theo mô hình ngữ cảnh; hai nhánh còn khác kênh và xử lý lượng tử.** Các tỷ lệ tiết kiệm trong report gốc là chẩn đoán của setup, không dùng như kết luận truyền thông tổng quát.

Bài học được giữ lại khi sang UAV3D: luôn so learned intermediate representation với message kết quả nhiệm vụ, cùng khả năng truy cập thông tin và ngân sách được định nghĩa rõ. Chuyển sang truyền segmentation map từng là đề xuất, không phải quyết định đề tài bắt buộc đã chốt.

## Vai trò toán học của SB

SB tối ưu KL trên không gian quỹ đạo so với reference dưới ràng buộc phân phối. Tối ưu đó không tự bảo đảm mIoU, tracking hay giá trị truyền tốt hơn mapping. Với squared error, conditional mean có tính tối ưu Bayes; điều đó không giải thích trực tiếp mọi chỉ số nhiệm vụ hoặc mọi recipe SB.

Nếu quay lại SB, phải định nghĩa phân phối/posterior cần học, reference, likelihood quan sát, quyền truy cập thông tin và quyết định downstream; sau đó so với baseline có cùng lịch sử/quan sát. Khoảng cách tới oracle hoặc memory sạch không tự là phần hiệu năng SB có thể thu hồi.

Nguồn: [nền tảng toán](sb-mathematical-foundations-2026-11-02.md), [deep research](sb-deep-research-2026-10-30.md), [tái lập paper](sb-paper-reproductions-2026-10-31.md), [hướng SB từng đề xuất](sb-huong-moi-2026-11-02.md). Báo cáo tiến độ đầu tiên và lệnh chạy đầy đủ: [30/10](bao-cao-tien-do-2026-10-30.md).
