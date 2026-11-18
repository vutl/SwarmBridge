# UAV3D — kết quả và những điểm đã kiểm toán

Tổng hợp đến 04/11/2026 từ report Claude và review Codex. Các kết quả budget/queue mới hơn nằm ở [report v1.1](uav3d-proto-v1.1-budget-queue-2026-11-04.md); quy tắc và checklist ở [protocol](uav3d-protocol-v1-2026-11-04.md). Bản tổng hợp này không phải một lượt đánh giá hoặc tái lập mới.

## Dữ liệu và phạm vi

UAV3D mini trong `data/v1.0-mini/`: 20 scene, 20 frame/scene, 5 UAV. Ba scene 11/13/15 được dùng làm dev sau khi đối chiếu với token trong file val của tác giả; 17 scene còn lại dùng fine-tune. Dev đã tham gia lựa chọn ngưỡng/recipe. Provenance checkpoint chưa được xác nhận đầy đủ chỉ từ file prediction val.

Kiểm tra displacement/vận tốc annotation ủng hộ bước mô phỏng khoảng 0.2 s, median khoảng 0.2006 s nhất quán giữa các scene. Timestamp cách khoảng 2.7 s, được giải thích là wall-clock nanosecond; không dùng trực tiếp làm dt mô phỏng của tracker. Nguồn đồng hồ/sampling trong paper vẫn cần ghi rõ khi công bố.

Visibility từ segmentation là heuristic, không phải nhãn che khuất từng instance đã được xác minh. Trajectory UAV cố định; đây chưa phải active sensing hoặc điều khiển bầy UAV.

## Kết quả protocol box v1

Sender gửi detection hoặc detection đã lọc Kalman kèm vận tốc; codec thật, ledger causal, cache tối đa ba frame, cold-start, cap 500 sau fusion. Bảng dưới là baseline protocol v1, chưa phải so cùng budget. Các kịch bản ngẫu nhiên lấy trung bình ba seed kênh.

| Phương án/kênh | mAP | AMOTA |
|---|---:|---:|
| Ego, không truyền | 0.512 | 0.542 |
| Box noisy-OR, sync | 0.665 | 0.594 |
| KF noisy-OR, sync | 0.648 | 0.690 |
| KF noisy-OR, trễ 1 frame | 0.594 | 0.631 |
| KF noisy-OR, trễ 2 frame | 0.519 | 0.529 |
| Box noisy-OR, trễ 2 frame | 0.433 | 0.408 |
| KF noisy-OR, mất 50% | 0.607 | 0.635 |

Nguồn đầy đủ: [report box v1](uav3d-proto-box-results-2026-11-04.md). Codex đã kiểm tra 61/61 hash input tracking khớp prediction và metadata dt 0.2; 70 checks của test script pass tại thời điểm review. AMOTA đã lưu khớp bảng; review không chạy lại official TrackingEval. Không dùng số checks làm bảo đảm toàn bộ pipeline đúng.

## Đính chính và kiểm toán cần nhớ

| Điểm | Diễn giải được giữ lại |
|---|---|
| Ghép nhầm mAP/AMOTA ở lượt đầu | Noisy-OR mAP 0.665 đi với AMOTA khoảng 0.588 ở cấu hình cũ; AMOTA 0.687 thuộc cấu hình khác. Chỉ dùng bảng có cùng file/hash. |
| Feature mAP 0.693 | Là cached FP16, chưa phải đo lại sau q8; không gán chi phí q8 cho kết quả FP16. |
| Box/feature ở lượt đầu | Có ước tính bit khác nhau và chưa có codec round-trip cho mọi nhánh; chưa kết luận “box rẻ hơn X lần” như một phép so công bằng. |
| Cap | Head mặc định 83 và cap 500 thay đổi kết quả; cần cùng nguồn helper và cap trước/sau fusion được công bố. |
| Cold-start | Script cũ dùng message mới hơn khi thiếu lịch sử; ledger mới xử lý không có message. Không trộn bảng hai protocol. |
| Warp | Code cũ chỉ dịch. Cần xác minh BEV frame, test SE(2) và tránh xoay hai lần nếu đã world-align. |
| Message KF | Nhánh hiện tại gửi box + velocity; không bật extension track ID/covariance. |
| Bit RX | Packet được dùng lại từ cache khác packet mới nhận. v1.1 đã tách tx/rx_arrived/used và tính byte padding. |
| Trễ | Có thể giảm detection dưới ego; không phải mọi phương án và mọi metric đều xuống dưới ego. |
| Tính mới | Độ trễ, AoI, VoI, fusion và task-oriented scheduling đã có prior art; kết quả dev chưa xác lập novelty. |

Nguồn: [audit độc lập](codex-uav3d-independent-audit-2026-11-04.md), [review protocol](codex-review-uav3d-protocol-v1-2026-11-04.md), [review box](codex-review-proto-box-results-2026-11-04.md), [report lượt đầu có đính chính](uav3d-comm-results-2026-11-04.md).

## Những câu hỏi chuyển sang v1.1

Report v1.1 đã bổ sung ablation KF-0/V/P/PV, kiểm tra velocity clipping, lựa chọn nội dung theo budget và hàng đợi. Các phần đó thay thế mục “việc tiếp theo” trong report v1 cũ. Một số lượt tracking còn đang hoàn thành tại snapshot được ghi trong report v1.1; xem trạng thái tại đó và JSON trong `runs/`, không suy ra từ bản tổng hợp này.

Cần tách: lợi ích velocity cho association; forecast vị trí; smoothing/track confirmation; và lựa chọn message. Mất gói có cache làm thay đổi cả độ tuổi dữ liệu, không chỉ số helper. Điều kiện kênh, budget, quyền biết receiver/CSI và phí feedback quyết định phép so có công bằng hay không.

Khoảng cách tới best-found dùng GT là chẩn đoán, không phải cận trên hoặc bảo đảm policy học được sẽ thu hồi. RL cần chứng minh giá trị quyết định tuần tự; MARL cần mô hình agent phân tán cụ thể. Chưa dùng kết quả một mô hình trên ba scene để loại toàn bộ phương pháp hoặc khẳng định một research gap.

Pilot `train_separate.py` do Codex khởi chạy rồi dừng theo yêu cầu; chưa có model hoàn thành. [Kế hoạch pilot lịch sử](uav3d-separated-source-pilot-2026-11-04.md) không phải kết quả của Claude hoặc một job cần tự khởi động lại.
