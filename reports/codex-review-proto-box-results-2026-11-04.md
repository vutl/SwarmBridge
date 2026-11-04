# Review kết quả protocol box và nhận xét ChatGPT — 04/11/2026

Phạm vi: đọc report, code tại commit `820d0d6d` và đoạn ChatGPT người dùng gửi; chạy lại các kiểm tra CPU hiện có, đối chiếu hash và metrics. Không sửa code/kết quả của Claude, không chạy tracking lại hoặc GPU, không đánh giá test và không xóa file.

## Những gì đã kiểm tra trực tiếp

- `test_proto.py data/v1.0-mini`: **70 checks pass**. Đây là các assertion trong test script, không phải 70 thí nghiệm hay bảo đảm toàn bộ pipeline đúng.
- **61/61** file prediction có hash khớp cả `box_proto.json` và `tracking/<method>__<ledger>/input.json`; cả 61 metadata tracking ghi dt `0.2`. Các metrics AMOTA đã lưu tổng hợp lại khớp bảng report. Không tự chạy lại official TrackingEval trong review này.
- Các ví dụ: ego AMOTA 0.5423; kf_nor sync 0.6900, fix1 0.6306, fix2 0.5293, loss0.5 trung bình 0.6347; box_nor fix2 0.4079. Test hiện có đo lỗi vị trí sau dự báo trên tập xe chuyển động được chọn: median 0.98 m so với 1.78 m nếu giữ vị trí cũ (n=2185). Đó là chẩn đoán hữu ích, chưa xác định nguyên nhân thay đổi AMOTA.
- `ledger.py` chọn packet đã đến theo send_frame mới nhất, có expiration và reset scene, xử lý cold-start và out-of-order đúng theo mô hình đã định nghĩa. Mô hình hiện tại chưa có hàng đợi, budget hay delay phụ thuộc payload/policy.

## Hai nhận xét code của ChatGPT đều đúng

1. `box_proto.py` bỏ `tid/logvar` qua `base(m)` và gọi `BC.encode(...)` không có `track=True`. Nhánh `kf` thực chất gửi **detection đã lọc Kalman, kèm vận tốc**, không truyền ID/covariance. Tên đầy đủ “probabilistic track-state communication” không phù hợp với nhánh này. Codec có hỗ trợ extension không có nghĩa pipeline đang dùng nó. Chưa cần thêm extension ngay nếu phép thử budget chưa cần đến; có thể định nghĩa một nhánh riêng sau này và tính đủ chi phí.
2. `kb_rx` cộng kích thước packet mỗi frame packet được dùng, kể cả khi nó đang ở cache. Đây là lượng dữ liệu được tiêu thụ từ cache mỗi frame, không phải throughput nhận qua wireless. Nên đổi tên thành cached/used-message bits, đồng thời nếu cần đo throughput nhận thì cộng từng packet một lần tại arrival_frame, kể cả packet cũ đã bị packet mới thay thế. TX vẫn phải tính các packet đã phát dù drop hoặc đến sau cuối episode; quy tắc tail của episode cần công bố.

Một chi tiết bổ sung: TX hiện cộng `st['bits']` (132+87n), còn bitstream thực có padding tới byte được lưu ở `st['bytes']`. Để gọi là bit vật lý của payload đã serialize, dùng `8*st['bytes']`; nếu tiếp tục dùng useful bits thì báo riêng useful bits và padded bits. Chênh tối đa 28 bit/receiver-frame với 4 packet, nhỏ so với 41–50 kbit, không giải thích các chênh AMOTA.

## Những kết luận cần giữ ở mức giả thuyết

**Ablation vị trí/vận tốc là ưu tiên hợp lý.** Bốn nhánh KF-0, KF-V, KF-P, KF-PV phải dùng cùng sender tracker, cùng tập detection/track được xác nhận, cùng codec, packet ledger, cap và tham số receiver tracker. Thao tác tắt vận tốc diễn ra sau khi đã dự báo vị trí (ở nhánh P), ngay trước receiver tracker; không vô tình tắt vận tốc dùng để forecast. Giữ message/chi phí giống nhau để đây là ablation cơ chế. Nếu sau này bỏ hẳn trường velocity để tiết kiệm bit, đó là thí nghiệm representation/budget khác. Sync so với delay trong cùng kf đo tác động của độ trễ, chưa đo riêng hiệu quả của việc bật forecast.

KF và box còn khác nhau ở smoothing vị trí, xác nhận MIN_HITS, lựa chọn track và số box; bởi vậy khác biệt AMOTA sync chưa thể quy toàn bộ hoặc phần lớn cho velocity. Tỷ lệ 76% box có velocity chỉ xác nhận tracker có quyền dùng velocity, không lượng hóa tác dụng nhân quả. Nếu cần giải thích phần lợi so box thì thêm kiểm soát track selection/smoothing, hoặc giữ kết luận ở phạm vi ablation bốn nhánh KF.

**Saturation không tự động vô hại.** Track ghép nhầm vẫn tham gia forecast, fusion và tracker; clipping một velocity sai có thể giúp hoặc làm hại. Ghi tần suất, mẫu lỗi và chênh metric so range rộng/FP32 hoặc codec velocity khác, giữ các yếu tố còn lại. Chẩn đoán dùng FP32 là oracle codec reference, không là phương án bitrate tương đương.

**“Mất gói ít hại hơn trễ” chỉ đúng ở các operating point và phương án đang so.** Với cache, gói mất có thể khiến hệ dùng packet cũ; không phải loss chỉ giảm số UAV có mặt. box_nor loss0.5 AMOTA 0.5074 vẫn dưới ego 0.5423, khác kf_nor. Báo phân phối tuổi packet thực sự dùng, tỷ lệ không có message và cache hits để giải thích. Không suy ra một quy luật tổng quát cho mọi link/budget.

**mAP khác AMOTA không chứng minh mọi policy chọn theo mAP sẽ chọn sai message.** Kết quả cho một ví dụ hai pipeline có thứ hạng đảo; chưa đo marginal utility hay policy lựa chọn message. Nó đủ biện minh phải chấm downstream tracking. Chưa chứng minh một loại representation hoặc scheduler tối ưu cho tracking.

## Bước tiếp theo đề nghị

Đồng ý với ChatGPT rằng sau chuẩn hóa ngắn nên chuyển sang **budgeted scheduling và kiểm tra phần lợi còn có thể đạt được**, thay vì tiếp tục mở nhiều model fusion. Hoàn thành bốn nhánh KF, sửa accounting, kiểm tra saturation và độ nhạy Kalman có giới hạn trên dev/train; cập nhật protocol v1.1 và lưu manifest/params/hash. Không gọi “bù trễ tử tế đã đủ mạnh” trước khi làm các kiểm tra này.

Pilot budget nên có codec/message cost thật, cùng input và channel realization cho mọi policy, một số budget cụ thể đã chốt, random/top-score/novelty/age-aware và greedy có ích cho tracking. Định nghĩa ai được biết receiver state/ACK, candidate object/track/packet, ID giữa các UAV và phí feedback. Nếu mục tiêu là scheduling gây delay, service time/queue phải phụ thuộc payload, rate và hành động; tiếp tục ledger delay độc lập chỉ kiểm tra lựa chọn nội dung ở tuổi message ngoại sinh. Với dataset cố định, action truyền có thể đổi receiver tracker/queue và quyết định tương lai; chưa có action bay hay active sensing.

**Oracle gap không chứng minh cần RL hoặc MARL.** Gap có thể do oracle biết GT/tương lai mà policy không thể suy ra, hoặc do heuristic yếu. Định nghĩa rõ oracle biết tương lai task, channel và receiver state nào; phân biệt oracle theo một bước và theo cả episode với continuation policy cố định. Dùng các nhánh gửi/không gửi từ cùng tracker/queue state, cùng exogenous noise để đo tác động counterfactual, không chỉ hiệu hai conditional averages. Beam search hoặc greedy dùng GT chỉ là “best found diagnostic”, không tự động là oracle tối ưu/cận trên. Với nhiều UAV và nhiều object, tối ưu cả episode thường không thể vét cạn; có thể kiểm tra chính xác trong tập candidate/horizon nhỏ rồi mở rộng xấp xỉ có ghi rõ.

Một gap khả quan biện minh thử supervised utility estimator, bandit hoặc planning trước; RL cần bằng chứng rằng quyết định hiện tại làm đổi chất lượng và cơ hội truyền tương lai đủ để vượt myopic utility. Scheduler trung tâm điều phối nhiều UAV có thể là một bài toán single-agent; MARL cần agent phân tán, observation và action riêng được định nghĩa. Không gắn MARL chỉ từ số UAV.

Có thể viết problem formulation/proposal làm việc ngay với những giả thuyết có khả năng bị bác bỏ, nhưng “Arrival-Time Semantic Value” hiện là cách đặt câu hỏi, chưa phải novelty đã xác lập. Prior-art và phép so cùng budget vẫn cần được xử lý trước khi tuyên bố đóng góp thuật toán.
