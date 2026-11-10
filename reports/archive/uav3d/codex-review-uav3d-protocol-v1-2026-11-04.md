# Review độc lập protocol UAV3D v1 — 04/11/2026

Đối tượng review: `reports/archive/uav3d/uav3d-protocol-v1-2026-11-04.md`, commit `5d0fb46d`. Chỉ đọc code và chạy phân tích CPU; không sửa protocol hay code của Claude, không chạy GPU. Có thể sửa code song song theo các điểm được thống nhất, nhưng **chưa khóa protocol và chưa chạy GPU** cho tới khi giải quyết các điểm chặn dưới đây và review diff.

## Kết luận

Protocol đi đúng hướng: phân biệt dev/test, khởi động lạnh, codec, nguồn prediction ghép mAP–AMOTA, đối chứng và kiểm tra hình học. Tuy nhiên mục 9 chưa đủ: cần đặc tả mô hình truyền theo thời gian, phép so ở cùng budget, codec và luật quyết định trước khi gọi kết quả là bằng chứng. Đề xuất gating là giả thuyết mới; không mặc định thắng Kalman, cũng không coi một lần thua là bằng chứng mọi mô hình bộ nhớ vô ích.

## Đính chính về thời gian

Tái tính độc lập từ `sample.json` và các trường `translation`, `vx`, `vy` của annotation: lấy các cặp cùng instance ở frame kế tiếp, tốc độ đầu ≥2 m/s, thay đổi độ lớn tốc độ dưới 5%, thu được **31.889 cặp**, median quãng đường/tốc độ đầu **0,200599 s**, p5 **0,198917**, p95 **0,242453**; median từng scene trong cả 20 scene đều khoảng 0,20059–0,20060 s. Khác 9 cặp và p5 so với Claude có thể do chi tiết lọc/cách tính; cần lưu script và định nghĩa chính xác để đối chiếu, không phải khác biệt đáng kể ở median. Bằng chứng này ủng hộ mạnh **bước mô phỏng khoảng 0,2 s** trên mini, đủ làm giả định để triển khai và kiểm tra độ nhạy. Nguồn sinh trường vận tốc/đồng hồ chưa được kiểm chứng độc lập từ mã tạo dataset trong review này; cách giải thích timestamp là wall-clock và tốc độ mô phỏng chậm ~13 lần là hợp lý nhưng vẫn là suy luận cần provenance. Mâu thuẫn 0.5 Hz trong paper cần ghi rõ. Trước sửa `dt`, đánh giá nhạy cảm AMOTA/tracking giữa 0.2 và timestamp trên cùng file, nhất là sau khi baseline Kalman tạo velocity khác 0. Không nhầm velocity do detector/track dự đoán với ground-truth simulator.

## Các điểm chặn bổ sung vào mục 9

### 1. Định nghĩa quá trình gửi–đến theo từng link, tránh nhìn tương lai

Lấy ngẫu nhiên d độc lập mỗi frame rồi đọc thẳng message `t-d` chỉ là **mô hình lấy mẫu tuổi message**, chưa chắc là kênh packet có thể thực hiện: packet có thể tới không theo thứ tự, cùng lúc hoặc trễ hơn packet mới, và có thể có nhiều bản sao. Chốt một trong hai: (a) gọi đúng là age-sampling oracle/proxy và nêu rõ giới hạn, hoặc (b) sinh packet tại thời điểm s, loss và arrival `s+d`, bên nhận chỉ dùng packet đã đến, chọn bản mới nhất theo timestamp, xử lý đến đồng thời/out-of-order và giữ hay xóa cache theo quy tắc đã công bố. Không được truy cập `s>t`; cache phải reset theo scene và các frame trước khi packet đầu tiên tới phải không có message. Lưu ledger `{scene, sender, send_frame, arrival_frame, drop, seq, age_used}` và hash; cùng một ledger cho tất cả phương án. Phải định nghĩa policy khi packet rơi: giữ packet cũ hay không có message; hai nghĩa không tương đương.

Delay, loss và seed cần chỉ rõ train/eval: phân phối d, p hoặc tổ hợp p khi train; có/không kết hợp loss ở từng scenario; tách RNG model, channel train và kịch bản eval; fixed test ledger độc lập, thống nhất theo từng scene/link và lưu file. `drops.json` cũ không đủ cho d=3 và cold-start theo protocol; có thể giữ làm một tập so sánh đã định danh, không mặc nhiên tái sử dụng làm ledger mới.

### 2. So phương án ở cùng chi phí, không so thắng/thua Kalman mơ hồ

Gating gửi feature dense q8 còn Kalman gửi box có bitrate, compute, delay và nội dung khác rất nhiều. AMOTA/mAP cao hơn Kalman trong tình huống đó **không chứng minh gating hiệu quả hơn dưới cùng budget**. Báo cả accuracy, payload/frame, p95, header và compute/latency của mỗi phương án; giới hạn budget per-frame hoặc kỳ vọng bằng nhau và chốt cách chọn/gửi khi vượt budget. Nếu chưa có feature codec đạt budget gần box, chỉ so sánh như hai operating point, không khẳng định model thắng/thua ở cùng tài nguyên.

Gate tối thiểu cho giả thuyết kiến trúc nên là feature-gating so **feature-mean robust** cùng q8, source masks, encoder, số epoch, dữ liệu, delay/loss ledger, bitrate và input availability; thêm gating không có age để tách hiệu ứng biết tuổi. Có thể thêm nhóm mean không/ có age nếu khả thi. Gating có thêm tham số, tính và báo tham số/compute; không mô tả là chỉ khác age khi cả kiến trúc khác. Cùng seed không bảo đảm cùng random stream: log seed dữ liệu/model/kênh riêng.

Trước khi chạy chốt metric chính, kịch bản chính, hiệu số tối thiểu, cách tổng hợp 3 seed và từng scene, hướng xử lý đồng bộ vs trễ; tránh chọn ngưỡng thắng sau khi xem số. Với 3 scene, quy tắc chỉ là gate pilot, không tuyên bố ý nghĩa thống kê hay đóng toàn bộ hướng sau một thất bại.

### 3. Codec là bitstream thật, không chỉ lượng tử float

85 bit/box hiện chỉ có phân bổ bit, chưa đủ để encode: phải định nghĩa ranges, origins, units, signedness, z/size/score/velocity, rounding, overflow, box count, bit order và pack/unpack, đặc biệt 85 không chia hết cho 8. Phải đo **số byte thực** sau padding/header, báo số box bão hòa/sai số; nếu x/y theo tọa độ ego nhận thì sender cần thông tin pose receiver hợp lệ tại thời điểm gửi và chi phí của nó; tránh dùng pose hiện tại của tương lai để encode packet trễ. Receiver phải dùng giá trị giải mã, không được dùng bản float trước nén. Xác nhận thông tin velocity là detector/track output khả dụng chứ không phải GT.

Q8 cần chốt scale/zero-point hoặc min/max: precision, bit phụ trợ từng kênh, đối xử kênh hằng, padding, bitmask/index, dtype, bão hòa và serialization; đo số byte thật thay vì chỉ tính 8×số phần tử. Với feature sparse/q4 phải tính chi phí vị trí và overhead để khớp budget. 96 bit header chỉ là mô hình nếu không thực sự serialize; nêu rõ tính cả pose/timestamp/count, framing, số lượng link và đơn vị gửi. Phân biệt bit **phát** (kể cả packet bị rơi) với bit được nhận: mặc định ngân sách wireless phải đếm bit phát, chỉ rõ overhead/FEC/retransmission chưa mô phỏng và topology.

### 4. Warp và phép thử hình học

Không mặc định thêm yaw là đúng: trước tiên xác minh cached BEV thực sự ở hệ trục nào, sensor/ego/global transforms, chiều quy ước yaw, pixel-center và trật tự `grid_sample`; tránh xoay hai lần nếu feature đã được world-align. Dùng **kiểm thử tổng hợp SE(2)**: identity, dịch ±x/±y, quay ±90°, kết hợp dịch/rot, inverse, biên và mask ngoài vùng; so điểm/ảnh dự kiến để phát hiện dấu/swap. Kiểm tra thực với xe tĩnh ở nhiều scene là kiểm tra bổ sung, không thay thế test tổng hợp; ground truth chỉ dùng để chấm/kiểm định, không cấp vào model. Cần định nghĩa thế nào là xe tĩnh, ngưỡng sai số và trường hợp pose không đổi; báo phân vị chứ không chỉ một ví dụ.

### 5. Baseline Kalman và cap

Chốt tracker helper có gì tại thời điểm s: input box đã qua codec ở đâu, cách gán ID, khởi tạo state/covariance, Q/R, association, stale/missing, tốc độ khởi tạo và dt; không dùng GT hay tương lai; pose/time/chi phí có sẵn tương đương các nhánh. Gọi **Kalman lấy cảm hứng CERF**, không gọi official CERF reproduction trừ khi chạy code/thiết lập chính thức. Nếu CERF mô tả thêm kỹ thuật, ghi khác biệt cụ thể.

Cap 500 ở head và sau fusion là quy ước v1 chấp nhận được, không suy ra tự nhiên là công bằng tuyệt đối. Fixed score threshold mỗi phương án chọn trên dev và lock trước test; nêu rõ tuning budget như nhau và tập dev dùng chung; không quét lại threshold trên test. AMOTA bản chất tích phân nhiều ngưỡng: prefilter 0.1–0.4 có thể cắt recall tối đa và ảnh hưởng AMOTA; thêm không-prefilter (hoặc ngưỡng head thấp nhất) làm baseline và ghi rõ thứ tự cap/filter/tracker. Lưu params tracker và exact file hash.

### 6. Gating, test, train và điều kiện hoàn thành

Đặc tả rõ ego luôn có mặt; helper missing được mask **trước** softmax (tránh NaN), mọi helper mất cho ra đúng ego; tuổi phải là tuổi packet đã đến, không dùng tuổi tương lai/ground truth. Gating hiện chỉ chấm mỗi nguồn độc lập theo `f_i, age_i, mask_i`, không trực tiếp thấy ego hoặc tương quan nguồn; ghi đây là thiết kế có chủ ý. Khởi tạo đồng đều cần kiểm bằng unit test, kể cả zero helper. Define train/eval model cùng packet protocol; danh sách mục 9 chưa có test per-link delay, gate mask/age, initialization, giới hạn thông tin, loss/cold-start và logging.

Tập mini train 17 scene có thể gồm official val: trước khi chọn full val làm test, liệt kê ID/tokens và loại toàn bộ train/dev cũng như scene có nguy cơ checkpoint đã train; lưu manifest, dataset version và checksums. Chỉ gọi test độc lập **với fine-tune/protocol hiện tại** khi đã xác minh; nếu checkpoint provenance chưa rõ, nêu rõ không chứng minh độc lập với pretraining. Freeze code, config, checkpoints, ledger generation, thresholds, baselines và seed cùng tag; một lần chạy đã định nghĩa bao gồm mọi seed/kịch bản, xử lý crash/retry và thay đổi protocol minh bạch. Không cần ngăn tải hợp pháp trước khi khóa: điều quan trọng là không dùng test để điều chỉnh.

## Checklist bổ sung để Claude triển khai

1. Ledger causal và test arrival/out-of-order/loss/đầu scene/cùng seed.
2. Codec thực và kiểm thử pack/unpack, bit count, saturation, sender pose/timestamp, không GT và không future.
3. Test warp tổng hợp SE(2), xác minh hệ tọa độ BEV, static-scene; log mask.
4. Cùng input, bộ kịch bản, ngân sách, decoder/epoch cho mean vs gating; chốt phương án age, stop rule và metric.
5. Kalman có chi tiết state/noise/association/đầu vào, baseline gốc, cap/threshold và thí nghiệm độ nhạy dt.
6. Freeze manifest scene/checkpoint/config/seed/ledger; lưu prediction hash và xác thực input tracker/evaluator bằng hash; phát hiện file thiếu/thừa token và báo riêng từng scene.

Lưu ý trách nhiệm: pilot `train_separate.py` do Codex tạo và chạy dở rồi dừng theo yêu cầu của người dùng; không phải kết quả của Claude và chưa có model hoàn thành.
