# Review đề xuất chọn block sau temporal baseline

Ngày 03/11/2026. Đọc toàn bộ transcript người dùng gửi, đối chiếu code temporal, channel và Q2 progressive hiện có. Đây là review và protocol đề nghị; chưa chạy thí nghiệm GPU mới.

## Nhận định

Nên làm một pilot chọn block dùng checkpoint có sẵn trước khi train thêm SB. Nhưng bảng ba chính sách trong transcript chưa đủ chặt để phân biệt hiệu quả của sender/receiver selection hay quyết định có tiếp tục SB. Cần sửa thông tin mỗi bên được phép dùng và cách đọc kết quả.

## 1. Những điều đồng ý

- Temporal rollout ba seed đã kết thúc; log có `TEMPORAL DONE`. Số liệu tổng hợp trong transcript khớp bản review trước.
- Memory fill là baseline bắt buộc. Khoảng cách với quá khứ sạch chưa phải hiệu năng SB có thể lấy lại.
- Town3 cho phép kiểm tra predictive communication và phân bổ tài nguyên giữa những UAV phụ trách vùng khác nhau. Nó chưa xác nhận helper-view completion cho cùng ROI. Không cần ép các UAV nhìn trùng nhau nếu mục tiêu là phủ vùng và scheduling.
- Chưa có căn cứ đưa một SB filter lớn vào latent BEV ngay. Một thí nghiệm inference có posterior giải tích vẫn có giá trị riêng để kiểm tra sampler.

## 2. ACK không đồng bộ chính xác receiver trong kênh hiện tại

`channel/channel.py` truyền latent thực qua AWGN/fading/erasure. Nhận được ACK chỉ cho biết block đến nơi, không cho sender biết hiện thực nhiễu trong giá trị mà receiver đã nhận.

Ví dụ scalar, age=0 và biết r: `z_p = r*z + epsilon`, với `Var(epsilon|z,r) = r*(1-r)`. Hai lần truyền cùng z, cùng r, cùng ACK vẫn có thể cho hai z_p khác nhau. Nếu receiver dùng z_p để cập nhật trí nhớ, UAV không suy ra chính xác trí nhớ đó từ z và ACK.

Mô hình video codec có decoder replica là đối chứng hợp lý khi các symbol được giải mã đúng và hai bên biết cùng trạng thái, model, pose và các cập nhật xác định. Để dùng lập luận đó ở đây, cần một trong các setup rõ:

- đường truyền digital đã giải mã, nội dung lượng tử hóa thành công được hai bên biết giống nhau;
- feedback bổ sung đủ để đồng bộ trạng thái, có tính chi phí;
- hoặc sender giữ một replica xấp xỉ và chịu lỗi đồng bộ.

Chính sách dùng latent sạch và **state thật của receiver** nên được gắn nhãn ideal-information diagnostic. Không gọi nó là sender policy triển khai được với ACK đơn thuần.

Ngay cả bản ideal chọn lỗi latent lớn nhất cũng không phải upper bound mIoU: lỗi latent lớn không nhất thiết là block có task gain lớn nhất, và việc truyền vẫn qua kênh.

## 3. Chọn block không phải phần hoàn toàn chưa được xét

`swarmbridge/experiments/q2_progressive/run.py` đã so `random`, `channel`, `unc_h1`, `unc_sb`, `oracle`. `oracle` chấm lỗi `(z_hat-z)^2`; các chính sách dùng cùng H1 decoder. Các JSON `runs/q2p_s*.json` đã tồn tại.

Phần mới hợp lý là đưa phép so đó sang **temporal state**, dùng checkpoint mới và đối chứng sender replica. Không nên ghi rằng toàn bộ Q2 trước đây đều gửi ngẫu nhiên. Đúng hơn: temporal rollout hiện tại đang gửi ngẫu nhiên.

## 4. Hai phép suy luận về SB cần thu hẹp

### MSE không giải thích toàn bộ kết quả âm

Conditional mean tối ưu MSE; mẫu posterior độc lập chịu expected MSE gấp hai Bayes risk. Nhưng one-step SB trong code trả một estimator xác định, còn các bảng chính chấm mIoU/IoU xe và có task loss. Định lý không chứng minh vì sao estimator hữu hạn đó thua mapping, và không giải thích trực tiếp vì sao spread-based scheduling thua.

Có paired data cũng chưa đủ tự động loại rủi ro correspondence: objective và conditioning phải dùng đúng quan hệ cặp. Nếu thật sự học p(z_k | z_{k-1}, pose, dt) và giữ điều kiện xuyên suốt, phản ví dụ chỉ khớp marginal không nhắm vào setup đó. Nếu chỉ lấy hai tập marginal để chạy SB, việc dataset lưu sẵn cặp không cứu objective.

### Ensemble thất bại không loại SB

Ensemble thắng random cho thấy một scorer cụ thể có giá trị; chưa chứng minh nó là posterior đúng hay SB sẽ cải thiện hơn. Ensemble thua có thể do scorer, calibration, tương quan giữa model, hoặc block uncertainty không khớp task utility. Chưa đủ để loại mọi posterior sampler.

Ngược lại, nếu ensemble đã thắng tốt thì SB vẫn phải đem lại lợi ích thêm đủ bù chi phí. Khả năng sender đồng bộ receiver không phải điều kiện duy nhất quyết định có cần uncertainty: dynamics tương lai, thông tin ngoài FOV và kết quả truyền tương lai vẫn có thể bất định.

`z_p` cần likelihood phù hợp là yêu cầu của filter sắp xây; temporal mapping hiện tại không dùng Bayesian likelihood. Vì vậy chưa có một bug likelihood đang chạy cần sửa trong model này.

## 5. Protocol pilot đề nghị

### Chính sách và quyền truy cập thông tin

| Chính sách | Được dùng | Ý nghĩa |
|---|---|---|
| Random | Mask hợp lệ, lịch sử nhận/gửi | Đối chứng |
| Receiver ensemble | Lịch sử/quan sát thực của trạm và ba temporal model | Heuristic uncertainty, chưa gọi là VOI |
| Sender innovation | Latent hiện tại ở UAV và replica do sender tự xây từ thông tin nó thực sự biết | Chính sách khả thi theo giả định đồng bộ đã công bố |
| Ideal residual | Latent sạch và trạng thái thật của receiver | Diagnostic có thêm thông tin, không coi là sender deployable/upper bound task |

Nếu chưa muốn xây đồng bộ sender thì có thể chạy ba hàng đầu tiên theo nghĩa random / ensemble / ideal residual, nhưng phải ghi rõ chưa kiểm tra sender policy khả thi.

### Cách giữ phép so công bằng

1. Chọn thời điểm quyết định rõ ràng. Pilot dễ diễn giải: cùng một vòng đầu ngẫu nhiên, sau đó chọn lượng block bổ sung bằng nhau; tránh dùng sai số sau khi đã nhận chính block đang được chọn.
2. Dùng cùng quy tắc candidate: gửi block mới hay được phép retransmit phải chốt trước. Trừ các block đã gửi nếu đang kiểm tra new-block selection.
3. Dùng cùng decoder và cùng thông tin đã nhận cho mọi policy; ensemble chỉ làm scorer trong phép so đầu tiên. Nếu đổi cả decoder thì không tách được lợi ích selection.
4. Presample hiện thực kênh theo frame/round/block rồi dùng chung giữa các policy. Giữ số block yêu cầu bằng nhau, ghi thêm số nhận thành công. Chi phí ACK/request/đồng bộ phải báo riêng; chưa gọi tỷ lệ block là bitrate thực tế.
5. Đánh giá val trước để chốt scorer, sau đó một lượt test đã cố định protocol. Ghi per-scene mIoU, vehicle IoU, tiền cảnh/nền và runtime. Không coi ba seed tối ưu là ba scene độc lập.
6. Phép so cùng một history giúp tách chất lượng chọn block; sau đó cần rollout mỗi policy bằng chính state của nó để đo tác động dài hạn. Không cho policy triển khai được lấy state sạch hay state của đối thủ.

Latent-error ranking có thể thắng mà vẫn bỏ sót xe. Nếu cần chẩn đoán task VOI, trên một subset nhỏ có thể đánh giá hậu nghiệm từng candidate query bằng task gain qua cùng decoder. Ground truth chỉ được dùng chấm/huấn luyện, không đưa vào receiver policy triển khai.

### Luật đọc kết quả

- Ideal residual hơn random: có khoảng trống để cải thiện content selection; chưa là đóng góp mới hoặc chứng minh sender biết làm được.
- Sender replica gần ideal: giả định đồng bộ đủ tốt trong setup đó; tiếp tục đo chi phí và độ bền khi feedback/kênh lệch.
- Ensemble hơn random: tiếp tục kiểm tra task alignment và chi phí, rồi mới so một posterior model khác nếu cần.
- Ensemble kém: chưa mở train SB lớn dựa riêng vào kết quả này; cần tách lỗi uncertainty khỏi việc task không hưởng lợi từ lựa chọn block.
- Không policy có thêm thông tin nào đem lại lợi ích đáng kể: bằng chứng bất lợi cho setup/decoder/budget đang thử, không phải định lý mọi SemCom hoặc SB vô ích.

Thời gian 30–60 phút trong transcript là ước lượng chưa kiểm chứng. Nên smoke trên một đoạn ngắn để đo runtime trước khi hứa thời gian hoàn tất.

## 6. Literature và dataset

[GOSC](https://arxiv.org/abs/2609.39477) đã nghiên cứu value-based closed-loop scheduling. Vì vậy chỉ thắng random hoặc thêm feedback chưa xác nhận novelty của SwarmBridge.

[MCOP](https://arxiv.org/abs/2510.12679) là công trình có thật về multi-UAV occupancy, mở rộng nhiều dataset và dùng feature selection. Task 3D occupancy khác BEV segmentation hiện tại; nó là nguồn để kiểm tra dataset/baseline, chưa là lý do chuyển toàn bộ pipeline ngay.

[Trang CoPerception-UAV chính thức](https://siheng-chen.github.io/dataset/coperception-uav/) mô tả nhiều town/formation, nhưng ở phần đọc được chưa có điều khoản cấp phép tái phân phối dataset rõ ràng. Tìm trong các file text/metadata local cũng chưa thấy license. Đó là trạng thái chưa xác minh quyền redistributing, không phải kết luận vi phạm. Cần kiểm tra trước khi tiếp tục phân phối archive dataset công khai; không tự suy giấy phép code sang dataset.

## Khuyến nghị

Giữ hướng predictive communication làm nền. Chạy pilot temporal block selection sau khi chốt quyền truy cập state và các đối chứng trên. SB mechanism gate là phép kiểm tra riêng về conditional sampling; kết quả của một heuristic ensemble chưa thay thế nó. Không mở thêm Mamba, SB filter BEV hay RL trajectory trước khi câu hỏi hiện tại được đo đúng.
