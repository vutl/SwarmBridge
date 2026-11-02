**Kiểm toán nguyên nhân SB-decoder — 31/10/2026**

Mục tiêu là phân biệt lỗi triển khai công thức, khác biệt recipe so với paper, và tác động của việc phân bổ huấn luyện. Chẩn đoán vùng trước đây chỉ xác định nơi model thua, không tự xác định nguyên nhân. Pilot conditioning-only vì vậy chưa đủ để kết luận dừng toàn bộ SB-decoder.

**Những gì đã kiểm tra trực tiếp trong code**

1. Đối chiếu marginal VE của `swarmbridge/bridges/audit.py` với `SBNoiseScheduleVE` và các hệ số trong `SchroedingerBridgeAudioToAudioModel._step` của NeMo 2.0.0. Float32, 100 thời điểm: sai lệch tối đa hệ số clean/source 2,38e-7; variance 2,68e-7, bao gồm epsilon regularization của NeMo. Chưa tìm được lỗi công thức forward trong phạm vi này.
2. Kiểm tra điều khiển: cùng khởi tạo, batch, loss MSE+CE, AdamW, clipping và EMA; một model mapping và một SB ép 100% endpoint. Ở chế độ xác định, sau ba updates, prediction, gradient, raw weights và EMA khớp chính xác. Việc này xác nhận đường train endpoint của SB quy về mapping như mong đợi, không chứng minh mọi công thức/sampler đều đúng.
3. Kiểm tra số học thông thường có AMP: hai mapping giống hệt nhau cũng không cập nhật bit-for-bit giống nhau. Raw-weight max difference sau ba updates khoảng 0,00109, tương tự mapping–SB endpoint khoảng 0,00112; predictions đầu tiên giống hệt. PyTorch báo `nll_loss2d_forward_out_cuda_template` không có deterministic implementation cho CE reduction mặc định. Trong reference control, dùng CE unreduced rồi sum/valid-count với deterministic algorithms để kiểm tra cùng objective. Không sửa pipeline train đã chạy. Đây là nguồn dao động số học cần ghi nhận, chưa phải bằng chứng giải thích khoảng cách mIoU SB–mapping ba seed.
4. Code NeMo lấy thời gian train uniform trong khoảng liên tục; không có quy tắc bắt buộc dành 50% mẫu tại endpoint. Vì vậy thiếu endpoint có thể là recipe không hợp bài toán mất block, không tự là một bug vi phạm paper.
5. `mean_path_4` của audit là kernel mean bỏ noise, không phải NeMo PF-ODE. Sai lệch này ảnh hưởng diễn giải sampler nhiều bước; không giải thích riêng vì sao SB một bước thua mapping vì một bước trả trực tiếp predictor.

Nguồn gốc: các file NeMo giữ nguyên tại `pdf/GithubModels/NVIDIA__NeMo_audio_2.0.0/`, source snapshot Rethinking trong `pdf/GithubModels/Dahan-Wang__Rethinking-Flow-and-Diffusion-Bridge-Models-for-Speech-Enhancement/`. Dữ liệu kiểm tra nằm trong `runs/sb_training_root_audit/`.

**Điều chỉnh cách diễn đạt giả thuyết endpoint**

Hiện code bốc Bernoulli riêng cho mỗi mẫu. Với batch 32 và xác suất endpoint 0,1, trung bình có 3,2 mẫu endpoint/update; xác suất một update chứa ít nhất một mẫu endpoint là 1−0,9^32 = 96,57%. Do đó không đúng nếu nói chỉ 10% optimizer updates có luyện endpoint. Đúng là số ví dụ tại đúng endpoint chỉ bằng khoảng 1/10 mapping ở cùng số updates, trước khi tính khả năng các thời điểm gần endpoint hỗ trợ học.

Objective thực tế là `L_p = p L_endpoint + (1−p) L_intermediate`. Mapping tối ưu riêng `L_endpoint`. Các đầu vào trung gian chứa thông tin từ clean target là thiết kế của bridge, không phải rò rỉ nhãn test. Tuy nhiên, với mạng dùng chung hữu hạn, học cả họ nhiệm vụ này có thể giúp hoặc làm hại nhiệm vụ endpoint. Đó là điều cần kiểm định bằng can thiệp, không suy ra từ tên SB.

Ở t=1, một-step SB và mapping có cùng thông tin đầu vào và đích. Lợi ích nếu có phải đến từ cách huấn luyện, biểu diễn hoặc regularization; SB không tạo thêm bằng chứng về cảnh thật bằng cách thêm noise nội bộ.

**Loss và gradient đã đo trên checkpoint**

Phân tích dùng 32 ví dụ cố định lấy từ **training cache**, không dùng test và không cập nhật các checkpoint gốc. Chọn gửi 15%, cùng channel/mask/noise, seed 0. Các giá trị dưới không phải mIoU validation và không được so trực tiếp với bảng audit 200 lưới.

| Thời điểm đầu vào, checkpoint SB 30k | MSE latent | CE nhiệm vụ | mIoU trên batch chẩn đoán |
|---|---:|---:|---:|
| t=0,25 | 0,0522 | 0,0255 | 0,9145 |
| t=0,50 | 0,0996 | 0,0254 | 0,9157 |
| t=0,90 | 0,2466 | 0,0332 | 0,8961 |
| t=0,99 | 0,4947 | 0,1118 | 0,7705 |
| t=1 | 0,6676 | 0,2801 | 0,6080 |
| Mapping 30k tại t=1, cùng batch | 0,5947 | 0,1919 | 0,6645 |

SB giải tốt hơn ở các trạng thái còn mang clean signal và kém rõ ở endpoint. Nhưng đường cong này tự nó chưa chứng minh model chỉ học chép: khi t→1, lượng thông tin về phần thiếu cũng thật sự giảm. Ví dụ ở t=0,5, MSE vùng thiếu của phép copy chưa train là 0,3887, trong khi SB đạt 0,2008; model đã học hơn một phép sao chép đơn giản.

Đã đo gradient MSE+CE trên hai batch con 16 mẫu. Ở 30k, norm gradient endpoint khoảng 1,68–2,11, so với 0,059–0,103 của uniform interior. Ngay sau nhân trọng số p=0,1 và 1−p=0,9, norm thành phần endpoint vẫn lớn hơn interior khoảng 1,82–3,98 lần. Cosine giữa hai gradient dương khoảng 0,31–0,34 trong hai batch này. Chưa thấy bằng chứng endpoint bị lấn át về norm hoặc gradient xung đột âm ở các batch đã đo.

Đây là gradient thô tại checkpoint EMA, trên một tập nhỏ, không phải hướng cập nhật AdamW thực tế hay một kiểm định gradient toàn bộ tập train. Nó không bác bỏ lợi ích của việc tăng số ví dụ endpoint, nhưng không ủng hộ cách giải thích đơn giản rằng 90% interior luôn lấn át endpoint.

**Can thiệp cục bộ đã hoàn tất**

Hai nhánh khởi đầu từ đúng `last.pt` SB seed 0 tại 30k, cùng raw weights, AdamW state, EMA, RNG data và RNG bridge. Tiếp tục đúng 500 updates; chỉ đổi endpoint sample probability 0,1 vs 0,5. Tất cả cập nhật giữ input x_t-only. Không thay conditioning, backbone, loss hoặc learning rate. Đo raw và EMA ở update 30.500 cố định.

Đây là phép kiểm tra tác động của endpoint allocation tại checkpoint hiện tại. Không phải pilot 6k from scratch; không thể lấy kết quả này làm chiến thắng có ngân sách công bằng trước mapping 30k. EMA decay 0,999 giữ khoảng 60,6% ảnh hưởng trọng số EMA cũ sau 500 updates, nên cần báo cả raw và EMA thay vì chọn loại có số đẹp hơn.

Script: `scripts/probe_sb_endpoint_intervention.py`; checkpoint gốc giữ nguyên. Kết quả nhánh riêng trong `runs/sb_endpoint_intervention/`.

Kết quả một-step validation, 200 lưới, seed 0, mốc cuối cố định:

| Nhánh / weights | 15% gửi | 30% gửi | 50% gửi | 100% gửi |
|---|---:|---:|---:|---:|
| SB 30k ban đầu, EMA | 0,58743 | 0,68615 | 0,74717 | 0,83866 |
| Tiếp tục 500, 10% endpoint, EMA | 0,58742 | 0,68688 | 0,74828 | 0,83882 |
| Tiếp tục 500, 50% endpoint, EMA | 0,59060 | 0,68762 | 0,74924 | 0,83873 |
| Tiếp tục 500, 10% endpoint, raw | 0,56044 | 0,67654 | 0,74332 | 0,83774 |
| Tiếp tục 500, 50% endpoint, raw | 0,58707 | 0,68280 | 0,74652 | 0,83606 |

Raw trước can thiệp đạt 0,57168 ở 15%, cho thấy raw dao động đáng kể so EMA. 50% endpoint cải thiện EMA ở 15% khoảng +0,00318 so với tiếp tục 10%, tức +0,318 điểm phần trăm; raw tăng +0,02663 giữa hai nhánh. Không chọn chỉ raw để tuyên bố giải quyết khoảng cách: EMA vẫn cách mapping 30k seed 0 khoảng 0,03079, và mapping đó không được cấp thêm 500 updates trong probe này.

Ở 70/100%, IoU xe EMA của nhánh 10% là 0,47823/0,59450; nhánh 50% là 0,47785/0,59396. Trong probe ngắn này, tăng trọng tâm endpoint có tín hiệu cải thiện vùng truyền ít nhưng không cải thiện mọi tiêu chí. Đây là một can thiệp một seed, có dao động số học AMP; chưa đủ suy ra kết quả từ đầu 6k hay kết luận đây là nguyên nhân duy nhất.

Kết luận hiện tại: endpoint allocation có tác động có thể đo trong điều kiện đã thử, nên cần đưa vào pilot. Chưa có bằng chứng rằng chỉ tăng endpoint sẽ tự lấp toàn bộ gap, hoặc conditioning có thể bỏ qua. Probe đã kết thúc lúc khoảng 19:59 ngày 31/10/2026, Asia/Seoul.

**Sửa pilot để kiểm tra được hai giả thuyết**

Cập nhật quan trọng: đã tìm thêm mismatch phân phối corruption. Phần thiết kế giữ nguyên sampler cũ dưới đây vẫn có thể dùng để kiểm tra riêng conditioning/endpoint, nhưng không được coi là train/eval đã khớp. Nếu sửa sampler trước, phải train lại cả các baseline x-only; không dùng các checkpoint cũ làm đối chứng cùng recipe. Chi tiết và hệ quả ở phần cuối.

Ba model mới như đề xuất là hợp lý, nhưng mỗi SB cần đối chiếu với mapping có input tương ứng. Có thể dùng lại seed 0 / mốc 6k của hai model x_t-only đã lưu, vì cấu hình chung và RNG stream được ghi lại; không gộp điểm pilot một seed với mean ba seed.

| Nhánh | Model | Trạng thái |
|---|---|---|
| x_t-only | Mapping(z_received, t=1) | Đã có checkpoint 6k seed 0 |
| x_t-only | SB, 10% endpoint | Đã có checkpoint 6k seed 0 |
| x_t-only | SB, 50% endpoint | Model mới; so với hai dòng trên |
| Có điều kiện | Mapping([z_received,z_received,r], t=1) | Model mới |
| Có điều kiện | SB([x_t,z_received,r],t), 10% endpoint | Model mới; so với mapping ngay trên |

Nếu cần kiểm tra tương tác, bổ sung có điều kiện + 50% endpoint sau đó. Hai thay đổi riêng lẻ đều thất bại không loại được trường hợp cần đồng thời cả hai; dừng có thể là quyết định ngân sách thực dụng, nhưng không phải chứng minh đã tìm được root hoặc mọi SB đều thất bại.

Ngưỡng mở rộng đề xuất ghi theo đơn vị rõ: `mIoU_SB − mIoU_mapping_matched >= −0,005` tại **cả 15% và 30%**, và trung bình IoU xe tại 70/100% không thấp hơn mapping_matched quá 0,005. Đây là gate không kém hơn đáng kể theo mức sai khác thực dụng đã chọn, **không phải kiểm định thống kê equivalence** và không phải bằng chứng SB thắng. Mốc đối chiếu là cùng seed, cùng 6k, cùng input family; không lấy lợi thế xe ở checkpoint 30k làm chuẩn cho pilot 6k.

Qua gate mới cân nhắc đủ ba seed; để dùng SB làm đóng góp chính còn phải có lợi ích dương rõ về task/communication/compute. Chưa khởi chạy ba model pilot mới trong lượt kiểm toán này.

**Mismatch cụ thể trong cách tạo dữ liệu train/eval**

`BevData` train dùng `CachedLatents.batch` với `TRAIN_CFG`: toàn bộ latent trong footprint được đưa vào channel, erasure probability uniform 0–0,9. Trong khi đó, validation 15% chọn ngẫu nhiên 15% block trước, rồi qua channel với physical erasure probability uniform 0–0,6. Việc dùng erasure như proxy cho phần chưa gửi có thể hợp lệ, nhưng range và tần suất hiện tại chưa khớp chế độ vận hành khó.

Tỷ lệ nhận kỳ vọng ở eval 15% xấp xỉ `0,15 × (1−p_erase)`, tức 6–15%. Train có tỷ lệ nhận kỳ vọng 10–100%. Không gọi đây là hoàn toàn ngoài support của mọi mask: binomial erasure trong train vẫn có thể tạo các hiện thực nhận dưới 10%. Vấn đề được chứng minh là khác phân phối/range của tham số và mức tập trung vào trường hợp khó, không phải mọi mask eval đều bất khả trong train.

Mô phỏng CPU 20.000 grids trên footprint training cache, đúng kích thước block và cách làm tròn số block được chọn:

| Đại lượng | Train hiện tại | Eval chọn gửi 15% |
|---|---:|---:|
| Tỷ lệ diện tích footprint nhận được trung bình | 55,07% | 10,70% |
| Tỷ lệ grid nhận được không quá 15% | 6,28% | 87,70% |
| Tỷ lệ grid nhận được không quá 10% | 2,09% | 43,65% |

Với endpoint probability 0,1 độc lập, khoảng `6,28% × 10% = 0,628%` **mẫu** train SB vừa ở đúng endpoint vừa có coverage không quá 15%. Ở batch 32, 30k updates, đó là khoảng 6.029 ví dụ thuộc loại này. Mapping 6k đã có khoảng 12.058 ví dụ cùng loại. Các con số là kỳ vọng Monte Carlo, không phải đếm log mẫu thực tế và không tính khả năng chuyển giao từ các thời điểm trung gian.

Đây là một nguyên nhân setup đáng kiểm tra cùng endpoint allocation: học khó tại t=1 đã ít, lại còn ít gặp mức mất thông tin đang là trọng tâm đánh giá. Nó tác động lên cả mapping lẫn SB, nên chưa được dùng một mình để quy toàn bộ chênh SB–mapping cho distribution mismatch. Can thiệp 10%/50% ở trên mới chỉ giữ sampler cũ cố định.

Script: `scripts/audit_sb_coverage_distribution.py`; dữ liệu `runs/sb_training_root_audit/coverage_distribution.json`.

Đề xuất sampler khi chuyển sang recipe bám đúng giao tiếp: bốc sender budget từ các mức train đã quy định, chọn block trong footprint, rồi mới áp dụng physical channel; giữ nguyên sampler này cho mọi model đối chứng. Nếu áp dụng ngay, pilot đầy đủ cần train lại mapping x-only, SB x-only 10%, SB x-only 50%, mapping conditioned và SB conditioned 10% — năm model, không còn là ba model mới cộng hai baseline cũ. Giới hạn một seed/6k cho vòng đầu vẫn giữ, ước tính khoảng hai giờ theo tốc độ hiện tại, chưa khởi chạy. Có thể bổ sung conditioned 50% sau để kiểm tra tương tác.

Như vậy, phần đã định vị được là **sự không khớp giữa bài train được phân bổ và bài endpoint thiếu thông tin cần giải**, không phải một lỗi công thức VE đã được tìm thấy. Để xác định đóng góp của từng yếu tố cần đối chứng theo từng thay đổi; chưa có căn cứ gọi đây là root duy nhất.
