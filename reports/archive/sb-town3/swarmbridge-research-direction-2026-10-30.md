**SwarmBridge sau khi đọc lại thư viện và các nguồn Deep Research**

Ngày 30/10/2026, Asia/Seoul. Bản này cập nhật khuyến nghị sau kết quả decoder của Claude. Đây là phân tích nghiên cứu, không phải kết quả train mới hay cam kết tính mới.

**Quyết định tôi đề xuất: giữ bài toán giao tiếp theo giá trị bằng chứng; dùng mapping làm decoder chính hiện tại; kiểm tra SB trong vai trò conditional inference bằng một thí nghiệm cơ chế nhỏ trước khi cấp ngân sách train dài.** Việc cần phân biệt tiếp là lợi ích của mô hình hóa phân phối đối với quyết định truyền. Không lấy thêm Mamba, VE, PPO hoặc số bước train làm lý do tự thân để mở rộng.

Tôi cần điều chỉnh cả lời khuyên trước của mình. “SB cho belief rồi chọn message” có lý do khoa học, nhưng quá rộng để làm đóng góp mới. Hướng đó cũng chưa có bằng chứng dương trong code hiện tại. Kết quả decoder âm không loại nó, nhưng việc chưa bị loại cũng không đủ để ưu tiên đầu tư lớn.

**Phạm vi đọc và truy nguồn**

Thư viện lúc bắt đầu có 67 PDF, 1.420 trang: 62 paper và 5 bản đề xuất/Deep Research do AI tổng hợp. Đã tạo cache toàn văn có số trang; rà phạm vi của toàn bộ, đọc lại các phần phương pháp/kết quả then chốt, đối chiếu với năm file reports người dùng nêu và lịch sử Claude mới gửi. Mức đọc từng tài liệu được ghi trong [bản đồ bằng chứng](../tai-lieu/literature-evidence-map-2026-10-30.md). Không tuyên bố đã kiểm chứng mọi chứng minh hoặc tái lập mọi benchmark.

Đã lần theo các nguồn trong Deep Research, đọc thêm các nhánh gần proposal và tìm nền tảng conditional inference còn thiếu. Tám paper bổ sung đã lưu vào pdf/; danh mục và checksum ở [manifest bổ sung](../../metadata/bo-sung-nguon-nghien-cuu-2026-10-30.json). Không dùng các bản tổng hợp AI như nguồn primary để khẳng định novelty.

**Điều dữ liệu hiện tại cho phép kết luận**

Nguồn thực nghiệm là `runs/decoders.json`, `runs/q1_b10_ema_s{0,1,2}.json`, code trong `swarmbridge/data/real.py`, `bridges/models.py` và `experiments/q2_progressive/`.

| Câu hỏi | Bằng chứng | Kết luận phù hợp |
|---|---|---|
| Decoder SB hiện tại có thắng mapping khi gửi ít? | 15%: map_up 0,600; SB4 0,527; SB1 0,550. 50%: 0,754 / 0,744 / 0,743. | Không; cấu hình hiện tại bất lợi ở chế độ tiết kiệm truyền. |
| SB giữ xe tốt hơn một cách rõ ràng? | So cùng nâng cấp: IoU xe 15% là 0,161 / 0,151; 50% là 0,396 / 0,390; 100% là 0,589 / 0,598. | Tín hiệu nhỏ khi gửi đủ không chứng minh lợi thế ở chế độ thiếu bằng chứng. |
| Lỗi H1 có giải thích bảng decoder? | Bảng dùng Mapping, không dùng H1 làm đối thủ. | Không. Lỗi H1 phải sửa riêng, không dùng để phủ nhận thất bại SB1/SB4. |
| Sai lệch stochastic sampler có giải thích SB1 thua? | Một bước trả trực tiếp dự đoán endpoint. | Không phải lời giải thích đầy đủ cho thất bại một bước. |
| Đã kiểm tra original swarm-completion? | Mỗi vùng chỉ có một nguồn, `N=1`; target là latent sạch của chính nguồn đó. | Chưa kiểm tra bổ sung bằng chứng từ helper nhìn cùng vùng. |
| Đã có kết quả cuối trên test cho decoder? | `CachedLatents` trong script mặc định dùng val. | Chưa. Không mở test nhiều lần để chọn recipe. |

Tỉ lệ gửi là tỉ lệ block chủ động chọn truyền, không phải tỉ lệ thông tin nhận thành công: vẫn có noise/fading/erasure. Task loss và residual được thêm cùng lúc, nên chưa thể tách đóng góp riêng.

Tính thử paired-t interval trên ba seed giúp định cỡ bất định, không thay thế đánh giá theo scene. SB4−mapping tại 15%: −0,0731, interval minh họa [−0,1183; −0,0278]. Tại 100%: +0,00284, interval [−0,00200; +0,00768]. Giả định t-test với ba seed rất hạn chế, chưa hiệu chỉnh nhiều phép so, và không chứa bất định do chọn scene. Không diễn giải bảng thành định lý SB vô ích.

**Một điều kiện nền trong kế hoạch cũ cần sửa**

Kế hoạch cũ đặt điều kiện “SB phải khôi phục tốt hơn mapping” trước khi nghiên cứu quyết định truyền. Điều kiện đó hợp lý nếu contribution là decoder tốt hơn. Nó không phải điều kiện toán học bắt buộc cho một contribution về belief.

Với MSE, mean có điều kiện là nghiệm Bayes. Một mẫu độc lập từ posterior đúng có expected squared error bằng hai lần trace covariance, còn conditional mean chỉ chịu một lần trace covariance. Điều này giả sử target thật và mẫu sinh độc lập có điều kiện theo cùng bằng chứng. Vì vậy mô hình sinh có thể có posterior đúng nhưng thua một point estimator về MSE. Lập luận này không tự chuyển thành định lý về mIoU qua decoder phi tuyến.

Thứ cần đánh giá ở nhánh belief là chất lượng phân phối và quyết định do phân phối đó tạo ra, cùng tổng cost. Đổi cách chấm sau khi thấy kết quả âm chỉ hợp lệ nếu ghi rõ đây là một câu hỏi nghiên cứu mới, có protocol/baseline mới; không dùng nó để đổi tên thất bại cũ thành thành công.

Mặt khác, một mapping đủ mạnh từ toàn bộ lịch sử sang action-value cũng có thể thực hiện quyết định tốt mà không biểu diễn posterior tường minh. Có lý do dùng belief không đồng nghĩa có lý do bắt buộc dùng SB.

**Literature làm thay đổi cách đặt vấn đề thế nào**

| Nhóm nguồn | Điều ảnh hưởng trực tiếp tới quyết định của ta |
|---|---|
| [CodeFilling](../../../pdf/Communication-Efficient%20Collaborative%20Perception%20via%20Information%20Filling%20with%20Codebook.pdf), [CoSDH](https://arxiv.org/html/2503.03430) | Receiver demand, sender supply và tránh redundancy đã có. Bộ chọn phải vượt các cơ chế này nếu claim cooperative perception. |
| [CoGMP](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Generative_Map_Priors_for_Collaborative_BEV_Semantic_Segmentation_CVPR_2025_paper.pdf), [GenComm](https://arxiv.org/html/2510.19618), [DiffCP](https://arxiv.org/html/2409.19592), [LR-V2X](https://arxiv.org/html/2610.11264) | Generative fusion, heterogeneous translation và packet-loss recovery đều có prior art gần. Thay tên sampler chưa đủ. |
| [SComCP](https://arxiv.org/html/2507.00895), [CoDS](https://arxiv.org/html/2512.22513), [Generative Feature Imputing](https://arxiv.org/html/2508.17957) | Channel uses, quantization/coding, packetization và reliability đã là phần của hệ thống. Không dùng “8 bit/value” để suy kết quả digital deployment. |
| [GOSC](https://arxiv.org/html/2609.39477) | Đã có vòng kín chọn nội dung/thời điểm/độ bảo vệ dựa trên value và common knowledge. “Nối loop” không còn đủ đặc thù. |
| [SBM](https://arxiv.org/html/2510.16834v2), [Jukić](https://www.isca-archive.org/interspeech_2024/jukic24_interspeech.pdf), [Rethinking](https://arxiv.org/html/2602.18355) | Speech cho thấy recipe, loss và backbone quan trọng; không chứng minh bridge phải thắng cùng latent/task của ta. |
| [CDSB, UAI 2022](https://proceedings.mlr.press/v180/shi22a.html), [Reich, Data Assimilation](https://arxiv.org/abs/1807.08351) | Có nền trực tiếp cho posterior sampling/filtering từ quan sát thiếu/nhiễu. Đây là nhánh cần đọc nếu SB dùng để cập nhật belief. |
| [BridgeTwist](https://arxiv.org/html/2610.18658v1) | Có hướng reuse SB prior với likelihood và SMC correction; không cần mặc định mọi thay đổi quan sát đều phải train một translator mới. |

GOSC hiện đã xác minh được; lời ghi trong log cũ rằng chưa tìm ra bài không còn đúng. Đây là preprint dùng grid missions và planner cụ thể; không phải kết quả BEV của ta. Nhưng sự tồn tại của nó đủ để loại claim “chưa ai làm closed-loop semantic value”. Phần limitations của bài cũng nói rõ chưa đối chứng mọi hệ thống công bố.

CoSDH đặc biệt đáng chú ý vì tách vùng nền quan sát rõ với vùng không quan sát được. Entropy/foreground confidence không tự phân biệt được hai trường hợp đó. Đây là đối chứng tốt cho cách ta dùng uncertainty.

CoGMP gần ở BEV segmentation, nhưng metric road/lane và task/map priors khác vehicle IoU của Town3. Không đem số IoU của hai hệ thống so trực tiếp. CoDS dùng reliability từ LLR của bộ giải mã; đó là bất định về truyền đúng feature, khác bất định về nội dung cảnh.

**SB thực ra đang được dùng theo ba nghĩa khác nhau**

1. **Paired data bridge / training objective:** học predictor trên các điểm trung gian giữa latent sạch và hỏng. Code hiện tại nằm gần nhánh này.
2. **Conditional inference / filtering:** sinh hoặc cập nhật các khả năng của trạng thái chưa biết từ lịch sử quan sát. Đây là nhánh có thể phục vụ action-value.
3. **Stochastic control:** điều khiển phân bố vị trí/vận tốc/quỹ đạo vật lý. Đây là một bài toán khác.

Không chuyển bảo đảm giữa ba nghĩa. Noise của internal bridge-time không tự là noise hay thời gian của kênh vô tuyến. Từ “belief latent” cũng cần thận trọng: một tensor teacher cố định chưa phải một phân phối xác suất đã calibrated.

Nếu chạy EOT giữa hai tập latent không condition theo scene, solver có thể ghép một cảnh với cảnh khác để giảm transport cost. Nó có thể giải đúng một bài toán OT mà làm sai bài toán giữ bằng chứng. Vì vậy “đổi sang SB chính danh” không tự sửa task correspondence.

**Một nhận định trong báo cáo Claude về GSBM cần thu hẹp**

Không đúng khi nói mọi biến thể SB có paired data đều quy về công thức Brownian hiện tại. Trong [GSBM](https://arxiv.org/abs/2310.02233), state cost thay đổi đường đi có điều kiện; Lemma 3 cho ví dụ quadratic cost khác Brownian bridge dù endpoints được cố định. Hai marginal cố định cũng không cố định toàn bộ coupling.

Tuy nhiên, điều đó chưa tạo lý do dùng GSBM cho code của ta. Nếu trạng thái trung gian chỉ là latent tưởng tượng và không có tác dụng vật lý/compute được đo, giảm một running cost tùy ý chưa giải thích được tiết kiệm truyền. Phải chỉ ra state cost ứng với rủi ro, thời gian hay chi phí nào có thể kiểm chứng. “Có thêm task cost nên chắc hơn” không đủ.

Feedback SBM dùng aligned pairs để hướng dẫn semi-supervised transport; chữ feedback ở đó không phải packet phản hồi của mạng. Multi-marginal SB xử lý nhiều marginal snapshots; thêm nhiều round nhận message chưa tự biến causal filtering thành bài toán đó.

**Phân tích thêm: vì sao sampler có thể thiếu variance ngay cả với predictor hoàn hảo**

Đây là suy luận bằng luật total variance, đã kiểm tra trên Gaussian giải tích; không phải claim thuật toán mới.

Trong code, khi đi từ t tới s<t, đặt a=s/t và predictor m_t=E[X0|Xt,y]. Kernel biết chính xác X0 có mean a Xt+(1−a)X0, variance q s(t−s)/t. Khi X0 còn bất định, reverse kernel đúng là mixture qua p(X0|Xt,y). Do đó:

`E[Xs|Xt,y] = a Xt + (1−a) m_t`

`Cov(Xs|Xt,y) = q s(t−s)/t I + (1−a)^2 Cov(X0|Xt,y)`.

Plugin sampler hiện tại giữ term đầu, thay X0 bằng mean và bỏ term covariance thứ hai. Ở bước cuối s=0, term bị bỏ là toàn bộ bất định endpoint còn lại. Với nhiều bước nhỏ, sai lệch có thể giảm; ở vài bước nó có thể lớn. Đây là một xấp xỉ thường gặp, không tự động là lỗi đánh máy của code.

| Posterior Gaussian có variance 0,5 | NFE | Plugin hiện tại | Reverse mixture Gaussian đầy đủ |
|---|---:|---:|---:|
| q=1 | 1 | 0,0000 | 0,5000 |
| q=1 | 4 | 0,2153 | 0,5000 |
| q=1 | 16 | 0,3690 | 0,5000 |
| q=10 | 4 | 0,0610 | 0,5000 |

Kết quả mới khớp thí nghiệm audit trước. Mã [analyze_sb_decision_mechanism.py](../../../scripts/analyze_sb_decision_mechanism.py), số liệu [sb-decision-mechanism-2026-10-30.json](../../metadata/sb-decision-mechanism-2026-10-30.json).

Covariance bổ sung là giải pháp đúng cho toy Gaussian đã biết likelihood/posterior. Với posterior nhiều mode, matching mean/covariance vẫn chưa tái tạo đúng mixture. Không lấy bảng này làm bản vá sampler BEV tổng quát. Cũng không dùng nó giải thích vì sao SB một bước thua mapping về mIoU.

BridgeTwist có một hướng khác: importance correction cho posterior từ prior đã học. Bảo đảm của nó nhắm learned/discretized model, không xóa sai số prior; cần score và velocity nên chưa cắm thẳng vào clean-prediction network hiện tại. Preprint này là nguồn tham khảo cơ chế, chưa là lựa chọn triển khai.

**Điểm có thể làm cho phân phối đáng giá: quan hệ giữa những khả năng, không chỉ một bản đồ variance**

Ví dụ giải tích của tôi: hai ô nhị phân đều có xác suất 0,5 và variance 0,25. Task loss là tổng số ô phân loại sai.

| Belief thật | Risk trước truyền | Message cho biết chính xác ô 1 | Expected risk giảm |
|---|---:|---|---:|
| Hai ô độc lập | 1,0 | Chỉ giúp ô 1 | 0,5 |
| Hai ô luôn bằng nhau | 1,0 | Biết ô 1 thì biết cả ô 2 | 1,0 |

Cùng marginal variance, khác message value. Mô hình joint distribution có thể biểu diễn thông tin mà score variance độc lập bỏ mất. Nhưng mixture, structured probabilistic model hoặc một value predictor trực tiếp cũng có thể làm được; ví dụ này chưa chứng minh SB cần thiết.

Một ví dụ khác: vùng A có p=0,5, variance 0,25 nhưng helper chỉ có xác suất 0,05 cung cấp quan sát hữu ích; vùng B có p=0,1, variance 0,09 nhưng helper nhìn rõ chắc chắn. Với cùng cost và thông điệp tiết lộ nhãn chính xác khi hữu ích, expected giảm risk là A=0,025 và B=0,1. Xếp theo uncertainty sẽ chọn A, xếp theo expected utility chọn B. Đây là ví dụ lý tưởng hóa, không phải mô hình channel/perception đã đo trên UAV.

Vì vậy, nếu mở lại nhánh SB, câu hỏi nên là: **với bằng chứng tương quan hoặc nhiều giả thuyết hợp lý, việc biểu diễn joint conditional uncertainty có giúp xếp hạng hành động hỏi thêm tốt hơn mô hình rẻ hơn, sau khi tính compute/latency?**

**Một formulation đủ cụ thể để biết model phải học gì**

Gọi S_t là trạng thái nhiệm vụ, H_t là lịch sử khả dụng ở receiver, gồm message đã nhận, mask/CRC, CSI ước lượng, pose/timestamp và các phản hồi đã trao đổi. Belief là b_t=p(S_t|H_t). Action a=(helper, ROI, chế độ truyền, mức bảo vệ), hoặc STOP.

Muốn tính giá trị của message chưa nhận còn cần mô hình p(M_a|S_t,pose,quality,channel) hoặc một value predictor được học từ intervention. Posterior của cảnh không tự cho biết camera của từng helper nhìn được gì. Receiver không được xem trước feature chưa truyền miễn phí.

Một target đơn giản có tính thời gian là:

`V(a|H_t) = E[L(d_no_message(t+tau_a), S_(t+tau_a)) − L(d_with_message(t+tau_a), S_(t+tau_a)) | H_t] − lambda C(a)`.

Hai nhánh counterfactual phải chấm cùng thời điểm t+tau_a. Nếu chấm baseline ở hiện tại và message sau một khoảng trễ thì target sẽ lẫn lợi ích thông tin với sự thay đổi của thế giới. Với chuỗi quyết định, dùng common terminal horizon hoặc cumulative task risk; không giả định greedy một bước luôn tối ưu.

Đây là định nghĩa nghiên cứu đề xuất của tôi. GOSC/PASC/AoII là related work, nên phần mới phải được chứng minh ở estimator, protocol, giới hạn hay regime cụ thể. Chưa được viết như một invention của “value of information”.

Để lấy nhãn offline, giữ chung scene và kịch bản nhiễu, thực sự cung cấp từng candidate message cho decoder/fusion rồi đo thay đổi task loss. Dùng common random numbers khi so lựa chọn để giảm phương sai Monte Carlo. Nhãn này có thể dùng GT trong training; policy deploy không có GT. Oracle utility phải là kết quả can thiệp thật, không gọi residual latent bình phương là upper bound task.

Target bước đầu nên là class-weighted task loss hoặc local vehicle risk có định nghĩa rõ; mIoU/IoU xe vẫn là metric đánh giá. IoU toàn tập là tỷ số gộp, không tự phân rã thành tổng utility từng block. Không hứa tổng score từng block bằng mIoU gain của cả tập.

**Đánh giá thông tin đầu vào trước khi mở rộng model**

Với Town3 hiện tại, có thể làm bài phân bổ ngân sách giữa các vùng UAV gần như rời nhau. Đây vẫn là bài multi-UAV communication hợp lệ. Nó khác bài helper nhìn vùng mà ego bị che. Không ép mô tả sau lên thiết lập trước.

Nếu muốn quay lại original collaborative perception, cần xác nhận ba điều:

1. Có quan sát đồng bộ, transform/visibility/GT thích hợp cho cùng ROI.
2. Helper thật sự có thông tin bổ sung: đo task gain khi thêm clean helper, bao gồm negative transfer, thay vì chỉ đo footprint overlap.
3. Có fusion teacher/baseline được huấn luyện và đánh giá. Warp rồi cộng latent của encoder một nguồn không tự tạo “full-information oracle”.

Có thể kiểm tra trên một phần nhỏ benchmark phù hợp trước khi tải cả dataset. UAV3D/CoPerception dynamic là ứng viên cần đo, không phải lời bảo đảm chỉ vì tên dataset có nhiều UAV. Trong trường hợp chưa có dữ liệu phù hợp, toy multi-view dùng likelihood biết trước đủ để kiểm tra cơ chế, nhưng chưa đủ claim thực nghiệm UAV.

Nếu tiếp tục với video hiện có, staleness cần timestamp, alignment và task evolution thật. Chỉ trộn latent với hệ số tương quan tùy chọn không xác nhận được future task value trên video.

**Làm giao tiếp thực tế hơn mà không dựng cả radio stack ngay**

Phải chọn rõ một nhánh:

- Digital: quantization/index thực, header/mask/pose/timestamp, coding/modulation hoặc BLER table được hiệu chuẩn theo MCS/blocklength, CRC/drop/retransmission và feedback.
- Analog JSCC: số real/complex channel uses và power constraint, không gán tùy ý 8 bit cho một giá trị float.

Cho prototype digital, uniform quantization dùng range học từ train là bước đầu đủ kiểm soát; chưa cần VQ phức tạp. Xác minh dequantized clean-channel performance trước khi thêm loss/coding. Nếu dùng BLER lookup thay modem, ghi rõ đó là abstraction và đối chiếu một vài operating points.

Ví dụ kế toán chi phí, không phải số liệu đã đo: với 144 block và một code index 10 bit/block, payload là 1.440 bit; một demand bitmap 144 bit đã bằng 10% payload trước header. Với 16×4×4 giá trị/block ở 8 bit/value, payload lớn hơn nhiều và tỉ trọng feedback nhỏ hơn. Vì vậy không thể vừa đổi sang codebook rất nén vừa mặc định feedback luôn không đáng kể.

Độ trễ cần gồm request/metadata, chờ lịch MAC, truyền, decode và inference. Chọn TDMA hay song song sẽ đổi cách cộng thời gian. NFE là số lần gọi mạng, không phải phép đo wall-clock; K×NFE có thể batch nhưng vẫn tiêu thụ memory/compute. So scheduling phải dùng cùng tổng cost và thông tin sẵn có.

Nơi có candidate mới và retransmission, tách hai loại hành động. Nếu action chỉ được gửi block chưa gửi thì phải mask block đã chọn. Nếu cho gửi lại thì phải mô hình hóa combining/channel evolution và đếm lại cost.

**Kế hoạch tiếp theo theo bằng chứng, không theo tên thuật toán**

| Cổng | Việc cụ thể | Điều quyết định |
|---|---|---|
| 0. Khóa phạm vi dữ liệu | Giữ sector scheduling trên Town3 hoặc kiểm chứng một subset có complementary views; xác định clean-helper gain. | Không có gain thì không có bài helper-filling trên subset đó. |
| 1. Kiểm tra inference nhỏ | Gaussian và một posterior hai mode có likelihood biết trước; so Bayes oracle, Gaussian/mixture, conditional flow/diffusion và bridge phù hợp. | Moment, mode mass, conditional coverage và action regret phải có ý nghĩa ở budget thực tế. |
| 2. Tạo target utility | Dùng map_up hoặc fusion baseline cố định, thêm candidate message thật và đo task gain; split theo scene. | Score phải dự đoán gain/action ranking, không chỉ lỗi hiện tại. |
| 3. Protocol có cost | Payload/metadata/feedback/coding/compute; đánh giá risk–cost và deadline. | Giảm cost tại cùng chất lượng hoặc cải thiện risk tại cùng cost, với sai số theo scene. |
| 4. Kiểm tra vai trò SB | Thêm belief model đã qua cổng 1 vào cùng selector/protocol và so value predictor trực tiếp cùng dữ liệu. | Chỉ giữ SB nếu nó tạo lợi ích quyết định sau khi tính chi phí. |
| 5. Sequential policy | Chạy greedy và finite-horizon baselines; dùng RL khi quan sát/hành động/budget thực sự có tác dụng tương lai. | PPO/DT/MARL phải thắng đối chứng phù hợp, không phải chỉ random. |

Không cần triển khai tất cả baseline lớn đồng thời. Bắt đầu oracle + Gaussian/mixture + direct value predictor; thêm conditional flow/diffusion nếu chúng là đối thủ thực chất cho giả thuyết phân phối.

Nếu mục tiêu trước mắt vẫn là kiểm recipe một bước trên Town3, endpoint-only training với cùng mạng là sanity check tốt, rồi mới thêm t trung gian có kiểm soát. Khi t1_frac=1, bài toán gần hồi quy có input lặp, chưa phải bằng chứng lợi ích bridge. Khóa loss, input, scheduler và budget; xem learning curves rồi mới tăng train. Checkpoint hiện tại là state_dict cuối/EMA, không có sẵn mọi learning curve hay optimizer state để “resume đúng như chưa bị dừng”.

Một audit theo paper cần khớp protocol, không chỉ dựng ảnh nhỏ và thấy sắc nét. Benchmark EOT phù hợp khi xác minh optimal coupling; benchmark conditional posterior phù hợp khi xác minh uncertainty; task benchmark phù hợp khi xác minh giao tiếp. Không dùng một kiểm tra thay cả ba.

**RL/DT/Mamba nên xuất hiện ở đâu**

Với static scene, reward từng round là giảm task loss trừ cost; nếu không discount và cùng một loss/target, tổng reward telescopes thành loss đầu trừ loss cuối trừ tổng cost. Discount hoặc scene chuyển động sẽ đổi objective, cần định nghĩa lại. PPO không tự giữ hard budget: action mask/termination hoặc constrained formulation phải thực hiện điều đó.

CDT đã xử lý điều kiện reward/cost; DT wireless và drone đã có. Thay RTG không bảo đảm generalize sang preference ngoài support dataset. Nếu làm offline RL, phải có offline trajectories đủ cover action và so BC/CQL/IQL thích hợp; chưa cần làm ngay khi không có decision dataset.

Mamba cần kiểm tra bằng thiết kế backbone × training objective. Thắng do backbone không chứng minh SB, và U-Net âm cũng không chứng minh Mamba âm. Đây là ablation về sau, không phải ưu tiên trước cổng dữ liệu/utility.

Repo DeepJSCC và UAV path planning hữu ích để học pipeline; UavNetSim hữu ích cho packet/MAC/mobility. Chúng không tự tạo camera observations/fusion teacher đồng bộ. Có thể học cách chạy chúng riêng mà không nhập tất cả vào SwarmBridge.

**Hướng SB nào còn đáng cân nhắc sau khi rà lại?**

| Vai trò | Đánh giá hiện tại | Điều kiện mở lại |
|---|---|---|
| Decoder khôi phục một nguồn | Hạ ưu tiên; map_up là lựa chọn thực dụng hiện tại. | Có giả thuyết recipe cụ thể và bằng chứng hội tụ/metric rõ. |
| Joint conditional inference / filtering | Có nền toán học phù hợp hơn với evidence update; còn rủi ro compute/calibration. | Vượt mixture/direct-value baselines về action utility ở cùng cost. |
| Cross-encoder / codec translation | Hợp với endpoint thực sự khác nhau. | Có deployment need thật, thắng adapter/GenComm-style baseline; không dùng để tránh kết quả âm. |
| Physical swarm control | SB có formulation tự nhiên nhưng là đề tài riêng. | Dynamics, reference policy, mục tiêu phân phối và baseline assignment/MPC/RL rõ. |

Trong nhánh conditional inference, CDSB đã có filtering demo và code; ví dụ Lorenz dùng basis regression, không phải một neural BEV model train một lần là chạy. Reich cung cấp nền coupling/filtering. Đây là nơi nên hiểu bài toán trước khi chọn kiến trúc. Không có claim “first SB for Bayesian filtering” hay “first SB in belief space”.

Các paper physical steering, kinetic swarming và MAPF tạo cơ sở cho một đề tài control khác; không có sẵn bảo đảm collision/connectivity cho cấu hình 5 UAV của ta. Không gắn thêm trajectory control vào perception chỉ để giữ chữ swarm.

**Khuyến nghị chốt cho dự án**

Giữ `map_up` để tiến hành phần đo giá trị thông tin. Trước đợt train dài tiếp theo, xác nhận setup có loại bằng chứng cần nghiên cứu và tạo được target utility hợp lệ. Cho SB một thử nghiệm conditional inference nhỏ có đáp án biết trước, vì nó trả lời trực tiếp câu hỏi uncertainty đang bị bỏ ngỏ.

Nếu posterior đúng hơn nhưng decision không tốt hơn hoặc cost quá lớn, dùng model đơn giản hơn. Nếu direct value predictor đã làm tốt, không bắt buộc xây generative belief. Nếu SB giúp trong một regime có tương quan/đa giả thuyết, mô tả đúng regime đó và cơ chế giúp; khi ấy mới có lý do mở rộng.

Phần đáng giữ của ý tưởng ban đầu là quyết định truyền dựa trên bằng chứng còn thiếu đối với nhiệm vụ. SB là một ứng viên giải quyết một phần bài toán đó. Bằng chứng hiện tại chưa cho phép đảo ngược thứ tự này.
