**SwarmBridge: nên làm gì với Schrödinger Bridge sau các kết quả Town3?**

Ngày rà soát: 30/10/2026, Asia/Seoul. Đây là bản phân tích nghiên cứu của Codex, đối chiếu nguồn gốc và code hiện tại; không phải kết quả tái lập đầy đủ các paper.

Cập nhật sau lần đọc lại toàn thư viện: xem [tổng hợp hướng nghiên cứu mới](swarmbridge-research-direction-2026-10-30.md). Bản mới xác minh GOSC, bổ sung conditional simulation/filtering và thu hẹp khuyến nghị ưu tiên SB cho belief; các khuyến nghị bên dưới ghi lại giai đoạn phân tích trước.

**Khuyến nghị của tôi là giữ SB như một giả thuyết có thể kiểm tra, hoàn tất kiểm toán nhỏ trước, rồi ưu tiên bài toán giá trị thông tin khi truyền nhiều vòng.** Không tiếp tục lấy việc tăng mIoU latent bằng cách đổi reference làm trục duy nhất. Mamba là một yếu tố kiến trúc cần đối chứng; RL là một lớp quyết định cần có môi trường chuyển trạng thái rõ. Ghép cả ba ngay bây giờ sẽ làm khó xác định nguyên nhân thành công hoặc thất bại.

Không có bằng chứng rằng sửa các điểm dưới đây chắc chắn giúp SB thắng. Ngược lại, cũng chưa đủ cơ sở để kết luận mọi cách dùng SB trong SwarmBridge đều không phù hợp. Cần tách ba câu: code tính đúng công thức đã chọn chưa; công thức có biểu diễn được đối tượng cần học không; và đối tượng đó có cải thiện quyết định truyền so với cách rẻ hơn không.

**Các kết quả hiện tại có giá trị, nhưng phạm vi hẹp hơn proposal gốc.**

| Điều đã kiểm tra | Kết quả / phạm vi |
|---|---|
| Dữ liệu Town3 | 340 mẫu; 1.100/200/400 lưới train/val/test. Teacher test mIoU 0,881. |
| Đối tượng phục hồi thực tế | Latent sạch của một UAV; `CachedLatents.N=1`. Không phải oracle latent hợp nhất nhiều nguồn. |
| Q1 cuối, severe, ba seed | Mapping 0,750; SB isotropic 0,748; empirical 0,750 ở chế độ tốt nhất trong các chế độ đã đo. |
| Q1b, severe | H1 AUROC trong bin r 0,766; SB khoảng 0,754–0,757. Đây là chỉ số phân biệt lỗi, không phải kiểm định calibration của posterior. |
| Q2 gửi hai vòng, hetero | 70% ngân sách quy ước: H1 chọn phần gửi thêm đạt 0,814; gửi đủ đạt 0,821. |
| Phần chưa có | Mamba; môi trường RL; điều khiển bay có phản hồi; đánh giá đầy đủ trên test cho các model khôi phục. |

Nguồn nội bộ: [báo cáo tiến độ](bao-cao-tien-do-2026-10-30.md), `runs/teacher/summary.json`, `runs/q1_b10_ema_s{0,1,2}.json`, `runs/q2p_s{0,1,2}.json`, `swarmbridge/data/real.py`. Các con số trong bảng là mô tả thực nghiệm, không phải chứng minh không có lợi ích ở mọi thiết lập.

Thí nghiệm decoder Claude nhắc tới nằm trong `swarmbridge/experiments/q2_progressive/decoders.py`: mapping/SB, mỗi loại có bản base và bản thêm task loss cùng residual; mặc định 6.000 bước. **Cập nhật sau khi nhận kết quả Claude:** đã xác nhận đủ 12 checkpoint và `runs/decoders.json` (ghi lúc 18:17 ngày 30/10). Đã đánh giá cả bridge một bước và bốn bước mean-only. Nó chưa phải audit VE/endpoint/Mamba.

| Tỉ lệ block chủ động gửi | Mapping nâng cấp mIoU | SB nâng cấp, 4 bước | SB nâng cấp, 1 bước |
|---:|---:|---:|---:|
| 15% | 0,600 | 0,527 | 0,550 |
| 30% | 0,693 | 0,664 | 0,669 |
| 50% | 0,754 | 0,744 | 0,743 |
| 70% | 0,794 | 0,791 | 0,790 |
| 100% | 0,837 | 0,840 | 0,838 |

Các số là mean ba seed trên val; tỉ lệ block được chọn gửi không bằng tỉ lệ thông tin nhận thành công vì vẫn có erasure/fading/noise. Ở 15%, chênh SB bốn bước so mapping là −0,0731 (−7,31 điểm phần trăm mIoU). Ở 100%, lợi thế mean chỉ +0,0028. IoU xe ở 15/50/100% lần lượt là mapping 0,161/0,396/0,589 và SB bốn bước 0,151/0,390/0,598. Không dùng sự cải thiện nhỏ khi gửi đủ để kết luận lợi ích ở chế độ cần tiết kiệm truyền.

Kết quả mới làm yếu thêm lý do dùng SB làm decoder chính của Town3. Đây là phép so với Mapping, không phải H1, nên lỗi gradient H1 không giải thích được thất bại này. Sai lệch sampler nhiều bước cũng không giải thích riêng được thất bại của dự đoán một bước, vì một bước trả thẳng output endpoint. Thêm task loss và residual là một gói hai thay đổi; bảng chưa tách được đóng góp riêng của mỗi thay đổi.

Khuyến nghị vận hành cập nhật: lấy `map_up` làm baseline decoder chính cho thí nghiệm tiếp theo; chưa có cơ sở gọi nó là tốt nhất trên test hoặc mọi mức truyền. Không đổi nhãn uncertainty của H1 thành của mapping này: `map_up` hiện chưa có variance head. Hướng belief/value-of-information bên dưới vẫn là giả thuyết dự phòng, chưa có bằng chứng dương cho SB. Nếu đầu tư tiếp, kiểm định phân phối và utility trước khi mở rộng RL hoặc huấn luyện nhiều cấu hình dài.

Một audit endpoint nhỏ nên gồm mạng bridge chỉ train tại t=1 làm sanity check (bài toán hồi quy endpoint với input bị lặp), rồi thêm t trung gian có kiểm soát và giữ một phần mẫu endpoint. Sanity check này không phải thành tích SB; nó kiểm tra pipeline và hiệu ứng huấn luyện trung gian. Giữ input/backbone/loss/budget rõ ràng; chỉ đổi sang VE sau khi baseline Brownian được kiểm tra. Dữ liệu hiện không chứng minh 30k bước là đủ hay cần. Tái lập một benchmark ảnh với đúng protocol là bằng chứng hẹp về implementation; chỉ thấy ảnh sắc hơn không chứng minh code hoặc posterior đúng.

**Trước tiên phải phân biệt các nghĩa của “dùng SB”.**

| Cách dùng | Đích thực sự | Hệ quả với SwarmBridge |
|---|---|---|
| SB tối ưu vận chuyển phân phối | Tìm cả coupling và quá trình tối ưu giữa các phân phối biên | Ghép cặp latent sạch/hỏng rồi hồi quy chưa tự động giải bài toán này. |
| Bridge có cặp, có điều kiện | Khôi phục hoặc sinh đầu ra đúng với quan sát cụ thể | Gần code hiện tại; điều kiện đầu vào và sampler phải bảo toàn ý nghĩa của cặp. |
| SB training, suy luận một bước | Học một hàm dự đoán nhờ các trạng thái trung gian | Có thể cải thiện tối ưu hóa, nhưng đầu ra một bước xác định không tự tạo posterior đa mẫu. |
| SB dịch latent giữa endpoint khác nhau | Nối biểu diễn của hai encoder/decoder | Hợp với heterogeneous swarm, song vẫn cần thắng adapter trực tiếp và diffusion. |
| SB điều khiển vật lý / sinh hành động | Phân phối vị trí, vận tốc hoặc chuỗi hành động | Có thể liên hệ RL, nhưng phải xây động lực và tiêu chí riêng. |

[Augmented Bridge Matching](https://arxiv.org/html/2311.06978v1) phân biệt việc giữ các marginal với giữ coupling của cặp dữ liệu. Điều kiện theo điểm xuất phát có lý do toán học; không nên bỏ `z_p` khỏi mọi bridge chỉ vì lo mạng học lối tắt. Cần so biến thể và kiểm tra đúng đại lượng cần giữ.

**Kiểm toán đã tìm được một lỗi baseline cụ thể và một vấn đề sampler cụ thể.**

1. H1 chưa thực hiện đầy đủ faithful heteroscedastic regression. `models.py` chặn gradient qua `mu.detach()`, nhưng `logvar` vẫn xuất phát từ cùng U-Net. NLL phương sai do đó vẫn cập nhật trunk. Paper yêu cầu chặn cả nhánh ảnh hưởng này vào tham số dùng chung. [Phương pháp gốc, mục 2 và phương trình 5](https://arxiv.org/html/2212.09184).

   Kiểm tra CPU trên model hiện tại: norm gradient từ riêng NLL phương sai vào `net.inc.weight` là **0,226710**, không bằng 0. Vì vậy không được dùng bảo đảm “phương sai không thể làm giảm độ chính xác mean” cho H1 hiện tại. Điều này không chứng minh toàn bộ khoảng cách H1–M1 do lỗi này; cũng không dự đoán chiều thay đổi của AUROC sau khi sửa.

2. `stochastic=False` hiện là cập nhật mean của kernel hậu nghiệm, bỏ phần ngẫu nhiên. Đó là một heuristic xác định, không phải tự động là probability-flow ODE của bridge đang có nhiễu. Công thức PF-ODE trong [DDBM](https://arxiv.org/abs/2309.16948) và sampler ở [Jukić, Table 2](https://www.isca-archive.org/interspeech_2024/jukic24_interspeech.pdf) có các hệ số khác. Code I²SB cũng có `ot_ode`, nhưng đó là lựa chọn OT-ODE riêng; không đồng nghĩa mọi thao tác tắt nhiễu lúc test đều là PF-ODE. [Code I²SB](https://github.com/NVlabs/I2SB/blob/master/i2sb/diffusion.py).

   Phản ví dụ độc lập: với hai endpoint cố định, q=1, đi từ t=0,75 tới s=0,5. Phương sai marginal cần là 0,25; cập nhật mean-only cho phương sai 0,08333. Kết quả “ODE” cũ vẫn mô tả đúng heuristic đã chạy; tên và diễn giải về bảo toàn phân phối cần sửa. Không có bảo đảm PF-ODE đúng sẽ cho mIoU cao hơn.

**Quan trọng hơn: test bridge hiện tại chưa đủ để xác nhận uncertainty.** Test cũ cho mạng biết chính xác từng target `z_star`. Nó xác nhận công thức bridge giữa hai điểm đã biết, nhưng mạng thật chỉ có thể học ước lượng có điều kiện của target. Hai tình huống này khác nhau khi target còn bất định.

Tôi bổ sung thí nghiệm CPU với đáp án Gaussian giải tích. Cho posterior `X0|y ~ N(y/2, 0,5)`, dùng chính sampler hiện tại và thay U-Net bằng hàm Bayes chính xác `E[X0|Xt,y]`. Không có lỗi học, không train, không chạy GPU. Phương sai đầu ra tính bằng đệ quy giải tích:

| q | NFE | Phương sai posterior đúng | Phương sai sampler hiện tại |
|---:|---:|---:|---:|
| 1 | 1 | 0,500 | 0,000 |
| 1 | 4 | 0,500 | 0,2153 |
| 1 | 16 | 0,500 | 0,3690 |
| 1 | 64 | 0,500 | 0,4487 |
| 1 | 256 | 0,500 | 0,4821 |
| 10 | 4 | 0,500 | 0,0610 |

Monte Carlo 32.768 mẫu khớp gần với đệ quy. Đây là hiện tượng thiếu độ phân tán ở số bước hữu hạn của plug-in sampler, không phải bằng chứng mọi bridge sai. Nó giải thích vì sao cần đo chất lượng posterior trước khi lấy phương sai mẫu làm tín hiệu quyết định. K=8 mẫu chỉ tăng độ chính xác ước lượng phân tán của sampler đó; nó không tự sửa bias do NFE=4. Kết quả Gaussian không được chuyển nguyên xi sang Town3.

Mã tái chạy: [audit_sb_assumptions.py](../../../scripts/audit_sb_assumptions.py). Số liệu: [sb-audit-checks-2026-10-30.json](../../metadata/sb-audit-checks-2026-10-30.json).

```powershell
.venv\Scripts\python.exe -m scripts.audit_sb_assumptions
```

**Một bước dự đoán và một phân phối dự đoán là hai sản phẩm khác nhau.**

Suy luận sau đây là phân tích của tôi cho objective đang dùng, không phải định lý về mọi SB. Nếu chỉ chấm MSE latent, nghiệm Bayes ở endpoint là `m(y,c)=E[z*|y,c]`. Mapping có cùng thông tin đầu vào và cùng loss cũng hướng tới nghiệm đó. Với mô hình liên tục theo t và giới hạn endpoint thích hợp, bridge data-prediction một bước không có một nghiệm MSE Bayes khác tốt hơn. Lợi ích có thể đến từ cách huấn luyện, regularization, backbone hoặc tối ưu hóa hữu hạn. Đây vẫn có thể là lợi ích đáng nghiên cứu, nhưng cần đo đúng.

Ngược lại, nếu cần phân phối nhiều khả năng để tính rủi ro hoặc giá trị của một quan sát mới, một output mean duy nhất không đủ. One-step SBM xác định cần thêm một cơ chế riêng nếu muốn uncertainty: ensemble, variance head hoặc mô hình sinh ngẫu nhiên được kiểm định. Không thể đồng thời mặc định “một forward xác định” và “độ phân tán posterior qua nhiều lần chạy”.

Không được suy từ định lý mean tối ưu MSE sang “mean latent chắc chắn tối ưu mIoU”. Decoder phi tuyến và metric theo lớp làm hai bài toán khác nhau. Tương tự, nhiều mode có thể là lý do để cần phân phối, nhưng không bảo đảm lấy một mẫu sẽ cải thiện mIoU.

Thêm task loss cũng thay đổi cách diễn giải về phân phối. Với MSE thuần, data predictor tối ưu là conditional mean cần cho các công thức score tương ứng. Thêm cross-entropy qua decoder nói chung làm nghiệm lệch khỏi conditional mean đó. Đây có thể là adaptation tốt cho segmentation, nhưng không còn được tự động dùng bảo đảm posterior của objective gốc. Vì vậy nên tách nhánh decoder tối ưu task và nhánh belief cần calibration; nếu dùng chung phải kiểm định lại phân phối sau fine-tuning. Nhận xét này không phủ nhận kết quả task-loss của các paper speech; nó giới hạn kết luận toán học có thể chuyển sang scheduler của ta.

**Các điểm khác cũng giới hạn cách diễn giải báo cáo.**

- `Reference.fit()` đo `E[(z*−z_p)^2 | bin(r)]`, không trừ mean residual. Đây là second moment; chỉ bằng variance khi mean residual có điều kiện bằng 0. Nó cũng không chứa tương quan giữa các vị trí/kênh. Không thể gọi nó là toàn bộ posterior corruption geometry.
- `1−r` có ý nghĩa theo prior và mô hình quan sát đã giả định. Chuẩn hóa latent trên toàn tập train chưa bảo đảm phương sai và mean có điều kiện đúng tại mọi vị trí. Test dữ liệu giả không chứng minh calibration trên latent thật.
- AUROC trong năm bin r là một probe hữu ích, nhưng bin rộng còn giữ biến thiên r bên trong; nó không chứng minh độc lập có điều kiện hoặc calibration. Cần thêm risk–coverage, reliability và so dự báo lỗi bằng metadata đầy đủ với metadata cộng tín hiệu bridge.
- Script aggregate dùng ngưỡng 2 standard errors với ba seed. Không nên diễn giải thành một kiểm định 95% chuẩn với mẫu lớn. Theo giả định paired t-test thông thường, ba hiệu seed chỉ cho df=2; ngưỡng hai phía 95% khoảng 4,303. Nhiều phép so sánh và lựa chọn chế độ tốt nhất còn tạo thêm chọn lọc. Đây là nhận xét thống kê, không phải tính lại p-value cho mọi bảng.
- Những cấu hình đã chỉnh trên val không thể dùng chính val để khẳng định tổng quát hóa cuối cùng. Town3 chỉ có hai scene val; frame và các UAV cùng scene có tương quan.
- `z_p + sqrt(1−r)·net` là cách tham số hóa residual mềm. Nó không cưỡng chế data consistency với quan sát nhiễu, cũng không phải phép chiếu range–null trong mọi trường hợp.
- “Oracle” Q2 đang xếp theo residual latent bình phương. Nó không tính marginal gain của hành động sau truyền và khôi phục, nên không phải upper bound bảo đảm cho task utility. Vòng hai cũng có thể chọn lại block vòng một; phải phân biệt gửi mới và retransmission.
- “70% bit” hiện là 70% lượt block theo cách đếm chung. Digital cần lượng tử hóa/coding thực; analog cần channel uses và power constraint. Chi phí request, thời gian inference và trễ vòng hai cần đi vào quyết định cuối.

**Nguồn mới làm thay đổi đánh giá tính mới, chứ không chỉ thêm danh sách đọc.**

| Nguồn gốc đã đối chiếu | Điều có thể dùng làm bằng chứng | Điều không được suy ra |
|---|---|---|
| [I²SB và code chính thức](https://github.com/NVlabs/I2SB) | Có tùy chọn condition `x1`; recipe chuẩn sử dụng pretrained ADM và đánh giá classifier accuracy/FID. | Thêm `z_p` không tự động là sai; thắng ảnh restoration không bảo đảm thắng Town3. |
| [SBM, Yang et al.](https://arxiv.org/html/2510.16834v2) | VE, loss phổ nhiều độ phân giải, suy luận endpoint một bước; có ablation mapping/SB trên nhiều backbone. | Chưa chứng minh một cơ chế phổ quát “Mamba bằng động lực SB”. Table 3 chủ yếu dùng chỉ số chất lượng speech, không phải MSE/segmentation. |
| [Jukić 2024](https://www.isca-archive.org/interspeech_2024/jukic24_interspeech.pdf) | Có loss phụ miền tín hiệu và phân biệt sampler SDE/ODE rõ ràng. | Không được sao chép mức nhiễu tuyệt đối giữa phổ speech và latent BEV khác chuẩn hóa. |
| [Rethinking, AAAI 2026](https://arxiv.org/html/2602.18355) | Phân tích predictive behavior; cải tiến loss, backbone và fine-tuning; một bước gần predictive baseline mạnh. | Không chứng minh mọi bridge đều vô ích, cũng không chứng minh bridge phải thắng ở mọi metric. |
| [DBIM, ICLR 2025](https://arxiv.org/html/2405.15885) | Sampler non-Markovian, bảo toàn marginal và xử lý khởi tạo ngẫu nhiên cho bridge endpoint. | Không thể thay công thức sampler bằng thao tác bỏ nhiễu rồi giữ nguyên mọi tuyên bố posterior. |
| [BridgeCodec](https://arxiv.org/html/2610.15453) | Dịch giữa hai codec cố định; huấn luyện latent rồi thêm loss qua decoder. | Nó không giải đúng bài toán một encoder sạch/hỏng của ta; không chứng minh mọi adapter trực tiếp đều thua. |
| [SBGSC](https://arxiv.org/html/2604.17802) | SB đã có trong generative semantic communication. | “SB + SemCom” không đủ làm tuyên bố tính mới. |
| [PRISM](https://arxiv.org/html/2608.06893) | Reference, schedule và số bước tương tác; lý thuyết theo mode Gaussian có giới hạn trên ảnh thật. | R2 diagonal theo bin r không tương đương toàn bộ phổ posterior; không có bảo đảm matched reference luôn thắng. |
| [DiffCP](https://arxiv.org/html/2409.19592) | Diffusion tái tạo feature cộng tác từ message nén, có điều kiện hình học và semantic. | Khôi phục BEV feature bằng mô hình sinh đã là prior art. |
| [GenComm](https://arxiv.org/html/2510.19618) | Conditional diffusion nối semantic space của agent dị thể. | Heterogeneous translation cũng cần khác biệt cụ thể, không chỉ đổi diffusion thành SB. |
| [LR-V2X, preprint 10/2026](https://arxiv.org/html/2610.11264) | Compact latent bị mất gói → prior BEV → reconstructor có ego context; có task fine-tuning. | Bài gần trực tiếp với hướng recovery; nó không được gọi là SB solver nếu objective không giải SB. |
| [Confidence-calibrated multi-UAV resource allocation, 7/2026](https://link.springer.com/article/10.1186/s13638-026-02655-0) | Confidence calibration, novelty và scheduling semantic/radio đã được nghiên cứu cho UAV. | “Uncertainty + UAV scheduling” không còn là mô tả đủ đặc thù cho đóng góp. |
| [Adaptive cross-modal UAV SemCom, 8/2026](https://link.springer.com/article/10.1186/s13638-026-02664-z) | Chọn modality/granularity bằng recurrent PPO, power bằng SAC trong MDP. | Thêm PPO không tự tạo tính mới hoặc hợp thức hóa một MDP thiếu dynamics. |
| [Significance-Driven Semantic Communication](https://arxiv.org/html/2608.28441) | Task-specific data value và phân bổ kênh theo restless bandit. | “Value of information + communication” không phải ý mới tự thân. |
| [Value of Communication: A Pareto Analysis](https://arxiv.org/html/2512.01454) | Có phân tích trade-off và marginal value trong mô hình Markov cụ thể. | Không được bê bảo đảm tối ưu của mô hình đó sang mạng perception của ta. |
| [Gao et al., bản tác giả và GLOBECOM 2020](https://ww3.math.ucla.edu/camreport/cam19-57.pdf) | Đã đặt điều khiển vận tốc bầy UAV thành bài toán SB/mean-field game. [Bản ghi hội nghị của Princeton](https://collaborate.princeton.edu/en/publications/energy-efficient-velocity-control-for-massive-numbers-of-rotary-w-2/) xác nhận năm 2020. | “SB chưa từng dùng cho UAV” là sai ở nghĩa tổng quát; hướng điều khiển cũng cần rà novelty. |
| [UAV3D, trang tác giả](https://huiyegit.github.io/UAV3D_Benchmark/) | Có benchmark single/cooperative 3D perception và dữ liệu đồng bộ. | Chưa đo overlap và utility trên phần dữ liệu định dùng thì chưa thể khẳng định nó giải nút thắt Town3. |

Giới hạn truy cập: đã đọc bài và một số file code I²SB/DBIM. Demo SBM truy cập được nhưng chưa xác minh được một repo huấn luyện chính thức đủ để tái lập nguyên bản. Đường dẫn NeMo qua connector không trả về file đọc được; nhận định sampler dựa trên công thức paper và code I²SB đã đọc, không phải tuyên bố đã audit NeMo. Không có thí nghiệm paper-scale nào được tái lập trong lần nghiên cứu này. Không có kết luận chắc chắn rằng một tổ hợp cụ thể chưa từng xuất hiện ở mọi venue.

**Đọc lại đề xuất của Claude: phần kiểm toán nên làm, nhưng cần bỏ những kết luận quá mạnh.**

| Nhận định / đề xuất | Đánh giá của tôi |
|---|---|
| Cùng U-Net nhưng training khác nhau | Đúng; SB không phải tên backbone. |
| Có thể khác recipe SBM | Đúng, cần đối chiếu schedule, representation, input, objective và endpoint. |
| Đổi VE, bỏ condition, thêm endpoint training và train 30k là “tái lập trung thành” | Đây là một gói adaptation. Nên có nhánh bám paper và nhánh adaptation riêng; thay nhiều yếu tố cùng lúc khó quy nguyên nhân. |
| Làm vậy vẫn thua thì biết chắc vấn đề ở bài toán, không ở code | Quá mạnh. Một recipe và một data regime không loại trừ mọi lỗi hoặc lựa chọn khác. |
| Sinh ảnh sắc hơn là chứng minh code đúng | Không đủ. Cần kiểm định moment, coupling, conditional coverage và so implementation tham chiếu. |
| SB giữ xe tốt hơn H1 là dấu hiệu tốt | Là tín hiệu cần theo dõi. H1 hiện có vấn đề gradient; phải so mapping mạnh và kiểm tra false positives. |
| Nhiễu lớn tốt là dấu hiệu SB không chính danh | Chỉ là giả thuyết. Scale, learning và discretization đều có thể ảnh hưởng; q không phải máy kiểm định “SB thật”. |
| Bọc Gymnasium rồi PPO là xong RL | Chỉ là khung phần mềm. Cần hành động hợp lệ, observation khả dụng, transition, horizon, chi phí và benchmark. |
| Không cần simulator 3D khi chỉ chọn message trên dữ liệu cố định | Hợp lý trong phạm vi offline communication. Không dùng kết quả đó để claim tối ưu đường bay hoặc sensing do action gây ra. |

**Nếu phải chọn một hướng nghiên cứu chính, tôi chọn bridge cho belief có điều kiện phục vụ giá trị thông tin.** Đây là đề xuất để kiểm nghiệm, không phải đóng góp đã được chứng minh mới hoặc hiệu quả.

Ví dụ trạm chưa biết một vùng có xe hay không. Một UAV nhìn rõ vùng này nhưng kênh hơi yếu; UAV khác có kênh tốt nhưng chỉ nhìn nền. Xếp hạng theo kênh chọn sai người. Xếp hạng theo variance cũng có thể chọn một vùng vốn không thể quan sát tốt. Một belief có cấu trúc có thể giúp dự báo: yêu cầu message nào sẽ làm giảm rủi ro nhiệm vụ nhiều nhất, sau khi tính khả năng nhận được và chi phí?

Đối tượng cần học là `b_t = p(z_task | các message đã nhận, mask, channel, pose, task)`. Việc lấy mẫu chỉ có ý nghĩa nếu các mẫu là các khả năng đúng có điều kiện với bằng chứng. Mô hình không được dùng prior để thay thế bằng chứng mà không phản ánh sự bất định đó.

Định nghĩa đề xuất:

`R(b) = min_d E[L_task(d,Z) | b]`

`V(a|b) = R(b) − E_{M_a|b,channel}[R(update(b,M_a,a))] − lambda * C(a)`.

Ở đây a có thể là xin block/ROI của UAV j, xin gửi lại, đổi độ phân giải hoặc dừng. Mức bất định hiện tại chỉ là một đầu vào. Phần quyết định là rủi ro nào có thể giảm nhờ hành động. Chi phí C phải bao gồm truyền, request và compute/latency phù hợp.

Bridge cung cấp belief hoặc các đặc trưng cho mô hình value; scheduler chọn action. Để dự đoán message chưa nhận, còn cần mô hình quan sát/helper hoặc value predictor học từ intervention. Sample variance của z* không tự cho ra phân phối message của mọi helper. Đây là một phần còn thiếu trong cả proposal lẫn code, không được che bằng ký hiệu kỳ vọng.

Trong giai đoạn đầu, có thể đo utility khi thật sự cung cấp từng candidate message trong dữ liệu offline, tạo nhãn marginal task gain rồi học value predictor. Bridge chỉ được giữ vai trò chính nếu nó giúp dự báo utility hoặc risk–cost tốt hơn mô hình không bridge. Baseline gồm mapping + uncertainty, ensemble, conditional Gaussian/mixture, conditional diffusion và flow được condition công bằng. Không chỉ so với r hoặc random.

**Tôi xếp các nhánh như sau.**

| Nhánh | Mức nên đầu tư hiện tại | Phép kiểm tra quyết định |
|---|---|---|
| Reference theo r để tăng latent mIoU | Giữ làm ablation, giảm ưu tiên | Sửa audit trước; chỉ mở rộng nếu có cơ chế/metric rõ vượt S1 và augmentation. |
| One-step task-aware SB, CNN/Mamba | Thí nghiệm chẩn đoán có giới hạn | Thiết kế 2×2; cùng input, task loss, data split, tuning và latency; không lấy CNN âm để loại Mamba trước. |
| Belief + marginal communication utility | Hướng ưu tiên có điều kiện | Posterior phải qua test; value ranking và policy phải cải thiện trên test với tổng cost bằng nhau. |
| Dịch latent dị thể | Nhánh thay thế hợp lý | Hai encoder cố định thật sự khác nhau; adapter và GenComm-style baseline; kiểm tra task fidelity, không chỉ MSE. |
| SB cho bay / action trajectories | Một đề tài riêng nếu chọn control | Dynamics, goal distribution, constraints và baseline MPC/RL/assignment phải rõ. Chưa gắn thêm vào Town3 hiện tại. |

Không có bước nào đòi SB phải thắng vì ta thích nó. Nhưng cũng không loại nó chỉ bởi một so sánh trên output mean không kiểm tra được điểm mạnh của phân phối.

**Thứ tự thí nghiệm nên được khóa trước khi dùng thêm GPU.**

| Cổng | Việc làm | Kết quả cho phép đi tiếp |
|---|---|---|
| A: công thức và baseline | Sửa faithful head/trunk; đặt tên sampler cũ rõ; kiểm tra sampler tham chiếu trên Gaussian với conditional-mean estimator; tách regression và sampling metrics. | Sai số moment/coupling được giải thích; test không chỉ dùng oracle biết target. |
| B: kiểm tra cơ chế trên bài toán có đáp án | Gaussian rồi một bài toán nhỏ hai mode, với observation/helper likelihood biết trước; so mẫu và calibration qua nhiều budget NFE. | Không collapse về mean khi cần hai mode; sample frequency và expected value gần đáp án biết trước. Đây là bằng chứng cơ chế, chưa là UAV result. |
| C: Town3 có kiểm soát | Dùng mapping mạnh với task loss, bridge Brownian/VE, cùng input và tuning; thí nghiệm one-step riêng với stochastic sampler; chọn config trên val rồi đánh giá test. | Có convergence curves và sai số theo scene; biết khác biệt do schedule, condition hay loss. |
| D: utility thực tế | Trên dữ liệu offline, can thiệp thêm message để đo marginal task gain; huấn luyện value predictor; so greedy/Bandit với các score r/U. | Xếp đúng helper/ROI hơn và giảm tổng cost ở cùng risk/recall, không chỉ AUROC lỗi hiện tại. |
| E: Mamba và RL | 2×2 backbone/training nếu dữ liệu và latency cho phép; RL khi có chuỗi quyết định và tác động tương lai cần tối ưu. | Lợi ích còn sau đối chứng cùng backbone, budget và thông tin đầu vào. |

Đây là cổng phụ thuộc bằng chứng, không phải lời hứa “1–2 ngày sẽ giải quyết xong”. Có thể dùng checkpoint cũ để đọc learning curves và đo thêm trước khi train. Chỉ tăng ngân sách khi còn dấu hiệu chưa hội tụ hoặc giả thuyết cụ thể cần phân biệt. So cùng số bước là một kiểm soát, nhưng không tự chứng minh cả hai phương pháp đã được tối ưu công bằng.

Tiêu chí định lượng nên chọn theo nhiệm vụ trước khi mở test: ví dụ xe nhỏ recall ở cùng false-positive rate, hoặc mức giảm số channel uses để đạt một risk mục tiêu với khoảng tin cậy theo scene. Nếu chọn ngưỡng nội bộ như +1 điểm IoU xe hoặc giảm 10% cost, phải ghi đó là mục tiêu engineering của dự án, không phải chuẩn do paper quy định. “Không có ý nghĩa thống kê” không được đổi thành “chắc chắn bằng nhau”; muốn tuyên bố tương đương cần biên tương đương đặt trước và đủ dữ liệu.

**Thiết kế RL có thể bắt đầu nhỏ mà vẫn trung thực về phạm vi.**

Một episode có thể là một cảnh với 2–4 vòng truyền. Observation gồm message đã nhận, mask, tóm tắt belief, CSI ước lượng và ngân sách còn lại. Action chọn UAV, ROI, mức chất lượng hoặc STOP. Transition áp mô hình kênh rồi cập nhật belief. Reward huấn luyện có thể dùng nhãn offline để tính task loss giảm trừ chi phí; policy triển khai không được thấy nhãn đó.

Nếu mọi lựa chọn độc lập theo frame và không gây hệ quả tương lai, hãy bắt đầu bằng supervised value ranking, greedy hoặc contextual bandit. Nếu quyết định thay đổi ngân sách, belief và cơ hội các vòng sau, có finite-horizon MDP thực sự để so RL. Policy có hàng nghìn block-action cần masking/cấu trúc, không chỉ đặt PPO lên vector lớn. Đặt budget/latency là constraint hoặc penalty có giải thích; đo cả wall-clock, không quy NFE thành độ trễ chính xác.

Nếu action thay đổi đường bay, phải có transition của vị trí, camera, channel và quan sát tương lai. Dữ liệu AirSim/CARLA đã ghi sẵn là dữ liệu mô phỏng, không phải một closed-loop flight environment ta có thể tự do điều khiển. Có thể dùng mô phỏng động học nhẹ ở giai đoạn đầu, nhưng phải chứng minh mô hình quan sát/channel phản ứng hợp lý với action. Không cần cài simulator 3D chỉ để làm đẹp pipeline.

**Việc nên làm ngay là sửa cách đặt câu hỏi trước khi mở rộng phương pháp.** Câu hỏi có ích hơn “SB có thắng U-Net không?” là: “Với cùng bằng chứng và tổng chi phí, một bridge được kiểm định về phân phối có giúp biết lúc nào cần hỏi thêm và hỏi thông tin nào để giảm lỗi nhiệm vụ không?” Nếu có, SB có vai trò khoa học rõ. Nếu không, dữ liệu đó chỉ ra giới hạn của SB trong cấu hình đã kiểm tra; không cần ép thêm Mamba hoặc PPO để giữ tên phương pháp.

Trong lần làm việc này đã xóa duy nhất file PDF `(1)` trùng SHA-256 `C02231541274B4C32170680913DEBEC96D519A0BF9F5A066D2094D72AEAC1143`; bản gốc được giữ. Đã thêm script chẩn đoán CPU và JSON kết quả; chưa sửa các model/sampler huấn luyện hoặc chạy thêm một đợt train dài.
