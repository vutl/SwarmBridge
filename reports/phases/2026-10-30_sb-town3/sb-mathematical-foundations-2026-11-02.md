# SB có ưu thế toán học gì, và điều đó dẫn tới ứng dụng nào?

Ngày 02/11/2026. Nguồn là các công trình lý thuyết gốc được liên kết bên dưới, cộng với các suy diễn và phản ví dụ tự xây cho thiết kế SwarmBridge. Không chạy GPU, không thay model/checkpoint. Các lập luận dưới đây phân biệt nghiệm lý tưởng với thuật toán học hữu hạn.

**Kết luận:** không có thứ tự tổng quát “SB tốt hơn mapping/diffusion/filtering khác”. SB tối ưu một bài toán xác định trên phân phối quỹ đạo. Có bảo đảm về nghiệm, năng lượng điều khiển và một số sai số phân phối dưới giả thiết cụ thể. Muốn suy ra lợi ích giao tiếp, phải nối những bảo đảm đó với posterior đúng, loss nhiệm vụ và quyết định truyền. Mối nối này chưa được thiết lập trong các thí nghiệm hiện tại.

## 1. SB tối ưu chính xác đại lượng gì?

Cho R là luật xác suất tham chiếu trên quỹ đạo X_τ, τ∈[0,1]. Bài toán cổ điển:

\[
P^*=\arg\min_{P:P_0=\mu_0,\;P_1=\mu_1}\operatorname{KL}(P\|R).
\]

Với nghiệm hữu hạn và các điều kiện tồn tại thích hợp, tính lồi nghiêm ngặt cho tính duy nhất. Phân rã KL:

\[
\operatorname{KL}(P\|R)=\operatorname{KL}(P_{01}\|R_{01})+
\mathbb E_{P_{01}}\operatorname{KL}(P^{x_0,x_1}\|R^{x_0,x_1}).
\]

Do đó nghiệm giữ các bridge có điều kiện của R và tối ưu coupling hai đầu. Nguồn: [Léonard, Proposition 2.3 và phần tồn tại](https://leonard.perso.math.cnrs.fr/papers/2014-Leonard-A%20survey%20of%20the%20Schroedinger%20problem%20and%20some%20of%20its%20connections%20with%20optimal%20transport.pdf).

Điều này bảo đảm **ít thay đổi giả định R nhất để thỏa ràng buộc**, không tự bảo đảm đúng từng cặp scene, mIoU cao hay ít bit. Nếu R là một giả định sai về cảnh, giải chính xác bài toán vẫn có thể cho inference sai.

### Dạng điều khiển và ý nghĩa của reference

Với cùng phân phối đầu và cùng ma trận khuếch tán khả nghịch σ, xét:

\[
R:\ dX_\tau=b_\tau(X_\tau)d\tau+\sigma_\tau dW_\tau,
\quad
P:\ dX_\tau=(b_\tau+v_\tau)(X_\tau)d\tau+\sigma_\tau dW_\tau.
\]

Dưới điều kiện Girsanov thích hợp, đặt a=σσᵀ:

\[
\operatorname{KL}(P\|R)
=\frac12\mathbb E_P\int_0^1 v_\tau^\top a_\tau^{-1}v_\tau\,d\tau.
\]

Nếu phân phối đầu khác, cần thêm KL đầu. SB tối thiểu hóa chi phí điều khiển này trong tập thỏa endpoint. Nguồn: [Chen–Georgiou–Pavon, mục 4–5](https://arxiv.org/pdf/2005.10963).

**Suy luận cho Q1:** chọn a theo CSI thay đổi cách phạt các hướng hiệu chỉnh. Nó là thiết kế regularizer/prior, chưa phải chứng minh reference “wireless-aware” cải thiện task. Thay a cũng thay bài toán; không được so hai giá trị KL đối với hai R khác nhau rồi kết luận reference nào tốt hơn cho nhận thức. Thay diffusion coefficient còn có thể làm các luật quỹ đạo liên tục suy biến tương hỗ, nên các phép so Girsanov cần kiểm tra tính tương thích.

## 2. Một phản ví dụ: SB tối ưu nhưng ghép sai hoàn toàn ý nghĩa

Đây là ví dụ tự xây, không phải số liệu Town3.

Cho X₀ và X₁ đều nhận ±1 với xác suất 1/2. Cặp dữ liệu thật có X₁=−X₀. Reference có xác suất giữ dấu 0.9 và đổi dấu 0.1:

\[
R_{01}=\begin{pmatrix}.45&.05\\.05&.45\end{pmatrix},
\qquad
\pi_{\rm true}=\begin{pmatrix}0&.5\\.5&0\end{pmatrix}.
\]

R đã thỏa hai marginal, nên nghiệm SB chính xác là P*=R với KL=0. Nhưng khi chấm lỗi khôi phục theo cặp thật:

\[
\mathbb E_{P^*}(X_1-(-X_0))^2=3.6,
\qquad
\mathbb E_{\pi_{\rm true}}(X_1-(-X_0))^2=0.
\]

**Hai phân phối biên không xác định được correspondence đúng.** Vì thế nối phân phối latent frame trước với phân phối latent frame sau, hoặc nối latent hỏng với sạch, chưa đủ để suy ra đúng scene của người dùng. Cần điều kiện/coupling đúng từ các cặp dữ liệu, mô hình động lực hay likelihood. Đây là vấn đề nhận dạng bài toán, không phải chỉ vấn đề số bước train.

Nếu bridge được xây riêng cho từng điều kiện y, từ δ_y tới p(x|y), coupling đầu/cuối có thể đã được xác định bởi điều kiện. Không được dùng phản ví dụ trên để phủ nhận mọi paired/conditional bridge. Vấn đề là phải nói rõ bài toán nào đang được giải.

## 3. Với MSE, dự đoán mean có một bảo đảm mà lấy mẫu không có

Phần này là suy diễn trực tiếp cho mục tiêu ước lượng của ta. Giả sử X có moment bậc hai hữu hạn; H chứa toàn bộ thông tin bộ thu thực sự có. Đặt m(H)=E[X|H]. Với mọi hàm dự đoán điểm f(H):

\[
\mathbb E\|X-f(H)\|^2
=\underbrace{\mathbb E\|X-m(H)\|^2}_{D_{\rm Bayes}}
+\mathbb E\|m(H)-f(H)\|^2.
\]

Chứng minh: khai triển bình phương và dùng E[X−m(H)|H]=0 để triệt tích chéo. Mean là nghiệm tối ưu MSE trên mọi estimator cùng thông tin H.

Nếu X' là một mẫu độc lập có điều kiện từ **posterior đúng** p(X|H), thì:

\[
\mathbb E\|X-X'\|^2=2D_{\rm Bayes}.
\]

Với trung bình K mẫu posterior độc lập có điều kiện, rủi ro là (1+1/K)D_Bayes. Tạo phân phối đúng và đoán đúng trạng thái thật là hai mục tiêu khác nhau. Ví dụ X=±1 đồng xác suất, không có quan sát: mean=0 đạt MSE=1; lấy một mẫu posterior đạt MSE=2.

Giới hạn của kết quả:

- Không chứng minh mapping hữu hạn đã đạt Bayes. SB-training vẫn có thể cho một estimator tốt hơn nhờ inductive bias/tối ưu hóa hữu hạn; đây là chỗ có thể xuất hiện lợi ích kiểu SBM.
- Không suy ra mean latent tối ưu mIoU. Decoder phi tuyến, CE và mIoU là những mục tiêu khác nhau. MSE+CE trong code hiện tại cũng không có nghiệm đúng bằng conditional mean nói chung.
- Nếu chỉ cần xác suất nhãn, một classifier được train bằng CE có thể học p(label|H) trực tiếp; không bắt buộc phải sinh toàn bộ latent posterior.

Kết quả về [perception–distortion của Blau–Michaeli](https://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf) là một bối cảnh liên quan: khớp phân phối đầu ra và giảm distortion có thể xung đột ở biên tối ưu. Không có nghĩa mọi model sinh luôn thua về task, hoặc một model chưa tối ưu không thể cải thiện đồng thời nhiều metric.

## 4. Vậy SB có kết quả “thắng” toán học nào không?

Có, nhưng cần ghi đúng đối tượng so sánh.

### Hội tụ về một nghiệm phân phối xác định

[DSBM, Theorem 8](https://arxiv.org/html/2303.16852v3) chứng minh hội tụ KL của các bước IMF lý tưởng dưới các giả thiết của định lý. Paper phân biệt bridge matching với phép lặp giải SB. Một lần regression trên các bridge từ coupling cho trước không tự mang toàn bộ bảo đảm đó. Kết quả hội tụ không nói 6k AdamW updates với mạng hữu hạn tốt hơn mapping trên test.

### Reference có điều kiện cho bài toán posterior

[CDSB, Proposition 3 và mục 5](https://proceedings.mlr.press/v180/shi22a/shi22a.pdf) cho các iterate IPF chính xác một chặn dạng:

\[
\mathbb E_Y\operatorname{KL}(q_n(\cdot|Y)\|p(\cdot|Y))
\le\frac{2C_R}{n},
\qquad C_R=\mathbb E_Y\operatorname{KL}(P_Y^*\|R_Y).
\]

Ở đây q_n là marginal sinh ra được chỉ rõ trong Proposition 3; n là vòng IPF, không phải optimizer updates. Paper cho phép reference khởi tạo từ xấp xỉ posterior bằng mạng/EnKF. Hằng số nhỏ hơn làm chặn chặt hơn, chưa bảo đảm tốc độ thực tế; sai số học và rời rạc chưa tự biến mất.

### Một so sánh tối ưu thực sự trong họ diffusion

Preprint [Chen–Vanden-Eijnden 2026, Theorem 3.2, 3.6, 4.1](https://arxiv.org/html/2602.10989v1) tối ưu diffusion coefficient trong một họ có cùng time marginals và estimator drift nền. Nghiệm liên hệ Föllmer process, tối thiểu hóa KL quỹ đạo giữa quá trình chính xác và xấp xỉ trên khoảng cắt endpoint. Sau tuning, biểu thức KL có tính bất biến theo schedule khi so tương ứng score errors.

Đây là ưu thế có điều kiện về độ nhạy với sai số ước lượng; chưa là bảo đảm mIoU, NFE=1 hay latent có điều kiện của ta. Là preprint; cần thẩm định giả thiết trước khi chuyển công thức sang code.

### Thời gian sinh hữu hạn

[DSB 2021](https://arxiv.org/abs/2106.01357) dùng SB để nối chính xác hai marginal trong thời gian hữu hạn ở mức bài toán lý tưởng, giảm nhu cầu đợi noising hội tụ về reference. Thời gian SDE hữu hạn không đồng nghĩa ít FLOPs hay ít training: vẫn cần học, rời rạc hóa và có thể lặp nhiều vòng.

**Vì thế câu “SB thường thắng không?” không có đáp án tần suất bằng toán.** Toán có thể chứng minh một bất đẳng thức với metric và lớp đối thủ xác định. Tần suất thắng benchmark là câu hỏi thực nghiệm, và kết quả thay đổi theo loss, dữ liệu, ngân sách và baseline.

## 5. Từ posterior tới quyết định truyền: mối nối còn thiếu

Các công thức sau là suy diễn quyết định Bayes cho hệ thống đề xuất, không phải định lý hiệu năng mới của SB.

Gọi S là trạng thái liên quan nhiệm vụ, H là lịch sử thông tin hiện có, b=p(S|H). Với loss nhiệm vụ ℓ(d,S), rủi ro Bayes:

\[
\mathcal R(b)=\inf_d\mathbb E_b[\ell(d,S)].
\]

Yêu cầu truyền a tạo một quan sát ngẫu nhiên O_a qua kênh. Giá trị kỳ vọng của thông tin:

\[
\operatorname{VOI}(a|H)=\mathcal R(b)
-\mathbb E_{O_a|H,a}\mathcal R\bigl(p(S|H,O_a,a)\bigr).
\]

Nếu inference là chính xác, loss không đổi, quan sát miễn phí và có thể bỏ qua, VOI≥0 vì bộ thu luôn có thể tiếp tục dùng quyết định cũ. Chi phí bit/độ trễ/năng lượng phải được trừ riêng khi chọn a. Đây không bảo đảm bộ thu xấp xỉ hiện tại luôn cải thiện sau khi nhận thêm gói.

### Uncertainty lớn chưa chắc đáng gửi

Toy: trạng thái S=(T,N); T là bit nhiệm vụ đồng xác suất, N là nuisance độc lập có bốn giá trị. Gói tiết lộ N giảm entropy 2 bit nhưng giảm lỗi phân loại T bằng 0; gói tiết lộ T giảm entropy chỉ 1 bit nhưng giảm lỗi từ 0.5 xuống 0. Mức giảm entropy, phương sai latent và giá trị nhiệm vụ không đồng nhất.

Do đó SB có posterior tốt hơn vẫn cần chứng minh nó xếp hạng yêu cầu truyền tốt hơn. “Mẫu phân tán ở đâu thì gửi ở đó” chỉ là heuristic.

### Một chặn nối sai số posterior với sai số quyết định

Giả sử 0≤ℓ≤L, p là posterior thật, q là xấp xỉ và d_q tối ưu theo q. Từ chênh lệch kỳ vọng bị chặn bởi L·TV, cộng hai lần và dùng Pinsker:

\[
\mathbb E_p\ell(d_q,S)-\inf_d\mathbb E_p\ell(d,S)
\le 2L\operatorname{TV}(p,q)
\le L\sqrt{2\operatorname{KL}(p\|q)}.
\]

Cũng có thể dùng KL(q||p) nếu đó là chiều hữu hạn được kiểm soát. Đây là lý do toán học để quan tâm chất lượng posterior. Nhưng KL(P*||R) được tối thiểu hóa trong định nghĩa SB **khác** KL(p||q) giữa posterior thật và model học. Chặn regret chưa được suy ra chỉ từ một giá trị bridge loss thấp. Chặn cũng chưa tự giải việc xấp xỉ kỳ vọng của các quan sát tương lai khi tính VOI.

## 6. Suy ra cách đặt SB cho SwarmBridge

### Hai trục thời gian cần phân biệt

- k: frame vật lý của scene/UAV.
- τ∈[0,1]: thời gian nhân tạo của thuật toán bridge để sinh/cập nhật một phân phối.

Một bridge theo τ không tự học đúng dynamics theo k. Chỉ khớp hai phân phối marginal giữa frame không xác định được chuyển động đúng của xe.

### Bài toán posterior phải được định nghĩa trước thuật toán

Với dynamics T_k và belief b_{k-1}, prediction/correction lý tưởng là:

\[
b_k^-(z)=\int T_k(z|z',\mathrm{pose},\Delta t)\,b_{k-1}(z')\,dz',
\qquad
b_k(z)\propto p(o_k|z,c_k)\,b_k^-(z).
\]

Mô hình Gaussian tuyến tính có thể thực hiện điều này bằng mean/covariance và Kalman; không cần sinh nhiều latent mẫu. SB chỉ đáng xét về mặt tính toán khi posterior phức tạp, xấp xỉ rẻ bỏ sót cấu trúc ảnh hưởng quyết định, và sampler cạnh tranh với các cách xấp xỉ khác.

### Một phát biểu SB cụ thể hơn “bridge frame cũ sang mới”

Đặt c_k là thông tin sẵn có gồm lịch sử, pose, Δt, mask và CSI/quan sát hiện tại. Tạo xấp xỉ posterior rẻ q_ref(z|c_k), chẳng hạn mean temporal mapping cùng covariance/ensemble. Sau đó học **conditional SB từ q_ref(·|c_k) tới posterior đích p(z_k|c_k)** theo τ. Điều kiện c_k được giữ xuyên suốt. Đây là ứng dụng kiểu conditional posterior refinement, không tự nhận là ý tưởng mới.

Posterior đích không được oracle cung cấp lúc suy luận: phải học từ joint scene–observation qua dữ liệu/simulator, rồi đánh giá ngoài mẫu. Khả năng rút ngắn đường sinh có lý do khi reference đã gần target; lợi ích hữu hạn phải đo. Không được ép mọi thông tin kênh thành diffusion variance nếu chưa có mô hình thống kê tương ứng.

### Kênh hiện tại có likelihood cụ thể

Suy ra trực tiếp từ `channel/channel.py`, với age=0, r>0 và điều kiện theo precision đã biết:

\[
z_p|z,r\sim\mathcal N(rz,\;r(1-r)I).
\]

Tương đương dùng z_p/r với noise variance (1-r)/r; r=0 không đóng góp likelihood. Nếu prior dự báo scalar là N(m,P), posterior Gaussian có:

\[
P_+^{-1}=P^{-1}+\frac{r}{1-r},
\qquad m_+=P_+\left(P^{-1}m+\frac{z_p}{1-r}\right).
\]

Đây là sanity check chính xác cho inference có điều kiện. `z_p` đã co về prior đơn vị, nên coi `z_p=z+noise` phương sai 1-r sẽ sai likelihood khi thay bằng temporal prior.

### Bayes tilt liên quan SB, nhưng không đồng nhất

Với prior path law R và likelihood dương L(o|X₁), đặt Z=E_R L. Với Q không bị áp thêm ràng buộc marginal:

\[
\operatorname{KL}(Q\|R)-\mathbb E_Q\log L
=\operatorname{KL}(Q\|R^o)-\log Z,
\quad \frac{dR^o}{dR}=\frac{L}{Z}.
\]

Nghiệm là Bayes posterior trên path. Công thức này giải thích mục tiêu correction theo likelihood. Nó là bài toán biến phân với terminal potential; không được gọi ngay là classical SB với hai marginal cố định. Nếu khóa Q₀=R₀, posterior toàn path nói chung không còn khả thi vì quan sát mới có thể cập nhật cả trạng thái quá khứ.

## 7. Điều code hiện tại thực sự kiểm tra

`AuditDenoiser.sample` với nfe=1 bắt đầu ở source, gọi mạng tại τ=1 rồi trả prediction. Không có random input sinh posterior ở nhánh này. Do đó nó là một estimator điểm được học qua các trạng thái bridge. Nó kiểm tra lợi ích của **SB-training**, chưa kiểm tra lợi ích của **posterior sampling**.

Thêm CE có thể giúp nhiệm vụ nhưng thay đổi nghiệm denoising conditional mean dùng để xây score/kernel. Một posterior sampler sau task fine-tuning cần được kiểm định lại. Kết quả finite-step Gaussian trong `scripts/audit_sb_assumptions.py` đã cho thấy kể cả estimator mean chính xác cũng có thể tạo phân tán sai với plug-in sampler ít bước.

Điều này giải thích một sự thiếu ăn khớp giữa kỳ vọng và thiết kế, không chứng minh đó là nguyên nhân duy nhất của mọi khoảng cách mIoU, và không phủ nhận khả năng one-step SB-training thắng trong bài toán khác.

## 8. Kế hoạch có căn cứ toán học, trước khi train lớn

1. Chốt loss nhiệm vụ và thông tin c_k mà mọi baseline được dùng. Có thể dùng 1−mIoU toàn scene làm bounded loss về mặt lý thuyết; CE hoặc lỗi xe là surrogate khác và cần ghi rõ.
2. Kiểm tra một conditional sampler trên Gaussian và mixture có posterior giải tích: mean, covariance, mode mass, coverage, lỗi theo NFE. Đối chiếu baseline Gaussian/Kalman trong trường hợp nó là nghiệm chính xác. Đây là kiểm tra đối tượng cần học, mạnh hơn chỉ nhìn sample đẹp.
3. Trên Town3, so temporal mapping, ensemble/covariance, và conditional SB cùng thông tin. Đánh giá held-out posterior diagnostics phù hợp, task risk và latency. Dữ liệu không cung cấp nhiều ground truth cho cùng một history nên không được gọi sai số latent trung bình là chứng nhận posterior đúng.
4. So quyết định xin thêm block bằng **cải thiện nhiệm vụ thực tế trên cùng chi phí truyền**. Có thể dùng simulator để đánh giá candidate queries ở val; true labels chỉ chấm utility/huấn luyện, không được lọt vào quyết định triển khai.
5. Chỉ mở RL sau khi một belief/action interface hoạt động; nếu SB có lợi, phải tách lợi ích inference khỏi lợi ích thêm history hay thêm compute. Nếu không có lợi, giới hạn kết luận vào cấu hình và bài toán đã thử.

Memory hữu ích không phải ưu thế riêng của SB. Tương tự, mô hình sinh không vượt được định nghĩa tối ưu rate–distortion bằng cách đoán thông tin chưa nhận. [Wyner–Ziv](https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf) cho một nền tảng về side information ở decoder; temporal wireless hiện tại cần mô hình nhân quả riêng, chưa thể lấy nguyên công thức i.i.d. đó làm bound hệ thống.

**Quyết định đề nghị:** chưa mở thêm recipe SB-decoder hoặc mặc định SB temporal sẽ thắng. Nếu tiếp tục đầu tư SB, chốt đối tượng là conditional posterior refinement phục vụ quyết định truyền; kiểm tra sampler và mối nối posterior → task utility trước. Đây là hướng có lập luận toán học rõ hơn việc lấy hai latent làm endpoint rồi kỳ vọng bridge tự cải thiện segmentation.

## Tái lập các phản ví dụ nhỏ

```powershell
.venv\Scripts\python.exe -m scripts.analyze_sb_mathematical_limits
```

Script chỉ dùng thư viện chuẩn, tính chính xác trên tập hữu hạn. Output: `runs/sb_math_2026-11-02/analytic_checks.json`. Xác nhận đồng nhất thức MSE, phản ví dụ đúng marginal/sai coupling, đồng nhất thức Bayes tilt và khác biệt giữa entropy với task VOI. Đây là minh họa toán, không phải kết quả mới trên dataset hoặc chứng minh SB sẽ thắng.
