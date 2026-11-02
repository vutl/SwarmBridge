**SB speech và codec: những gì đã chạy thực tế ngày 31/10/2026**

Đã chạy inference bằng checkpoint tác giả của NVIDIA SB, Sony SBCTM và BridgeVoC phiên bản mở rộng. Đã chạy forward/backward và sampler của code Rethinking trên dữ liệu tổng hợp. Đây là kiểm tra khả năng chạy và chất lượng trên một tập nhỏ; chưa tái lập đầy đủ bảng benchmark, chưa train lại speech và chưa có phép so SB–mapping cùng backbone trên speech.

**Kết quả đo tại máy**

Dùng 12 cặp clean/noisy đầu tiên lấy được từ archive test VoiceBank-DEMAND, tất cả thuộc speaker p232. Các entry đã giải nén hoàn chỉnh và kiểm tra CRC. Không chọn mẫu theo kết quả model. Tập nhỏ và chỉ một speaker không đủ đại diện toàn bộ test set. Đã dừng tải phần archive còn lại khi đủ tập probe; không tuyên bố đã tải xong toàn bộ dataset.

Khử nhiễu: cùng 12 cặp được resample 48→16 kHz bằng `scipy.signal.resample_poly`; so tín hiệu đầu ra với clean tương ứng. Cả hai chỉ số dưới đây càng cao càng tốt.

| Model / sampler | SI-SDR (dB) | ESTOI |
|---|---:|---:|
| Noisy input | 9,0961 | 0,8151 |
| NVIDIA SB, official ODE, 1 step | 12,8901 | 0,8947 |
| NVIDIA SB, official ODE, 10 steps | 12,6189 | 0,8949 |
| NVIDIA SB, official SDE, 10 steps | 12,4595 | 0,8962 |
| NVIDIA SB, official SDE, 50 steps | 12,2290 | 0,8883 |
| Sony SBCTM, 1 step | 13,2877 | 0,8979 |
| Sony SBCTM, 4 steps | 14,3679 | 0,9011 |

Đây là **thắng đầu vào noisy**, không phải bằng chứng thắng mapping cùng backbone. Hai checkpoint có dữ liệu và cách huấn luyện khác nhau; không dùng bảng nhỏ này để xếp hạng hai paper. NVIDIA dùng WSJ0/CHiME3 theo model card; Sony công bố weights train VoiceBank-DEMAND. Probe dùng test VoiceBank nên càng không được xem là phép so cùng điều kiện huấn luyện.

Nguồn weights/code: [NVIDIA model card](https://huggingface.co/nvidia/se_den_sb_16k_small), [Sony SBCTM](https://github.com/sony/sbctm). Model NVIDIA là 25.161.882 tham số; Sony loader giữ cả student và target network, tổng 131.983.246 tham số trong instance, nhưng inference dùng nhánh `dnn_tgt`.

Vocoder là một nhiệm vụ riêng: 12 waveform clean được chuyển thành mel 100 bands ở 24 kHz; script gốc dựng prior từ mel và tái tạo waveform bằng BridgeVoC single-step. Không đưa noisy waveform vào phép thử vocoder. ESTOI sau resample về 16 kHz đạt **0,9857**; sai số L1 log-mel theo hàm mel gốc đạt **0,1019**. Hai giá trị này không so trực tiếp với bảng khử nhiễu. Chưa đối chứng một neural vocoder predictive.

BridgeVoC đã chạy là checkpoint LibriTTS single-step của [Andong-Li-speech/BridgeVoC](https://github.com/Andong-Li-speech/BridgeVoC), phiên bản mở rộng của tác giả. Paper PDF trong thư viện là bản IJCAI với [repo Tencent](https://github.com/tencent-ailab/BridgeVocoder). Đã tải cả hai source snapshot nhưng chỉ chạy weights phiên bản mở rộng; không gọi đây là tái lập chính xác checkpoint IJCAI.

**Có code gì, và đã chạy tới đâu**

| Công trình | Nguồn xác minh | Trạng thái tại máy |
|---|---|---|
| Jukić/NVIDIA SB speech enhancement | NeMo 2.0.0 + `nvidia/se_den_sb_16k_small` | Inference 12 mẫu × 4 sampler settings, có waveform và metric |
| Sony SBCTM | `sony/sbctm`, checkpoint OSF trong README | Inference 12 mẫu × 2 settings, có waveform và metric |
| BridgeVoC mở rộng | `Andong-Li-speech/BridgeVoC`, weights Hugging Face | Chạy nguyên `enhancement_single.py`, đủ 12 waveform |
| BridgeVoC IJCAI | `tencent-ailab/BridgeVocoder` | Đã tải source; không chạy checkpoint bản gốc |
| Rethinking Flow and Diffusion Bridge Models | `Dahan-Wang/Rethinking-Flow-and-Diffusion-Bridge-Models-for-Speech-Enhancement` | Forward/backward hai backbone và sampler bridge trên GPU; random weights, không đo chất lượng |
| SBM Mamba | `sbmse/sbmse.github.io` được paper dẫn tới | Snapshot được kiểm tra chỉ là demo; không có Python/weights trong repo này |
| BridgeCodec | `thuhcsi/interspeech2026-BridgeCodec` | Snapshot là demo/audio; không có Python/weights trong repo này |
| FNSE-SBGAN | `Taltt/FNSE-SBGAN` | README ghi code coming soon; không có Python ở snapshot đã kiểm tra |
| SE-MSB unpaired | Link anonymous trong paper | Truy cập HTTP 403; chưa tải/chạy được |
| A²SB | `NVIDIA/diffusion-audio-restoration` | Đã tải source, chưa chạy weights; khác nhiệm vụ và model với NeMo denoising |
| SBGSC | Tìm tiêu đề và repo GitHub | Chưa xác minh được repo chính thức; không kết luận tuyệt đối rằng tác giả không có code |

Các trạng thái demo/coming-soon chỉ nói về các repo được kiểm tra ngày 31/10, không bao quát mọi bản code có thể tồn tại ở nơi khác. [SE-MSB](https://arxiv.org/html/2609.26000) dùng học vận chuyển từ dữ liệu **không có cặp**, khác các bridge supervised trên từng cặp của SwarmBridge. Không gọi việc áp dụng công thức marginal lên latent có cặp là tái lập thuật toán unpaired của họ.

**Họ dùng SB khác ta ở đâu**

NVIDIA đưa quan sát nhiễu vào làm điều kiện cho predictor trong suốt quá trình, dùng STFT có biến đổi biên độ và chuẩn hóa, thêm loss waveform, và có sampler SDE/ODE riêng. Đã đọc trực tiếp cấu hình trong checkpoint, không suy từ hình minh họa. Audit BEV mới dùng `x_t`-only và không chạy PF-ODE; `mean_path_4` của ta là heuristic khác. Tuy nhiên sampler nhiều bước không giải thích riêng thất bại của SB một bước.

Sony SBCTM dùng consistency trajectory distillation để tạo model ít bước từ teacher. Gọi một SB predictor thường đúng một lần không tương đương tái lập distillation của họ. [README Sony](https://github.com/sony/sbctm) có code train và inference rõ ràng; probe hiện chỉ chạy pretrained inference.

[SBM Mamba](https://arxiv.org/html/2510.16834v2) dùng backbone và loss phổ nhiều thành phần, huấn luyện trên kho speech lớn; trong ablation của tác giả, SB-training có lợi so với mapping. Nhưng main Mamba-base và SBM khác nhẹ số tham số do conditioning theo thời gian. Không được diễn giải rằng mọi mạng U-Net đổi sang SB đều phải thắng. Chưa có code/weights chạy được từ demo repo đã kiểm tra, nên chưa tái lập kết luận này tại máy.

[BridgeCodec](https://arxiv.org/html/2610.15453), mục 2.4 và 3.2: SB nối latent lượng tử hóa của codec nguồn 8 kHz với codec đích 48 kHz. Train latent trước, rồi fine-tune qua decoder đông cứng bằng loss waveform và mel. Recipe dùng VP và 10 NFE; không phải VE one-step. Bảng chính so codec nguồn/đích và biến thể SB không Mamba, chưa phải bằng chứng SB vượt direct translator cùng backbone. Bài toán này gần SemCom hơn khử nhiễu, nhưng không đồng nhất với lấp block của cùng frozen teacher latent.

BridgeVoC tạo endpoint nguồn có cấu trúc bằng cách nâng mel trở lại không gian phổ tuyến tính, rồi học phục hồi phổ/phase. Đây là ví dụ cụ thể của việc thiết kế endpoint và biểu diễn cho bài toán nén. Bản single-step mở rộng có giai đoạn distillation; không coi một forward của predictor bất kỳ là tương đương. Nguồn: [repo tác giả](https://github.com/Andong-Li-speech/BridgeVoC).

[Rethinking](https://arxiv.org/html/2602.18355), Table 3: predictive TF-GridNet đạt SI-SNR 16,448 và PESQ 3,187; proposed 5 NFE đạt 16,424 và 3,213. Cải thiện phụ thuộc chỉ số, không phải thắng tuyệt đối. Họ thay backbone, loss và fine-tune reverse process; repo có cả `config.yaml`, `config_predictive.yaml` và `config_finetuning.yaml`. Code đã chạy được hai TF-GridNet (2,158M / 2,112M tham số) và sampler trên GPU; chưa thấy weights được dẫn trong README/tree đã kiểm tra nên chưa tái lập các số này.

Vì vậy, câu trong báo cáo cũ rằng **không paper nào thắng mapping cùng backbone về chỉ số độ chính xác** là quá rộng. Đồng thời, thắng diffusion khác hoặc noisy input không tự chứng minh lợi ích riêng của SB so với mapping. Cần giữ cả hai nhận định khi đánh giá tài liệu.

**Hệ quả cho SwarmBridge — suy luận của mình, chưa phải kết quả nhân quả**

Speech restoration và semantic reconstruction đều có thể là bài toán suy luận có điều kiện từ thông tin suy giảm. Không có lý do lý thuyết chỉ từ tên modality để nói SB dùng được cho speech nhưng không dùng được cho SemCom. Ngược lại, không có bảo đảm thắng vì cả hai đều là inverse problem.

Audit Town3 trả lời một câu hẹp: paired VE bridge, input x_t-only, cùng U-Net và MSE+CE, 30k updates có thắng direct mapping trên val không? Ở mức gửi ít, kết quả là âm. Nó chưa kiểm tra đầy đủ source-conditioned SB, distillation, reverse-process fine-tuning, unpaired DSB hay codec translation. Bỏ qua sự khác nhau đó rồi kết luận SB không hợp UAV là quá sớm.

Các hướng kiểm tra có giá trị nhất tiếp theo là:

1. Xác minh lợi ích **riêng của SB** trên một recipe speech có cả predictive đối chứng: Rethinking là candidate cụ thể vì đã có code hai nhánh. Cần checkpoint đúng hoặc train lại hai nhánh bằng cùng dữ liệu/compute; không dùng 12 test clips để train rồi báo cáo kết quả test.
2. Trên BEV, phân tích checkpoint đã có theo vùng thực sự nhận được và vùng không có bằng chứng, cùng IoU từng lớp. Điều này giúp chọn giả thuyết tiếp, thay vì đổi backbone vì tên Mamba.
3. Nếu thử source-conditioned bridge/PF-ODE hoặc fine-tune ở trạng thái suy luận, thay từng yếu tố và giữ mapping cùng budget. Một bước thắng sau fine-tune cũng cần so mapping fine-tune bằng đúng số updates.
4. Nếu chọn hướng codec translation, phải có nhu cầu encoder/decoder khác nhau và bottleneck thật. Nếu chọn task-oriented reconstruction, phải đánh giá khả năng giữ nhãn/đối tượng và cost thực; không dùng độ đẹp thay thế độ đúng của cảnh.

Lượt này chưa chạy thêm một queue BEV dài. Tất cả probe inference/code đã kết thúc; không để GPU train nền sau khi trả kết quả.

**Tính tái lập và các giới hạn kỹ thuật**

- Source snapshots có commit ID trong `pdf/GithubModels/*/SWARMBRIDGE_SOURCE.json`; modules NeMo giữ nguyên source số học và license. Weights có SHA-256 trong manifest.
- `.venv-sb-papers` dùng PyTorch 2.5.1/cu124, torchaudio 2.5.1; các dependency bổ sung và NumPy riêng. `.venv` SwarmBridge vẫn dùng PyTorch 2.12.0/cu130. File `.pth` cho phép môi trường paper dùng lại các dependency phụ đã có; không coi nó là container độc lập hoàn toàn.
- NeMo: adapter bỏ framework metadata/type decorators và eager ASR-training imports; giữ nguyên estimator, encoder/decoder, VE và SBSampler. Load strict toàn bộ 536 tensor keys. Chưa chạy native `AudioToAudioModel.restore_from` vì dependency ASR của framework, nên không gọi đây là chạy nguyên toàn bộ NeMo application.
- Sony: dùng native PyTorch FIR/activation có sẵn trong source thay custom CUDA extension trên Windows; không thay weights/kiến trúc. Bỏ riêng 647 keys của teacher chỉ dùng khi train, rồi strict-load mọi key còn lại vào student/target. Các dependency chỉ dùng ở training được chặn trong adapter; không tạo giả metric.
- BridgeVoC: chạy trực tiếp script inference gốc, không sửa code hay weights. Script ghi PCM16 theo mặc định. Variant/checkpoint và cấu hình inference được ghi rõ ở trên.
- Round-trip STFT/ISTFT của adapter NeMo có max error 1,19e-7; VE variance so closed form lệch tối đa 3,58e-7; SI-SDR so TorchMetrics lệch 3,55e-15. Chúng kiểm tra tính nhất quán số học, không chứng minh chất lượng posterior.
- PESQ chưa đo: package chuẩn cần Microsoft Visual C++ Build Tools và build thất bại trên máy này. Không thay bằng metric xấp xỉ rồi gọi đó là PESQ. Chưa đo WER/DNSMOS hay làm nghe đánh giá chủ quan.

Chạy lại probe bằng môi trường đã chuẩn bị:

    .venv-sb-papers\Scripts\python.exe scripts/run_sb_speech_probe.py --engine nvidia
    .venv-sb-papers\Scripts\python.exe scripts/run_sb_speech_probe.py --engine sony
    .venv-sb-papers\Scripts\python.exe scripts/probe_rethinking_sb_code.py
    .venv-sb-papers\Scripts\python.exe scripts/check_sb_paper_adapters.py

Kết quả, waveform clean/noisy/processed và metadata nằm trong `runs/sb_paper_reproductions/`. `nvidia/summary.json`, `sony/summary.json`, `bridgevoc_results.json`, `rethinking_code_probe.json`, `compatibility_checks.json` là các file kết quả chính. Âm thanh BridgeVoC nằm trong `bridgevoc_NFE1/`.
