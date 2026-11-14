# Tài liệu, mã nguồn và prior art

## Danh sách 12 paper về SB, CSI và NLoS — 04/11/2026

Đối chiếu theo danh sách người dùng gửi. Có đủ 12/12 PDF (bản cập nhật 23:40: thêm #1, #2, #5). Kế hoạch dựa trên các paper này: [ke-hoach-proposal-abc-2026-11-04.md](../huang-abc/ke-hoach-proposal-abc-2026-11-04.md). Các PDF mới được kiểm tra chữ ký, mở bằng `pypdf` và đối chiếu từ khóa tiêu đề ở trang đầu. Nhãn dưới đây chỉ phản ánh lần tải này.

| # | Paper | Trạng thái / đường dẫn |
|---|---|---|
| 1 | Diffusion Schrödinger Bridge-Based Channel Estimation for MIMO ISAC Systems | [PDF](<../../../pdf/Diffusion Schrödinger Bridge-Based Channel Estimation for MIMO ISAC Systems.pdf>); [IEEE](https://ieeexplore.ieee.org/document/11189857), [DOI](https://doi.org/10.1109/LWC.2025.3617170) |
| 2 | Brownian Bridge-Based Diffusion Channel Denoising for ISAC Massive MIMO Systems | [PDF](<../../../pdf/Brownian Bridge-Based Diffusion Channel Denoising for ISAC Massive MIMO Systems.pdf>); [IEEE](https://ieeexplore.ieee.org/document/10859257), [DOI](https://doi.org/10.1109/TCCN.2025.3537103) |
| 3 | Diffusion-based Generative Prior for Low-Complexity MIMO Channel Estimation | [PDF mới](<../../../pdf/Diffusion-based Generative Prior for Low-Complexity MIMO Channel Estimation.pdf>) |
| 4 | Generative Diffusion Models for High Dimensional Channel Estimation | [PDF mới](<../../../pdf/Generative Diffusion Models for High Dimensional Channel Estimation.pdf>) |
| 5 | Generative Diffusion Model-Based Variational Inference for MIMO Channel Estimation | [PDF](<../../../pdf/Generative Diffusion Model-Based Variational Inference for MIMO Channel Estimation.pdf>); [IEEE](https://ieeexplore.ieee.org/document/10946972/), [accepted manuscript tại QMUL](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/106727/Nallanathan%20Generative%20Diffusion%20Model-based%20Variational%20Inference%20for%20MIMO%20Channel%20Estimation%202025%20Accepted.pdf?isAllowed=y&sequence=2) |
| 6 | Environment-Aware Diffusion Model for Massive MIMO-OFDM Channel Estimation | [PDF mới](<../../../pdf/Environment-Aware Diffusion Model for Massive MIMO-OFDM Channel Estimation.pdf>) |
| 7 | Environment-Aware Channel Inference via Cross-Modal Flow | [PDF mới](<../../../pdf/Environment-Aware Channel Inference via Cross-Modal Flow.pdf>) |
| 8 | Semantic-Driven Multi-UAV Cooperative Communication With Joint Positioning and Resource Allocation | [Đã có](../../../pdf/Semantic-Driven_Multi-UAV_Cooperative_Communication_With_Joint_Positioning_and_Resource_Allocation.pdf) |
| 9 | Optimally Bridging Semantics and Data: Generative Semantic Communication via Schrödinger Bridge | [Đã có](<../../../pdf/Optimally Bridging Semantics and Data Generative Semantic Communication via Schrodinger Bridge.pdf>) |
| 10 | DiffCP: Ultra-Low Bit Collaborative Perception via Diffusion Model | [PDF mới](<../../../pdf/DiffCP Ultra-Low Bit Collaborative Perception via Diffusion Model.pdf>) |
| 11 | BridgeCodec | [Đã có](<../../../pdf/BridgeCodec Mamba Enhanced Neural Audio Codec with Schrödinger Bridge at Low Bitrate.pdf>) |
| 12 | I²SB | [Đã có](<../../../pdf/I2SB Image-to-Image Schrodinger Bridge.pdf>) |

IEEE trả trang HTML xác thực thay vì PDF cho #1 và #2. Với #5, IEEE không cung cấp PDF cho phiên tải; bản accepted manuscript có thể xem qua trang QMUL nhưng máy tải nhận kết nối đóng hoặc trang chặn 403. Bạn có thể thử các link bằng trình duyệt hoặc tài khoản trường. Không lưu HTML dưới tên PDF.

Đã clone nông vào `pdf/GithubModels/`:

- Paper #3: [Diffusion_channel_est](../../../pdf/GithubModels/benediktfesl__Diffusion_channel_est) — `7eec6bd`, repo tác giả, có mã train/eval.
- Paper #4: [Diffusion-models-for-channel-estimation](../../../pdf/GithubModels/STARainZ__Diffusion-models-for-channel-estimation) — `2b92244`, repo liên kết từ trang nhóm tác giả.
- Paper #7: [Environment-Aware-Channel-Inference](../../../pdf/GithubModels/GM-Leung__Environment-Aware-Channel-Inference) — `51ba3c6`. **Hiện chỉ có README**, tác giả ghi code sẽ phát hành sau.
- Paper #12: [I2SB](../../../pdf/GithubModels/NVlabs__I2SB) — `1ffdfaa`, repo tác giả.
- Paper #11: [BridgeCodec](../../../pdf/GithubModels/thuhcsi__interspeech2026-BridgeCodec) đã có từ trước.

Chưa tìm được repo chính thức có thể xác minh cho các paper còn lại. Dataset, checkpoint bổ sung và dependency chưa được tải/cài trong đợt này; các repo chưa được chạy thử. Nhật ký tải và SHA-256 của 5 file mới: `runs/csi_paper_downloads_2026-11-04.json`.

Trang đọc chung cho các đợt thu thập tài liệu từ 30/10 đến 04/11/2026. Paper ở `pdf/`, mã nguồn tham khảo ở `pdf/GithubModels/`, dữ liệu nghiên cứu ở `data/`; lần sắp xếp report không di chuyển chúng.

## Nguồn nên mở

| Mục đích | Nguồn chi tiết |
|---|---|
| Đối thủ của bài toán arrival-aware cooperative tracking, ma trận so sánh và mức độ đã đọc | [Prior-art map](prior-art-arrival-aware-2026-11-04.md) |
| CERF/CoSDH: phương pháp, giới hạn code công bố và hệ quả cho baseline | [Review paper + code](cerf-review-2026-11-04.md) |
| Literature SB, vai trò và evidence | [SB evidence map](literature-evidence-map-2026-10-30.md), [tổng hợp SB](../sb-town3/sb-va-town3.md) |
| Repo UAV3D, DHD, How2comm, CodeFilling, U2UData và repo hỗ trợ | [Nhật ký repo](github-swarm-downloads-2026-11-04.md) |
| Những paper tải thêm cho hướng swarm | [Nhật ký PDF](tai-them-tai-lieu-swarm-2026-11-04.md) |
| Nguồn “Tóm tắt điều hành.pdf” | [Đối chiếu nguồn và repo](nguon-tom-tat-dieu-hanh-2026-11-03.md) |

Các nhật ký tải là snapshot theo thời điểm. Ví dụ GOST được ghi chưa tải được ở đợt đầu, nhưng PDF đã có trong `pdf/` lúc sắp xếp report; prior-art map mới hơn có phân tích nó. Không đọc một lỗi tải HTTP cũ thành trạng thái hiện tại của thư viện. Chưa đọc toàn văn/không tìm thấy code phải được phân biệt với kết luận paper không có kỹ thuật hoặc tác giả không công bố code.

## Mã nguồn tham khảo

Nhóm collaborative perception gồm UAV3D, DHD/Air-Co-Pred, Where2comm, CodeFilling, CoSDH, CoGMP, CoCa3D, How2comm, OpenCOOD, CoBEVT và CoAlign. Nhóm tracking có TrackEval/BoxMOT; nhóm simulator/MARL có U2UData/U2UData+, AirSim, DSSE, PettingZoo và Gazebo; nhóm nhập môn SemCom có DeepJSCC, DeepSC và SemanticRL.

Nguồn commit/revision chính xác ở manifest bên dưới. Clone nông, source archive, demo, checkpoint và simulator binary có mức hoàn chỉnh khác nhau; “đã tải” không đồng nghĩa “đã cài, chạy hay tái lập paper”. U2USim-2 và client đã được tải ở snapshot riêng từ Hugging Face theo nhật ký, không đồng nghĩa dataset đầy đủ U2UData+ đã tải. UAV3D hiện dùng mini.

## Manifest và JSON

Các file JSON giữ nguyên nội dung khi chuyển sang `metadata/`:

- [Danh sách tài liệu gốc](../../metadata/danh-sach-tai-lieu-2026-10-30.json), [nguồn bổ sung](../../metadata/bo-sung-nguon-nghien-cuu-2026-10-30.json).
- [Paper swarm bổ sung](../../metadata/tai-them-tai-lieu-swarm-2026-11-03.json), [nhật ký thử tải lại](../../metadata/paper-download-retry-2026-11-04.json).
- [Repo swarm](../../metadata/github-swarm-downloads-2026-11-04.json), [repo hỗ trợ](../../metadata/swarm-support-repos-2026-11-04.json), [gói simulator/client](../../metadata/swarm-packages-2026-11-04.json).
- [SB audit checks](../../metadata/sb-audit-checks-2026-10-30.json), [SB decision mechanism](../../metadata/sb-decision-mechanism-2026-10-30.json).
- [Bảng đổi đường dẫn report](../../metadata/report-moves-2026-11-04.json).

## Cách dùng khi nghiên cứu tiếp

Chọn công trình gần nhất theo observation/action/task/bit/channel, rồi đối chiếu toàn văn và implementation. Không suy novelty từ tiêu đề hay từ việc chưa tìm thấy một repo. Những claim trong prior-art map và các đề xuất lịch sử cần được đọc cùng giới hạn nguồn và các đính chính mới nhất trong [tổng hợp UAV3D](../uav3d/uav3d-ket-qua-va-kiem-toan.md).
