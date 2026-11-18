# Nguồn của “Tóm tắt điều hành.pdf”

Đối chiếu ngày 2026-11-03. Paper được lưu trong `pdf/`, repo trong `pdf/GithubModels/`. Không tải dataset hoặc checkpoint bổ sung. Các repo được clone nông; LFS smudge bị tắt, nên tài nguyên LFS nếu có cần tải riêng. Không cài dependency hay chạy thử.

## Bài báo

| Bài / nguồn | Trạng thái | Nguồn |
|---|---|---|
| MCOP: Multi-UAV Collaborative Occupancy Prediction | Đã tải `MCOP Multi-UAV Collaborative Occupancy Prediction.pdf` | [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Lin_MCOP_Multi-UAV_Collaborative_Occupancy_Prediction_ICCV_2025_paper.pdf) |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Đã tải `Mamba Linear-Time Sequence Modeling with Selective State Spaces.pdf` | [arXiv](https://arxiv.org/abs/2312.00752) |
| DSSE: A Drone Swarm Search Environment | Đã tải `DSSE A Drone Swarm Search Environment.pdf` | [arXiv](https://arxiv.org/abs/2307.06240) |
| Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps | Đã tải `Where2comm Communication-Efficient Collaborative Perception via Spatial Confidence Maps.pdf`; trang dataset CoPerception-UAV yêu cầu trích dẫn bài này | [Trang dataset](https://siheng-chen.github.io/dataset/coperception-uav/), [arXiv](https://arxiv.org/abs/2209.12836) |
| Few-step Adversarial Schrödinger Bridge for Generative Speech Enhancement | Đã có trong `pdf/` | [ISCA](https://www.isca-archive.org/interspeech_2025/han25b_interspeech.html) |
| Schrödinger Bridge for Generative Speech Enhancement | Đã có trong `pdf/` | [arXiv](https://arxiv.org/abs/2407.16074) |
| Efficient Multi-agent Collaboration Learning via Posterior Mamba | **Chưa tải được PDF**. IEEE trả phản hồi chặn/yêu cầu xác thực trong môi trường tải; repo tác giả đã tải. | [IEEE](https://ieeexplore.ieee.org/document/11458732/), [DOI](https://doi.org/10.1109/TBDATA.2026.3679580) |
| Cooperative decision-making of unmanned aerial vehicles: A multi-agent reinforcement learning approach (FOFE-MMAPPO) | **Chưa tải được PDF**. ScienceDirect và bản thảo SSRN trả 403 trong môi trường tải. | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197626002617), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5191422) |

Nguồn ghi “CoPerception-UAV: A Virtual Collaborative Perception Dataset for UAVs (ICCVW 2021)” chưa được xác minh là một bài riêng; trang chính thức giới thiệu dataset và dẫn Where2comm 2022. Không khẳng định dataset đó là một bài 2021.

## Repo đã tải

| Repo | Thư mục | Commit |
|---|---|---|
| [Posterior-Mamba](https://github.com/BUPT-zeld151/Posterior-Mamba) | `BUPT-zeld151__Posterior-Mamba` | `73dcddb` |
| [Mamba](https://github.com/state-spaces/mamba) | `state-spaces__mamba` | `e9594ce` |
| [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm) | `MediaBrain-SJTU__Where2comm` | `d343252` |
| [DSSE](https://github.com/HNU-ICS/UUV-Swarm-Search-via-RL) | `HNU-ICS__UUV-Swarm-Search-via-RL` | `dbe25e4` |
| [Drone Swarm RL / AirSim](https://github.com/Lauqz/Drone-Swarm-RL-airsim-sb3) | `Lauqz__Drone-Swarm-RL-airsim-sb3` | `b0f3181` |
| [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | `Farama-Foundation__PettingZoo` | `6b3f495` |
| [AirSim](https://github.com/microsoft/AirSim) | `microsoft__AirSim` | `1ca93f6` |
| [Gazebo Sim](https://github.com/gazebosim/gz-sim) | `gazebosim__gz-sim` | `dad15e2` |

Chưa tìm thấy repo triển khai chính thức có thể xác minh cho MCOP, FOFE-MMAPPO và bài Few-step Adversarial SB. Đối với Jukić, repo liên quan NVIDIA `NVIDIA__diffusion-audio-restoration` và `NVIDIA__NeMo_audio_2.0.0` đã có sẵn; cần đối chiếu riêng phiên bản/cấu hình trước khi coi đó là bản tái lập chính xác. Không đồng nhất repo demo với code huấn luyện.

Các tập Air-to-Pred-Occ, UAV3D-Occ và GauUScene-Occ được MCOP nêu là các biến thể dataset; không suy ra chúng đã được công khai chỉ từ việc các dataset gốc tồn tại. Thư mục dự án hiện hành chính là SwarmBridge nên không clone thêm bản GitHub của nó.
