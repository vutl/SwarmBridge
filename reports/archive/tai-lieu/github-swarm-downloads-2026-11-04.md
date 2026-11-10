# Repo bổ sung ngày 04/11/2026

**Cập nhật sau yêu cầu “Tải hết”:** đã tải thêm đầy đủ snapshot U2USim-2 Windows/Linux và AirDrone client từ Hugging Face: **82 file, 4.317.501.139 byte**, tất cả được kiểm tra theo SHA-256 của LFS hoặc Git blob SHA-1 do nguồn công bố. Xem [manifest gói đầy đủ](../../metadata/swarm-packages-2026-11-04.json). Các mục bên dưới mô tả đợt clone GitHub ban đầu; simulator/client không còn ở trạng thái chưa tải.

Đợt đầu đã tải 10 repo chính thức còn thiếu vào `pdf/GithubModels/`. Các repo được clone nông; chưa cài dependency hoặc chạy thí nghiệm. Đợt đầu không tải dataset, checkpoint hoặc simulator binaries từ các đường dẫn ngoài; simulator/client được bổ sung ở cập nhật phía trên. Git LFS smudge được tắt cho thao tác clone; đã kiểm tra 10 repo trong bảng và không phát hiện con trỏ LFS chưa tải. Không repo nào trong bảng khai báo `.gitmodules` ở gốc. [Manifest và commit đầy đủ](../../metadata/github-swarm-downloads-2026-11-04.json).

| Repo | Thư mục | Commit |
|---|---|---|
| [UAV3D](https://github.com/huiyegit/UAV3D) | `huiyegit__UAV3D` | `5cee0e9497` |
| [DHD / Air-Co-Pred](https://github.com/WangzcBruce/DHD) | `WangzcBruce__DHD` | `27c2e6b279` |
| [M3OT](https://github.com/M3OT/M3OT) | `M3OT__M3OT` | `c56a75c60a` |
| [How2comm](https://github.com/ydk122024/How2comm) | `ydk122024__How2comm` | `8d357abdcb` |
| [U2UData](https://github.com/fengtt42/U2UData) | `fengtt42__U2UData` | `237d2c659e` |
| [U2UData+ / U2UData-2](https://github.com/fengtt42/U2UData-2) | `fengtt42__U2UData-2` | `b7db055edd` |
| [CoCa3D](https://github.com/MediaBrain-SJTU/CoCa3D) | `MediaBrain-SJTU__CoCa3D` | `b39e5846bc` |
| [CodeFilling](https://github.com/PhyllisH/CodeFilling) | `PhyllisH__CodeFilling` | `57f83ccc7d` |
| [CoSDH](https://github.com/Xu2729/CoSDH) | `Xu2729__CoSDH` | `b270b69fe1` |
| [CoGMP](https://github.com/buaa-colalab/CoGMP) | `buaa-colalab__CoGMP` | `2e62bdb7c6` |

## Sáu repo nền tảng đã bổ sung

- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) → `DerrickXuNu__OpenCOOD`
- [CoBEVT](https://github.com/DerrickXuNu/CoBEVT) → `DerrickXuNu__CoBEVT`
- [CoAlign](https://github.com/yifanlu0227/CoAlign) → `yifanlu0227__CoAlign`
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) → `JonathonLuiten__TrackEval`
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) → `mikel-brostrom__boxmot`
- [Ultralytics](https://github.com/ultralytics/ultralytics) → `ultralytics__ultralytics`

Tất cả nằm trong `pdf/GithubModels/`. Xem [manifest repo hỗ trợ](../../metadata/swarm-support-repos-2026-11-04.json) để biết commit cụ thể.

## Phạm vi tải bổ sung

Đã tải trọn 82 file của hai snapshot Hugging Face công khai ở revision được ghi trong manifest, gồm cả file `.pak`, executable Windows/Linux, client và tài liệu. Không thực thi binary, cài dependency hoặc chạy huấn luyện. Dataset ghi sẵn `fengtt42/U2UData-2` là gói riêng khoảng 61,8 GB theo API tại thời điểm khảo sát, chưa tải trong đợt hoàn tất mã nguồn/simulator này; các dataset và checkpoint liên kết ngoài khác cũng chưa được tải.

Đã thử lại bốn PDF còn thiếu: GOST và Posterior-Mamba vẫn bị IEEE trả HTTP 418; Relay-aware semantic resource allocation và FOFE-MMAPPO vẫn bị ScienceDirect trả HTTP 403. Xem [nhật ký thử lại](../../metadata/paper-download-retry-2026-11-04.json).

## Những giới hạn đã kiểm tra

- Repo U2UData-2 có tài liệu/hình minh họa, hiện không có tệp Python. **Đã bổ sung** simulator Windows/Linux và client mà README dẫn tới, tại `pdf/GithubModels/HuggingFace/fengtt42__U2USim-2/` và `pdf/GithubModels/HuggingFace/fengtt42__U2UData_2_Client/`. Chưa chạy simulator hoặc cài đặt client. U2UData gốc là repo khác và có mã Python.
- Repo M3OT hiện nhỏ, có 2 tệp Python và phần giới thiệu dataset; không mặc định là pipeline huấn luyện đầy đủ cho tất cả benchmark trong paper.
- CoGMP có mã triển khai công khai, README ghi rõ official implementation và hướng dẫn training/evaluation. Thông tin này bổ sung cho việc trước đây mới xác minh được trang dự án.
- Where2comm, Posterior-Mamba, Mamba, AirSim, PettingZoo, Gazebo Sim và các repo speech đã tải ở các đợt trước được giữ nguyên.
- Chưa xác minh được repo triển khai chính thức có thể tải cho MCOP, CoDS (Digital Semantic Communication), SComCP, GMF, GOST, Relay-aware semantic resource allocation và hai bài Springer mới tải. Đây là trạng thái tìm kiếm hiện tại, không phải khẳng định tác giả không công bố code. Tránh nhầm CoDS digital semantic communication với các dự án khác cũng tên CoDS.
- SBM speech hiện có repo demo `sbmse__sbmse.github.io` từ trước; không tính demo là mã huấn luyện.
- `pdf/GithubModels/` vẫn được `.gitignore` bỏ qua; đã kiểm tra với README UAV3D.
