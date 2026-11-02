# SwarmBridge — khung code Q1

Q1: đưa độ tin cậy kênh vào **reference process** của conditional Schrödinger Bridge có tạo lợi ích
riêng, vượt cả "đưa CSI vào mạng" lẫn "augmentation nhiễu theo kênh" không?

## Luồng dữ liệu

```
z* (oracle, fusion đầy đủ) ──► channel/ (FOV, stale, Rayleigh, AWGN, erasure) ──► z_p, r
                                                                                     │
                     bridges/ : Mapping | FlowMatching | SchrodingerBridge(Reference R0/R1/R2)
                                                                                     │
                                         ẑ* ──► task head ──► mIoU ; mẫu ──► U (Q1b)
```

| File | Vai trò |
|---|---|
| `channel/channel.py` | Mô phỏng kênh trên latent; gộp LMMSE; `r = G/(1+G)` nên `1−r` = phương sai lỗi còn lại của `z_p` |
| `data/synthetic.py` | Dữ liệu giả thay cho latent cache CoPerception-UAV (chỉ để kiểm tra pipeline) |
| `bridges/backbone.py` | Một U-Net dùng chung cho mọi model (cùng capacity) |
| `bridges/reference.py` | R0 isotropic, R1 theo r (hand-crafted), R2 empirical `Var[z*−z_p \| r]`; cùng tổng nhiễu |
| `bridges/models.py` | Mapping (±augmentation), Flow Matching, SB; đều condition rõ trên `z_p` |
| `uncertainty/metrics.py` | Q1b: AUROC của U, của 1−r, và của U **trong từng bin r** |
| `experiments/q1_reference/grid.py` | Lưới đối chứng M0–M4, F1, S0–S4 |
| `experiments/q1_reference/run.py` | Train + đánh giá theo điều kiện kênh easy/mid/severe, NFE 1/4 |

## Chạy

```powershell
.venv\Scripts\python.exe -m tests.test_q1_core
.venv\Scripts\python.exe -m swarmbridge.experiments.q1_reference.run --steps 2000
.venv\Scripts\python.exe -m swarmbridge.experiments.q1_reference.run --models M3 S1 S3 --steps 500
```

## Đọc kết quả (khi có dữ liệu thật)

- **Q1a sống** chỉ khi S3/S4 thắng **cả S1 (CSI trong mạng) lẫn M3 (augmentation theo kênh)**, ổn định qua nhiều điều kiện kênh, cùng NFE.
- **SB đáng giữ** khi S1 ít nhất ngang M1/F1; nếu thua cả hai về chất lượng–latency thì bỏ SB khỏi MVP.
- **Q1b sống** khi `U|r` rõ ràng > 0.5 (U mang thông tin lỗi ngoài r). `U|r ≈ 0.5` nghĩa là U chỉ phản chiếu r.

## Chưa có (TODO)

- Loader latent thật (teacher, encoder đóng băng, cache) thay `data/synthetic.py`
- Conditional diffusion baseline; heteroscedastic mapping (baseline Q1b)
- Test kênh lạ (train AWGN → test Rayleigh), test evidence-removal, nhiều seed
