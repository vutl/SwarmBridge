import time

import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| available:", torch.cuda.is_available())
props = torch.cuda.get_device_properties(0)
print(props.name, "| sm_%d%d" % torch.cuda.get_device_capability(0), "| %.2f GiB" % (props.total_memory / 2**30))

x = torch.randn(4096, 4096, device="cuda")
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    a = x.to(dtype)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    print(f"{str(dtype):15s} matmul: {20 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOPS")

# One training step of a small BEV-like conv net under bf16 autocast
m = torch.nn.Sequential(
    torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.SiLU(), torch.nn.Conv2d(128, 64, 3, padding=1)
).cuda()
opt = torch.optim.AdamW(m.parameters())
z = torch.randn(8, 64, 128, 128, device="cuda")
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = m(z).square().mean()
loss.backward()
opt.step()
print("bf16 train step OK | peak %d MiB" % (torch.cuda.max_memory_allocated() // 2**20))
