"""Per-image reconstruction metrics: PSNR, MS-SSIM, LPIPS (AlexNet) and CLIP image-image cosine (supplementary)."""
import numpy as np
import torch


class Metrics:
    def __init__(self, device="cuda"):
        import lpips
        import open_clip
        from pytorch_msssim import ms_ssim
        self.device = device
        self._ms_ssim = ms_ssim
        self.lp = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        self.clip, _, self.clip_pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.clip = self.clip.eval().to(device)

    def _t(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def clip_embed(self, img):
        from PIL import Image
        x = self.clip_pre(Image.fromarray(img)).unsqueeze(0).to(self.device)
        e = self.clip.encode_image(x)
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def __call__(self, ref, rec, ref_clip=None):
        mse = float(np.mean((ref.astype(np.float64) - rec.astype(np.float64)) ** 2))
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))
        a, b = self._t(ref), self._t(rec)
        msssim = float(self._ms_ssim(a, b, data_range=1.0, size_average=True))
        lp = float(self.lp(a * 2 - 1, b * 2 - 1))
        if ref_clip is None:
            ref_clip = self.clip_embed(ref)
        clip = float((ref_clip * self.clip_embed(rec)).sum())
        return dict(psnr=psnr, ms_ssim=msssim, lpips=lp, clip=clip)
