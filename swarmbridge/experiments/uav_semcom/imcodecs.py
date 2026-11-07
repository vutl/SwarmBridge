"""Image transmission modes with REAL bitstreams and decoders (direction A, step A1).

Every mode exposes encode(img_uint8_HWC) -> bytes and decode(bytes) -> img_uint8_HWC. The byte count returned by
encode is the full payload a receiver needs (container / header bytes included, nothing side-loaded except the
codec definition and, for learned codecs, the pretrained weights that both ends hold).
"""
import io
import struct

import numpy as np
import torch
from PIL import Image


class PillowCodec:
    """JPEG / JPEG2000 / AVIF through Pillow's bundled libjpeg-turbo / OpenJPEG / libavif."""

    def __init__(self, fmt, level):
        self.fmt, self.level = fmt, level
        self.name = f"{fmt.lower()}_{level}"
        self.family = fmt.lower()
        self.device = "cpu"

    def encode(self, img):
        buf = io.BytesIO()
        im = Image.fromarray(img)
        if self.fmt == "JPEG":
            im.save(buf, "JPEG", quality=self.level, optimize=True)
        elif self.fmt == "JPEG2000":
            # raw J2K codestream (no JP2 box wrapper); level = compression ratio vs 24-bit RGB
            im.save(buf, "JPEG2000", no_jp2=True, irreversible=True, quality_mode="rates",
                    quality_layers=[self.level])
        elif self.fmt == "AVIF":
            im.save(buf, "AVIF", quality=self.level, speed=6, subsampling="4:2:0")
        else:
            raise ValueError(self.fmt)
        return buf.getvalue()

    def decode(self, data):
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


class CompressAICodec:
    """Pretrained learned image codec from CompressAI (real rANS bitstream via model.compress / decompress).
    Payload = all entropy-coded strings + header (latent H, W as uint16, string count, string lengths). Images whose
    sides are not multiples of 64 (native-resolution frames) are edge-padded before coding; then the count byte's
    high bit is set and the original H, W (2 x uint16) follow, and the decoder crops. 64-aligned images (the 512^2
    A1 set) are coded exactly as before (byte-identical payloads)."""

    def __init__(self, arch, quality, device="cuda"):
        from compressai.zoo import image_models
        self.arch, self.quality, self.device = arch, quality, device
        self.name = f"{arch}_q{quality}"
        self.family = arch
        # entropy coding needs bit-identical hyper-latent decoding on both ends; non-deterministic cuDNN kernels
        # can desynchronise encoder and decoder (catastrophic decode). Force deterministic kernels.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.net = image_models[arch](quality=quality, metric="mse", pretrained=True).eval().to(device)
        self.net.update(force=True)

    @torch.no_grad()
    def encode(self, img):
        H, W = img.shape[:2]
        ph, pw = (-H) % 64, (-W) % 64
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        out = self.net.compress(x)
        strings = [s[0] for s in out["strings"]]
        h, w = out["shape"]
        n = len(strings) | (0x80 if (ph or pw) else 0)
        head = struct.pack("<HHB", h, w, n) + b"".join(struct.pack("<I", len(s)) for s in strings)
        if ph or pw:
            head += struct.pack("<HH", H, W)
        return head + b"".join(strings)

    @torch.no_grad()
    def decode(self, data):
        h, w, n = struct.unpack("<HHB", data[:5])
        padded, n = bool(n & 0x80), n & 0x7F
        lens = struct.unpack("<" + "I" * n, data[5:5 + 4 * n])
        pos, strings = 5 + 4 * n, []
        if padded:
            H, W = struct.unpack("<HH", data[pos:pos + 4])
            pos += 4
        for L in lens:
            strings.append([data[pos:pos + L]])
            pos += L
        x = self.net.decompress(strings, (h, w))["x_hat"].clamp(0, 1)
        rec = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return rec[:H, :W] if padded else rec

    def header_bytes(self):
        return 5 + 4 * 2


def default_modes(device="cuda"):
    modes = [PillowCodec("JPEG", q) for q in (5, 15, 30, 50, 75, 90)]
    modes += [PillowCodec("JPEG2000", r) for r in (400, 200, 100, 50, 25, 12)]
    modes += [PillowCodec("AVIF", q) for q in (5, 20, 35, 50, 65, 80)]
    modes += [CompressAICodec("bmshj2018-hyperprior", q, device) for q in (1, 2, 3, 4, 5, 6, 7, 8)]
    modes += [CompressAICodec("mbt2018-mean", q, device) for q in (1, 2, 3, 4, 5, 6, 7, 8)]
    return modes
