"""Teacher for the leader setup: per-UAV BEV RGB -> latent message z -> BEV classes.

The latent z (C x 48 x 48 for a 384 x 384 grid) is what a UAV transmits; its clean value is
the oracle z* for Q1. The decoder is the frozen task head the bridges are judged through.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

K = 3  # other, road, vehicle
IGNORE = 255


class Encoder(nn.Module):
    """ResNet18 stem..layer2 (stride 8, ImageNet-pretrained) + 1x1 bottleneck to C channels."""

    def __init__(self, C=16, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        self.proj = nn.Conv2d(128, C, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb_uint8):
        x = (rgb_uint8.float() / 255 - self.mean) / self.std
        return self.proj(self.body(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
                                 nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.silu(x + self.net(x))


class Decoder(nn.Module):
    """Latent (C, 48, 48) -> logits (K, 192, 192), i.e. 0.5 m cells."""

    def __init__(self, C=16, width=128):
        super().__init__()
        self.inp = nn.Conv2d(C, width, 3, padding=1)
        self.b0 = nn.Sequential(Block(width), Block(width))
        self.b1 = Block(width)
        self.b2 = Block(width)
        self.out = nn.Conv2d(width, K, 1)

    def forward(self, z):
        h = self.b0(self.inp(z))
        h = self.b1(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False))
        h = self.b2(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False))
        return self.out(h)


def miou(pred, gt, k=K, ignore=IGNORE):
    """mIoU over classes present (pred/gt: int tensors, same shape). Returns (mean, per-class)."""
    valid = gt != ignore
    ious = []
    for c in range(k):
        p, g = (pred == c) & valid, (gt == c) & valid
        union = (p | g).sum().item()
        ious.append((p & g).sum().item() / union if union else float("nan"))
    t = torch.tensor(ious)
    return t[~t.isnan()].mean().item(), ious
