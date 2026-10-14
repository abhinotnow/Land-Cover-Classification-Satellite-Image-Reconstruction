"""Partial-Convolution U-Net (Liu et al. 2018, simplified: single-channel mask, no VGG loss).
Convention here: valid mask m has 1 = VALID pixel, 0 = hole."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PConv(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, bn=True, act="relu"):
        super().__init__()
        self.k, self.s, self.p = k, s, k // 2
        self.conv = nn.Conv2d(cin, cout, k, s, self.p)
        self.bn = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.act = {"relu": nn.ReLU(), "leaky": nn.LeakyReLU(0.2), None: nn.Identity()}[act]
        self.register_buffer("ones", torch.ones(1, 1, k, k))

    def forward(self, x, m):
        with torch.no_grad():
            msum = F.conv2d(m, self.ones, stride=self.s, padding=self.p)
            new_m = (msum > 0).float()
            ratio = (self.k ** 2) / msum.clamp(min=1.0) * new_m   # re-normalise by valid count
        b = self.conv.bias.view(1, -1, 1, 1)
        out = ((self.conv(x * m) - b) * ratio + b) * new_m
        return self.act(self.bn(out)), new_m


class PConvUNet(nn.Module):
    """Input size must be divisible by 16 (64 and 256 both fine)."""

    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.e1 = PConv(3, ch[0], 5, 2, bn=False)
        self.e2 = PConv(ch[0], ch[1], 3, 2)
        self.e3 = PConv(ch[1], ch[2], 3, 2)
        self.e4 = PConv(ch[2], ch[3], 3, 2)
        self.d4 = PConv(ch[3] + ch[2], ch[2], act="leaky")
        self.d3 = PConv(ch[2] + ch[1], ch[1], act="leaky")
        self.d2 = PConv(ch[1] + ch[0], ch[0], act="leaky")
        self.d1 = PConv(ch[0] + 3, 3, bn=False, act=None)

    @staticmethod
    def _up(x, m):
        return (F.interpolate(x, scale_factor=2, mode="nearest"),
                F.interpolate(m, scale_factor=2, mode="nearest"))

    def forward(self, x, m):
        x0 = x * m
        x1, m1 = self.e1(x0, m)
        x2, m2 = self.e2(x1, m1)
        x3, m3 = self.e3(x2, m2)
        x4, m4 = self.e4(x3, m3)
        u, um = self._up(x4, m4)
        h, hm = self.d4(torch.cat([u, x3], 1), torch.max(um, m3))
        u, um = self._up(h, hm)
        h, hm = self.d3(torch.cat([u, x2], 1), torch.max(um, m2))
        u, um = self._up(h, hm)
        h, hm = self.d2(torch.cat([u, x1], 1), torch.max(um, m1))
        u, um = self._up(h, hm)
        h, _ = self.d1(torch.cat([u, x0], 1), torch.max(um, m))
        return h


@torch.no_grad()
def restore(model, imgs, missing_masks, device, bs=128):
    """imgs float32 (N,H,W,3); missing_masks float32 (N,H,W) with 1=missing.
    Returns composited float32 (N,H,W,3): network output in holes, original elsewhere."""
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.ascontiguousarray(imgs[i:i + bs])).permute(0, 3, 1, 2).float().to(device)
        valid = 1 - torch.from_numpy(np.ascontiguousarray(missing_masks[i:i + bs])).float().unsqueeze(1).to(device)
        y = model(x, valid).clamp(0, 1)
        comp = y * (1 - valid) + x * valid
        out.append(comp.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)
