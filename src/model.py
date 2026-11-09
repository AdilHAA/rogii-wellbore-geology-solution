"""GeoSteerNet by medali1992 (Apache-2.0), adapted by Adil Khabibullin."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pad=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.proj = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.proj(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.res(x)
        return skip, self.pool(skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))


class GeoSteerNet(nn.Module):
    """Nine input channels, three pooling stages, four SDF candidates."""

    def __init__(self):
        super().__init__()
        b = 32
        self.stem = ConvBnRelu(9, b, k=5, pad=2)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        self.bottleneck = nn.Sequential(
            ResidualBlock(b * 8, b * 8), ResidualBlock(b * 8, b * 8)
        )
        self.up3 = UpBlock(b * 16, b * 4)
        self.up2 = UpBlock(b * 8, b * 2)
        self.up1 = UpBlock(b * 4, b)
        self.sdf_head = nn.Conv2d(b, 4, 1)
        self.logit_head = nn.Conv2d(b, 4, 1)

    def forward(self, batch):
        x = self.stem(batch["image"])
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return {
            "sdf": torch.tanh(self.sdf_head(x)) * 3.0,
            "logit": self.logit_head(x).mean(2),
        }
