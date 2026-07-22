"""U-Net architecture for lung/nodule segmentation."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net with 4 downsampling / upsampling stages."""

    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super().__init__()
        f = base_features

        self.down1 = DoubleConv(in_channels, f)
        self.down2 = DoubleConv(f, f * 2)
        self.down3 = DoubleConv(f * 2, f * 4)
        self.down4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.upconv4 = DoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.upconv3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.upconv2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.upconv1 = DoubleConv(f * 2, f)

        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        bottleneck = self.bottleneck(self.pool(d4))

        u4 = self.upconv4(torch.cat([self.up4(bottleneck), d4], dim=1))
        u3 = self.upconv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.upconv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.upconv1(torch.cat([self.up1(u2), d1], dim=1))

        return torch.sigmoid(self.out_conv(u1))