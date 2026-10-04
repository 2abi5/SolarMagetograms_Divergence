"""
E7 vanilla U-Net (Ronneberger et al. 2015) for x4 super resolution.

Standard pre-upsampling SR use of a U-Net: the LR input is bicubically
upsampled x4 to the HR grid, then a 3-level U-Net (double 3x3 conv + ReLU per
level, 2x2 max-pool down, 2x2 transposed-conv up, skip concatenation, 1x1 conv
head) maps 1 -> 3 channels. 'Same' padding; no normalisation layers, the same
choice as every other model in the study. Base width 16 keeps the parameter
count near HighRes-Net's.

Three 2x poolings need HR sizes divisible by 8, i.e. LR sizes divisible by 2
(`lr_multiple`); evaluation reflect-pads full regions to that multiple.
"""

import torch
import torch.nn as nn


def _double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNetSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base=16, levels=3, upscale_factor=4):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode="bicubic", align_corners=False)
        widths = [base * 2 ** i for i in range(levels + 1)]
        self.down = nn.ModuleList([_double_conv(in_channels, widths[0])] +
                                  [_double_conv(widths[i], widths[i + 1]) for i in range(levels)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
                                 for i in reversed(range(levels))])
        self.merge = nn.ModuleList([_double_conv(2 * widths[i], widths[i]) for i in reversed(range(levels))])
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        self.lr_multiple = 2 ** levels // upscale_factor if 2 ** levels > upscale_factor else 1

    def forward(self, lr):
        x = self.upsample(lr)
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, merge in zip(self.up, self.merge):
            x = merge(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)
