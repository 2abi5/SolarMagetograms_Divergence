"""
E8 vanilla EDSR (Lim et al. 2017, "Enhanced Deep Residual Networks for Single
Image Super-Resolution"), following the reference EDSR-PyTorch layout:

    head conv -> n_resblocks x [conv-ReLU-conv, x res_scale, +skip] -> conv
    -> + head (global skip) -> upsampler (conv to 4C, PixelShuffle(2)) x 2 -> tail conv

No BatchNorm (EDSR's defining change from SRResNet). Adapted to 1 input and
3 output channels; the RGB mean-shift layers do not apply to magnetograms and
are omitted. Width 32 and 8 blocks keep the parameter count near HighRes-Net's.
Fully convolutional: any LR size.
"""

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    lr_multiple = 1

    def __init__(self, in_channels=1, out_channels=3, n_feats=32, n_resblocks=8, res_scale=1.0,
                 upscale_factor=4):
        super().__init__()
        assert upscale_factor in (2, 4), "x2 or x4 (PixelShuffle(2) stages)"
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)],
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))
        stages = []
        for _ in range({2: 1, 4: 2}[upscale_factor]):
            stages += [nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1), nn.PixelShuffle(2)]
        self.upsample = nn.Sequential(*stages)
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)

    def forward(self, lr):
        x = self.head(lr)
        x = x + self.body(x)
        return self.tail(self.upsample(x))
