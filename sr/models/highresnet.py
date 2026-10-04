"""
HighRes-Net as used by the converter (source/models/highresnet_rprcdo.py),
adapted to this study: 1 input channel (MDI LOS), 3 output channels
(Bp, Bt, Br), x4, single frame (K = 1).

The Encoder, ResidualBlock and FusionBlock classes are the converter's own.
The converter's Decoder hard-codes one output channel, so Decoder3 subclasses
it and only swaps the final 1x1 convolution for a 64 -> 3 one.

At K = 1 the FusionBlock never runs (its recursion needs >= 2 views). It is
kept so the architecture matches the converter, but its parameters receive no
gradient; report `count_params(model, active_only=True)` alongside the total.
No normalisation layers (the converter has none).
"""

import torch.nn as nn

from source.models.highresnet_rprcdo import FusionBlock
from source.models.model_utils_RPRCDO import Decoder, Encoder


class Decoder3(Decoder):
    def __init__(self, channels=64, upscale_factor=4, out_channels=3, final_kernel_size=1, p=0.0):
        super().__init__(deconv_in_channels=channels, deconv_out_channels=channels,
                         upscale_factor=upscale_factor, final_kernel_size=final_kernel_size, p=p)
        self.final = nn.Sequential(nn.ReflectionPad2d(final_kernel_size // 2),
                                   nn.Conv2d(channels, out_channels, final_kernel_size, padding=0))

    def forward(self, x):
        return self.final(self.deconv(x))


class HighResNetSR(nn.Module):
    lr_multiple = 1          # fully convolutional, bilinear upsampling: any LR size

    def __init__(self, in_channels=1, out_channels=3, hidden_channels=64, enc_num_layers=2,
                 kernel_size=3, upscale_factor=4, final_kernel_size=1, p=0.0):
        super().__init__()
        self.encode = Encoder(in_channels=in_channels, num_layers=enc_num_layers,
                              kernel_size=kernel_size, channel_size=hidden_channels, p=p)
        self.fuse = FusionBlock(input_channels=hidden_channels, kernel_size=kernel_size, p=p)
        self.decode = Decoder3(channels=hidden_channels, upscale_factor=upscale_factor,
                               out_channels=out_channels, final_kernel_size=final_kernel_size, p=p)
        self.inactive_prefixes = ("fuse.",)   # unused at K = 1

    def forward(self, lr):
        hidden = self.encode(lr)               # (B, C_h, H, W)
        hidden = self.fuse(hidden[:, None])    # K = 1 view: fusion loop is skipped
        return self.decode(hidden)             # (B, 3, 4H, 4W)
