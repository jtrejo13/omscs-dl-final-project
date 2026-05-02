# ------------------------------------------------------------------------
# Adapted from NAFNet (https://github.com/megvii-research/NAFNet)
# "Simple Baselines for Image Restoration", Chen et al., ECCV 2022
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.arch_util import LayerNorm2d

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GELUGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * F.gelu(x2)


class AsymmetricSimpleGate(nn.Module):
    """SimpleGate with an unequal channel split.

    Splits the input into (ratio_big * c, ratio_small * c) along channels,
    projects the small half up to ratio_big * c with a 1x1 conv, then
    multiplies element-wise. Output channels: ratio_big * c.
    """

    def __init__(self, c, ratio_big=2, ratio_small=1):
        super().__init__()
        self.big = ratio_big * c
        self.small = ratio_small * c
        self.proj = nn.Conv2d(self.small, self.big, kernel_size=1, bias=True)

    def forward(self, x):
        x_big, x_small = torch.split(x, [self.big, self.small], dim=1)
        return x_big * self.proj(x_small)


class SkipGate(nn.Module):
    """Per-channel sigmoid gate over enc_skip, conditioned on concat([x, enc_skip]).

    Output: x + sigmoid(conv1x1([x, enc_skip])) * enc_skip.
    """

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Conv2d(2 * c, c, kernel_size=1, bias=True)

    def forward(self, x, enc_skip):
        g = torch.sigmoid(self.gate(torch.cat([x, enc_skip], dim=1)))
        return x + g * enc_skip


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlockA(NAFBlock):
    """Variant A: GELU gate (x1 * gelu(x2)) instead of SimpleGate (x1 * x2)."""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__(c, DW_Expand, FFN_Expand, drop_out_rate)
        self.sg = GELUGate()


class NAFBlockB(NAFBlock):
    """Variant B: SCA (Simplified Channel Attention) removed entirely."""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__(c, DW_Expand, FFN_Expand, drop_out_rate)
        del self.sca

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlockC(NAFBlock):
    """Variant C: nn.BatchNorm2d in place of LayerNorm2d for norm1/norm2."""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__(c, DW_Expand, FFN_Expand, drop_out_rate)
        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)


class NAFBlockE(NAFBlock):
    """Variant E: AsymmetricSimpleGate with 2:1 split (instead of 1:1).

    Both the depthwise gate and the FFN gate become asymmetric. DW_Expand and
    FFN_Expand are set to 3 so the gate input splits cleanly into (2c, c);
    the smaller half is projected up to 2c via a 1x1 conv before the
    element-wise multiplication. Output of each gate has 2c channels, which
    feeds wider conv3 / conv5 / SCA layers than baseline.
    """

    def __init__(self, c, DW_Expand=3, FFN_Expand=3, drop_out_rate=0.):
        nn.Module.__init__(self)
        dw_channel = c * DW_Expand
        ffn_channel = c * FFN_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sg = AsymmetricSimpleGate(c, ratio_big=2, ratio_small=1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * c, 2 * c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.conv3 = nn.Conv2d(2 * c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg_ffn = AsymmetricSimpleGate(c, ratio_big=2, ratio_small=1)
        self.conv5 = nn.Conv2d(2 * c, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg_ffn(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], block_cls=None, skip_fusion="add"):
        super().__init__()

        if block_cls is None:
            block_cls = NAFBlock

        if skip_fusion not in ("add", "gated"):
            raise ValueError(f"Unknown skip_fusion: {skip_fusion!r} (expected 'add' or 'gated')")
        self.skip_fusion = skip_fusion

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[block_cls(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[block_cls(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[block_cls(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        if skip_fusion == "gated":
            self.skip_gates = nn.ModuleList()
            chan_g = width
            for _ in enc_blk_nums:
                chan_g *= 2
            for _ in dec_blk_nums:
                chan_g //= 2
                self.skip_gates.append(SkipGate(chan_g))

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        gates = self.skip_gates if self.skip_fusion == "gated" else [None] * len(self.decoders)
        for decoder, up, enc_skip, gate in zip(self.decoders, self.ups, encs[::-1], gates):
            x = up(x)
            x = gate(x, enc_skip) if gate is not None else (x + enc_skip)
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

