"""ISG-Net reference architecture aligned with the manuscript's module definitions."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .modules import (ConvBlock, VisualMambaBlock, build_sensitive_attention,
                      build_cross_stream_module)


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, use_mamba, kernel, attention,
                 interaction, alpha_init, dual, forward_type):
        super().__init__()
        def branch():
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                ConvBlock(out_ch, out_ch),
                VisualMambaBlock(out_ch, forward_type) if use_mamba else nn.Identity(),
            )
        self.inv = branch()
        self.sen = branch() if dual else None
        self.attn = build_sensitive_attention(attention, out_ch, kernel) if dual else None
        self.cross = build_cross_stream_module(interaction, out_ch, alpha_init) if dual else None

    def forward(self, fi, fs):
        xi = self.inv(fi)
        xs = self.attn(self.sen(fs)) if self.sen is not None else torch.zeros_like(xi)
        residual = torch.zeros_like(xi)
        if self.cross is not None:
            xi, residual = self.cross(xi, xs)
        return xi, xs, residual


class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, kernel, attention, interaction, alpha_init, dual):
        super().__init__()
        self.up_proj = nn.Conv2d(in_ch, skip_ch, 1)
        self.fuse = ConvBlock(2 * skip_ch, skip_ch)
        self.main = ConvBlock(skip_ch, skip_ch)
        self.guide = build_sensitive_attention(attention, skip_ch, kernel) if dual else None
        self.cross = build_cross_stream_module(interaction, skip_ch, alpha_init) if dual else None

    def forward(self, x, skip_i, skip_s):
        up = self.up_proj(F.interpolate(x, size=skip_i.shape[-2:], mode="bilinear",
                                        align_corners=False))
        x = self.main(self.fuse(torch.cat([up, skip_i], 1)))
        residual = torch.zeros_like(x)
        if self.cross is not None:
            x, residual = self.cross(x, self.guide(skip_s))
        return x, residual


class ISGNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_stages=4,
                 alpha_init=0.5, local_contrast_kernel=7, use_dual_stream=True,
                 mamba_stages=None, sensitive_attention_encoder="contrast_guided",
                 sensitive_attention_decoder="contrast_guided",
                 encoder_cross_stream_mode="purification",
                 decoder_cross_stream_mode="purification", projection_dim=64,
                 vss_forward_type="v05"):
        super().__init__()
        if num_stages != 4:
            raise ValueError("The manuscript architecture has four encoder stages")
        self.use_dual_stream = use_dual_stream
        widths = [base_channels * 2 ** (i + 1) for i in range(num_stages)]
        stages = [False, False, True, True] if mamba_stages is None else mamba_stages
        if len(stages) != num_stages:
            raise ValueError("mamba_stages must contain four flags")
        self.stem_i = ConvBlock(in_channels, base_channels)
        self.stem_s = ConvBlock(in_channels, base_channels) if use_dual_stream else None
        self.encoders = nn.ModuleList([
            EncoderStage(base_channels if i == 0 else widths[i-1], width, stages[i],
                         local_contrast_kernel, sensitive_attention_encoder,
                         encoder_cross_stream_mode, alpha_init, use_dual_stream,
                         vss_forward_type)
            for i, width in enumerate(widths)
        ])
        # Each stage projector is shared across streams and across paired views.
        self.projectors = nn.ModuleList([nn.Linear(width, projection_dim) for width in widths])
        skips = list(reversed([base_channels] + widths[:-1]))
        inputs = [widths[-1]] + skips[:-1]
        self.decoders = nn.ModuleList([
            DecoderStage(a, b, local_contrast_kernel, sensitive_attention_decoder,
                         decoder_cross_stream_mode, alpha_init, use_dual_stream)
            for a, b in zip(inputs, skips)
        ])
        self.head = nn.Conv2d(base_channels, 1, 1)

    def _descriptors(self, features):
        result = []
        for projector, feature in zip(self.projectors, features):
            h = projector(F.adaptive_avg_pool2d(feature, 1).flatten(1))
            result.append(h / (h.norm(p=2, dim=1, keepdim=True) + 1e-12))
        return result

    def forward_single(self, image):
        fi = self.stem_i(image)
        fs = self.stem_s(image) if self.stem_s is not None else torch.zeros_like(fi)
        inv, sen, enc_res = [], [], []
        inv_skips, sen_skips = [fi], [fs]
        for encoder in self.encoders:
            fi, fs, residual = encoder(fi, fs)
            inv.append(fi); sen.append(fs); enc_res.append(residual)
            inv_skips.append(fi); sen_skips.append(fs)
        x, bottleneck = fi, fi
        dec_res = []
        for decoder, skip_i, skip_s in zip(self.decoders, inv_skips[-2::-1], sen_skips[-2::-1]):
            x, residual = decoder(x, skip_i, skip_s)
            dec_res.append(residual)
        return {"pred": self.head(x).sigmoid(), "inv_feats": inv, "sen_feats": sen,
                "inv_desc": self._descriptors(inv), "sen_desc": self._descriptors(sen),
                "encoder_residuals": enc_res, "decoder_residuals": dec_res,
                "bottleneck": bottleneck, "decoder_feat": x}

    def forward(self, x_o, x_a=None):
        result = {"original": self.forward_single(x_o)}
        if x_a is not None:
            result["augmented"] = self.forward_single(x_a)
        return result
