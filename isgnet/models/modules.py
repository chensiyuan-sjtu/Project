"""Convolution, contrast attention, and sensitive-guided residual correction."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def local_mean(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Local windows must have a positive odd size")
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(x, (pad,) * 4, mode="replicate"), kernel_size, stride=1)


def normalized_local_contrast(x: torch.Tensor, kernel_size: int = 7,
                              eps: float = 1e-3) -> torch.Tensor:
    """Local standard deviation / (local mean + eps), on a nonnegative proxy."""
    mean = local_mean(x, kernel_size)
    variance = (local_mean(x.square(), kernel_size) - mean.square()).clamp_min(0)
    # Match sqrt(max(variance, 0)) while keeping the derivative finite at zero.
    std = torch.where(variance > 0, variance.clamp_min(1e-12).sqrt(), 0.0)
    return std / (mean + eps)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch normalization, GELU, and a residual skip."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


@lru_cache(maxsize=1)
def _vmamba_class():
    """Load the upstream standalone vmamba.py from an explicit dependency location."""
    root = os.environ.get("VMAMBA_ROOT")
    try:
        if root:
            path = Path(root).expanduser() / "vmamba.py"
            if not path.is_file():
                raise ImportError(f"VMAMBA_ROOT does not contain vmamba.py: {path}")
            name = "_isgnet_vmamba_backend"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import VMamba from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(name, None)
                raise
        else:
            module = importlib.import_module("vmamba")
        return module.VSSBlock
    except (ImportError, AttributeError, NameError) as exc:
        raise ImportError(
            "Install upstream VMamba dependencies and set VMAMBA_ROOT to its checkout "
            "(the directory containing vmamba.py). Install the listed dependencies. The full model "
            "requires a real VSSBlock; there is no silent convolutional fallback."
        ) from exc


class VisualMambaBlock(nn.Module):
    """Adapter to upstream VMamba VSSBlock, never a convolutional substitute."""
    def __init__(self, channels: int, forward_type: str = "v05"):
        super().__init__()
        self.block = _vmamba_class()(
            hidden_dim=channels, channel_first=True, drop_path=0.0,
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_conv=3,
            ssm_drop_rate=0.0, mlp_drop_rate=0.0, forward_type=forward_type,
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(),
                                 nn.Conv2d(hidden, channels, 1))

    def forward(self, x):
        weights = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) +
                                self.mlp(F.adaptive_max_pool2d(x, 1)))
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        descriptor = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(descriptor))


class ContrastGuidedAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1))

    def forward(self, x):
        low, high = x.amin((-2, -1), keepdim=True), x.amax((-2, -1), keepdim=True)
        proxy = (x - low) / (high - low + 1e-6)
        contrast = normalized_local_contrast(proxy, self.kernel_size, eps=1e-3)
        weights = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(contrast, 1)) +
                                self.mlp(F.adaptive_max_pool2d(contrast, 1)))
        return x * weights


class AdditiveInteraction(nn.Module):
    """Concatenation/projection control used in the naive dual-stream ablation."""
    def __init__(self, channels: int, use_cbam: bool = False):
        super().__init__()
        layers = [nn.Conv2d(2 * channels, channels, 1)]
        if use_cbam:
            layers += [ChannelAttention(channels), SpatialAttention()]
        self.fuse = nn.Sequential(*layers)

    def forward(self, main_feat, guide_feat):
        fused = self.fuse(torch.cat([main_feat, guide_feat], 1))
        return fused, fused - main_feat


class SensitiveGuidedPurification(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 0.5):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1), nn.GELU(),
            nn.BatchNorm2d(channels), nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, main_feat, guide_feat):
        residual = self.predictor(torch.cat([main_feat, guide_feat], 1))
        return main_feat - self.alpha * residual, residual


def build_sensitive_attention(mode, channels, kernel_size):
    if mode == "contrast_guided":
        return ContrastGuidedAttention(channels, kernel_size=kernel_size)
    if mode == "cbam":
        return nn.Sequential(ChannelAttention(channels), SpatialAttention())
    if mode == "channel":
        return ChannelAttention(channels)
    if mode == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown attention: {mode}")


def build_cross_stream_module(mode, channels, alpha_init):
    if mode == "purification":
        return SensitiveGuidedPurification(channels, alpha_init)
    if mode in ("concat", "additive", "cat_cbam"):
        return AdditiveInteraction(channels, use_cbam=mode == "cat_cbam")
    if mode == "none":
        return None
    raise ValueError(f"Unknown stream interaction: {mode}")
