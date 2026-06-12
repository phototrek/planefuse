"""Device-agnostic image operations — the one true implementation (SPEC §4).

All image tensors are float32 (C, H, W); integer maps are (H, W).
The device is wherever the input tensor lives; ops never move tensors
between devices except documented explicit CPU round-trips for MPS gaps.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from focusstack.backend.device import Device


def to_tensor(arr: np.ndarray, device: Device) -> torch.Tensor:
    """(H, W, C) float32 numpy -> (C, H, W) float32 tensor on device."""
    if arr.ndim != 3:
        raise ValueError(f"expected (H, W, C) array, got shape {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).to(device.torch_device)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor -> (H, W, C) float32 numpy on host."""
    return t.detach().permute(1, 2, 0).contiguous().cpu().numpy()


def gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding. img: (C, H, W)."""
    k = gaussian_kernel1d(sigma, img.device)
    r = (k.numel() - 1) // 2
    c = img.shape[0]
    x = img.unsqueeze(0)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous(), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous(), groups=c)
    return x.squeeze(0)


def downsample2(img: torch.Tensor) -> torch.Tensor:
    """Anti-aliased x2 downsample: blur(sigma=1.0) then stride-2 decimation."""
    return gaussian_blur(img, 1.0)[..., ::2, ::2]


def upsample_to(img: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Bilinear upsample to an exact (H, W)."""
    return F.interpolate(
        img.unsqueeze(0), size=shape, mode="bilinear", align_corners=False
    ).squeeze(0)


def box_filter3(img: torch.Tensor) -> torch.Tensor:
    """3x3 mean filter with reflect padding. img: (C, H, W)."""
    c = img.shape[0]
    w = torch.full((c, 1, 3, 3), 1.0 / 9.0, dtype=torch.float32, device=img.device)
    x = F.pad(img.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x, w, groups=c).squeeze(0)


def median_filter2d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """(2r+1)^2 median on an integer (H, W) map via unfold (SPEC §4: no native 2D median).

    Median of an odd-count window of integers is an integer, so the float
    round-trip is exact for the int32 winner maps this is used on.
    """
    if radius < 1:
        return x
    k = 2 * radius + 1
    f = x.to(torch.float32).unsqueeze(0).unsqueeze(0)
    f = F.pad(f, (radius, radius, radius, radius), mode="replicate")
    patches = f.unfold(2, k, 1).unfold(3, k, 1)
    med = patches.reshape(*patches.shape[:4], -1).median(dim=-1).values
    return med.squeeze(0).squeeze(0).round().to(x.dtype)


def local_energy(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel energy = 3x3-smoothed mean |x| over channels -> (1, H, W). SPEC §7.1."""
    return box_filter3(x.abs().mean(dim=0, keepdim=True))
