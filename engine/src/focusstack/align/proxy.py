"""Downscaled luminance proxy for alignment estimation (SPEC §6 step 3a)."""

from __future__ import annotations

import numpy as np
import torch

from focusstack.backend import Device, ops


def make_proxy(frame: np.ndarray, device: Device, max_long_edge: int = 2048
               ) -> tuple[torch.Tensor, float]:
    """(H, W, 3) numpy -> ((1, h, w) luminance tensor, downscale_factor).

    factor = original_long_edge / proxy_long_edge >= 1.0 (never upscales).
    """
    t = ops.to_tensor(frame, device)
    lum = ops.rgb_to_luminance(t)
    long_edge = max(lum.shape[-2:])
    if long_edge <= max_long_edge:
        return lum, 1.0
    factor = long_edge / max_long_edge
    new_h = max(1, round(lum.shape[-2] / factor))
    new_w = max(1, round(lum.shape[-1] / factor))
    return ops.upsample_to(lum, (new_h, new_w)), float(long_edge) / float(max(new_h, new_w))
