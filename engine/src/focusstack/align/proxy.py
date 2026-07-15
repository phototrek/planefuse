"""Downscaled luminance proxy for alignment estimation (SPEC §6 step 3a)."""

from __future__ import annotations

import numpy as np
import torch

from focusstack.backend import Device, ops


def scene_linear_proxy_luminance(frame: np.ndarray) -> np.ndarray:
    """Build a neutral, robust measurement proxy without altering fusion data."""

    neutral = np.asarray(frame, dtype=np.float32).mean(axis=2)
    low, high = np.percentile(neutral, (0.5, 99.5))
    span = float(high - low)
    if span <= np.finfo(np.float32).eps:
        return np.zeros_like(neutral)
    return np.clip((neutral - low) / span, 0.0, 1.0).astype(np.float32)


def make_proxy(
    frame: np.ndarray,
    device: Device,
    max_long_edge: int = 2048,
    *,
    normalize_scene_linear: bool = False,
) -> tuple[torch.Tensor, float]:
    """(H, W, 3) numpy -> ((1, h, w) luminance tensor, downscale_factor).

    factor = original_long_edge / proxy_long_edge >= 1.0 (never upscales).
    """
    if normalize_scene_linear:
        neutral = scene_linear_proxy_luminance(frame)
        lum = torch.from_numpy(neutral[None]).to(device.torch_device)
    else:
        t = ops.to_tensor(frame, device)
        lum = ops.rgb_to_luminance(t)
    long_edge = max(lum.shape[-2:])
    if long_edge <= max_long_edge:
        return lum, 1.0
    factor = long_edge / max_long_edge
    new_h = max(1, round(lum.shape[-2] / factor))
    new_w = max(1, round(lum.shape[-1] / factor))
    return ops.upsample_to(lum, (new_h, new_w)), float(long_edge) / float(max(new_h, new_w))
