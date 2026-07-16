"""Full-resolution warp + validity mask (SPEC §6 step 5)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from planefuse.align.transforms import scale_transform_to_resolution
from planefuse.backend import Device, ops


def warp_full(frame: np.ndarray, matrix_proxy: np.ndarray, device: Device,
              out_shape: tuple[int, int], proxy_factor: float,
              interp: str = "bilinear") -> tuple[np.ndarray, np.ndarray]:
    """Warp a full-res (H, W, 3) frame by a proxy-resolution transform.

    Returns (aligned (H, W, 3) float32, validity mask (H, W) bool). Out-of-frame
    pixels are edge-clamped in the image and False in the mask, so stacking can
    exclude them (PMax forces their energy to -inf via masks=).
    """
    m_full = scale_transform_to_resolution(matrix_proxy, proxy_factor)
    t = ops.to_tensor(frame, device)
    m_t = torch.from_numpy(m_full.astype(np.float32)).to(device.torch_device)
    aligned = ops.warp(t, m_t, out_shape=out_shape, interp=interp)
    # validity: warp an all-ones plane with zero padding; ~1 inside, <1 at edges
    ones = torch.ones((1, frame.shape[0], frame.shape[1]), device=device.torch_device)
    grid = ops._normalized_grid(  # noqa: SLF001
        m_t,
        out_shape,
        (frame.shape[0], frame.shape[1]),
    )
    sampled = F.grid_sample(ones.unsqueeze(0), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    mask = (sampled.squeeze(0).squeeze(0) > 0.999).cpu().numpy()
    return ops.to_numpy(aligned), mask
