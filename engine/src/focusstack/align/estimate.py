"""Per-pair transform estimation (SPEC §6 steps a-d), at proxy resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from focusstack.backend import Device, ops
from focusstack.align.initial import estimate_scale_rotation, estimate_translation
from focusstack.align.proxy import make_proxy
from focusstack.align.refine import brightness_gain, refine_ecc
from focusstack.align.transforms import similarity_matrix, translation_matrix


@dataclass
class PairResult:
    matrix: np.ndarray      # 3x3 output->input at PROXY resolution
    correlation: float      # ECC correlation (quality gate input)
    gain: float             # brightness gain for `b` relative to `a`
    proxy_factor: float     # multiply translation by this for full-res


def _to_tensor3(m: np.ndarray, device: Device) -> torch.Tensor:
    return torch.from_numpy(m.astype(np.float32)).to(device.torch_device)


def estimate_pair(frame_a: np.ndarray, frame_b: np.ndarray, device: Device,
                  max_long_edge: int = 2048, model: str = "similarity",
                  normalize_brightness: bool = True) -> PairResult:
    """Estimate the transform warping frame_b onto frame_a (consecutive pair)."""
    pa, factor = make_proxy(frame_a, device, max_long_edge)
    pb, _ = make_proxy(frame_b, device, max_long_edge)
    h, w = pa.shape[-2:]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    if model == "translation":
        dy, dx = estimate_translation(pa, pb)
        init = translation_matrix(dx, dy)
    else:
        scale, angle = estimate_scale_rotation(pa, pb)
        sr = similarity_matrix(scale, angle, 0.0, 0.0, cx, cy)
        pb_corr = ops.warp(pb, _to_tensor3(sr, device), out_shape=(h, w), interp="bilinear")
        dy, dx = estimate_translation(pa, pb_corr)
        init = translation_matrix(dx, dy) @ sr

    a_np = pa.squeeze(0).cpu().numpy()
    b_np = pb.squeeze(0).cpu().numpy()
    m, corr = refine_ecc(a_np, b_np, init=init, cx=cx, cy=cy)
    if model == "translation":
        m[:2, :2] = np.eye(2)  # enforce translation model

    gain = 1.0
    if normalize_brightness:
        mask = np.ones((h, w), dtype=bool)
        gain = brightness_gain(a_np, b_np, mask)
    return PairResult(matrix=m, correlation=corr, gain=gain, proxy_factor=factor)
