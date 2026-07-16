"""Per-pair transform estimation (SPEC §6 steps a-d), at proxy resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from planefuse.backend import Device, ops
from planefuse.align.initial import estimate_scale_rotation, estimate_translation
from planefuse.align.proxy import make_proxy
from planefuse.align.refine import brightness_gain, refine_ecc
from planefuse.align.transforms import similarity_matrix, translation_matrix


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
                  normalize_brightness: bool = True,
                  normalize_scene_linear: bool = False) -> PairResult:
    """Estimate the transform warping frame_b onto frame_a (consecutive pair)."""
    pa, factor = make_proxy(
        frame_a,
        device,
        max_long_edge,
        normalize_scene_linear=normalize_scene_linear,
    )
    pb, _ = make_proxy(
        frame_b,
        device,
        max_long_edge,
        normalize_scene_linear=normalize_scene_linear,
    )
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
    m, corr = refine_ecc(a_np, b_np, init=init, cx=cx, cy=cy, model=model)
    if model == "translation":
        m[:2, :2] = np.eye(2)  # enforce translation model

    # Divergence guard (SPEC §6 quality gate): ECC on a heavily-defocused or
    # low-texture consecutive pair can diverge to an implausible warp while still
    # reporting a high correlation. A real focus stack's inter-frame scale change
    # is sub-percent (focus breathing), so a similarity scale far from 1.0 is
    # physically impossible — reject it, fall back to the gentler initial guess,
    # and report zero correlation so the quality gate can flag/drop the pair.
    sc = float(np.sqrt(abs(np.linalg.det(m[:2, :2]))))
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=np.float64)
    mapped = (m @ corners.T).T
    valid_projective = bool(
        bool(np.all(np.isfinite(m)))
        and abs(float(np.linalg.det(m))) > 1e-8
        and bool(np.all(np.abs(mapped[:, 2]) > 1e-6))
    )
    if valid_projective and model == "perspective":
        mapped_xy = mapped[:, :2] / mapped[:, 2:3]
        displacement = np.linalg.norm(mapped_xy - corners[:, :2], axis=1)
        valid_projective = bool(np.max(displacement) < np.hypot(h, w) * 0.5)
    if (model == "perspective" and not valid_projective) or (
        model != "perspective" and not (0.8 < sc < 1.25)
    ):
        m = init.copy()
        if model == "translation":
            m[:2, :2] = np.eye(2)
        corr = 0.0

    gain = 1.0
    if normalize_brightness:
        mask = np.ones((h, w), dtype=bool)
        gain = brightness_gain(a_np, b_np, mask)
    return PairResult(matrix=m, correlation=corr, gain=gain, proxy_factor=factor)
