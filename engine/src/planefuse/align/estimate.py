"""Per-pair transform estimation (SPEC §6 steps a-d), at proxy resolution."""

from __future__ import annotations

from dataclasses import dataclass

from planefuse.align.initial import estimate_scale_rotation, estimate_translation
from planefuse.align.proxy import make_proxy
from planefuse.align.refine import brightness_gain, refine_ecc
from planefuse.align.transforms import (
    scale_transform_to_resolution,
    similarity_matrix,
    translation_matrix,
)
from planefuse.backend import Device, ops

import numpy as np
import torch

# Rec. 709 luma weights (R, G, B), matching ops.rgb_to_luminance.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _luminance_full(frame: np.ndarray) -> np.ndarray:
    """(H, W, 3) float32 RGB -> contiguous (H, W) float32 luminance for ECC."""
    return np.ascontiguousarray(frame.astype(np.float32) @ _LUMA)


def _alignment_plausible(m: np.ndarray, model: str, shape: tuple[int, int]) -> bool:
    h, w = shape
    corners = np.array(
        [[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]],
        dtype=np.float64,
    )
    mapped = (m @ corners.T).T
    if not (
        bool(np.all(np.isfinite(m)))
        and abs(float(np.linalg.det(m))) > 1e-8
        and bool(np.all(np.abs(mapped[:, 2]) > 1e-6))
    ):
        return False
    if model == "perspective":
        mapped_xy = mapped[:, :2] / mapped[:, 2:3]
        displacement = np.linalg.norm(mapped_xy - corners[:, :2], axis=1)
        return bool(np.max(displacement) < np.hypot(h, w) * 0.5)
    scale = float(np.sqrt(abs(np.linalg.det(m[:2, :2]))))
    return 0.8 < scale < 1.25


@dataclass
class PairResult:
    matrix: np.ndarray      # 3x3 output->input at proxy or full resolution
    correlation: float      # ECC correlation (quality gate input)
    gain: float             # brightness gain for `b` relative to `a`
    proxy_factor: float     # multiply translation by this for full-res


def _to_tensor3(m: np.ndarray, device: Device) -> torch.Tensor:
    return torch.from_numpy(m.astype(np.float32)).to(device.torch_device)


def estimate_pair(frame_a: np.ndarray, frame_b: np.ndarray, device: Device,
                  max_long_edge: int = 2048, model: str = "similarity",
                  normalize_brightness: bool = True,
                  normalize_scene_linear: bool = False,
                  refine_full_res: bool = False) -> PairResult:
    """Estimate the transform warping frame_b onto frame_a (consecutive pair).

    With `refine_full_res`, a warm-started ECC polish is run at the frames'
    native resolution and the returned matrix is full-res (proxy_factor 1.0).
    """
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
    if not _alignment_plausible(m, model, (h, w)):
        m = init.copy()
        if model == "translation":
            m[:2, :2] = np.eye(2)
        corr = 0.0

    # Full-res polish: scale the proxy estimate up and run one warm-started ECC
    # solve at native resolution. Every pair returns a full-res matrix so the
    # chain stays consistent; a diverged polish falls back to the scaled proxy.
    if refine_full_res and factor > 1.0:
        m_full = scale_transform_to_resolution(m, factor)
        la, lb = _luminance_full(frame_a), _luminance_full(frame_b)
        fh, fw = la.shape
        cxf, cyf = (fw - 1) / 2.0, (fh - 1) / 2.0
        m_ref, corr_ref = refine_ecc(
            la, lb, init=m_full, cx=cxf, cy=cyf, levels=1, model=model
        )
        if model == "translation":
            m_ref[:2, :2] = np.eye(2)
        if _alignment_plausible(m_ref, model, (fh, fw)):
            m, corr = m_ref, corr_ref
        else:
            m = m_full  # rejected polish: keep the scaled proxy estimate (and its corr)
        factor = 1.0

    gain = 1.0
    if normalize_brightness:
        mask = np.ones((h, w), dtype=bool)
        gain = brightness_gain(a_np, b_np, mask)
    return PairResult(matrix=m, correlation=corr, gain=gain, proxy_factor=factor)
