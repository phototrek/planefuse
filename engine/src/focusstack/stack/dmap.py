"""DMap: depth-map focus stacking (SPEC §7.2).

Streaming argmax of a per-frame sharpness map -> integer index map + max-sharpness
map; the index map is then contrast-thresholded, edge-aware smoothed, and its
undecided regions diffusion-filled, before a second streaming pass composites the
result with fractional-index blending. Memory is O(image), not O(frames).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from focusstack.backend import Device, ops
from focusstack.stack.base import (
    CancelFn,
    FrameSource,
    ParamSpec,
    ProgressFn,
    StackResult,
    register,
)


def _sharp_of(frame_np: np.ndarray, device: Device, radius: int) -> torch.Tensor:
    t = ops.to_tensor(frame_np, device)
    gray = ops.rgb_to_luminance(t)
    return ops.sharpness_map(gray, radius).squeeze(0)  # (H, W)


def dmap_fold(source: FrameSource, device: Device, radius: int,
              masks: FrameSource | None = None,
              progress=None, cancel=None) -> tuple[np.ndarray, np.ndarray]:
    """Streaming argmax fold. Returns (index map int32 (H,W), max-sharpness (H,W) float32)."""
    n = len(source)
    best_idx: torch.Tensor | None = None
    best_sharp: torch.Tensor | None = None
    for i in range(n):
        if cancel is not None and cancel():
            raise InterruptedError("dmap fold cancelled")
        if progress is not None:
            progress(f"DMap sharpness {i + 1}/{n}", i / max(2 * n, 1))
        s = _sharp_of(source.read(i), device, radius)  # (H, W)
        if masks is not None:
            m = torch.from_numpy(np.ascontiguousarray(masks.read(i))).to(device.torch_device)
            # Invalid pixels get -inf so they can never win the argmax, even on a
            # tie with a valid zero-sharpness (flat) pixel.
            s = torch.where(m.bool(), s, torch.full_like(s, float("-inf")))
        if best_idx is None or best_sharp is None:
            best_idx = torch.zeros(s.shape, dtype=torch.int32, device=s.device)
            best_sharp = s
            continue
        better = s > best_sharp
        best_idx = torch.where(better, torch.full_like(best_idx, i), best_idx)
        best_sharp = torch.where(better, s, best_sharp)
    assert best_idx is not None and best_sharp is not None
    return best_idx.cpu().numpy(), best_sharp.cpu().numpy()


def refine_index(index: np.ndarray, max_sharp: np.ndarray, device: Device,
                 contrast_percentile: float, smoothing_radius: int) -> np.ndarray:
    """SPEC §7.2 steps 3-4. index (H,W) int/float, max_sharp (H,W) float ->
    smoothed fractional index map (H,W) float32 with undecided regions filled."""
    idx_t = torch.from_numpy(index.astype(np.float32)).unsqueeze(0).to(device.torch_device)
    sharp_t = torch.from_numpy(max_sharp.astype(np.float32)).unsqueeze(0).to(device.torch_device)
    thr = float(np.percentile(max_sharp, contrast_percentile))
    decided = sharp_t > thr
    guide = sharp_t / (sharp_t.max() + 1e-8)
    smoothed = ops.guided_filter(guide, idx_t, radius=smoothing_radius, eps=1e-4)
    if not bool(decided.any()):
        # Degenerate / uniform contrast: nothing exceeds the threshold, so there is
        # nothing to diffuse from. Treat the whole map as decided.
        return smoothed.squeeze(0).cpu().numpy()
    filled = ops.masked_diffuse(smoothed, decided, radius=max(2, smoothing_radius // 2), iters=64)
    return filled.squeeze(0).cpu().numpy()


def composite(source: FrameSource, frac_index: np.ndarray, device: Device,
              masks: FrameSource | None = None, progress=None, cancel=None) -> np.ndarray:
    """Second streaming pass (SPEC §7.2 step 5): for each pixel, blend the two
    frames bracketing its fractional index. frac_index (H,W) float in [0, n-1]."""
    n = len(source)
    fi = np.clip(frac_index, 0, n - 1)
    lo = np.floor(fi).astype(np.int64)
    w_hi = (fi - lo).astype(np.float32)
    out = np.zeros((*fi.shape, 3), dtype=np.float32)
    for i in range(n):
        if cancel is not None and cancel():
            raise InterruptedError("dmap composite cancelled")
        if progress is not None:
            progress(f"DMap composite {i + 1}/{n}", (n + i) / max(2 * n, 1))
        frame = source.read(i)
        w = np.zeros(fi.shape, dtype=np.float32)
        w += np.where(lo == i, 1.0 - w_hi, 0.0)
        w += np.where(lo + 1 == i, w_hi, 0.0)
        out += frame * w[..., None]
    return out


@register("dmap")
class DMap:
    @staticmethod
    def params() -> list[ParamSpec]:
        return [
            ParamSpec("estimation_radius", "Estimation radius", "int", default=8, min=2, max=40,
                      tooltip="Radius (px) of the local sharpness window."),
            ParamSpec("contrast_threshold", "Contrast threshold", "float", default=7.0, min=0.0,
                      max=50.0, tooltip="Percentile of max-sharpness below which a pixel is "
                                        "undecided and filled from neighbours."),
            ParamSpec("smoothing_radius", "Smoothing radius", "int", default=16, min=1, max=64,
                      tooltip="Edge-aware (guided-filter) smoothing radius for the depth map."),
        ]

    def run(self, source: FrameSource, device: Device, params: dict[str, Any],
            progress: ProgressFn | None = None, cancel: CancelFn | None = None,
            masks: FrameSource | None = None) -> StackResult:
        radius = int(params.get("estimation_radius", 8))
        pct = float(params.get("contrast_threshold", 7.0))
        smooth = int(params.get("smoothing_radius", 16))
        index, max_sharp = dmap_fold(source, device, radius, masks, progress, cancel)
        frac = refine_index(index, max_sharp, device, pct, smooth)
        image = composite(source, frac, device, masks, progress, cancel)
        if progress is not None:
            progress("done", 1.0)
        n = len(source)
        depth = (frac / max(n - 1, 1)).astype(np.float32)
        return StackResult(image=image, aux={"depth": depth, "index": frac.astype(np.float32)})
