"""PMax: Laplacian pyramid, max-energy selection, streaming fold (SPEC §7.1).

Memory complexity is O(image_size), not O(frames * image_size): only the
current-best coefficient maps and the current frame are live at any time.
A second streaming pass corrects halo artefacts by re-reading frames
(streaming — no buffering) and pasting coefficients for smoothed winner
regions that changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from planefuse.backend import Device, ops
from planefuse.stack.base import (
    CancelFn,
    FrameSource,
    ParamSpec,
    ProgressFn,
    StackResult,
    register,
)
from planefuse.stack.pyramid import build_laplacian, collapse_laplacian, pyramid_depth
from planefuse.stack.sources import read_tensor

_NEG_INF = float("-inf")


@dataclass
class _FoldState:
    lap: list[torch.Tensor]  # best coefficients per level (C, h_l, w_l)
    energy: list[torch.Tensor]  # best energy per level (1, h_l, w_l)
    winner: list[torch.Tensor]  # winning frame index per level (h_l, w_l) int32
    res_num: torch.Tensor  # residual weighted sum (C, h_r, w_r)
    res_den: torch.Tensor  # residual weight sum   (1, h_r, w_r)


def _read_frame(source: FrameSource, idx: int) -> np.ndarray | torch.Tensor:
    """The frame as the tensor the source holds, else as numpy."""
    held = read_tensor(source, idx)
    return held if held is not None else source.read(idx)


def _frame_pyramid(
    frame_np: np.ndarray | torch.Tensor, device: Device, depth: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Convert a frame ((H, W, C) numpy, or a held (C, H, W) tensor, which is
    used in place when already on `device`) to a Laplacian pyramid with
    per-level energy maps.

    Returns:
        lap       - Laplacian coefficient tensors (C, h_l, w_l)
        energies  - local energy tensors (1, h_l, w_l)
        residual  - coarsest Gaussian level (C, h_r, w_r)
        res_weight - soft weight for residual contribution (1, h_r, w_r)
    """
    if isinstance(frame_np, torch.Tensor):
        t = frame_np.to(device.torch_device)
    else:
        t = ops.to_tensor(frame_np, device)
    lap, residual = build_laplacian(t, depth)
    energies = [ops.local_energy(lvl) for lvl in lap]
    # Residual weight: smoothed magnitude of high-frequency content at coarsest
    # level — frames with more detail there contribute more to the weighted avg.
    res_weight = ops.box_filter3((residual - ops.box_filter3(residual)).abs().mean(0, keepdim=True)) + 1e-6
    return lap, energies, residual, res_weight


class _RangeBounds:
    """Per-pixel, per-channel minimum and maximum over the aligned frames.

    A fused pixel cannot honestly be darker or brighter than every source frame at
    that place; PMax can overshoot there, which shows as a dark rim along a
    high-contrast edge. The bounds are built in the fold pass from the frames it
    already reads, and `apply` limits the collapsed result to them.
    """

    def __init__(self) -> None:
        self.lo: torch.Tensor | None = None
        self.hi: torch.Tensor | None = None

    def update(self, frame: np.ndarray | torch.Tensor, device: Device, valid: torch.Tensor | None) -> None:
        t = frame.to(device.torch_device) if isinstance(frame, torch.Tensor) else ops.to_tensor(frame, device)
        lo_src = t if valid is None else torch.where(valid, t, torch.full_like(t, float("inf")))
        hi_src = t if valid is None else torch.where(valid, t, torch.full_like(t, float("-inf")))
        if self.lo is None or self.hi is None:
            self.lo, self.hi = lo_src.clone(), hi_src.clone()
        else:
            self.lo = torch.minimum(self.lo, lo_src)
            self.hi = torch.maximum(self.hi, hi_src)

    def apply(self, result: torch.Tensor) -> torch.Tensor:
        if self.lo is None or self.hi is None:
            return result
        covered = torch.isfinite(self.lo) & torch.isfinite(self.hi)
        clamped = torch.minimum(torch.maximum(result, self.lo), self.hi)
        return torch.where(covered, clamped, result)


def _read_mask(masks: FrameSource, idx: int) -> np.ndarray | torch.Tensor:
    held = read_tensor(masks, idx)
    return held if held is not None else masks.read(idx)


def _mask_levels(mask_np: np.ndarray | torch.Tensor, device: Device, depth: int) -> list[torch.Tensor]:
    """Downsample a boolean validity mask through the pyramid levels.

    Returns one boolean tensor per Laplacian level (depth tensors) then one for
    the residual — total depth + 1 entries, indexed [0 .. depth].
    A pixel is considered valid when the averaged mask value exceeds 0.5.
    """
    # Work with (1, H, W) float so we can reuse downsample2 (which expects C,H,W)
    if isinstance(mask_np, torch.Tensor):
        m = mask_np.to(device=device.torch_device, dtype=torch.float32).unsqueeze(0)
    else:
        m = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).to(device.torch_device)
    levels: list[torch.Tensor] = []
    for _ in range(depth + 1):  # depth Laplacian levels + 1 residual level
        levels.append(m > 0.5)
        m = ops.downsample2(m)
    return levels


@register("pmax")
class PMax:
    """Pyramid-max focus stacking with streaming fold and halo suppression."""

    @staticmethod
    def params() -> list[ParamSpec]:
        return [
            ParamSpec(
                "selection_smoothing",
                "Selection smoothing",
                "int",
                default=1,
                min=0,
                max=3,
                tooltip=(
                    "Median-filters the per-level winner maps to suppress halos. "
                    "0 = off (fastest). PMax can amplify noise and contrast; that is "
                    "expected and matches Zerene."
                ),
            ),
            ParamSpec(
                "range_clamp",
                "Range clamp",
                "bool",
                default=True,
                tooltip=(
                    "Limits every output pixel to the lowest and highest value that pixel has "
                    "in any aligned frame. Removes the dark rims PMax draws along high-contrast "
                    "edges; leaves fine detail alone."
                ),
            ),
        ]

    def run(
        self,
        source: FrameSource,
        device: Device,
        params: dict[str, Any],
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
        masks: FrameSource | None = None,
    ) -> StackResult:
        smoothing = int(params.get("selection_smoothing", 1))
        bounds = _RangeBounds() if bool(params.get("range_clamp", True)) else None
        n = len(source)
        first = _read_frame(source, 0)
        h, w = first.shape[-2:] if isinstance(first, torch.Tensor) else first.shape[:2]
        depth = int(params.get("_pyramid_depth") or pyramid_depth(h, w))

        # total_steps: n (fold pass) + n (halo pass, if needed) + 1 (collapse)
        # We use a conservative n*2+1 so fractions are consistent regardless of
        # whether the halo pass is actually triggered.
        total_steps = n * (2 if smoothing > 0 else 1) + 1

        state: _FoldState | None = None

        def _tick(i: int, msg: str) -> None:
            if cancel is not None and cancel():
                raise InterruptedError("stack job cancelled")
            if progress is not None:
                progress(msg, i / total_steps)

        # ------------------------------------------------------------------ #
        # Pass 1: streaming max-energy fold                                   #
        # ------------------------------------------------------------------ #
        for idx in range(n):
            _tick(idx, f"PMax fold {idx + 1}/{n}")
            frame = first if idx == 0 else _read_frame(source, idx)
            lap, energies, residual, res_w = _frame_pyramid(frame, device, depth)

            # Apply validity masks by zeroing energy in invalid regions so those
            # pixels can never win, and by zeroing the residual weight contribution.
            mlv: list[torch.Tensor] | None = None
            if masks is not None:
                mlv = _mask_levels(_read_mask(masks, idx), device, depth)
                energies = [e.masked_fill(~m, _NEG_INF) for e, m in zip(energies, mlv[:depth])]
                res_w = res_w * mlv[depth].to(res_w.dtype)
            if bounds is not None:
                bounds.update(frame, device, mlv[0] if mlv is not None else None)

            if state is None:
                state = _FoldState(
                    lap=lap,
                    energy=energies,
                    winner=[torch.zeros(e.shape[-2:], dtype=torch.int32, device=e.device) for e in energies],
                    res_num=residual * res_w,
                    res_den=res_w.clone(),
                )
                continue

            # Update best coefficients where this frame has higher energy.
            for lvl in range(depth):
                better = energies[lvl] > state.energy[lvl]  # (1, h_l, w_l)
                state.lap[lvl] = torch.where(better, lap[lvl], state.lap[lvl])
                state.energy[lvl] = torch.where(better, energies[lvl], state.energy[lvl])
                state.winner[lvl] = torch.where(
                    better.squeeze(0),
                    torch.full_like(state.winner[lvl], idx),
                    state.winner[lvl],
                )

            # Accumulate residual via weighted average (energy-weighted blending).
            state.res_num = state.res_num + residual * res_w
            state.res_den = state.res_den + res_w

        assert state is not None, "source must have at least one frame"

        # ------------------------------------------------------------------ #
        # Halo control: smooth winner maps, then re-read frames to paste      #
        # coefficients for pixels where the winner changed.                   #
        # ------------------------------------------------------------------ #
        if smoothing > 0:
            filtered = [ops.median_filter2d(wm, smoothing) for wm in state.winner]
            changed = [f != wm for f, wm in zip(filtered, state.winner)]

            if any(c.any() for c in changed):
                # Update state.winner to the smoothed version for aux output.
                state.winner = filtered

                for idx in range(n):
                    _tick(n + idx, f"PMax halo pass {idx + 1}/{n}")
                    lap, _, _, _ = _frame_pyramid(_read_frame(source, idx), device, depth)
                    mlv = _mask_levels(_read_mask(masks, idx), device, depth) if masks is not None else None
                    for lvl in range(depth):
                        # Pixels that changed winner to this frame index.
                        sel = changed[lvl] & (filtered[lvl] == idx)
                        if mlv is not None:
                            sel = sel & mlv[lvl].squeeze(0)
                        if sel.any():
                            state.lap[lvl] = torch.where(sel.unsqueeze(0), lap[lvl], state.lap[lvl])
            else:
                # No pixels actually changed; still update winner for consistency.
                state.winner = filtered

        # ------------------------------------------------------------------ #
        # Collapse                                                             #
        # ------------------------------------------------------------------ #
        residual = state.res_num / state.res_den.clamp_min(1e-8)
        result = collapse_laplacian(state.lap, residual)
        if bounds is not None:
            result = bounds.apply(result)

        _tick(total_steps - 1, "collapse")
        if progress is not None:
            progress("done", 1.0)

        return StackResult(
            image=ops.to_numpy(result),
            aux={f"winner_l{i}": wm.cpu().numpy() for i, wm in enumerate(state.winner)},
        )
