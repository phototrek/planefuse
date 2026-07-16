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

_NEG_INF = float("-inf")


@dataclass
class _FoldState:
    lap: list[torch.Tensor]  # best coefficients per level (C, h_l, w_l)
    energy: list[torch.Tensor]  # best energy per level (1, h_l, w_l)
    winner: list[torch.Tensor]  # winning frame index per level (h_l, w_l) int32
    res_num: torch.Tensor  # residual weighted sum (C, h_r, w_r)
    res_den: torch.Tensor  # residual weight sum   (1, h_r, w_r)


def _frame_pyramid(
    frame_np: np.ndarray, device: Device, depth: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Convert a numpy frame to a Laplacian pyramid with per-level energy maps.

    Returns:
        lap       - Laplacian coefficient tensors (C, h_l, w_l)
        energies  - local energy tensors (1, h_l, w_l)
        residual  - coarsest Gaussian level (C, h_r, w_r)
        res_weight - soft weight for residual contribution (1, h_r, w_r)
    """
    t = ops.to_tensor(frame_np, device)
    lap, residual = build_laplacian(t, depth)
    energies = [ops.local_energy(lvl) for lvl in lap]
    # Residual weight: smoothed magnitude of high-frequency content at coarsest
    # level — frames with more detail there contribute more to the weighted avg.
    res_weight = ops.box_filter3((residual - ops.box_filter3(residual)).abs().mean(0, keepdim=True)) + 1e-6
    return lap, energies, residual, res_weight


def _mask_levels(mask_np: np.ndarray, device: Device, depth: int) -> list[torch.Tensor]:
    """Downsample a boolean validity mask through the pyramid levels.

    Returns one boolean tensor per Laplacian level (depth tensors) then one for
    the residual — total depth + 1 entries, indexed [0 .. depth].
    A pixel is considered valid when the averaged mask value exceeds 0.5.
    """
    # Work with (1, H, W) float so we can reuse downsample2 (which expects C,H,W)
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
        n = len(source)
        first = source.read(0)
        h, w = first.shape[:2]
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
            frame = first if idx == 0 else source.read(idx)
            lap, energies, residual, res_w = _frame_pyramid(frame, device, depth)

            # Apply validity masks by zeroing energy in invalid regions so those
            # pixels can never win, and by zeroing the residual weight contribution.
            mlv: list[torch.Tensor] | None = None
            if masks is not None:
                mlv = _mask_levels(masks.read(idx), device, depth)
                energies = [e.masked_fill(~m, _NEG_INF) for e, m in zip(energies, mlv[:depth])]
                res_w = res_w * mlv[depth].to(res_w.dtype)

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
                    lap, _, _, _ = _frame_pyramid(source.read(idx), device, depth)
                    mlv = _mask_levels(masks.read(idx), device, depth) if masks is not None else None
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

        _tick(total_steps - 1, "collapse")
        if progress is not None:
            progress("done", 1.0)

        return StackResult(
            image=ops.to_numpy(result),
            aux={f"winner_l{i}": wm.cpu().numpy() for i, wm in enumerate(state.winner)},
        )
