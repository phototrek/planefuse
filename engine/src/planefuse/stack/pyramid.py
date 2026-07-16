"""Laplacian pyramid build/collapse (SPEC §7.1).

Build subtracts the upsampled next-coarser Gaussian level, so
collapse(build(x)) == x exactly (up to float error) by construction.
"""

from __future__ import annotations

import math

import torch

from planefuse.backend import ops


def pyramid_depth(h: int, w: int) -> int:
    """SPEC §7.1: floor(log2(min(H, W))) - 5, clamped to >= 1 (coarsest >= 32 px)."""
    return max(1, int(math.floor(math.log2(min(h, w)))) - 5)


def build_laplacian(img: torch.Tensor, depth: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    lap: list[torch.Tensor] = []
    cur = img
    for _ in range(depth):
        down = ops.downsample2(cur)
        up = ops.upsample_to(down, (cur.shape[-2], cur.shape[-1]))
        lap.append(cur - up)
        cur = down
    return lap, cur


def collapse_laplacian(lap: list[torch.Tensor], residual: torch.Tensor) -> torch.Tensor:
    cur = residual
    for level in reversed(lap):
        cur = ops.upsample_to(cur, (level.shape[-2], level.shape[-1])) + level
    return cur
