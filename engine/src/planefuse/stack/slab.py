"""Slabbing: hierarchical stacking for deep stacks (SPEC §7.4).

Note: validity masks are not propagated into slab sub-stacks in this milestone
(slabbing is used on already-aligned or pre-aligned stacks); masks= is accepted
for interface compatibility but ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from planefuse.backend import Device
from planefuse.stack.base import (
    CancelFn,
    FrameSource,
    ParamSpec,
    ProgressFn,
    StackResult,
    get_algorithm,
    register,
)
from planefuse.stack.sources import ArrayFrameSource


def plan_slabs(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping [start, end) windows covering range(n). step = size - overlap."""
    if size <= overlap:
        raise ValueError(f"slab size {size} must exceed overlap {overlap}")
    if n <= size:
        return [(0, n)]
    step = size - overlap
    slabs: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        slabs.append((start, end))
        if end == n:
            break
        start += step
    return slabs


@register("slab")
class Slab:
    @staticmethod
    def params() -> list[ParamSpec]:
        return [
            ParamSpec("slab_size", "Slab size", "int", default=10, min=2, max=100,
                      tooltip="Frames per sub-stack."),
            ParamSpec("slab_overlap", "Slab overlap", "int", default=2, min=0, max=50,
                      tooltip="Overlapping frames between consecutive slabs."),
            ParamSpec("inner_method", "Inner method", "choice", default="pmax",
                      choices=("pmax", "dmap", "weighted"), tooltip="Method for each slab."),
            ParamSpec("outer_method", "Outer method", "choice", default="dmap",
                      choices=("pmax", "dmap", "weighted"), tooltip="Method combining slabs."),
        ]

    def run(self, source: FrameSource, device: Device, params: dict[str, Any],
            progress: ProgressFn | None = None, cancel: CancelFn | None = None,
            masks: FrameSource | None = None) -> StackResult:
        n = len(source)
        size = int(params.get("slab_size", 10))
        overlap = int(params.get("slab_overlap", 2))
        inner = get_algorithm(str(params.get("inner_method", "pmax")))
        outer = get_algorithm(str(params.get("outer_method", "dmap")))
        slabs = plan_slabs(n, size, overlap)

        slab_results: list[np.ndarray] = []
        for si, (a, b) in enumerate(slabs):
            if cancel is not None and cancel():
                raise InterruptedError("slab stack cancelled")
            if progress is not None:
                progress(f"slab {si + 1}/{len(slabs)}", si / (len(slabs) + 1))
            sub = ArrayFrameSource([source.read(i) for i in range(a, b)])
            res = inner.run(sub, device, params)
            slab_results.append(res.image)

        if progress is not None:
            progress("combining slabs", len(slabs) / (len(slabs) + 1))
        combined = outer.run(ArrayFrameSource(slab_results), device, params)
        if progress is not None:
            progress("done", 1.0)
        return StackResult(image=combined.image, aux=combined.aux)
