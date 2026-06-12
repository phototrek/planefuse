"""Tiled processing with feathered overlap blending (SPEC §8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from focusstack.backend import Device
from focusstack.stack.base import FrameSource
from focusstack.stack.pyramid import pyramid_depth


@dataclass(frozen=True)
class Tile:
    y0: int
    x0: int
    y1: int
    x1: int


def plan_tiles(h: int, w: int, tile: int, overlap: int) -> list[Tile]:
    if overlap >= tile:
        raise ValueError(f"overlap {overlap} must be smaller than tile {tile}")
    if h <= tile and w <= tile:
        return [Tile(0, 0, h, w)]
    step = tile - overlap
    ys = list(range(0, max(h - overlap, 1), step))
    xs = list(range(0, max(w - overlap, 1), step))
    out = []
    for y in ys:
        for x in xs:
            out.append(Tile(y, x, min(y + tile, h), min(x + tile, w)))
    return out


def estimate_stack_bytes(h: int, w: int, c: int = 3) -> int:
    """Rough upper bound of device memory for an untiled PMax fold (SPEC §4).

    State: pyramid coefficients (4/3 * C floats) + energy + winner per pixel,
    plus ~3 transient frame-sized buffers during the per-frame pyramid build.
    Frame count does not appear: the fold is streaming.
    """
    px = h * w
    pyr = px * 4 / 3
    state = pyr * (c * 4 + 4 + 4)
    transient = 3 * px * c * 4 + pyr * (c * 4 + 4)
    return int(state + transient)


def _feather(t: Tile, h: int, w: int, overlap: int) -> np.ndarray:
    """Per-tile weights: 1 in the interior, linear ramp across overlap,
    full weight at image borders (no neighbor there)."""
    th, tw = t.y1 - t.y0, t.x1 - t.x0
    ramp = max(overlap, 1)

    def axis_w(n: int, lo_edge: bool, hi_edge: bool) -> np.ndarray:
        a = np.ones(n, dtype=np.float32)
        r = min(ramp, n)
        if not lo_edge:
            a[:r] = np.linspace(1.0 / (r + 1), 1.0, r, dtype=np.float32)
        if not hi_edge:
            a[n - r :] = np.linspace(1.0, 1.0 / (r + 1), r, dtype=np.float32)
        return a

    wy = axis_w(th, t.y0 == 0, t.y1 == h)
    wx = axis_w(tw, t.x0 == 0, t.x1 == w)
    return np.outer(wy, wx)[..., None]  # (th, tw, 1)


class _RegionSource:
    """Restricts a FrameSource to one tile rectangle."""

    def __init__(self, inner: FrameSource, t: Tile):
        self._inner = inner
        self._t = t

    def __len__(self) -> int:
        return len(self._inner)

    def read(self, idx: int, region=None) -> np.ndarray:
        t = self._t
        if region is None:
            return self._inner.read(idx, region=(t.y0, t.x0, t.y1, t.x1))
        y0, x0, y1, x1 = region
        return self._inner.read(idx, region=(t.y0 + y0, t.x0 + x0, t.y0 + y1, t.x0 + x1))


def stack_tiled(
    method: str,
    source: FrameSource,
    device: Device,
    params: dict[str, Any],
    tile: int = 2048,
    overlap: int = 128,
    progress=None,
    cancel=None,
    masks: FrameSource | None = None,
) -> np.ndarray:
    from focusstack.stack.base import get_algorithm  # local import avoids cycle

    probe = source.read(0)
    h, w, c = probe.shape
    del probe
    full_depth = pyramid_depth(h, w)
    # SPEC §8: pyramid algorithms need overlap >= 2^depth to avoid seams
    overlap = max(overlap, 2**full_depth)
    # plan_tiles needs tile > overlap or the tile grid degenerates
    tile = max(tile, 4 * overlap)
    params = {**params, "_pyramid_depth": full_depth}
    tiles = plan_tiles(h, w, tile, overlap)
    num = np.zeros((h, w, c), dtype=np.float32)
    den = np.zeros((h, w, 1), dtype=np.float32)
    algo = get_algorithm(method)
    for i, t in enumerate(tiles):
        if progress is not None:
            progress(f"tile {i + 1}/{len(tiles)}", i / len(tiles))
        sub = _RegionSource(source, t)
        sub_masks = _RegionSource(masks, t) if masks is not None else None
        kwargs = {"masks": sub_masks} if sub_masks is not None else {}
        res = algo.run(sub, device, params, cancel=cancel, **kwargs)
        fw = _feather(t, h, w, overlap)
        num[t.y0 : t.y1, t.x0 : t.x1] += res.image * fw
        den[t.y0 : t.y1, t.x0 : t.x1] += fw
    if progress is not None:
        progress("done", 1.0)
    return num / np.maximum(den, 1e-8)
