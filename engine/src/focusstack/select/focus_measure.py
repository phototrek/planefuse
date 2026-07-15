"""Grid focus measures for frame selection (SPEC §7.0 step 1).

Per cell (i,j) and frame p: phi = mean over the cell's VALID pixels of the
absolute vertical second difference of luminance |−f(x,y−1)+2 f(x,y)−f(x,y+1)|.
A cell with >25% invalid pixels in ANY frame is marked globally unreliable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from focusstack.backend import Device, ops
from focusstack.align.proxy import scene_linear_proxy_luminance
from focusstack.io.metadata import ProcessingDomain
from focusstack.stack.base import FrameSource

_MAX_INVALID_FRACTION = 0.25


def _cell_ids(h: int, w: int, grid_rows: int, grid_cols: int) -> np.ndarray:
    """(H, W) int array mapping each pixel to a flat cell id in [0, rows*cols)."""
    row_edges = np.linspace(0, h, grid_rows + 1).astype(np.int64)
    col_edges = np.linspace(0, w, grid_cols + 1).astype(np.int64)
    row_id = np.zeros(h, dtype=np.int64)
    for r in range(grid_rows):
        row_id[row_edges[r]:row_edges[r + 1]] = r
    col_id = np.zeros(w, dtype=np.int64)
    for c in range(grid_cols):
        col_id[col_edges[c]:col_edges[c + 1]] = c
    return row_id[:, None] * grid_cols + col_id[None, :]


def _vertical_second_diff(gray: np.ndarray) -> np.ndarray:
    """|−f(x,y−1)+2 f(x,y)−f(x,y+1)| with edge replication. gray (H, W)."""
    up = np.empty_like(gray)
    dn = np.empty_like(gray)
    up[1:, :] = gray[:-1, :]
    up[0, :] = gray[0, :]
    dn[:-1, :] = gray[1:, :]
    dn[-1, :] = gray[-1, :]
    return np.abs(-up + 2.0 * gray - dn)


def compute_focus_measures(source: FrameSource, device: Device,
                           grid_rows: int = 32, grid_cols: int = 48,
                           masks: FrameSource | None = None,
                           progress: Callable[[str, float], None] | None = None,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (phi (n, grid_rows, grid_cols) float32,
    cell_reliable (grid_rows, grid_cols) bool)."""
    n = len(source)
    probe = source.read(0)
    h, w = probe.shape[:2]
    cell_id = _cell_ids(h, w, grid_rows, grid_cols).ravel()
    ncells = grid_rows * grid_cols
    phi = np.zeros((n, grid_rows, grid_cols), dtype=np.float32)
    cell_unreliable = np.zeros(ncells, dtype=bool)

    for p in range(n):
        if progress is not None:
            progress(f"focus measure {p + 1}/{n}", p / n)
        frame = probe if p == 0 else source.read(p)
        if source.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
            gray = scene_linear_proxy_luminance(frame)
        else:
            gray = ops.to_numpy(ops.rgb_to_luminance(ops.to_tensor(frame, device)))[..., 0]
        d = _vertical_second_diff(gray).ravel()
        if masks is not None:
            valid = np.ascontiguousarray(masks.read(p)).astype(bool).ravel()
        else:
            valid = np.ones(h * w, dtype=bool)
        valid_f = valid.astype(np.float64)
        vcount = np.bincount(cell_id, weights=valid_f, minlength=ncells)
        total = np.bincount(cell_id, minlength=ncells).astype(np.float64)
        dsum = np.bincount(cell_id, weights=(d * valid_f), minlength=ncells)
        phi[p] = (dsum / np.maximum(vcount, 1.0)).reshape(grid_rows, grid_cols)
        invalid_frac = 1.0 - (vcount / np.maximum(total, 1.0))
        cell_unreliable |= invalid_frac > _MAX_INVALID_FRACTION

    return phi, (~cell_unreliable).reshape(grid_rows, grid_cols)
