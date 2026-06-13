"""Per-cell depth, in-focus intervals, and peak-set augmentation (SPEC §7.0 steps 4-5)."""

from __future__ import annotations

import numpy as np


def peak_index(curve: np.ndarray) -> int:
    return int(np.argmax(curve))


def in_focus_interval(curve: np.ndarray, focus_tolerance: float) -> tuple[int, int]:
    """Maximal contiguous run of frames CONTAINING the peak whose value is
    >= focus_tolerance * peak. Returns inclusive (lo, hi)."""
    p = peak_index(curve)
    thr = focus_tolerance * float(curve[p])
    lo = p
    while lo - 1 >= 0 and curve[lo - 1] >= thr:
        lo -= 1
    hi = p
    while hi + 1 < len(curve) and curve[hi + 1] >= thr:
        hi += 1
    return lo, hi
