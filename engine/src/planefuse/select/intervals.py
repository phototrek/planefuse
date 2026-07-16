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


def build_rows(smoothed: np.ndarray, reliable: np.ndarray, focus_tolerance: float,
               n_frames: int) -> list[tuple[tuple[int, int], str]]:
    """Coverage rows for the set cover (SPEC §7.0 step 5). Returns a list of
    ((lo, hi) inclusive interval, kind) where kind is 'cell' or 'synthetic'.

    - one 'cell' row per reliable cell, with its in-focus interval;
    - peak-set augmentation: L = sorted distinct reliable-peak indices; for each
      maximal run of consecutive indices in L, add a synthetic row for the index
      immediately before and after the run, whose interval is the union of the
      in-focus intervals of all reliable cells peaking at the adjacent run
      endpoint, shifted by ∓1 and clipped; empty rows dropped.
    """
    r, c = reliable.shape
    rows: list[tuple[tuple[int, int], str]] = []
    peak_to_intervals: dict[int, list[tuple[int, int]]] = {}
    peaks: set[int] = set()
    for i in range(r):
        for j in range(c):
            if not reliable[i, j]:
                continue
            curve = smoothed[:, i, j]
            iv = in_focus_interval(curve, focus_tolerance)
            rows.append((iv, "cell"))
            p = peak_index(curve)
            peaks.add(p)
            peak_to_intervals.setdefault(p, []).append(iv)

    if not peaks:
        return rows

    sorted_peaks = sorted(peaks)
    runs: list[list[int]] = [[sorted_peaks[0]]]
    for idx in sorted_peaks[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])

    def _clip(v: int) -> int:
        return max(0, min(n_frames - 1, v))

    def _union(endpoint: int, shift: int) -> tuple[int, int] | None:
        ivs = peak_to_intervals.get(endpoint, [])
        if not ivs:
            return None
        lo = _clip(min(a for a, _ in ivs) + shift)
        hi = _clip(max(b for _, b in ivs) + shift)
        if lo > hi:
            return None
        return (lo, hi)

    for run in runs:
        before = _union(run[0], -1)
        after = _union(run[-1], +1)
        if before is not None:
            rows.append((before, "synthetic"))
        if after is not None:
            rows.append((after, "synthetic"))
    return rows
