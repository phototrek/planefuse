"""Interval-stabbing set cover + degenerate floor (SPEC §7.0 steps 6-7)."""

from __future__ import annotations

_MIN_KEEP = 3


def min_stab_cover(intervals: list[tuple[int, int]]) -> list[int]:
    """Minimum set of integer points stabbing every inclusive interval.
    Exact (consecutive-ones / interval point cover): sort by right endpoint,
    repeatedly take the right endpoint of the first not-yet-stabbed interval."""
    if not intervals:
        return []
    pts: list[int] = []
    last: int | None = None
    for lo, hi in sorted(intervals, key=lambda iv: iv[1]):
        if last is None or lo > last:
            last = hi
            pts.append(hi)
    return pts


def select_indices(intervals: list[tuple[int, int]], n_frames: int
                   ) -> tuple[list[int], list[int], str | None]:
    """Returns (kept sorted, redundant sorted, warning|None). Degenerate floor:
    if no rows or fewer than 3 kept, keep all frames with a warning (SPEC §7.0 step 7)."""
    if not intervals:
        return list(range(n_frames)), [], "scene too low-contrast for frame selection"
    kept = sorted(set(min_stab_cover(intervals)))
    if len(kept) < _MIN_KEEP:
        return list(range(n_frames)), [], "scene too low-contrast for frame selection"
    redundant = [i for i in range(n_frames) if i not in set(kept)]
    return kept, redundant, None
