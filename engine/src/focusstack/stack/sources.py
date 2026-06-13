"""Frame sources: stream frames from memory or disk (SPEC §7 streaming)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from focusstack.io import load_image

Region = tuple[int, int, int, int]  # (y0, x0, y1, x1)


def _crop(arr: np.ndarray, region: Region | None) -> np.ndarray:
    if region is None:
        return arr
    y0, x0, y1, x1 = region
    return arr[y0:y1, x0:x1]


class ArrayFrameSource:
    """In-memory frames (tests, small stacks)."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return _crop(self._frames[idx], region)


class DirFrameSource:
    """Frames loaded lazily from disk paths; re-reads on every call (streaming)."""

    def __init__(self, paths: list[Path]):
        self.paths = [Path(p) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return _crop(load_image(self.paths[idx]).pixels, region)


class _SubsetSource:
    """Exposes only a subset of another source's frames, re-indexed to 0..k-1."""

    def __init__(self, inner, indices: list[int]):
        self._inner = inner
        self._idx = list(indices)

    def __len__(self) -> int:
        return len(self._idx)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        return self._inner.read(self._idx[idx], region=region)
