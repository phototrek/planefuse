"""Aligned-frame + validity-mask cache (SPEC §9 cache/, §12 atomic writes).

Aligned frames are stored as 16-bit TIFF (lossless, matches the working pipeline);
masks as 8-bit single-channel PNG (0/255). Readers are FrameSource-compatible so
PMax and tiled stacking consume them with no changes (their masks= parameter).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from focusstack.io import load_image, save_image
from focusstack.stack.sources import Region, _crop  # noqa: SLF001


class _CachedSource:
    def __init__(self, paths: list[Path], as_mask: bool = False):
        self.paths = paths
        self._as_mask = as_mask

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        arr = load_image(self.paths[idx]).pixels  # (H, W, 3) float32
        if self._as_mask:
            arr = arr[..., 0] > 0.5  # (H, W) bool
        return _crop(arr, region)


class AlignedCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.frames_dir = self.root / "aligned"
        self.masks_dir = self.root / "masks"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_save(self, arr: np.ndarray, path: Path, **kw) -> None:
        tmp = path.with_name(path.stem + ".__tmp__" + path.suffix)
        save_image(arr, tmp, **kw)
        os.replace(tmp, path)

    def write(self, idx: int, aligned: np.ndarray, mask: np.ndarray) -> None:
        fpath = self.frames_dir / f"{idx:04d}.tif"
        mpath = self.masks_dir / f"{idx:04d}.png"
        self._atomic_save(aligned, fpath, bit_depth=16, compression="zlib")
        mask3 = np.repeat(mask.astype(np.float32)[..., None], 3, axis=2)
        self._atomic_save(mask3, mpath, bit_depth=8)

    def frame_source(self) -> _CachedSource:
        return _CachedSource(sorted(self.frames_dir.glob("*.tif")))

    def mask_source(self) -> _CachedSource:
        return _CachedSource(sorted(self.masks_dir.glob("*.png")), as_mask=True)
