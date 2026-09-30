"""Aligned-frame + validity-mask cache (SPEC §9 cache/, §12 atomic writes).

Aligned frames are stored as 16-bit TIFF (lossless, matches the working pipeline);
masks as 8-bit single-channel PNG (0/255). Readers are FrameSource-compatible so
PMax and tiled stacking consume them with no changes (their masks= parameter).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile
import torch

from planefuse.errors import ValidationError
from planefuse.io import load_image, save_image
from planefuse.io.atomic import atomic_output
from planefuse.io.metadata import ImageMetadata, ProcessingDomain
from planefuse.stack.sources import Region, _crop  # noqa: SLF001


def cache_equivalent(aligned: torch.Tensor, domain: ProcessingDomain) -> torch.Tensor:
    """The values AlignedCache.write then read would hand back for `aligned`
    ((3, H, W) float32), computed where the tensor lives.

    Scene-linear frames are stored as float32 (lossless), so they come back
    unchanged; rendered frames are clipped to [0, 1] and stored as 16-bit, so
    they come back quantized exactly as save_image + load_image do it.
    """
    if domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
        return aligned
    return torch.floor(aligned.clamp(0.0, 1.0) * 65535.0 + 0.5) / 65535.0


class _CachedSource:
    def __init__(
        self,
        paths: list[Path],
        *,
        as_mask: bool = False,
        domain: ProcessingDomain = ProcessingDomain.RENDERED_RGB,
        metadata: ImageMetadata | None = None,
        source_hash: str = "",
    ):
        self.paths = paths
        self._as_mask = as_mask
        self.domain = domain
        self.metadata = metadata
        self.source_hash = source_hash

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, idx: int, region: Region | None = None) -> np.ndarray:
        if self.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB and not self._as_mask:
            arr = np.asarray(tifffile.imread(self.paths[idx]), dtype=np.float32)
        else:
            arr = load_image(self.paths[idx]).pixels  # (H, W, 3) float32
        if self._as_mask:
            arr = arr[..., 0] > 0.5  # (H, W) bool
        return _crop(arr, region)


class AlignedCache:
    def __init__(
        self,
        root: Path,
        *,
        domain: ProcessingDomain = ProcessingDomain.RENDERED_RGB,
        metadata: ImageMetadata | None = None,
        source_hash: str = "",
    ):
        self.root = Path(root)
        self.domain = domain
        self.metadata = metadata
        self.source_hash = source_hash
        calibration = None
        if metadata is not None:
            calibration_payload = repr((
                metadata.black_level,
                metadata.white_level,
                metadata.color_matrix1,
                metadata.color_matrix2,
                metadata.calibration_illuminant1,
                metadata.calibration_illuminant2,
            )).encode("utf-8")
            calibration = hashlib.sha256(calibration_payload).hexdigest()
        expected = {
            "version": 2,
            "domain": domain.value,
            "source_hash": source_hash,
            "decoder": dict(metadata.decoder) if metadata is not None else {},
            "calibration_hash": calibration,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = ("version", "domain") + (("source_hash",) if source_hash else ())
            if any(existing.get(key) != expected.get(key) for key in keys):
                raise ValidationError(
                    f"cache manifest does not match requested domain/source: {manifest_path}"
                )
            self.manifest = existing
        else:
            with atomic_output(manifest_path) as temporary:
                temporary.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
            self.manifest = expected
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
        if self.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
            with atomic_output(fpath) as temporary:
                tifffile.imwrite(
                    temporary,
                    aligned.astype(np.float32),
                    compression="zlib",
                    photometric="rgb",
                )
        else:
            self._atomic_save(aligned, fpath, bit_depth=16, compression="zlib")
        mask3 = np.repeat(mask.astype(np.float32)[..., None], 3, axis=2)
        self._atomic_save(mask3, mpath, bit_depth=8)

    def frame_source(self) -> _CachedSource:
        return _CachedSource(
            sorted(self.frames_dir.glob("*.tif")),
            domain=self.domain,
            metadata=self.metadata,
            source_hash=self.source_hash,
        )

    def mask_source(self) -> _CachedSource:
        return _CachedSource(
            sorted(self.masks_dir.glob("*.png")),
            as_mask=True,
            domain=ProcessingDomain.RENDERED_RGB,
            source_hash=self.source_hash,
        )
