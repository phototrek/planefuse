"""Image export (SPEC §5). Clamping to [0, 1] happens HERE and only here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

_ICC_TAG = 34675
_TIFF_COMPRESSION = {"none": None, "lzw": "lzw", "zlib": "zlib"}


def save_image(
    arr: np.ndarray,
    path: Path,
    *,
    bit_depth: int = 16,
    icc: bytes | None = None,
    compression: str = "zlib",
    jpeg_quality: int = 95,
) -> None:
    """arr: float32 (H, W, 3) RGB or (H, W) grayscale, possibly outside [0, 1] (clamped here, SPEC §7.1)."""
    path = Path(path)
    clipped = np.clip(arr, 0.0, 1.0)
    suffix = path.suffix.lower()

    # Depth-map / grayscale export: 2-D (H, W) arrays (SPEC §5 DMap depth map).
    if clipped.ndim == 2:
        if suffix in (".tif", ".tiff"):
            if bit_depth == 16:
                data = (clipped * 65535.0 + 0.5).astype(np.uint16)
            elif bit_depth == 8:
                data = (clipped * 255.0 + 0.5).astype(np.uint8)
            else:
                raise ValueError(f"unsupported TIFF bit depth {bit_depth}")
            if compression not in _TIFF_COMPRESSION:
                raise ValueError(f"unknown compression {compression!r}")
            tifffile.imwrite(path, data, compression=_TIFF_COMPRESSION[compression])
            return
        if suffix == ".png":
            if bit_depth == 16:
                data = (clipped * 65535.0 + 0.5).astype(np.uint16)
                Image.fromarray(data, mode="I;16").save(path)
            else:
                Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
            return
        raise ValueError(f"grayscale export supports .tif/.png, not {suffix!r}")

    # RGB export: 3-D (H, W, 3) arrays.
    if suffix in (".tif", ".tiff"):
        if bit_depth == 16:
            data = (clipped * 65535.0 + 0.5).astype(np.uint16)
        elif bit_depth == 8:
            data = (clipped * 255.0 + 0.5).astype(np.uint8)
        else:
            raise ValueError(f"unsupported TIFF bit depth {bit_depth}")
        if compression not in _TIFF_COMPRESSION:
            raise ValueError(f"unknown compression {compression!r}")
        extratags = [(_ICC_TAG, 7, len(icc), icc, False)] if icc else []  # type 7 = UNDEFINED
        tifffile.imwrite(path, data, compression=_TIFF_COMPRESSION[compression], extratags=extratags)
        return
    if suffix in (".jpg", ".jpeg"):
        im = Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8))
        im.save(path, quality=jpeg_quality, icc_profile=icc)
        return
    if suffix == ".png":
        if bit_depth == 16:
            raise ValueError("16-bit PNG output lands in M6 (SPEC §5); use TIFF for 16-bit now")
        im = Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8))
        im.save(path, icc_profile=icc)
        return
    raise ValueError(f"unsupported output format {suffix!r} (use .tif/.jpg/.png)")
