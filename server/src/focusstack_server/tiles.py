"""Deep-zoom viewer tile pyramid (SPEC §9). 256px JPEG tiles per zoom level.
Stored as cache/tiles/<image_id>/<z>/<x>_<y>.jpg; z=0 coarsest."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 256


def _to_pil(image: np.ndarray) -> Image.Image:
    clamped = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(clamped)


def _level_dims(w: int, h: int, levels: int, z: int) -> tuple[int, int, int]:
    scale = 2 ** (levels - 1 - z)
    return max(1, w // scale), max(1, h // scale), scale


def _save_tile(image: Image.Image, out_dir: Path, tx: int, ty: int, w: int, h: int) -> None:
    box = (tx * TILE, ty * TILE, min((tx + 1) * TILE, w), min((ty + 1) * TILE, h))
    image.crop(box).save(out_dir / f"{tx}_{ty}.jpg", quality=85)


def _levels_for(w: int, h: int) -> int:
    return max(1, math.ceil(math.log2(max(1, max(h, w) / TILE))) + 1)


def build_pyramid(image: np.ndarray, out_dir: Path) -> int:
    """Build tiles for a float32 image and return its zoom-level count."""
    out_dir = Path(out_dir)
    h, w = image.shape[:2]
    base = _to_pil(image)
    levels = _levels_for(w, h)
    for z in range(levels):
        lw, lh, _ = _level_dims(w, h, levels, z)
        lvl_img = base.resize((lw, lh), Image.Resampling.BILINEAR)
        zdir = out_dir / str(z)
        zdir.mkdir(parents=True, exist_ok=True)
        for ty in range(math.ceil(lh / TILE)):
            for tx in range(math.ceil(lw / TILE)):
                _save_tile(lvl_img, zdir, tx, ty, lw, lh)
    return levels


def rebuild_region(
    image: np.ndarray,
    out_dir: Path,
    levels: int,
    bbox: tuple[int, int, int, int],
) -> list[dict[str, int]]:
    """Rewrite tiles overlapping the base-resolution bbox."""
    y0, x0, y1, x1 = bbox
    if y1 <= y0 or x1 <= x0:
        return []

    out_dir = Path(out_dir)
    h, w = image.shape[:2]
    base = _to_pil(image)
    dirty: list[dict[str, int]] = []
    for z in range(levels):
        lw, lh, scale = _level_dims(w, h, levels, z)
        level = base.resize((lw, lh), Image.Resampling.BILINEAR)
        zdir = out_dir / str(z)
        zdir.mkdir(parents=True, exist_ok=True)
        tx0 = max(0, x0 // scale // TILE)
        ty0 = max(0, y0 // scale // TILE)
        tx1 = min(math.ceil(lw / TILE) - 1, (math.ceil(x1 / scale) - 1) // TILE)
        ty1 = min(math.ceil(lh / TILE) - 1, (math.ceil(y1 / scale) - 1) // TILE)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                _save_tile(level, zdir, tx, ty, lw, lh)
                dirty.append({"z": z, "x": tx, "y": ty})
    return dirty


def tile_path(tiles_root: Path, image_id: str, z: int, x: int, y: int) -> Path:
    return tiles_root / image_id / str(z) / f"{x}_{y}.jpg"
