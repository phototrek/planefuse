"""Deep-zoom viewer tile pyramid (SPEC §9). 256px JPEG tiles per zoom level.
Stored as cache/tiles/<image_id>/<z>/<x>_<y>.jpg; z=0 coarsest."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 256


def build_pyramid(image: np.ndarray, out_dir: Path) -> int:
    """Build tiles for a float32 (H,W,3) image (clamped to [0,1] for display).
    Returns the number of zoom levels (z in [0, levels-1], levels-1 = full res)."""
    out_dir = Path(out_dir)
    h, w = image.shape[:2]
    clamped = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    base = Image.fromarray(clamped)
    levels = max(1, math.ceil(math.log2(max(1, max(h, w) / TILE))) + 1)
    for z in range(levels):
        scale = 2 ** (levels - 1 - z)
        lw = max(1, w // scale)
        lh = max(1, h // scale)
        lvl_img = base.resize((lw, lh), Image.Resampling.BILINEAR)
        zdir = out_dir / str(z)
        zdir.mkdir(parents=True, exist_ok=True)
        for ty in range(math.ceil(lh / TILE)):
            for tx in range(math.ceil(lw / TILE)):
                box = (tx * TILE, ty * TILE, min((tx + 1) * TILE, lw), min((ty + 1) * TILE, lh))
                lvl_img.crop(box).save(zdir / f"{tx}_{ty}.jpg", quality=85)
    return levels


def tile_path(tiles_root: Path, image_id: str, z: int, x: int, y: int) -> Path:
    return tiles_root / image_id / str(z) / f"{x}_{y}.jpg"
