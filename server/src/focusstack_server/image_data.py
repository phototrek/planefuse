"""Load persisted project images, including unclamped float working results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from focusstack.io import load_image
from focusstack_server.projects import Project


def load_project_pixels(project: Project, path: Path) -> np.ndarray:
    path = Path(path)
    matching = [info for info in project.images.values() if info.get("path") == str(path)]
    if any(info.get("storage") == "float32_tiff" for info in matching):
        pixels = np.asarray(tifffile.imread(path), dtype=np.float32)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError(f"working image must be RGB, got shape {pixels.shape}")
        return pixels
    return load_image(path).pixels
