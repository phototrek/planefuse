"""Load persisted project images, including unclamped float working results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from planefuse.io import ImageMetadata, ProcessingDomain, load_image, metadata_from_dict
from planefuse_server.projects import Project


def load_project_frame(
    project: Project, path: Path
) -> tuple[np.ndarray, ProcessingDomain, ImageMetadata | None]:
    path = Path(path)
    matching = [info for info in project.images.values() if info.get("path") == str(path)]
    working = next((info for info in matching if info.get("storage") == "float32_tiff"), None)
    if working is not None:
        pixels = np.asarray(tifffile.imread(path), dtype=np.float32)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError(f"working image must be RGB, got shape {pixels.shape}")
        domain = ProcessingDomain(working.get("domain", ProcessingDomain.RENDERED_RGB.value))
        metadata_values = working.get("metadata")
        metadata = metadata_from_dict(metadata_values) if metadata_values else None
        return pixels, domain, metadata
    frame = load_image(path)
    return frame.pixels, frame.domain, frame.metadata


def load_project_pixels(project: Project, path: Path) -> np.ndarray:
    return load_project_frame(project, path)[0]
