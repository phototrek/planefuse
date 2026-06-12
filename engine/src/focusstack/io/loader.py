"""Image loading and stack validation (SPEC §5, §12).

Working format: float32 (H, W, C) in [0, 1], source gamma kept, ICC bytes
carried through untouched. RGB only; grayscale and alpha are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from focusstack.errors import ValidationError

SUPPORTED = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
_ICC_TAG = 34675


@dataclass
class Frame:
    pixels: np.ndarray  # float32 (H, W, 3) in [0, 1]
    bit_depth: int  # 8 or 16
    icc: bytes | None
    path: Path


@dataclass
class FileStatus:
    path: Path
    status: str  # ok | wrong_size | wrong_bit_depth | unreadable | unsupported | not_rgb
    message: str = ""


@dataclass
class ValidationReport:
    files: list[FileStatus] = field(default_factory=list)
    width: int = 0
    height: int = 0
    bit_depth: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.files) and all(s.status == "ok" for s in self.files)


def _scale(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0, 8
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0, 16
    raise ValidationError(f"unsupported sample type {arr.dtype}; expected uint8 or uint16")


def load_image(path: Path) -> Frame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValidationError(f"{path.name}: unsupported format {suffix}")
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            arr = page.asarray()
            icc_tag = page.tags.get(_ICC_TAG) if isinstance(page, tifffile.TiffPage) else None
            icc = bytes(icc_tag.value) if icc_tag is not None else None
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValidationError(f"{path.name}: RGB input required (got shape {arr.shape})")
        pixels, depth = _scale(arr)
        return Frame(pixels, depth, icc, path)
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValidationError(f"{path.name}: RGB input required (got mode {im.mode})")
        icc = im.info.get("icc_profile")
        arr = np.asarray(im)
    pixels, depth = _scale(arr)
    return Frame(pixels, depth, bytes(icc) if icc else None, path)


def validate_stack(paths: list[Path]) -> ValidationReport:
    """Per-file validation report (SPEC §12). Never raises for bad files."""
    report = ValidationReport()
    for p in paths:
        p = Path(p)
        try:
            frame = load_image(p)
        except ValidationError as e:
            if p.suffix.lower() not in SUPPORTED:
                kind = "unsupported"
            elif "RGB" in str(e):
                kind = "not_rgb"
            else:
                kind = "unsupported"
            report.files.append(FileStatus(p, kind, str(e)))
            continue
        except Exception as e:  # noqa: BLE001 - corrupted files land here by design
            report.files.append(FileStatus(p, "unreadable", str(e)))
            continue
        h, w = frame.pixels.shape[:2]
        if report.width == 0:
            report.width, report.height, report.bit_depth = w, h, frame.bit_depth
        if (w, h) != (report.width, report.height):
            report.files.append(FileStatus(p, "wrong_size", f"{w}x{h} != {report.width}x{report.height}"))
        elif frame.bit_depth != report.bit_depth:
            report.files.append(FileStatus(p, "wrong_bit_depth", f"{frame.bit_depth} != {report.bit_depth}"))
        else:
            report.files.append(FileStatus(p, "ok"))
    return report
