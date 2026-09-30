"""Deterministic, no-aesthetic-development camera RAW decoding."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from planefuse.errors import RawDecodeError
from planefuse.io.exiv2 import read_metadata
from planefuse.io.metadata import ImageMetadata, ProcessingDomain, parse_xmp_packet

RAW_EXTENSIONS = {
    ".3fr", ".arw", ".cr2", ".cr3", ".dng", ".erf", ".fff", ".iiq",
    ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".x3f",
}


def _tag_values(path: Path) -> dict[int, Any]:
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                return {}
            return {int(code): tag.value for code, tag in page.tags.items()}
    except Exception:  # noqa: BLE001 - many proprietary RAWs are not TIFF containers
        return {}


def _rational_values(value: Any, count: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    items = list(value) if isinstance(value, (tuple, list, np.ndarray)) else [value]
    if len(items) == count * 2 and all(isinstance(item, (int, np.integer)) for item in items):
        return tuple(float(items[i]) / float(items[i + 1]) for i in range(0, len(items), 2))
    if len(items) == count:
        return tuple(float(item) for item in items)
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8", errors="replace")
    return str(value).rstrip("\x00")


def _exifread(path: Path) -> dict[str, Any]:
    try:
        import exifread

        with path.open("rb") as stream:
            tags = exifread.process_file(stream, details=False, extract_thumbnail=False, strict=False)
        return {str(key): str(value) for key, value in tags.items()}
    except Exception:  # noqa: BLE001 - LibRaw remains the decode authority
        return {}


def _exiftool_tags(path: Path) -> dict[str, Any]:
    """Fallback for ISOBMFF RAWs (CR3) that exifread cannot parse; needs exiftool on PATH."""
    import json, subprocess
    try:
        out = subprocess.run(["exiftool", "-j", "-n", "-Make", "-Model", "-DateTimeOriginal",
                              "-LensModel", "-SerialNumber", str(path)],
                             capture_output=True, text=True, timeout=30).stdout
        d = json.loads(out)[0]
    except Exception:  # noqa: BLE001
        return {}
    m = {"Image Make": d.get("Make"), "Image Model": d.get("Model"),
         "EXIF DateTimeOriginal": d.get("DateTimeOriginal"), "EXIF LensModel": d.get("LensModel"),
         "Image BodySerialNumber": d.get("SerialNumber")}
    return {k: str(v) for k, v in m.items() if v not in (None, "")}


def _first(values: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = values.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _color_matrix(raw: Any, tags: dict[int, Any]) -> tuple[float, ...] | None:
    source = _rational_values(tags.get(50721), 9)
    if source is not None:
        return source
    # LibRaw's cam_xyz (rawpy.rgb_xyz_matrix) is already the DNG ColorMatrix1
    # convention (XYZ -> cameraRGB); an all-zeros matrix means unknown camera.
    xyz_to_camera = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    if xyz_to_camera.shape != (3, 3) or abs(float(np.linalg.det(xyz_to_camera))) < 1e-10:
        return None
    return tuple(float(value) for value in xyz_to_camera.reshape(-1))


def _neutral(raw: Any, tags: dict[int, Any]) -> tuple[float, ...] | None:
    source = _rational_values(tags.get(50728), 3)
    if source is not None:
        return source
    wb = np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
    if wb.shape != (3,) or not np.all(np.isfinite(wb)) or np.any(wb <= 0):
        return None
    neutral = 1.0 / wb
    neutral /= neutral[1]
    return tuple(float(value) for value in neutral)


def load_raw(path: Path):
    path = Path(path)
    if path.suffix.lower() not in RAW_EXTENSIONS:
        raise RawDecodeError(f"{path.name}: unsupported camera RAW extension {path.suffix.lower()}")
    try:
        import rawpy
    except (ImportError, OSError) as exc:
        raise RawDecodeError(f"{path.name}: RAW support is not installed ({exc})") from exc

    tags = _tag_values(path)
    portable = read_metadata(path)
    exifread_tags = _exifread(path) or _exiftool_tags(path)
    exif = dict(portable.exif)
    exif.update({f"ExifRead.{key}": value for key, value in exifread_tags.items()})
    make = _text(tags.get(271)) or _first(exifread_tags, "Image Make", "Make")
    model = _text(tags.get(272)) or _first(exifread_tags, "Image Model", "Model")
    # Canon (and others) already put the make into the model string; do not repeat it.
    unique_model = _text(tags.get(50708)) or (
        model if (make and model and model.lower().startswith(make.lower())) else
        " ".join(part for part in (make, model) if part)
    ) or None
    xmp_bytes = portable.raw_xmp
    if tags.get(700) is not None:
        xmp_bytes = bytes(tags[700])

    decoder = {
        "library": "rawpy/LibRaw",
        "rawpy_version": rawpy.__version__,
        "libraw_version": list(rawpy.libraw_version),
        "demosaic": "AHD",
        "white_balance": [1.0, 1.0, 1.0, 1.0],
        "gamma": [1.0, 1.0],
        "auto_brightness": False,
        "auto_scale": False,
        "output_color": "raw",
        "output_bps": 16,
        "denoise": False,
        "median_filter_passes": 0,
        "highlight_mode": "clip",
    }
    try:
        with rawpy.imread(str(path)) as raw:
            black = tuple(float(value) for value in raw.black_level_per_channel)
            white = float(raw.white_level)
            pattern = raw.raw_pattern
            color_description = raw.color_desc.decode("ascii", errors="replace")
            params = rawpy.Params(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                median_filter_passes=0,
                use_camera_wb=False,
                use_auto_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0],
                output_color=rawpy.ColorSpace.raw,
                output_bps=16,
                no_auto_bright=True,
                no_auto_scale=True,
                gamma=(1.0, 1.0),
                highlight_mode=rawpy.HighlightMode.Clip,
            )
            decoded = raw.postprocess(params=params)
            if decoded.ndim != 3 or decoded.shape[2] != 3:
                raise RawDecodeError(f"{path.name}: LibRaw returned unsupported shape {decoded.shape}")
            green_black = (black[1] + black[3]) / 2.0
            denominators = np.array(
                [white - black[0], white - green_black, white - black[2]], dtype=np.float32
            )
            if np.any(denominators <= 0):
                raise RawDecodeError(f"{path.name}: invalid black/white calibration")
            pixels = decoded.astype(np.float32) / denominators.reshape(1, 1, 3)
            metadata = ImageMetadata(
                source_path=path,
                icc=portable.icc,
                exif=exif,
                xmp=portable.xmp or parse_xmp_packet(xmp_bytes),
                xmp_bytes=xmp_bytes,
                camera_make=make,
                camera_model=model,
                unique_camera_model=unique_model,
                camera_serial=_first(exifread_tags, "Image BodySerialNumber", "EXIF BodySerialNumber"),
                lens_model=_first(exifread_tags, "EXIF LensModel", "Image LensModel"),
                capture_time=_first(exifread_tags, "EXIF DateTimeOriginal", "Image DateTime"),
                orientation=1,
                source_orientation=int(getattr(raw.sizes, "flip", 0) or 1),
                active_size=(int(raw.sizes.height), int(raw.sizes.width)),
                cfa_pattern=tuple(int(value) for value in pattern.reshape(-1)) if pattern is not None else None,
                color_description=color_description,
                bits_per_sample=max(1, int(math.ceil(math.log2(white + 1.0)))),
                black_level=black,
                white_level=white,
                color_matrix1=_color_matrix(raw, tags),
                calibration_illuminant1=int(tags.get(50778, 21)),
                as_shot_neutral=_neutral(raw, tags),
                analog_balance=(1.0, 1.0, 1.0),
                baseline_exposure=0.0,
                decoder=decoder,
            )
    except RawDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize rawpy/LibRaw's typed exception set
        camera = " ".join(part for part in (make, model) if part)
        detail = f" ({camera})" if camera else ""
        raise RawDecodeError(f"{path.name}{detail}: LibRaw decode failed: {exc}") from exc

    from planefuse.io.loader import Frame

    return Frame(
        pixels=pixels,
        bit_depth=metadata.bits_per_sample or 16,
        metadata=metadata,
        domain=ProcessingDomain.SCENE_LINEAR_CAMERA_RGB,
    )
