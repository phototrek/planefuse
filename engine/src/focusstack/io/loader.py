"""Image loading and stack validation (SPEC §5, §12).

Working format: float32 (H, W, C) in [0, 1], source gamma kept, ICC bytes
carried through untouched. RGB only; grayscale and alpha are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import imagecodecs
import tifffile
from PIL import Image, ImageOps

from focusstack.errors import RawDecodeError, ValidationError
from focusstack.io.metadata import (
    ImageMetadata,
    ProcessingDomain,
    metadata_from_pillow,
    normalize_orientation,
    parse_xmp_packet,
)
from focusstack.io.raw import RAW_EXTENSIONS

RENDERED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
SUPPORTED = RENDERED_EXTENSIONS | RAW_EXTENSIONS
_ICC_TAG = 34675


@dataclass
class Frame:
    pixels: np.ndarray  # float32 (H, W, 3) in [0, 1]
    bit_depth: int  # 8 or 16
    metadata: ImageMetadata
    domain: ProcessingDomain = ProcessingDomain.RENDERED_RGB

    @property
    def icc(self) -> bytes | None:
        return self.metadata.icc

    @property
    def path(self) -> Path:
        return self.metadata.source_path


@dataclass
class FileStatus:
    path: Path
    status: str
    message: str = ""


@dataclass
class ValidationReport:
    files: list[FileStatus] = field(default_factory=list)
    width: int = 0
    height: int = 0
    bit_depth: int = 0
    domain: str = ""
    camera: str = ""
    decoder: dict[str, object] = field(default_factory=dict)

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
    if suffix in RAW_EXTENSIONS:
        from focusstack.io.raw import load_raw

        return load_raw(path)
    if suffix in (".tif", ".tiff"):
        metadata: ImageMetadata
        try:
            with Image.open(path) as im:
                metadata = metadata_from_pillow(path, im)
        except Exception:  # noqa: BLE001 - metadata absence must not block valid pixels
            metadata = ImageMetadata(source_path=path)
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            arr = page.asarray()
            if not isinstance(page, tifffile.TiffPage):
                raise ValidationError(f"{path.name}: unsupported TIFF frame layout")
            icc_tag = page.tags.get(_ICC_TAG)
            icc = bytes(icc_tag.value) if icc_tag is not None else metadata.icc
            make_tag = page.tags.get(271)
            model_tag = page.tags.get(272)
            xmp_tag = page.tags.get(700)
            make_value = str(make_tag.value) if make_tag is not None else None
            model_value = str(model_tag.value) if model_tag is not None else None
            xmp_value = bytes(xmp_tag.value) if xmp_tag is not None else None
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValidationError(f"{path.name}: RGB input required (got shape {arr.shape})")
        arr = normalize_orientation(arr, metadata.source_orientation)
        pixels, depth = _scale(arr)
        exif = dict(metadata.exif)
        make = make_value or metadata.camera_make
        model = model_value or metadata.camera_model
        if make:
            exif["Exif.Image.Make"] = make
        if model:
            exif["Exif.Image.Model"] = model
        xmp_bytes = xmp_value or metadata.xmp_bytes
        metadata = replace(
            metadata,
            icc=icc,
            exif=exif,
            xmp_bytes=xmp_bytes,
            xmp={**metadata.xmp, **parse_xmp_packet(xmp_bytes)},
            camera_make=make,
            camera_model=model,
            unique_camera_model=model or metadata.unique_camera_model,
        )
        return Frame(pixels, depth, metadata)
    if suffix == ".png":
        with Image.open(path) as im:
            metadata = metadata_from_pillow(path, im)
        arr = imagecodecs.png_decode(path.read_bytes())
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValidationError(f"{path.name}: RGB input required (got shape {arr.shape})")
        arr = normalize_orientation(arr, metadata.source_orientation)
        pixels, depth = _scale(arr)
        return Frame(pixels, depth, metadata)
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValidationError(f"{path.name}: RGB input required (got mode {im.mode})")
        metadata = metadata_from_pillow(path, im)
        arr = np.asarray(ImageOps.exif_transpose(im))
    pixels, depth = _scale(arr)
    return Frame(pixels, depth, metadata)


def validate_stack(paths: list[Path]) -> ValidationReport:
    """Per-file validation report (SPEC §12). Never raises for bad files."""
    report = ValidationReport()
    reference: Frame | None = None

    def mismatch(path: Path, status: str, field_name: str, expected: object, actual: object) -> FileStatus:
        return FileStatus(
            path,
            status,
            f"{field_name}: expected {expected!r}, actual {actual!r}",
        )

    def raw_calibration_missing(frame: Frame) -> list[str]:
        metadata = frame.metadata
        required = {
            "camera model": metadata.unique_camera_model,
            "black level": metadata.black_level,
            "white level": metadata.white_level,
            "color matrix": metadata.color_matrix1,
            "as-shot neutral": metadata.as_shot_neutral,
            "calibration illuminant": metadata.calibration_illuminant1,
        }
        return [name for name, value in required.items() if value is None]

    for p in paths:
        p = Path(p)
        try:
            frame = load_image(p)
        except RawDecodeError as e:
            report.files.append(FileStatus(p, "raw_decode_error", str(e)))
            continue
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
        if reference is None:
            reference = frame
            report.width, report.height, report.bit_depth = w, h, frame.bit_depth
            report.domain = frame.domain.value
            report.camera = frame.metadata.unique_camera_model or ""
            report.decoder = dict(frame.metadata.decoder)
            if frame.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
                missing = raw_calibration_missing(frame)
                if missing:
                    report.files.append(
                        FileStatus(p, "missing_raw_calibration", "missing: " + ", ".join(missing))
                    )
                    continue
            report.files.append(FileStatus(p, "ok"))
            continue

        if frame.domain is not reference.domain:
            report.files.append(
                mismatch(p, "mixed_domain", "processing domain", reference.domain.value, frame.domain.value)
            )
            continue

        if frame.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
            expected_metadata = reference.metadata
            actual_metadata = frame.metadata
            missing = raw_calibration_missing(frame)
            if missing:
                report.files.append(
                    FileStatus(p, "missing_raw_calibration", "missing: " + ", ".join(missing))
                )
                continue
            expected_camera = expected_metadata.unique_camera_model
            actual_camera = actual_metadata.unique_camera_model
            if actual_camera != expected_camera:
                report.files.append(
                    mismatch(p, "incompatible_camera", "camera", expected_camera, actual_camera)
                )
                continue
            expected_sensor = (
                expected_metadata.active_size,
                expected_metadata.source_orientation,
                expected_metadata.bits_per_sample,
            )
            actual_sensor = (
                actual_metadata.active_size,
                actual_metadata.source_orientation,
                actual_metadata.bits_per_sample,
            )
            if actual_sensor != expected_sensor:
                report.files.append(
                    mismatch(p, "incompatible_sensor_mode", "sensor mode", expected_sensor, actual_sensor)
                )
                continue
            expected_cfa = (expected_metadata.cfa_pattern, expected_metadata.color_description)
            actual_cfa = (actual_metadata.cfa_pattern, actual_metadata.color_description)
            if actual_cfa != expected_cfa:
                report.files.append(
                    mismatch(p, "incompatible_cfa", "CFA", expected_cfa, actual_cfa)
                )
                continue
            expected_calibration = (
                expected_metadata.black_level,
                expected_metadata.white_level,
                expected_metadata.color_matrix1,
                expected_metadata.color_matrix2,
                expected_metadata.calibration_illuminant1,
                expected_metadata.calibration_illuminant2,
            )
            actual_calibration = (
                actual_metadata.black_level,
                actual_metadata.white_level,
                actual_metadata.color_matrix1,
                actual_metadata.color_matrix2,
                actual_metadata.calibration_illuminant1,
                actual_metadata.calibration_illuminant2,
            )
            if actual_calibration != expected_calibration:
                report.files.append(
                    mismatch(
                        p,
                        "incompatible_raw_calibration",
                        "RAW calibration",
                        expected_calibration,
                        actual_calibration,
                    )
                )
                continue
            report.files.append(FileStatus(p, "ok"))
            continue

        if (w, h) != (report.width, report.height):
            report.files.append(FileStatus(p, "wrong_size", f"{w}x{h} != {report.width}x{report.height}"))
        elif frame.bit_depth != report.bit_depth:
            report.files.append(FileStatus(p, "wrong_bit_depth", f"{frame.bit_depth} != {report.bit_depth}"))
        else:
            report.files.append(FileStatus(p, "ok"))
    return report
