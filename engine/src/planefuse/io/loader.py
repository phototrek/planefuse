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

from planefuse.errors import RawDecodeError, ValidationError
from planefuse.io.metadata import (
    ImageMetadata,
    ProcessingDomain,
    metadata_from_pillow,
    normalize_orientation,
    parse_xmp_packet,
)
from planefuse.io.raw import RAW_EXTENSIONS

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
        from planefuse.io.raw import load_raw

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


@dataclass
class _FrameProbe:
    path: Path
    width: int
    height: int
    bit_depth: int
    domain: ProcessingDomain
    metadata: ImageMetadata | None = None
    decoder: dict[str, object] = field(default_factory=dict)


def _oriented_size(width: int, height: int, orientation: int) -> tuple[int, int]:
    if orientation in {5, 6, 7, 8}:
        return height, width
    return width, height


def _probe_png(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as stream:
        header = stream.read(33)
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValidationError(f"{path.name}: unreadable PNG header")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    bit_depth = int(header[24])
    color_type = int(header[25])
    if color_type != 2:
        raise ValidationError(f"{path.name}: RGB input required (got PNG color type {color_type})")
    if bit_depth not in (8, 16):
        raise ValidationError(f"unsupported sample type {bit_depth}-bit; expected uint8 or uint16")
    return width, height, bit_depth


def _probe_rendered_image(path: Path) -> tuple[int, int, int]:
    suffix = path.suffix.lower()
    if suffix not in RENDERED_EXTENSIONS:
        raise ValidationError(f"{path.name}: unsupported format {suffix}")
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                raise ValidationError(f"{path.name}: unsupported TIFF frame layout")
            samples = int(page.samplesperpixel)
            if samples != 3:
                raise ValidationError(f"{path.name}: RGB input required (got {samples} channels)")
            bits = page.bitspersample
            if isinstance(bits, tuple):
                depth = int(bits[0])
                if any(int(item) != depth for item in bits):
                    raise ValidationError(f"{path.name}: RGB input required (mixed bit depths)")
            else:
                depth = int(bits)
            if depth not in (8, 16):
                raise ValidationError(f"unsupported sample type {depth}-bit; expected uint8 or uint16")
            orientation_tag = page.tags.get(274)
            orientation = int(orientation_tag.value) if orientation_tag is not None else 1
            width, height = _oriented_size(
                int(page.imagewidth),
                int(page.imagelength),
                orientation,
            )
            return width, height, depth
    if suffix == ".png":
        return _probe_png(path)
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValidationError(f"{path.name}: RGB input required (got mode {im.mode})")
        orientation = int(im.getexif().get(274, 1) or 1)
        width, height = _oriented_size(im.width, im.height, orientation)
        return width, height, 8


def _probe_from_frame(path: Path, frame: Frame) -> _FrameProbe:
    height, width = frame.pixels.shape[:2]
    return _FrameProbe(
        path=path,
        width=width,
        height=height,
        bit_depth=frame.bit_depth,
        domain=frame.domain,
        metadata=frame.metadata,
        decoder=dict(frame.metadata.decoder),
    )


def _probe_frame(path: Path, loaded: Frame | BaseException | None = None) -> _FrameProbe:
    """Probe one file. A RAW file is probed by decoding it; when the caller has
    already decoded it, `loaded` (the Frame, or the exception its decode raised)
    stands in for that decode so the file is not decoded twice."""
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        if isinstance(loaded, BaseException):
            raise loaded
        frame = loaded if loaded is not None else load_image(path)
        return _probe_from_frame(path, frame)
    width, height, depth = _probe_rendered_image(path)
    return _FrameProbe(
        path=path,
        width=width,
        height=height,
        bit_depth=depth,
        domain=ProcessingDomain.RENDERED_RGB,
    )


def probe_frame_shape(path: Path) -> tuple[int, int] | None:
    """Cheap (height, width) of the decoded frame, without decoding pixels.

    Used only to size memory budgets; returns None when the header cannot be
    read (validation reports the file properly).
    """
    path = Path(path)
    try:
        if path.suffix.lower() in RAW_EXTENSIONS:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                sizes = raw.sizes
                height, width = int(sizes.iheight), int(sizes.iwidth)
                if int(getattr(sizes, "flip", 0) or 0) in (5, 6):
                    height, width = width, height
                return height, width
        width, height, _depth = _probe_rendered_image(path)
        return height, width
    except Exception:  # noqa: BLE001 - a sizing probe must never fail the job
        return None


def validate_stack(
    paths: list[Path],
    loaded: list[Frame | BaseException | None] | None = None,
) -> ValidationReport:
    """Per-file validation report (SPEC §12). Never raises for bad files.

    `loaded`, when given, holds each path's already-decoded Frame (or the
    exception its decode raised), index-aligned with `paths`; RAW files are then
    validated from it instead of being decoded again. The checks and messages
    are the same either way.
    """
    report = ValidationReport()
    reference: _FrameProbe | None = None

    def mismatch(path: Path, status: str, field_name: str, expected: object, actual: object) -> FileStatus:
        return FileStatus(
            path,
            status,
            f"{field_name}: expected {expected!r}, actual {actual!r}",
        )

    def raw_calibration_missing(metadata: ImageMetadata) -> list[str]:
        required = {
            "camera model": metadata.unique_camera_model,
            "black level": metadata.black_level,
            "white level": metadata.white_level,
            "color matrix": metadata.color_matrix1,
            "as-shot neutral": metadata.as_shot_neutral,
            "calibration illuminant": metadata.calibration_illuminant1,
        }
        return [name for name, value in required.items() if value is None]

    for index, p in enumerate(paths):
        p = Path(p)
        try:
            probe = _probe_frame(p, loaded[index] if loaded is not None else None)
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
        if reference is None:
            reference = probe
            report.width, report.height, report.bit_depth = (
                probe.width,
                probe.height,
                probe.bit_depth,
            )
            report.domain = probe.domain.value
            report.camera = (probe.metadata.unique_camera_model or "") if probe.metadata is not None else ""
            report.decoder = dict(probe.decoder)
            if probe.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
                if probe.metadata is None:
                    report.files.append(FileStatus(p, "missing_raw_calibration", "missing metadata"))
                    continue
                missing = raw_calibration_missing(probe.metadata)
                if missing:
                    report.files.append(
                        FileStatus(p, "missing_raw_calibration", "missing: " + ", ".join(missing))
                    )
                    continue
            report.files.append(FileStatus(p, "ok"))
            continue

        if probe.domain is not reference.domain:
            report.files.append(
                mismatch(p, "mixed_domain", "processing domain", reference.domain.value, probe.domain.value)
            )
            continue

        if probe.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
            if reference.metadata is None or probe.metadata is None:
                report.files.append(FileStatus(p, "missing_raw_calibration", "missing metadata"))
                continue
            expected_metadata = reference.metadata
            actual_metadata = probe.metadata
            missing = raw_calibration_missing(actual_metadata)
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

        if (probe.width, probe.height) != (report.width, report.height):
            report.files.append(
                FileStatus(
                    p,
                    "wrong_size",
                    f"{probe.width}x{probe.height} != {report.width}x{report.height}",
                )
            )
        elif probe.bit_depth != report.bit_depth:
            report.files.append(
                FileStatus(p, "wrong_bit_depth", f"{probe.bit_depth} != {report.bit_depth}")
            )
        else:
            report.files.append(FileStatus(p, "ok"))
    return report
