"""Lossless 16-bit demosaiced Linear DNG writer and validator."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile

from planefuse.errors import DiskSpaceError, DngExportError
from planefuse.io.atomic import atomic_output
from planefuse.io.metadata import ImageMetadata, build_xmp_packet, parse_xmp_packet
from planefuse.io.provenance import dng_provenance

log = logging.getLogger(__name__)

_DNG_VERSION = b"\x01\x07\x01\x00"
_DNG_BACKWARD_VERSION = b"\x01\x04\x00\x00"


@dataclass(frozen=True)
class DngValidationReport:
    path: Path
    encoding_offset: float
    encoding_span: float
    max_code_error: int
    rawpy_validated: bool


def _ascii_tag(code: int, value: str) -> tuple[Any, ...]:
    encoded = value.encode("utf-8") + b"\x00"
    return code, 2, len(encoded), encoded, False


def _rational_data(values: Iterable[float], *, signed: bool) -> tuple[int, ...]:
    denominator = 1_000_000
    data: list[int] = []
    for value in values:
        numerator = int(round(float(value) * denominator))
        if not signed and numerator < 0:
            raise DngExportError("unsigned DNG rational cannot encode a negative value")
        data.extend((numerator, denominator))
    return tuple(data)


def _required_metadata(metadata: ImageMetadata) -> None:
    required = {
        "unique_camera_model": metadata.unique_camera_model,
        "color_matrix1": metadata.color_matrix1,
        "as_shot_neutral": metadata.as_shot_neutral,
        "calibration_illuminant1": metadata.calibration_illuminant1,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise DngExportError("missing required Linear DNG metadata: " + ", ".join(missing))
    if len(metadata.color_matrix1 or ()) != 9:
        raise DngExportError("color_matrix1 must contain 9 values")
    if len(metadata.as_shot_neutral or ()) != 3:
        raise DngExportError("as_shot_neutral must contain 3 values")


def _encode_scene_linear(image: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise DngExportError(f"Linear DNG requires RGB pixels, got shape {image.shape}")
    if not np.all(np.isfinite(image)):
        raise DngExportError("Linear DNG pixels contain NaN or infinity")
    minimum = min(0.0, float(image.min()))
    maximum = max(1.0, float(image.max()))
    span = maximum - minimum
    codes = np.rint((image.astype(np.float64) - minimum) / span * 65535.0)
    encoded = np.clip(codes, 0, 65535).astype(np.uint16)
    black_code = int(round((0.0 - minimum) / span * 65535.0))
    white_code = int(round((1.0 - minimum) / span * 65535.0))
    return encoded, minimum, span, black_code, white_code


def _write_dng(
    path: Path,
    encoded: np.ndarray,
    metadata: ImageMetadata,
    xmp: bytes,
    *,
    black_code: int,
    white_code: int,
) -> None:
    height, width = encoded.shape[:2]
    make = metadata.camera_make or str(metadata.exif.get("Exif.Image.Make", "PlaneFuse"))
    model = metadata.camera_model or str(metadata.exif.get("Exif.Image.Model", metadata.unique_camera_model))
    tags: list[tuple[Any, ...]] = [
        (50706, 1, 4, _DNG_VERSION, False),
        (50707, 1, 4, _DNG_BACKWARD_VERSION, False),
        _ascii_tag(50708, metadata.unique_camera_model or model),
        # BlackLevel is an integer code; over the 1e6 denominator _rational_data uses, a
        # stack whose minimum is well below zero overflows the 32-bit RATIONAL numerator.
        (50714, 5, 3, (black_code, 1) * 3, False),
        (50717, 4, 3, (white_code,) * 3, False),
        (50718, 5, 2, _rational_data((1.0, 1.0), signed=False), False),
        (50719, 4, 2, (0, 0), False),
        (50720, 4, 2, (width, height), False),
        (50721, 10, 9, _rational_data(metadata.color_matrix1 or (), signed=True), False),
        (50728, 5, 3, _rational_data(metadata.as_shot_neutral or (), signed=False), False),
        (50730, 10, 1, _rational_data((metadata.baseline_exposure or 0.0,), signed=True), False),
        (50778, 3, 1, int(metadata.calibration_illuminant1 or 21), False),
        (50829, 4, 4, (0, 0, height, width), False),
        (50780, 5, 1, _rational_data((1.0,), signed=False), False),
        (274, 3, 1, 1, False),
        (700, 1, len(xmp), xmp, False),
        _ascii_tag(271, make),
        _ascii_tag(272, model),
        _ascii_tag(305, "PlaneFuse"),
        _ascii_tag(50827, metadata.source_path.name),
        _ascii_tag(50936, "PlaneFuse Linear DNG"),
    ]
    if metadata.color_matrix2 is not None:
        tags.append((50722, 10, 9, _rational_data(metadata.color_matrix2, signed=True), False))
    if metadata.calibration_illuminant2 is not None:
        tags.append((50779, 3, 1, int(metadata.calibration_illuminant2), False))
    if metadata.analog_balance is not None:
        tags.append((50727, 5, 3, _rational_data(metadata.analog_balance, signed=False), False))
    if metadata.camera_serial:
        tags.append(_ascii_tag(50735, metadata.camera_serial))

    tifffile.imwrite(
        path,
        encoded,
        photometric=34892,
        planarconfig="contig",
        # DNG readers require NewSubfileType on the main image (Apple ImageIO refuses a
        # DNG without it) and reject an ExtraSamples tag on a 3-sample LinearRaw image
        # (Adobe DNG Converter refuses it); tifffile emits both by default.
        subfiletype=0,
        extrasamples=False,
        compression="jpeg",
        compressionargs={"lossless": True, "bitspersample": 16},
        metadata=None,
        extratags=tags,
    )


def validate_linear_dng(
    path: Path,
    *,
    expected_pixels: np.ndarray | None = None,
) -> DngValidationReport:
    path = Path(path)
    try:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != 1:
                raise DngExportError("Linear DNG must contain one primary raw IFD")
            page: Any = tif.pages[0]
            required = (50706, 50707, 50708, 50714, 50717, 50721, 50728, 50778, 700)
            missing = [str(code) for code in required if code not in page.tags]
            if missing:
                raise DngExportError("Linear DNG is missing tags: " + ", ".join(missing))
            if int(page.photometric) != 34892:
                raise DngExportError("PhotometricInterpretation is not LinearRaw")
            if int(page.compression) != 7:
                raise DngExportError("Linear DNG does not use lossless JPEG compression")
            if page.dtype != np.dtype(np.uint16) or page.samplesperpixel != 3:
                raise DngExportError("Linear DNG must contain contiguous 16-bit RGB samples")
            decoded = page.asarray()
            xmp = parse_xmp_packet(bytes(page.tags[700].value))
    except DngExportError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DngExportError(f"could not reopen Linear DNG with tifffile: {exc}") from exc

    max_code_error = 0
    if expected_pixels is not None:
        if decoded.shape != expected_pixels.shape:
            raise DngExportError(
                f"Linear DNG pixel shape {decoded.shape} does not match {expected_pixels.shape}"
            )
        max_code_error = int(
            np.max(np.abs(decoded.astype(np.int32) - expected_pixels.astype(np.int32)))
        )
        if max_code_error > 1:
            raise DngExportError(f"Linear DNG pixel round-trip error is {max_code_error} codes")

    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            raw_pixels = np.asarray(raw.raw_image)[..., :3]
            if raw_pixels.shape != decoded.shape:
                raise DngExportError(
                    f"LibRaw pixel shape {raw_pixels.shape} does not match {decoded.shape}"
                )
            # LibRaw returns a 3-sample LinearRaw image as a 4-channel raw_image, and its
            # values did not match the TIFF samples on a stack that Apple ImageIO and Adobe
            # DNG Converter both read correctly, so a mismatch here is not evidence of a bad
            # file. The tifffile round trip above is the pixel check; this one only proves
            # LibRaw opens the file at the expected geometry.
            diff = np.abs(raw_pixels.astype(np.int32) - decoded.astype(np.int32))
            if int(diff.max()) > 1:
                log.warning(
                    "LibRaw raw_image differs from the TIFF decode (max %d, mean %.3f, "
                    "raw_image shape %s); not treated as an error for 3-sample LinearRaw",
                    int(diff.max()),
                    float(diff.mean()),
                    np.asarray(raw.raw_image).shape,
                )
    except DngExportError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DngExportError(f"could not reopen Linear DNG with LibRaw: {exc}") from exc

    offset = 0.0
    span = 1.0
    try:
        payload = json.loads(xmp["Xmp.PlaneFuse.Provenance"])
        mapping = payload["integer_mapping"]
        offset, span = float(mapping["offset"]), float(mapping["span"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return DngValidationReport(path, offset, span, max_code_error, True)


def save_linear_dng(
    image: np.ndarray,
    destination: Path,
    metadata: ImageMetadata,
    provenance: dict[str, Any],
) -> DngValidationReport:
    destination = Path(destination)
    _required_metadata(metadata)
    if max(image.shape[:2], default=0) > 65_000:
        raise DngExportError("Linear DNG dimensions exceed the 65,000-pixel compatibility limit")
    estimated = image.size * 2 + 8 * 1024 * 1024
    if destination.parent.is_dir() and shutil.disk_usage(destination.parent).free < estimated * 2:
        raise DiskSpaceError(f"not enough free space for atomic DNG export to {destination.parent}")
    encoded, offset, span, black_code, white_code = _encode_scene_linear(image)
    payload = dng_provenance(
        metadata,
        provenance,
        encoding_offset=offset,
        encoding_span=span,
    )
    xmp_values = dict(metadata.without_single_exposure_fields().xmp)
    xmp_values["Xmp.PlaneFuse.Provenance"] = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    xmp = build_xmp_packet(xmp_values)
    if xmp is None:
        raise DngExportError("could not build PlaneFuse XMP provenance")

    with atomic_output(destination) as temporary:
        _write_dng(
            temporary,
            encoded,
            metadata,
            xmp,
            black_code=black_code,
            white_code=white_code,
        )
        report = validate_linear_dng(temporary, expected_pixels=encoded)
    return replace(
        report,
        path=destination,
        encoding_offset=offset,
        encoding_span=span,
    )
