"""Deterministic, no-aesthetic-development camera RAW decoding."""

from __future__ import annotations

import math
import struct
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


# EXIF Orientation -> LibRaw's sizes.flip code, as LibRaw maps tag 274 when it parses a TIFF IFD.
_LIBRAW_FLIP = {1: 0, 2: 1, 3: 3, 4: 2, 5: 4, 6: 6, 7: 7, 8: 5}
_LINEAR_RAW = 34892


def _tag_numbers(tag: Any) -> tuple[float, ...]:
    """A numeric tag's values; RATIONAL / SRATIONAL arrive as flat (num, den) pairs."""
    value = tag.value
    items = list(value) if isinstance(value, (tuple, list, np.ndarray)) else [value]
    if int(tag.dtype) in (5, 10):
        return tuple(float(items[i]) / float(items[i + 1]) for i in range(0, len(items), 2))
    return tuple(float(item) for item in items)


def _per_sample(values: tuple[float, ...], samples: int, name: str, path: Path) -> tuple[float, ...]:
    if len(values) == 1:
        return values * samples
    if len(values) == samples:
        return values
    raise RawDecodeError(f"{path.name}: {name} has {len(values)} values for {samples} samples")


def _dng_ifds(tif: tifffile.TiffFile):
    def walk(page: Any):
        yield page
        if getattr(page, "subifds", None):
            for sub in tifffile.TiffPages(page):
                yield from walk(sub)

    for page in tif.pages:
        yield from walk(page)


def _main_dng_ifd(tif: tifffile.TiffFile) -> Any | None:
    for page in _dng_ifds(tif):
        subfile = page.tags.get(254)
        if subfile is None or int(subfile.value) == 0:
            return page
    return None


def _describe_ifd(page: Any) -> str:
    return (
        f"PhotometricInterpretation {int(page.photometric)}, {int(page.samplesperpixel)} samples, "
        f"compression {int(page.compression)}"
    )


def _linear_dng_ifd(tif: tifffile.TiffFile, path: Path, reason: str) -> tuple[Any, int]:
    """The full-resolution IFD of a 3-sample LinearRaw DNG and the file's EXIF orientation."""
    page = _main_dng_ifd(tif)
    if page is None:
        raise RawDecodeError(f"{path.name}: LibRaw cannot open this DNG ({reason}) and it has no main image")
    if int(page.photometric) != _LINEAR_RAW or int(page.samplesperpixel) != 3:
        raise RawDecodeError(
            f"{path.name}: LibRaw cannot open this DNG ({reason}); its main image is "
            f"{_describe_ifd(page)}, and only a 3-sample LinearRaw DNG is read without LibRaw"
        )
    for code, name in ((50715, "BlackLevelDeltaH"), (50716, "BlackLevelDeltaV")):
        if code in page.tags:
            raise RawDecodeError(f"{path.name}: LinearRaw DNG with a {name} is not supported")
    repeat = page.tags.get(50713)
    if repeat is not None and tuple(int(v) for v in repeat.value) != (1, 1):
        raise RawDecodeError(
            f"{path.name}: LinearRaw DNG with BlackLevelRepeatDim {tuple(repeat.value)} is not supported"
        )
    orientation_tag = tif.pages[0].tags.get(274)
    orientation = int(orientation_tag.value) if orientation_tag is not None else 1
    return page, orientation


def _active_area(page: Any) -> tuple[int, int, int, int]:
    height, width = int(page.imagelength), int(page.imagewidth)
    area = page.tags.get(50829)
    if area is None:
        return 0, 0, height, width
    top, left, bottom, right = (int(v) for v in area.value)
    return top, left, bottom, right


_OPCODE_NAMES = {
    1: "WarpRectilinear", 2: "WarpFisheye", 3: "FixVignetteRadial", 4: "FixBadPixelsConstant",
    5: "FixBadPixelsList", 6: "TrimBounds", 7: "MapTable", 8: "MapPolynomial", 9: "GainMap",
    10: "DeltaPerRow", 11: "DeltaPerColumn", 12: "ScalePerRow", 13: "ScalePerColumn",
    14: "WarpRectilinear2",
}
_OPCODE_OPTIONAL = 1
_OPCODE_LISTS = {51008: "OpcodeList1", 51009: "OpcodeList2", 51022: "OpcodeList3"}
_MAP_POLYNOMIAL = 8
# The newest DNG version whose opcodes this reader implements; a newer opcode is not processed.
_OPCODE_VERSION_CEILING = (1, 7, 0, 0)


def _opcode_name(opcode_id: int) -> str:
    return f"{_OPCODE_NAMES.get(opcode_id, 'unknown opcode')} ({opcode_id})"


def _parse_opcodes(blob: bytes, list_name: str, path: Path) -> list[tuple[int, tuple[int, ...], int, bytes]]:
    """(opcode id, DNG version, flags, parameter bytes) per opcode of a DNG opcode list (big-endian)."""
    try:
        (count,) = struct.unpack_from(">I", blob, 0)
        position, opcodes = 4, []
        for _ in range(count):
            opcode_id, version, flags, size = struct.unpack_from(">I4sII", blob, position)
            position += 16
            if position + size > len(blob):
                raise struct.error("parameters run past the end of the list")
            opcodes.append((opcode_id, tuple(version), flags, blob[position:position + size]))
            position += size
    except struct.error as exc:
        raise RawDecodeError(f"{path.name}: malformed {list_name}: {exc}") from exc
    return opcodes


def _map_polynomial(pixels: np.ndarray, params: bytes, path: Path) -> None:
    """DNG MapPolynomial, in place on normalized [0, 1] pixels: within the area, on every
    row_pitch-th row and col_pitch-th column, planes [plane, plane + planes) become
    sum(coefficient[i] * x**i), clipped to [0, 1]."""
    try:
        top, left, bottom, right, plane, planes, row_pitch, col_pitch, degree = struct.unpack_from(
            ">9I", params, 0
        )
        coefficients = struct.unpack_from(f">{degree + 1}d", params, 36)
    except struct.error as exc:
        raise RawDecodeError(f"{path.name}: malformed MapPolynomial parameters: {exc}") from exc
    height, width, channels = pixels.shape
    if row_pitch < 1 or col_pitch < 1 or planes < 1 or plane >= channels:
        raise RawDecodeError(
            f"{path.name}: MapPolynomial with plane {plane}, planes {planes}, pitch {row_pitch}x{col_pitch} "
            f"does not fit a {channels}-plane image"
        )
    area = pixels[
        top:min(bottom, height):row_pitch,
        left:min(right, width):col_pitch,
        plane:min(plane + planes, channels),
    ]
    if area.size == 0:
        return
    result = np.full(area.shape, np.float32(coefficients[-1]), dtype=np.float32)
    for coefficient in reversed(coefficients[:-1]):
        result *= area
        result += np.float32(coefficient)
    np.clip(result, 0.0, 1.0, out=result)
    area[...] = result


def _check_opcode_lists(page: Any, path: Path) -> list[bytes]:
    """The parameters of each OpcodeList2 MapPolynomial to apply. An opcode this reader does not
    implement, or one written for a DNG version newer than it supports, is skipped when optional
    and refused by name when not, in all three lists."""
    to_apply: list[bytes] = []
    for code, list_name in _OPCODE_LISTS.items():
        tag = page.tags.get(code)
        if tag is None:
            continue
        for opcode_id, version, flags, params in _parse_opcodes(bytes(tag.value), list_name, path):
            optional = bool(flags & _OPCODE_OPTIONAL)
            if version > _OPCODE_VERSION_CEILING:
                if optional:
                    continue
                raise RawDecodeError(
                    f"{path.name}: {list_name} holds {_opcode_name(opcode_id)} of DNG version "
                    f"{'.'.join(map(str, version))}, newer than this reader's "
                    f"{'.'.join(map(str, _OPCODE_VERSION_CEILING))}, and it is not optional"
                )
            if code == 51009 and opcode_id == _MAP_POLYNOMIAL:
                to_apply.append(params)
            elif not optional:
                raise RawDecodeError(
                    f"{path.name}: {list_name} holds {_opcode_name(opcode_id)}, which is not optional and is "
                    "not supported"
                )
    return to_apply


def linear_dng_has_map_polynomial(path: Path) -> bool:
    """True for a DNG whose full-resolution IFD is a 3-sample LinearRaw image with a MapPolynomial in
    OpcodeList2 (DxO's tone encoding). LibRaw silently ignores that curve, so such a DNG is read with
    tifffile even when LibRaw can open it; every other DNG goes to LibRaw first. Reads tags only."""
    try:
        with tifffile.TiffFile(path) as tif:
            page = _main_dng_ifd(tif)
            if page is None or int(page.photometric) != _LINEAR_RAW or int(page.samplesperpixel) != 3:
                return False
            tag = page.tags.get(51009)
            if tag is None:
                return False
            opcodes = _parse_opcodes(bytes(tag.value), "OpcodeList2", path)
    except Exception:  # noqa: BLE001 - unreadable header or list: the LibRaw path decides
        return False
    return any(opcode_id == _MAP_POLYNOMIAL for opcode_id, _, _, _ in opcodes)


def linear_dng_shape(path: Path) -> tuple[int, int] | None:
    """(height, width) of the frame `load_raw` makes from a LinearRaw DNG LibRaw cannot open."""
    path = Path(path)
    try:
        with tifffile.TiffFile(path) as tif:
            page, orientation = _linear_dng_ifd(tif, path, "header probe")
            top, left, bottom, right = _active_area(page)
    except Exception:  # noqa: BLE001 - a sizing probe never fails the job
        return None
    height, width = bottom - top, right - left
    if orientation in (5, 6, 7, 8):
        height, width = width, height
    return height, width


def _load_linear_dng(path: Path, reason: str, common: dict[str, Any], tags: dict[int, Any]):
    """Read a 3-sample LinearRaw DNG with tifffile (imagecodecs decodes the tiles or strips): the
    LinearizationTable if any, (sample - BlackLevel) / (WhiteLevel - BlackLevel) per channel,
    clipped to [0, 1], then the OpcodeList2 MapPolynomial curves (DxO stores a curve-encoded
    signal they linearize), then oriented. Without opcodes this is the frame LibRaw gives for the
    same DNG, and the metadata keeps LibRaw's per-channel black layout."""
    import imagecodecs

    from planefuse.io.metadata import normalize_orientation

    try:
        with tifffile.TiffFile(path) as tif:
            page, orientation = _linear_dng_ifd(tif, path, reason)
            compression = int(page.compression)
            if page.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
                raise RawDecodeError(f"{path.name}: LinearRaw DNG has unsupported sample type {page.dtype}")
            bits_per_sample = page.bitspersample
            bits = int(bits_per_sample[0] if isinstance(bits_per_sample, tuple) else bits_per_sample)
            black_tag = page.tags.get(50714)
            white_tag = page.tags.get(50717)
            black = _per_sample(_tag_numbers(black_tag), 3, "BlackLevel", path) if black_tag else (0.0,) * 3
            white = (
                _per_sample(_tag_numbers(white_tag), 3, "WhiteLevel", path)
                if white_tag
                else (float(2**bits - 1),) * 3
            )
            map_polynomials = _check_opcode_lists(page, path)
            linearization = page.tags.get(50712)
            top, left, bottom, right = _active_area(page)
            samples = page.asarray()
            if int(page.planarconfig) == 2:
                samples = np.moveaxis(samples, 0, -1)
            samples = samples[top:bottom, left:right]
            if linearization is not None:
                # DNG LinearizationTable: sample -> table[min(sample, len - 1)], before BlackLevel.
                table = np.asarray(linearization.value, dtype=np.uint16).reshape(-1)
                if table.size - 1 < np.iinfo(samples.dtype).max:
                    samples = np.minimum(samples, samples.dtype.type(table.size - 1))
                samples = table[samples]

        black_arr = np.array(black, dtype=np.float32)
        denominators = np.array(white, dtype=np.float32) - black_arr
        if np.any(denominators <= 0):
            raise RawDecodeError(f"{path.name}: invalid black/white calibration")
        pixels = samples.astype(np.float32)
        del samples
        pixels -= black_arr
        np.clip(pixels, 0.0, denominators, out=pixels)
        pixels /= denominators
        for params in map_polynomials:
            _map_polynomial(pixels, params, path)
        pixels = normalize_orientation(pixels, orientation)
    except RawDecodeError:
        raise
    except Exception as exc:  # tifffile, imagecodecs and numpy raise many types
        raise RawDecodeError(f"{path.name}: could not read the LinearRaw DNG with tifffile: {exc}") from exc
    white_level = max(white)
    metadata = ImageMetadata(
        **common,
        source_orientation=_LIBRAW_FLIP.get(orientation, 0) or 1,
        active_size=(bottom - top, right - left),
        cfa_pattern=None,
        color_description="RGBG",
        bits_per_sample=max(1, math.ceil(math.log2(white_level + 1.0))),
        black_level=(*black, 0.0),
        white_level=white_level,
        color_matrix1=_rational_values(tags.get(50721), 9),
        calibration_illuminant1=int(tags.get(50778, 21)),
        as_shot_neutral=_rational_values(tags.get(50728), 3),
        analog_balance=(1.0, 1.0, 1.0),
        decoder={
            "library": "tifffile/imagecodecs",
            "tifffile_version": tifffile.__version__,
            "imagecodecs_version": imagecodecs.__version__,
            "reason": reason,
            "compression": compression,
            "opcode_list2": [_opcode_name(_MAP_POLYNOMIAL)] * len(map_polynomials),
            "demosaic": "none (LinearRaw)",
            "white_balance": [1.0, 1.0, 1.0, 1.0],
            "gamma": [1.0, 1.0],
            "auto_brightness": False,
            "auto_scale": False,
            "output_color": "raw",
            "output_bps": 16,
            "highlight_mode": "clip",
        },
    )
    from planefuse.io.loader import Frame

    return Frame(
        pixels=pixels,
        bit_depth=metadata.bits_per_sample or 16,
        metadata=metadata,
        domain=ProcessingDomain.SCENE_LINEAR_CAMERA_RGB,
    )


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

    baseline_exposure = _rational_values(tags.get(50730), 1)
    common: dict[str, Any] = {
        "source_path": path,
        "icc": portable.icc,
        "exif": exif,
        "xmp": portable.xmp or parse_xmp_packet(xmp_bytes),
        "xmp_bytes": xmp_bytes,
        "camera_make": make,
        "camera_model": model,
        "unique_camera_model": unique_model,
        "camera_serial": _first(exifread_tags, "Image BodySerialNumber", "EXIF BodySerialNumber"),
        "lens_model": _first(exifread_tags, "EXIF LensModel", "Image LensModel"),
        "capture_time": _first(exifread_tags, "EXIF DateTimeOriginal", "Image DateTime"),
        "orientation": 1,
        "baseline_exposure": baseline_exposure[0] if baseline_exposure else 0.0,
    }

    if path.suffix.lower() == ".dng" and linear_dng_has_map_polynomial(path):
        return _load_linear_dng(path, "OpcodeList2 MapPolynomial, which LibRaw ignores", common, tags)

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
                **common,
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
                decoder=decoder,
            )
    except RawDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize rawpy/LibRaw's typed exception set
        if path.suffix.lower() == ".dng" and isinstance(exc, rawpy.LibRawFileUnsupportedError):
            # LibRaw cannot open a DNG 1.7 with JPEG XL tiles (DxO PhotoLab writes them).
            detail = exc.args[0] if exc.args else exc
            reason = detail.decode("ascii", errors="replace") if isinstance(detail, bytes) else str(detail)
            return _load_linear_dng(path, f"LibRaw: {reason}", common, tags)
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
