"""Image export (SPEC §5). Clamping to [0, 1] happens HERE and only here."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import binascii
import json
import struct
import zlib

import imagecodecs
import numpy as np
import tifffile
from PIL import ExifTags, Image

from planefuse.io.atomic import atomic_output
from planefuse.io.metadata import ImageMetadata, build_xmp_packet

_TIFF_COMPRESSION = {"none": None, "lzw": "lzw", "zlib": "zlib"}


def _exif_bytes(metadata: ImageMetadata | None) -> bytes | None:
    if metadata is None:
        return None
    safe = metadata.without_single_exposure_fields()
    exif = Image.Exif()
    inverse = {name: tag for tag, name in ExifTags.TAGS.items()}
    for key, value in safe.exif.items():
        tag = inverse.get(key.rsplit(".", 1)[-1])
        if tag is not None and isinstance(value, (str, int, float, bytes)):
            exif[tag] = value
    exif[274] = 1
    return exif.tobytes() if len(exif) else None


def _xmp_bytes(metadata: ImageMetadata | None, provenance: dict[str, Any] | None) -> bytes | None:
    values: dict[str, Any] = {}
    if metadata is not None:
        values.update(metadata.without_single_exposure_fields().xmp)
    if provenance is not None:
        values["Xmp.PlaneFuse.Provenance"] = json.dumps(
            provenance, sort_keys=True, separators=(",", ":")
        )
    packet = build_xmp_packet(values)
    if packet is None and metadata is not None:
        return metadata.xmp_bytes
    return packet


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def _inject_png_metadata(path: Path, metadata: ImageMetadata | None, xmp: bytes | None) -> None:
    if metadata is None and xmp is None:
        return
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise OSError("PNG encoder returned an invalid signature")
    chunks: list[bytes] = []
    icc = metadata.icc if metadata is not None else None
    if icc:
        chunks.append(_png_chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc)))
    exif = _exif_bytes(metadata)
    if exif:
        chunks.append(_png_chunk(b"eXIf", exif.removeprefix(b"Exif\x00\x00")))
    if xmp:
        itxt = b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + xmp
        chunks.append(_png_chunk(b"iTXt", itxt))
    if not chunks:
        return
    position = 8
    ihdr_length = struct.unpack(">I", payload[position:position + 4])[0]
    after_ihdr = position + 12 + ihdr_length
    path.write_bytes(payload[:after_ihdr] + b"".join(chunks) + payload[after_ihdr:])


def _integer_pixels(clipped: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 16:
        return (clipped * 65535.0 + 0.5).astype(np.uint16)
    if bit_depth == 8:
        return (clipped * 255.0 + 0.5).astype(np.uint8)
    raise ValueError(f"unsupported bit depth {bit_depth}")


def _encode_pixels(
    clipped: np.ndarray,
    path: Path,
    *,
    bit_depth: int,
    compression: str,
    jpeg_quality: int,
    metadata: ImageMetadata | None,
    xmp: bytes | None,
) -> None:
    suffix = path.suffix.lower()
    if clipped.ndim not in (2, 3) or (clipped.ndim == 3 and clipped.shape[2] != 3):
        raise ValueError(f"expected grayscale or RGB image, got shape {clipped.shape}")
    if suffix in (".tif", ".tiff"):
        if compression not in _TIFF_COMPRESSION:
            raise ValueError(f"unknown compression {compression!r}")
        extratags: list[tuple[Any, ...]] = []
        if metadata is not None:
            safe = metadata.without_single_exposure_fields()
            if safe.icc:
                extratags.append((34675, 7, len(safe.icc), safe.icc, False))
            for tag, value in (
                (271, safe.camera_make or safe.exif.get("Exif.Image.Make")),
                (272, safe.camera_model or safe.exif.get("Exif.Image.Model")),
            ):
                if value:
                    encoded = value.encode("utf-8") + b"\x00"
                    extratags.append((tag, "s", len(encoded), encoded, False))
            extratags.append((274, "H", 1, 1, False))
        if xmp:
            extratags.append((700, 1, len(xmp), xmp, False))
        tifffile.imwrite(
            path,
            _integer_pixels(clipped, bit_depth),
            compression=_TIFF_COMPRESSION[compression],
            extratags=extratags,
        )
        return
    if suffix == ".png":
        path.write_bytes(imagecodecs.png_encode(_integer_pixels(clipped, bit_depth)))
        _inject_png_metadata(path, metadata, xmp)
        return
    if suffix in (".jpg", ".jpeg"):
        if clipped.ndim != 3:
            raise ValueError("JPEG export requires RGB pixels")
        kwargs: dict[str, Any] = {"quality": jpeg_quality}
        if metadata is not None and metadata.icc:
            kwargs["icc_profile"] = metadata.icc
        exif = _exif_bytes(metadata)
        if exif:
            kwargs["exif"] = exif
        if xmp:
            kwargs["xmp"] = xmp
        Image.fromarray(_integer_pixels(clipped, 8), mode="RGB").save(path, **kwargs)
        return
    if clipped.ndim == 2:
        raise ValueError(f"grayscale export supports .tif/.png, not {suffix!r}")
    raise ValueError(f"unsupported output format {suffix!r} (use .tif/.jpg/.png)")


def _validate_encoded(path: Path, expected_shape: tuple[int, ...]) -> None:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        decoded = tifffile.imread(path)
    elif suffix == ".png":
        decoded = imagecodecs.png_decode(path.read_bytes())
    else:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            decoded = np.asarray(image)
    if decoded.shape != expected_shape:
        raise OSError(f"encoded image shape {decoded.shape} does not match {expected_shape}")


def save_image(
    arr: np.ndarray,
    path: Path,
    *,
    bit_depth: int = 16,
    icc: bytes | None = None,
    compression: str = "zlib",
    jpeg_quality: int = 95,
    metadata: ImageMetadata | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """arr: float32 (H, W, 3) RGB or (H, W) grayscale, possibly outside [0, 1] (clamped here, SPEC §7.1)."""
    path = Path(path)
    clipped = np.clip(arr, 0.0, 1.0)
    if metadata is None and icc is not None:
        metadata = ImageMetadata(source_path=path, icc=icc)
    elif metadata is not None and icc is not None and metadata.icc != icc:
        from dataclasses import replace

        metadata = replace(metadata, icc=icc)

    with atomic_output(path) as temporary:
        xmp = _xmp_bytes(metadata, provenance)
        _encode_pixels(
            clipped,
            temporary,
            bit_depth=bit_depth,
            compression=compression,
            jpeg_quality=jpeg_quality,
            metadata=metadata,
            xmp=xmp,
        )
        _validate_encoded(temporary, clipped.shape)


def save_float_tiff(
    arr: np.ndarray,
    path: Path,
    *,
    compression: str = "zlib",
    metadata: ImageMetadata | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Write an unclamped 32-bit float RGB TIFF companion.

    This preserves the working scene-linear numbers, including negative values
    and highlight headroom. It is deliberately identified as TIFF, never as RAW.
    """

    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("32-bit float companion output must use .tif or .tiff")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"float TIFF requires RGB pixels, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("float TIFF pixels contain NaN or infinity")
    if compression not in _TIFF_COMPRESSION:
        raise ValueError(f"unknown compression {compression!r}")

    extratags: list[tuple[Any, ...]] = []
    if metadata is not None:
        safe = metadata.without_single_exposure_fields()
        if safe.icc:
            extratags.append((34675, 7, len(safe.icc), safe.icc, False))
        for tag, value in ((271, safe.camera_make), (272, safe.camera_model)):
            if value:
                encoded = value.encode("utf-8") + b"\x00"
                extratags.append((tag, "s", len(encoded), encoded, False))
        extratags.append((274, "H", 1, 1, False))
    xmp = _xmp_bytes(metadata, provenance)
    if xmp:
        extratags.append((700, 1, len(xmp), xmp, False))

    expected = np.asarray(arr, dtype=np.float32)
    with atomic_output(path) as temporary:
        tifffile.imwrite(
            temporary,
            expected,
            photometric="rgb",
            compression=_TIFF_COMPRESSION[compression],
            metadata=None,
            extratags=extratags,
        )
        decoded = tifffile.imread(temporary)
        if decoded.dtype != np.float32 or decoded.shape != expected.shape:
            raise OSError("float TIFF validation returned the wrong dtype or shape")
        if not np.array_equal(decoded, expected):
            raise OSError("float TIFF validation detected changed pixel values")
