"""Typed image metadata and processing-domain contracts.

Metadata is kept separate from pixel decoding so rendered and camera-RAW
frames can share the same downstream stack interfaces without mixing their
color/transfer-function assumptions.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from PIL import ExifTags, Image

from planefuse.io.exiv2 import read_metadata


class ProcessingDomain(str, Enum):
    RENDERED_RGB = "rendered_rgb"
    SCENE_LINEAR_CAMERA_RGB = "scene_linear_camera_rgb"


_SINGLE_EXPOSURE_KEY_PARTS = (
    "focusdistance",
    "subjectdistance",
    "depthoffield",
)

_XMP_NAMESPACES = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "aux": "http://ns.adobe.com/exif/1.0/aux/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "PlaneFuse": "https://planefuse.app/ns/1.0/",
}
_XMP_PREFIXES = {namespace: prefix for prefix, namespace in _XMP_NAMESPACES.items()}


def _frozen_map(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ImageMetadata:
    source_path: Path
    icc: bytes | None = None
    exif_bytes: bytes | None = None
    xmp_bytes: bytes | None = None
    exif: Mapping[str, Any] = field(default_factory=dict)
    xmp: Mapping[str, Any] = field(default_factory=dict)
    camera_make: str | None = None
    camera_model: str | None = None
    unique_camera_model: str | None = None
    camera_serial: str | None = None
    lens_model: str | None = None
    capture_time: str | None = None
    orientation: int = 1
    source_orientation: int = 1
    active_size: tuple[int, int] | None = None
    cfa_pattern: tuple[int, ...] | None = None
    color_description: str | None = None
    bits_per_sample: int | None = None
    black_level: tuple[float, ...] | None = None
    white_level: float | None = None
    color_matrix1: tuple[float, ...] | None = None
    color_matrix2: tuple[float, ...] | None = None
    calibration_illuminant1: int | None = None
    calibration_illuminant2: int | None = None
    as_shot_neutral: tuple[float, ...] | None = None
    analog_balance: tuple[float, ...] | None = None
    baseline_exposure: float | None = None
    decoder: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "exif", _frozen_map(self.exif))
        object.__setattr__(self, "xmp", _frozen_map(self.xmp))
        object.__setattr__(self, "decoder", _frozen_map(self.decoder))

    def without_single_exposure_fields(self) -> ImageMetadata:
        def safe(values: Mapping[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in values.items()
                if not any(part in key.replace("_", "").lower() for part in _SINGLE_EXPOSURE_KEY_PARTS)
            }

        return replace(self, exif=safe(self.exif), xmp=safe(self.xmp))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


def metadata_to_dict(metadata: ImageMetadata) -> dict[str, Any]:
    """Return a complete JSON-safe representation for project persistence."""

    values: dict[str, Any] = {}
    for name in ImageMetadata.__dataclass_fields__:
        value = getattr(metadata, name)
        values[name] = str(value) if name == "source_path" else _json_safe(value)
    return values


def _decode_blob(value: Any) -> bytes | None:
    if isinstance(value, dict) and isinstance(value.get("base64"), str):
        return base64.b64decode(value["base64"])
    return None


def metadata_from_dict(values: Mapping[str, Any]) -> ImageMetadata:
    """Restore metadata persisted by :func:`metadata_to_dict`."""

    tuple_fields = {
        "active_size",
        "cfa_pattern",
        "black_level",
        "color_matrix1",
        "color_matrix2",
        "as_shot_neutral",
        "analog_balance",
    }
    payload = dict(values)
    payload["source_path"] = Path(str(payload.get("source_path", "")))
    for name in ("icc", "exif_bytes", "xmp_bytes"):
        payload[name] = _decode_blob(payload.get(name))
    for name in tuple_fields:
        if payload.get(name) is not None:
            payload[name] = tuple(payload[name])
    allowed = set(ImageMetadata.__dataclass_fields__)
    return ImageMetadata(**{key: value for key, value in payload.items() if key in allowed})


def _exif_key(tag: int) -> str:
    name = ExifTags.TAGS.get(tag, str(tag))
    if name in {"Make", "Model", "Orientation", "DateTime", "Software"}:
        return f"Exif.Image.{name}"
    return f"Exif.Photo.{name}"


def parse_xmp_packet(packet: bytes | None) -> dict[str, str]:
    if not packet:
        return {}
    try:
        root = ET.fromstring(packet.decode("utf-8", errors="replace"))
    except ET.ParseError:
        return {}
    values: dict[str, str] = {}
    for element in root.iter():
        for raw_key, value in element.attrib.items():
            if not raw_key.startswith("{"):
                continue
            namespace, local = raw_key[1:].split("}", 1)
            prefix = _XMP_PREFIXES.get(namespace)
            if prefix:
                values[f"Xmp.{prefix}.{local}"] = value
        if element.tag.startswith("{") and element.text and element.text.strip():
            namespace, local = element.tag[1:].split("}", 1)
            prefix = _XMP_PREFIXES.get(namespace)
            if prefix:
                values[f"Xmp.{prefix}.{local}"] = element.text.strip()
    return values


def build_xmp_packet(values: Mapping[str, Any]) -> bytes | None:
    if not values:
        return None
    for prefix, namespace in _XMP_NAMESPACES.items():
        ET.register_namespace(prefix, namespace)
    ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    ET.register_namespace("x", "adobe:ns:meta/")
    root = ET.Element("{adobe:ns:meta/}xmpmeta")
    rdf = ET.SubElement(root, "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF")
    description = ET.SubElement(rdf, "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    for key, value in sorted(values.items()):
        parts = key.split(".", 2)
        if len(parts) != 3 or parts[0] != "Xmp":
            continue
        tag_namespace = _XMP_NAMESPACES.get(parts[1])
        if tag_namespace is None:
            continue
        description.set(f"{{{tag_namespace}}}{parts[2]}", str(value))
    return ET.tostring(root, encoding="utf-8", xml_declaration=False)


def metadata_from_pillow(path: Path, image: Image.Image) -> ImageMetadata:
    """Extract portable metadata available through Pillow.

    A pyexiv2 adapter augments this representation when that optional native
    dependency is installed; Pillow keeps loading functional on unsupported
    metadata/platform combinations.
    """

    exif_obj = image.getexif()
    portable = read_metadata(path)
    exif = {_exif_key(int(tag)): value for tag, value in exif_obj.items()}
    exif.update(portable.exif)
    orientation = int(exif_obj.get(274, 1) or 1)
    info = image.info
    exif_bytes = info.get("exif")
    xmp_bytes = info.get("xmp") or portable.raw_xmp
    icc = info.get("icc_profile") or portable.icc
    make = exif.get("Exif.Image.Make") or exif_obj.get(271)
    model = exif.get("Exif.Image.Model") or exif_obj.get(272)
    lens = exif.get("Exif.Photo.LensModel") or exif_obj.get(42036)
    captured = exif.get("Exif.Photo.DateTimeOriginal") or exif_obj.get(36867) or exif_obj.get(306)
    return ImageMetadata(
        source_path=path,
        icc=bytes(icc) if icc else None,
        exif_bytes=bytes(exif_bytes) if exif_bytes else None,
        xmp_bytes=bytes(xmp_bytes) if xmp_bytes else None,
        exif=exif,
        xmp={**parse_xmp_packet(bytes(xmp_bytes) if xmp_bytes else None), **portable.xmp},
        camera_make=str(make) if make is not None else None,
        camera_model=str(model) if model is not None else None,
        unique_camera_model=str(model) if model is not None else None,
        lens_model=str(lens) if lens is not None else None,
        capture_time=str(captured) if captured is not None else None,
        orientation=1,
        source_orientation=orientation,
    )


def capture_time_from_file(path: Path) -> str | None:
    """Read capture time without decoding full-resolution image pixels."""

    path = Path(path)
    portable = read_metadata(path)
    for key in (
        "Exif.Photo.DateTimeOriginal",
        "Exif.Image.DateTime",
        "Exif.Photo.DateTimeDigitized",
    ):
        value = portable.exif.get(key)
        if value not in (None, ""):
            return str(value)

    if path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}:
        try:
            with Image.open(path) as image:
                exif = image.getexif()
                value = exif.get(36867) or exif.get(306) or exif.get(36868)
                if value not in (None, ""):
                    return str(value)
        except Exception:  # noqa: BLE001 - absence/corruption falls through to ExifRead
            pass

    try:
        import exifread

        with path.open("rb") as stream:
            tags = exifread.process_file(stream, details=False, extract_thumbnail=False, strict=False)
        for key in ("EXIF DateTimeOriginal", "Image DateTime", "EXIF DateTimeDigitized"):
            value = tags.get(key)
            if value not in (None, ""):
                return str(value)
    except Exception:  # noqa: BLE001 - grouping has a deterministic filename fallback
        pass
    return None


def normalize_orientation(arr: Any, orientation: int) -> Any:
    """Return pixels in top-left orientation for TIFF/RAW array loaders."""

    import numpy as np

    transforms = {
        1: lambda x: x,
        2: lambda x: np.flip(x, axis=1),
        3: lambda x: np.flip(np.flip(x, axis=0), axis=1),
        4: lambda x: np.flip(x, axis=0),
        5: lambda x: np.swapaxes(x, 0, 1),
        6: lambda x: np.rot90(x, k=3),
        7: lambda x: np.flip(np.swapaxes(x, 0, 1), axis=(0, 1)),
        8: lambda x: np.rot90(x, k=1),
    }
    return np.ascontiguousarray(transforms.get(orientation, transforms[1])(arr))
