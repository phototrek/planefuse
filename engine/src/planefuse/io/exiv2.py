"""Narrow, failure-tolerant pyexiv2 adapter.

pyexiv2 is native and not thread-safe, so access is serialized here instead of
leaking that constraint through the loader and writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
import json


@dataclass(frozen=True)
class Exiv2Metadata:
    exif: dict[str, Any] = field(default_factory=dict)
    xmp: dict[str, Any] = field(default_factory=dict)
    raw_xmp: bytes | None = None
    icc: bytes | None = None


_LOCK = Lock()
_PLANEFUSE_NS = "https://planefuse.app/ns/1.0/"


def read_metadata(path: Path) -> Exiv2Metadata:
    try:
        import pyexiv2
    except (ImportError, OSError):
        return Exiv2Metadata()

    try:
        with _LOCK, pyexiv2.Image(str(path)) as image:
            raw_xmp = image.read_raw_xmp()
            return Exiv2Metadata(
                exif=dict(image.read_exif()),
                xmp=dict(image.read_xmp()),
                raw_xmp=raw_xmp.encode("utf-8") if raw_xmp else None,
                icc=image.read_icc() or None,
            )
    except Exception:  # noqa: BLE001 - pixel loading must survive unsupported metadata
        return Exiv2Metadata()


def write_metadata(path: Path, metadata: Any, provenance: dict[str, Any] | None = None) -> None:
    import pyexiv2

    safe = metadata.without_single_exposure_fields()
    with _LOCK:
        pyexiv2.registerNs(_PLANEFUSE_NS, "PlaneFuse")
        with pyexiv2.Image(str(path)) as image:
            if safe.exif:
                image.modify_exif(dict(safe.exif))
            if metadata.xmp_bytes:
                image.modify_raw_xmp(metadata.xmp_bytes.decode("utf-8", errors="replace"))
            xmp = dict(safe.xmp)
            for key in metadata.xmp:
                if key not in safe.xmp:
                    xmp[key] = None
            if provenance is not None:
                xmp["Xmp.PlaneFuse.Provenance"] = json.dumps(
                    provenance, sort_keys=True, separators=(",", ":")
                )
            if xmp:
                image.modify_xmp(xmp)
            if safe.icc:
                image.modify_icc(safe.icc)
