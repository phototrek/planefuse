"""Engine-calling job runners (SPEC §9). Each returns a closure (progress, cancel)->result."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tifffile

from focusstack.align import AlignParams
from focusstack.errors import DngExportError
from focusstack.io import (
    ProcessingDomain,
    load_image,
    metadata_from_dict,
    metadata_to_dict,
    save_float_tiff,
    save_image,
    save_linear_dng,
)
from focusstack.pipeline import stack_frames
from focusstack.select import SelectParams
from focusstack_server.projects import Project, ProjectStore


def make_stack_runner(store: ProjectStore, proj: Project, params: dict[str, Any]) -> Callable:
    # An explicit frame list (e.g. a batch group) overrides the project's frames.
    paths = [Path(p) for p in (params.get("frames") or proj.frames)]
    method = params.get("method", "pmax")
    device = params.get("device", "auto")
    align = AlignParams(**params["align"]) if params.get("align") else None
    select = SelectParams(**params["select"]) if params.get("select") else None
    algo_params = params.get("algo_params", {})

    def run(progress: Callable[[str, float], None], cancel: Callable[[], bool]) -> dict:
        result = stack_frames(paths, method=method, params=algo_params, device_pref=device,
                              align=align, select=select, cache_dir=proj.cache,
                              progress=progress, cancel=cancel)
        image_id = uuid.uuid4().hex[:12]
        out_path = proj.cache / f"{image_id}.tif"
        if result.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
            save_float_tiff(
                result.image,
                out_path,
                metadata=result.metadata,
                provenance=result.provenance,
            )
            storage = "float32_tiff"
        else:
            save_image(
                result.image,
                out_path,
                bit_depth=16,
                metadata=result.metadata,
                provenance=result.provenance,
            )
            storage = "rendered_16bit"
        # Reload the latest project before recording the result: another job may
        # have added images since this runner captured `proj` at enqueue time,
        # and store.save writes the whole project.json (would clobber them).
        latest = store.get(proj.id) or proj
        latest.images[image_id] = {
            "kind": "result",
            "path": str(out_path),
            "method": method,
            "frames": len(paths),
            "domain": result.domain.value,
            "storage": storage,
            "reference_path": str(paths[len(paths) // 2]),
            "metadata": metadata_to_dict(result.metadata) if result.metadata is not None else None,
            "decoder": dict(result.metadata.decoder) if result.metadata is not None else {},
            "provenance": result.provenance,
        }
        if "depth" in result.aux:
            depth_id = uuid.uuid4().hex[:12]
            dpath = proj.cache / f"{depth_id}.tif"
            save_image(result.aux["depth"], dpath, bit_depth=16)
            latest.images[depth_id] = {"kind": "depth", "path": str(dpath)}
        store.save(latest)
        return {"image_id": image_id}

    return run


def make_export_runner(
    store: ProjectStore,
    proj: Project,
    params: dict[str, Any],
) -> Callable:
    image_id = params["image_id"]
    dest = Path(params["dest"])
    bit_depth = int(params.get("bit_depth", 16))
    jpeg_quality = int(params.get("jpeg_quality", 95))
    info = proj.images.get(image_id)

    def run(progress: Callable[[str, float], None], cancel: Callable[[], bool]) -> dict:
        if info is None:
            raise ValueError(f"unknown image_id {image_id}")
        if "path" not in info:
            raise ValueError(f"image {image_id} is not exportable (no path)")
        progress("export", 0.1)
        if info.get("storage") == "float32_tiff":
            img = tifffile.imread(Path(info["path"]))
        else:
            img = load_image(Path(info["path"])).pixels
        dest.parent.mkdir(parents=True, exist_ok=True)
        domain = ProcessingDomain(info.get("domain", ProcessingDomain.RENDERED_RGB.value))
        metadata_values = info.get("metadata")
        metadata = metadata_from_dict(metadata_values) if metadata_values else None
        provenance = info.get("provenance") or {}
        requested_format = str(params.get("format", dest.suffix.lstrip("."))).lower()
        if requested_format == "dng" or dest.suffix.lower() == ".dng":
            if domain is not ProcessingDomain.SCENE_LINEAR_CAMERA_RGB:
                raise DngExportError(
                    "Linear DNG export requires a scene_linear_camera_rgb result"
                )
            if metadata is None:
                raise DngExportError("Linear DNG export requires reference camera metadata")
            report = save_linear_dng(img, dest, metadata, provenance)
            validation = {
                "path": str(dest),
                "rawpy_validated": report.rawpy_validated,
                "max_code_error": report.max_code_error,
                "encoding_offset": report.encoding_offset,
                "encoding_span": report.encoding_span,
            }
            latest = store.get(proj.id) or proj
            if image_id in latest.images:
                latest.images[image_id]["dng_validation"] = validation
                store.save(latest)
            payload: dict[str, Any] = {"dest": str(dest), "dng_validation": validation}
        elif bit_depth == 32:
            save_float_tiff(
                img,
                dest,
                metadata=metadata,
                provenance=provenance,
            )
            payload = {"dest": str(dest), "float_tiff": True}
        else:
            save_image(
                img,
                dest,
                bit_depth=bit_depth,
                jpeg_quality=jpeg_quality,
                metadata=metadata,
                provenance=provenance,
            )
            payload = {"dest": str(dest)}
        progress("done", 1.0)
        return payload

    return run
