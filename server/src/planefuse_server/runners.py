"""Engine-calling job runners (SPEC §9). Each returns a closure (progress, cancel)->result."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tifffile

from planefuse.align import AlignParams
from planefuse.align import align_stack
from planefuse.backend import get_device
from planefuse.errors import DngExportError, ValidationError
from planefuse.io import (
    ProcessingDomain,
    load_image,
    metadata_from_dict,
    metadata_to_dict,
    save_float_tiff,
    save_image,
    save_linear_dng,
    validate_stack,
)
from planefuse.pipeline import stack_frames
from planefuse.select import SelectParams
from planefuse.select import select_frames
from planefuse.stack import DirFrameSource
from planefuse.stack.base import FrameSource
from planefuse_server.projects import Project, ProjectStore


def make_stack_runner(store: ProjectStore, proj: Project, params: dict[str, Any]) -> Callable:
    # An explicit frame list (e.g. a batch group) overrides the project's frames.
    paths = [Path(p) for p in (params.get("frames") or proj.frames)]
    method = params.get("method", "pmax")
    device = params.get("device", "auto")
    align = AlignParams(**params["align"]) if params.get("align") else None
    select = SelectParams(**params["select"]) if params.get("select") else None
    algo_params = params.get("algo_params", {})

    def run(progress: Callable[[str, float], None], cancel: Callable[[], bool]) -> dict:
        # The workspace UI can contain rejected files while the valid remainder
        # is still stackable. Skip those here instead of failing the whole job.
        report = validate_stack(paths)
        frames = paths
        if not report.ok:
            frames = [Path(status.path) for status in report.files if status.status == "ok"]
            skipped = [status for status in report.files if status.status != "ok"]
            shown = ", ".join(status.path.name for status in skipped[:8])
            more = "" if len(skipped) <= 8 else f" (+{len(skipped) - 8} more)"
            progress(f"skipped {len(skipped)} mismatched frame(s): {shown}{more}", 0.0)
            if len(frames) < 2:
                detail = "\n  ".join(f"{status.path.name}: {status.status}" for status in skipped)
                raise ValidationError("no usable frames after skipping mismatched:\n  " + detail)

        result = stack_frames(frames, method=method, params=algo_params, device_pref=device,
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
            "frames": len(frames),
            "domain": result.domain.value,
            "storage": storage,
            "reference_path": str(frames[len(frames) // 2]),
            "metadata": metadata_to_dict(result.metadata) if result.metadata is not None else None,
            "decoder": dict(result.metadata.decoder) if result.metadata is not None else {},
            "provenance": result.provenance,
        }
        if "depth" in result.aux:
            depth_id = uuid.uuid4().hex[:12]
            dpath = proj.cache / f"{depth_id}.tif"
            save_image(result.aux["depth"], dpath, bit_depth=16)
            latest.images[depth_id] = {
                "kind": "depth",
                "path": str(dpath),
                "parent": image_id,
            }
        store.save(latest)
        return {"image_id": image_id}

    return run


def make_select_runner(store: ProjectStore, proj: Project, params: dict[str, Any]) -> Callable:
    paths = [Path(path) for path in (params.get("frames") or proj.frames)]
    device = get_device(str(params.get("device", "auto")))
    align_params = AlignParams(**params["align"]) if params.get("align") else None
    select_params = SelectParams(**(params.get("select") or {}))

    def run(progress: Callable[[str, float], None], cancel: Callable[[], bool]) -> dict:
        source: FrameSource = DirFrameSource(paths)
        masks = None
        if align_params is not None:
            cache = proj.cache / f"selection-{uuid.uuid4().hex[:8]}"
            report = align_stack(
                source,
                cache,
                device,
                align_params,
                progress=progress,
                cancel=cancel,
            )
            source = report.cache.frame_source()
            masks = report.cache.mask_source()
        proposal = select_frames(
            source,
            device,
            select_params,
            masks=masks,
            progress=progress,
        )
        payload = {
            "kept": proposal.kept,
            "redundant": proposal.redundant,
            "warning": proposal.warning,
            "coverage": {
                "reliable_cells": int(proposal.reliable.sum()),
                "total_cells": int(proposal.reliable.size),
            },
            "params": {
                "focus_tolerance": select_params.focus_tolerance,
                "kurtosis_threshold": select_params.kurtosis_threshold,
                "grid_rows": select_params.grid_rows,
                "grid_cols": select_params.grid_cols,
            },
        }
        latest = store.get(proj.id) or proj
        latest.ui_state["selectionProposal"] = payload
        store.save(latest)
        return payload

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
    compression = str(params.get("compression", "zlib"))
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
                compression=compression,
                jpeg_quality=jpeg_quality,
                metadata=metadata,
                provenance=provenance,
            )
            payload = {"dest": str(dest)}

        float_tiff_value = params.get("float_tiff_dest")
        if float_tiff_value:
            float_tiff_dest = Path(str(float_tiff_value))
            float_tiff_dest.parent.mkdir(parents=True, exist_ok=True)
            save_float_tiff(
                img,
                float_tiff_dest,
                metadata=metadata,
                provenance=provenance,
            )
            payload["float_tiff_dest"] = str(float_tiff_dest)

        depth_value = params.get("depth_dest")
        if depth_value:
            depth_info = next(
                (
                    candidate
                    for candidate in proj.images.values()
                    if candidate.get("kind") == "depth"
                    and candidate.get("parent") == image_id
                    and candidate.get("path")
                ),
                None,
            )
            if depth_info is None:
                raise ValueError("this result has no depth-map companion")
            depth_dest = Path(str(depth_value))
            depth_dest.parent.mkdir(parents=True, exist_ok=True)
            depth = load_image(Path(str(depth_info["path"]))).pixels[..., 0]
            save_image(depth, depth_dest, bit_depth=16, compression=compression)
            payload["depth_dest"] = str(depth_dest)
        progress("done", 1.0)
        return payload

    return run
