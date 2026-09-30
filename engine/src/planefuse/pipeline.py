"""Job orchestration: validate -> stack (direct/tiled) -> fallback chain (SPEC §4, §12).

M1 scope: alignment is M2; frames are assumed pre-aligned here.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planefuse.align import AlignParams, align_stack
from planefuse.backend import Device, empty_cache, free_memory, get_device
from planefuse.errors import ValidationError
from planefuse.io import ValidationReport, validate_stack
from planefuse.io.loader import probe_frame_shape
from planefuse.select import SelectParams, select_frames
from planefuse.stack import DirFrameSource, MemoryFrameSource, StackResult, get_algorithm
from planefuse.stack.base import FrameSource
from planefuse.stack.sources import (
    TensorFrameSource,
    _SubsetSource,
    decode_frames,
    default_decode_workers,
    read_tensor,
)
from planefuse.tiles import estimate_stack_bytes, stack_tiled

log = logging.getLogger(__name__)

_HOST_MEMORY_FRACTION = 0.6
_DEVICE_MEMORY_FRACTION = 0.6
_WORKING_FRAMES = 2
# Rough peak of one in-flight RAW decode beyond the frame it returns (LibRaw's
# buffers, the uint16 image, the float conversion temporary), in frames.
_DECODE_TRANSIENT_FRAMES = 2.5


def host_memory_budget() -> int:
    """Bytes the in-memory frame path may hold: PLANEFUSE_MEMORY_BUDGET_MB when
    set, else 60% of physical RAM."""
    configured = os.getenv("PLANEFUSE_MEMORY_BUDGET_MB", "").strip()
    if configured:
        try:
            return max(0, int(float(configured) * (1 << 20)))
        except ValueError:
            log.warning("ignoring PLANEFUSE_MEMORY_BUDGET_MB=%r: not a number", configured)
    try:
        physical = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        import psutil

        physical = int(psutil.virtual_memory().total)
    return int(physical * _HOST_MEMORY_FRACTION)


def _aligned_store(device: Device, n: int, height: int, width: int, host_held: int) -> str:
    """Where align_stack keeps the aligned frames: "device" when they fit 60%
    of the device's free memory, else "host" when they fit the host budget,
    else "cache" (the on-disk AlignedCache).

    `host_held` is what the job already holds in host memory while warping
    (the decoded input frames). On MPS the device memory is host memory, so
    the device store must fit the host budget as well.
    """
    frame_bytes = height * width * 3 * 4
    aligned = n * (frame_bytes + height * width)  # frames + bool masks
    host_fits = aligned + host_held + _WORKING_FRAMES * frame_bytes <= host_memory_budget()
    if device.kind == "cpu":
        return "device" if host_fits else "cache"
    device_fits = aligned <= int(free_memory(device) * _DEVICE_MEMORY_FRACTION)
    if device_fits and (device.kind != "mps" or host_fits):
        return "device"
    return "host" if host_fits else "cache"


def _frame_hw(source: FrameSource) -> tuple[int, int]:
    held = read_tensor(source, 0)
    if held is not None:
        return int(held.shape[-2]), int(held.shape[-1])
    height, width = source.read(0).shape[:2]
    return height, width


def _validate_and_preload(
    paths: list[Path], reference_index: int
) -> tuple[ValidationReport, MemoryFrameSource | None]:
    """Validate the stack; when every frame fits the host budget, decode each
    frame exactly once (in parallel) and validate from those decodes.

    Returns (report, memory source or None). None means the streaming path:
    validation ran exactly as before and the caller builds a DirFrameSource.
    """
    n = len(paths)
    if n < 2:
        return validate_stack(paths), None
    shapes = {probe_frame_shape(p) for p in paths}
    if len(shapes) != 1 or None in shapes:
        # Unreadable header or mixed sizes: validation reports it, from disk.
        return validate_stack(paths), None
    (shape,) = shapes
    assert shape is not None
    frame_bytes = shape[0] * shape[1] * 3 * 4
    resident = (n + _WORKING_FRAMES) * frame_bytes
    transient = int(frame_bytes * _DECODE_TRANSIENT_FRAMES)
    budget = host_memory_budget()
    if resident + transient > budget:
        log.info("frames need %d MB (+%d MB per decode) > host budget %d MB; streaming from disk",
                 resident >> 20, transient >> 20, budget >> 20)
        return validate_stack(paths), None
    spare = budget - resident
    workers = min(default_decode_workers(n), max(1, spare // transient))
    log.info("decoding %d frames in memory (%d MB of %d MB budget, %d workers)",
             n, resident >> 20, budget >> 20, workers)
    loaded = decode_frames(paths, workers=workers)
    report = validate_stack(paths, loaded)
    if not report.ok:
        return report, None
    frames = [item for item in loaded if not isinstance(item, BaseException)]
    if len(frames) != n:
        # A rendered file whose header validates but whose pixels do not
        # decode: hand it to the streaming path, which raises on reading it.
        return validate_stack(paths), None
    return report, MemoryFrameSource(paths, frames, reference_index=reference_index, workers=workers)


def stack_frames(
    paths: list[Path],
    *,
    method: str = "pmax",
    params: dict[str, Any] | None = None,
    device_pref: str = "auto",
    tile_mode: str = "auto",  # auto | always | never
    tile: int = 2048,
    align: AlignParams | None = None,
    cache_dir: Path | None = None,
    select: SelectParams | None = None,
    progress=None,
    cancel=None,
) -> StackResult:
    params = dict(params or {})
    reference_index = (align.reference if align is not None and align.reference is not None else len(paths) // 2)
    report, memory_source = _validate_and_preload(list(paths), reference_index)
    if not report.ok:
        bad = [f"{s.path.name}: {s.status}" for s in report.files if s.status != "ok"]
        raise ValidationError("stack validation failed:\n  " + "\n  ".join(bad))
    if len(paths) < 2:
        raise ValidationError("a stack needs at least 2 frames")

    device = get_device(device_pref)

    def _is_oom(e: Exception) -> bool:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
        return device.kind == "mps" and isinstance(e, RuntimeError) and "memory" in str(e).lower()

    input_source: DirFrameSource | MemoryFrameSource = (
        memory_source if memory_source is not None
        else DirFrameSource(list(paths), reference_index=reference_index)
    )
    source: FrameSource
    masks: FrameSource | None = None
    alignment_report = None
    held: list[TensorFrameSource] = []
    if align is not None:
        cdir = Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp(prefix="fs-align-"))
        # align_stack frees each decoded input frame right after warping it,
        # so while warping about one input frame is held beside the aligned ones.
        host_held = (report.height * report.width * 3 * 4
                     if isinstance(input_source, MemoryFrameSource) else 0)
        store = _aligned_store(device, len(paths), report.height, report.width, host_held)
        log.info("aligned frames: %s", store)
        try:
            alignment_report = align_stack(input_source, cache_dir=cdir, device=device,
                                           params=align, progress=progress, cancel=cancel,
                                           store=store)
        except Exception as e:
            if store == "cache" or not _is_oom(e):
                raise
            log.warning("device OOM holding aligned frames; retrying alignment with the disk cache")
            empty_cache(device)
            if isinstance(input_source, MemoryFrameSource):
                # Frames already warped were freed: stream the inputs from disk.
                input_source.release()
                input_source = DirFrameSource(list(paths), reference_index=reference_index)
            alignment_report = align_stack(input_source, cache_dir=cdir, device=device,
                                           params=align, progress=progress, cancel=cancel,
                                           store="cache")
        if isinstance(input_source, MemoryFrameSource):
            input_source.release()  # aligned frames replace them from here on
        source = alignment_report.frame_source()
        masks = alignment_report.mask_source()
        held = [s for s in (alignment_report.frames, alignment_report.masks) if s is not None]
        height, width = _frame_hw(source)
    else:
        source = input_source
        height, width = report.height, report.width

    selected_indices = list(range(len(paths)))
    if select is not None:
        sel = select_frames(source, device, select, masks=masks, progress=progress)
        if progress is not None:
            msg = f"selected {len(sel.kept)}/{len(source)} frames"
            if sel.warning:
                msg += f" ({sel.warning})"
            progress(msg, 0.0)
        source = _SubsetSource(source, sel.kept)
        masks = _SubsetSource(masks, sel.kept) if masks is not None else None
        selected_indices = list(sel.kept)
        height, width = _frame_hw(source)

    use_tiled = tile_mode == "always"
    if tile_mode == "auto" and device.kind != "cpu":
        needed = estimate_stack_bytes(height, width)
        budget = int(free_memory(device) * 0.8)
        use_tiled = needed > budget
        if use_tiled:
            log.warning("estimated %d MB > budget %d MB; using tiled mode", needed >> 20, budget >> 20)

    def _run(dev: Device, tiled: bool) -> StackResult:
        if tiled:
            img = stack_tiled(method, source, dev, params, tile=tile,
                              progress=progress, cancel=cancel, masks=masks)
            result = StackResult(image=img, domain=source.domain, metadata=source.metadata)
        else:
            algo = get_algorithm(method)
            if masks is not None:
                result = algo.run(source, dev, params, progress=progress, cancel=cancel, masks=masks)  # type: ignore[call-arg]
            else:
                result = algo.run(source, dev, params, progress=progress, cancel=cancel)
            result.domain = source.domain
            result.metadata = source.metadata

        if alignment_report is None:
            alignment_payload: dict[str, Any] = {
                "model": "none",
                "reference": reference_index,
                "transforms": [np.eye(3).tolist() for _ in paths],
                "quality": {},
                "flagged": [],
                "recovered": [],
                "excluded": [],
            }
        else:
            alignment_payload = {
                "model": align.model if align is not None else "none",
                "reference": alignment_report.reference,
                "transforms": [matrix.tolist() for matrix in alignment_report.matrices],
                "quality": {
                    str(key): value for key, value in alignment_report.correlations.items()
                },
                "flagged": sorted(alignment_report.flagged),
                "recovered": sorted(alignment_report.recovered),
                "excluded": sorted(alignment_report.dropped),
            }
        result.provenance = {
            "method": method,
            "parameters": params,
            "device": dev.kind,
            "tiled": tiled,
            "sources": [
                {"name": path.name, "sha256": digest}
                for path, digest in zip(paths, input_source.source_hashes, strict=True)
            ],
            "alignment": alignment_payload,
            "selected": selected_indices,
            "excluded": sorted(set(range(len(paths))) - set(selected_indices)),
        }
        return result

    # SPEC §4 fallback chain: direct -> tiled -> CPU. A job never fails on OOM.
    try:
        return _run(device, use_tiled)
    except Exception as e:
        if not _is_oom(e):
            raise
        log.warning("device OOM; retrying tiled")
        empty_cache(device)
    try:
        return _run(device, True)
    except Exception as e:
        if not _is_oom(e):
            raise
        log.warning("device OOM even tiled; falling back to CPU")
        for tensors in held:
            tensors.offload()
        empty_cache(device)
        return _run(get_device("cpu"), True)
