"""Job orchestration: validate -> stack (direct/tiled) -> fallback chain (SPEC §4, §12).

M1 scope: alignment is M2; frames are assumed pre-aligned here.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from focusstack.align import AlignParams, align_stack
from focusstack.backend import Device, empty_cache, free_memory, get_device
from focusstack.errors import ValidationError
from focusstack.io import validate_stack
from focusstack.select import SelectParams, select_frames
from focusstack.stack import DirFrameSource, StackResult, get_algorithm
from focusstack.stack.base import FrameSource
from focusstack.stack.sources import _SubsetSource
from focusstack.tiles import estimate_stack_bytes, stack_tiled

log = logging.getLogger(__name__)


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
    report = validate_stack(paths)
    if not report.ok:
        bad = [f"{s.path.name}: {s.status}" for s in report.files if s.status != "ok"]
        raise ValidationError("stack validation failed:\n  " + "\n  ".join(bad))
    if len(paths) < 2:
        raise ValidationError("a stack needs at least 2 frames")

    device = get_device(device_pref)

    reference_index = (align.reference if align is not None and align.reference is not None else len(paths) // 2)
    disk_source = DirFrameSource(list(paths), reference_index=reference_index)
    source: FrameSource
    masks: FrameSource | None = None
    alignment_report = None
    if align is not None:
        cdir = Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp(prefix="fs-align-"))
        alignment_report = align_stack(disk_source, cache_dir=cdir, device=device,
                                       params=align, progress=progress, cancel=cancel)
        source = alignment_report.cache.frame_source()
        masks = alignment_report.cache.mask_source()
        height, width = source.read(0).shape[:2]
    else:
        source = disk_source
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
        height, width = source.read(0).shape[:2]

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
                for path, digest in zip(paths, disk_source.source_hashes, strict=True)
            ],
            "alignment": alignment_payload,
            "selected": selected_indices,
            "excluded": sorted(set(range(len(paths))) - set(selected_indices)),
        }
        return result

    def _is_oom(e: Exception) -> bool:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
        return device.kind == "mps" and isinstance(e, RuntimeError) and "memory" in str(e).lower()

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
        empty_cache(device)
        return _run(get_device("cpu"), True)
