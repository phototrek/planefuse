"""Job orchestration: validate -> stack (direct/tiled) -> fallback chain (SPEC §4, §12).

M1 scope: alignment is M2; frames are assumed pre-aligned here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from focusstack.backend import Device, empty_cache, free_memory, get_device
from focusstack.errors import ValidationError
from focusstack.io import validate_stack
from focusstack.stack import DirFrameSource, StackResult, get_algorithm
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
    source = DirFrameSource(list(paths))

    use_tiled = tile_mode == "always"
    if tile_mode == "auto" and device.kind != "cpu":
        needed = estimate_stack_bytes(report.height, report.width)
        budget = int(free_memory(device) * 0.8)
        use_tiled = needed > budget
        if use_tiled:
            log.warning("estimated %d MB > budget %d MB; using tiled mode", needed >> 20, budget >> 20)

    def _run(dev: Device, tiled: bool) -> StackResult:
        if tiled:
            img = stack_tiled(method, source, dev, params, tile=tile, progress=progress, cancel=cancel)
            return StackResult(image=img)
        return get_algorithm(method).run(source, dev, params, progress=progress, cancel=cancel)

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
