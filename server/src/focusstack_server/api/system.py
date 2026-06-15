"""System + algorithm-registry endpoints (SPEC §9)."""

from __future__ import annotations

from typing import Any, cast

import torch
from fastapi import APIRouter

from focusstack import __version__ as engine_version
from focusstack.backend import free_memory, get_device
from focusstack.stack import REGISTRY

router = APIRouter(prefix="/api")


@router.get("/system")
def system() -> dict:
    dev = get_device("auto")
    return {
        "device": dev.kind,
        "device_name": dev.name,
        "free_memory": free_memory(dev),
        "version": engine_version,
        "torch": torch.__version__,
    }


@router.get("/algorithms")
def algorithms() -> list[dict]:
    out = []
    for name, cls in sorted(REGISTRY.items()):
        params = [
            {"name": p.name, "label": p.label, "type": p.type, "default": p.default,
             "min": p.min, "max": p.max, "choices": list(p.choices) if p.choices else None,
             "tooltip": p.tooltip}
            for p in cast(Any, cls).params()
        ]
        out.append({"name": name, "params": params})
    return out
