"""System + algorithm-registry endpoints (SPEC §9)."""

from __future__ import annotations

from typing import Any, cast

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from planefuse import __version__ as engine_version
from planefuse.backend import free_memory, get_device
from planefuse.stack import REGISTRY
from planefuse.tiles import estimate_stack_bytes

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


class EstimateBody(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: int = Field(gt=1)
    method: str = "pmax"
    align: bool = True


@router.post("/estimate")
def estimate(body: EstimateBody) -> dict:
    # Conservative planning estimate, intentionally not a hardware benchmark.
    memory = estimate_stack_bytes(body.height, body.width)
    method_factor = {"pmax": 1.0, "weighted": 0.7, "dmap": 1.35, "slab": 1.6}.get(
        body.method, 1.0
    )
    device = get_device("auto")
    throughput = {"cuda": 80.0, "mps": 28.0, "cpu": 4.0}[device.kind]
    input_megapixels = body.width * body.height * body.frames / 1_000_000
    seconds = input_megapixels / throughput * method_factor
    if body.align:
        seconds += body.frames * 0.12
    return {
        "approximate": True,
        "memory_bytes": memory,
        "seconds": max(0.1, round(seconds, 1)),
        "basis": f"planning model for {device.kind}; actual speed depends on image content",
    }
