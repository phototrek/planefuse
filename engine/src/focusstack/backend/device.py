"""Runtime device selection (SPEC §4). One implementation, three devices."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from focusstack.errors import BackendError

log = logging.getLogger(__name__)

_VALID = ("auto", "cuda", "mps", "cpu")


@dataclass(frozen=True)
class Device:
    kind: str  # "cuda" | "mps" | "cpu"
    torch_device: torch.device
    name: str


def get_device(prefer: str = "auto") -> Device:
    if prefer not in _VALID:
        raise ValueError(f"unknown device preference {prefer!r}; expected one of {_VALID}")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        d = Device("cuda", torch.device("cuda:0"), torch.cuda.get_device_name(0))
        log.info("using device: %s (%s)", d.kind, d.name)
        return d
    if prefer == "cuda":
        raise BackendError("CUDA requested but no CUDA device is available")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        d = Device("mps", torch.device("mps"), "Apple silicon (MPS)")
        log.info("using device: %s", d.kind)
        return d
    if prefer == "mps":
        raise BackendError("MPS requested but not available (requires Apple silicon macOS)")
    d = Device("cpu", torch.device("cpu"), "CPU")
    log.info("using device: cpu")
    return d


def free_memory(device: Device) -> int:
    """Bytes of memory available on the device (SPEC §4 memory discipline)."""
    if device.kind == "cuda":
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    if device.kind == "mps":
        return max(0, int(torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()))
    import psutil

    return int(psutil.virtual_memory().available)


def empty_cache(device: Device) -> None:
    """Release cached allocations after OOM, before fallback (SPEC §4)."""
    if device.kind == "cuda":
        torch.cuda.empty_cache()
    elif device.kind == "mps":
        torch.mps.empty_cache()
