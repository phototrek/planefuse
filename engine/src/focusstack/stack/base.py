"""Algorithm registry with typed parameters (SPEC §7).

The UI and CLI build their parameter forms from ParamSpec metadata, so
adding an algorithm requires no UI changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

ProgressFn = Callable[[str, float], None]  # (message, fraction 0..1)
CancelFn = Callable[[], bool]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    type: str  # "int" | "float" | "choice" | "bool"
    default: Any
    min: Any = None
    max: Any = None
    choices: tuple[str, ...] | None = None
    tooltip: str = ""


@dataclass
class StackResult:
    image: np.ndarray  # float32 (H, W, 3), UNclamped (SPEC §7.1)
    aux: dict[str, np.ndarray] = field(default_factory=dict)


class FrameSource(Protocol):
    def __len__(self) -> int: ...

    def read(self, idx: int, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """float32 (H, W, 3); region=(y0, x0, y1, x1) returns that crop."""
        ...


class StackAlgorithm(Protocol):
    name: str

    @staticmethod
    def params() -> list[ParamSpec]: ...

    def run(
        self,
        source: FrameSource,
        device: Any,
        params: dict[str, Any],
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> StackResult: ...


REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def get_algorithm(name: str) -> StackAlgorithm:
    if name not in REGISTRY:
        raise KeyError(f"unknown stacking algorithm {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]()
