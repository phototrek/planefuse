from planefuse.stack.base import (
    REGISTRY,
    FrameSource,
    ParamSpec,
    StackResult,
    get_algorithm,
    register,
)
from planefuse.stack.sources import ArrayFrameSource, DirFrameSource
import planefuse.stack.pmax  # noqa: E402,F401  (registers "pmax")
import planefuse.stack.dmap  # noqa: E402,F401  (registers "dmap")
import planefuse.stack.weighted  # noqa: E402,F401  (registers "weighted")
import planefuse.stack.slab  # noqa: E402,F401  (registers "slab")

__all__ = [
    "REGISTRY",
    "ArrayFrameSource",
    "DirFrameSource",
    "FrameSource",
    "ParamSpec",
    "StackResult",
    "get_algorithm",
    "register",
]
