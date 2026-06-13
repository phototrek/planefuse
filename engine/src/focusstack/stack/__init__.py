from focusstack.stack.base import (
    REGISTRY,
    FrameSource,
    ParamSpec,
    StackResult,
    get_algorithm,
    register,
)
from focusstack.stack.sources import ArrayFrameSource, DirFrameSource
import focusstack.stack.pmax  # noqa: E402,F401  (registers "pmax")
import focusstack.stack.dmap  # noqa: E402,F401  (registers "dmap")
import focusstack.stack.weighted  # noqa: E402,F401  (registers "weighted")
import focusstack.stack.slab  # noqa: E402,F401  (registers "slab")

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
