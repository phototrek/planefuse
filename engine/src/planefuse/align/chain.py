"""Chain consecutive pair transforms to the reference frame (SPEC §6 step 2, 4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from planefuse.align.transforms import compose, invert


@dataclass
class ChainResult:
    matrices: list[np.ndarray]      # per-frame output->input to the reference canvas
    reference: int
    correlations: dict[int, float]  # pair index i -> corr for pair (i, i+1)
    flagged: set[int] = field(default_factory=set)   # frame indices below threshold
    dropped: set[int] = field(default_factory=set)


def chain_to_reference(n: int, pair_transforms: dict[int, np.ndarray],
                       correlations: dict[int, float], ref: int,
                       threshold: float, drop: bool = False) -> ChainResult:
    """pair_transforms[i] maps frame (i+1) -> frame i (consecutive, output->input).

    Composing from frame k toward the reference yields the transform mapping the
    reference canvas to frame k's pixels.
    """
    # warp(img, M)[x] = img(M x), and warp(warp(img, A), B) = warp(img, A @ B)
    # (A applied first). To align frame k to the reference we apply the pair
    # transform closest to k first, so each newly added link left-multiplies the
    # accumulator. Order matters: scale/rotation transforms do not commute.
    matrices: list[np.ndarray] = [np.eye(3) for _ in range(n)]
    acc = np.eye(3)
    for k in range(ref + 1, n):
        acc = compose(pair_transforms[k - 1], acc)  # ref->...->k
        matrices[k] = acc.copy()
    acc = np.eye(3)
    for k in range(ref - 1, -1, -1):
        acc = compose(invert(pair_transforms[k]), acc)  # ref->...->k
        matrices[k] = acc.copy()

    flagged = {i for i, c in correlations.items() if c < threshold}
    flagged_frames = {i + 1 for i in flagged}
    result = ChainResult(matrices=matrices, reference=ref,
                         correlations=correlations, flagged=flagged_frames)
    if drop:
        result.dropped = {f for f in flagged_frames if f != ref}
    return result
