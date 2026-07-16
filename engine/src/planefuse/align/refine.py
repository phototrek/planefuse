"""ECC refinement and brightness normalization (SPEC §6 steps c, d).

ECC runs on CPU (OpenCV) over downscaled luminance proxies — cheap and the spec
mandates it. The refined affine is projected to a similarity (the output model).
"""

from __future__ import annotations

import cv2
import numpy as np

from planefuse.align.transforms import project_to_similarity

# cv2.findTransformECC parallelizes its per-iteration reductions across
# whatever thread count OpenCV sees at import time. That changes the
# floating-point summation order of the Gauss-Newton solve, which can shift a
# hard case's Hessian just enough to converge to a different local optimum on
# an 8-core laptop than on a 4-core CI runner — a real, reproducible source of
# machine-dependent alignment accuracy, not routine numerical noise. Pinning
# to one thread makes ECC deterministic and gives every machine the same
# (best-observed) convergence; proxy images are small enough that the
# single-threaded cost is negligible next to full-resolution warping/fusion.
cv2.setNumThreads(1)


def _ecc_at_level(a: np.ndarray, b: np.ndarray, warp_init: np.ndarray,
                  iters: int, eps: float, motion: int) -> tuple[np.ndarray, float]:
    """One ECC solve at one pyramid level."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    warp = warp_init.astype(np.float32)
    try:
        cc, out = cv2.findTransformECC(
            a, b, warp, motion, criteria, None, 5)  # type: ignore[call-overload]
    except TypeError:
        cc, out = cv2.findTransformECC(a, b, warp, motion, criteria)
    return np.asarray(out, dtype=np.float32), float(cc)


def _as_homogeneous(warp: np.ndarray) -> np.ndarray:
    if warp.shape == (3, 3):
        return warp.astype(np.float64)
    out = np.eye(3, dtype=np.float64)
    out[:2] = warp
    return out


def _scale_for_level(warp: np.ndarray, factor: float) -> np.ndarray:
    scale = np.diag([factor, factor, 1.0])
    return np.linalg.inv(scale) @ _as_homogeneous(warp) @ scale


def refine_ecc(a: np.ndarray, b: np.ndarray, init: np.ndarray,
               cx: float, cy: float, levels: int = 3,
               model: str = "similarity") -> tuple[np.ndarray, float]:
    """Refine the b->a alignment with a 3-level ECC pyramid warm-started from
    `init` (3x3 output->input). Returns (3x3 similarity matrix, ECC correlation).

    a, b: float32 (h, w) luminance proxies in [0, 1].
    """
    if model not in {"translation", "similarity", "perspective"}:
        raise ValueError(f"unknown alignment model {model!r}")
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    pyr_a = [a]
    pyr_b = [b]
    for _ in range(levels - 1):
        pyr_a.insert(0, cv2.pyrDown(pyr_a[0]))
        pyr_b.insert(0, cv2.pyrDown(pyr_b[0]))
    coarsest_factor = 2 ** (levels - 1)
    warp_full = _scale_for_level(init, coarsest_factor)
    motion = cv2.MOTION_HOMOGRAPHY if model == "perspective" else cv2.MOTION_AFFINE
    warp = warp_full.astype(np.float32) if model == "perspective" else warp_full[:2].astype(np.float32)
    cc = 0.0
    for lvl, (la, lb) in enumerate(zip(pyr_a, pyr_b)):
        try:
            warp, cc = _ecc_at_level(la, lb, warp, iters=100, eps=1e-5, motion=motion)
        except cv2.error:
            cc = 0.0
        if lvl < len(pyr_a) - 1:
            fine = _scale_for_level(warp, 0.5)
            warp = fine.astype(np.float32) if model == "perspective" else fine[:2].astype(np.float32)
    matrix = _as_homogeneous(warp)
    matrix /= matrix[2, 2]
    if model == "perspective":
        return matrix, cc
    return project_to_similarity(matrix, cx=cx, cy=cy), cc


def brightness_gain(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Multiplicative gain g so that mean(b*g) ~ mean(a) inside `mask`
    (SPEC §6 step d, mean ratio). Falls back to 1.0 on empty/zero support."""
    m = mask.astype(bool)
    if m.sum() < 16:
        return 1.0
    mean_a = float(a[m].mean())
    mean_b = float(b[m].mean())
    if mean_b < 1e-6:
        return 1.0
    return mean_a / mean_b
