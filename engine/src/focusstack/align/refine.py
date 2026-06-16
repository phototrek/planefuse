"""ECC refinement and brightness normalization (SPEC §6 steps c, d).

ECC runs on CPU (OpenCV) over downscaled luminance proxies — cheap and the spec
mandates it. The refined affine is projected to a similarity (the output model).
"""

from __future__ import annotations

import cv2
import numpy as np

from focusstack.align.transforms import project_to_similarity


def _ecc_at_level(a: np.ndarray, b: np.ndarray, warp_init: np.ndarray,
                  iters: int, eps: float) -> tuple[np.ndarray, float]:
    """One ECC solve with a 2x3 affine warp. Returns (2x3 warp, correlation)."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    warp = warp_init.astype(np.float32)
    try:
        cc, out = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_AFFINE, criteria, None, 5)  # type: ignore[call-overload]
    except TypeError:
        cc, out = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, criteria)
    return np.asarray(out, dtype=np.float32), float(cc)


def refine_ecc(a: np.ndarray, b: np.ndarray, init: np.ndarray,
               cx: float, cy: float, levels: int = 3, iters: int = 100) -> tuple[np.ndarray, float]:
    """Refine the b->a alignment with an ECC pyramid warm-started from
    `init` (3x3 output->input). Returns (3x3 similarity matrix, ECC correlation).

    a, b: float32 (h, w) luminance images in [0, 1]. `levels=1` does a single
    solve at the input resolution (used for the warm-started full-res polish).
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    pyr_a = [a]
    pyr_b = [b]
    for _ in range(levels - 1):
        pyr_a.insert(0, cv2.pyrDown(pyr_a[0]))
        pyr_b.insert(0, cv2.pyrDown(pyr_b[0]))
    warp = init[:2, :].astype(np.float32).copy()
    coarsest_factor = 2 ** (levels - 1)
    warp[:, 2] /= coarsest_factor
    cc = 0.0
    for lvl, (la, lb) in enumerate(zip(pyr_a, pyr_b)):
        try:
            warp, cc = _ecc_at_level(la, lb, warp, iters=iters, eps=1e-5)
        except cv2.error:
            cc = 0.0
        if lvl < len(pyr_a) - 1:
            warp[:, 2] *= 2.0  # translation doubles going one level finer
    m = np.eye(3, dtype=np.float64)
    m[:2, :] = warp
    return project_to_similarity(m, cx=cx, cy=cy), cc


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
