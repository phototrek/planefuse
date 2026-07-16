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
# removes thread-count-dependent variation on one machine. CPU/OpenCV builds
# can still select different local minima; the geometric and directional
# checks below handle those. Proxy images are small enough that the
# single-threaded cost is negligible next to full-resolution warping/fusion.
cv2.setNumThreads(1)

_RETRY_CORRELATION_FLOOR = 0.5
_MAX_SCALE_REFINEMENT = 0.05
_MAX_CORNER_REFINEMENT = 0.05


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


def _run_ecc_pyramid(pyr_a: list[np.ndarray], pyr_b: list[np.ndarray],
                     init: np.ndarray, motion: int) -> tuple[np.ndarray, float]:
    """Run one directional ECC solve from coarse to fine."""
    coarsest_factor = 2 ** (len(pyr_a) - 1)
    warp_at_level = _scale_for_level(init, coarsest_factor)
    if motion != cv2.MOTION_HOMOGRAPHY:
        warp_at_level = warp_at_level[:2]
    warp = warp_at_level.astype(np.float32)
    cc = 0.0
    for lvl, (la, lb) in enumerate(zip(pyr_a, pyr_b)):
        try:
            warp, cc = _ecc_at_level(
                la, lb, warp, iters=100, eps=1e-5, motion=motion)
        except cv2.error:
            # _ecc_at_level works on a copy, so the last stable warm start is
            # still safe to carry to the next level.
            cc = 0.0
        if lvl < len(pyr_a) - 1:
            fine = _scale_for_level(warp, 0.5)
            warp = fine.astype(np.float32)
            if motion != cv2.MOTION_HOMOGRAPHY:
                warp = warp[:2]
    return _as_homogeneous(warp), cc


def _project_affine(matrix: np.ndarray, cx: float, cy: float) -> np.ndarray:
    return project_to_similarity(_as_homogeneous(matrix), cx=cx, cy=cy)


def _is_stable_refinement(candidate: np.ndarray, init: np.ndarray,
                          shape: tuple[int, int]) -> bool:
    """Whether an ECC similarity is a local refinement of its global guess.

    ECC's affine objective can prefer scale/shear changes that mimic a focus
    difference. The FFT/log-polar and phase-correlation stages already found
    the global basin, so ECC is allowed to correct that guess but not replace
    it with substantially different geometry.
    """
    if not np.all(np.isfinite(candidate)):
        return False
    candidate_scale = float(np.sqrt(abs(np.linalg.det(candidate[:2, :2]))))
    init_scale = float(np.sqrt(abs(np.linalg.det(init[:2, :2]))))
    if candidate_scale < 1e-8 or init_scale < 1e-8:
        return False
    scale_ratio = candidate_scale / init_scale
    if not 1.0 - _MAX_SCALE_REFINEMENT <= scale_ratio <= 1.0 + _MAX_SCALE_REFINEMENT:
        return False

    h, w = shape
    corners = np.array(
        [[0.0, 0.0, 1.0], [w - 1.0, 0.0, 1.0],
         [w - 1.0, h - 1.0, 1.0], [0.0, h - 1.0, 1.0]],
        dtype=np.float64,
    )
    candidate_xy = (candidate @ corners.T)[:2]
    init_xy = (init @ corners.T)[:2]
    correction = np.linalg.norm(candidate_xy - init_xy, axis=0)
    return bool(np.max(correction) <= np.hypot(h, w) * _MAX_CORNER_REFINEMENT)


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
    motion = cv2.MOTION_HOMOGRAPHY if model == "perspective" else cv2.MOTION_AFFINE
    matrix, cc = _run_ecc_pyramid(pyr_a, pyr_b, init, motion)
    if model == "perspective":
        if not np.all(np.isfinite(matrix)) or abs(matrix[2, 2]) < 1e-8:
            return _as_homogeneous(init), 0.0
        return matrix / matrix[2, 2], cc

    initial = _project_affine(init, cx, cy)
    forward = _project_affine(matrix, cx, cy)
    forward_stable = _is_stable_refinement(forward, initial, a.shape)
    if forward_stable and np.isfinite(cc) and cc >= _RETRY_CORRELATION_FLOOR:
        return forward, cc

    # ECC is direction-sensitive when the two frames have different focus
    # blur. A false basin in b->a is often well-conditioned in a->b. Retry only
    # after an obvious photometric or geometric failure, then invert the result
    # back to the requested b->a convention.
    try:
        reverse_init = np.linalg.inv(initial)
    except np.linalg.LinAlgError:
        return initial, 0.0
    reverse_matrix, reverse_cc = _run_ecc_pyramid(
        pyr_b, pyr_a, reverse_init, motion)
    reverse = _project_affine(reverse_matrix, cx, cy)
    try:
        reverse_as_forward = np.linalg.inv(reverse)
    except np.linalg.LinAlgError:
        reverse_as_forward = np.full((3, 3), np.nan, dtype=np.float64)
    reverse_stable = _is_stable_refinement(reverse_as_forward, initial, a.shape)

    stable: list[tuple[np.ndarray, float]] = []
    if forward_stable and np.isfinite(cc):
        stable.append((forward, cc))
    if reverse_stable and np.isfinite(reverse_cc):
        stable.append((reverse_as_forward, reverse_cc))
    if any(result[1] >= _RETRY_CORRELATION_FLOOR for result in stable):
        return max(stable, key=lambda result: result[1])

    # On rare pairs the coarsest image has too little texture and sends both
    # directions into bad basins. Retry from the original global initializer
    # at full proxy resolution, where the missing detail is available.
    full_matrix, full_cc = _run_ecc_pyramid(
        pyr_a[-1:], pyr_b[-1:], initial, motion)
    full = _project_affine(full_matrix, cx, cy)
    if _is_stable_refinement(full, initial, a.shape) and np.isfinite(full_cc):
        stable.append((full, full_cc))
    if stable:
        return max(stable, key=lambda result: result[1])
    return initial, 0.0


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
