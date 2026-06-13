"""Pure 3x3 transform math (SPEC §6). numpy float64; output->input pixel coords.

A matrix M maps output pixel (col, row, 1) -> input pixel; warp(img, M) samples
img at M·x. Composition matches matrix multiplication: compose(A, B) == A @ B.
"""

from __future__ import annotations

import numpy as np


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def similarity_matrix(scale: float, angle: float, tx: float, ty: float,
                      cx: float, cy: float) -> np.ndarray:
    """Similarity (uniform scale + rotation + translation) about center (cx, cy)."""
    ca, sa = np.cos(angle), np.sin(angle)
    rot = np.array([[scale * ca, -scale * sa, 0],
                    [scale * sa, scale * ca, 0],
                    [0, 0, 1]], dtype=np.float64)
    pre = translation_matrix(-cx, -cy)
    post = translation_matrix(cx + tx, cy + ty)
    return post @ rot @ pre


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float64) @ b.astype(np.float64)


def invert(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m.astype(np.float64))


def scale_transform_to_resolution(m: np.ndarray, factor: float) -> np.ndarray:
    """Re-express an output->input transform estimated at downscaled resolution
    for use at `factor`× higher resolution: M_full = S · M_low · S^{-1} with
    S = diag(factor, factor, 1). The 2x2 block is unchanged; translation scales
    by `factor` (SPEC §6 step 5)."""
    s = np.diag([factor, factor, 1.0])
    return s @ m.astype(np.float64) @ np.linalg.inv(s)


def project_to_similarity(affine: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Project an affine output->input matrix to the nearest similarity via
    orthogonal Procrustes on the 2x2 block (SPEC §6 step c). Translation is
    recomputed so the image center maps consistently."""
    a = affine[:2, :2].astype(np.float64)
    u, s, vt = np.linalg.svd(a)
    r = u @ vt                          # nearest rotation
    if np.linalg.det(r) < 0:            # reflect-free
        u[:, -1] *= -1
        r = u @ vt
    scale = float(s.mean())             # uniform scale = mean singular value
    sim2 = scale * r
    center = np.array([cx, cy], dtype=np.float64)
    # Recompute translation to preserve the image center mapping
    t = affine[:2, 2] + (affine[:2, :2] - sim2) @ center
    out = np.eye(3, dtype=np.float64)
    out[:2, :2] = sim2
    out[:2, 2] = t
    return out
