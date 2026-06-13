"""Curve smoothing + kurtosis reliability (SPEC §7.0 steps 2-3)."""

from __future__ import annotations

import numpy as np


def smooth_curves(phi: np.ndarray) -> np.ndarray:
    """Replace each cell's across-frames curve with the SUM of itself + its 8
    neighbours (fewer at borders). phi (n, rows, cols) -> same shape."""
    n, r, c = phi.shape
    padded = np.pad(phi, ((0, 0), (1, 1), (1, 1)), mode="constant")
    out = np.zeros_like(phi)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            out += padded[:, di:di + r, dj:dj + c]
    return out


def excess_kurtosis(curve: np.ndarray) -> float:
    """Fisher (excess) kurtosis, population/biased estimator: m4/m2^2 − 3.
    A constant (zero-variance) curve is defined as -3.0 (maximally unreliable)."""
    x = curve.astype(np.float64)
    mean = x.mean()
    d = x - mean
    m2 = float((d * d).mean())
    if m2 < 1e-12:
        return -3.0
    m4 = float((d ** 4).mean())
    return m4 / (m2 * m2) - 3.0


def classify_reliable(smoothed: np.ndarray, cell_reliable: np.ndarray,
                      kurtosis_threshold: float) -> np.ndarray:
    """(rows, cols) bool: cells whose smoothed curve kurtosis >= threshold AND
    that were not already marked unreliable (validity mask)."""
    n, r, c = smoothed.shape
    out = np.zeros((r, c), dtype=bool)
    for i in range(r):
        for j in range(c):
            if not cell_reliable[i, j]:
                continue
            out[i, j] = excess_kurtosis(smoothed[:, i, j]) >= kurtosis_threshold
    return out
