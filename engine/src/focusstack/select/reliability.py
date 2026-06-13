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
                      kurtosis_threshold: float,
                      min_amplitude_frac: float = 0.3) -> np.ndarray:
    """(rows, cols) bool: cells whose smoothed curve kurtosis >= threshold AND
    that were not already marked unreliable (validity mask).

    A textureless cell (e.g. a blank wall) produces a near-zero focus measure;
    its smoothed curve is dominated by floating-point noise, so its kurtosis is
    meaningless (it can land anywhere). Kurtosis alone therefore cannot reject
    such cells. Before the kurtosis test we apply a scene-relative amplitude
    gate: a cell is only eligible if its smoothed curve peaks at >=
    ``min_amplitude_frac`` of the brightest cell's peak. This is the
    standard-deviation rule SPEC §7.0 step 3 notes the kurtosis test
    *dominates* but does not eliminate, and it is what lets known textureless
    regions be rejected as required by the §13.2 calibration check.
    """
    n, r, c = smoothed.shape
    out = np.zeros((r, c), dtype=bool)
    global_peak = float(smoothed.max())
    amp_floor = min_amplitude_frac * global_peak
    for i in range(r):
        for j in range(c):
            if not cell_reliable[i, j]:
                continue
            curve = smoothed[:, i, j]
            if float(curve.max()) < amp_floor:
                continue  # textureless: focus signal below the scene floor
            out[i, j] = excess_kurtosis(curve) >= kurtosis_threshold
    return out
