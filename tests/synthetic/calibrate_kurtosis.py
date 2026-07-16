"""Kurtosis-threshold calibration (SPEC §7.0 step 3, §13.2).

Generates a stack with a known textureless rectangle (top-left quarter), computes
smoothed focus curves, and reports the excess-kurtosis distribution for textured
vs textureless cells. The recommended threshold separates the two populations.
Run: uv run python -m tests.synthetic.calibrate_kurtosis
"""

from __future__ import annotations

import numpy as np

from planefuse.backend import get_device
from planefuse.select.focus_measure import compute_focus_measures
from planefuse.select.reliability import excess_kurtosis, smooth_curves
from planefuse.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def calibrate(grid_rows: int = 16, grid_cols: int = 24) -> dict:
    stack = generate_stack(h=160, w=200, n_frames=30, max_sigma=5.0, seed=77,
                           flat_region=True)
    src = ArrayFrameSource(stack.frames)
    phi, _ = compute_focus_measures(src, get_device("cpu"), grid_rows, grid_cols)
    smoothed = smooth_curves(phi)
    fr_rows = grid_rows // 4
    fr_cols = grid_cols // 4
    textureless, textured = [], []
    for i in range(grid_rows):
        for j in range(grid_cols):
            k = excess_kurtosis(smoothed[:, i, j])
            (textureless if (i < fr_rows and j < fr_cols) else textured).append(k)
    tl = np.array(textureless)
    tx = np.array(textured)
    sep = tl.max() < tx.min()
    threshold = float((tl.max() + tx.min()) / 2) if sep else float(np.percentile(tx, 25))
    return {"textureless_n": int(tl.size), "textured_n": int(tx.size),
            "textureless_max": float(tl.max()), "textureless_mean": float(tl.mean()),
            "textured_min": float(tx.min()), "textured_median": float(np.median(tx)),
            "cleanly_separable": bool(sep), "recommended_threshold": threshold}


if __name__ == "__main__":
    import json
    print(json.dumps(calibrate(), indent=2))
