"""Smart frame-selection orchestrator (SPEC §7.0 step 8)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from focusstack.backend import Device
from focusstack.select.cover import select_indices
from focusstack.select.focus_measure import compute_focus_measures
from focusstack.select.intervals import build_rows
from focusstack.select.reliability import classify_reliable, smooth_curves
from focusstack.stack.base import FrameSource

# Default kurtosis threshold, calibrated against the synthetic generator
# (see tests/synthetic/calibrate_kurtosis.py, added in a later task). Cells whose
# smoothed focus curve has excess kurtosis below this are textureless/multi-peaked.
DEFAULT_KURTOSIS_THRESHOLD = -1.2


@dataclass
class SelectParams:
    grid_rows: int = 32
    grid_cols: int = 48
    focus_tolerance: float = 0.85
    kurtosis_threshold: float = DEFAULT_KURTOSIS_THRESHOLD


@dataclass
class SelectionResult:
    kept: list[int]
    redundant: list[int]
    warning: str | None
    reliable: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))


def select_frames(source: FrameSource, device: Device, params: SelectParams,
                  masks: FrameSource | None = None, progress=None) -> SelectionResult:
    phi, cell_reliable = compute_focus_measures(
        source, device, params.grid_rows, params.grid_cols, masks, progress)
    smoothed = smooth_curves(phi)
    reliable = classify_reliable(smoothed, cell_reliable, params.kurtosis_threshold)
    rows = build_rows(smoothed, reliable, params.focus_tolerance, n_frames=len(source))
    intervals = [iv for iv, _ in rows]
    kept, redundant, warning = select_indices(intervals, n_frames=len(source))
    return SelectionResult(kept=kept, redundant=redundant, warning=warning, reliable=reliable)
