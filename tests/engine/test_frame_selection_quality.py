import numpy as np
from skimage.metrics import structural_similarity

from focusstack.backend import get_device
from focusstack.select import SelectParams, select_frames
from focusstack.select.focus_measure import compute_focus_measures
from focusstack.select.reliability import classify_reliable, smooth_curves
from focusstack.stack.base import get_algorithm
from focusstack.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def _stack_subset(frames, indices):
    sub = ArrayFrameSource([frames[i] for i in indices])
    return get_algorithm("pmax").run(sub, get_device("cpu"), {"selection_smoothing": 1}).image


def test_textureless_cells_classified_unreliable():
    stack = generate_stack(h=160, w=200, n_frames=24, max_sigma=5.0, seed=77,
                           flat_region=True)
    src = ArrayFrameSource(stack.frames)
    phi, cell_rel = compute_focus_measures(src, get_device("cpu"), grid_rows=16, grid_cols=24)
    reliable = classify_reliable(smooth_curves(phi), cell_rel,
                                 kurtosis_threshold=SelectParams().kurtosis_threshold)
    fr = reliable[:4, :6]  # textureless top-left quarter (16//4=4 rows, 24//4=6 cols)
    assert fr.mean() < 0.25, f"textureless reliable fraction {fr.mean()}"
    assert reliable[8:, 8:].mean() > 0.4, f"textured reliable fraction {reliable[8:, 8:].mean()}"


def test_selection_coverage_and_quality():
    stack = generate_stack(h=140, w=180, n_frames=48, max_sigma=5.0, seed=78)
    src = ArrayFrameSource(stack.frames)
    res = select_frames(src, get_device("cpu"), SelectParams(grid_rows=16, grid_cols=24))
    assert res.warning is None
    assert 3 <= len(res.kept) <= 28, f"kept {len(res.kept)}"
    full = _stack_subset(stack.frames, list(range(48)))
    subset = _stack_subset(stack.frames, res.kept)
    s = structural_similarity(np.clip(subset, 0, 1), np.clip(full, 0, 1),
                              channel_axis=2, data_range=1.0)
    assert s > 0.97, f"subset-vs-full SSIM {s}"
