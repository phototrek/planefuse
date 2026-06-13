import numpy as np

from focusstack.backend import get_device
from focusstack.select import SelectParams, select_frames
from focusstack.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_select_thins_oversampled_stack():
    stack = generate_stack(h=120, w=160, n_frames=40, max_sigma=5.0, seed=80)
    src = ArrayFrameSource(stack.frames)
    res = select_frames(src, get_device("cpu"),
                        SelectParams(grid_rows=16, grid_cols=24, focus_tolerance=0.85))
    assert res.warning is None
    assert sorted(res.kept) == res.kept
    assert set(res.kept) | set(res.redundant) == set(range(40))
    assert 3 <= len(res.kept) < 40
    assert all(0 <= k < 40 for k in res.kept)


def test_select_low_contrast_keeps_all():
    frames = [np.full((64, 64, 3), 0.5, dtype=np.float32) + i * 1e-4 for i in range(6)]
    src = ArrayFrameSource(frames)
    res = select_frames(src, get_device("cpu"), SelectParams(grid_rows=8, grid_cols=8))
    assert res.kept == list(range(6))
    assert res.warning is not None
