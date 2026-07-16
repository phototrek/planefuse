import numpy as np

from planefuse.backend import get_device
from planefuse.stack.dmap import refine_index


def test_refine_smooths_and_fills_undecided():
    rng = np.random.default_rng(0)
    h, w = 40, 40
    index = rng.integers(0, 5, (h, w)).astype(np.float32)
    sharp = rng.uniform(0.5, 1.0, (h, w)).astype(np.float32)
    sharp[:, :8] = 0.001  # a low-contrast band -> undecided at any sane percentile
    out = refine_index(index, sharp, get_device("cpu"),
                       contrast_percentile=7.0, smoothing_radius=6)
    assert out.shape == (h, w)
    assert out.dtype == np.float32
    assert out[:, :8].std() < index[:, :8].std()


def test_refine_all_decided_when_high_contrast():
    h, w = 24, 24
    index = np.full((h, w), 2.0, dtype=np.float32)
    sharp = np.full((h, w), 0.9, dtype=np.float32)  # uniformly high contrast
    out = refine_index(index, sharp, get_device("cpu"),
                       contrast_percentile=7.0, smoothing_radius=4)
    np.testing.assert_allclose(out, 2.0, atol=1e-3)
