import numpy as np

from focusstack.backend import get_device, ops  # noqa: F401
from focusstack.align.warp import warp_full
from focusstack.align.transforms import translation_matrix


def test_warp_full_identity_returns_frame_and_full_mask():
    rng = np.random.default_rng(0)
    frame = rng.uniform(0, 1, (64, 80, 3)).astype(np.float32)
    aligned, mask = warp_full(frame, np.eye(3), get_device("cpu"),
                              out_shape=(64, 80), proxy_factor=1.0, interp="bilinear")
    np.testing.assert_allclose(aligned, frame, atol=1e-4)
    assert mask.dtype == bool
    assert mask.all()


def test_warp_full_translation_marks_revealed_border_invalid():
    frame = np.ones((64, 80, 3), dtype=np.float32)
    m = translation_matrix(10.0, 0.0)
    aligned, mask = warp_full(frame, m, get_device("cpu"), out_shape=(64, 80),
                              proxy_factor=1.0, interp="bilinear")
    assert not mask[:, -1].any()
    assert mask[:, 0].all()


def test_warp_full_scales_proxy_translation():
    frame = np.ones((100, 100, 3), dtype=np.float32)
    m_proxy = translation_matrix(2.0, 0.0)   # estimated at /4
    _, mask = warp_full(frame, m_proxy, get_device("cpu"), out_shape=(100, 100),
                        proxy_factor=4.0, interp="bilinear")
    invalid_cols = (~mask).any(axis=0).sum()
    assert 6 <= invalid_cols <= 10
