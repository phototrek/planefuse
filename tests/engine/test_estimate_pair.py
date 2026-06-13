import numpy as np

from focusstack.backend import get_device
from focusstack.align.estimate import PairResult, estimate_pair
from tests.synthetic.generate import generate_stack


def test_estimate_pair_returns_result():
    stack = generate_stack(h=240, w=300, n_frames=2, max_sigma=0.4, seed=12,
                           scale_step=0.01, trans_jitter=3.0, rot_jitter=0.01)
    res = estimate_pair(stack.frames[0], stack.frames[1], get_device("cpu"),
                        max_long_edge=256, model="similarity")
    assert isinstance(res, PairResult)
    assert res.matrix.shape == (3, 3)        # at PROXY resolution
    assert 0.0 <= res.correlation <= 1.0
    assert res.proxy_factor >= 1.0
    assert np.isfinite(res.gain)


def test_estimate_pair_translation_model_skips_logpolar():
    stack = generate_stack(h=200, w=200, n_frames=2, max_sigma=0.4, seed=13,
                           trans_jitter=4.0)
    res = estimate_pair(stack.frames[0], stack.frames[1], get_device("cpu"),
                        max_long_edge=256, model="translation")
    np.testing.assert_allclose(res.matrix[:2, :2], np.eye(2), atol=1e-6)
