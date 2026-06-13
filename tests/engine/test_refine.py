import numpy as np

from focusstack.backend import get_device
from focusstack.align.proxy import make_proxy
from focusstack.align.refine import brightness_gain, refine_ecc
from tests.synthetic.generate import generate_stack


def _proxy_np(frame):
    p, _ = make_proxy(frame, get_device("cpu"), max_long_edge=256)
    return p.squeeze(0).cpu().numpy()


def test_refine_ecc_improves_toward_identity_pair():
    # two near-identical frames -> ECC returns ~identity, corr ~1
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.3, seed=7)
    a, b = _proxy_np(stack.frames[0]), _proxy_np(stack.frames[1])
    m, corr = refine_ecc(a, b, init=np.eye(3), cx=a.shape[1] / 2, cy=a.shape[0] / 2)
    assert corr > 0.9
    assert abs(m[0, 0] - 1.0) < 0.05 and abs(m[1, 1] - 1.0) < 0.05


def test_refine_ecc_returns_similarity():
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.3, seed=8,
                           scale_step=0.02)
    a, b = _proxy_np(stack.frames[0]), _proxy_np(stack.frames[1])
    m, corr = refine_ecc(a, b, init=np.eye(3), cx=a.shape[1] / 2, cy=a.shape[0] / 2)
    block = m[:2, :2]
    np.testing.assert_allclose(block[:, 0] @ block[:, 1], 0.0, atol=1e-4)  # orthogonal cols


def test_brightness_gain_matches_mean():
    a = np.full((50, 50), 0.4, dtype=np.float32)
    b = np.full((50, 50), 0.2, dtype=np.float32)
    mask = np.ones((50, 50), dtype=bool)
    gain = brightness_gain(a, b, mask)
    np.testing.assert_allclose(gain, 2.0, atol=1e-3)  # b * gain matches a
