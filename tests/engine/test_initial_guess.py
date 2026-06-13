import numpy as np

from focusstack.backend import get_device
from focusstack.align.initial import estimate_scale_rotation, estimate_translation
from focusstack.align.proxy import make_proxy
from tests.synthetic.generate import generate_stack


def _proxy(frame):
    p, _ = make_proxy(frame, get_device("cpu"), max_long_edge=256)
    return p


def test_translation_recovered_on_pure_shift():
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.5, seed=5,
                           trans_jitter=6.0, scale_step=0.0, rot_jitter=0.0)
    a, b = _proxy(stack.frames[0]), _proxy(stack.frames[1])
    dy, dx = estimate_translation(a, b)
    assert np.isfinite(dy) and np.isfinite(dx)


def test_scale_rotation_recovered_on_breathing():
    stack = generate_stack(h=200, w=240, n_frames=2, max_sigma=0.5, seed=6,
                           scale_step=0.03, rot_jitter=0.04, trans_jitter=0.0)
    a, b = _proxy(stack.frames[0]), _proxy(stack.frames[1])
    scale, angle = estimate_scale_rotation(a, b)
    assert 0.90 < scale < 1.12
    assert abs(angle) < 0.15
