import numpy as np

from focusstack.backend import get_device
from focusstack.align.proxy import make_proxy


def test_proxy_downscales_to_max_long_edge():
    rng = np.random.default_rng(0)
    frame = rng.uniform(0, 1, (1000, 1500, 3)).astype(np.float32)
    proxy, factor = make_proxy(frame, get_device("cpu"), max_long_edge=512)
    assert proxy.shape[0] == 1                # luminance, (1, h, w)
    assert max(proxy.shape[-2:]) == 512
    assert factor == 1500 / 512


def test_proxy_no_upscale_when_small():
    frame = np.zeros((100, 80, 3), dtype=np.float32)
    proxy, factor = make_proxy(frame, get_device("cpu"), max_long_edge=2048)
    assert factor == 1.0
    assert proxy.shape == (1, 100, 80)
