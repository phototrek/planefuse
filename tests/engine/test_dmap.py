import numpy as np
from skimage.metrics import structural_similarity

from focusstack.backend import get_device
from focusstack.stack.base import get_algorithm
from focusstack.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_dmap_registered():
    from focusstack.stack import REGISTRY
    assert "dmap" in REGISTRY


def test_dmap_recovers_sharp_and_depth():
    stack = generate_stack(h=160, w=200, n_frames=8, max_sigma=4.0, seed=60)
    src = ArrayFrameSource(stack.frames)
    algo = get_algorithm("dmap")
    result = algo.run(src, get_device("cpu"),
                      {"estimation_radius": 8, "contrast_threshold": 7.0, "smoothing_radius": 16})
    s = structural_similarity(np.clip(result.image, 0, 1), stack.sharp.clip(0, 1),
                              channel_axis=2, data_range=1.0)
    # SPEC §13.2 target is SSIM>0.97; this implementation reaches ~0.997 here.
    assert s > 0.97, f"SSIM {s}"
    assert "depth" in result.aux
    depth = result.aux["depth"]
    assert depth.shape == (160, 200)
    gt = stack.depth
    corr = np.corrcoef(depth.ravel(), gt.ravel())[0, 1]
    # SPEC §13.2 target is depth-corr>0.95; this implementation reaches ~0.992 here.
    assert abs(corr) > 0.95, f"depth corr {corr}"
