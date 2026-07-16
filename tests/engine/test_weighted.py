import numpy as np
from skimage.metrics import structural_similarity

from planefuse.backend import get_device
from planefuse.stack.base import get_algorithm
from planefuse.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_weighted_registered():
    from planefuse.stack import REGISTRY
    assert "weighted" in REGISTRY


def test_weighted_recovers_sharp():
    stack = generate_stack(h=140, w=180, n_frames=8, max_sigma=4.0, seed=61)
    src = ArrayFrameSource(stack.frames)
    algo = get_algorithm("weighted")
    result = algo.run(src, get_device("cpu"), {"temperature": 0.05, "sharpness_radius": 8})
    s = structural_similarity(np.clip(result.image, 0, 1), stack.sharp.clip(0, 1),
                              channel_axis=2, data_range=1.0)
    assert s > 0.95, f"SSIM {s}"


def test_weighted_low_temperature_approaches_hard_max():
    h, w = 32, 32
    rng = np.random.default_rng(0)
    blurry = np.full((h, w, 3), 0.5, dtype=np.float32)
    sharp = rng.uniform(0, 1, (h, w, 3)).astype(np.float32)
    src = ArrayFrameSource([blurry, sharp])
    algo = get_algorithm("weighted")
    out = algo.run(src, get_device("cpu"), {"temperature": 0.01, "sharpness_radius": 4}).image
    assert np.abs(out - sharp).mean() < np.abs(out - blurry).mean()
