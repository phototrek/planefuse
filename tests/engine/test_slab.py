import numpy as np
from skimage.metrics import structural_similarity

from planefuse.backend import get_device
from planefuse.stack.base import get_algorithm
from planefuse.stack.slab import plan_slabs
from planefuse.stack.sources import ArrayFrameSource
from tests.synthetic.generate import generate_stack


def test_plan_slabs_overlap():
    slabs = plan_slabs(n=25, size=10, overlap=2)
    assert slabs[0] == (0, 10)
    assert slabs[1][0] == 8
    assert slabs[-1][1] == 25
    covered = set()
    for a, b in slabs:
        covered |= set(range(a, b))
    assert covered == set(range(25))


def test_slab_registered_and_recovers_sharp():
    from planefuse.stack import REGISTRY
    assert "slab" in REGISTRY
    stack = generate_stack(h=120, w=150, n_frames=16, max_sigma=4.0, seed=62)
    src = ArrayFrameSource(stack.frames)
    algo = get_algorithm("slab")
    result = algo.run(src, get_device("cpu"),
                      {"slab_size": 6, "slab_overlap": 2, "inner_method": "pmax",
                       "outer_method": "pmax"})
    s = structural_similarity(np.clip(result.image, 0, 1), stack.sharp.clip(0, 1),
                              channel_axis=2, data_range=1.0)
    assert s > 0.95, f"SSIM {s}"
