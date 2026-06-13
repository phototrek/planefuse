import numpy as np
from skimage.metrics import structural_similarity

from focusstack.io import save_image
from focusstack.pipeline import stack_frames
from focusstack.align import AlignParams
from tests.synthetic.generate import generate_stack


def test_stack_with_alignment_recovers_sharp(tmp_path):
    stack = generate_stack(h=200, w=260, n_frames=7, max_sigma=4.0, seed=30,
                           scale_step=0.003, trans_jitter=2.5, rot_jitter=0.01)
    src_dir = tmp_path / "frames"
    src_dir.mkdir()
    paths = []
    for i, f in enumerate(stack.frames):
        p = src_dir / f"f_{i:03d}.tif"
        save_image(f, p, bit_depth=16)
        paths.append(p)
    result = stack_frames(paths, method="pmax", device_pref="cpu",
                          align=AlignParams(max_long_edge=256),
                          cache_dir=tmp_path / "cache")
    s = structural_similarity(np.clip(result.image, 0, 1), stack.sharp.clip(0, 1),
                              channel_axis=2, data_range=1.0)
    assert s > 0.90


def test_stack_pre_aligned_path_unchanged(tmp_path):
    stack = generate_stack(h=120, w=140, n_frames=5, max_sigma=4.0, seed=31)
    paths = []
    for i, f in enumerate(stack.frames):
        p = tmp_path / f"f_{i:03d}.tif"
        save_image(f, p, bit_depth=16)
        paths.append(p)
    result = stack_frames(paths, method="pmax", device_pref="cpu", align=None)
    assert result.image.shape == (120, 140, 3)
