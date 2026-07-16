import numpy as np
from skimage.metrics import structural_similarity

from planefuse.backend import get_device
from planefuse.stack import ArrayFrameSource, get_algorithm
from planefuse.tiles import estimate_stack_bytes, plan_tiles, stack_tiled
from tests.synthetic.generate import generate_stack


def test_plan_tiles_covers_image_exactly():
    tiles = plan_tiles(500, 700, tile=256, overlap=32)
    covered = np.zeros((500, 700), dtype=int)
    for t in tiles:
        covered[t.y0 : t.y1, t.x0 : t.x1] += 1
    assert (covered >= 1).all()


def test_plan_tiles_single_when_image_fits():
    tiles = plan_tiles(200, 200, tile=256, overlap=32)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.y0, t.x0, t.y1, t.x1) == (0, 0, 200, 200)


def test_estimate_scales_with_pixels():
    small = estimate_stack_bytes(1000, 1500)
    big = estimate_stack_bytes(4000, 6000)
    assert big > small * 10
    assert small > 1000 * 1500 * 3 * 4  # at least one frame's worth


def test_tiled_matches_untiled():
    # SPEC §8: tiled and untiled agree within atol=2e-3
    synth = generate_stack(h=192, w=256, n_frames=8, max_sigma=4.0, seed=11)
    device = get_device("cpu")
    algo = get_algorithm("pmax")
    src = ArrayFrameSource(synth.frames)
    params = {"selection_smoothing": 0}
    untiled = algo.run(src, device, params).image
    tiled = stack_tiled("pmax", src, device, params, tile=96, overlap=48)
    np.testing.assert_allclose(tiled, untiled, atol=2e-3)
    # belt and braces: quality preserved
    s = structural_similarity(tiled.clip(0, 1), synth.sharp, channel_axis=2, data_range=1.0)
    assert s > 0.95
