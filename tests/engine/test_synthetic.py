import numpy as np

from tests.synthetic.generate import generate_stack, make_scene


def test_scene_shapes_and_ranges():
    sharp, depth = make_scene(96, 128, seed=3)
    assert sharp.shape == (96, 128, 3)
    assert depth.shape == (96, 128)
    assert sharp.dtype == np.float32 and depth.dtype == np.float32
    assert 0.0 <= depth.min() and depth.max() <= 1.0
    assert sharp.std() > 0.05  # must have texture for sharpness metrics


def test_stack_basic_properties():
    stack = generate_stack(h=96, w=128, n_frames=8, seed=3)
    assert len(stack.frames) == 8
    assert all(f.shape == (96, 128, 3) for f in stack.frames)
    assert stack.sharp.shape == (96, 128, 3)
    assert len(stack.transforms) == 8
    np.testing.assert_allclose(stack.transforms[0], np.eye(3), atol=1e-6)


def test_each_frame_sharpest_at_its_focus_depth():
    """Frame p must be the sharpest frame at pixels whose depth ~ p/(n-1)."""
    n = 8
    stack = generate_stack(h=96, w=128, n_frames=n, max_sigma=5.0, seed=3)

    def grad_energy(img, mask):
        gray = img.mean(axis=2)
        gy, gx = np.gradient(gray)
        return float(((gx**2 + gy**2) * mask).sum() / max(mask.sum(), 1))

    for p in [0, n // 2, n - 1]:
        focus = p / (n - 1)
        band = np.abs(stack.depth - focus) < 0.04
        if band.sum() < 200:
            continue
        energies = [grad_energy(f, band) for f in stack.frames]
        assert int(np.argmax(energies)) == p


def test_brightness_flicker_and_noise():
    a = generate_stack(h=64, w=64, n_frames=4, seed=1, flicker=0.1, noise=0.02)
    b = generate_stack(h=64, w=64, n_frames=4, seed=1, flicker=0.0, noise=0.0)
    assert not np.allclose(a.frames[1], b.frames[1])


def test_geometric_jitter_changes_frames():
    a = generate_stack(h=64, w=64, n_frames=4, seed=1, scale_step=0.002)
    b = generate_stack(h=64, w=64, n_frames=4, seed=1)
    np.testing.assert_allclose(a.frames[0], b.frames[0], atol=1e-6)  # frame 0 = identity
    assert not np.allclose(a.frames[3], b.frames[3])
    assert a.transforms[3][0, 0] != 1.0  # scale recorded in ground truth
