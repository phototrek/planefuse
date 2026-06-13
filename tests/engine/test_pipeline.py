import numpy as np
import pytest

from focusstack.errors import ValidationError
from focusstack.io import save_image
from focusstack.pipeline import stack_frames
from tests.synthetic.generate import generate_stack


@pytest.fixture
def stack_dir(tmp_path):
    synth = generate_stack(h=96, w=128, n_frames=6, max_sigma=4.0, seed=2)
    for i, f in enumerate(synth.frames):
        save_image(f, tmp_path / f"frame_{i:03d}.tif", bit_depth=16)
    return tmp_path, synth


def test_stack_frames_from_dir(stack_dir):
    d, synth = stack_dir
    result = stack_frames(sorted(d.glob("*.tif")), method="pmax", device_pref="cpu")
    assert result.image.shape == synth.sharp.shape


def test_validation_failure_raises_with_filenames(stack_dir):
    d, _ = stack_dir
    bad = np.zeros((10, 10, 3), dtype=np.float32)
    save_image(bad, d / "zz_wrong_size.tif")
    with pytest.raises(ValidationError, match="zz_wrong_size"):
        stack_frames(sorted(d.glob("*.tif")), method="pmax", device_pref="cpu")


def test_forced_tiled_mode_matches_direct(stack_dir):
    d, _ = stack_dir
    paths = sorted(d.glob("*.tif"))
    direct = stack_frames(paths, method="pmax", device_pref="cpu", tile_mode="never")
    tiled = stack_frames(paths, method="pmax", device_pref="cpu", tile_mode="always", tile=64)
    np.testing.assert_allclose(tiled.image, direct.image, atol=2e-3)
