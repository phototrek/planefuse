import numpy as np
import pytest

from focusstack.io import save_image
from focusstack.stack.base import REGISTRY, ParamSpec, StackResult, get_algorithm, register
from focusstack.stack.sources import ArrayFrameSource, DirFrameSource


def test_register_and_lookup():
    @register("dummy")
    class Dummy:
        name = "dummy"

        @staticmethod
        def params():
            return [ParamSpec("k", "K", "int", default=1, min=0, max=5, tooltip="test")]

        def run(self, source, device, params, progress=None, cancel=None):
            return StackResult(image=source.read(0))

    assert "dummy" in REGISTRY
    algo = get_algorithm("dummy")
    assert algo.params()[0].name == "k"
    del REGISTRY["dummy"]


def test_unknown_algorithm_raises():
    with pytest.raises(KeyError, match="nope"):
        get_algorithm("nope")


def test_array_source_read_and_region():
    frames = [np.full((20, 30, 3), i / 10, dtype=np.float32) for i in range(4)]
    src = ArrayFrameSource(frames)
    assert len(src) == 4
    assert src.read(2).shape == (20, 30, 3)
    crop = src.read(1, region=(5, 10, 15, 25))  # (y0, x0, y1, x1)
    assert crop.shape == (10, 15, 3)
    np.testing.assert_allclose(crop, 0.1)


def test_dir_source(tmp_path):
    for i in range(3):
        arr = np.full((16, 24, 3), i / 4, dtype=np.float32)
        save_image(arr, tmp_path / f"f{i}.tif", bit_depth=16)
    src = DirFrameSource(sorted(tmp_path.glob("*.tif")))
    assert len(src) == 3
    assert src.read(0).shape == (16, 24, 3)
    crop = src.read(2, region=(0, 0, 8, 8))
    assert crop.shape == (8, 8, 3)
    np.testing.assert_allclose(crop, 0.5, atol=1e-3)
