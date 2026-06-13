import numpy as np
import tifffile

from focusstack.io.writer import save_image


def test_depth_map_16bit_tiff_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    depth = rng.uniform(0, 1, (40, 60)).astype(np.float32)   # 2-D (H, W)
    out = tmp_path / "depth.tif"
    save_image(depth, out, bit_depth=16)
    back = tifffile.imread(out)
    assert back.shape == (40, 60)
    assert back.dtype == np.uint16
    np.testing.assert_allclose(back.astype(np.float32) / 65535.0, depth,
                               atol=1.0 / 65535 + 1e-6)


def test_depth_map_8bit_png(tmp_path):
    depth = np.linspace(0, 1, 50 * 40, dtype=np.float32).reshape(50, 40)
    out = tmp_path / "depth.png"
    save_image(depth, out, bit_depth=8)
    assert out.exists()
