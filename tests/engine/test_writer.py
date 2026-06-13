import numpy as np
import pytest

from focusstack.io import load_image
from focusstack.io.writer import save_image


@pytest.fixture
def img():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 1, (32, 48, 3)).astype(np.float32)


def test_tiff16_roundtrip(tmp_path, img):
    out = tmp_path / "r.tif"
    save_image(img, out, bit_depth=16, compression="zlib")
    back = load_image(out)
    assert back.bit_depth == 16
    np.testing.assert_allclose(back.pixels, img, atol=1.0 / 65535 + 1e-6)


def test_tiff_icc_embedded(tmp_path, img):
    icc = b"\x00\x00\x02\x00fake-icc"
    out = tmp_path / "icc.tif"
    save_image(img, out, bit_depth=16, icc=icc)
    assert load_image(out).icc == icc


def test_unclamped_input_is_clamped_at_export(tmp_path, img):
    hot = img.copy()
    hot[0, 0] = 1.7  # PMax overshoot (SPEC §7.1: clamp only at I/O boundary)
    hot[1, 1] = -0.3
    out = tmp_path / "c.tif"
    save_image(hot, out, bit_depth=16)
    back = load_image(out)
    assert back.pixels.max() <= 1.0
    assert back.pixels.min() >= 0.0


def test_jpeg_quality_and_icc(tmp_path, img):
    out = tmp_path / "q.jpg"
    save_image(img, out, jpeg_quality=90, icc=b"\x00\x00\x02\x00fake-icc")
    back = load_image(out)
    assert back.pixels.shape == img.shape
    assert back.icc == b"\x00\x00\x02\x00fake-icc"


def test_png8(tmp_path, img):
    out = tmp_path / "p.png"
    save_image(img, out, bit_depth=8)
    back = load_image(out)
    assert back.bit_depth == 8


def test_png16_rejected_until_m6(tmp_path, img):
    with pytest.raises(ValueError, match="M6"):
        save_image(img, tmp_path / "p16.png", bit_depth=16)


def test_unknown_extension_raises(tmp_path, img):
    with pytest.raises(ValueError):
        save_image(img, tmp_path / "x.webp")
