import numpy as np
import pytest
import imagecodecs

from planefuse.io import load_image
from planefuse.io.writer import save_float_tiff, save_image


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


def test_png16_rgb_roundtrip_without_8bit_quantization(tmp_path, img):
    out = tmp_path / "p16.png"
    save_image(img, out, bit_depth=16)
    encoded = imagecodecs.png_decode(out.read_bytes())
    assert encoded.dtype == np.uint16
    assert encoded.shape == img.shape
    np.testing.assert_allclose(encoded.astype(np.float32) / 65535.0, img, atol=1.0 / 65535 + 1e-6)
    back = load_image(out)
    assert back.bit_depth == 16


def test_failed_export_never_publishes_partial_destination(tmp_path, img, monkeypatch):
    out = tmp_path / "partial.tif"

    def fail_encode(*_args, **_kwargs):
        raise OSError("simulated encoder failure")

    monkeypatch.setattr("planefuse.io.writer._encode_pixels", fail_encode)
    with pytest.raises(OSError, match="simulated"):
        save_image(img, out, bit_depth=16)
    assert not out.exists()


def test_unknown_extension_raises(tmp_path, img):
    with pytest.raises(ValueError):
        save_image(img, tmp_path / "x.webp")


def test_float_tiff_companion_preserves_negative_and_highlight_headroom(tmp_path):
    import tifffile

    arr = np.linspace(-0.25, 1.5, 5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
    destination = tmp_path / "scene-linear-float.tif"
    save_float_tiff(arr, destination)
    decoded = tifffile.imread(destination)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, arr)
