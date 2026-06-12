from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from focusstack.errors import ValidationError
from focusstack.io.loader import load_image, validate_stack


@pytest.fixture
def tmp_images(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr16 = rng.integers(0, 65535, (40, 60, 3), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "a_16bit.tif", arr16)
    arr8 = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    Image.fromarray(arr8).save(tmp_path / "b_8bit.jpg", quality=95)
    Image.fromarray(arr8).save(tmp_path / "c_8bit.png")
    return tmp_path


def test_load_tiff16(tmp_images):
    f = load_image(tmp_images / "a_16bit.tif")
    assert f.pixels.shape == (40, 60, 3)
    assert f.pixels.dtype == np.float32
    assert f.bit_depth == 16
    assert 0.0 <= f.pixels.min() and f.pixels.max() <= 1.0


def test_load_jpeg_and_png(tmp_images):
    for name, depth in [("b_8bit.jpg", 8), ("c_8bit.png", 8)]:
        f = load_image(tmp_images / name)
        assert f.pixels.shape == (40, 60, 3)
        assert f.bit_depth == depth


def test_grayscale_rejected(tmp_path):
    gray = np.zeros((20, 20), dtype=np.uint8)
    Image.fromarray(gray, mode="L").save(tmp_path / "gray.png")
    with pytest.raises(ValidationError, match="RGB"):
        load_image(tmp_path / "gray.png")


def test_icc_passthrough(tmp_path):
    icc = b"\x00\x00\x02\x00fake-icc-profile-bytes"
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "with_icc.jpg", icc_profile=icc)
    f = load_image(tmp_path / "with_icc.jpg")
    assert f.icc == icc


def test_validate_stack_ok(tmp_images):
    report = validate_stack(sorted(tmp_images.glob("*8bit*")))
    assert report.ok
    assert all(s.status == "ok" for s in report.files)


def test_validate_stack_mixed_sizes(tmp_path):
    a = np.zeros((20, 20, 3), dtype=np.uint8)
    b = np.zeros((30, 20, 3), dtype=np.uint8)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    report = validate_stack([tmp_path / "a.png", tmp_path / "b.png"])
    assert not report.ok
    statuses = {s.path.name: s.status for s in report.files}
    assert statuses["a.png"] == "ok"
    assert statuses["b.png"] == "wrong_size"


def test_validate_stack_unreadable(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a tiff at all")
    report = validate_stack([bad])
    assert not report.ok
    assert report.files[0].status == "unreadable"
