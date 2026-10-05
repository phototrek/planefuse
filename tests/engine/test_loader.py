from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from planefuse.errors import ValidationError
from planefuse.io.loader import load_image, validate_stack
from planefuse.io.metadata import ProcessingDomain
from tests.engine.test_raw_loader import write_test_raw


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


def test_rendered_frame_carries_exif_xmp_and_normalizes_orientation(tmp_path):
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    exif = Image.Exif()
    exif[271] = "PlaneFuse Camera Co"
    exif[272] = "SameCam Pro"
    exif[274] = 6
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><label>source</label></x:xmpmeta>'
    path = tmp_path / "oriented.jpg"
    Image.fromarray(pixels).save(path, exif=exif, xmp=xmp, icc_profile=b"test-icc")

    frame = load_image(path)

    assert frame.domain is ProcessingDomain.RENDERED_RGB
    assert frame.pixels.shape == (3, 2, 3)
    assert frame.metadata.camera_make == "PlaneFuse Camera Co"
    assert frame.metadata.camera_model == "SameCam Pro"
    assert frame.metadata.source_orientation == 6
    assert frame.metadata.orientation == 1
    assert frame.metadata.exif_bytes
    assert frame.metadata.xmp_bytes == xmp
    assert frame.metadata.icc == b"test-icc"
    assert frame.icc == frame.metadata.icc
    assert frame.path == path


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


def test_validate_same_camera_raw_stack_reports_domain_and_decoder(tmp_path):
    paths = [write_test_raw(tmp_path / f"frame-{index}.dng") for index in range(2)]
    report = validate_stack(paths)
    assert report.ok
    assert report.domain == ProcessingDomain.SCENE_LINEAR_CAMERA_RGB.value
    assert report.camera == "PlaneFuse Camera Co SameCam Pro"
    assert report.decoder["demosaic"] == "AHD"


@pytest.mark.parametrize(
    ("second_options", "expected_status"),
    [
        ({"model": "Different Camera"}, "incompatible_camera"),
        ({"width": 42}, "incompatible_sensor_mode"),
        ({"cfa": (2, 1, 1, 0)}, "incompatible_cfa"),
        ({"black": 514}, "incompatible_raw_calibration"),
    ],
)
def test_validate_raw_stack_rejects_incompatible_frames(tmp_path, second_options, expected_status):
    first = write_test_raw(tmp_path / "first.dng")
    second = write_test_raw(tmp_path / "second.dng", **second_options)
    report = validate_stack([first, second])
    assert not report.ok
    assert report.files[0].status == "ok"
    assert report.files[1].status == expected_status
    assert "expected" in report.files[1].message
    assert "actual" in report.files[1].message


@pytest.mark.parametrize("second_black", [511, 513])
def test_validate_raw_stack_accepts_one_count_of_black_level_difference(tmp_path, second_black):
    first = write_test_raw(tmp_path / "first.dng")
    second = write_test_raw(tmp_path / "second.dng", black=second_black)
    report = validate_stack([first, second])
    assert report.ok, [(status.status, status.message) for status in report.files]


def test_validate_stack_rejects_mixed_rendered_and_raw_domains(tmp_path):
    raw = write_test_raw(tmp_path / "first.dng")
    rendered = tmp_path / "second.png"
    Image.fromarray(np.zeros((32, 40, 3), dtype=np.uint8)).save(rendered)
    report = validate_stack([raw, rendered])
    assert not report.ok
    assert report.files[1].status == "mixed_domain"
