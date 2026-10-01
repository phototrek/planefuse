from pathlib import Path

import numpy as np
import pytest
import rawpy
import tifffile

from planefuse.errors import DiskSpaceError, DngExportError
from planefuse.io import load_raw
from planefuse.io.dng import save_linear_dng, validate_linear_dng
from planefuse.io.metadata import parse_xmp_packet
from tests.engine.test_raw_loader import write_linear_dng, write_test_raw


def test_linear_dng_roundtrips_pixels_tags_and_provenance(tmp_path: Path):
    metadata = load_raw(write_test_raw(tmp_path / "source.dng")).metadata
    image = np.zeros((24, 30, 3), dtype=np.float32)
    image[..., 0] = np.linspace(-0.02, 1.25, 24 * 30, dtype=np.float32).reshape(24, 30)
    image[..., 1] = 0.5
    image[..., 2] = 0.25
    destination = tmp_path / "stacked.dng"
    provenance = {
        "method": "pmax",
        "sources": ["source-0001.dng", "source-0002.dng"],
        "transforms": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        "excluded": [],
    }

    report = save_linear_dng(image, destination, metadata, provenance)

    assert report.rawpy_validated
    with tifffile.TiffFile(destination) as tif:
        page = tif.pages[0]
        assert int(page.photometric) == 34892
        assert int(page.compression) == 7
        assert page.dtype == np.dtype(np.uint16)
        assert page.shape == image.shape
        assert page.tags[50706].value == b"\x01\x07\x01\x00"
        assert page.tags[50707].value == b"\x01\x04\x00\x00"
        assert page.tags[50708].value == "PlaneFuse Camera Co SameCam Pro"
        assert 50721 in page.tags
        assert 50728 in page.tags
        assert 50778 in page.tags
        assert 50714 in page.tags
        assert 50717 in page.tags
        xmp = parse_xmp_packet(bytes(page.tags[700].value))
        assert "pmax" in xmp["Xmp.PlaneFuse.Provenance"]
        encoded = page.asarray()

    restored = encoded.astype(np.float32) / 65535.0 * report.encoding_span + report.encoding_offset
    np.testing.assert_allclose(restored, image, atol=report.encoding_span / 65535.0 + 1e-6)

    with rawpy.imread(str(destination)) as raw:
        np.testing.assert_array_equal(raw.raw_image[..., :3], encoded)

    reopened = validate_linear_dng(destination, expected_pixels=encoded)
    assert reopened.rawpy_validated


def test_linear_dng_missing_camera_calibration_is_never_published(tmp_path: Path):
    metadata = load_raw(write_test_raw(tmp_path / "source.dng")).metadata
    from dataclasses import replace

    invalid = replace(metadata, color_matrix1=None)
    destination = tmp_path / "invalid.dng"
    with pytest.raises(DngExportError, match="color_matrix1"):
        save_linear_dng(np.zeros((8, 8, 3), dtype=np.float32), destination, invalid, {})
    assert not destination.exists()


def test_linear_dng_truncated_write_is_never_published(tmp_path: Path, monkeypatch):
    metadata = load_raw(write_test_raw(tmp_path / "source.dng")).metadata
    destination = tmp_path / "truncated.dng"

    def write_truncated(path, *_args, **_kwargs):
        path.write_bytes(b"II*\x00")

    monkeypatch.setattr("planefuse.io.dng._write_dng", write_truncated)
    with pytest.raises(DngExportError, match="tifffile"):
        save_linear_dng(np.zeros((8, 8, 3), dtype=np.float32), destination, metadata, {})
    assert not destination.exists()


def test_linear_dng_rejects_over_limit_dimensions_before_encoding(tmp_path: Path):
    metadata = load_raw(write_test_raw(tmp_path / "source.dng")).metadata
    image = np.zeros((65_001, 1, 3), dtype=np.float32)
    with pytest.raises(DngExportError, match="65,000"):
        save_linear_dng(image, tmp_path / "too-tall.dng", metadata, {})


def test_linear_dng_reports_insufficient_atomic_output_space(tmp_path: Path, monkeypatch):
    metadata = load_raw(write_test_raw(tmp_path / "source.dng")).metadata

    class Usage:
        free = 0

    monkeypatch.setattr("planefuse.io.dng.shutil.disk_usage", lambda _path: Usage())
    destination = tmp_path / "no-space.dng"
    with pytest.raises(DiskSpaceError, match="not enough free space"):
        save_linear_dng(np.zeros((8, 8, 3), dtype=np.float32), destination, metadata, {})
    assert not destination.exists()


def test_linear_dng_black_level_fits_a_rational_when_the_minimum_is_deeply_negative(tmp_path: Path):
    # A stack of linear renders (DxO DNGs) can undershoot to -0.2: the black code is then
    # 10922, which as a RATIONAL over a 1e6 denominator overflowed the 32-bit numerator.
    metadata = load_raw(write_test_raw(tmp_path / "source.dng")).metadata
    image = np.full((24, 30, 3), 0.5, dtype=np.float32)
    image[0, 0] = -0.2
    image[0, 1] = 1.0
    destination = tmp_path / "undershoot.dng"

    report = save_linear_dng(image, destination, metadata, {})

    assert report.encoding_offset == pytest.approx(-0.2)
    assert report.encoding_span == pytest.approx(1.2)
    expected_black = round(0.2 / 1.2 * 65535.0)
    with tifffile.TiffFile(destination) as tif:
        page = tif.pages[0]
        assert page.tags[50714].value == (expected_black, 1) * 3
        white_code = int(page.tags[50717].value[0])
        encoded = page.asarray()
    assert int(encoded[0, 0, 0]) == 0
    assert int(encoded[0, 1, 0]) == white_code


def test_linear_dng_carries_a_linear_dng_sources_baseline_exposure(tmp_path: Path):
    samples = np.random.default_rng(5).integers(0, 65535, size=(32, 40, 3), dtype=np.uint16)
    source = write_linear_dng(tmp_path / "render.dng", samples, compression="jpegxl", baseline_exposure=(52, 100))
    metadata = load_raw(source).metadata
    assert metadata.baseline_exposure == pytest.approx(0.52)
    destination = tmp_path / "stacked.dng"

    save_linear_dng(samples.astype(np.float32) / 65535.0, destination, metadata, {})

    with tifffile.TiffFile(destination) as tif:
        numerator, denominator = tif.pages[0].tags[50730].value
    assert numerator / denominator == pytest.approx(0.52)
