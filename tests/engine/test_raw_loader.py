from pathlib import Path

import numpy as np
import pytest
import tifffile

from focusstack.errors import RawDecodeError
from focusstack.io import ProcessingDomain, load_image
from focusstack.io.raw import RAW_EXTENSIONS, load_raw


def write_test_raw(
    path: Path,
    *,
    make: str = "FocusStack Camera Co",
    model: str = "SameCam Pro",
    width: int = 40,
    height: int = 32,
    cfa: tuple[int, int, int, int] = (0, 1, 1, 2),
    include_calibration: bool = True,
) -> Path:
    black = 512
    mosaic = np.empty((height, width), dtype=np.uint16)
    mosaic[0::2, 0::2] = black + 8000
    mosaic[0::2, 1::2] = black + 4000
    mosaic[1::2, 0::2] = black + 4000
    mosaic[1::2, 1::2] = black + 2000

    def ascii_tag(code: int, value: str):
        encoded = value.encode("utf-8") + b"\x00"
        return code, 2, len(encoded), encoded, False

    tags = [
        (50706, 1, 4, b"\x01\x07\x01\x00", False),
        (50707, 1, 4, b"\x01\x04\x00\x00", False),
        ascii_tag(50708, f"{make} {model}"),
        (33421, 3, 2, (2, 2), False),
        (33422, 1, 4, bytes(cfa), False),
        (50713, 3, 2, (2, 2), False),
        (50714, 5, 4, (black, 1, black, 1, black, 1, black, 1), False),
        (50717, 4, 1, 16383, False),
        ascii_tag(271, make),
        ascii_tag(272, model),
    ]
    if include_calibration:
        tags.extend([
            (50721, 10, 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), False),
            (50728, 5, 3, (1, 2, 1, 1, 2, 3), False),
            (50778, 3, 1, 21, False),
        ])
    tifffile.imwrite(path, mosaic, photometric=32803, metadata=None, extratags=tags)
    return path


def test_raw_extensions_cover_current_camera_families():
    assert {".dng", ".cr3", ".nef", ".arw", ".raf"} <= RAW_EXTENSIONS


def test_load_raw_is_scene_linear_no_bake(tmp_path: Path):
    path = write_test_raw(tmp_path / "frame.dng")

    frame = load_raw(path)

    assert frame.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
    assert frame.pixels.dtype == np.float32
    assert frame.pixels.shape == (32, 40, 3)
    denominator = 16383 - 512
    np.testing.assert_allclose(
        frame.pixels[16, 20],
        np.array([8000, 4000, 2000], dtype=np.float32) / denominator,
        atol=2.0 / denominator,
    )
    assert frame.metadata.camera_make == "FocusStack Camera Co"
    assert frame.metadata.camera_model == "SameCam Pro"
    assert frame.metadata.unique_camera_model == "FocusStack Camera Co SameCam Pro"
    assert frame.metadata.cfa_pattern == (0, 1, 3, 2)
    assert frame.metadata.black_level == (512.0, 512.0, 512.0, 512.0)
    assert frame.metadata.white_level == 16383.0
    assert frame.metadata.decoder["demosaic"] == "AHD"
    assert frame.metadata.decoder["white_balance"] == [1.0, 1.0, 1.0, 1.0]
    assert frame.metadata.decoder["gamma"] == [1.0, 1.0]
    assert frame.metadata.decoder["auto_brightness"] is False
    assert frame.metadata.decoder["output_color"] == "raw"


def test_generic_loader_dispatches_dng_to_raw_mode(tmp_path: Path):
    path = write_test_raw(tmp_path / "frame.dng")
    assert load_image(path).domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB


def test_invalid_raw_has_typed_camera_decode_error(tmp_path: Path):
    path = tmp_path / "broken.nef"
    path.write_bytes(b"not a camera raw")
    with pytest.raises(RawDecodeError, match="broken.nef"):
        load_raw(path)
