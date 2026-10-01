import struct
from pathlib import Path

import numpy as np
import pytest
import tifffile

from planefuse.errors import RawDecodeError
from planefuse.io import ProcessingDomain, load_image
from planefuse.io.raw import RAW_EXTENSIONS, load_raw


def write_test_raw(
    path: Path,
    *,
    make: str = "PlaneFuse Camera Co",
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
    assert frame.metadata.camera_make == "PlaneFuse Camera Co"
    assert frame.metadata.camera_model == "SameCam Pro"
    assert frame.metadata.unique_camera_model == "PlaneFuse Camera Co SameCam Pro"
    assert frame.metadata.cfa_pattern == (0, 1, 3, 2)
    assert frame.metadata.black_level == (512.0, 512.0, 512.0, 512.0)
    assert frame.metadata.white_level == 16383.0
    assert frame.metadata.decoder["demosaic"] == "AHD"
    assert frame.metadata.decoder["white_balance"] == [1.0, 1.0, 1.0, 1.0]
    assert frame.metadata.decoder["gamma"] == [1.0, 1.0]
    assert frame.metadata.decoder["auto_brightness"] is False
    assert frame.metadata.decoder["output_color"] == "raw"


def test_color_matrix_fallback_keeps_dng_xyz_to_camera_convention(tmp_path: Path):
    # Without an embedded ColorMatrix1 tag, LibRaw fills cam_xyz from its
    # Adobe table for known cameras. That matrix is already XYZ -> cameraRGB
    # (DNG ColorMatrix1 convention) and must be stored verbatim, not inverted:
    # an inverted matrix renders previews (and exported DNGs) with wildly
    # rotated hues.
    path = write_test_raw(
        tmp_path / "frame.dng",
        make="Canon",
        model="EOS 5D Mark III",
        include_calibration=False,
    )
    frame = load_raw(path)
    matrix = frame.metadata.color_matrix1
    assert matrix is not None
    expected = np.array(
        [6722, -635, -963, -4287, 12460, 2028, -908, 2162, 5668], dtype=np.float64
    ) / 10000.0
    np.testing.assert_allclose(np.asarray(matrix), expected, atol=1e-4)


def test_generic_loader_dispatches_dng_to_raw_mode(tmp_path: Path):
    path = write_test_raw(tmp_path / "frame.dng")
    assert load_image(path).domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB


def test_invalid_raw_has_typed_camera_decode_error(tmp_path: Path):
    path = tmp_path / "broken.nef"
    path.write_bytes(b"not a camera raw")
    with pytest.raises(RawDecodeError, match="broken.nef"):
        load_raw(path)


_LINEAR_DNG_COMPRESSIONS = {
    "none": {},
    "deflate": {"compression": "adobe_deflate", "tile": (16, 16)},
    "deflate-strips": {"compression": "adobe_deflate", "rowsperstrip": 8},
    "ljpeg": {"compression": "jpeg", "compressionargs": {"lossless": True, "bitspersample": 16}},
    "jpegxl": {"compression": "jpegxl_dng", "tile": (16, 16)},
}


def write_linear_dng(
    path: Path,
    pixels: np.ndarray,
    *,
    compression: str,
    black: int = 0,
    white: int = 65535,
    orientation: int = 1,
    photometric: int = 34892,
    opcode_lists: dict[int, bytes] | None = None,
    baseline_exposure: tuple[int, int] | None = None,
    linearization: tuple[int, ...] | None = None,
) -> Path:
    """A DNG laid out like a DxO PhotoLab render: a small RGB preview in IFD0 carrying the
    camera tags, and the full-resolution image (a _LINEAR_DNG_COMPRESSIONS key) in a SubIFD."""

    def ascii_tag(code: int, value: str):
        encoded = value.encode("utf-8") + b"\x00"
        return code, 2, len(encoded), encoded, False

    ifd0_tags = [
        (50706, 1, 4, b"\x01\x07\x00\x00", False),
        (50707, 1, 4, b"\x01\x07\x00\x00" if compression == "jpegxl" else b"\x01\x04\x00\x00", False),
        ascii_tag(50708, "PlaneFuse Camera Co SameCam Pro"),
        ascii_tag(271, "PlaneFuse Camera Co"),
        ascii_tag(272, "SameCam Pro"),
        (274, 3, 1, orientation, False),
        (50721, 10, 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), False),
        (50728, 5, 3, (1, 2, 1, 1, 2, 3), False),
        (50778, 3, 1, 21, False),
    ]
    if baseline_exposure is not None:
        ifd0_tags.append((50730, 10, 1, baseline_exposure, False))
    samples = 1 if pixels.ndim == 2 else pixels.shape[2]
    raw_tags = [
        (50713, 3, 2, (1, 1), False),
        (50714, 5, samples, (black, 1) * samples, False),
        (50717, 4, samples, (white,) * samples, False),
    ]
    if linearization is not None:
        raw_tags.append((50712, 3, len(linearization), linearization, False))
    for code, blob in (opcode_lists or {}).items():
        raw_tags.append((code, 7, len(blob), blob, False))
    with tifffile.TiffWriter(path) as writer:
        writer.write(
            np.zeros((6, 8, 3), dtype=np.uint8),
            photometric="rgb",
            subfiletype=1,
            subifds=1,
            metadata=None,
            extratags=ifd0_tags,
        )
        options = _LINEAR_DNG_COMPRESSIONS[compression]
        writer.write(
            pixels,
            photometric=photometric,
            planarconfig="contig" if samples > 1 else None,
            subfiletype=0,
            extrasamples=False,
            metadata=None,
            extratags=raw_tags,
            **options,
        )
    return path


def opcode(opcode_id: int, params: bytes, *, optional: bool = False) -> bytes:
    return struct.pack(">I4sII", opcode_id, b"\x01\x03\x00\x00", int(optional), len(params)) + params


def opcode_list(*opcodes: bytes) -> bytes:
    return struct.pack(">I", len(opcodes)) + b"".join(opcodes)


def map_polynomial(area, plane, planes, pitch, coefficients) -> bytes:
    top, left, bottom, right = area
    return opcode(
        8,
        struct.pack(">9I", top, left, bottom, right, plane, planes, pitch[0], pitch[1], len(coefficients) - 1)
        + struct.pack(f">{len(coefficients)}d", *coefficients),
    )


def _linear_samples(height: int = 48, width: int = 64, low: int = 0, high: int = 65535) -> np.ndarray:
    return np.random.default_rng(7).integers(low, high + 1, size=(height, width, 3), dtype=np.uint16)


def test_jpegxl_linear_dng_decodes_without_libraw(tmp_path: Path):
    samples = _linear_samples(low=256, high=60000)
    path = write_linear_dng(tmp_path / "render.dng", samples, compression="jpegxl", black=256, white=60000)

    frame = load_image(path)

    assert frame.domain is ProcessingDomain.SCENE_LINEAR_CAMERA_RGB
    assert frame.pixels.dtype == np.float32
    assert frame.pixels.shape == (48, 64, 3)
    expected = (samples.astype(np.float32) - 256.0) / np.float32(60000 - 256)
    np.testing.assert_array_equal(frame.pixels, expected)
    assert frame.metadata.decoder["library"] == "tifffile/imagecodecs"
    assert frame.metadata.unique_camera_model == "PlaneFuse Camera Co SameCam Pro"
    assert frame.metadata.black_level == (256.0, 256.0, 256.0, 0.0)
    assert frame.metadata.white_level == 60000.0


@pytest.mark.parametrize("orientation", [1, 8])
def test_jpegxl_linear_dng_frame_equals_libraw_frame_of_uncompressed_twin(tmp_path: Path, orientation: int):
    samples = _linear_samples()
    jxl = load_raw(write_linear_dng(tmp_path / "jxl.dng", samples, compression="jpegxl", orientation=orientation))
    plain = load_raw(write_linear_dng(tmp_path / "plain.dng", samples, compression="none", orientation=orientation))

    assert plain.metadata.decoder["library"] == "rawpy/LibRaw"
    assert jxl.metadata.decoder["library"] == "tifffile/imagecodecs"
    np.testing.assert_array_equal(jxl.pixels, plain.pixels)
    assert jxl.bit_depth == plain.bit_depth
    for name in (
        "source_orientation", "active_size", "cfa_pattern", "color_description", "bits_per_sample",
        "black_level", "white_level", "color_matrix1", "as_shot_neutral", "calibration_illuminant1",
        "unique_camera_model",
    ):
        assert getattr(jxl.metadata, name) == getattr(plain.metadata, name), name


def test_jpegxl_linear_dng_validates_and_sizes_as_a_raw_stack(tmp_path: Path):
    from planefuse.io.loader import probe_frame_shape, validate_stack

    paths = [
        write_linear_dng(tmp_path / f"{index:03d}.dng", _linear_samples(), compression="jpegxl", orientation=8)
        for index in (1, 2)
    ]
    assert probe_frame_shape(paths[0]) == (64, 48)
    report = validate_stack(paths)
    assert report.ok, [(status.status, status.message) for status in report.files]
    assert report.domain == ProcessingDomain.SCENE_LINEAR_CAMERA_RGB.value


def test_unreadable_non_linear_dng_still_fails_naming_what_it_found(tmp_path: Path):
    mosaic = np.random.default_rng(3).integers(0, 65535, size=(32, 32), dtype=np.uint16)
    path = write_linear_dng(tmp_path / "cfa.dng", mosaic, compression="jpegxl", photometric=32803)
    with pytest.raises(RawDecodeError, match=r"cfa\.dng: .*PhotometricInterpretation 32803, 1 samples"):
        load_raw(path)


def _polynomial(x: np.ndarray, coefficients) -> np.ndarray:
    return np.clip(sum(c * x.astype(np.float64) ** i for i, c in enumerate(coefficients)), 0.0, 1.0)


@pytest.mark.parametrize("compression", sorted(_LINEAR_DNG_COMPRESSIONS))
def test_linear_dng_applies_opcode_list2_map_polynomials_for_every_compression(
    tmp_path: Path, compression: str
):
    samples = _linear_samples()
    # DxO's shape (one cubic per plane over the whole image) plus a generic one: a sub-area,
    # every other row and column, two planes at once, a curve that overshoots and is clipped.
    curves = [
        ((0, 0, 48, 64), 0, 1, (1, 1), (0.003, 0.016, 0.0, 0.246)),
        ((0, 0, 48, 64), 1, 1, (1, 1), (0.005, 0.031, 0.0, 0.461)),
        ((0, 0, 48, 64), 2, 1, (1, 1), (0.003, 0.020, 0.0, 0.301)),
        ((8, 4, 40, 60), 1, 2, (2, 2), (0.1, 3.0)),
    ]
    blob = opcode_list(*(map_polynomial(*curve) for curve in curves))
    path = write_linear_dng(tmp_path / "curves.dng", samples, compression=compression, opcode_lists={51009: blob})

    frame = load_raw(path)

    expected = samples.astype(np.float64) / 65535.0
    for (top, left, bottom, right), plane, planes, (row_pitch, col_pitch), coefficients in curves:
        region = (slice(top, bottom, row_pitch), slice(left, right, col_pitch), slice(plane, plane + planes))
        expected[region] = _polynomial(expected[region], coefficients)
    np.testing.assert_allclose(frame.pixels, expected, atol=2e-6)
    assert frame.metadata.decoder["library"] == "tifffile/imagecodecs"
    assert frame.metadata.decoder["opcode_list2"] == ["MapPolynomial (8)"] * 4


def test_unsupported_opcode_is_refused_by_name_unless_optional(tmp_path: Path):
    samples = _linear_samples()
    gain_map = opcode(9, b"\x00" * 16)
    refused = write_linear_dng(
        tmp_path / "gainmap.dng", samples, compression="jpegxl", opcode_lists={51009: opcode_list(gain_map)}
    )
    with pytest.raises(RawDecodeError, match=r"gainmap\.dng: OpcodeList2 holds GainMap \(9\)"):
        load_raw(refused)

    late = write_linear_dng(
        tmp_path / "late.dng", samples, compression="jpegxl", opcode_lists={51022: opcode_list(opcode(77, b""))}
    )
    with pytest.raises(RawDecodeError, match=r"late\.dng: OpcodeList3 holds unknown opcode \(77\)"):
        load_raw(late)

    skipped = write_linear_dng(
        tmp_path / "optional.dng",
        samples,
        compression="jpegxl",
        opcode_lists={
            51008: opcode_list(opcode(4, b"\x00" * 8, optional=True)),
            51009: opcode_list(opcode(9, b"\x00" * 16, optional=True)),
        },
    )
    plain = load_raw(write_linear_dng(tmp_path / "plain.dng", samples, compression="jpegxl"))
    np.testing.assert_array_equal(load_raw(skipped).pixels, plain.pixels)


@pytest.mark.parametrize("compression", ["none", "ljpeg"])
def test_linear_dng_without_opcodes_keeps_the_libraw_path(tmp_path: Path, compression: str):
    from planefuse.io.loader import probe_frame_shape

    samples = _linear_samples()
    plain = write_linear_dng(tmp_path / "plain.dng", samples, compression=compression)
    empty = write_linear_dng(
        tmp_path / "empty.dng", samples, compression=compression, opcode_lists={51009: opcode_list()}
    )
    for path in (plain, empty):
        frame = load_raw(path)
        assert frame.metadata.decoder["library"] == "rawpy/LibRaw"
        np.testing.assert_array_equal(frame.pixels, samples.astype(np.float32) / np.float32(65535))
        assert probe_frame_shape(path) == (48, 64)


def test_linear_dng_with_opcodes_sizes_and_validates_through_tifffile(tmp_path: Path):
    from planefuse.io.loader import probe_frame_shape, validate_stack

    curve = opcode_list(map_polynomial((0, 0, 48, 64), 0, 3, (1, 1), (0.0, 0.5)))
    paths = [
        write_linear_dng(
            tmp_path / f"{index:03d}.dng", _linear_samples(), compression="none", orientation=8,
            opcode_lists={51009: curve},
        )
        for index in (1, 2)
    ]
    assert probe_frame_shape(paths[0]) == (64, 48)
    report = validate_stack(paths)
    assert report.ok, [(status.status, status.message) for status in report.files]
    assert report.decoder["library"] == "tifffile/imagecodecs"


@pytest.mark.parametrize("compression", ["jpegxl", "none"])
def test_linear_dng_applies_the_linearization_table_before_black(tmp_path: Path, compression: str):
    # DxO's JPEG XL renders without opcodes store a 10-bit code and a quadratic 1024-entry table.
    table = tuple(int(round(65535 * (i / 1023) ** 2)) for i in range(1024))
    samples = np.random.default_rng(11).integers(0, 1100, size=(48, 64, 3), dtype=np.uint16)
    path = write_linear_dng(
        tmp_path / "table.dng", samples, compression=compression, black=256, linearization=table,
        opcode_lists={51009: opcode_list(opcode(9, b"", optional=True))},
    )

    frame = load_raw(path)

    linear = np.asarray(table, dtype=np.float32)[np.minimum(samples, 1023)]
    expected = np.clip(linear - 256.0, 0.0, 65535 - 256) / np.float32(65535 - 256)
    np.testing.assert_allclose(frame.pixels, expected, atol=1e-6)
    assert frame.metadata.decoder["library"] == "tifffile/imagecodecs"


def test_libraw_path_carries_the_dng_baseline_exposure(tmp_path: Path):
    path = write_linear_dng(tmp_path / "plain.dng", _linear_samples(), compression="none", baseline_exposure=(52, 100))
    frame = load_raw(path)
    assert frame.metadata.decoder["library"] == "rawpy/LibRaw"
    assert frame.metadata.baseline_exposure == pytest.approx(0.52)
    assert load_raw(write_test_raw(tmp_path / "cfa.dng")).metadata.baseline_exposure == 0.0
